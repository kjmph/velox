/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/experimental/cudf/connectors/hive/PinnedStagingArena.h"

#include "velox/common/base/Exceptions.h"

#include <cuda_runtime_api.h>

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

std::string readProcStatusList(std::string_view name) {
  std::ifstream status("/proc/self/status");
  std::string line;
  const auto prefix = std::string{name} + ":";
  while (std::getline(status, line)) {
    if (!line.starts_with(prefix)) {
      continue;
    }
    const auto value = line.find_first_not_of(" \t", prefix.size());
    return value == std::string::npos ? "" : line.substr(value);
  }
  return "";
}

std::string systemError(int error) {
  return std::string(std::strerror(error)) +
      " (errno=" + std::to_string(error) + ")";
}

const char* allocationStrategyName(
    PinnedStagingArena::HostAllocationStrategy strategy) {
  switch (strategy) {
    case PinnedStagingArena::HostAllocationStrategy::kCudaHostAlloc:
      return "cudaHostAlloc";
    case PinnedStagingArena::HostAllocationStrategy::kMmapMbindCudaHostRegister:
      return "mmap-mbind-first-touch-cudaHostRegister";
    case PinnedStagingArena::HostAllocationStrategy::kUninitialized:
      return "uninitialized";
  }
  return "unknown";
}

struct NumaPolicyDetection {
  std::optional<uint32_t> node;
  std::string detail;
};

NumaPolicyDetection resolveSingleNodePolicy(
    bool relativeNodes,
    std::span<const uint32_t> policyNodes,
    std::span<const uint32_t> allowedNodes) {
  std::vector<uint32_t> resolvedNodes;
  resolvedNodes.reserve(policyNodes.size());
  for (const auto policyNode : policyNodes) {
    if (relativeNodes) {
      if (policyNode >= allowedNodes.size()) {
        throw std::runtime_error(
            "Relative MPOL_BIND node " + std::to_string(policyNode) +
            " is outside the process's allowed memory-node set");
      }
      resolvedNodes.push_back(allowedNodes[policyNode]);
    } else {
      resolvedNodes.push_back(policyNode);
    }
  }
  std::sort(resolvedNodes.begin(), resolvedNodes.end());
  resolvedNodes.erase(
      std::unique(resolvedNodes.begin(), resolvedNodes.end()),
      resolvedNodes.end());
  if (resolvedNodes.size() != 1) {
    return {
        std::nullopt,
        "effective MPOL_BIND policy resolves to " +
            std::to_string(resolvedNodes.size()) + " physical NUMA nodes"};
  }
  return {
      resolvedNodes.front(),
      relativeNodes ? "single-node relative MPOL_BIND policy"
                    : "single-node absolute MPOL_BIND policy"};
}

#if defined(__linux__)

constexpr uint32_t kFallbackMaximumNumaNodes = 1024;

uint32_t maximumPossibleNumaNodes() {
  std::ifstream possible("/sys/devices/system/node/possible");
  std::string value;
  if (!(possible >> value)) {
    return kFallbackMaximumNumaNodes;
  }

  uint64_t maximum = 0;
  size_t begin = 0;
  while (begin < value.size()) {
    const auto end = value.find(',', begin);
    const auto token = value.substr(
        begin, end == std::string::npos ? std::string::npos : end - begin);
    const auto dash = token.find('-');
    const auto last = token.substr(dash == std::string::npos ? 0 : dash + 1);
    char* parseEnd = nullptr;
    errno = 0;
    const auto parsed = std::strtoull(last.c_str(), &parseEnd, 10);
    if (errno != 0 || parseEnd == last.c_str() || *parseEnd != '\0' ||
        parsed >= std::numeric_limits<uint32_t>::max()) {
      return kFallbackMaximumNumaNodes;
    }
    maximum = std::max(maximum, static_cast<uint64_t>(parsed));
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return static_cast<uint32_t>(maximum + 1);
}

NumaPolicyDetection detectSingleNodeBind() {
  int mode = MPOL_DEFAULT;
  if (::syscall(SYS_get_mempolicy, &mode, nullptr, 0, nullptr, 0) != 0) {
    const auto error = errno;
    return {std::nullopt, "get_mempolicy unavailable: " + systemError(error)};
  }

  const auto baseMode = mode & ~MPOL_MODE_FLAGS;
  if (baseMode != MPOL_BIND) {
    return {
        std::nullopt,
        "effective process policy is not MPOL_BIND (mode=" +
            std::to_string(baseMode) + ")"};
  }

  const auto maxNodes = maximumPossibleNumaNodes();
  constexpr auto kBitsPerWord = sizeof(unsigned long) * CHAR_BIT;
  std::vector<unsigned long> mask(
      (static_cast<uint64_t>(maxNodes) + kBitsPerWord - 1) / kBitsPerWord, 0);
  if (::syscall(SYS_get_mempolicy, &mode, mask.data(), maxNodes, nullptr, 0) !=
      0) {
    const auto error = errno;
    throw std::runtime_error(
        "Cannot read the node mask of the active MPOL_BIND policy: " +
        systemError(error));
  }

  std::vector<uint32_t> policyNodes;
  for (uint32_t candidate = 0; candidate < maxNodes; ++candidate) {
    if ((mask[candidate / kBitsPerWord] &
         (1UL << (candidate % kBitsPerWord))) == 0) {
      continue;
    }
    policyNodes.push_back(candidate);
  }
  if (policyNodes.empty()) {
    throw std::runtime_error(
        "The active MPOL_BIND policy has an empty NUMA node mask");
  }

  std::vector<uint32_t> allowedNodes;
  if ((mode & MPOL_F_RELATIVE_NODES) != 0) {
    std::vector<unsigned long> allowedMask(mask.size(), 0);
    int ignoredMode = MPOL_DEFAULT;
    if (::syscall(
            SYS_get_mempolicy,
            &ignoredMode,
            allowedMask.data(),
            maxNodes,
            nullptr,
            MPOL_F_MEMS_ALLOWED) != 0) {
      const auto error = errno;
      throw std::runtime_error(
          "Cannot resolve relative MPOL_BIND policy against the process's "
          "allowed memory-node set: " +
          systemError(error));
    }
    for (uint32_t candidate = 0; candidate < maxNodes; ++candidate) {
      if ((allowedMask[candidate / kBitsPerWord] &
           (1UL << (candidate % kBitsPerWord))) != 0) {
        allowedNodes.push_back(candidate);
      }
    }
    if (allowedNodes.empty()) {
      throw std::runtime_error(
          "The process's allowed memory-node set is empty");
    }
  }

  return resolveSingleNodePolicy(
      (mode & MPOL_F_RELATIVE_NODES) != 0, policyNodes, allowedNodes);
}

void bestEffortMappingAdvice(void* data, size_t bytes) {
#if defined(MADV_DONTDUMP)
  if (::madvise(data, bytes, MADV_DONTDUMP) != 0) {
    const auto error = errno;
    VLOG(1) << "MADV_DONTDUMP failed for pinned staging: "
            << systemError(error);
  }
#endif
#if defined(MADV_DONTFORK)
  if (::madvise(data, bytes, MADV_DONTFORK) != 0) {
    const auto error = errno;
    VLOG(1) << "MADV_DONTFORK failed for pinned staging: "
            << systemError(error);
  }
#endif
}

#else

NumaPolicyDetection detectSingleNodeBind() {
  return {std::nullopt, "explicit NUMA placement is Linux-only"};
}

#endif

bool productionRequiresNumaLocalAllocation() {
  const auto* value = std::getenv("PRESTO_GPU_NUMA_BINDING");
  return value != nullptr && std::string_view(value) == "required";
}

class PackThreadPool {
 public:
  explicit PackThreadPool(uint32_t numThreads) {
    threads_.reserve(numThreads);
    try {
      for (uint32_t i = 0; i < numThreads; ++i) {
        threads_.emplace_back([this]() { run(); });
      }
    } catch (...) {
      stop();
      throw;
    }
  }

  ~PackThreadPool() {
    stop();
  }

  PackThreadPool(const PackThreadPool&) = delete;
  PackThreadPool& operator=(const PackThreadPool&) = delete;

  void add(std::function<void()> task) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) {
        throw std::runtime_error("Pinned staging pack pool is stopping");
      }
      tasks_.push_back(std::move(task));
    }
    cv_.notify_one();
  }

  [[nodiscard]] uint32_t numThreads() const {
    return threads_.size();
  }

 private:
  void run() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return stopping_ || !tasks_.empty(); });
        if (stopping_ && tasks_.empty()) {
          return;
        }
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      task();
    }
  }

  void stop() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    cv_.notify_all();
    for (auto& thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> tasks_;
  std::vector<std::thread> threads_;
  bool stopping_{false};
};

struct PackJob {
  struct Work {
    const uint8_t* source;
    uint8_t* destination;
    uint64_t size;
  };

  PackJob(std::vector<Work> work, uint32_t workers)
      : work(std::move(work)), remaining(workers) {}

  void copyAll() noexcept {
    while (true) {
      const auto index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= work.size()) {
        return;
      }
      const auto& item = work[index];
      std::memcpy(item.destination, item.source, item.size);
    }
  }

  void complete(uint32_t count = 1) noexcept {
    if (remaining.fetch_sub(count, std::memory_order_acq_rel) == count) {
      std::lock_guard<std::mutex> lock(mutex);
      cv.notify_all();
    }
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this]() {
      return remaining.load(std::memory_order_acquire) == 0;
    });
  }

  std::vector<Work> work;
  std::atomic<uint64_t> next{0};
  std::atomic<uint32_t> remaining;
  std::mutex mutex;
  std::condition_variable cv;
};

} // namespace

struct PinnedStagingArena::Impl {
  struct Window {
    ~Window() {
      if (data == nullptr) {
        return;
      }
      if (strategy == HostAllocationStrategy::kCudaHostAlloc) {
        const auto status = cudaFreeHost(data);
        if (status != cudaSuccess) {
          LOG(ERROR) << "cudaFreeHost failed for a pinned staging window: "
                     << cudaGetErrorString(status);
        }
        return;
      }

#if defined(__linux__)
      const auto status = cudaHostUnregister(data);
      if (status != cudaSuccess) {
        // The arena lease protocol should make this impossible during normal
        // shutdown. If it nevertheless happens, retain the mapping until
        // process exit rather than unmapping pages that CUDA may still regard
        // as DMA-visible.
        LOG(ERROR) << "cudaHostUnregister failed for a NUMA-local pinned "
                      "staging window; leaking the mapping until process exit: "
                   << cudaGetErrorString(status);
        return;
      }
      if (::munmap(data, mappedBytes) != 0) {
        const auto error = errno;
        LOG(ERROR) << "munmap failed for a NUMA-local pinned staging window: "
                   << systemError(error);
      }
#endif
    }

    Window() = default;
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    uint8_t* data{nullptr};
    size_t mappedBytes{0};
    HostAllocationStrategy strategy{HostAllocationStrategy::kUninitialized};
    std::mutex packMutex;
  };

  struct Config {
    bool enabled{false};
    uint64_t windowBytes{0};
    uint32_t packThreads{0};
    uint32_t windowSetCount{0};
  };

  struct WindowSet {
    std::array<Window, PinnedStagingArena::kWindowCount> windows;
    // Keep an independent pool per set so two concurrent leases cannot
    // monopolize one process-wide FIFO with coarse copyAll() jobs. Declare the
    // pool after the windows so it is stopped before their storage is freed.
    std::unique_ptr<PackThreadPool> packPool;
  };

  Impl(
      uint64_t windowBytes,
      uint32_t packThreads,
      uint32_t windowSetCount,
      uint32_t failAllocationAt,
      uint32_t failRegistrationAt,
      std::optional<uint32_t> numaNodeOverride,
      bool hasNumaNodeOverride,
      bool requireNumaLocalOverride,
      bool hasRequireNumaLocalOverride)
      : windowBytes_(windowBytes) {
    NumaPolicyDetection numaPolicy;
    if (hasNumaNodeOverride) {
      numaPolicy = {
          numaNodeOverride,
          numaNodeOverride.has_value()
              ? "test override selecting NUMA node " +
                  std::to_string(*numaNodeOverride)
              : "test override selecting portable allocation"};
    } else {
      numaPolicy = detectSingleNodeBind();
    }
    requireNumaLocal_ = hasRequireNumaLocalOverride
        ? requireNumaLocalOverride
        : productionRequiresNumaLocalAllocation();
    if (requireNumaLocal_ && !numaPolicy.node.has_value()) {
      throw std::runtime_error(
          "PRESTO_GPU_NUMA_BINDING=required, but a strict single-node "
          "MPOL_BIND policy could not be proven: " +
          numaPolicy.detail);
    }
    allocationStrategy_ = numaPolicy.node.has_value()
        ? HostAllocationStrategy::kMmapMbindCudaHostRegister
        : HostAllocationStrategy::kCudaHostAlloc;
    numaNode_ = numaPolicy.node;
    numaPolicyDetail_ = std::move(numaPolicy.detail);

    windowSets_.reserve(windowSetCount);
    uint32_t allocationOrdinal = 0;
    uint32_t registrationOrdinal = 0;
    for (uint32_t windowSetIndex = 0; windowSetIndex < windowSetCount;
         ++windowSetIndex) {
      try {
        auto windowSet = std::make_unique<WindowSet>();
        for (auto& window : windowSet->windows) {
          ++allocationOrdinal;
          if (failAllocationAt == allocationOrdinal) {
            throw std::runtime_error(
                "Injected pinned staging allocation failure");
          }
          if (allocationStrategy_ == HostAllocationStrategy::kCudaHostAlloc) {
            allocatePortableWindow(window);
          } else {
            ++registrationOrdinal;
            allocateNumaLocalWindow(
                window, *numaNode_, failRegistrationAt == registrationOrdinal);
          }
        }
        windowSet->packPool = std::make_unique<PackThreadPool>(packThreads);
        windowSets_.push_back(std::move(windowSet));
      } catch (const std::exception& error) {
        if (windowSets_.empty()) {
          throw;
        }
        LOG(WARNING) << "Pinned host staging initialized only "
                     << windowSets_.size() << " of " << windowSetCount
                     << " requested window sets after later window-set "
                        "initialization failed: "
                     << error.what();
        break;
      } catch (...) {
        if (windowSets_.empty()) {
          throw;
        }
        LOG(WARNING) << "Pinned host staging initialized only "
                     << windowSets_.size() << " of " << windowSetCount
                     << " requested window sets after an unknown later "
                        "window-set initialization failure";
        break;
      }
    }

    // Publish the successfully initialized sets to the lease allocator only
    // after the allocation loop. If bookkeeping allocation itself fails, let
    // construction fail rather than exposing a set without a free-list entry.
    windowSetInUse_.resize(windowSets_.size(), false);
    for (uint32_t windowSetIndex = 0;
         windowSetIndex < static_cast<uint32_t>(windowSets_.size());
         ++windowSetIndex) {
      freeWindowSets_.push_back(windowSetIndex);
    }

    const auto cpuList = readProcStatusList("Cpus_allowed_list");
    const auto memoryNodeList = readProcStatusList("Mems_allowed_list");
    int cudaRuntimeVersion = 0;
    int cudaDriverVersion = 0;
    const auto runtimeVersionStatus =
        cudaRuntimeGetVersion(&cudaRuntimeVersion);
    const auto driverVersionStatus = cudaDriverGetVersion(&cudaDriverVersion);
    if (runtimeVersionStatus != cudaSuccess ||
        driverVersionStatus != cudaSuccess) {
      // Version reporting is diagnostic only. Do not leave a failed query as
      // the calling thread's last CUDA error.
      (void)cudaGetLastError();
    }
    uint64_t totalPinnedBytes = 0;
    for (const auto& windowSet : windowSets_) {
      for (const auto& window : windowSet->windows) {
        totalPinnedBytes += window.mappedBytes;
      }
    }
    LOG(INFO) << "Initialized " << windowSets_.size()
              << " pinned host staging window sets ("
              << PinnedStagingArena::kWindowCount << " windows per set, "
              << windowSets_.size() * PinnedStagingArena::kWindowCount
              << " windows total) of " << windowBytes_ << " bytes each; "
              << "total_pinned_bytes=" << totalPinnedBytes
              << ", total_pack_threads="
              << static_cast<uint64_t>(windowSets_.size()) * packThreads
              << ", allocation_strategy="
              << allocationStrategyName(allocationStrategy_) << ", numa_node="
              << (numaNode_.has_value() ? std::to_string(*numaNode_) : "none")
              << ", numa_policy_detail=" << numaPolicyDetail_
              << ", numa_binding_required="
              << (requireNumaLocal_ ? "true" : "false")
              << ", cudart_compile_version=" << CUDART_VERSION
              << ", cuda_runtime_version="
              << (runtimeVersionStatus == cudaSuccess ? cudaRuntimeVersion : 0)
              << ", cuda_driver_version="
              << (driverVersionStatus == cudaSuccess ? cudaDriverVersion : 0)
#if CUDART_VERSION >= 13000
              << ", native_cuda_memcpy_batch_async_compiled=true"
                 ", native_cuda_memcpy_batch_async_requires_non_default_"
                 "stream=true"
#else
              << ", native_cuda_memcpy_batch_async_compiled=false"
#endif
              << ", Cpus_allowed_list="
              << (cpuList.empty() ? "unknown" : cpuList)
              << ", Mems_allowed_list_cgroup_eligibility="
              << (memoryNodeList.empty() ? "unknown" : memoryNodeList);
    // Mems_allowed_list is the cgroup eligibility mask, not the process's
    // active allocation policy. The worker launcher records `numactl --show`
    // before exec; use its `policy` and `membind` fields to verify strict NUMA
    // placement instead of inferring policy from this diagnostic mask.
  }

  void allocatePortableWindow(Window& window) {
    void* data = nullptr;
    const auto status =
        cudaHostAlloc(&data, windowBytes_, cudaHostAllocDefault);
    if (status != cudaSuccess) {
      // Do not leave a failed allocation as the CUDA thread's last error when
      // the caller falls back to a smaller arena or pageable transfers.
      (void)cudaGetLastError();
      throw std::runtime_error(
          std::string("cudaHostAlloc failed for pinned staging: ") +
          cudaGetErrorString(status));
    }
    window.data = static_cast<uint8_t*>(data);
    window.mappedBytes = windowBytes_;
    window.strategy = HostAllocationStrategy::kCudaHostAlloc;

    // Initialization is intentionally lazy. Prefault and warm every page so
    // the first transfer does not pay that cost.
    std::memset(window.data, 0, windowBytes_);
  }

  void allocateNumaLocalWindow(
      Window& window,
      uint32_t numaNode,
      bool injectRegistrationFailure) {
#if defined(__linux__)
    const auto pageSizeResult = ::sysconf(_SC_PAGESIZE);
    if (pageSizeResult <= 0) {
      throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
    }
    const auto pageSize = static_cast<size_t>(pageSizeResult);
    if (windowBytes_ > std::numeric_limits<size_t>::max() - (pageSize - 1)) {
      throw std::runtime_error(
          "NUMA-local pinned staging window size overflows size_t");
    }
    const auto mappedBytes = static_cast<size_t>(
        (windowBytes_ + pageSize - 1) / pageSize * pageSize);
    void* mapping = ::mmap(
        nullptr,
        mappedBytes,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1,
        0);
    if (mapping == MAP_FAILED) {
      const auto error = errno;
      throw std::runtime_error(
          "mmap failed for NUMA-local pinned staging: " + systemError(error));
    }

    const auto releaseMapping = [&]() {
      if (::munmap(mapping, mappedBytes) != 0) {
        const auto error = errno;
        LOG(ERROR) << "munmap failed while unwinding NUMA-local pinned "
                      "staging initialization: "
                   << systemError(error);
      }
    };

    constexpr auto kBitsPerWord = sizeof(unsigned long) * CHAR_BIT;
    const auto words = static_cast<size_t>(numaNode / kBitsPerWord + 1);
    std::vector<unsigned long> mask(words, 0);
    mask[numaNode / kBitsPerWord] |= 1UL << (numaNode % kBitsPerWord);
    const auto maxNode = static_cast<unsigned long>(words * kBitsPerWord);
    if (::syscall(
            SYS_mbind,
            mapping,
            mappedBytes,
            MPOL_BIND,
            mask.data(),
            maxNode,
            MPOL_MF_STRICT) != 0) {
      const auto error = errno;
      releaseMapping();
      throw std::runtime_error(
          "mbind failed for NUMA-local pinned staging on node " +
          std::to_string(numaNode) + ": " + systemError(error));
    }

    bestEffortMappingAdvice(mapping, mappedBytes);

    // mmap must remain unpopulated until after mbind. Touch one byte in every
    // page to materialize the complete window under the VMA's strict binding
    // before CUDA pins and maps those pages for DMA.
    auto* touch = static_cast<volatile uint8_t*>(mapping);
    for (size_t offset = 0; offset < mappedBytes; offset += pageSize) {
      touch[offset] = 0;
    }
    touch[mappedBytes - 1] = 0;

    const auto registrationStatus = injectRegistrationFailure
        ? cudaErrorMemoryAllocation
        : cudaHostRegister(mapping, mappedBytes, cudaHostRegisterPortable);
    if (registrationStatus != cudaSuccess) {
      (void)cudaGetLastError();
      releaseMapping();
      throw std::runtime_error(
          std::string(
              "cudaHostRegister failed for NUMA-local pinned staging: ") +
          cudaGetErrorString(registrationStatus));
    }

    window.data = static_cast<uint8_t*>(mapping);
    window.mappedBytes = mappedBytes;
    window.strategy = HostAllocationStrategy::kMmapMbindCudaHostRegister;
#else
    VELOX_UNREACHABLE("NUMA-local pinned staging allocation is Linux-only");
#endif
  }

  ~Impl() {
    shutdown();
  }

  [[nodiscard]] std::optional<PinnedStagingArena::WindowSetLease> acquirePair(
      std::shared_ptr<Impl> self) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto ticket = nextTicket_++;
    const bool wasContended =
        ticket != servingTicket_ || freeWindowSets_.empty();
    ++waitingLeaseCount_;
    cv_.wait(lock, [this, ticket]() {
      return stopping_ ||
          (ticket == servingTicket_ && !freeWindowSets_.empty());
    });
    VELOX_DCHECK_GT(waitingLeaseCount_, 0);
    --waitingLeaseCount_;
    if (stopping_) {
      return std::nullopt;
    }

    const auto windowSetIndex = freeWindowSets_.front();
    freeWindowSets_.pop_front();
    VELOX_DCHECK_LT(windowSetIndex, windowSetInUse_.size());
    VELOX_DCHECK(!windowSetInUse_[windowSetIndex]);
    windowSetInUse_[windowSetIndex] = true;
    ++activeLeaseCount_;
    ++servingTicket_;
    cv_.notify_all();
    return PinnedStagingArena::WindowSetLease(
        std::move(self), windowSetIndex, activeLeaseCount_, wasContended);
  }

  void releasePair(uint32_t windowSetIndex) noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      VELOX_DCHECK_LT(windowSetIndex, windowSetInUse_.size());
      VELOX_DCHECK(windowSetInUse_[windowSetIndex]);
      VELOX_DCHECK_GT(activeLeaseCount_, 0);
      windowSetInUse_[windowSetIndex] = false;
      --activeLeaseCount_;
      freeWindowSets_.push_back(windowSetIndex);
    }
    cv_.notify_all();
  }

  [[nodiscard]] uint32_t waitingLeaseCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    return waitingLeaseCount_;
  }

  [[nodiscard]] uint8_t* data(uint32_t windowSetIndex, uint32_t windowIndex)
      const {
    VELOX_CHECK_LT(windowSetIndex, windowSets_.size());
    VELOX_CHECK_LT(windowIndex, PinnedStagingArena::kWindowCount);
    return windowSets_[windowSetIndex]->windows[windowIndex].data;
  }

  void pack(
      uint32_t windowSetIndex,
      uint32_t windowIndex,
      std::span<const PinnedStagingArena::Copy> copies) {
    VELOX_CHECK_LT(windowSetIndex, windowSets_.size());
    VELOX_CHECK_LT(windowIndex, PinnedStagingArena::kWindowCount);
    auto& window = windowSets_[windowSetIndex]->windows[windowIndex];
    auto& packPool = *windowSets_[windowSetIndex]->packPool;
    std::lock_guard<std::mutex> windowLock(window.packMutex);

    // Validate every destination before scheduling any copy, so a malformed
    // later range cannot leave the window partially modified.
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    ranges.reserve(copies.size());
    for (const auto& copy : copies) {
      VELOX_CHECK_LE(copy.destinationOffset, windowBytes_);
      VELOX_CHECK_LE(copy.size, windowBytes_ - copy.destinationOffset);
      if (copy.size == 0) {
        continue;
      }
      VELOX_CHECK_NOT_NULL(copy.source);
      ranges.emplace_back(
          copy.destinationOffset, copy.destinationOffset + copy.size);
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) {
      VELOX_CHECK_LE(
          ranges[i - 1].second,
          ranges[i].first,
          "Pinned staging pack destinations must not overlap");
    }
    if (ranges.empty()) {
      return;
    }

    std::vector<PackJob::Work> work;
    for (const auto& copy : copies) {
      uint64_t copied = 0;
      while (copied < copy.size) {
        const auto bytes = std::min<uint64_t>(
            PinnedStagingArena::kPackQuantumBytes, copy.size - copied);
        work.push_back(
            PackJob::Work{
                static_cast<const uint8_t*>(copy.source) + copied,
                window.data + copy.destinationOffset + copied,
                bytes});
        copied += bytes;
      }
    }

    const auto numWorkers = static_cast<uint32_t>(
        std::min<size_t>(packPool.numThreads(), work.size()));
    auto job = std::make_shared<PackJob>(std::move(work), numWorkers);
    uint32_t scheduled = 0;
    try {
      for (; scheduled < numWorkers; ++scheduled) {
        packPool.add([job]() {
          job->copyAll();
          job->complete();
        });
      }
    } catch (...) {
      job->complete(numWorkers - scheduled);
      if (scheduled != 0) {
        job->wait();
      }
      throw;
    }
    job->wait();
  }

  void shutdown() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    cv_.notify_all();
  }

  static std::mutex globalMutex;
  static Config config;
  static std::shared_ptr<Impl> current;
  static bool initializationAttempted;
  static uint32_t failAllocationAt;
  static uint32_t failRegistrationAt;
  static bool hasNumaNodeOverride;
  static std::optional<uint32_t> numaNodeOverride;
  static bool hasRequireNumaLocalOverride;
  static bool requireNumaLocalOverride;

  const uint64_t windowBytes_;
  HostAllocationStrategy allocationStrategy_{
      HostAllocationStrategy::kUninitialized};
  std::optional<uint32_t> numaNode_;
  std::string numaPolicyDetail_;
  bool requireNumaLocal_{false};
  std::vector<std::unique_ptr<WindowSet>> windowSets_;
  std::deque<uint32_t> freeWindowSets_;
  std::vector<bool> windowSetInUse_;
  std::mutex mutex_;
  std::condition_variable cv_;
  uint64_t nextTicket_{0};
  uint64_t servingTicket_{0};
  uint32_t waitingLeaseCount_{0};
  uint32_t activeLeaseCount_{0};
  bool stopping_{false};
};

std::mutex PinnedStagingArena::Impl::globalMutex;
PinnedStagingArena::Impl::Config PinnedStagingArena::Impl::config;
std::shared_ptr<PinnedStagingArena::Impl> PinnedStagingArena::Impl::current;
bool PinnedStagingArena::Impl::initializationAttempted{false};
uint32_t PinnedStagingArena::Impl::failAllocationAt{0};
uint32_t PinnedStagingArena::Impl::failRegistrationAt{0};
bool PinnedStagingArena::Impl::hasNumaNodeOverride{false};
std::optional<uint32_t> PinnedStagingArena::Impl::numaNodeOverride;
bool PinnedStagingArena::Impl::hasRequireNumaLocalOverride{false};
bool PinnedStagingArena::Impl::requireNumaLocalOverride{false};

PinnedStagingArena::WindowSetLease::WindowSetLease(
    std::shared_ptr<Impl> arena,
    uint32_t windowSetIndex,
    uint32_t activeLeasesAtAcquire,
    bool wasContended)
    : arena_(std::move(arena)),
      windowSetIndex_(windowSetIndex),
      activeLeasesAtAcquire_(activeLeasesAtAcquire),
      wasContended_(wasContended) {}

PinnedStagingArena::WindowSetLease::~WindowSetLease() {
  release();
}

PinnedStagingArena::WindowSetLease::WindowSetLease(
    WindowSetLease&& other) noexcept
    : arena_(std::move(other.arena_)),
      windowSetIndex_(other.windowSetIndex_),
      activeLeasesAtAcquire_(other.activeLeasesAtAcquire_),
      wasContended_(other.wasContended_) {}

PinnedStagingArena::WindowSetLease&
PinnedStagingArena::WindowSetLease::operator=(WindowSetLease&& other) noexcept {
  if (this != &other) {
    release();
    arena_ = std::move(other.arena_);
    windowSetIndex_ = other.windowSetIndex_;
    activeLeasesAtAcquire_ = other.activeLeasesAtAcquire_;
    wasContended_ = other.wasContended_;
  }
  return *this;
}

uint8_t* PinnedStagingArena::WindowSetLease::data(uint32_t windowIndex) const {
  VELOX_CHECK_NOT_NULL(arena_);
  return arena_->data(windowSetIndex_, windowIndex);
}

uint64_t PinnedStagingArena::WindowSetLease::capacity() const {
  VELOX_CHECK_NOT_NULL(arena_);
  return arena_->windowBytes_;
}

uint32_t PinnedStagingArena::WindowSetLease::windowSetCount() const {
  VELOX_CHECK_NOT_NULL(arena_);
  return static_cast<uint32_t>(arena_->windowSets_.size());
}

uint32_t PinnedStagingArena::WindowSetLease::activeLeasesAtAcquire() const {
  VELOX_CHECK_NOT_NULL(arena_);
  return activeLeasesAtAcquire_;
}

bool PinnedStagingArena::WindowSetLease::wasContended() const {
  VELOX_CHECK_NOT_NULL(arena_);
  return wasContended_;
}

void PinnedStagingArena::WindowSetLease::pack(
    uint32_t windowIndex,
    std::span<const Copy> copies) const {
  VELOX_CHECK_NOT_NULL(arena_);
  arena_->pack(windowSetIndex_, windowIndex, copies);
}

void PinnedStagingArena::WindowSetLease::release() {
  if (arena_ != nullptr) {
    auto arena = std::move(arena_);
    arena->releasePair(windowSetIndex_);
  }
}

void PinnedStagingArena::configure(
    bool enabled,
    uint64_t windowBytes,
    uint32_t packThreads,
    uint32_t windowSetCount) {
  if (enabled) {
    VELOX_USER_CHECK_GT(
        windowBytes, 0, "Pinned staging window must be nonzero");
    VELOX_USER_CHECK_LE(
        windowBytes,
        std::numeric_limits<size_t>::max() / kWindowCount,
        "Pinned staging windows do not fit size_t");
    VELOX_USER_CHECK_GT(
        packThreads, 0, "Pinned staging pack thread count must be nonzero");
    VELOX_USER_CHECK_GT(
        windowSetCount, 0, "Pinned staging window-set count must be nonzero");
    const auto windowSetBytes = windowBytes * kWindowCount;
    VELOX_USER_CHECK_LE(
        windowSetCount,
        std::numeric_limits<size_t>::max() / windowSetBytes,
        "Pinned staging arena does not fit size_t");
  }

  std::shared_ptr<Impl> old;
  {
    std::lock_guard<std::mutex> lock(Impl::globalMutex);
    old = std::move(Impl::current);
    Impl::config =
        Impl::Config{enabled, windowBytes, packThreads, windowSetCount};
    Impl::initializationAttempted = false;
  }
  if (old != nullptr) {
    old->shutdown();
  }
}

bool PinnedStagingArena::enabled() {
  std::lock_guard<std::mutex> lock(Impl::globalMutex);
  return Impl::config.enabled;
}

std::optional<PinnedStagingArena::WindowSetLease>
PinnedStagingArena::acquirePair() {
  std::shared_ptr<Impl> arena;
  {
    std::lock_guard<std::mutex> lock(Impl::globalMutex);
    if (!Impl::config.enabled || Impl::initializationAttempted) {
      arena = Impl::current;
      if (arena == nullptr) {
        return std::nullopt;
      }
    } else {
      Impl::initializationAttempted = true;
      try {
        Impl::current = std::make_shared<Impl>(
            Impl::config.windowBytes,
            Impl::config.packThreads,
            Impl::config.windowSetCount,
            Impl::failAllocationAt,
            Impl::failRegistrationAt,
            Impl::numaNodeOverride,
            Impl::hasNumaNodeOverride,
            Impl::requireNumaLocalOverride,
            Impl::hasRequireNumaLocalOverride);
      } catch (const std::exception& error) {
        LOG(ERROR) << "Disabling pinned host staging after initialization "
                      "failure: "
                   << error.what();
        return std::nullopt;
      } catch (...) {
        LOG(ERROR) << "Disabling pinned host staging after an unknown "
                      "initialization failure";
        return std::nullopt;
      }
      arena = Impl::current;
    }
  }
  return arena->acquirePair(std::move(arena));
}

void PinnedStagingArena::reset() {
  std::shared_ptr<Impl> old;
  {
    std::lock_guard<std::mutex> lock(Impl::globalMutex);
    old = std::move(Impl::current);
    Impl::config = Impl::Config{};
    Impl::initializationAttempted = false;
  }
  if (old != nullptr) {
    old->shutdown();
  }
}

void PinnedStagingArena::setAllocationFailureForTesting(bool fail) {
  setAllocationFailureAtForTesting(fail ? 1 : 0);
}

void PinnedStagingArena::setAllocationFailureAtForTesting(
    uint32_t allocationOrdinal) {
  reset();
  std::lock_guard<std::mutex> lock(Impl::globalMutex);
  Impl::failAllocationAt = allocationOrdinal;
}

void PinnedStagingArena::setNumaNodeForTesting(
    std::optional<uint32_t> numaNode) {
  reset();
  std::lock_guard<std::mutex> lock(Impl::globalMutex);
  Impl::hasNumaNodeOverride = true;
  Impl::numaNodeOverride = numaNode;
}

void PinnedStagingArena::clearNumaNodeForTesting() {
  reset();
  std::lock_guard<std::mutex> lock(Impl::globalMutex);
  Impl::hasNumaNodeOverride = false;
  Impl::numaNodeOverride.reset();
}

void PinnedStagingArena::setRequireNumaLocalForTesting(bool required) {
  reset();
  std::lock_guard<std::mutex> lock(Impl::globalMutex);
  Impl::hasRequireNumaLocalOverride = true;
  Impl::requireNumaLocalOverride = required;
}

void PinnedStagingArena::clearRequireNumaLocalForTesting() {
  reset();
  std::lock_guard<std::mutex> lock(Impl::globalMutex);
  Impl::hasRequireNumaLocalOverride = false;
  Impl::requireNumaLocalOverride = false;
}

std::optional<uint32_t> PinnedStagingArena::resolveNumaNodeForTesting(
    bool relativeNodes,
    std::span<const uint32_t> policyNodes,
    std::span<const uint32_t> allowedNodes) {
  return resolveSingleNodePolicy(relativeNodes, policyNodes, allowedNodes).node;
}

void PinnedStagingArena::setRegistrationFailureAtForTesting(
    uint32_t registrationOrdinal) {
  reset();
  std::lock_guard<std::mutex> lock(Impl::globalMutex);
  Impl::failRegistrationAt = registrationOrdinal;
}

PinnedStagingArena::HostAllocationStrategy
PinnedStagingArena::hostAllocationStrategyForTesting() {
  std::lock_guard<std::mutex> lock(Impl::globalMutex);
  return Impl::current == nullptr ? HostAllocationStrategy::kUninitialized
                                  : Impl::current->allocationStrategy_;
}

uint32_t PinnedStagingArena::waitingLeaseCountForTesting() {
  std::shared_ptr<Impl> arena;
  {
    std::lock_guard<std::mutex> lock(Impl::globalMutex);
    arena = Impl::current;
  }
  return arena == nullptr ? 0 : arena->waitingLeaseCount();
}

} // namespace facebook::velox::cudf_velox::connector::hive
