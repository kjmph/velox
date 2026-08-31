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
#include <condition_variable>
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

bool isSingleNodeList(std::string_view nodes) {
  return !nodes.empty() && nodes.find(',') == std::string_view::npos &&
      nodes.find('-') == std::string_view::npos;
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
      if (data != nullptr) {
        const auto status = cudaFreeHost(data);
        if (status != cudaSuccess) {
          LOG(ERROR) << "cudaFreeHost failed for a pinned staging window: "
                     << cudaGetErrorString(status);
        }
      }
    }

    Window() = default;
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    uint8_t* data{nullptr};
    std::mutex packMutex;
  };

  struct Config {
    bool enabled{false};
    uint64_t windowBytes{0};
    uint32_t packThreads{0};
  };

  Impl(uint64_t windowBytes, uint32_t packThreads, bool forceAllocationFailure)
      : windowBytes_(windowBytes), packPool_(packThreads) {
    for (auto& window : windows_) {
      void* data = nullptr;
      const auto status = forceAllocationFailure
          ? cudaErrorMemoryAllocation
          : cudaHostAlloc(&data, windowBytes_, cudaHostAllocDefault);
      if (status != cudaSuccess) {
        // Preserve the returned status for diagnostics, but do not leave a
        // failed allocation as the CUDA thread's last error when the caller
        // falls back to pageable transfers.
        (void)cudaGetLastError();
        throw std::runtime_error(
            std::string("cudaHostAlloc failed for pinned staging: ") +
            cudaGetErrorString(status));
      }
      window.data = static_cast<uint8_t*>(data);

      // Initialization is intentionally lazy. Prefault and warm every page so
      // the first transfer does not pay that cost. NUMA placement itself is
      // determined by cudaHostAlloc under the worker process's memory policy.
      std::memset(window.data, 0, windowBytes_);
    }

    const auto cpuList = readProcStatusList("Cpus_allowed_list");
    const auto memoryNodeList = readProcStatusList("Mems_allowed_list");
    LOG(INFO) << "Initialized " << windows_.size()
              << " pinned host staging windows of " << windowBytes_
              << " bytes each; Cpus_allowed_list="
              << (cpuList.empty() ? "unknown" : cpuList)
              << ", Mems_allowed_list="
              << (memoryNodeList.empty() ? "unknown" : memoryNodeList);
    if (!isSingleNodeList(memoryNodeList)) {
      LOG(WARNING)
          << "Pinned host staging is not restricted to one NUMA memory node; "
             "launch the GPU worker with a GPU-local memory binding for "
             "predictable cache-to-GPU bandwidth";
    }
  }

  ~Impl() {
    shutdown();
  }

  [[nodiscard]] std::optional<PinnedStagingArena::WindowSetLease> acquirePair(
      std::shared_ptr<Impl> self) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto ticket = nextTicket_++;
    cv_.wait(lock, [this, ticket]() {
      return stopping_ || (ticket == servingTicket_ && !windowsInUse_);
    });
    if (stopping_) {
      return std::nullopt;
    }

    windowsInUse_ = true;
    ++servingTicket_;
    cv_.notify_all();
    return PinnedStagingArena::WindowSetLease(std::move(self));
  }

  void releasePair() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      VELOX_DCHECK(windowsInUse_);
      windowsInUse_ = false;
    }
    cv_.notify_all();
  }

  [[nodiscard]] uint8_t* data(uint32_t windowIndex) const {
    VELOX_CHECK_LT(windowIndex, windows_.size());
    return windows_[windowIndex].data;
  }

  void pack(
      uint32_t windowIndex,
      std::span<const PinnedStagingArena::Copy> copies) {
    VELOX_CHECK_LT(windowIndex, windows_.size());
    std::lock_guard<std::mutex> windowLock(windows_[windowIndex].packMutex);

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
                windows_[windowIndex].data + copy.destinationOffset + copied,
                bytes});
        copied += bytes;
      }
    }

    const auto numWorkers = static_cast<uint32_t>(
        std::min<size_t>(packPool_.numThreads(), work.size()));
    auto job = std::make_shared<PackJob>(std::move(work), numWorkers);
    uint32_t scheduled = 0;
    try {
      for (; scheduled < numWorkers; ++scheduled) {
        packPool_.add([job]() {
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
  static bool forceAllocationFailure;

  const uint64_t windowBytes_;
  PackThreadPool packPool_;
  std::array<Window, PinnedStagingArena::kWindowCount> windows_;
  std::mutex mutex_;
  std::condition_variable cv_;
  uint64_t nextTicket_{0};
  uint64_t servingTicket_{0};
  bool windowsInUse_{false};
  bool stopping_{false};
};

std::mutex PinnedStagingArena::Impl::globalMutex;
PinnedStagingArena::Impl::Config PinnedStagingArena::Impl::config;
std::shared_ptr<PinnedStagingArena::Impl> PinnedStagingArena::Impl::current;
bool PinnedStagingArena::Impl::initializationAttempted{false};
bool PinnedStagingArena::Impl::forceAllocationFailure{false};

PinnedStagingArena::WindowSetLease::WindowSetLease(std::shared_ptr<Impl> arena)
    : arena_(std::move(arena)) {}

PinnedStagingArena::WindowSetLease::~WindowSetLease() {
  release();
}

PinnedStagingArena::WindowSetLease::WindowSetLease(
    WindowSetLease&& other) noexcept
    : arena_(std::move(other.arena_)) {}

PinnedStagingArena::WindowSetLease&
PinnedStagingArena::WindowSetLease::operator=(WindowSetLease&& other) noexcept {
  if (this != &other) {
    release();
    arena_ = std::move(other.arena_);
  }
  return *this;
}

uint8_t* PinnedStagingArena::WindowSetLease::data(uint32_t windowIndex) const {
  VELOX_CHECK_NOT_NULL(arena_);
  return arena_->data(windowIndex);
}

uint64_t PinnedStagingArena::WindowSetLease::capacity() const {
  VELOX_CHECK_NOT_NULL(arena_);
  return arena_->windowBytes_;
}

void PinnedStagingArena::WindowSetLease::pack(
    uint32_t windowIndex,
    std::span<const Copy> copies) const {
  VELOX_CHECK_NOT_NULL(arena_);
  arena_->pack(windowIndex, copies);
}

void PinnedStagingArena::WindowSetLease::release() {
  if (arena_ != nullptr) {
    auto arena = std::move(arena_);
    arena->releasePair();
  }
}

void PinnedStagingArena::configure(
    bool enabled,
    uint64_t windowBytes,
    uint32_t packThreads) {
  if (enabled) {
    VELOX_USER_CHECK_GT(
        windowBytes, 0, "Pinned staging window must be nonzero");
    VELOX_USER_CHECK_LE(
        windowBytes,
        std::numeric_limits<size_t>::max() / kWindowCount,
        "Pinned staging windows do not fit size_t");
    VELOX_USER_CHECK_GT(
        packThreads, 0, "Pinned staging pack thread count must be nonzero");
  }

  std::shared_ptr<Impl> old;
  {
    std::lock_guard<std::mutex> lock(Impl::globalMutex);
    old = std::move(Impl::current);
    Impl::config = Impl::Config{enabled, windowBytes, packThreads};
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
            Impl::forceAllocationFailure);
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
  reset();
  std::lock_guard<std::mutex> lock(Impl::globalMutex);
  Impl::forceAllocationFailure = fail;
}

} // namespace facebook::velox::cudf_velox::connector::hive
