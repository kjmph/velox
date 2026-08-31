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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"
#include "velox/experimental/cudf/connectors/hive/PinnedStagingArena.h"

#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/CacheInputStream.h"

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/types.hpp>

#include <rmm/cuda_device.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <future>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace {
using DeviceReadRequest = cudf::io::datasource::device_read_request;
using facebook::velox::IoStats;
using facebook::velox::RuntimeCounter;
using facebook::velox::cudf_velox::connector::hive::PinnedStagingArena;
using facebook::velox::dwio::common::BufferedInput;
using facebook::velox::dwio::common::CachedRegion;
using facebook::velox::dwio::common::CacheInputStream;

constexpr size_t kMaximumCopiesPerBatch = 32;
constexpr size_t kMinimumPinnedStagingBytes = 1ULL << 20;

const std::string kDeviceReadBatches = "cudfBufferedDeviceReadBatches";
const std::string kDeviceReadRequests = "cudfBufferedDeviceReadRequests";
const std::string kDeviceReadBytes = "cudfBufferedDeviceReadBytes";
const std::string kDeviceReadFragments = "cudfBufferedDeviceReadFragments";
const std::string kCacheBackedSourceBytes = "cudfCacheBackedSourceBytes";
const std::string kCopiedSourceBytes = "cudfCopiedSourceBytes";
const std::string kStagingAttempts = "cudfPinnedStagingAttempts";
const std::string kStagingTransfers = "cudfPinnedStagingTransfers";
const std::string kStagingBytes = "cudfPinnedStagingBytes";
const std::string kStagingWindows = "cudfPinnedStagingWindows";
const std::string kStagingAcquireNanos = "cudfPinnedStagingAcquireNanos";
const std::string kStagingPackNanos = "cudfPinnedStagingPackNanos";
const std::string kStagingH2DWaitNanos = "cudfPinnedStagingH2DWaitNanos";
const std::string kStagingFallbacks = "cudfPinnedStagingFallbacks";
const std::string kStagingDisabledBypasses =
    "cudfPinnedStagingDisabledBypasses";
const std::string kStagingSmallReadBypasses =
    "cudfPinnedStagingSmallReadBypasses";
const std::string kDirectH2DBytes = "cudfDirectHostToDeviceBytes";

void addIoCounter(
    const std::shared_ptr<IoStats>& ioStats,
    const std::string& name,
    uint64_t value,
    RuntimeCounter::Unit unit = RuntimeCounter::Unit::kNone) {
  if (ioStats == nullptr) {
    return;
  }
  ioStats->addCounter(
      name, RuntimeCounter(facebook::velox::saturateCast(value), unit));
}

uint64_t elapsedNanos(std::chrono::steady_clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}

int getStreamDevice(rmm::cuda_stream_view stream) {
  int device{};
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
  CUDF_CUDA_TRY(cudaStreamGetDevice(stream.value(), &device));
#else
  // CUDA versions before 12.8 cannot query a stream's device. Retain the
  // historical requirement that the stream belongs to the current device.
  CUDF_CUDA_TRY(cudaGetDevice(&device));
#endif
  return device;
}

void synchronizeStream(rmm::cuda_stream_view stream, int device) {
  auto const deviceScope =
      rmm::cuda_set_device_raii{rmm::cuda_device_id{device}};
  try {
    stream.synchronize();
  } catch (...) {
    const auto primaryError = std::current_exception();
    // Returning while a host buffer may still be in use would be unsafe. A
    // device-wide fence is the last recoverable fallback.
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::terminate();
    }
    std::rethrow_exception(primaryError);
  }
}

// std::future returned by an executor does not wait when discarded. Wrapping
// it in a deferred future whose captured state drains in its destructor makes
// discard obey the datasource contract: once the outer future is gone, no
// destination or retained host source remains in use.
template <typename T>
class DrainingCompletion {
 public:
  explicit DrainingCompletion(std::future<T> completion)
      : completion_(std::move(completion)) {}

  DrainingCompletion(DrainingCompletion&& other) noexcept
      : completion_(std::move(other.completion_)) {}

  DrainingCompletion(const DrainingCompletion&) = delete;
  DrainingCompletion& operator=(const DrainingCompletion&) = delete;
  DrainingCompletion& operator=(DrainingCompletion&&) = delete;

  ~DrainingCompletion() noexcept {
    if (completion_.valid()) {
      try {
        std::ignore = completion_.get();
      } catch (...) {
      }
    }
  }

  T get() {
    return completion_.get();
  }

 private:
  std::future<T> completion_;
};

template <typename T>
std::future<T> makeDrainingFuture(std::future<T> completion) {
  auto state = DrainingCompletion<T>{std::move(completion)};
  return std::async(
      std::launch::deferred,
      [state = std::move(state)]() mutable { return state.get(); });
}

class CudaEvent {
 public:
  CudaEvent() {
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }

  ~CudaEvent() {
    if (event_ != nullptr) {
      // Event destruction releases bookkeeping only. Completion has already
      // been established before this object leaves the transfer routine.
      std::ignore = cudaEventDestroy(event_);
    }
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  cudaEvent_t get() const {
    return event_;
  }

  void record(rmm::cuda_stream_view stream) {
    CUDF_CUDA_TRY(cudaEventRecord(event_, stream.value()));
    recorded_ = true;
  }

  void synchronize() {
    if (!recorded_) {
      return;
    }
    CUDF_CUDA_TRY(cudaEventSynchronize(event_));
    recorded_ = false;
  }

 private:
  cudaEvent_t event_{nullptr};
  bool recorded_{false};
};

// Owns every host source referenced by a batch of H2D descriptors. Cache hits
// retain cache pins directly; the fallback owns copied pageable buffers. The
// submission layer is intentionally independent of source ownership: the same
// prepared plan can use bounded double-buffered pinned staging or safely fall
// back to direct pageable copies without changing lifetime/completion rules.
class HostToDeviceTransferPlan {
 public:
  explicit HostToDeviceTransferPlan(std::shared_ptr<IoStats> ioStats)
      : ioStats_(std::move(ioStats)) {}

  void addCachedRegion(CachedRegion region, uint8_t* destination) {
    VELOX_CHECK_LE(
        region.size(),
        std::numeric_limits<size_t>::max() - cachedSourceBytes_,
        "Cached source byte count overflows size_t");
    cachedSourceBytes_ += region.size();
    const auto ownerIndex = retainedRegions_.size();
    retainedRegions_.emplace_back(std::move(region));
    retainedDescriptorCounts_.push_back(0);
    size_t destinationOffset = 0;
    for (const auto range : retainedRegions_.back()->ranges()) {
      addDescriptor(
          destination + destinationOffset,
          range.data(),
          range.size(),
          SourceOwner{SourceOwnerKind::kCached, ownerIndex});
      ++retainedDescriptorCounts_.back();
      destinationOffset += range.size();
    }
    VELOX_CHECK_EQ(
        destinationOffset,
        retainedRegions_.back()->size(),
        "Cached region ranges do not cover the retained region");
    VELOX_CHECK_GT(
        retainedDescriptorCounts_.back(),
        0,
        "A nonempty cached region must contain at least one range");
  }

  void addCopiedRegion(std::vector<uint8_t> region, uint8_t* destination) {
    if (region.empty()) {
      return;
    }
    VELOX_CHECK_LE(
        region.size(),
        std::numeric_limits<size_t>::max() - copiedSourceBytes_,
        "Copied source byte count overflows size_t");
    copiedSourceBytes_ += region.size();
    const auto ownerIndex = copiedRegions_.size();
    copiedRegions_.push_back(std::move(region));
    copiedDescriptorCounts_.push_back(1);
    addDescriptor(
        destination,
        copiedRegions_.back().data(),
        copiedRegions_.back().size(),
        SourceOwner{SourceOwnerKind::kCopied, ownerIndex});
  }

  void submitAndWait(
      rmm::cuda_stream_view stream,
      int device,
      std::optional<PinnedStagingArena::WindowSetLease> windows) {
    if (destinations_.empty()) {
      return;
    }

    auto const deviceScope =
        rmm::cuda_set_device_raii{rmm::cuda_device_id{device}};
    addIoCounter(ioStats_, kDeviceReadFragments, destinations_.size());
    if (cachedSourceBytes_ != 0) {
      addIoCounter(
          ioStats_,
          kCacheBackedSourceBytes,
          cachedSourceBytes_,
          RuntimeCounter::Unit::kBytes);
    }
    if (copiedSourceBytes_ != 0) {
      addIoCounter(
          ioStats_,
          kCopiedSourceBytes,
          copiedSourceBytes_,
          RuntimeCounter::Unit::kBytes);
    }
    if (windows.has_value()) {
      addIoCounter(ioStats_, kStagingTransfers, 1);
      addIoCounter(
          ioStats_, kStagingBytes, totalBytes_, RuntimeCounter::Unit::kBytes);
      submitStagedAndWait(*windows, stream, device);
      return;
    }

    addIoCounter(
        ioStats_, kDirectH2DBytes, totalBytes_, RuntimeCounter::Unit::kBytes);
    submitDirectAndWait(stream, device);
  }

 private:
  enum class SourceOwnerKind : uint8_t { kCached, kCopied };

  struct SourceOwner {
    SourceOwnerKind kind;
    size_t index;
  };

  struct Cursor {
    size_t descriptor{0};
    size_t offset{0};
  };

  struct WindowBatch {
    std::vector<PinnedStagingArena::Copy> packCopies;
    std::vector<void*> destinations;
    std::vector<const void*> sources;
    std::vector<size_t> sizes;
    std::vector<size_t> completedDescriptors;
    size_t bytes{0};
  };

  void submitDirectAndWait(rmm::cuda_stream_view stream, int device) {
    // Allocate the tail fence before submitting any copy. A failure here is
    // therefore a synchronous scheduling failure with no source lifetime to
    // drain.
    CudaEvent tailEvent;
    std::exception_ptr scheduleError;

    for (size_t begin = 0; begin < destinations_.size();
         begin += kMaximumCopiesPerBatch) {
      const auto count =
          std::min(kMaximumCopiesPerBatch, destinations_.size() - begin);
      try {
        CUDF_CUDA_TRY(
            cudf::detail::memcpy_batch_async(
                destinations_.data() + begin,
                sources_.data() + begin,
                sizes_.data() + begin,
                count,
                stream));
      } catch (...) {
        scheduleError = std::current_exception();
        break;
      }
    }

    // One event after the final successfully submitted group is sufficient to
    // keep every retained pin/buffer alive until all preceding copies finish.
    // Even after a scheduling error the CUDA call may have submitted a prefix,
    // so always attempt the tail fence and conservatively drain on failure.
    std::exception_ptr fenceError;
    const auto recordStatus = cudaEventRecord(tailEvent.get(), stream.value());
    if (recordStatus == cudaSuccess) {
      const auto synchronizeStatus = cudaEventSynchronize(tailEvent.get());
      if (synchronizeStatus != cudaSuccess) {
        try {
          CUDF_CUDA_TRY(synchronizeStatus);
        } catch (...) {
          fenceError = std::current_exception();
        }
      }
    } else {
      try {
        CUDF_CUDA_TRY(recordStatus);
      } catch (...) {
        fenceError = std::current_exception();
      }
    }

    if (fenceError != nullptr) {
      try {
        synchronizeStream(stream, device);
      } catch (...) {
        // synchronizeStream either established device completion before
        // throwing or terminated because source lifetime could not be proven.
      }
    }
    if (scheduleError != nullptr) {
      std::rethrow_exception(scheduleError);
    }
    if (fenceError != nullptr) {
      std::rethrow_exception(fenceError);
    }
  }

  WindowBatch makeWindowBatch(Cursor& cursor, uint8_t* staging, size_t capacity)
      const {
    WindowBatch batch;
    while (cursor.descriptor < sizes_.size() && batch.bytes < capacity) {
      const auto descriptor = cursor.descriptor;
      VELOX_DCHECK_LT(cursor.offset, sizes_[descriptor]);
      const auto bytes =
          std::min(sizes_[descriptor] - cursor.offset, capacity - batch.bytes);
      VELOX_CHECK_GT(bytes, 0);

      const auto* source =
          static_cast<const uint8_t*>(sources_[descriptor]) + cursor.offset;
      auto* destination =
          static_cast<uint8_t*>(destinations_[descriptor]) + cursor.offset;
      auto* stagedSource = staging + batch.bytes;
      batch.packCopies.push_back(
          PinnedStagingArena::Copy{source, batch.bytes, bytes});

      const auto canCoalesce = !batch.sizes.empty() &&
          static_cast<uint8_t*>(batch.destinations.back()) +
                  batch.sizes.back() ==
              destination &&
          static_cast<const uint8_t*>(batch.sources.back()) +
                  batch.sizes.back() ==
              stagedSource;
      if (canCoalesce) {
        batch.sizes.back() += bytes;
      } else {
        batch.destinations.push_back(destination);
        batch.sources.push_back(stagedSource);
        batch.sizes.push_back(bytes);
      }

      batch.bytes += bytes;
      cursor.offset += bytes;
      if (cursor.offset == sizes_[descriptor]) {
        batch.completedDescriptors.push_back(descriptor);
        ++cursor.descriptor;
        cursor.offset = 0;
      }
    }
    VELOX_CHECK_GT(batch.bytes, 0, "Pinned staging made no transfer progress");
    return batch;
  }

  void releasePackedSources(const std::vector<size_t>& descriptors) {
    for (const auto descriptor : descriptors) {
      VELOX_CHECK_LT(descriptor, owners_.size());
      const auto owner = owners_[descriptor];
      if (owner.kind == SourceOwnerKind::kCached) {
        VELOX_CHECK_LT(owner.index, retainedDescriptorCounts_.size());
        auto& remaining = retainedDescriptorCounts_[owner.index];
        VELOX_CHECK_GT(remaining, 0);
        if (--remaining == 0) {
          retainedRegions_[owner.index].reset();
        }
      } else {
        VELOX_CHECK_LT(owner.index, copiedDescriptorCounts_.size());
        auto& remaining = copiedDescriptorCounts_[owner.index];
        VELOX_CHECK_GT(remaining, 0);
        if (--remaining == 0) {
          std::vector<uint8_t>{}.swap(copiedRegions_[owner.index]);
        }
      }
    }
  }

  static void submitWindow(
      const WindowBatch& batch,
      CudaEvent& event,
      rmm::cuda_stream_view stream,
      bool& cudaWorkMayBePending) {
    VELOX_CHECK(!batch.destinations.empty());
    cudaWorkMayBePending = true;
    for (size_t begin = 0; begin < batch.destinations.size();
         begin += kMaximumCopiesPerBatch) {
      const auto count =
          std::min(kMaximumCopiesPerBatch, batch.destinations.size() - begin);
      CUDF_CUDA_TRY(
          cudf::detail::memcpy_batch_async(
              batch.destinations.data() + begin,
              batch.sources.data() + begin,
              batch.sizes.data() + begin,
              count,
              stream));
    }
    event.record(stream);
  }

  void submitStagedAndWait(
      PinnedStagingArena::WindowSetLease& windows,
      rmm::cuda_stream_view stream,
      int device) {
    std::array<CudaEvent, PinnedStagingArena::kWindowCount> completionEvents;
    std::array<bool, PinnedStagingArena::kWindowCount> submitted{};
    Cursor cursor;
    bool cudaWorkMayBePending = false;

    try {
      size_t batchIndex = 0;
      while (cursor.descriptor < sizes_.size()) {
        const auto windowIndex = batchIndex % PinnedStagingArena::kWindowCount;
        // Before rolling onto a window again, establish that its previous H2D
        // copy is complete. The other window remains in flight while host pack
        // threads fill this one.
        if (submitted[windowIndex]) {
          const auto waitStart = std::chrono::steady_clock::now();
          completionEvents[windowIndex].synchronize();
          addIoCounter(
              ioStats_,
              kStagingH2DWaitNanos,
              elapsedNanos(waitStart),
              RuntimeCounter::Unit::kNanos);
          submitted[windowIndex] = false;
        }

        auto batch = makeWindowBatch(
            cursor,
            windows.data(windowIndex),
            static_cast<size_t>(windows.capacity()));
        const auto packStart = std::chrono::steady_clock::now();
        windows.pack(windowIndex, batch.packCopies);
        addIoCounter(
            ioStats_,
            kStagingPackNanos,
            elapsedNanos(packStart),
            RuntimeCounter::Unit::kNanos);
        addIoCounter(ioStats_, kStagingWindows, 1);
        // pack() is the source-lifetime boundary: cache pins and owned
        // pageable fallbacks are no longer needed once their last fragment is
        // resident in the leased pinned window.
        releasePackedSources(batch.completedDescriptors);
        submitWindow(
            batch, completionEvents[windowIndex], stream, cudaWorkMayBePending);
        submitted[windowIndex] = true;
        ++batchIndex;
      }

      for (uint32_t windowIndex = 0;
           windowIndex < PinnedStagingArena::kWindowCount;
           ++windowIndex) {
        if (submitted[windowIndex]) {
          const auto waitStart = std::chrono::steady_clock::now();
          completionEvents[windowIndex].synchronize();
          addIoCounter(
              ioStats_,
              kStagingH2DWaitNanos,
              elapsedNanos(waitStart),
              RuntimeCounter::Unit::kNanos);
        }
      }
      windows.release();
    } catch (...) {
      const auto primaryError = std::current_exception();
      if (cudaWorkMayBePending) {
        try {
          synchronizeStream(stream, device);
        } catch (...) {
          // synchronizeStream establishes device completion before throwing,
          // or terminates if safe window reuse cannot be proven.
        }
      }
      windows.release();
      std::rethrow_exception(primaryError);
    }
  }

  void addDescriptor(
      void* destination,
      const void* source,
      size_t size,
      SourceOwner owner) {
    if (size == 0) {
      return;
    }
    VELOX_CHECK_NOT_NULL(destination);
    VELOX_CHECK_NOT_NULL(source);
    VELOX_CHECK_LE(
        size,
        std::numeric_limits<size_t>::max() - totalBytes_,
        "Host-to-device transfer size overflows size_t");
    destinations_.push_back(destination);
    sources_.push_back(source);
    sizes_.push_back(size);
    owners_.push_back(owner);
    totalBytes_ += size;
  }

  std::vector<std::optional<CachedRegion>> retainedRegions_;
  std::vector<size_t> retainedDescriptorCounts_;
  std::vector<std::vector<uint8_t>> copiedRegions_;
  std::vector<size_t> copiedDescriptorCounts_;
  std::vector<void*> destinations_;
  std::vector<const void*> sources_;
  std::vector<size_t> sizes_;
  std::vector<SourceOwner> owners_;
  std::shared_ptr<IoStats> ioStats_;
  size_t cachedSourceBytes_{0};
  size_t copiedSourceBytes_{0};
  size_t totalBytes_{0};
};

std::vector<size_t> executeDeviceReadBatch(
    const std::shared_ptr<BufferedInput>& input,
    const std::shared_ptr<std::mutex>& inputMutex,
    const std::vector<DeviceReadRequest>& requests,
    size_t fileSize,
    rmm::cuda_stream_view stream,
    int device,
    const std::shared_ptr<IoStats>& ioStats) {
  addIoCounter(ioStats, kDeviceReadBatches, 1);
  addIoCounter(ioStats, kDeviceReadRequests, requests.size());
  std::vector<size_t> results(requests.size());
  std::vector<
      std::unique_ptr<facebook::velox::dwio::common::SeekableInputStream>>
      inputStreams(requests.size());
  HostToDeviceTransferPlan transfer(ioStats);
  std::optional<PinnedStagingArena::WindowSetLease> stagingWindows;
  size_t totalReadBytes = 0;

  {
    // BufferedInput stores enqueue/load bookkeeping in the input object. Keep
    // batches for the same input atomic while allowing unrelated files and
    // drivers to progress concurrently.
    std::lock_guard<std::mutex> lock(*inputMutex);
    bool hasReads = false;
    for (size_t index = 0; index < requests.size(); ++index) {
      const auto& request = requests[index];
      // Deliberately avoid offset + size: the mathematical end can exceed
      // size_t, while datasource semantics still allow a short (or empty)
      // result. Subtract only after proving that offset is inside the file.
      const auto readSize = request.offset < fileSize
          ? std::min(request.size, fileSize - request.offset)
          : 0;
      results[index] = readSize;
      if (readSize == 0) {
        continue;
      }
      VELOX_CHECK_LE(
          readSize,
          std::numeric_limits<size_t>::max() - totalReadBytes,
          "Total buffered device read size overflows size_t");
      totalReadBytes += readSize;
      hasReads = true;
      inputStreams[index] = input->enqueue({request.offset, readSize});
      VELOX_CHECK_NOT_NULL(
          inputStreams[index], "BufferedInput::enqueue returned null stream");
    }

    if (hasReads) {
      // CachedBufferedInput may schedule coalesced loads asynchronously, while
      // a singleton demand region may not be scheduled at all. The Next()/
      // readFully() materialization below is the authoritative completion
      // barrier for both cases, and it still runs before staging is acquired.
      input->load(facebook::velox::dwio::common::LogType::FILE);
    }

    addIoCounter(
        ioStats,
        kDeviceReadBytes,
        totalReadBytes,
        RuntimeCounter::Unit::kBytes);

    for (size_t index = 0; index < requests.size(); ++index) {
      const auto readSize = results[index];
      if (readSize == 0) {
        continue;
      }

      auto* cacheStream =
          dynamic_cast<CacheInputStream*>(inputStreams[index].get());
      if (cacheStream == nullptr) {
        std::vector<uint8_t> copied(readSize);
        inputStreams[index]->readFully(
            reinterpret_cast<char*>(copied.data()), readSize);
        transfer.addCopiedRegion(std::move(copied), requests[index].dst);
        continue;
      }

      size_t copiedBytes = 0;
      while (copiedBytes < readSize) {
        const void* data = nullptr;
        int runSize = 0;
        VELOX_CHECK(
            cacheStream->Next(&data, &runSize),
            "Cached input ended after {} of {} bytes",
            copiedBytes,
            readSize);
        VELOX_CHECK_GT(runSize, 0, "Cached input returned an empty run");
        VELOX_CHECK_LE(
            static_cast<size_t>(runSize),
            readSize - copiedBytes,
            "Cached input returned bytes beyond the requested region");

        auto retained = cacheStream->retainedRegionForLastNext();
        VELOX_CHECK_EQ(
            retained.size(),
            static_cast<size_t>(runSize),
            "Retained cache region does not match Next() result");
        VELOX_CHECK_EQ(
            retained.ranges().front().data(),
            data,
            "Retained cache region does not begin at the Next() result");
        transfer.addCachedRegion(
            std::move(retained), requests[index].dst + copiedBytes);
        copiedBytes += runSize;
      }
    }

    // The transfer plan now owns an independent pin for every cache fragment
    // (or an owned copy for non-cache input), so release the input streams and
    // their original pins before H2D begins. Staged transfers release each
    // owner after its last fragment is packed; the direct fallback retains the
    // owners until its CUDA completion fence.
    inputStreams.clear();
  }

  // Only reserve the bounded pinned arena after all storage work is complete
  // and the transfer plan owns exact cache pins or private copied buffers.
  // AsyncDataCache entries can be exclusive, stale-sized, cancelled, or
  // evicted between a load barrier and Next(); preparing sources first closes
  // that residency gap and guarantees no remote I/O can hold both windows.
  if (totalReadBytes >= kMinimumPinnedStagingBytes &&
      PinnedStagingArena::enabled()) {
    addIoCounter(ioStats, kStagingAttempts, 1);
    const auto acquireStart = std::chrono::steady_clock::now();
    stagingWindows = PinnedStagingArena::acquirePair();
    addIoCounter(
        ioStats,
        kStagingAcquireNanos,
        elapsedNanos(acquireStart),
        RuntimeCounter::Unit::kNanos);
    if (!stagingWindows.has_value()) {
      addIoCounter(ioStats, kStagingFallbacks, 1);
    }
  } else if (totalReadBytes >= kMinimumPinnedStagingBytes) {
    addIoCounter(ioStats, kStagingDisabledBypasses, 1);
  } else if (totalReadBytes != 0) {
    addIoCounter(ioStats, kStagingSmallReadBypasses, 1);
  }

  transfer.submitAndWait(stream, device, std::move(stagingWindows));
  return results;
}

// Keeps the datasource alive while cuDF drains its completion future. The
// explicit destructor is important when the outer deferred future is discarded:
// lambda-capture destruction order alone cannot provide this lifetime rule.
class RetainedReadCompletion {
 public:
  RetainedReadCompletion(
      std::shared_ptr<cudf::io::datasource> dataSource,
      std::future<void> completion)
      : dataSource_(std::move(dataSource)),
        completion_(std::move(completion)) {}

  RetainedReadCompletion(RetainedReadCompletion&& other) noexcept
      : dataSource_(std::move(other.dataSource_)),
        completion_(std::move(other.completion_)) {}

  RetainedReadCompletion(const RetainedReadCompletion&) = delete;
  RetainedReadCompletion& operator=(const RetainedReadCompletion&) = delete;
  RetainedReadCompletion& operator=(RetainedReadCompletion&&) = delete;

  ~RetainedReadCompletion() noexcept {
    if (completion_.valid()) {
      try {
        completion_.get();
      } catch (...) {
      }
    }
  }

  void get() {
    completion_.get();
  }

 private:
  std::shared_ptr<cudf::io::datasource> dataSource_;
  std::future<void> completion_;
};

class BufferedReadCompletion {
 public:
  BufferedReadCompletion(
      std::shared_ptr<cudf::io::datasource> dataSource,
      std::future<std::vector<size_t>> completion,
      std::vector<size_t> expectedSizes)
      : dataSource_(std::move(dataSource)),
        completion_(std::move(completion)),
        expectedSizes_(std::move(expectedSizes)) {}

  BufferedReadCompletion(BufferedReadCompletion&& other) noexcept
      : dataSource_(std::move(other.dataSource_)),
        completion_(std::move(other.completion_)),
        expectedSizes_(std::move(other.expectedSizes_)) {}

  BufferedReadCompletion(const BufferedReadCompletion&) = delete;
  BufferedReadCompletion& operator=(const BufferedReadCompletion&) = delete;
  BufferedReadCompletion& operator=(BufferedReadCompletion&&) = delete;

  ~BufferedReadCompletion() noexcept {
    if (completion_.valid()) {
      try {
        std::ignore = completion_.get();
      } catch (...) {
      }
    }
  }

  void get() {
    const auto actualSizes = completion_.get();
    VELOX_CHECK_EQ(
        actualSizes.size(),
        expectedSizes_.size(),
        "Buffered device batch returned the wrong number of results");
    for (size_t index = 0; index < actualSizes.size(); ++index) {
      VELOX_CHECK_EQ(
          actualSizes[index],
          expectedSizes_[index],
          "Buffered device read was unexpectedly short");
    }
  }

 private:
  std::shared_ptr<cudf::io::datasource> dataSource_;
  std::future<std::vector<size_t>> completion_;
  std::vector<size_t> expectedSizes_;
};
} // namespace

namespace facebook::velox::cudf_velox::connector::hive {

std::string normalizeKvikioUri(std::string_view path) {
  constexpr std::string_view kS3aPrefix = "s3a://";
  constexpr std::string_view kS3nPrefix = "s3n://";
  if (path.starts_with(kS3aPrefix) || path.starts_with(kS3nPrefix)) {
    return "s3://" + std::string(path.substr(kS3aPrefix.size()));
  }
  return std::string(path);
}

BufferedInputDataSource::BufferedInputDataSource(
    std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input,
    std::shared_ptr<facebook::velox::IoStats> ioStats)
    : input_(std::move(input)),
      ioStats_(std::move(ioStats)),
      fileSize_(input_->getReadFile()->size()) {}

size_t BufferedInputDataSource::size() const {
  return fileSize_;
}

std::unique_ptr<cudf::io::datasource::buffer>
BufferedInputDataSource::host_read(size_t offset, size_t size) {
  if (offset >= fileSize_) {
    return cudf::io::datasource::buffer::create(std::vector<uint8_t>{});
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  std::vector<uint8_t> data(readSize);
  readContiguous(offset, readSize, data.data());
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t
BufferedInputDataSource::host_read(size_t offset, size_t size, uint8_t* dst) {
  if (offset >= fileSize_) {
    return 0;
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  readContiguous(offset, readSize, dst);
  return readSize;
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
BufferedInputDataSource::host_read_async(size_t offset, size_t size) {
  return std::async(std::launch::deferred, [this, offset, size]() {
    return this->host_read(offset, size);
  });
}

std::future<size_t> BufferedInputDataSource::host_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  return std::async(std::launch::deferred, [this, offset, size, dst]() {
    return this->host_read(offset, size, dst);
  });
}

std::unique_ptr<cudf::io::datasource::buffer>
BufferedInputDataSource::device_read(
    size_t offset,
    size_t size,
    rmm::cuda_stream_view stream) {
  const auto readSize =
      offset < fileSize_ ? std::min(size, fileSize_ - offset) : 0;
  rmm::device_buffer result(
      readSize, stream, rmm::mr::get_current_device_resource_ref());
  const auto bytesRead = device_read(
      offset, readSize, static_cast<uint8_t*>(result.data()), stream);
  result.resize(bytesRead, stream);
  return datasource::buffer::create(std::move(result));
}

size_t BufferedInputDataSource::device_read(
    size_t offset,
    size_t size,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  return device_read_async(offset, size, dst, stream).get();
}

std::future<size_t> BufferedInputDataSource::device_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  const DeviceReadRequest request{offset, size, dst};
  auto completion = device_read_batch_async(
      cudf::host_span<DeviceReadRequest const>{&request, 1}, stream);
  return std::async(
      std::launch::deferred, [completion = std::move(completion)]() mutable {
        auto results = completion.get();
        VELOX_CHECK_EQ(results.size(), 1);
        return results.front();
      });
}

std::future<std::vector<size_t>>
BufferedInputDataSource::device_read_batch_async(
    cudf::host_span<DeviceReadRequest const> requests,
    rmm::cuda_stream_view stream) {
  // Validate the complete descriptor list before scheduling any destination
  // access, and copy it because the caller's span expires on return.
  std::vector<DeviceReadRequest> copiedRequests(
      requests.begin(), requests.end());
  for (const auto& request : copiedRequests) {
    VELOX_CHECK(
        request.size == 0 || request.dst != nullptr,
        "A nonempty device read requires a non-null destination");
  }
  if (copiedRequests.empty()) {
    return std::async(
        std::launch::deferred, [] { return std::vector<size_t>{}; });
  }

  const auto device = getStreamDevice(stream);
  auto const deviceScope =
      rmm::cuda_set_device_raii{rmm::cuda_device_id{device}};
  auto completion = cudf::detail::host_worker_pool().submit_task(
      [input = input_,
       inputMutex = inputMutex_,
       ioStats = ioStats_,
       requests = std::move(copiedRequests),
       fileSize = fileSize_,
       stream,
       device]() {
        auto const taskDeviceScope =
            rmm::cuda_set_device_raii{rmm::cuda_device_id{device}};
        return executeDeviceReadBatch(
            input, inputMutex, requests, fileSize, stream, device, ioStats);
      });
  return makeDrainingFuture(std::move(completion));
}

bool BufferedInputDataSource::supports_device_read() const {
  return true;
}

void BufferedInputDataSource::readContiguous(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  using namespace facebook::velox::dwio::common;
  // read() consults BufferedInput's current merged-region state, which load()
  // replaces. Serialize host/footer reads with device batches for this input.
  std::lock_guard<std::mutex> lock(*inputMutex_);
  // BufferedInput::read gives us a stream over the exact region.
  auto stream = input_->read(offset, size, LogType::FILE);
  VELOX_CHECK(stream != nullptr, "read() returned null stream");
  stream->readFully(reinterpret_cast<char*>(dst), size);
}

std::tuple<
    std::vector<rmm::device_buffer>,
    std::vector<cudf::device_span<const uint8_t>>,
    std::future<void>>
fetchByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto bufferedInput = dynamic_cast<BufferedInputDataSource*>(dataSource.get());
  if (bufferedInput == nullptr) {
    // cuDF owns the generic KvikIO/host implementation. Delegating avoids a
    // second copy of its coalescing, short-read validation, failure draining,
    // and host-buffer lifetime rules at the Velox boundary.
    auto [buffers, spans, completion] =
        cudf::io::parquet::fetch_byte_ranges_to_device_async(
            *dataSource, byteRanges, stream, mr);
    auto completionState =
        RetainedReadCompletion{std::move(dataSource), std::move(completion)};
    auto retainedCompletion = std::async(
        std::launch::deferred,
        [completionState = std::move(completionState)]() mutable {
          completionState.get();
        });
    return {
        std::move(buffers), std::move(spans), std::move(retainedCompletion)};
  }

  // Pad buffer sizes to be a multiple of 8 bytes. Required by
  // `decode_page_data_kernel` in cuDF Parquet reader.
  constexpr auto kBufferPaddingMultiple = 8;

  // Allocate device spans for each column chunk
  std::vector<cudf::device_span<const uint8_t>> columnChunkData{};
  columnChunkData.reserve(byteRanges.size());

  // Validate before arithmetic or allocation. byte_range_info uses signed
  // fields, so accumulating an invalid negative size into size_t would
  // otherwise underflow before the request builder rejected it.
  std::vector<size_t> rangeOffsets;
  std::vector<size_t> rangeSizes;
  rangeOffsets.reserve(byteRanges.size());
  rangeSizes.reserve(byteRanges.size());
  size_t totalSize = 0;
  for (const auto& byteRange : byteRanges) {
    const auto offset = byteRange.offset();
    const auto size = byteRange.size();
    VELOX_CHECK_GE(offset, 0, "Device read offset must be nonnegative");
    VELOX_CHECK_GE(size, 0, "Device read size must be nonnegative");
    VELOX_CHECK_LE(
        static_cast<uint64_t>(offset),
        std::numeric_limits<size_t>::max(),
        "Device read offset does not fit size_t");
    VELOX_CHECK_LE(
        static_cast<uint64_t>(size),
        std::numeric_limits<size_t>::max(),
        "Device read size does not fit size_t");
    const auto requestSize = static_cast<size_t>(size);
    VELOX_CHECK_LE(
        requestSize,
        std::numeric_limits<size_t>::max() - totalSize,
        "Total device read size overflows size_t");
    rangeOffsets.push_back(static_cast<size_t>(offset));
    rangeSizes.push_back(requestSize);
    totalSize += requestSize;
  }
  VELOX_CHECK_LE(
      totalSize,
      std::numeric_limits<size_t>::max() - (kBufferPaddingMultiple - 1),
      "Padded device read size overflows size_t");

  // Allocate single device buffer for all column chunks
  std::vector<rmm::device_buffer> columnChunkBuffers{};
  columnChunkBuffers.emplace_back(
      cudf::util::round_up_safe<size_t>(totalSize, kBufferPaddingMultiple),
      stream,
      mr);

  // Compute device spans for each column chunk
  auto bufferData = static_cast<uint8_t*>(columnChunkBuffers.back().data());
  std::ignore = std::accumulate(
      rangeSizes.begin(),
      rangeSizes.end(),
      std::size_t{0},
      [&](auto acc, const auto rangeSize) {
        // A nonempty list of empty ranges has a zero-byte device allocation,
        // whose data pointer is null. Even adding zero to null is undefined.
        auto* rangeData = bufferData == nullptr ? nullptr : bufferData + acc;
        columnChunkData.emplace_back(rangeData, static_cast<size_t>(rangeSize));
        return acc + rangeSize;
      });

  // Submit one immutable batch. This shares the regular cuDF device-read path
  // and avoids mutable cross-call enqueue/rollback state at the hybrid reader
  // boundary.
  std::vector<DeviceReadRequest> requests;
  std::vector<size_t> expectedSizes;
  requests.reserve(byteRanges.size());
  expectedSizes.reserve(byteRanges.size());
  for (size_t index = 0; index < byteRanges.size(); ++index) {
    requests.push_back(
        {rangeOffsets[index],
         rangeSizes[index],
         const_cast<uint8_t*>(columnChunkData[index].data())});
    expectedSizes.push_back(rangeSizes[index]);
  }

  auto batchCompletion = cudf::io::device_read_batch_async(
      *bufferedInput,
      cudf::host_span<DeviceReadRequest const>{
          requests.data(), requests.size()},
      stream);
  auto completionState = BufferedReadCompletion{
      std::move(dataSource),
      std::move(batchCompletion),
      std::move(expectedSizes)};
  auto completion = std::async(
      std::launch::deferred,
      [completionState = std::move(completionState)]() mutable {
        completionState.get();
      });
  return {
      std::move(columnChunkBuffers),
      std::move(columnChunkData),
      std::move(completion)};
}

} // namespace facebook::velox::cudf_velox::connector::hive
