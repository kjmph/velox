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
#pragma once

#include <cudf/contiguous_split.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <cinttypes>
#include <memory>
#include "velox/common/base/Exceptions.h"
#include "velox/common/future/VeloxPromise.h"

namespace facebook::velox::ucx_exchange {

/// Struct that bundles a packed_table with the CUDA stream that was used
/// to allocate its memory. This allows the receiver to reuse the same stream
/// for subsequent operations on the data.
struct PackedTableWithStream {
  std::unique_ptr<cudf::packed_table> packedTable;
  rmm::cuda_stream_view stream;

  PackedTableWithStream() = default;
  PackedTableWithStream(
      std::unique_ptr<cudf::packed_table>&& table,
      rmm::cuda_stream_view s)
      : packedTable(std::move(table)), stream(s) {}

  /// Returns the size of the GPU data buffer, or 0 if packedTable is null.
  size_t gpuDataSize() const {
    return packedTable ? packedTable->data.gpu_data->size() : 0;
  }
};

using PackedTableWithStreamPtr = std::unique_ptr<PackedTableWithStream>;

class UcxExchangeQueue {
 public:
  explicit UcxExchangeQueue(
      int32_t numberOfConsumers,
      uint64_t receiveHighWaterBytes = 0)
      : numberOfConsumers_{numberOfConsumers},
        receiveHighWaterBytes_{receiveHighWaterBytes},
        receiveLowWaterBytes_{(receiveHighWaterBytes * kContinuePct) / 100} {
    VELOX_CHECK_GE(numberOfConsumers, 1);
  }

  ~UcxExchangeQueue() {
    clearAllPromises();
  }

  std::mutex& mutex() {
    return mutex_;
  }

  bool empty() const {
    return queue_.empty();
  }

  /// Enqueues 'data' to the queue. One random promise(top of promise queue)
  /// associated with the future that is waiting for the data from the queue
  /// is returned in 'promises' if 'data' is not nullptr. When 'data' is
  /// nullptr and the queue is completed serving data, all left over promises
  /// will be returned in 'promises'. When 'data' is nullptr and the queue is
  /// not completed serving data, no 'promises' will be added and returned.
  void enqueueLocked(
      PackedTableWithStreamPtr&& data,
      std::vector<ContinuePromise>& promises);

  /// If data is permanently not available, e.g. the source cannot be
  /// contacted, this registers an error message and causes the reading
  /// Exchanges to throw with the message.
  void setError(std::string_view error);

  bool isInError() {
    return !error_.empty();
  }
  /// Returns a PackedTableWithStream object.
  ///
  /// Returns a nullptr if no data is available. If data is still expected,
  /// sets 'atEnd' to false and 'future' to a Future that will complete when
  /// data arrives. If no more data is expected, sets 'atEnd' to true. Returns
  /// one PackedTableWithStream if data is available.
  /// It's possible that the same consumer is already waiting for data. In this
  /// case, a stalePromise is returned which needs to be cleaned up.
  PackedTableWithStreamPtr dequeueLocked(
      int consumerId,
      bool* atEnd,
      ContinueFuture* future,
      ContinuePromise* stalePromise);

  int32_t size() const {
    return queue_.size();
  }

  /// Returns the total bytes held by packed tables in 'this'.
  int64_t totalBytes() const {
    return totalBytes_;
  }

  uint64_t retainedReceiveBytesLocked() const {
    return totalBytes_ + reservedBytes_ + inFlightBytes_;
  }

  uint64_t queuedReceiveBytesLocked() const {
    return totalBytes_ + reservedBytes_;
  }

  uint64_t receiveHighWaterBytes() const {
    return receiveHighWaterBytes_;
  }

  uint64_t receiveLowWaterBytes() const {
    return receiveLowWaterBytes_;
  }

  bool receiveBytesAtHighWaterLocked() const {
    return receiveHighWaterBytes_ > 0 &&
        queuedReceiveBytesLocked() >= receiveHighWaterBytes_;
  }

  bool receiveBytesBelowLowWaterLocked() const {
    return receiveHighWaterBytes_ == 0 ||
        queuedReceiveBytesLocked() <= receiveLowWaterBytes_;
  }

  bool tryReserveReceiveBytesLocked(uint64_t bytes);

  void releaseReceiveBytesLocked(uint64_t bytes);

  void releaseInFlightReceiveBytesLocked(uint64_t bytes);

  /// Returns the maximum retained receive bytes.
  uint64_t peakBytes() const {
    return peakBytes_;
  }

  /// Returns total number of packed tables received from all sources.
  uint64_t receivedTables() const {
    return receivedTables_;
  }

  /// Returns an average size of tables. Returns 0 if hasn't received
  /// any tables yet.
  uint64_t averageReceivedTablesBytes() const {
    return receivedTables_ > 0 ? receivedBytes_ / receivedTables_ : 0;
  }

  void addSourceLocked() {
    VELOX_CHECK(!noMoreSources_, "addSource called after noMoreSources");
    numSources_++;
  }

  void noMoreSources();

  void close();

 private:
  std::vector<ContinuePromise> closeLocked() {
    queue_.clear();
    totalBytes_ = 0;
    reservedBytes_ = 0;
    return clearAllPromisesLocked();
  }

  std::vector<ContinuePromise> checkCompleteLocked() {
    if (noMoreSources_ && numCompleted_ == numSources_) {
      atEnd_ = true;
      return clearAllPromisesLocked();
    }
    return {};
  }

  void addPromiseLocked(
      int consumerId,
      ContinueFuture* future,
      ContinuePromise* stalePromise);

  void clearAllPromises() {
    std::vector<ContinuePromise> promises;
    {
      std::lock_guard<std::mutex> l(mutex_);
      promises = clearAllPromisesLocked();
    }
    clearPromises(promises);
  }

  std::vector<ContinuePromise> clearAllPromisesLocked() {
    std::vector<ContinuePromise> promises;
    promises.reserve(promises_.size());
    auto it = promises_.begin();
    while (it != promises_.end()) {
      promises.push_back(std::move(it->second));
      it = promises_.erase(it);
    }
    VELOX_CHECK(promises_.empty());
    return promises;
  }

  static void clearPromises(std::vector<ContinuePromise>& promises) {
    for (auto& promise : promises) {
      promise.setValue();
    }
  }

  static constexpr uint32_t kContinuePct{70};

  const int32_t numberOfConsumers_;

  int numCompleted_{0};
  int numSources_{0};
  bool noMoreSources_{false};
  bool atEnd_{false};

  std::mutex mutex_;
  std::deque<PackedTableWithStreamPtr> queue_;
  // The map from consumer id to the waiting promise
  folly::F14FastMap<int, ContinuePromise> promises_;

  // When set, all promises will be realized and the next dequeue will
  // throw an exception with this message.
  std::string error_;
  // Total size of packed tables in queue.
  int64_t totalBytes_{0};
  // Bytes reserved for posted or pending UCX receives not yet enqueued.
  int64_t reservedBytes_{0};
  // Bytes dequeued by an Exchange operator but still retained by downstream
  // CudfVectors that own the packed table.
  int64_t inFlightBytes_{0};
  int64_t inFlightTables_{0};
  const uint64_t receiveHighWaterBytes_{0};
  const uint64_t receiveLowWaterBytes_{0};
  // Number of packed tables received.
  int64_t receivedTables_{0};
  // Total size of packed tables received. Used to calculate an average
  // expected size.
  int64_t receivedBytes_{0};
  // Maximum value of retained receive bytes.
  int64_t peakBytes_{0};
  // Peak queue size (number of items). Used for high-water-mark alerts.
  int64_t peakSize_{0};
};

} // namespace facebook::velox::ucx_exchange
