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

#include <folly/io/IOBuf.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include "velox/core/PlanNode.h"
#include "velox/exec/OutputBuffer.h" // for the Stats structure
#include "velox/exec/Task.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowShm.h"

/// CPU RowVector mirror of UcxQueues.h. Structure intentionally tracks
/// the cudf version closely so future merging/templating is mechanical.
/// Differences vs cudf:
///   - Payload type is std::shared_ptr<UcxCpuRowPayload> (IOBuf chain plus
///     row/byte counters) instead of std::shared_ptr<cudf::packed_columns>.
///   - No GPU stream / RMM concerns.

namespace facebook::velox::ucx_exchange {

struct UcxCpuRowShmSlotLease {
  std::shared_ptr<UcxCpuRowShmSlotPool> pool;
  UcxCpuRowShmSlotPool::Slot slot;
};

/// One serialized RowVector chunk plus enough metadata for the queue to
/// report stats and apply backpressure without walking the IOBuf chain.
///
/// `data` holds the bytes produced by PrestoVectorSerde::flush(); typically
/// a multi-buffer IOBuf chain. When CPU SHM direct-TX is enabled, `data`
/// wraps `shmSegment` so UCX fallback can still read the bytes without
/// reserializing. `numBytes` is the total chain length cached at enqueue time.
/// `numRows` is the row count of the original RowVector before serialization.
struct UcxCpuRowPayload {
  ~UcxCpuRowPayload();

  std::unique_ptr<folly::IOBuf> data;
  std::shared_ptr<UcxCpuRowShmSegment> shmSegment;
  std::shared_ptr<UcxCpuRowShmSlotPool> shmSlotPool;
  size_t shmOffset{0};
  uint32_t shmSlotId{std::numeric_limits<uint32_t>::max()};
  bool releaseShmSlotOnDestroy{false};
  int32_t numRows{0};
  int64_t numBytes{0};
};

/// Callback fired when data becomes available on a destination queue. A
/// nullptr `data` indicates end-of-stream. The CPU-row exchange does not
/// use the cudf `remainingBytes` prefetch contract, so the vector is empty.
using UcxCpuRowDataAvailableCallback = std::function<void(
    std::shared_ptr<UcxCpuRowPayload> data,
    std::vector<int64_t> remainingBytes)>;

struct UcxCpuRowDataAvailable {
  UcxCpuRowDataAvailableCallback callback{nullptr};
  std::shared_ptr<UcxCpuRowPayload> data;
  std::vector<int64_t> remainingBytes;

  void notify() {
    if (callback) {
      callback(std::move(data), std::move(remainingBytes));
    }
  }
};

/// Per-destination FIFO of serialized RowVector chunks for one source
/// task. Producers (UcxCpuRowPartitionedOutput) call enqueueBack;
/// consumers (UcxCpuRowExchangeServer) call getData. Mirrors
/// UcxDestinationQueue.
class UcxCpuRowDestinationQueue {
 public:
  struct Stats {
    void recordEnqueue(const UcxCpuRowPayload* payload);
    void recordDequeue(const UcxCpuRowPayload* payload);
    void recordRequeueFront(const UcxCpuRowPayload* payload);

    exec::DestinationBuffer::Stats toOutputBufferStats() const;

    bool finished{false};

    int64_t bytesQueued{0};
    int64_t rowsQueued{0};
    int64_t payloadsQueued{0};

    int64_t bytesSent{0};
    int64_t rowsSent{0};
    int64_t payloadsSent{0};
  };

  /// Enqueues at the back. The data field is `std::shared_ptr` so that
  /// the same payload can be referenced from multiple destination queues
  /// in broadcast mode without copying.
  void enqueueBack(std::shared_ptr<UcxCpuRowPayload> data);

  /// Enqueues at the front; used when a transfer fails and the payload
  /// needs to be retried.
  void enqueueFront(std::shared_ptr<UcxCpuRowPayload> data);

  struct Data {
    std::shared_ptr<UcxCpuRowPayload> data;
    std::vector<int64_t> remainingBytes;
    /// True when the result is delivered synchronously via the return
    /// value rather than the notify callback.
    bool immediate{false};
  };

  /// Pulls the head of the queue. If empty, installs `notify` to fire
  /// when data arrives and returns Data{nullptr, {}, false}.
  [[nodiscard]] Data getData(UcxCpuRowDataAvailableCallback notify);

  /// Drops every queued payload. Used when the query is aborted.
  void deleteResults();

  /// Synchronous, no-callback variant of getData. Returns the head of
  /// the queue if a payload is immediately available; nullptr if the
  /// queue is empty OR the next entry is the end-of-stream marker
  /// (caller must consume the marker via the regular getData path).
  /// Caller must hold the parent UcxCpuRowOutputQueue's mutex.
  std::shared_ptr<UcxCpuRowPayload> tryDequeueLocked();

  /// Returns and clears the pending notify callback (if any), packaged
  /// alongside the data that should be passed to it.
  UcxCpuRowDataAvailable getAndClearNotify();

  /// Marks this destination as drained; used to populate finishing
  /// stats reported up to the OutputBuffer view.
  void finish();

  Stats stats() const;

  std::string toString();

 private:
  void clearNotify();

  std::deque<std::shared_ptr<UcxCpuRowPayload>> queue_;
  UcxCpuRowDataAvailableCallback notify_{nullptr};
  Stats stats_;
};

/// Top-level queue for one source task. Owns one
/// UcxCpuRowDestinationQueue per partition (or one shared list of
/// destinations for broadcast).
class UcxCpuRowOutputQueue
    : public std::enable_shared_from_this<UcxCpuRowOutputQueue> {
 public:
  UcxCpuRowOutputQueue(
      std::shared_ptr<exec::Task> task,
      uint32_t numDestinations,
      uint32_t numDrivers,
      core::PartitionedOutputNode::Kind kind =
          core::PartitionedOutputNode::Kind::kPartitioned);

  /// Late-initialization variant. Mirrors UcxOutputQueue::initialize:
  /// supports the case where a downstream getData() arrives before the
  /// producing task has registered. The queue manager creates a
  /// placeholder UcxCpuRowOutputQueue, stores the callback, and calls
  /// initialize() once the source task starts.
  bool initialize(
      std::shared_ptr<exec::Task> task,
      uint32_t numDestinations,
      uint32_t numDrivers,
      core::PartitionedOutputNode::Kind kind =
          core::PartitionedOutputNode::Kind::kPartitioned);

  core::PartitionedOutputNode::Kind kind() const {
    return kind_;
  }

  /// Acquire-load read of `initialized_`, paired with the release-store
  /// at the end of initialize(). Lock-free read so other queries can
  /// quickly check whether the queue is ready.
  bool isInitialized() const {
    return initialized_.load(std::memory_order_acquire);
  }

  /// For grouped execution, the producing driver count is finalized
  /// after splits arrive. This lets the queue manager update it.
  void updateNumDrivers(uint32_t newNumDrivers);

  /// Append the payload to `destination`'s FIFO. Caller transfers ownership.
  void enqueue(
      int destination,
      std::unique_ptr<UcxCpuRowPayload> data,
      int32_t numRows);

  /// After a flush, returns true (and populates `future`) if total
  /// queued bytes have crossed the high-water mark and producers should
  /// block until the consumer drains.
  bool checkBlocked(ContinueFuture* future);

  /// Async pop from `destination`. If a payload is available, `notify`
  /// is invoked synchronously (and `getData` returns). Otherwise the
  /// callback is installed and fires when data arrives.
  void getData(int destination, UcxCpuRowDataAvailableCallback notify);

  /// Non-blocking pop from `destination`. Returns the next payload if
  /// one is immediately available; nullptr otherwise. Used for
  /// server-side bundling; after the first chunk arrives via getData()
  /// the server drains additional chunks via tryGetData() until either
  /// the bundle target is hit or the queue is empty. Does NOT register
  /// a notify callback when the queue is empty.
  std::shared_ptr<UcxCpuRowPayload> tryGetData(int destination);

  /// Put a payload back at the head of a destination queue after an
  /// optimistic tryGetData() drain. Used by the exchange server when
  /// preserving transport-homogeneous bundles.
  void requeueFront(int destination, std::shared_ptr<UcxCpuRowPayload> data);

  std::optional<UcxCpuRowShmSlotLease> tryAcquireSlot(
      int destination,
      size_t bytes);

  /// Indicates that one driver finished producing. The queue closes
  /// once all drivers report and all queued data has drained.
  void noMoreData();

  /// Used by broadcast mode to add new destinations after start. New
  /// destinations are backfilled with previously-seen broadcast
  /// payloads.
  void updateOutputBuffers(int numBuffers, bool noMoreBuffers);

  bool isFinished();
  bool isFinishedLocked();

  /// Drops all queued payloads for `destination` and makes subsequent
  /// getData() calls return immediately with empty data.
  void deleteResults(int destination);

  /// Wakes up any waiting producers. Used when the task is aborted.
  void terminate();

  std::string toString();

  /// Shoehorns UcxCpuRowOutputQueue stats into the standard Velox
  /// OutputBuffer::Stats so the existing Presto stat-reporting plumbing
  /// works without modification.
  exec::OutputBuffer::Stats stats();

 private:
  // Backpressure: UCX drains large bundles, so waking all producers at 90%
  // leaves too little headroom and causes immediate re-blocking. Resume at
  // half full to keep the pipe fed without a producer herd.
  static constexpr int32_t kContinuePct = 50;

  void updateStatsWithEnqueuedLocked(int64_t bytes, int64_t rows);
  void updateStatsWithFreedLocked(
      int64_t bytes,
      int64_t numPayloads,
      std::vector<ContinuePromise>& promises);
  void recordDirectHandoffLocked(const UcxCpuRowDataAvailable& notification);
  void acknowledgeDirectHandoffLocked(int64_t bytes, int64_t numPayloads);
  void reconcileQueuedStatsLocked(const char* reason);
  void maybeUnblockProducersLocked(std::vector<ContinuePromise>& promises);
  void updateTotalQueuedBytesMsLocked();
  int64_t getAverageQueueTimeMsLocked() const;

  void noMoreDrivers();
  void checkIfDone(bool oneDriverFinished);

  bool enqueuePartitionedOutputLocked(
      int destination,
      std::shared_ptr<UcxCpuRowPayload> data,
      std::vector<UcxCpuRowDataAvailable>& dataAvailableCbs);

  void enqueueBroadcastOutputLocked(
      std::shared_ptr<UcxCpuRowPayload> data,
      std::vector<UcxCpuRowDataAvailable>& dataAvailableCbs);

  uint32_t expectedSlotPoolOpenersLocked(int destination) const;
  void maybeStartSlotPoolCreateLocked(
      int destination,
      size_t bytes,
      uint32_t expectedOpeners);

  struct SlotPoolBuildState {
    uint32_t nextNumSlots{0};
    uint32_t inFlightNumSlots{0};
    bool createInFlight{false};
  };

  std::shared_ptr<exec::Task> task_{nullptr};

  core::PartitionedOutputNode::Kind kind_{
      core::PartitionedOutputNode::Kind::kPartitioned};

  std::atomic<bool> initialized_{false};

  // For broadcast: payloads kept around so late-arriving destinations
  // can be backfilled. Cleared once noMoreQueues_ is set.
  std::vector<std::shared_ptr<UcxCpuRowPayload>> dataToBroadcast_;

  uint64_t maxSize_{0};
  uint64_t continueSize_{0};

  uint32_t numDrivers_{0};

  bool noMoreQueues_{false};

  std::mutex mutex_;

  std::vector<std::unique_ptr<UcxCpuRowDestinationQueue>> queues_;
  std::vector<std::vector<std::shared_ptr<UcxCpuRowShmSlotPool>>> slotPools_;
  std::vector<SlotPoolBuildState> slotPoolBuild_;
  std::vector<exec::DestinationBuffer::Stats> finishedBufferStats_;

  uint32_t numFinished_{0};

  bool atEnd_ = false;

  std::vector<ContinuePromise> promises_;

  int64_t queuedBytes_{0};
  int64_t queuedPayloads_{0};
  int64_t pendingDirectHandoffBytes_{0};
  int64_t pendingDirectHandoffPayloads_{0};

  int64_t totalBytesSent_{0};
  int64_t totalRowsSent_{0};
  int64_t totalPayloadsSent_{0};

  int64_t slotPoolAcquireSuccesses_{0};
  int64_t slotPoolAcquireMisses_{0};
  int64_t slotPoolAcquireTooLarge_{0};
  int64_t slotPoolPoolsCreated_{0};
  int64_t slotPoolPoolLimitMisses_{0};
  int64_t slotPoolCreateFailures_{0};
  int64_t slotPoolAsyncCreatesStarted_{0};
  int64_t slotPoolAsyncCreatesCompleted_{0};
  int64_t slotPoolAsyncCreatesDropped_{0};
  int64_t slotPoolAcquireSuccessBytes_{0};
  int64_t slotPoolAcquireMissBytes_{0};
  int64_t slotPoolAcquireTooLargeBytes_{0};

  uint64_t queueStartMs_{0};
  double totalQueuedBytesMs_{0};
};

} // namespace facebook::velox::ucx_exchange
