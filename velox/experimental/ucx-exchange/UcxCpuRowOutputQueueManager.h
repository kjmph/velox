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

#include <functional>
#include <map>
#include <optional>
#include <string_view>
#include <unordered_map>
#include "velox/exec/OutputBufferManager.h"
#include "velox/exec/Task.h"
#include "velox/experimental/ucx-exchange/PartitionKey.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowQueues.h"

/// Singleton mapping (taskId -> UcxCpuRowOutputQueue). Mirrors
/// UcxOutputQueueManager for CPU row-vector UCX exchange.

namespace facebook::velox::ucx_exchange {

/// Minimal lifecycle interface used by the output queue manager to stop
/// producer-side UCX sends when their task terminates abnormally. Keeping this
/// separate from the concrete server makes the registry independently
/// testable without constructing UCX endpoints.
class UcxCpuRowExchangeServerLifecycle {
 public:
  virtual ~UcxCpuRowExchangeServerLifecycle() = default;

  virtual const PartitionKey& getPartitionKey() const = 0;

  /// Requests asynchronous, idempotent shutdown on the communicator thread.
  virtual void requestAbort() = 0;

  /// Atomically enables communicator processing after registry admission.
  /// Returns false if shutdown won the race.
  virtual bool activate() = 0;

  /// Lock-free lifecycle check used to close registration-vs-shutdown races.
  virtual bool isClosed() const = 0;
};

class UcxCpuRowOutputQueueManager : public exec::OutputBufferManager {
 public:
  enum class ExchangeServerAdmission {
    kAccepted,
    kTaskRemoved,
    kDuplicateServer,
    kServerUnavailable,
  };

  static std::shared_ptr<UcxCpuRowOutputQueueManager> getInstanceRef();

  UcxCpuRowOutputQueueManager() = default;
  UcxCpuRowOutputQueueManager(const UcxCpuRowOutputQueueManager&) = delete;
  UcxCpuRowOutputQueueManager& operator=(const UcxCpuRowOutputQueueManager&) =
      delete;

  /// Initialize the queue for a task. If a placeholder queue was already
  /// created by an early getData() arrival, finalize it via
  /// UcxCpuRowOutputQueue::initialize().
  void initializeTask(
      std::shared_ptr<exec::Task> task,
      core::PartitionedOutputNode::Kind kind,
      int numDestinations,
      int numDrivers,
      const std::string& transportOptions = {}) override;

  /// For broadcast mode, propagate destination-buffer count changes to
  /// the underlying queue. Mirrors OutputBufferManager's same-name call.
  bool updateOutputBuffers(
      const std::string& taskId,
      int numBuffers,
      bool noMoreBuffers) override;

  /// For grouped execution, update the producing driver count after Velox
  /// discovers the actual split groups assigned to this task.
  bool updateNumDrivers(const std::string& taskId, uint32_t newNumDrivers)
      override;

  /// Enqueue a serialized RowVector chunk into the destination's queue.
  /// Caller transfers ownership of the payload.
  void enqueue(
      std::string_view taskId,
      int destination,
      std::unique_ptr<UcxCpuRowPayload> txData,
      int32_t numRows);

  /// Returns true (and populates `future`) if the queue is over the
  /// high-water mark and producers should block.
  bool checkBlocked(std::string_view taskId, ContinueFuture* future);

  /// Indicates that no more data will be coming for this task.
  void noMoreData(std::string_view taskId);

  /// True iff noMoreData has been called and all data has been
  /// fetched + acknowledged.
  bool isFinished(std::string_view taskId);

  void deleteResults(std::string_view taskId, int destination);

  /// Async pop from the head of `destination`'s queue. The notify
  /// callback fires synchronously if data is available; otherwise it
  /// fires when data arrives. A nullptr `data` argument signals end of
  /// stream. If the destination doesn't yet exist, additional queues
  /// are created (placeholder mechanism for late getData arrivals).
  std::shared_ptr<UcxCpuRowOutputQueue> getData(
      std::string_view taskId,
      int destination,
      UcxCpuRowDataAvailableCallback notify);

  /// Non-blocking variant of getData. Returns the next payload if one
  /// is immediately queued, nullptr otherwise. Server-side bundling
  /// uses this to drain additional chunks after a first chunk arrives,
  /// without registering a fresh notify callback.
  std::shared_ptr<UcxCpuRowPayload> tryGetData(
      std::string_view taskId,
      int destination);

  /// Returns the stable queue instance for an initialized producer task.
  /// Task IDs cannot be reused after removal and early placeholder queues are
  /// initialized in place, so output operators can safely cache this identity.
  std::shared_ptr<UcxCpuRowOutputQueue> getTaskQueue(std::string_view taskId);

  /// Registers a producer-side exchange server for task-lifecycle cleanup.
  /// Registration and task removal are linearized by the same mutex. A server
  /// arriving after task removal, or while another server owns the same exact
  /// partition key, is rejected and asynchronously aborted. Accepted servers
  /// are activated after releasing the lifecycle lock.
  ExchangeServerAdmission registerExchangeServer(
      const std::shared_ptr<UcxCpuRowExchangeServerLifecycle>& server);

  /// Removes a server from lifecycle tracking. Safe to call repeatedly.
  void unregisterExchangeServer(
      const std::shared_ptr<UcxCpuRowExchangeServerLifecycle>& server);

  /// Removes the queue for the given task. Calls `terminate` on the
  /// queue to wake up any waiting producers/consumers. Servers for a normally
  /// finished task remain alive long enough to complete their final metadata
  /// send. Servers for canceled, aborted, failed, or uninitialized tasks are
  /// asynchronously aborted.
  void removeTask(const std::string& taskId) override;

  /// Returns the queue stats, or nullopt if no queue exists.
  std::optional<exec::OutputBufferStats> stats(
      const std::string& taskId) override;

  std::optional<double> getUtilization(const std::string& taskId) override;

  std::optional<bool> isOverutilized(const std::string& taskId) override;

  std::string toString(const std::string& taskId) override;

 private:
  std::shared_ptr<UcxCpuRowOutputQueue> getQueueIfExists(
      std::string_view taskId);

  // Throws if no queue exists for the taskId.
  std::shared_ptr<UcxCpuRowOutputQueue> getQueue(std::string_view taskId);

  enum class TaskRemovalKind {
    kFinished,
    kAborted,
  };

  using ServerWeakPtr = std::weak_ptr<UcxCpuRowExchangeServerLifecycle>;

  struct State {
    std::unordered_map<std::string, std::shared_ptr<UcxCpuRowOutputQueue>>
        queues;
    // Tombstones prevent late handshakes from recreating placeholder queues.
    // They persist because UCX tags have no generation and task IDs are unique.
    std::unordered_map<std::string, TaskRemovalKind> removedTasks;
    // An entry claims the exact key for the task lifetime. Its weak pointer is
    // cleared when a server closes before task removal, but the claim remains:
    // a replacement would restart sequence zero on a partially consumed
    // stream. Task removal replaces these claims with a task tombstone.
    std::map<PartitionKey, ServerWeakPtr> activeServers;
  };

  folly::Synchronized<State, std::mutex> state_;
};

} // namespace facebook::velox::ucx_exchange
