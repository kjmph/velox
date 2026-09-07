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
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "velox/exec/OutputBufferManager.h"
#include "velox/exec/Task.h"
#include "velox/experimental/ucx-exchange/PartitionKey.h"
#include "velox/experimental/ucx-exchange/UcxQueues.h"

namespace facebook::velox::ucx_exchange {

/// Minimal lifecycle surface used to stop producer-side GPU exchange sends
/// when their task terminates abnormally. Keeping this independent of UCX
/// endpoints makes admission and task-removal races independently testable.
class UcxExchangeServerLifecycle {
 public:
  virtual ~UcxExchangeServerLifecycle() = default;

  virtual const PartitionKey& getPartitionKey() const = 0;

  /// Requests asynchronous, idempotent shutdown on the communicator thread.
  virtual void requestAbort() = 0;

  /// Enables communicator processing after registry admission. Returns false
  /// if shutdown won the race.
  virtual bool activate() = 0;

  /// Lock-free lifecycle check used to close registration/shutdown races.
  virtual bool isClosed() const = 0;
};

class UcxOutputQueueManager : public exec::OutputBufferManager {
 public:
  enum class ExchangeServerAdmission {
    kAccepted,
    kTaskRemoved,
    kDuplicateServer,
    kServerUnavailable,
  };

  /// Factory method to retrieve a reference to the output queue manager.
  static std::shared_ptr<UcxOutputQueueManager> getInstanceRef();

  // no constructor to prevent direct instantiation.
  UcxOutputQueueManager() = default;
  // no copy constructor.
  UcxOutputQueueManager(const UcxOutputQueueManager&) = delete;
  // no copy assignment.
  UcxOutputQueueManager& operator=(const UcxOutputQueueManager&) = delete;

  /// @brief Initializes a task and creates the corresponding output queues that
  /// are associated with this task.
  /// @param task The task.
  /// @param kind The output mode (partitioned, broadcast, etc.)
  /// @param numDestinations The number of queues (destinations or partitions)
  /// associated with this task.
  /// @param numDrivers The number of drivers that contribute data to these
  /// queues. Used to recognize when the queues are complete.
  void initializeTask(
      std::shared_ptr<exec::Task> task,
      core::PartitionedOutputNode::Kind kind,
      int numDestinations,
      int numDrivers) override;

  /// @brief Updates the number of destination buffers for a task.
  /// For broadcast mode, new destinations are backfilled with previously
  /// broadcast data.
  bool updateOutputBuffers(
      const std::string& taskId,
      int numBuffers,
      bool noMoreBuffers) override;

  bool updateNumDrivers(const std::string& taskId, uint32_t newNumDrivers)
      override;

  /// @brief Enqueues a cudf packed column into the queue.
  /// @param taskId The unique task Id.
  /// @param destination The destination (partition, queue number) into which
  /// the data is queued.
  /// @param txData The data to enqueue.
  /// @param numRows The number of rows in the data.
  void enqueue(
      std::string_view taskId,
      int destination,
      std::unique_ptr<cudf::packed_columns> txData,
      int32_t numRows,
      int64_t transferReservationBytes = 0);

  /// @brief Checks if the queue for a task is over capacity.
  /// Producers call this before accepting more input and after enqueueing a
  /// batch.
  /// @param taskId The unique task Id.
  /// @param future Output parameter - populated with a future if blocked.
  /// @return True if blocked (queue over capacity), false otherwise.
  bool checkBlocked(std::string_view taskId, ContinueFuture* future);

  /// @brief Checks active producer queued/in-flight transfer bytes.
  bool checkTransferCapacity(
      std::string_view taskId,
      int destination,
      int64_t maxBytes,
      ContinueFuture* future);

  /// @brief Reserves destination-local transfer capacity before GPU output
  /// materialization.
  bool reserveTransferBytes(
      std::string_view taskId,
      int destination,
      int64_t bytes,
      int64_t maxBytes,
      ContinueFuture* future);

  /// @brief Reserves full contiguous_split payload capacity before GPU output
  /// materialization.
  bool reserveFullTransferBytes(
      std::string_view taskId,
      int destination,
      int64_t bytes,
      ContinueFuture* future);

  /// @brief Blocks until the learned full-transfer retained-byte window has
  /// room. Does not reserve bytes.
  bool waitForFullTransferCapacity(
      std::string_view taskId,
      int64_t bytes,
      ContinueFuture* future);

  /// @brief Releases destination-local transfer capacity.
  void releaseTransferReservation(
      std::string_view taskId,
      int destination,
      int64_t bytes);

  /// @brief Returns the destination-local adaptive transfer admission window.
  int64_t transferWindowBytes(
      std::string_view taskId,
      int destination,
      int64_t baseBytes,
      int64_t normalBytes,
      int64_t maxBytes);

  /// @brief Records allocation/admission congestion for a destination.
  void recordTransferCongestion(
      std::string_view taskId,
      int destination,
      int64_t baseBytes);

  /// @brief Records larger payload demand for adaptive transfer probing.
  void recordTransferDemand(
      std::string_view taskId,
      int destination,
      int64_t targetBytes,
      int64_t baseBytes,
      int64_t maxBytes);

  /// @brief Records full contiguous_split allocation/admission pressure.
  void recordFullTransferCongestion(std::string_view taskId);

  /// @brief Returns queued/in-flight pressure for a single destination.
  UcxDestinationTransferStats transferStats(
      std::string_view taskId,
      int destination);

  /// @brief Reserves bytes for producer-side GPU output materialization.
  bool reserveOutputBytes(
      std::string_view taskId,
      int64_t bytes,
      ContinueFuture* future);

  /// @brief Releases bytes reserved for output materialization.
  void releaseOutputReservation(std::string_view taskId, int64_t bytes);

  /// @brief Releases bytes retained by an in-flight exchange transfer.
  void releaseInFlightBytes(
      std::string_view taskId,
      int destination,
      int64_t bytes,
      int64_t numPackedCols);

  /// @brief Indicates that no more data will be coming for this task.
  void noMoreData(std::string_view taskId);

  /// @returns true if noMoreData has been called and all accumulated data has
  /// been fetched.
  bool isFinished(std::string_view taskId);

  /// @brief
  void deleteResults(std::string_view taskId, int destination);

  /// @brief Asynchronously returns the head of the queue. If data is available,
  /// the callback function is triggered immediately and true is returned.
  /// Otherwise, the callback function is registered and called once data is
  /// available. If the queue is closed and no more data is available, then the
  /// callback function is called immediately with a nullptr. If the destination
  /// doesn't exist, then additional queues will be added for this task.
  /// @param taskId The unique taskId.
  /// @param destination The destination.
  /// @param notify The callback function.
  std::shared_ptr<UcxOutputQueue> getData(
      std::string_view taskId,
      int destination,
      UcxDataAvailableCallback notify);

  /// Registers a producer-side server for task lifecycle cleanup. Exact
  /// partition keys remain claimed for the task lifetime so sequence zero
  /// cannot be restarted against a partially consumed stream.
  ExchangeServerAdmission registerExchangeServer(
      const std::shared_ptr<UcxExchangeServerLifecycle>& server);

  /// Removes a server from lifecycle tracking. Safe to call repeatedly.
  void unregisterExchangeServer(
      const std::shared_ptr<UcxExchangeServerLifecycle>& server);

  /// Returns true if the given task can use intra-node transfer.
  /// Returns false if the task is not yet initialized (placeholder queue
  /// from early sink connections) or if the task uses broadcast mode
  /// (broadcast shares packed_columns across destinations - the intra-node
  /// source's destructive move would corrupt data for other servers).
  bool canUseIntraNode(std::string_view taskId);

  /// @brief Removes the queue for the given task from the queue manager.
  /// Calls "terminate" on the queue to awake waiting producers.
  void removeTask(const std::string& taskId) override;

  /// @brief Returns the queue statistics of the queue associated with the given
  /// task. Returns nullopt when the specified output queue doesn't exist.
  std::optional<exec::OutputBufferStats> stats(
      const std::string& taskId) override;

  std::optional<double> getUtilization(const std::string& taskId) override;

  std::optional<bool> isOverutilized(const std::string& taskId) override;

  std::string toString(const std::string& taskId) override;

  /// Retrieves the queue for a task if it exists, or NULL. Public because
  /// useDynamicUcx() registers the task elsewhere.
  std::shared_ptr<UcxOutputQueue> getQueueIfExists(std::string_view taskId);

 private:
  // Retrieves the queue for a task if it exists. Returns NULL only when the
  // task is known to have been removed; unknown task IDs still fail fast.
  std::shared_ptr<UcxOutputQueue> getQueueIfActive(std::string_view taskId);

  // Throws an exception if queue doesn't exist.
  std::shared_ptr<UcxOutputQueue> getQueue(std::string_view taskId);

  enum class TaskRemovalKind {
    kFinished,
    kAborted,
  };

  using ServerWeakPtr = std::weak_ptr<UcxExchangeServerLifecycle>;

  struct State {
    std::unordered_map<std::string, std::shared_ptr<UcxOutputQueue>> queues;
    // UCX tags carry no generation. Task tombstones therefore persist and
    // prevent late handshakes from recreating a zombie placeholder queue.
    std::unordered_map<std::string, TaskRemovalKind> removedTasks;
    // A closed server leaves an empty exact-key claim until task removal. A
    // replacement cannot safely restart sequence zero on the old stream.
    std::map<PartitionKey, ServerWeakPtr> activeServers;
  };

  folly::Synchronized<State, std::mutex> state_;
};

} // namespace facebook::velox::ucx_exchange
