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

#include <velox/exec/Task.h>
#include <functional>
#include <string_view>
#include <unordered_set>
#include "velox/experimental/ucx-exchange/UcxCpuRowQueues.h"

/// Singleton mapping (taskId -> UcxCpuRowOutputQueue). Mirrors
/// UcxOutputQueueManager for CPU row-vector UCX exchange.

namespace facebook::velox::ucx_exchange {

class UcxCpuRowOutputQueueManager {
 public:
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
      int numDrivers);

  /// For broadcast mode, propagate destination-buffer count changes to
  /// the underlying queue. Mirrors OutputBufferManager's same-name call.
  void updateOutputBuffers(
      std::string_view taskId,
      int numBuffers,
      bool noMoreBuffers);

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
  void getData(
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

  /// Reinsert a payload at the head of a destination queue. This is the
  /// inverse of tryGetData() for server-side bundle assembly.
  void requeueFront(
      std::string_view taskId,
      int destination,
      std::shared_ptr<UcxCpuRowPayload> data);

  std::optional<UcxCpuRowShmSlotLease>
  tryAcquireSlot(std::string_view taskId, int destination, size_t bytes);

  /// Removes the queue for the given task. Calls `terminate` on the
  /// queue to wake up any waiting producers/consumers.
  void removeTask(std::string_view taskId);

  /// Returns the queue stats, or nullopt if no queue exists.
  std::optional<exec::OutputBuffer::Stats> stats(std::string_view taskId);

 private:
  std::shared_ptr<UcxCpuRowOutputQueue> getQueueIfExists(
      std::string_view taskId);

  // Throws if no queue exists for the taskId.
  std::shared_ptr<UcxCpuRowOutputQueue> getQueue(std::string_view taskId);

  folly::Synchronized<
      std::unordered_map<std::string, std::shared_ptr<UcxCpuRowOutputQueue>>,
      std::mutex>
      queues_;

  // Tracks tasks that have been removed via removeTask(). Prevents
  // getData() from re-creating placeholder queues for dead tasks,
  // which would cause crashes when deleteResults() runs on undersized
  // placeholders. Mirrors UcxOutputQueueManager's removedTasks_.
  folly::Synchronized<std::unordered_set<std::string>, std::mutex>
      removedTasks_;
};

} // namespace facebook::velox::ucx_exchange
