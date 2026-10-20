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
// Assisted by watsonx Code Assistant
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <rmm/exec_policy.hpp>
#include <cuda/stream>
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"

namespace facebook::velox::ucx_exchange {

/* static */
const std::shared_ptr<UcxOutputQueueManager>&
UcxOutputQueueManager::getInstanceRef() {
  // In C++11, the static local variable is guaranteed to only be initialized
  // once even in a multi-threaded context.
  static const std::shared_ptr<UcxOutputQueueManager> instance =
      std::make_shared<UcxOutputQueueManager>();
  return instance;
}

void UcxOutputQueueManager::initializeTask(
    std::shared_ptr<exec::Task> task,
    core::PartitionedOutputNode::Kind kind,
    int numDestinations,
    int numDrivers,
    const std::string& /*transportOptions*/) {
  VELOX_CHECK_NOT_NULL(task);
  VELOX_CHECK_GE(numDestinations, 0);
  VELOX_CHECK_GE(numDrivers, 0);
  const auto& taskId = task->taskId();
  state_.withLock([&](auto& state) {
    // UCX tags do not carry a task generation. Even after requests are
    // canceled, an old unmatched message can remain in the worker's
    // unexpected-message queue. Presto task IDs are unique, so reject reuse.
    VELOX_CHECK(
        state.removedTasks.count(taskId) == 0,
        "Cannot reuse removed GPU exchange task ID {}",
        taskId);
    auto it = state.queues.find(taskId);
    if (it == state.queues.end()) {
      state.queues[taskId] = std::make_shared<UcxOutputQueue>(
          std::move(task), numDestinations, numDrivers, kind);
    } else {
      if (!it->second->initialize(task, numDestinations, numDrivers, kind)) {
        VELOX_FAIL(
            "Registering a cudf output queue for pre-existing taskId {}",
            taskId);
      }
    }
  });
}

bool UcxOutputQueueManager::updateOutputBuffers(
    const std::string& taskId,
    int numBuffers,
    bool noMoreBuffers) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->updateOutputBuffers(numBuffers, noMoreBuffers);
    return true;
  }
  return false;
}

void UcxOutputQueueManager::enqueue(
    std::string_view taskId,
    int destination,
    std::unique_ptr<cudf::packed_columns> txData,
    vector_size_t numRows,
    int64_t transferReservationBytes) {
  if (auto queue = getQueueIfActive(taskId)) {
    queue->enqueue(
        destination, std::move(txData), numRows, transferReservationBytes);
  }
}

bool UcxOutputQueueManager::checkBlocked(
    std::string_view taskId,
    ContinueFuture* future) {
  if (auto queue = getQueueIfActive(taskId)) {
    return queue->checkBlocked(future);
  }
  return false;
}

bool UcxOutputQueueManager::checkTransferCapacity(
    std::string_view taskId,
    int destination,
    int64_t maxBytes,
    ContinueFuture* future) {
  if (auto queue = getQueueIfActive(taskId)) {
    return queue->checkTransferCapacity(destination, maxBytes, future);
  }
  return false;
}

bool UcxOutputQueueManager::reserveTransferBytes(
    std::string_view taskId,
    int destination,
    int64_t bytes,
    int64_t maxBytes,
    ContinueFuture* future) {
  if (auto queue = getQueueIfActive(taskId)) {
    return queue->reserveTransferBytes(destination, bytes, maxBytes, future);
  }
  return false;
}

bool UcxOutputQueueManager::reserveFullTransferBytes(
    std::string_view taskId,
    int destination,
    int64_t bytes,
    ContinueFuture* future) {
  if (auto queue = getQueueIfActive(taskId)) {
    return queue->reserveFullTransferBytes(destination, bytes, future);
  }
  return false;
}

bool UcxOutputQueueManager::waitForFullTransferCapacity(
    std::string_view taskId,
    int64_t bytes,
    ContinueFuture* future) {
  if (auto queue = getQueueIfActive(taskId)) {
    return queue->waitForFullTransferCapacity(bytes, future);
  }
  return false;
}

void UcxOutputQueueManager::releaseTransferReservation(
    std::string_view taskId,
    int destination,
    int64_t bytes) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->releaseTransferReservation(destination, bytes);
  }
}

int64_t UcxOutputQueueManager::transferWindowBytes(
    std::string_view taskId,
    int destination,
    int64_t baseBytes,
    int64_t normalBytes,
    int64_t maxBytes) {
  if (auto queue = getQueueIfActive(taskId)) {
    return queue->transferWindowBytes(
        destination, baseBytes, normalBytes, maxBytes);
  }
  return baseBytes;
}

void UcxOutputQueueManager::recordTransferCongestion(
    std::string_view taskId,
    int destination,
    int64_t baseBytes) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->recordTransferCongestion(destination, baseBytes);
  }
}

void UcxOutputQueueManager::recordTransferDemand(
    std::string_view taskId,
    int destination,
    int64_t targetBytes,
    int64_t baseBytes,
    int64_t maxBytes) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->recordTransferDemand(destination, targetBytes, baseBytes, maxBytes);
  }
}

void UcxOutputQueueManager::recordFullTransferCongestion(
    std::string_view taskId) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->recordFullTransferCongestion();
  }
}

void UcxOutputQueueManager::releaseInFlightBytes(
    std::string_view taskId,
    int destination,
    int64_t bytes,
    int64_t numPackedColumns) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->releaseInFlightBytes(destination, bytes, numPackedColumns);
  }
}

void UcxOutputQueueManager::noMoreData(std::string_view taskId) {
  if (auto queue = getQueueIfActive(taskId)) {
    queue->noMoreData();
  }
}

bool UcxOutputQueueManager::isFinished(std::string_view taskId) {
  if (auto queue = getQueueIfActive(taskId)) {
    return queue->isFinished();
  }
  return true;
}

void UcxOutputQueueManager::deleteResults(
    std::string_view taskId,
    int destination) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->deleteResults(destination);
  }
}

std::shared_ptr<UcxOutputQueue> UcxOutputQueueManager::getData(
    std::string_view taskId,
    int destination,
    UcxDataAvailableCallback notify) {
  return getDataWithQueue(
      taskId,
      destination,
      [notify = std::move(notify)](
          std::shared_ptr<UcxOutputQueue> /*outputQueue*/,
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t numRows,
          std::vector<int64_t> remainingBytes) mutable {
        notify(std::move(data), numRows, std::move(remainingBytes));
      });
}

std::shared_ptr<UcxOutputQueue> UcxOutputQueueManager::getDataWithQueue(
    std::string_view taskId,
    int destination,
    UcxManagedDataAvailableCallback notify) {
  VELOX_CHECK_GE(destination, 0);
  std::shared_ptr<UcxOutputQueue> outputQueue;
  bool taskRemoved = false;
  std::string taskIdStr{taskId};
  state_.withLock([&](auto& state) {
    auto it = state.queues.find(taskIdStr);
    if (it == state.queues.end()) {
      // Check if the task was already removed. If so, don't re-create a
      // placeholder — the task is dead and any server calling getData() is a
      // stale leftover. Re-creating would produce an undersized queue that
      // crashes when deleteResults() is called for other destinations.
      if (state.removedTasks.count(taskIdStr) > 0) {
        taskRemoved = true;
        return;
      }
      // create the queue structures such that the notify callback can be
      // stored. It will be later initialized once the task is being created.
      VLOG(2)
          << "[QUEUE-MGR] task=" << taskId << " dest=" << destination
          << " creating placeholder queue (server arrived before task init)";
      outputQueue = std::make_shared<UcxOutputQueue>(nullptr, destination, 0);
      state.queues[taskIdStr] = outputQueue;
    } else {
      // queue exists.
      outputQueue = it->second;
    }
  });
  if (taskRemoved) {
    // Fire callback immediately with nullptr to signal end-of-stream.
    notify(nullptr, nullptr, /*numRows=*/0, {});
    return nullptr;
  }
  // outside of lock. Queue must exist.
  // get the data or install the notify callback. Capture a weak pointer to
  // avoid a queue -> callback -> queue ownership cycle while still making the
  // exact queue available before invoking the server callback.
  std::weak_ptr<UcxOutputQueue> weakQueue = outputQueue;
  outputQueue->getData(
      destination,
      [weakQueue, notify = std::move(notify)](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t numRows,
          std::vector<int64_t> remainingBytes) mutable {
        auto stableQueue = weakQueue.lock();
        VELOX_CHECK_NOT_NULL(
            stableQueue,
            "Output queue disappeared while dispatching dequeued data");
        notify(
            std::move(stableQueue),
            std::move(data),
            numRows,
            std::move(remainingBytes));
      });
  return outputQueue;
}

bool UcxOutputQueueManager::canUseIntraNode(std::string_view taskId) {
  auto queue = getQueueIfExists(taskId);
  return queue && queue->isInitialized() &&
      queue->kind() != core::PartitionedOutputNode::Kind::kBroadcast;
}

UcxOutputQueueManager::ExchangeServerAdmission
UcxOutputQueueManager::registerExchangeServer(
    const std::shared_ptr<UcxExchangeServerLifecycle>& server) {
  VELOX_CHECK_NOT_NULL(server);
  const auto key = server->getPartitionKey();
  const auto admission = state_.withLock([&](auto& state) {
    if (state.removedTasks.count(key.taskId) > 0) {
      return ExchangeServerAdmission::kTaskRemoved;
    }
    if (state.activeServers.count(key) > 0) {
      return ExchangeServerAdmission::kDuplicateServer;
    }

    state.activeServers.emplace(key, server);
    if (server->isClosed()) {
      // close() can win before registration. Keep the exact-key claim until
      // task removal so a replacement cannot restart the old stream.
      return ExchangeServerAdmission::kServerUnavailable;
    }
    return ExchangeServerAdmission::kAccepted;
  });

  if (admission == ExchangeServerAdmission::kAccepted && server->activate()) {
    return admission;
  }
  if (!server->isClosed()) {
    server->requestAbort();
  }
  return admission == ExchangeServerAdmission::kAccepted
      ? ExchangeServerAdmission::kServerUnavailable
      : admission;
}

void UcxOutputQueueManager::unregisterExchangeServer(
    const std::shared_ptr<UcxExchangeServerLifecycle>& server) {
  if (server == nullptr) {
    return;
  }
  const auto key = server->getPartitionKey();
  state_.withLock([&](auto& state) {
    auto active = state.activeServers.find(key);
    if (active == state.activeServers.end()) {
      return;
    }
    auto registered = active->second.lock();
    if (registered != nullptr && registered.get() == server.get()) {
      if (state.removedTasks.count(key.taskId) > 0) {
        state.activeServers.erase(active);
      } else {
        active->second.reset();
      }
    }
  });
}

void UcxOutputQueueManager::removeTask(const std::string& taskId) {
  std::vector<std::shared_ptr<UcxExchangeServerLifecycle>> serversToAbort;
  auto removeTaskServers = [&](auto& state, bool abortServers) {
    for (auto active = state.activeServers.begin();
         active != state.activeServers.end();) {
      if (active->first.taskId != taskId) {
        ++active;
        continue;
      }
      if (abortServers) {
        if (auto server = active->second.lock()) {
          serversToAbort.push_back(std::move(server));
        }
      }
      active = state.activeServers.erase(active);
    }
  };

  // If no queue exists, establish the abnormal-removal tombstone now. This
  // linearizes remove-before-initialize against initializeTask().
  auto queue = state_.withLock([&](auto& state) {
    auto queueIt = state.queues.find(taskId);
    if (queueIt != state.queues.end()) {
      return queueIt->second;
    }
    auto removedIt =
        state.removedTasks.emplace(taskId, TaskRemovalKind::kAborted).first;
    removeTaskServers(state, removedIt->second == TaskRemovalKind::kAborted);
    return std::shared_ptr<UcxOutputQueue>{};
  });

  if (queue == nullptr) {
    for (const auto& server : serversToAbort) {
      server->requestAbort();
    }
    IntraNodeTransferRegistry::getInstance()->cancelTask(taskId);
    return;
  }

  // Do not hold the manager mutex while reading Task state. Task termination
  // can call removeTask() while holding Task-internal lifecycle locks.
  const auto proposedRemovalKind =
      queue->taskState() == exec::TaskState::kFinished
      ? TaskRemovalKind::kFinished
      : TaskRemovalKind::kAborted;

  auto queueToTerminate = state_.withLock([&](auto& state) {
    auto queueIt = state.queues.find(taskId);
    if (queueIt == state.queues.end() || queueIt->second.get() != queue.get()) {
      return std::shared_ptr<UcxOutputQueue>{};
    }
    auto removedQueue = std::move(queueIt->second);
    state.queues.erase(queueIt);

    // Preserve the first removal classification. A repeated removal must not
    // clear the tombstone or change whether live servers are aborted.
    auto removedIt =
        state.removedTasks.emplace(taskId, proposedRemovalKind).first;
    removeTaskServers(state, removedIt->second == TaskRemovalKind::kAborted);
    return removedQueue;
  });

  // Mark every server before terminate() wakes pending getData callbacks.
  for (const auto& server : serversToAbort) {
    server->requestAbort();
  }
  if (queueToTerminate != nullptr) {
    queueToTerminate->terminate();
  }
  IntraNodeTransferRegistry::getInstance()->cancelTask(taskId);
}

std::shared_ptr<UcxOutputQueue> UcxOutputQueueManager::getQueueIfExists(
    std::string_view taskId) {
  std::string taskIdStr{taskId};
  return state_.withLock([&](auto& state) {
    auto it = state.queues.find(taskIdStr);
    return it == state.queues.end() ? nullptr : it->second;
  });
}

std::shared_ptr<UcxOutputQueue> UcxOutputQueueManager::getQueueIfActive(
    std::string_view taskId) {
  std::string taskIdStr{taskId};
  return state_.withLock([&](auto& state) {
    auto it = state.queues.find(taskIdStr);
    if (it != state.queues.end()) {
      return it->second;
    }
    VELOX_CHECK(
        state.removedTasks.count(taskIdStr) > 0,
        "Output cudf queue for task not found: {}",
        taskId);
    return std::shared_ptr<UcxOutputQueue>{};
  });
}

std::shared_ptr<UcxOutputQueue> UcxOutputQueueManager::getQueue(
    std::string_view taskId) {
  std::string taskIdStr{taskId};
  return state_.withLock([&](auto& state) {
    auto it = state.queues.find(taskIdStr);
    VELOX_CHECK(
        it != state.queues.end(),
        "Output cudf queue for task not found: {}",
        taskId);
    return it->second;
  });
}

std::optional<exec::OutputBuffer::Stats> UcxOutputQueueManager::stats(
    const std::string& taskId) {
  auto queue = getQueueIfExists(taskId);
  if (queue != nullptr) {
    return queue->stats();
  }
  return std::nullopt;
}

bool UcxOutputQueueManager::updateNumDrivers(
    const std::string& taskId,
    uint32_t newNumDrivers) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->updateNumDrivers(newNumDrivers);
    return true;
  }
  return false;
}

std::optional<double> UcxOutputQueueManager::getUtilization(
    const std::string& taskId) {
  if (auto queue = getQueueIfExists(taskId); queue && queue->isInitialized()) {
    return queue->getUtilization();
  }
  return std::nullopt;
}

std::optional<bool> UcxOutputQueueManager::isOverutilized(
    const std::string& taskId) {
  if (auto queue = getQueueIfExists(taskId); queue && queue->isInitialized()) {
    return queue->isOverutilized();
  }
  return std::nullopt;
}

std::string UcxOutputQueueManager::toString(const std::string& taskId) {
  auto queue = getQueueIfExists(taskId);
  if (queue != nullptr) {
    return queue->toString();
  }
  return "UcxOutputQueue[" + taskId + " not found]";
}

} // namespace facebook::velox::ucx_exchange
