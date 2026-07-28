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
#include "velox/experimental/ucx-exchange/UcxCpuRowOutputQueueManager.h"

namespace facebook::velox::ucx_exchange {

/* static */
std::shared_ptr<UcxCpuRowOutputQueueManager>
UcxCpuRowOutputQueueManager::getInstanceRef() {
  // C++11 guarantees thread-safe one-time initialization of static
  // locals.
  static std::shared_ptr<UcxCpuRowOutputQueueManager> instance =
      std::make_shared<UcxCpuRowOutputQueueManager>();
  return instance;
}

void UcxCpuRowOutputQueueManager::initializeTask(
    std::shared_ptr<exec::Task> task,
    core::PartitionedOutputNode::Kind kind,
    int numDestinations,
    int numDrivers,
    const std::string& /*transportOptions*/) {
  const auto& taskId = task->taskId();
  state_.withLock([&](auto& state) {
    // UCX tags do not carry a task generation. Even after requests are
    // canceled, an unmatched message from the old task can remain in the
    // worker's unexpected-message queue. Presto task IDs are unique, so keep
    // tombstones and reject reuse instead of risking cross-generation matches.
    VELOX_CHECK(
        state.removedTasks.count(taskId) == 0,
        "Cannot reuse removed CPU row exchange task ID {}",
        taskId);
    auto it = state.queues.find(taskId);
    if (it == state.queues.end()) {
      state.queues[taskId] = std::make_shared<UcxCpuRowOutputQueue>(
          std::move(task), numDestinations, numDrivers, kind);
    } else {
      if (!it->second->initialize(task, numDestinations, numDrivers, kind)) {
        VELOX_FAIL(
            "Registering a UcxCpuRow output queue for pre-existing taskId {}",
            taskId);
      }
    }
  });
}

bool UcxCpuRowOutputQueueManager::updateOutputBuffers(
    const std::string& taskId,
    int numBuffers,
    bool noMoreBuffers) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->updateOutputBuffers(numBuffers, noMoreBuffers);
    return true;
  }
  return false;
}

bool UcxCpuRowOutputQueueManager::updateNumDrivers(
    const std::string& taskId,
    uint32_t newNumDrivers) {
  auto queue = getQueueIfExists(taskId);
  if (queue == nullptr) {
    return false;
  }
  queue->updateNumDrivers(newNumDrivers);
  return true;
}

void UcxCpuRowOutputQueueManager::enqueue(
    std::string_view taskId,
    int destination,
    std::unique_ptr<UcxCpuRowPayload> txData,
    int32_t numRows) {
  getQueue(taskId)->enqueue(destination, std::move(txData), numRows);
}

bool UcxCpuRowOutputQueueManager::checkBlocked(
    std::string_view taskId,
    ContinueFuture* future) {
  return getQueue(taskId)->checkBlocked(future);
}

void UcxCpuRowOutputQueueManager::noMoreData(std::string_view taskId) {
  getQueue(taskId)->noMoreData();
}

bool UcxCpuRowOutputQueueManager::isFinished(std::string_view taskId) {
  return getQueue(taskId)->isFinished();
}

void UcxCpuRowOutputQueueManager::deleteResults(
    std::string_view taskId,
    int destination) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->deleteResults(destination);
  }
}

std::shared_ptr<UcxCpuRowOutputQueue> UcxCpuRowOutputQueueManager::getData(
    std::string_view taskId,
    int destination,
    UcxCpuRowDataAvailableCallback notify) {
  std::shared_ptr<UcxCpuRowOutputQueue> outputQueue;
  bool taskRemoved = false;
  std::string taskIdStr{taskId};
  state_.withLock([&](auto& state) {
    auto it = state.queues.find(taskIdStr);
    if (it == state.queues.end()) {
      // If the task was already removed, refuse to recreate a
      // placeholder. The undersized placeholder would crash on
      // subsequent deleteResults(destination) for destinations beyond
      // its capacity.
      if (state.removedTasks.count(taskIdStr) > 0) {
        taskRemoved = true;
        return;
      }
      // Server arrived before initializeTask. Create a placeholder
      // queue to hold the notify callback; it'll be promoted to a real
      // queue when the producer task initializes.
      outputQueue =
          std::make_shared<UcxCpuRowOutputQueue>(nullptr, destination, 0);
      state.queues[taskIdStr] = outputQueue;
    } else {
      outputQueue = it->second;
    }
  });
  if (taskRemoved) {
    notify(nullptr, {});
    return nullptr;
  }
  outputQueue->getData(destination, notify);
  return outputQueue;
}

std::shared_ptr<UcxCpuRowPayload> UcxCpuRowOutputQueueManager::tryGetData(
    std::string_view taskId,
    int destination) {
  auto queue = getQueueIfExists(taskId);
  if (!queue) {
    return nullptr;
  }
  return queue->tryGetData(destination);
}

std::shared_ptr<UcxCpuRowOutputQueue> UcxCpuRowOutputQueueManager::getTaskQueue(
    std::string_view taskId) {
  auto queue = getQueue(taskId);
  VELOX_CHECK(
      queue->isInitialized(),
      "CPU row output queue for task {} is not initialized",
      taskId);
  return queue;
}

UcxCpuRowOutputQueueManager::ExchangeServerAdmission
UcxCpuRowOutputQueueManager::registerExchangeServer(
    const std::shared_ptr<UcxCpuRowExchangeServerLifecycle>& server) {
  VELOX_CHECK_NOT_NULL(server);
  const auto key = server->getPartitionKey();
  const auto admission = state_.withLock([&](auto& state) {
    auto removed = state.removedTasks.find(key.taskId);
    if (removed != state.removedTasks.end()) {
      return ExchangeServerAdmission::kTaskRemoved;
    }

    if (state.activeServers.count(key) > 0) {
      // The exact key remains claimed for the task lifetime even after its
      // server closes. Restarting sequence zero against a partially consumed
      // queue would corrupt the stream, and an expired weak pointer cannot
      // prove that no unexpected UCX message remains.
      return ExchangeServerAdmission::kDuplicateServer;
    }
    state.activeServers.emplace(key, server);
    if (server->isClosed()) {
      // close() may have completed before registration and therefore could not
      // preserve the exact-key claim through unregisterExchangeServer().
      // Keep this claim until task removal prevents any tag reuse.
      return ExchangeServerAdmission::kServerUnavailable;
    }
    return ExchangeServerAdmission::kAccepted;
  });

  // Crossing into the communicator while holding the manager mutex would
  // create an unnecessary lock-order dependency. Once admitted, removal can
  // safely race activation: abnormal removal requests an idempotent abort,
  // normal removal preserves the pre-admitted server, and close wins through
  // the atomics in activate().
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

void UcxCpuRowOutputQueueManager::unregisterExchangeServer(
    const std::shared_ptr<UcxCpuRowExchangeServerLifecycle>& server) {
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
        // Preserve an empty claim until task removal. A replacement server
        // cannot resume the old stream safely after sequence or queue state
        // has advanced.
        active->second.reset();
      }
    }
  });
}

void UcxCpuRowOutputQueueManager::removeTask(const std::string& taskId) {
  std::vector<std::shared_ptr<UcxCpuRowExchangeServerLifecycle>> serversToAbort;
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
      // The task tombstone now rejects every late handshake, so its exact-key
      // claims are redundant. Live abnormal servers remain owned by the
      // communicator until their asynchronous abort reaches close().
      active = state.activeServers.erase(active);
    }
  };

  // If no queue exists, establish the abnormal-removal tombstone in this
  // first critical section. This makes remove-before-initialize linearizable:
  // initializeTask() either creates the queue before removal observes it, or
  // sees the tombstone and rejects initialization of the removed task.
  auto queue = state_.withLock([&](auto& state) {
    auto queueIt = state.queues.find(taskId);
    if (queueIt != state.queues.end()) {
      return queueIt->second;
    }
    auto removedIt =
        state.removedTasks.emplace(taskId, TaskRemovalKind::kAborted).first;
    removeTaskServers(state, removedIt->second == TaskRemovalKind::kAborted);
    return std::shared_ptr<UcxCpuRowOutputQueue>{};
  });

  if (queue == nullptr) {
    for (const auto& server : serversToAbort) {
      server->requestAbort();
    }
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
      // Another removal already won. Never let this stale callback alter the
      // tombstone or server set established by the winner.
      return std::shared_ptr<UcxCpuRowOutputQueue>{};
    }
    auto removedQueue = std::move(queueIt->second);
    state.queues.erase(queueIt);

    // Preserve the first removal classification. In particular, a second
    // removeTask() must not erase the tombstone and allow a late handshake to
    // recreate a zombie placeholder queue.
    auto removedIt =
        state.removedTasks.emplace(taskId, proposedRemovalKind).first;
    removeTaskServers(state, removedIt->second == TaskRemovalKind::kAborted);
    return removedQueue;
  });

  // Mark every server before terminating its queue. terminate() wakes pending
  // getData callbacks; the server checks the abort flag before publishing any
  // newly available payload or final marker.
  for (const auto& server : serversToAbort) {
    server->requestAbort();
  }
  if (queueToTerminate != nullptr) {
    queueToTerminate->terminate();
  }
}

std::shared_ptr<UcxCpuRowOutputQueue>
UcxCpuRowOutputQueueManager::getQueueIfExists(std::string_view taskId) {
  std::string taskIdStr{taskId};
  return state_.withLock([&](auto& state) {
    auto it = state.queues.find(taskIdStr);
    return it == state.queues.end() ? nullptr : it->second;
  });
}

std::shared_ptr<UcxCpuRowOutputQueue> UcxCpuRowOutputQueueManager::getQueue(
    std::string_view taskId) {
  std::string taskIdStr{taskId};
  return state_.withLock([&](auto& state) {
    auto it = state.queues.find(taskIdStr);
    VELOX_CHECK(
        it != state.queues.end(),
        "UcxCpuRow output queue for task not found: {}",
        taskId);
    return it->second;
  });
}

std::optional<exec::OutputBufferStats> UcxCpuRowOutputQueueManager::stats(
    const std::string& taskId) {
  auto queue = getQueueIfExists(taskId);
  if (queue != nullptr) {
    return queue->stats();
  }
  return std::nullopt;
}

std::optional<double> UcxCpuRowOutputQueueManager::getUtilization(
    const std::string& taskId) {
  if (auto queue = getQueueIfExists(taskId)) {
    return queue->getUtilization();
  }
  return std::nullopt;
}

std::optional<bool> UcxCpuRowOutputQueueManager::isOverutilized(
    const std::string& taskId) {
  if (auto queue = getQueueIfExists(taskId)) {
    return queue->isOverutilized();
  }
  return std::nullopt;
}

std::string UcxCpuRowOutputQueueManager::toString(const std::string& taskId) {
  if (auto queue = getQueueIfExists(taskId)) {
    return queue->toString();
  }
  return "UcxCpuRowOutputQueue[" + taskId + " not found]";
}

} // namespace facebook::velox::ucx_exchange
