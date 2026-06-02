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
    int numDrivers) {
  const auto& taskId = task->taskId();
  queues_.withLock([&](auto& queues) {
    auto it = queues.find(taskId);
    if (it == queues.end()) {
      queues[taskId] = std::make_shared<UcxCpuRowOutputQueue>(
          std::move(task), numDestinations, numDrivers, kind);
    } else {
      if (!it->second->initialize(task, numDestinations, numDrivers, kind)) {
        VELOX_FAIL(
            "Registering a UcxCpuRow output queue for pre-existing taskId {}",
            taskId);
      }
    }
  });
  // Clear stale "removed" state so post-removal getData() calls can
  // create proper placeholders.
  removedTasks_.withLock([&](auto& removed) { removed.erase(taskId); });
}

void UcxCpuRowOutputQueueManager::updateOutputBuffers(
    std::string_view taskId,
    int numBuffers,
    bool noMoreBuffers) {
  getQueue(taskId)->updateOutputBuffers(numBuffers, noMoreBuffers);
}

bool UcxCpuRowOutputQueueManager::updateNumDrivers(
    std::string_view taskId,
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

void UcxCpuRowOutputQueueManager::getData(
    std::string_view taskId,
    int destination,
    UcxCpuRowDataAvailableCallback notify) {
  std::shared_ptr<UcxCpuRowOutputQueue> outputQueue;
  bool taskRemoved = false;
  std::string taskIdStr{taskId};
  queues_.withLock([&](auto& queues) {
    auto it = queues.find(taskIdStr);
    if (it == queues.end()) {
      // If the task was already removed, refuse to recreate a
      // placeholder. The undersized placeholder would crash on
      // subsequent deleteResults(destination) for destinations beyond
      // its capacity.
      if (removedTasks_.withLock(
              [&](auto& removed) { return removed.count(taskIdStr) > 0; })) {
        taskRemoved = true;
        return;
      }
      // Server arrived before initializeTask. Create a placeholder
      // queue to hold the notify callback; it'll be promoted to a real
      // queue when the producer task initializes.
      outputQueue =
          std::make_shared<UcxCpuRowOutputQueue>(nullptr, destination, 0);
      queues[taskIdStr] = outputQueue;
    } else {
      outputQueue = it->second;
    }
  });
  if (taskRemoved) {
    notify(nullptr, {});
    return;
  }
  outputQueue->getData(destination, notify);
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

void UcxCpuRowOutputQueueManager::requeueFront(
    std::string_view taskId,
    int destination,
    std::shared_ptr<UcxCpuRowPayload> data) {
  if (!data) {
    return;
  }
  auto queue = getQueueIfExists(taskId);
  if (!queue) {
    return;
  }
  queue->requeueFront(destination, std::move(data));
}

void UcxCpuRowOutputQueueManager::removeTask(std::string_view taskId) {
  std::string taskIdStr{taskId};
  auto queue = queues_.withLock(
      [&](auto& queues) -> std::shared_ptr<UcxCpuRowOutputQueue> {
        auto it = queues.find(taskIdStr);
        if (it == queues.end()) {
          // Already removed. Clear stale "removed" state so the task
          // ID can be reused.
          removedTasks_.withLock(
              [&](auto& removed) { removed.erase(taskIdStr); });
          return nullptr;
        }
        auto taskQueue = it->second;
        queues.erase(it);
        // Insert into removedTasks_ inside the same lock to prevent a
        // concurrent getData() from racing into the gap and creating a
        // zombie placeholder.
        removedTasks_.withLock(
            [&](auto& removed) { removed.insert(taskIdStr); });
        return taskQueue;
      });
  if (queue != nullptr) {
    queue->terminate();
  }
}

std::shared_ptr<UcxCpuRowOutputQueue>
UcxCpuRowOutputQueueManager::getQueueIfExists(std::string_view taskId) {
  std::string taskIdStr{taskId};
  return queues_.withLock([&](auto& queues) {
    auto it = queues.find(taskIdStr);
    return it == queues.end() ? nullptr : it->second;
  });
}

std::shared_ptr<UcxCpuRowOutputQueue> UcxCpuRowOutputQueueManager::getQueue(
    std::string_view taskId) {
  std::string taskIdStr{taskId};
  return queues_.withLock([&](auto& queues) {
    auto it = queues.find(taskIdStr);
    VELOX_CHECK(
        it != queues.end(),
        "UcxCpuRow output queue for task not found: {}",
        taskId);
    return it->second;
  });
}

std::optional<exec::OutputBuffer::Stats> UcxCpuRowOutputQueueManager::stats(
    std::string_view taskId) {
  auto queue = getQueueIfExists(taskId);
  if (queue != nullptr) {
    return queue->stats();
  }
  return std::nullopt;
}

} // namespace facebook::velox::ucx_exchange
