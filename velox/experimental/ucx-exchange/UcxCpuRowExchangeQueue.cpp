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
#include "velox/experimental/ucx-exchange/UcxCpuRowExchangeQueue.h"

namespace facebook::velox::ucx_exchange {

void UcxCpuRowExchangeQueue::noMoreSources() {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    noMoreSources_ = true;
    promises = checkCompleteLocked();
  }
  clearPromises(promises);
}

void UcxCpuRowExchangeQueue::close() {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    promises = closeLocked();
  }
  clearPromises(promises);
}

void UcxCpuRowExchangeQueue::enqueueLocked(
    UcxCpuRowReceivedPtr&& data,
    std::vector<ContinuePromise>& promises) {
  if (data == nullptr) {
    ++numCompleted_;
    auto completedPromises = checkCompleteLocked();
    promises.reserve(promises.size() + completedPromises.size());
    for (auto& promise : completedPromises) {
      promises.push_back(std::move(promise));
    }
    return;
  }

  queue_.push_back(std::move(data));

  while (!promises_.empty()) {
    VELOX_CHECK_LE(promises_.size(), numberOfConsumers_);
    const int32_t unblockedConsumers = numberOfConsumers_ - promises_.size();
    const int64_t unassignedPayloads = queue_.size() - unblockedConsumers;
    if (unassignedPayloads <= 0) {
      break;
    }
    auto it = promises_.begin();
    promises.push_back(std::move(it->second));
    promises_.erase(it);
  }
}

void UcxCpuRowExchangeQueue::addPromiseLocked(
    int consumerId,
    ContinueFuture* future,
    ContinuePromise* stalePromise) {
  ContinuePromise promise{"UcxCpuRowExchangeQueue::dequeue"};
  *future = promise.getSemiFuture();
  auto it = promises_.find(consumerId);
  if (it != promises_.end()) {
    *stalePromise = std::move(it->second);
    it->second = std::move(promise);
  } else {
    promises_[consumerId] = std::move(promise);
  }
  VELOX_CHECK_LE(promises_.size(), numberOfConsumers_);
}

bool UcxCpuRowExchangeQueue::registerBackpressuredSourceLocked(
    const void* source,
    int32_t highWaterMark,
    std::function<void()> resume) {
  VELOX_CHECK_NOT_NULL(source);
  VELOX_CHECK_GE(highWaterMark, 0);
  VELOX_CHECK(resume, "Backpressured source resume callback is empty");
  if (queue_.size() <= highWaterMark) {
    return false;
  }
  const bool inserted =
      backpressuredSources_.emplace(source, std::move(resume)).second;
  VELOX_CHECK(inserted, "Source registered for backpressure more than once");
  return true;
}

std::vector<std::function<void()>>
UcxCpuRowExchangeQueue::takeBackpressuredSourcesLocked(int32_t lowWaterMark) {
  VELOX_CHECK_GE(lowWaterMark, 0);
  std::vector<std::function<void()>> callbacks;
  if (queue_.size() > lowWaterMark || backpressuredSources_.empty()) {
    return callbacks;
  }
  callbacks.reserve(backpressuredSources_.size());
  for (auto& entry : backpressuredSources_) {
    callbacks.push_back(std::move(entry.second));
  }
  backpressuredSources_.clear();
  return callbacks;
}

UcxCpuRowReceivedPtr UcxCpuRowExchangeQueue::dequeueLocked(
    int consumerId,
    bool* atEnd,
    ContinueFuture* future,
    ContinuePromise* stalePromise) {
  VELOX_CHECK_NOT_NULL(future);
  if (!error_.empty()) {
    *atEnd = true;
    VELOX_FAIL(error_);
  }

  *atEnd = false;

  UcxCpuRowReceivedPtr data = nullptr;
  if (queue_.empty()) {
    if (atEnd_) {
      *atEnd = true;
    } else {
      addPromiseLocked(consumerId, future, stalePromise);
    }
    return data;
  }

  data = std::move(queue_.front());
  queue_.pop_front();

  return data;
}

void UcxCpuRowExchangeQueue::setError(std::string_view error) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (!error_.empty()) {
      return;
    }
    error_ = error;
    atEnd_ = true;
    queue_.clear();
    promises = clearAllPromisesLocked();
  }
  clearPromises(promises);
}

} // namespace facebook::velox::ucx_exchange
