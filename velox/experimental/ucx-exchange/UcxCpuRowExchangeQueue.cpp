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
#include <glog/logging.h>

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
    VLOG(4) << "[EX-QUEUE-CPU] source completed (null enqueued)"
            << " numCompleted=" << numCompleted_
            << " numSources=" << numSources_
            << " noMoreSources=" << noMoreSources_;
    auto completedPromises = checkCompleteLocked();
    promises.reserve(promises.size() + completedPromises.size());
    for (auto& promise : completedPromises) {
      promises.push_back(std::move(promise));
    }
    return;
  }

  auto dataSize = data->numBytes;
  totalBytes_ += dataSize;
  if (peakBytes_ < totalBytes_) {
    peakBytes_ = totalBytes_;
  }

  ++receivedPayloads_;
  receivedBytes_ += dataSize;

  queue_.push_back(std::move(data));

  // High-water-mark alerts: log when queue size crosses thresholds.
  auto newSize = static_cast<int64_t>(queue_.size());
  if (newSize > peakSize_) {
    if ((peakSize_ < 100 && newSize >= 100) ||
        (peakSize_ < 1000 && newSize >= 1000) ||
        (peakSize_ < 10000 && newSize >= 10000)) {
      VLOG(1) << "[EX-QUEUE-CPU] high water mark: queueSize=" << newSize
              << " peakBytes=" << peakBytes_
              << " receivedPayloads=" << receivedPayloads_;
    }
    peakSize_ = newSize;
  }

  size_t wokenConsumers = 0;
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
    ++wokenConsumers;
  }
  if (wokenConsumers > 0) {
    VLOG(4) << "[EX-QUEUE-CPU] waking " << wokenConsumers << " consumers"
            << " queueSize=" << queue_.size();
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
      VLOG(4) << "[EX-QUEUE-CPU] consumer=" << consumerId
              << " blocked (empty queue, waiting for data)"
              << " numSources=" << numSources_
              << " numCompleted=" << numCompleted_
              << " waitingConsumers=" << (promises_.size() + 1);
      addPromiseLocked(consumerId, future, stalePromise);
    }
    return data;
  }

  data = std::move(queue_.front());
  queue_.pop_front();
  totalBytes_ -= data->numBytes;

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
