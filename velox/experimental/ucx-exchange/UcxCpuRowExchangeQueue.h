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
#include <cinttypes>
#include <memory>
#include "velox/common/base/Exceptions.h"
#include "velox/common/future/VeloxPromise.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowQueues.h"

/// CPU RowVector mirror of UcxExchangeQueue. Same promise/dequeue
/// plumbing, but the queue element is `std::unique_ptr<UcxCpuRowPayload>`
/// - the IOBuf-chain payload type is reused on both producer and
/// consumer sides since PrestoSerializer-produced bytes are a single
/// self-describing chunk.

namespace facebook::velox::ucx_exchange {

using UcxCpuRowReceivedPtr = std::unique_ptr<UcxCpuRowPayload>;

class UcxCpuRowExchangeQueue {
 public:
  explicit UcxCpuRowExchangeQueue(int32_t numberOfConsumers)
      : numberOfConsumers_{numberOfConsumers} {
    VELOX_CHECK_GE(numberOfConsumers, 1);
  }

  ~UcxCpuRowExchangeQueue() {
    clearAllPromises();
  }

  std::mutex& mutex() {
    return mutex_;
  }

  bool empty() const {
    return queue_.empty();
  }

  /// Enqueues `data`. If `data` is nullptr, signals one source completed;
  /// when the last source finishes, all waiting consumer promises are
  /// returned in `promises`. Otherwise, wakes one waiting consumer (if
  /// any) and returns its promise.
  void enqueueLocked(
      UcxCpuRowReceivedPtr&& data,
      std::vector<ContinuePromise>& promises);

  /// Mark the queue as errored. All pending promises are returned and
  /// the next dequeue throws.
  void setError(std::string_view error);

  bool isInError() {
    return !error_.empty();
  }

  /// Pulls one chunk off the queue. Returns nullptr when blocked (sets
  /// `future`) or when at end (sets `*atEnd = true`). If a stale promise
  /// for this consumer existed, it is returned in `stalePromise` so the
  /// caller can fulfill it outside the lock.
  UcxCpuRowReceivedPtr dequeueLocked(
      int consumerId,
      bool* atEnd,
      ContinueFuture* future,
      ContinuePromise* stalePromise);

  int32_t size() const {
    return queue_.size();
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

  const int32_t numberOfConsumers_;

  int numCompleted_{0};
  int numSources_{0};
  bool noMoreSources_{false};
  bool atEnd_{false};

  std::mutex mutex_;
  std::deque<UcxCpuRowReceivedPtr> queue_;
  // Map from consumer id to the waiting promise.
  folly::F14FastMap<int, ContinuePromise> promises_;

  std::string error_;
};

} // namespace facebook::velox::ucx_exchange
