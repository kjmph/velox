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

#include <list>
#include <memory>
#include <mutex>

template <typename T>
class WorkQueue {
 public:
  WorkQueue() = default;
  ~WorkQueue() = default;

  // non-copyable
  WorkQueue(const WorkQueue&) = delete;
  WorkQueue& operator=(const WorkQueue&) = delete;

  // push never blocks. Returns false after stopAccepting() has closed intake.
  bool push(std::shared_ptr<T> item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!accepting_) {
      return false;
    }
    queue_.emplace_back(std::move(item));
    return true;
  }

  // Serializes queue shutdown with in-flight producers. Once this returns, a
  // producer that has not already enqueued an item can no longer do so.
  void stopAccepting() {
    std::lock_guard<std::mutex> lock(mutex_);
    accepting_ = false;
  }

  // pop never blocks; returns nullptr if empty
  std::shared_ptr<T> pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return nullptr;
    }

    auto item = std::move(queue_.front());
    queue_.pop_front();
    return item;
  }

  // remove an element from the queue.
  bool erase(std::shared_ptr<T> item) {
    bool erased = false;
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = queue_.begin(); it != queue_.end();) {
      if (*it != item) {
        ++it;
        continue;
      }
      it = queue_.erase(it);
      erased = true;
    }
    return erased;
  }

  // helpers
  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  mutable std::mutex mutex_;
  std::list<std::shared_ptr<T>> queue_;
  bool accepting_{true};
};
