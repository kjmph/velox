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

#include "velox/experimental/ucx-exchange/DynamicUcxOutputBufferReader.h"

#include <glog/logging.h>

#include "velox/common/base/Exceptions.h"
#include "velox/exec/DefaultOutputBufferManager.h"

namespace facebook::velox::ucx_exchange {

DynamicUcxOutputBufferReader::DynamicUcxOutputBufferReader(
    std::shared_ptr<exec::DefaultOutputBufferManager> manager,
    std::string taskId,
    int destination)
    : manager_(std::move(manager)),
      taskId_(std::move(taskId)),
      destination_(destination),
      sequence_(0) {
  VELOX_CHECK_NOT_NULL(manager_, "Output buffer manager is null");
  VELOX_CHECK_GE(destination_, 0);
}

DynamicUcxOutputBufferReader::~DynamicUcxOutputBufferReader() {
  close();
}

void DynamicUcxOutputBufferReader::request(
    uint64_t maxBytes,
    DataCallback callback) {
  VELOX_CHECK(callback != nullptr, "Data callback is null");

  if (closed_.load(std::memory_order_acquire) ||
      atEnd_.load(std::memory_order_acquire)) {
    callback(Data{{}, sequence_.load(std::memory_order_acquire), {}, true});
    return;
  }

  const int64_t requested = sequence_.load(std::memory_order_acquire);

  auto deliver = callback;

  const bool served = manager_->getData(
      taskId_,
      destination_,
      maxBytes,
      requested,
      [weakSelf =
           std::weak_ptr<DynamicUcxOutputBufferReader>(shared_from_this()),
       callback = std::move(callback)](
          std::vector<std::unique_ptr<folly::IOBuf>> pages,
          int64_t sequence,
          std::vector<int64_t> remainingBytes) mutable {
        auto self = weakSelf.lock();
        if (self == nullptr || self->closed_.load(std::memory_order_acquire)) {
          return;
        }
        Data data;
        data.sequence = sequence;
        data.remainingBytes = std::move(remainingBytes);

        for (auto& page : pages) {
          if (page == nullptr) {
            // A null entry is the producer's end marker. It is only ever the
            // last element, and it is not a page to ship.
            data.atEnd = true;
            break;
          }
          data.pages.push_back(std::move(page));
        }

        // Advance past the pages actually delivered so the next request both
        // asks for new data and releases these.
        self->sequence_.store(
            sequence + static_cast<int64_t>(data.pages.size()),
            std::memory_order_release);
        if (data.atEnd) {
          self->atEnd_.store(true, std::memory_order_release);
        }

        callback(std::move(data));
      });

  if (!served) {
    atEnd_.store(true, std::memory_order_release);
    deliver(Data{{}, sequence_.load(std::memory_order_acquire), {}, true});
  }
}

void DynamicUcxOutputBufferReader::close() {
  closed_.store(true, std::memory_order_release);
}

void DynamicUcxOutputBufferReader::deleteResults() {
  // Not called from close(): several readers may serve one task, and this
  // releases the destination for all of them.
  try {
    manager_->deleteResults(taskId_, destination_);
  } catch (const std::exception& e) {
    // Reached from teardown paths, where throwing would mask the original
    // failure.
    LOG(WARNING) << "Failed to delete results for task " << taskId_
                 << " destination " << destination_ << ": " << e.what();
  }
}

} // namespace facebook::velox::ucx_exchange
