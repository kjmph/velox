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
#include "velox/experimental/ucx-exchange/UcxCpuRowExchangeClient.h"

namespace facebook::velox::ucx_exchange {

void UcxCpuRowExchangeClient::addRemoteTaskId(std::string_view remoteTaskId) {
  {
    std::lock_guard<std::mutex> l(queue_->mutex());

    bool duplicate = !remoteTaskIds_.insert(std::string{remoteTaskId}).second;
    if (duplicate) {
      return;
    }

    if (closed_) {
      return;
    }

    auto source =
        UcxCpuRowExchangeSource::create(taskId_, remoteTaskId, queue_);
    sources_.push_back(source);
    queue_->addSourceLocked();
    source->setRegistered();
    source->start();
  }
}

void UcxCpuRowExchangeClient::noMoreRemoteTasks() {
  queue_->noMoreSources();
}

void UcxCpuRowExchangeClient::close() {
  std::vector<std::shared_ptr<UcxCpuRowExchangeSource>> sources;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    if (closed_) {
      return;
    }
    closed_ = true;
    sources = std::move(sources_);
  }

  for (auto& source : sources) {
    source->close();
  }
  queue_->close();
}

UcxCpuRowReceivedPtr UcxCpuRowExchangeClient::next(
    int consumerId,
    bool* atEnd,
    ContinueFuture* future) {
  UcxCpuRowReceivedPtr data;
  ContinuePromise stalePromise = ContinuePromise::makeEmpty();
  std::vector<std::shared_ptr<UcxCpuRowExchangeSource>> sourcesToResume;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    if (closed_) {
      *atEnd = true;
      return data;
    }

    *atEnd = false;
    data = queue_->dequeueLocked(consumerId, atEnd, future, &stalePromise);
    if (*atEnd) {
      return data;
    }

    // A parked consumer on an empty/low queue is also a receive-credit signal.
    // If sources stopped posting receives due to backpressure, requiring a
    // successful dequeue to resume them can deadlock: no source is active, so
    // no future dequeue can happen. resumeFromBackpressure() is idempotent via
    // CAS, so waking all sources at low-water is cheap and closes that edge.
    if (queue_->size() <=
        UcxCpuRowExchangeSource::backpressureLowWaterMark()) {
      sourcesToResume.assign(sources_.begin(), sources_.end());
    }
  }

  for (auto& source : sourcesToResume) {
    source->resumeFromBackpressure();
  }
  if (stalePromise.valid()) {
    stalePromise.setValue();
  }
  return data;
}

UcxCpuRowExchangeClient::~UcxCpuRowExchangeClient() {
  close();
}

std::string UcxCpuRowExchangeClient::toString() const {
  std::stringstream out;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    for (auto& source : sources_) {
      out << source->toString() << std::endl;
    }
  }
  return out.str();
}

folly::dynamic UcxCpuRowExchangeClient::toJson() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["taskId"] = taskId_;
  obj["closed"] = closed_;
  folly::dynamic clientsObj = folly::dynamic::object;
  int index = 0;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    for (auto& source : sources_) {
      clientsObj[std::to_string(index++)] = source->toJson();
    }
  }
  obj["clients"] = clientsObj;
  return obj;
}

} // namespace facebook::velox::ucx_exchange
