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
#include "velox/experimental/ucx-exchange/UcxCpuRowDriverAdapter.h"

namespace facebook::velox::ucx_exchange {

void UcxCpuRowExchangeClient::addRemoteTaskId(std::string_view remoteTaskId) {
  // Task creates clients while holding its mutex. Defer listener startup until
  // producers are attached, before taking the receive queue mutex.
  if (cpuUcxExchangeEnabled()) {
    VELOX_CHECK(startCpuUcxExchange(), "CPU UCX communicator failed to start");
  }
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
  std::vector<std::function<void()>> sourcesToResume;
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
    // Source registration and this drain are linearized by queue_->mutex(), so
    // a source cannot park after the final low-water dequeue and lose its wake.
    sourcesToResume = queue_->takeBackpressuredSourcesLocked(
        UcxCpuRowExchangeSource::backpressureLowWaterMark());
  }

  for (auto& resume : sourcesToResume) {
    resume();
  }
  if (stalePromise.valid()) {
    stalePromise.setValue();
  }
  return data;
}

UcxCpuRowExchangeClient::~UcxCpuRowExchangeClient() {
  close();
}

folly::F14FastMap<std::string, RuntimeMetric>
UcxCpuRowExchangeClient::stats() const {
  // Receive bytes and rows are recorded by UcxCpuRowExchange. The shared
  // communicator does not currently attribute its CPU time to individual tasks.
  return {};
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
  folly::dynamic clientsObj = folly::dynamic::object;
  int index = 0;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    obj["closed"] = closed_;
    for (auto& source : sources_) {
      clientsObj[std::to_string(index++)] = source->toJson();
    }
  }
  obj["clients"] = clientsObj;
  return obj;
}

} // namespace facebook::velox::ucx_exchange
