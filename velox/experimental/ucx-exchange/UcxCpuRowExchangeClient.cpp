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
  std::shared_ptr<UcxCpuRowExchangeSource> sourceToStart;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    if (!remoteTaskIds_.emplace(remoteTaskId).second) {
      // Task updates may repeat producers, including after the client closes.
      return;
    }
    if (!closed_) {
      sourceToStart = UcxCpuRowExchangeSource::create(taskId_, remoteTaskId, queue_);
      sources_.push_back(sourceToStart);
      queue_->addSourceLocked();
      sourceToStart->setRegistered();
    }
  }

  if (!sourceToStart) {
    // Task can discover producers after its receive client closes. Establish
    // their control connection and abort it so producer output is released.
    // This source must not participate in the closed receive queue's counts.
    sourceToStart = UcxCpuRowExchangeSource::create(taskId_, remoteTaskId, queue_);
    sourceToStart->close();
  }
  // Registration can synchronously reject during communicator shutdown and
  // close() may deliver an end marker, so both run outside the queue mutex.
  sourceToStart->start();
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
