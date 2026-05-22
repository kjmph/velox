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
    VLOG(3) << "@" << taskId_
            << " Added remote split for task: " << remoteTaskId;
  }
}

void UcxCpuRowExchangeClient::noMoreRemoteTasks() {
  VLOG(3) << "@" << taskId_
          << " UcxCpuRowExchangeClient::noMoreRemoteTasks called.";
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

folly::F14FastMap<std::string, RuntimeMetric> UcxCpuRowExchangeClient::stats()
    const {
  folly::F14FastMap<std::string, RuntimeMetric> stats;
  std::lock_guard<std::mutex> l(queue_->mutex());
  for (const auto& source : sources_) {
    for (const auto& [name, value] : source->metrics()) {
      auto [iter, inserted] = stats.try_emplace(name, value.unit);
      iter->second.merge(value);
    }
  }

  stats.insert_or_assign(
      "ucxCpuRowExchangeClient.peakBytes",
      RuntimeMetric(queue_->peakBytes(), RuntimeCounter::Unit::kBytes));
  stats.insert_or_assign(
      "ucxCpuRowExchangeClient.numReceivedPayloads",
      RuntimeMetric(queue_->receivedPayloads()));
  stats.insert_or_assign(
      "ucxCpuRowExchangeClient.peakQueuedPayloads",
      RuntimeMetric(queue_->peakSize()));
  stats.insert_or_assign(
      "ucxCpuRowExchangeClient.maxQueuedPayloads",
      RuntimeMetric(maxQueuedPayloads_));
  stats.insert_or_assign(
      "ucxCpuRowExchangeClient.averageReceivedPayloadBytes",
      RuntimeMetric(
          queue_->averageReceivedPayloadBytes(), RuntimeCounter::Unit::kBytes));
  return stats;
}

UcxCpuRowReceivedPtr UcxCpuRowExchangeClient::next(
    int consumerId,
    bool* atEnd,
    ContinueFuture* future) {
  VLOG(3) << "@" << taskId_
          << " UcxCpuRowExchangeClient::next consumerId=" << consumerId;
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

    // The push-based UCX exchange has no "request more" knob; this
    // counter is purely diagnostic. Real backpressure lives in
    // UcxCpuRowExchangeSource::process()'s ReadyToReceive branch.
    if (data != nullptr && queue_->size() > maxQueuedPayloads_) {
      if (!inFlowControl_) {
        inFlowControl_ = true;
        VLOG(1) << "[FLOW-CTRL-CPU] @" << taskId_ << " consumer=" << consumerId
                << " entering flow control" << " queueSize=" << queue_->size()
                << " maxQueued=" << maxQueuedPayloads_;
      }
    } else if (inFlowControl_ && data != nullptr) {
      inFlowControl_ = false;
      VLOG(1) << "[FLOW-CTRL-CPU] @" << taskId_ << " consumer=" << consumerId
              << " leaving flow control" << " queueSize=" << queue_->size()
              << " maxQueued=" << maxQueuedPayloads_;
    }

    if (data != nullptr) {
      ++totalDequeued_;
      if (totalDequeued_ % 1000 == 0) {
        VLOG(1) << "[PROGRESS-CPU] @" << taskId_ << " consumer=" << consumerId
                << " dequeued=" << totalDequeued_
                << " queueSize=" << queue_->size()
                << " queueBytes=" << queue_->totalBytes();
      }
    }

    // Collect sources to resume while holding the queue mutex; the
    // resume call grabs the WorkQueue mutex, so we must not nest the
    // two (queue mutex then WorkQueue mutex would deadlock against the
    // Communicator thread).
    if (data != nullptr &&
        queue_->size() <= UcxCpuRowExchangeSource::backpressureLowWaterMark()) {
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
