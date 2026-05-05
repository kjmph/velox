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

#include <unordered_set>
#include "velox/experimental/ucx-exchange/UcxCpuRowExchangeQueue.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowExchangeSource.h"

namespace facebook::velox::ucx_exchange {

/// CPU RowVector mirror of UcxExchangeClient. One per consumer task,
/// fronts a UcxCpuRowExchangeQueue and the set of remote sources that
/// feed it.
class UcxCpuRowExchangeClient
    : public std::enable_shared_from_this<UcxCpuRowExchangeClient> {
 public:
  static constexpr std::chrono::milliseconds kRequestDataMaxWait{100};

  UcxCpuRowExchangeClient(
      std::string taskId,
      int destination,
      int32_t numberOfConsumers,
      int32_t requestDataSizesMaxWaitSec = 10)
      : taskId_{std::move(taskId)},
        destination_(destination),
        maxQueuedPayloads_(
            UcxCpuRowExchangeSource::backpressureHighWaterMark()),
        kRequestDataSizesMaxWaitSec_(requestDataSizesMaxWaitSec),
        queue_(std::make_shared<UcxCpuRowExchangeQueue>(numberOfConsumers)) {
    VELOX_CHECK_GE(
        destination, 0, "Exchange client destination must not be negative");
  }

  ~UcxCpuRowExchangeClient();

  void addRemoteTaskId(std::string_view remoteTaskId);

  void noMoreRemoteTasks();

  void close();

  folly::F14FastMap<std::string, RuntimeMetric> stats() const;

  const std::shared_ptr<UcxCpuRowExchangeQueue>& queue() const {
    return queue_;
  }

  /// Pulls the next received payload off the queue. Returns nullptr and
  /// sets atEnd if the stream is finished, or sets `future` if blocked
  /// waiting for data.
  UcxCpuRowReceivedPtr
  next(int consumerId, bool* atEnd, ContinueFuture* future);

  std::string toString() const;

  folly::dynamic toJson() const;

  std::chrono::seconds requestDataSizesMaxWaitSec() const {
    return kRequestDataSizesMaxWaitSec_;
  }

  const std::unordered_set<std::string>& getRemoteTaskIdList() const {
    return remoteTaskIds_;
  }

 private:
  const std::string taskId_;
  const int destination_;
  const int32_t maxQueuedPayloads_;
  const std::chrono::seconds kRequestDataSizesMaxWaitSec_;

  const std::shared_ptr<UcxCpuRowExchangeQueue> queue_;

  std::unordered_set<std::string> remoteTaskIds_;
  std::vector<std::shared_ptr<UcxCpuRowExchangeSource>> sources_;
  bool closed_{false};

  int64_t totalDequeued_{0};
  bool inFlowControl_{false};
};

} // namespace facebook::velox::ucx_exchange
