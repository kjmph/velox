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

#include <string_view>

#include "velox/exec/ExchangeClient.h"
#include "velox/experimental/ucx-exchange/UcxExchangeQueue.h"
#include "velox/experimental/ucx-exchange/UcxExchangeSource.h"

namespace facebook::velox::ucx_exchange {

// Handle for a set of producers. This may be shared by multiple UcxExchanges,
// one per consumer thread.
class UcxExchangeClient
    : public exec::ExchangeClient,
      public std::enable_shared_from_this<UcxExchangeClient> {
 public:
  // used for some primitive type of flow control, limits the size of elements
  // in the UcxExchangeQueue
  static constexpr int32_t kDefaultMaxQueuedColumns = 32;
  static constexpr std::chrono::milliseconds kRequestDataMaxWait{100};

  /// @param taskId Id of the consuming task, for logging.
  /// @param destination Partition the consuming task reads. Only validated:
  /// each split URL names the partition its source fetches.
  /// @param numberOfConsumers Number of UcxExchange operators sharing this
  /// client.
  /// @param receiveHighWaterBytes Query-level target for receive buffering.
  /// Zero disables byte accounting and retains the legacy table-count limit.
  /// @param requestDataSizesMaxWaitSec Max wait for a data-size request.
  UcxExchangeClient(
      std::string taskId,
      int destination,
      int32_t numberOfConsumers,
      uint64_t receiveHighWaterBytes = 0,
      int32_t requestDataSizesMaxWaitSec = 10);

  ~UcxExchangeClient() override;

  // Creates a UCX exchange source and starts fetching data from the specified
  // upstream task. If 'close' has been called already, creates an exchange
  // source and immediately closes it to notify the upstream task that data is
  // no longer needed. Repeated calls with the same 'taskId' are ignored.
  void addRemoteTaskId(std::string_view remoteTaskId) override;

  void noMoreRemoteTasks() override;

  // Closes all exchange sources.
  void close() override;

  // Returns runtime statistics aggregated across all of the exchange sources.
  // ExchangeClient is expected to report background CPU time by including a
  // runtime metric named Operator::kBackgroundCpuTimeNanos.
  folly::F14FastMap<std::string, RuntimeMetric> stats() const override;

  const std::shared_ptr<UcxExchangeQueue>& queue() const {
    return queue_;
  }

  /// Returns a PackedTableWithStream object from the queue or null.
  ///
  /// If no data is available returns a nullptr and sets 'atEnd' to true if no
  /// more data is expected. If data is still expected, sets 'atEnd' to false
  /// and sets 'future' to a Future that will complete when data arrives.
  ///
  PackedTableWithStreamPtr
  next(int consumerId, bool* atEnd, ContinueFuture* future);

  /// Called exactly once when a dequeued packed table's CudfVector releases
  /// its GPU storage.
  void releaseInFlightReceiveBytes(uint64_t bytes);

  /// Terminates receive processing after downstream discovers that retained
  /// GPU data is not safe to release (for example, stream synchronization
  /// failed). Safe to call from a CudfVector release callback.
  void failReceive(std::string_view error) noexcept;

  std::string toString() const override;

  folly::dynamic toJson() const override;

  bool tracksInFlightReceiveBytes() const {
    return queue_->tracksInFlightReceiveBytes();
  }

  /// Max wait for a data-size request to a producer.
  std::chrono::seconds requestDataSizesMaxWaitSec() const {
    return requestDataSizesMaxWaitSec_;
  }

  /// Ids of the upstream tasks added so far.
  const std::unordered_set<std::string>& getRemoteTaskIdList() const {
    return remoteTaskIds_;
  }

 private:
  // Handy for ad-hoc logging.
  const std::string taskId_;
  const int destination_;
  const int32_t maxQueuedColumns_;
  const std::chrono::seconds requestDataSizesMaxWaitSec_;

  const std::shared_ptr<UcxExchangeQueue> queue_;

  std::unordered_set<std::string> remoteTaskIds_;
  std::vector<std::shared_ptr<UcxExchangeSource>> sources_;
  bool closed_{false};

  // Total number of packed_clumns in flight.
  int64_t totalPendingColumns_{0};

  // Diagnostic counters for progress and flow control.
  int64_t totalDequeued_{0};
  bool inFlowControl_{false};
};

} // namespace facebook::velox::ucx_exchange
