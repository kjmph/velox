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

#include <random>
#include "velox/common/memory/ByteStream.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/Operator.h"
#include "velox/exec/SerializedPage.h"
#include "velox/exec/Task.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowExchangeClient.h"

/// CPU RowVector mirror of UcxExchange. Pulls received chunks from the
/// UcxCpuRowExchangeClient, wraps each into a PrestoSerializedPage, and
/// hands the page's stream to PrestoVectorSerde::deserialize to produce
/// output RowVectors. Mirrors the cudf operator's split-handling and
/// blocking-future plumbing.

namespace facebook::velox::ucx_exchange {

using exec::BlockingReason;
using exec::DriverCtx;
using exec::SourceOperator;

class UcxCpuRowExchange : public SourceOperator {
 public:
  UcxCpuRowExchange(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::PlanNode>& planNode,
      std::shared_ptr<UcxCpuRowExchangeClient> exchangeClient,
      std::string_view operatorType = "UcxCpuRowExchange");

  ~UcxCpuRowExchange() override;

  [[nodiscard]] BlockingReason isBlocked(ContinueFuture* future) override;

  [[nodiscard]] bool isFinished() override;

  [[nodiscard]] RowVectorPtr getOutput() override;

  void close() override;

 private:
  void addRemoteTaskIds(std::vector<std::string>& remoteTaskIds);

  void getSplits(ContinueFuture* future);

  void recordInputStats(uint64_t rawInputBytes, const RowVectorPtr& result);

  std::shared_ptr<UcxCpuRowExchangeClient> exchangeClient_;

  const uint64_t preferredOutputBatchBytes_;
  const std::unique_ptr<VectorSerde::Options> serdeOptions_;
  bool processSplits_{false};
  const int driverId_;
  bool noMoreSplits_{false};

  ContinueFuture splitFuture_{ContinueFuture::makeEmpty()};

  // The chunk we're currently deserializing. One UCX recv is one page.
  // The page may contain many bundled Presto serialized vectors and is
  // drained across multiple getOutput() calls, each returning a
  // bounded-size RowVector. Cleared only once its stream is exhausted.
  UcxCpuRowReceivedPtr currentReceived_;
  std::unique_ptr<exec::PrestoSerializedPage> currentPage_;
  std::unique_ptr<ByteInputStream> currentStream_;
  // Number of bytes the current page contributed to rawInputBytes
  // stats. Charged once on first touch so we don't double-count when
  // a single page produces multiple getOutput() returns.
  uint64_t currentPageBytesPending_{0};

  // Result vector accumulated across deserialize calls within one
  // getOutput() call only; reset on every call. A single bundled
  // page contains many serialized vectors, but we cap the per-call
  // result to ~preferredOutputBatchBytes worth of rows so downstream
  // operators don't get one giant batch that breaks their per-batch
  // assumptions.
  RowVectorPtr result_;
  std::optional<uint64_t> estimatedRowSize_;
  static constexpr vector_size_t kInitialOutputRows = 64;

  bool atEnd_{false};
  bool closed_{false};
  std::default_random_engine rng_{std::random_device{}()};
};

} // namespace facebook::velox::ucx_exchange
