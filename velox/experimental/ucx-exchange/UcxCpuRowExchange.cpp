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
#include "velox/experimental/ucx-exchange/UcxCpuRowExchange.h"
#include "velox/common/Casts.h"
#include "velox/exec/OperatorUtils.h"

#include <algorithm>

using facebook::velox::exec::PrestoSerializedPage;
using facebook::velox::exec::RemoteConnectorSplit;

namespace facebook::velox::ucx_exchange {

UcxCpuRowExchange::UcxCpuRowExchange(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::PlanNode>& planNode,
    std::shared_ptr<UcxCpuRowExchangeClient> exchangeClient,
    std::string_view operatorType)
    : SourceOperator(
          driverCtx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          operatorType),
      preferredOutputBatchBytes_{
          driverCtx->queryConfig().preferredOutputBatchBytes()},
      serdeOptions_{exec::getVectorSerdeOptions(
          common::stringToCompressionKind(operatorCtx_->driverCtx()
                                              ->queryConfig()
                                              .shuffleCompressionKind()),
          // The CPU UCX exchange currently sends Presto-serde pages.
          // Keep the receive side aligned with the producer-side check.
          VectorSerde::kindName(VectorSerde::Kind::kPresto),
          /*minCompressionRatio=*/std::nullopt,
          operatorCtx_->driverCtx()
              ->queryConfig()
              .minShuffleCompressionPageSizeBytes())},
      driverId_{driverCtx->driverId} {
  if (exchangeClient) {
    exchangeClient_ = std::move(exchangeClient);
  } else {
    auto task = operatorCtx_->task();
    exchangeClient_ = std::make_shared<UcxCpuRowExchangeClient>(
        task->taskId(), task->destination(), /*numConsumers=*/1);
  }
  // The CPU UCX adapter shares one exchange client across all drivers in a
  // task pipeline. Claim split ownership on that shared client instead of
  // assuming task-global driverId 0 belongs to this pipeline.
  processSplits_ = exchangeClient_->claimSplitProcessor();
}

UcxCpuRowExchange::~UcxCpuRowExchange() {
  close();
}

void UcxCpuRowExchange::addRemoteTaskIds(
    std::vector<std::string>& remoteTaskIds) {
  std::shuffle(std::begin(remoteTaskIds), std::end(remoteTaskIds), rng_);
  for (const std::string& remoteTaskId : remoteTaskIds) {
    exchangeClient_->addRemoteTaskId(remoteTaskId);
  }
  stats_.wlock()->numSplits += remoteTaskIds.size();
}

void UcxCpuRowExchange::getSplits(ContinueFuture* future) {
  if (!processSplits_) {
    return;
  }
  if (noMoreSplits_) {
    return;
  }
  std::vector<std::string> remoteTaskIds;
  for (;;) {
    exec::Split split;
    auto reason = operatorCtx_->task()->getSplitOrFuture(
        operatorCtx_->driverCtx()->driverId,
        operatorCtx_->driverCtx()->splitGroupId,
        planNodeId(),
        /*maxPreloadSplits=*/0,
        /*preload=*/nullptr,
        split,
        *future);
    if (reason != BlockingReason::kNotBlocked) {
      addRemoteTaskIds(remoteTaskIds);
      return;
    }

    if (split.hasConnectorSplit()) {
      auto remoteSplit =
          std::dynamic_pointer_cast<RemoteConnectorSplit>(split.connectorSplit);
      VELOX_CHECK_NOT_NULL(remoteSplit, "Wrong type of split");
      remoteTaskIds.push_back(remoteSplit->taskId);
      continue;
    }

    addRemoteTaskIds(remoteTaskIds);
    exchangeClient_->noMoreRemoteTasks();
    noMoreSplits_ = true;
    if (atEnd_) {
      operatorCtx_->task()->multipleSplitsFinished(
          false, stats_.rlock()->numSplits, 0);
    }
    return;
  }
}

BlockingReason UcxCpuRowExchange::isBlocked(ContinueFuture* future) {
  if (currentReceived_ || atEnd_) {
    return BlockingReason::kNotBlocked;
  }

  if (!splitFuture_.valid()) {
    getSplits(&splitFuture_);
  }

  ContinueFuture dataFuture = ContinueFuture::makeEmpty();
  currentReceived_ = exchangeClient_->next(driverId_, &atEnd_, &dataFuture);
  if (currentReceived_ || atEnd_) {
    if (atEnd_ && noMoreSplits_) {
      const auto numSplits = stats_.rlock()->numSplits;
      operatorCtx_->task()->multipleSplitsFinished(false, numSplits, 0);
    }
    return BlockingReason::kNotBlocked;
  }

  if (splitFuture_.valid()) {
    std::vector<ContinueFuture> futures;
    futures.push_back(std::move(splitFuture_));
    futures.push_back(std::move(dataFuture));
    *future = folly::collectAny(futures).unit();
    return BlockingReason::kWaitForSplit;
  }

  *future = std::move(dataFuture);
  return BlockingReason::kWaitForProducer;
}

bool UcxCpuRowExchange::isFinished() {
  return atEnd_ && !currentReceived_;
}

RowVectorPtr UcxCpuRowExchange::getOutput() {
  if (!currentReceived_) {
    return nullptr;
  }

  // Convert the received chunk into a PrestoSerializedPage on first
  // touch. The page owns the IOBuf chain and `currentStream_` is a view
  // into it; both stay alive until the stream is fully drained, which
  // may take many getOutput() calls when the bundle contains many
  // serialized vectors.
  if (!currentPage_) {
    currentPage_ = std::make_unique<PrestoSerializedPage>(
        std::move(currentReceived_->data),
        /*onDestructionCb=*/nullptr,
        /*numRows=*/std::nullopt);
    currentStream_ = currentPage_->prepareStreamForDeserialize();
    // Charge the page bytes once, when it arrives, regardless of how
    // many getOutput() calls it spans.
    currentPageBytesPending_ = currentPage_->size();
  }

  auto* serde =
      getNamedVectorSerde(VectorSerde::kindName(VectorSerde::Kind::kPresto));

  // Cap per-call output at ~preferredOutputBatchBytes worth of rows.
  // Mirrors standard Velox Exchange::getOutputFromColumnarPages: a
  // 64 MB UCX bundle can hold hundreds of thousands of rows, and
  // returning all of them as one RowVector causes downstream operators
  // to thrash on cache and lose pipelining benefits. The first call
  // uses kInitialOutputRows; subsequent calls use the running estimate.
  const auto numRows = estimatedRowSize_.has_value()
      ? std::max(
            (preferredOutputBatchBytes_ / estimatedRowSize_.value()),
            static_cast<uint64_t>(kInitialOutputRows))
      : static_cast<uint64_t>(kInitialOutputRows);

  vector_size_t resultOffset = 0;
  while (!currentStream_->atEnd() &&
         resultOffset < static_cast<vector_size_t>(numRows)) {
    serde->deserialize(
        currentStream_.get(),
        pool(),
        outputType_,
        &result_,
        resultOffset,
        serdeOptions_.get());
    resultOffset = result_->size();
  }

  RowVectorPtr out;
  std::swap(out, result_);
  VELOX_CHECK_NOT_NULL(out, "Deserialize must produce a result");

  // Update the running row-size estimate so subsequent batches size
  // correctly relative to preferredOutputBatchBytes.
  estimatedRowSize_ = std::max<uint64_t>(
      out->estimateFlatSize() / std::max<vector_size_t>(out->size(), 1),
      estimatedRowSize_.value_or(1));

  // Charge the bundle's bytes to stats only once, on the call that
  // first touches it. Later calls drain more rows from the same bundle
  // for free.
  const uint64_t bytesToCharge = currentPageBytesPending_;
  currentPageBytesPending_ = 0;

  // If the stream is exhausted, drop the bundle so isBlocked() can pull
  // the next chunk. Otherwise keep state for the next call to resume
  // where we left off.
  if (currentStream_->atEnd()) {
    currentStream_.reset();
    currentPage_.reset();
    currentReceived_.reset();
  }

  if (bytesToCharge > 0) {
    recordInputStats(bytesToCharge, out);
  } else {
    auto lockedStats = stats_.wlock();
    lockedStats->rawInputPositions += out->size();
    lockedStats->addInputVector(out->estimateFlatSize(), out->size());
  }
  return out;
}

void UcxCpuRowExchange::recordInputStats(
    uint64_t rawInputBytes,
    const RowVectorPtr& result) {
  auto lockedStats = stats_.wlock();
  lockedStats->rawInputBytes += rawInputBytes;
  lockedStats->rawInputPositions += result->size();
  lockedStats->addInputVector(result->estimateFlatSize(), result->size());
}

void UcxCpuRowExchange::close() {
  if (closed_) {
    return;
  }
  closed_ = true;
  SourceOperator::close();
  currentStream_.reset();
  currentPage_.reset();
  currentReceived_.reset();
  // The CPU UCX adapter shares one exchange client across all drivers in a
  // task pipeline. Individual operator close must not tear down the shared
  // client while sibling drivers are still blocked on it; the client destructor
  // closes it when the last operator releases its shared_ptr.
  exchangeClient_ = nullptr;
}

} // namespace facebook::velox::ucx_exchange
