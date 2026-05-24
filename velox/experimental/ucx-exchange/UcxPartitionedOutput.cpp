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
#include "velox/experimental/ucx-exchange/UcxPartitionedOutput.h"
#include <fmt/format.h>
#include "velox/core/PlanNode.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/partitioning.hpp>

#include <algorithm>
#include <limits>

using namespace facebook::velox::cudf_velox;
using facebook::velox::exec::Task;
namespace facebook::velox::ucx_exchange {

namespace {
uint64_t multiplySaturated(uint64_t value, uint64_t multiplier) {
  if (value == 0) {
    return 0;
  }
  if (multiplier > std::numeric_limits<uint64_t>::max() / value) {
    return std::numeric_limits<uint64_t>::max();
  }
  return value * multiplier;
}

uint64_t divideCeil(uint64_t value, uint64_t divisor) {
  VELOX_CHECK_GT(divisor, 0);
  return value / divisor + (value % divisor == 0 ? 0 : 1);
}
} // namespace

// Computes a mapping from names in n2 to names in n1
// and returns that mapping in remap.
// Names in n2 must occurs in n1.
static void getRemapping(
    const std::vector<std::string>& n1,
    const std::vector<std::string>& n2,
    std::vector<uint32_t>& remap) {
  std::unordered_map<std::string, uint32_t> lookup;
  for (uint32_t i = 0; i < n1.size(); ++i) {
    lookup[n1[i]] = i;
  }

  remap.clear();
  remap.reserve(n2.size());
  for (const auto& key : n2) {
    remap.push_back(lookup.at(key));
  }
}

UcxPartitionedOutput::UcxPartitionedOutput(
    int32_t operatorId,
    exec::DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
    bool eagerFlush)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "cudfPartitionedOutput"),
      NvtxHelper(
          nvtx3::rgb{255, 215, 0}, // Gold
          operatorId,
          fmt::format("[{}]", planNode->id())),
      queueManager_(UcxOutputQueueManager::getInstanceRef()),
      numPartitions_(planNode->numPartitions()),
      pipelineId_(ctx->pipelineId),
      driverId_(ctx->driverId),
      targetRowsPerChunk_(ctx->queryConfig().get<int64_t>(
          core::QueryConfig::kUcxPartitionedOutputBatchRows,
          CudfConfig::getInstance().partitionedOutputBatchRows)),
      initialPayloadBytes_(
          std::max<uint64_t>(ctx->queryConfig().maxOutputBufferSize(), 1)) {
  this->initPartitionKeys(planNode);
  auto sources = planNode->sources();
  std::vector<std::string> inNames, outNames;
  inNames.reserve(planNode->inputType()->size());
  for (int i = 0; i < planNode->inputType()->size(); ++i) {
    inNames.push_back(planNode->inputType()->nameOf(i));
  }
  outNames.reserve(planNode->outputType()->size());
  for (int i = 0; i < planNode->outputType()->size(); ++i) {
    outNames.push_back(planNode->outputType()->nameOf(i));
  }
  if (inNames != outNames) {
    getRemapping(inNames, outNames, remap_);
  }
}

void UcxPartitionedOutput::addInput(RowVectorPtr input) {
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " addInput";
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfVector, "Input must be a CudfVector");
  VELOX_CHECK(
      !future_.valid() || future_.hasValue(),
      "addInput with outstanding future!");

  const auto inputBytes = input->estimateFlatSize();
  // Record stats per-input (before buffering).
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addOutputVector(inputBytes, input->size());
  }

  pendingRows_ += cudfVector->getTableView().num_rows();
  pendingBytes_ += inputBytes;
  pendingInputs_.push_back(std::move(cudfVector));

  if (shouldFlushPending()) {
    if (!ensureOutputReservation(&future_)) {
      blockingReason_ = exec::BlockingReason::kWaitForConsumer;
      return;
    }
    flushPending();
  }
}

void UcxPartitionedOutput::flushPending() {
  if (pendingPartitionedBatch_) {
    drainPendingPartitionedBatch();
    return;
  }

  if (pendingInputs_.empty()) {
    return;
  }

  try {
    cudf::table_view tableView;
    rmm::cuda_stream_view stream = pendingInputs_.back()->stream();
    // Keeps the merged table alive while tableView references it.
    std::unique_ptr<cudf::table> mergedTable;

    if (pendingInputs_.size() == 1) {
      // Fast path: use the single input's view directly (no GPU alloc).
      auto& cv = pendingInputs_[0];
      stream = cv->stream();
      tableView = remap_.empty()
          ? cv->getTableView()
          : cv->getTableView().select(remap_.begin(), remap_.end());
    } else {
      // Collect (remapped) table views.
      std::vector<cudf::table_view> views;
      std::vector<rmm::cuda_stream_view> inputStreams;
      views.reserve(pendingInputs_.size());
      inputStreams.reserve(pendingInputs_.size());
      for (auto& v : pendingInputs_) {
        inputStreams.push_back(v->stream());
        views.push_back(
            remap_.empty()
                ? v->getTableView()
                : v->getTableView().select(remap_.begin(), remap_.end()));
      }

      cudf::detail::join_streams(inputStreams, stream);
      mergedTable = cudf::concatenate(
          views, stream, cudf::get_current_device_resource_ref());

      orderCudfVectorDeallocationsAfterStream(
          pendingInputs_, inputStreams, stream);

      // Free input GPU memory before partitioning (peak = 2x -> 1x).
      pendingInputs_.clear();
      pendingBytes_ = 0;

      tableView = mergedTable->view();
    }

    // Partition + enqueue (identical to previous addInput logic).
    auto queueManager = sharedQueueManager();
    if (numPartitions_ > 1) {
      if (partitionKeyIndices_.size() > 0 || spec_ == "gather") {
        hashPartition(tableView, stream);
        if (mergedTable == nullptr) {
          std::vector<rmm::cuda_stream_view> inputStreams;
          inputStreams.reserve(pendingInputs_.size());
          for (const auto& input : pendingInputs_) {
            inputStreams.push_back(input->stream());
          }
          orderCudfVectorDeallocationsAfterStream(
              pendingInputs_, inputStreams, stream);
        }
      } else {
        std::vector<CudfVectorPtr> vectorOwners;
        if (mergedTable == nullptr) {
          vectorOwners = std::move(pendingInputs_);
        }
        equalPartition(
            tableView, stream, std::move(mergedTable), std::move(vectorOwners));
      }
      pendingInputs_.clear();
      pendingRows_ = 0;
      pendingBytes_ = 0;
      drainPendingPartitionedBatch();
    } else {
      auto packedCols = cudf::pack(
          tableView, stream, cudf::get_current_device_resource_ref());
      stream.synchronize();
      auto packedColsPtr = std::make_unique<cudf::packed_columns>(
          std::move(packedCols.metadata), std::move(packedCols.gpu_data));
      queueManager->enqueue(
          this->taskId(), 0, std::move(packedColsPtr), tableView.num_rows());

      // Check backpressure after enqueue.
      auto blocked = queueManager->checkBlocked(this->taskId(), &future_);
      blockingReason_ = blocked ? exec::BlockingReason::kWaitForConsumer
                                : exec::BlockingReason::kNotBlocked;

      pendingInputs_.clear();
      pendingRows_ = 0;
      pendingBytes_ = 0;
      releaseOutputReservation();
    }

  } catch (const rmm::bad_alloc& e) {
    VLOG(1)
        << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
        << " caught memory alloc error, removing all memory in output queues";
    pendingInputs_.clear();
    pendingPartitionedBatch_.reset();
    pendingRows_ = 0;
    pendingBytes_ = 0;
    releaseOutputReservation();
    for (int i = 0; i < numPartitions_; i++) {
      sharedQueueManager()->deleteResults(this->taskId(), i);
    }
    throw;
  }
}

exec::BlockingReason UcxPartitionedOutput::isBlocked(ContinueFuture* future) {
  if (blockingReason_ != exec::BlockingReason::kNotBlocked) {
    *future = std::move(future_);
    blockingReason_ = exec::BlockingReason::kNotBlocked;
    return exec::BlockingReason::kWaitForConsumer;
  }
  if (shouldDrainPending()) {
    if (!pendingPartitionedBatch_ && !ensureOutputReservation(future)) {
      return exec::BlockingReason::kWaitForConsumer;
    }
    return exec::BlockingReason::kNotBlocked;
  }
  if (!finished_ &&
      sharedQueueManager()->checkBlocked(this->taskId(), future)) {
    return exec::BlockingReason::kWaitForConsumer;
  }
  return exec::BlockingReason::kNotBlocked;
}

RowVectorPtr UcxPartitionedOutput::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (finished_) {
    return nullptr;
  }
  if (shouldDrainPending()) {
    if (!pendingPartitionedBatch_ && outputReservationBytes_ == 0) {
      return nullptr;
    }
    flushPending();
    if (shouldDrainPending() ||
        blockingReason_ != exec::BlockingReason::kNotBlocked) {
      return nullptr;
    }
  }
  if (noMoreInput_) {
    sharedQueueManager()->noMoreData(this->taskId());
    finished_ = true;
  }
  return nullptr;
}

bool UcxPartitionedOutput::isFinished() {
  return finished_;
}

std::shared_ptr<facebook::velox::ucx_exchange::UcxOutputQueueManager>
UcxPartitionedOutput::sharedQueueManager() {
  auto shared_queueManager = queueManager_.lock();
  VELOX_CHECK_NOT_NULL(
      shared_queueManager, "OutputQueueManager was already destructed");
  return shared_queueManager;
}

bool UcxPartitionedOutput::shouldFlushPending() const {
  return !pendingInputs_.empty() &&
      (targetRowsPerChunk_ <= 0 || pendingRows_ >= targetRowsPerChunk_);
}

bool UcxPartitionedOutput::shouldDrainPending() const {
  return pendingPartitionedBatch_.has_value() || shouldFlushPending() ||
      (noMoreInput_ && !pendingInputs_.empty());
}

bool UcxPartitionedOutput::ensureOutputReservation(ContinueFuture* future) {
  if (outputReservationBytes_ > 0) {
    return true;
  }

  auto queueManager = sharedQueueManager();
  const auto reservationBytes = std::max<int64_t>(pendingBytes_, 1);
  if (queueManager->reserveOutputBytes(
          this->taskId(), reservationBytes, future)) {
    return false;
  }
  outputReservationBytes_ = reservationBytes;
  return true;
}

void UcxPartitionedOutput::releaseOutputReservation() {
  if (outputReservationBytes_ == 0) {
    return;
  }
  const auto reservationBytes = outputReservationBytes_;
  outputReservationBytes_ = 0;
  sharedQueueManager()->releaseOutputReservation(
      this->taskId(), reservationBytes);
}

void UcxPartitionedOutput::initPartitionKeys(
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode) {
  // Following Logic copied direcly from CudLocalPartition (!)

  // Following is IMO a hacky way to get the partition key indices. It is to
  // workaround the fact that the partition spec constructs the hash function
  // directly and has no public methods to get the partition key indices.

  // When the operator is of type kRepartition, the partition spec is a string
  // in the format "HASH(key1, key2, ...)"
  // We're going to extract the keys between HASH( and ) and find their indices
  // in the output row type.

  // When operator is of type kGather, we don't need to store any partition key
  // indices because we're going to merge all the incoming streams together.

  // Get partition function specification string
  spec_ = planNode->partitionFunctionSpec().toString();

  // Only parse keys if it's a hash function
  if (spec_.find("HASH(") != std::string::npos) {
    // Extract keys between HASH( and )
    size_t start = spec_.find("HASH(") + 5;
    size_t end = spec_.find(")", start);
    if (start != std::string::npos && end != std::string::npos) {
      std::string keysStr = spec_.substr(start, end - start);

      // Split by comma to get individual keys.
      std::vector<std::string> keys;
      size_t pos = 0;
      while ((pos = keysStr.find(",")) != std::string::npos) {
        std::string key = keysStr.substr(0, pos);
        keys.push_back(key);
        keysStr.erase(0, pos + 1);
      }
      keys.push_back(keysStr); // Add the last key.

      // Find field indices for each key.
      const auto& rowType = planNode->outputType();
      for (const auto& key : keys) {
        auto trimmedKey = key;
        // Trim whitespace
        trimmedKey.erase(0, trimmedKey.find_first_not_of(" "));
        trimmedKey.erase(trimmedKey.find_last_not_of(" ") + 1);

        auto fieldIndex = rowType->getChildIdx(trimmedKey);
        partitionKeyIndices_.push_back(fieldIndex);
      }
    }
  }
}

void UcxPartitionedOutput::hashPartition(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " Hashing and partitioning into " << numPartitions_ << " chunks";

  // Use cudf hash partitioning
  std::vector<cudf::size_type> partitionKeyIndices;
  for (const auto& idx : partitionKeyIndices_) {
    partitionKeyIndices.push_back(static_cast<cudf::size_type>(idx));
  }

  auto [partitionedTable, partitionOffsets] = cudf::hash_partition(
      tableView,
      partitionKeyIndices,
      numPartitions_,
      cudf::hash_id::HASH_MURMUR3,
      cudf::DEFAULT_HASH_SEED,
      stream);

  VELOX_CHECK_EQ(partitionOffsets.size(), numPartitions_ + 1);
  VELOX_CHECK_EQ(partitionOffsets[0], 0);

  auto partitionedView = partitionedTable->view();
  pendingPartitionedBatch_.emplace(PendingPartitionedBatch{
      std::move(partitionedTable),
      {},
      partitionedView,
      std::move(partitionOffsets),
      outputReservationBytes_,
      stream,
      0});
}

void UcxPartitionedOutput::equalPartition(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream,
    std::unique_ptr<cudf::table> tableOwner,
    std::vector<CudfVectorPtr> vectorOwners) {
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " Splitting into " << numPartitions_ << " chunks";
  std::vector<cudf::size_type> offsets;
  cudf::size_type size = tableView.num_rows();
  offsets.reserve(numPartitions_ + 1);
  offsets.push_back(0);
  for (int i = 1; i < numPartitions_; ++i) {
    cudf::size_type idx = size * i / numPartitions_;
    offsets.push_back(idx);
  }
  offsets.push_back(size);

  pendingPartitionedBatch_.emplace(PendingPartitionedBatch{
      std::move(tableOwner),
      std::move(vectorOwners),
      tableView,
      std::move(offsets),
      outputReservationBytes_,
      stream,
      0});
}

// Estimate a row chunk size that keeps each packed UCX payload under the given
// byte target. The caller increases the target geometrically, so large
// partitions do not become many tiny messages, while the first allocations stay
// small enough to avoid an immediate memory spike.
cudf::size_type UcxPartitionedOutput::rowsForPayloadTarget(
    const PendingPartitionedBatch& batch,
    cudf::size_type start,
    cudf::size_type end,
    uint64_t targetBytes) const {
  const auto partitionRows = end - start;
  if (partitionRows <= 0) {
    return 0;
  }

  const auto totalRows = batch.tableView.num_rows();
  if (totalRows <= 0 || batch.estimatedBytes <= 0 || targetBytes == 0) {
    return partitionRows;
  }

  const auto estimatedPartitionBytes =
      static_cast<long double>(batch.estimatedBytes) *
      static_cast<long double>(partitionRows) /
      static_cast<long double>(totalRows);
  if (estimatedPartitionBytes <= static_cast<long double>(targetBytes)) {
    return partitionRows;
  }

  const auto rows = static_cast<cudf::size_type>(
      static_cast<long double>(partitionRows) *
      static_cast<long double>(targetBytes) / estimatedPartitionBytes);
  return std::max<cudf::size_type>(rows, 1);
}

bool UcxPartitionedOutput::shouldSplitPayload(
    const PendingPartitionedBatch& batch,
    cudf::size_type start,
    cudf::size_type end) const {
  const auto partitionRows = end - start;
  const auto totalRows = batch.tableView.num_rows();
  if (partitionRows <= 0 || totalRows <= 0 || batch.estimatedBytes <= 0) {
    return false;
  }

  const auto estimatedPartitionBytes =
      static_cast<long double>(batch.estimatedBytes) *
      static_cast<long double>(partitionRows) /
      static_cast<long double>(totalRows);
  return estimatedPartitionBytes >
      static_cast<long double>(transferWindowBytes(batch));
}

uint64_t UcxPartitionedOutput::transferWindowBytes(
    const PendingPartitionedBatch& batch) const {
  if (batch.estimatedBytes <= 0 || numPartitions_ == 0) {
    return initialPayloadBytes_;
  }

  const auto averagePartitionBytes = divideCeil(
      static_cast<uint64_t>(batch.estimatedBytes),
      static_cast<uint64_t>(numPartitions_));
  return std::max<uint64_t>(initialPayloadBytes_, averagePartitionBytes);
}

bool UcxPartitionedOutput::drainPendingPartitionedBatch() {
  VELOX_CHECK(pendingPartitionedBatch_.has_value());
  VELOX_CHECK_GT(outputReservationBytes_, 0);

  auto queueManager = sharedQueueManager();
  auto& batch = pendingPartitionedBatch_.value();

  VELOX_CHECK_EQ(
      batch.offsets.size(), numPartitions_ + 1, "mismatch in numPartitions_");

  while (batch.nextPartition < numPartitions_) {
    const auto partition = batch.nextPartition;
    const auto start = batch.offsets[partition];
    const auto end = batch.offsets[partition + 1];
    VELOX_CHECK_LE(start, end);
    if (start == end) {
      ++batch.nextPartition;
      continue;
    }

    const auto transferWindow = transferWindowBytes(batch);
    if (batch.nextRow == 0) {
      batch.nextRow = start;
      batch.nextPayloadBytes =
          shouldSplitPayload(batch, start, end) ? initialPayloadBytes_
                                                : transferWindow;
    }

    while (batch.nextRow < end) {
      if (queueManager->checkTransferCapacity(
              this->taskId(), transferWindow, &future_)) {
        blockingReason_ = exec::BlockingReason::kWaitForConsumer;
        return false;
      }

      const auto chunkStart = batch.nextRow;
      const auto rowsPerChunk =
          rowsForPayloadTarget(batch, chunkStart, end, batch.nextPayloadBytes);
      VELOX_CHECK_GT(rowsPerChunk, 0);
      const auto chunkEnd =
          std::min<cudf::size_type>(end, chunkStart + rowsPerChunk);
      std::vector<cudf::size_type> sliceOffsets{chunkStart, chunkEnd};
      auto tableSlices = cudf::slice(batch.tableView, sliceOffsets);
      VELOX_CHECK_EQ(tableSlices.size(), 1);

      auto packedCols = cudf::pack(
          tableSlices[0],
          batch.stream,
          cudf::get_current_device_resource_ref());

      // UCXX/UCX is not stream-aware, so the packed payload must be complete
      // before exposing its raw device pointer to the exchange server.
      batch.stream.synchronize();

      auto packedColsPtr = std::make_unique<cudf::packed_columns>(
          std::move(packedCols.metadata), std::move(packedCols.gpu_data));

      queueManager->enqueue(
          this->taskId(),
          partition,
          std::move(packedColsPtr),
          chunkEnd - chunkStart);

      batch.nextRow = chunkEnd;
      batch.nextPayloadBytes = std::min<uint64_t>(
          transferWindow, multiplySaturated(batch.nextPayloadBytes, 2));

      if (queueManager->checkTransferCapacity(
              this->taskId(), transferWindow, &future_)) {
        blockingReason_ = exec::BlockingReason::kWaitForConsumer;
        return false;
      }
    }

    ++batch.nextPartition;
    batch.nextRow = 0;
    batch.nextPayloadBytes = 0;
  }

  pendingPartitionedBatch_.reset();
  pendingRows_ = 0;
  pendingBytes_ = 0;
  releaseOutputReservation();
  blockingReason_ = exec::BlockingReason::kNotBlocked;
  return true;
}

} // namespace facebook::velox::ucx_exchange
