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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfLocalPartition.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/RoundRobinPartitionFunction.h"
#include "velox/exec/Task.h"

#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>

#include <algorithm>

namespace facebook::velox::cudf_velox {

namespace {
template <class... Deriveds, class Base>
bool isAnyOf(const Base* p) {
  return ((dynamic_cast<const Deriveds*>(p) != nullptr) || ...);
}

std::vector<cudf::size_type> toCudfIndices(
    const std::vector<column_index_t>& indices) {
  std::vector<cudf::size_type> result;
  result.reserve(indices.size());
  for (const auto& index : indices) {
    result.push_back(static_cast<cudf::size_type>(index));
  }
  return result;
}

std::unique_ptr<cudf::column> createKeyNullMask(
    cudf::table_view tableView,
    const std::vector<cudf::size_type>& partitionKeyIndices,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(!partitionKeyIndices.empty());

  auto result =
      cudf::is_null(tableView.column(partitionKeyIndices[0]), stream, mr);
  for (size_t i = 1; i < partitionKeyIndices.size(); ++i) {
    auto keyIsNull =
        cudf::is_null(tableView.column(partitionKeyIndices[i]), stream, mr);
    result = cudf::binary_operation(
        result->view(),
        keyIsNull->view(),
        cudf::binary_operator::BITWISE_OR,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
  }
  return result;
}
} // namespace

bool CudfLocalPartition::shouldReplace(
    const std::shared_ptr<const core::LocalPartitionNode>& planNode) {
  if (planNode->isReplicateNulls() &&
      !isAnyOf<exec::HashPartitionFunctionSpec>(
          &planNode->partitionFunctionSpec())) {
    LOG(WARNING)
        << "CudfLocalPartition replicateNulls requires hash partitioning";
    return false;
  }

  // Only replace for Hash, Round Robin, and Round Robin Row-Wise Partitioning.
  if (isAnyOf<
          exec::HashPartitionFunctionSpec,
          exec::RoundRobinPartitionFunctionSpec,
          core::GatherPartitionFunctionSpec>(
          &planNode->partitionFunctionSpec())) {
    return true;
  }
  std::string spec = planNode->partitionFunctionSpec().toString();
  if (spec.find("ROUND ROBIN ROW") != std::string::npos) {
    return true;
  }
  LOG(WARNING) << "CudfLocalPartition unsupported spec = " << spec;
  return false;
}

CudfLocalPartition::CudfLocalPartition(
    int32_t operatorId,
    exec::DriverCtx* ctx,
    const std::shared_ptr<const core::LocalPartitionNode>& planNode)
    : CudfOperatorBase(
          operatorId,
          ctx,
          planNode->outputType(),
          planNode->id(),
          "CudfLocalPartition",
          nvtx3::rgb{255, 215, 0}, // Gold
          NvtxMethodFlag::kAll,
          std::nullopt,
          planNode),
      queues_{
          ctx->task->getLocalExchangeQueues(ctx->splitGroupId, planNode->id())},
      numPartitions_{queues_.size()},
      replicateNulls_{planNode->isReplicateNulls()} {
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
  std::string spec = planNode->partitionFunctionSpec().toString();
  auto* hashFunctionSpec = dynamic_cast<const exec::HashPartitionFunctionSpec*>(
      &planNode->partitionFunctionSpec());

  // Only parse keys if it's a hash function
  if (hashFunctionSpec) {
    // Extract keys between HASH( and )
    size_t start = spec.find("HASH(") + 5;
    size_t end = spec.find(")", start);
    if (start != std::string::npos && end != std::string::npos) {
      std::string keysStr = spec.substr(start, end - start);

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
    partitionFunctionType_ = PartitionFunctionType::kHash;
    if (replicateNulls_) {
      VELOX_USER_CHECK_EQ(
          partitionKeyIndices_.size(),
          1,
          "CudfLocalPartition replicateNulls requires exactly one partition key");
    }
  } else if (
      dynamic_cast<const exec::RoundRobinPartitionFunctionSpec*>(
          &planNode->partitionFunctionSpec()) &&
      numPartitions_ > 1) {
    partitionFunctionType_ = PartitionFunctionType::kRoundRobin;
  } else if (
      numPartitions_ > 1 && spec.find("ROUND ROBIN ROW") != std::string::npos) {
    partitionFunctionType_ = PartitionFunctionType::kRoundRobinRow;
  }
  VELOX_CHECK(
      numPartitions_ == 1 || partitionKeyIndices_.size() > 0 ||
      partitionFunctionType_ == PartitionFunctionType::kRoundRobin ||
      partitionFunctionType_ == PartitionFunctionType::kRoundRobinRow);

  // Since we're replacing the LocalPartition with CudfLocalPartition, the
  // number of producers is already set. Adding producer only adds to a counter
  // which we don't have to do again.
  // Normally, this is what we'd have to do:
  // for (auto& queue : queues_) {
  //   queue->addProducer();
  // }
}

void CudfLocalPartition::recordOutputStats(RowVectorPtr& input) {
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addOutputVector(input->estimateFlatSize(), input->size());
  }
}

void CudfLocalPartition::flushVectorPool() {
  // We reuse the LocalExchangeQueue from the CPU implementation. That impl
  // stores used vectors in a vector pool for the CPU LocalPartition to re-use.
  // CudfLocalPartition does not need it and does not extract it. This results
  // in unnecessary extension of the lifetimes of vectors that were exchanged,
  // resulting in kind of a memory leak.
  // This is a hack to forcefully flush the vector pools.

  for (auto& queue : queues_) {
    queue->getVector();
  }
}

void CudfLocalPartition::enqueuePartition(
    int partitionIndex,
    const CudfVectorPtr& cudfVector) {
  if (cudfVector->size() == 0) {
    // Skip empty partitions.
    return;
  }

  ContinueFuture future;
  auto blockingReason =
      queues_[partitionIndex]->enqueue(cudfVector, cudfVector->size(), &future);
  if (blockingReason != exec::BlockingReason::kNotBlocked) {
    blockingReasons_.push_back(blockingReason);
    futures_.push_back(std::move(future));
  }
}

void CudfLocalPartition::doAddInput(RowVectorPtr input) {
  flushVectorPool();
  recordOutputStats(input);
  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK(cudfVector, "Input must be a CudfVector");
  auto stream = cudfVector->stream();

  if (numPartitions_ > 1) {
    if (partitionFunctionType_ == PartitionFunctionType::kRoundRobin) {
      const auto partition = counter_ % numPartitions_;
      ++counter_;
      enqueuePartition(partition, cudfVector);
      return;
    }
    std::shared_ptr<cudf::table> replicatedNullRowsOwner;
    std::vector<cudf::size_type> nativeNullPartitionOffsets;

    auto [partitionedTable, partitionOffsets] = [&]() {
      auto tableView = cudfVector->getTableView();
      // Use cudf hash partitioning
      if (partitionFunctionType_ == PartitionFunctionType::kHash) {
        auto partitionKeyIndices = toCudfIndices(partitionKeyIndices_);

        if (replicateNulls_) {
          auto nullMask = createKeyNullMask(
              tableView, partitionKeyIndices, stream, get_temp_mr());
          auto replicatedNullRows = cudf::apply_boolean_mask(
              tableView, nullMask->view(), stream, get_temp_mr());
          if (replicatedNullRows->num_rows() > 0) {
            auto nullPartitionResult = cudf::hash_partition(
                replicatedNullRows->view(),
                partitionKeyIndices,
                numPartitions_,
                cudf::hash_id::HASH_MURMUR3,
                cudf::DEFAULT_HASH_SEED,
                stream,
                get_temp_mr());
            nativeNullPartitionOffsets = std::move(nullPartitionResult.second);
            replicatedNullRowsOwner =
                std::shared_ptr<cudf::table>(std::move(replicatedNullRows));
          }
        }

        return cudf::hash_partition(
            tableView,
            partitionKeyIndices,
            numPartitions_,
            cudf::hash_id::HASH_MURMUR3,
            cudf::DEFAULT_HASH_SEED,
            stream,
            get_temp_mr());
      } else if (
          partitionFunctionType_ == PartitionFunctionType::kRoundRobinRow) {
        auto result = cudf::round_robin_partition(
            tableView, numPartitions_, counter_, stream, get_temp_mr());
        counter_ = (counter_ + cudfVector->size()) % numPartitions_;
        return result;
      }
      VELOX_FAIL("Unsupported partition function");
    }();

    // cuDF partitioning APIs return num_partitions + 1 offsets where:
    // - offsets[i] is the starting row index for partition i
    // - offsets[num_partitions] is the total row count
    VELOX_CHECK(partitionOffsets.size() == numPartitions_ + 1);
    VELOX_CHECK(partitionOffsets[0] == 0);

    // cudf::split expects split points (excluding first 0 and last totalRows).
    // Erase first element (always 0) and last element (total row count).
    partitionOffsets.erase(partitionOffsets.begin());
    partitionOffsets.pop_back();

    auto partitionedTables =
        cudf::split(partitionedTable->view(), partitionOffsets, stream);

    auto partitionedTableOwner =
        std::shared_ptr<cudf::table>(std::move(partitionedTable));
    const auto inputBytes = cudfVector->estimateFlatSize();
    const auto inputRows = std::max<vector_size_t>(cudfVector->size(), 1);
    for (int i = 0; i < numPartitions_; ++i) {
      auto partitionData = partitionedTables[i];
      if (partitionData.num_rows() == 0) {
        continue;
      }

      auto partitionBytes = inputBytes *
          static_cast<uint64_t>(partitionData.num_rows()) /
          static_cast<uint64_t>(inputRows);
      if (partitionBytes == 0) {
        partitionBytes = 1;
      }
      auto partitionCudfVector = std::make_shared<CudfVector>(
          pool(),
          outputType_,
          partitionData.num_rows(),
          partitionData,
          partitionedTableOwner,
          stream,
          partitionBytes);
      enqueuePartition(i, partitionCudfVector);
    }

    if (replicatedNullRowsOwner) {
      VELOX_CHECK_EQ(nativeNullPartitionOffsets.size(), numPartitions_ + 1);
      const auto nullRowsView = replicatedNullRowsOwner->view();
      const auto nullRowsBytes = std::max<uint64_t>(
          1,
          inputBytes * static_cast<uint64_t>(nullRowsView.num_rows()) /
              static_cast<uint64_t>(inputRows));
      for (int i = 0; i < numPartitions_; ++i) {
        if (nativeNullPartitionOffsets[i] < nativeNullPartitionOffsets[i + 1]) {
          continue;
        }
        auto nullRowsCudfVector = std::make_shared<CudfVector>(
            pool(),
            outputType_,
            nullRowsView.num_rows(),
            nullRowsView,
            replicatedNullRowsOwner,
            stream,
            nullRowsBytes);
        enqueuePartition(i, nullRowsCudfVector);
      }
    }
  } else {
    // Single partition case.
    ContinueFuture future;
    auto blockingReason =
        queues_[0]->enqueue(input, input->retainedSize(), &future);
    if (blockingReason != exec::BlockingReason::kNotBlocked) {
      blockingReasons_.push_back(blockingReason);
      futures_.push_back(std::move(future));
    }
  }
}

exec::BlockingReason CudfLocalPartition::isBlocked(ContinueFuture* future) {
  if (!futures_.empty()) {
    auto blockingReason = blockingReasons_.front();
    *future = folly::collectAll(futures_.begin(), futures_.end()).unit();
    futures_.clear();
    blockingReasons_.clear();
    return blockingReason;
  }

  return exec::BlockingReason::kNotBlocked;
}

void CudfLocalPartition::doNoMoreInput() {
  Operator::noMoreInput();
  for (const auto& queue : queues_) {
    queue->noMoreData();
  }
}

bool CudfLocalPartition::isFinished() {
  if (!futures_.empty() || !noMoreInput_) {
    return false;
  }
  flushVectorPool();

  return true;
}

} // namespace facebook::velox::cudf_velox
