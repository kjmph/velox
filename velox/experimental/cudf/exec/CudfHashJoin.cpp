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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/AstExpressionUtils.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/common/testutil/TestValue.h"
#include "velox/core/Expressions.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Task.h" // NOLINT(misc-unused-headers)
#include "velox/type/TypeUtil.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/mark_join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/reshape.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/search.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <limits>

namespace facebook::velox::cudf_velox {

namespace {

constexpr size_t kDefaultMinMaxSummaryBatchRows = 100'000'000;
constexpr uint64_t kMinHashBuildBatchTargetBytes = 128ULL << 20;
constexpr uint64_t kMaxHashBuildBatchTargetBytes = 1ULL << 30;
constexpr uint64_t kMinDeferredHashHydrationBytes = 1ULL << 30;
constexpr uint64_t kMaxDeferredHashHydrationBytes = 8ULL << 30;
constexpr uint64_t kHashBuildWorkspaceBytesPerRow =
    2 * (sizeof(cudf::hash_value_type) + sizeof(cudf::size_type));

uint64_t saturatedAdd(uint64_t lhs, uint64_t rhs) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
    return std::numeric_limits<uint64_t>::max();
  }
  return lhs + rhs;
}

uint64_t saturatedMultiply(uint64_t lhs, uint64_t rhs) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    return std::numeric_limits<uint64_t>::max();
  }
  return lhs * rhs;
}

uint64_t ceilingDivide(uint64_t dividend, uint64_t divisor) {
  VELOX_CHECK_GT(divisor, 0);
  return dividend / divisor + (dividend % divisor != 0);
}

int64_t saturateRuntimeCounter(uint64_t value) {
  return static_cast<int64_t>(std::min<uint64_t>(
      value, static_cast<uint64_t>(std::numeric_limits<int64_t>::max())));
}

uint64_t hashBuildBatchTargetBytes() {
  uint64_t targetBytes = kMaxHashBuildBatchTargetBytes;
  if (auto info = currentDeviceMemoryInfo(); info.has_value()) {
    targetBytes = info->totalBytes / 64;
  }
  targetBytes = std::clamp(
      targetBytes,
      kMinHashBuildBatchTargetBytes,
      kMaxHashBuildBatchTargetBytes);
  common::testutil::TestValue::adjust(
      "facebook::velox::cudf_velox::CudfHashJoinBuild::doNoMoreInput::materializationTargetBytes",
      &targetBytes);
  VELOX_CHECK_GT(
      targetBytes, 0, "cuDF hash-build byte target must be positive");
  return targetBytes;
}

uint64_t hashBuildBatchWorkBytes(uint64_t rows, uint64_t flatBytes) {
  // During materialization, the source and concatenated copy overlap. Once
  // the source is released, the copy overlaps the hash table, whose 0.5 load
  // factor requires about two hash/index slots per row.
  return std::max(
      saturatedMultiply(flatBytes, 2),
      saturatedAdd(
          flatBytes, saturatedMultiply(rows, kHashBuildWorkspaceBytesPerRow)));
}

uint64_t hashBuildBatchMaxRows() {
  uint64_t maxRows =
      static_cast<uint64_t>(std::numeric_limits<cudf::size_type>::max());
  const auto& config = CudfConfig::getInstance();
  if (config.batchSizeMaxThreshold.has_value()) {
    VELOX_CHECK_GT(
        config.batchSizeMaxThreshold.value(),
        0,
        "cuDF max batch size must be positive");
    maxRows = std::min<uint64_t>(
        maxRows, static_cast<uint64_t>(config.batchSizeMaxThreshold.value()));
  }
  return maxRows;
}

uint64_t deferredHashHydrationThresholdBytes() {
  uint64_t thresholdBytes = kMaxDeferredHashHydrationBytes;
  if (auto info = currentDeviceMemoryInfo(); info.has_value()) {
    // A build larger than one eighth of the device is large enough to starve
    // independent branches that execute before its probe input is ready. Cap
    // the threshold so large-memory GPUs do not retain arbitrarily large idle
    // builds, while keeping the resident fast path for ordinary joins.
    thresholdBytes = info->totalBytes / 8;
  }
  thresholdBytes = std::clamp(
      thresholdBytes,
      kMinDeferredHashHydrationBytes,
      kMaxDeferredHashHydrationBytes);
  common::testutil::TestValue::adjust(
      "facebook::velox::cudf_velox::CudfHashJoinBuild::doNoMoreInput::deferredHydrationThresholdBytes",
      &thresholdBytes);
  VELOX_CHECK_GT(
      thresholdBytes,
      0,
      "cuDF deferred hash hydration threshold must be positive");
  return thresholdBytes;
}

/// Creates extended table view by appending precomputed columns
cudf::table_view createExtendedTableView(
    cudf::table_view originalView,
    std::vector<ColumnOrView>& precomputedColumns) {
  if (precomputedColumns.empty()) {
    return originalView;
  }

  std::vector<cudf::column_view> allViews;
  allViews.reserve(originalView.num_columns() + precomputedColumns.size());

  for (cudf::size_type i = 0; i < originalView.num_columns(); ++i) {
    allViews.push_back(originalView.column(i));
  }
  for (auto& col : precomputedColumns) {
    allViews.push_back(asView(col));
  }

  return cudf::table_view(allViews);
}

vector_size_t filteredOutputNumRows(
    bool zeroColumnOutput,
    cudf::column_view filterColumn,
    const std::vector<std::unique_ptr<cudf::column>>& joinedCols,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref tempMr) {
  if (!zeroColumnOutput) {
    return joinedCols.empty() ? 0 : joinedCols[0]->size();
  }

  auto trueCountScalar = cudf::reduce(
      filterColumn,
      *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{cudf::type_id::INT32},
      stream,
      tempMr);
  return static_cast<vector_size_t>(
      static_cast<cudf::numeric_scalar<int32_t>*>(trueCountScalar.get())
          ->value(stream));
}

// Returns row indices where mask is true as a column of size_type.
std::unique_ptr<cudf::column> getRetainedIndices(
    cudf::column_view mask,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto seq = cudf::sequence(
      mask.size(),
      cudf::numeric_scalar<cudf::size_type>(0, true, stream, mr),
      cudf::numeric_scalar<cudf::size_type>(1, true, stream, mr),
      stream,
      mr);

  auto indicesTable = cudf::apply_boolean_mask(
      cudf::table_view{{seq->view()}}, mask, stream, mr);

  return std::move(indicesTable->release().front());
}

// Tracks which probe rows have been matched across multiple build batches.
// Maintains a boolean column that accumulates matches via cudf::contains +
// BITWISE_OR, and provides a method to retrieve unmatched probe row indices.
class ProbeMatchTracker {
 public:
  ProbeMatchTracker(
      cudf::size_type numProbeRows,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    auto falseScalar = cudf::numeric_scalar<bool>(false, true, stream, mr);
    matchCol_ =
        cudf::make_column_from_scalar(falseScalar, numProbeRows, stream, mr);
    probeRowIndices_ = cudf::sequence(
        numProbeRows,
        cudf::numeric_scalar<cudf::size_type>(0, true, stream, mr),
        cudf::numeric_scalar<cudf::size_type>(1, true, stream, mr),
        stream,
        mr);
  }

  // Mark probe rows present in matchedLeftIndices as matched.
  void update(
      cudf::column_view matchedLeftIndices,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    auto matchedInBatch = cudf::contains(
        matchedLeftIndices, probeRowIndices_->view(), stream, mr);
    auto updatedMatch = cudf::binary_operation(
        matchCol_->view(),
        matchedInBatch->view(),
        cudf::binary_operator::BITWISE_OR,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    stream.synchronize();
    matchCol_ = std::move(updatedMatch);
  }

  // Returns indices of probe rows that were never matched.
  std::unique_ptr<cudf::column> getUnmatchedIndices(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    auto unmatchedMask = cudf::unary_operation(
        matchCol_->view(), cudf::unary_operator::NOT, stream, mr);
    return getRetainedIndices(unmatchedMask->view(), stream, mr);
  }

 private:
  std::unique_ptr<cudf::column> matchCol_;
  std::unique_ptr<cudf::column> probeRowIndices_;
};

bool usesPrebuiltCudfHashJoin(const core::HashJoinNode& joinNode) {
  return joinNode.isInnerJoin() || joinNode.isLeftJoin() ||
      joinNode.isRightJoin() || joinNode.isFullJoin() ||
      (joinNode.isLeftSemiProjectJoin() && joinNode.filter());
}

bool supportsByteBoundedHashBuild(const core::HashJoinNode& joinNode) {
  // Inner and left joins already combine matches across multiple build
  // tables. Other join kinds still contain single-build assumptions or drain
  // unmatched build rows through one final concatenation. Preserve their
  // existing row-limit batching until those probe paths support routine
  // byte-driven fanout.
  return joinNode.isInnerJoin() || joinNode.isLeftJoin();
}

bool useProbeUniqueInnerJoin(const core::HashJoinNode& joinNode) {
  return CudfConfig::getInstance().probeUniqueInnerJoinEnabled &&
      CudfConfig::getInstance().distinctHashJoinEnabled &&
      joinNode.isInnerJoin() && !joinNode.filter() &&
      joinNode.leftKeysUnique() && joinNode.leftKeysNonNull() &&
      !joinNode.rightKeysUnique();
}

bool supportsDeferredHashHydration(const core::HashJoinNode& joinNode) {
  // This first deferred-build lifecycle is intentionally narrow. Trusted
  // unique and non-null build keys use a shared collection of distinct hashes
  // after hydration, and an unfiltered INNER join can stream independent
  // build-shard outputs without cross-shard match state. Other join kinds and
  // filters retain the existing resident path.
  return joinNode.isInnerJoin() && !joinNode.filter() &&
      joinNode.rightKeysUnique() && joinNode.rightKeysNonNull() &&
      joinNode.leftKeysNonNull() &&
      CudfConfig::getInstance().distinctHashJoinEnabled &&
      !useProbeUniqueInnerJoin(joinNode);
}

cudf::size_type countNullJoinKeys(
    cudf::table_view tableView,
    const std::vector<cudf::size_type>& keyIndices,
    rmm::cuda_stream_view stream) {
  if (keyIndices.empty()) {
    return 0;
  }
  auto [_, nullCount] =
      cudf::bitmask_and(tableView.select(keyIndices), stream, get_temp_mr());
  return nullCount;
}

std::optional<cudf::size_type> fieldIndexOnSide(
    const core::TypedExprPtr& expr,
    const RowTypePtr& rowType) {
  const auto fieldExpr =
      std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr);
  if (fieldExpr == nullptr) {
    return std::nullopt;
  }

  const auto& fieldName = fieldExpr->name();
  if (!rowType->containsChild(fieldName)) {
    return std::nullopt;
  }
  return static_cast<cudf::size_type>(rowType->getChildIdx(fieldName));
}

bool isNotEqualCallName(const std::string& callName) {
  auto name =
      stripPrefix(callName, CudfConfig::getInstance().functionNamePrefix);
  name = stripPrefix(name, "presto.default.");
  return name == "neq" || name == "$operator$not_equal";
}

std::optional<std::pair<cudf::size_type, cudf::size_type>>
detectSimpleCrossSideNotEqual(
    const core::TypedExprPtr& expr,
    const RowTypePtr& leftType,
    const RowTypePtr& rightType) {
  const auto callExpr =
      std::dynamic_pointer_cast<const core::CallTypedExpr>(expr);
  if (callExpr == nullptr) {
    return std::nullopt;
  }

  if (!isNotEqualCallName(callExpr->name())) {
    return std::nullopt;
  }
  if (callExpr->inputs().size() != 2) {
    return std::nullopt;
  }

  auto left0 = fieldIndexOnSide(callExpr->inputs()[0], leftType);
  auto right0 = fieldIndexOnSide(callExpr->inputs()[0], rightType);
  auto left1 = fieldIndexOnSide(callExpr->inputs()[1], leftType);
  auto right1 = fieldIndexOnSide(callExpr->inputs()[1], rightType);

  if (left0.has_value() && right1.has_value() && !right0.has_value() &&
      !left1.has_value()) {
    return std::make_pair(left0.value(), right1.value());
  }
  if (right0.has_value() && left1.has_value() && !left0.has_value() &&
      !right1.has_value()) {
    return std::make_pair(left1.value(), right0.value());
  }
  return std::nullopt;
}

std::vector<cudf::size_type> sequenceIndices(cudf::size_type size) {
  std::vector<cudf::size_type> indices;
  indices.reserve(size);
  for (cudf::size_type i = 0; i < size; ++i) {
    indices.push_back(i);
  }
  return indices;
}

RowTypePtr makeMinMaxSummaryType(
    const RowTypePtr& buildType,
    const std::vector<cudf::size_type>& keyIndices,
    cudf::size_type valueIndex) {
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  names.reserve(keyIndices.size() + 2);
  types.reserve(keyIndices.size() + 2);

  for (auto keyIndex : keyIndices) {
    names.push_back(buildType->nameOf(keyIndex));
    types.push_back(buildType->childAt(keyIndex));
  }
  names.push_back(buildType->nameOf(valueIndex) + "_min");
  types.push_back(buildType->childAt(valueIndex));
  names.push_back(buildType->nameOf(valueIndex) + "_max");
  types.push_back(buildType->childAt(valueIndex));

  return ROW(std::move(names), std::move(types));
}

std::unique_ptr<cudf::table> makeEmptyMinMaxSummaryTable(
    cudf::table_view inputView,
    const std::vector<cudf::size_type>& keyIndices,
    cudf::size_type valueIndex) {
  std::vector<std::unique_ptr<cudf::column>> emptyColumns;
  emptyColumns.reserve(keyIndices.size() + 2);
  for (auto keyIndex : keyIndices) {
    emptyColumns.push_back(cudf::empty_like(inputView.column(keyIndex)));
  }
  emptyColumns.push_back(cudf::empty_like(inputView.column(valueIndex)));
  emptyColumns.push_back(cudf::empty_like(inputView.column(valueIndex)));
  return std::make_unique<cudf::table>(std::move(emptyColumns));
}

std::unique_ptr<cudf::table> buildMinMaxSummaryTable(
    cudf::table_view inputView,
    const std::vector<cudf::size_type>& keyIndices,
    cudf::size_type valueIndex,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (inputView.num_rows() == 0) {
    return makeEmptyMinMaxSummaryTable(inputView, keyIndices, valueIndex);
  }

  cudf::groupby::groupby groupByOwner(
      inputView.select(keyIndices), cudf::null_policy::EXCLUDE);
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = inputView.column(valueIndex);
  requests[0].aggregations.push_back(
      cudf::make_min_aggregation<cudf::groupby_aggregation>());
  requests[0].aggregations.push_back(
      cudf::make_max_aggregation<cudf::groupby_aggregation>());

  auto [summaryKeys, results] = groupByOwner.aggregate(requests, stream, mr);
  VELOX_CHECK_EQ(results.size(), 1);
  VELOX_CHECK_EQ(results[0].results.size(), 2);

  auto summaryColumns = summaryKeys->release();
  summaryColumns.push_back(std::move(results[0].results[0]));
  summaryColumns.push_back(std::move(results[0].results[1]));
  return std::make_unique<cudf::table>(std::move(summaryColumns));
}

std::unique_ptr<cudf::table> compactMinMaxSummaryTable(
    cudf::table_view inputView,
    cudf::size_type numKeyColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const keyIndices = sequenceIndices(numKeyColumns);
  if (inputView.num_rows() == 0) {
    return makeEmptyMinMaxSummaryTable(inputView, keyIndices, numKeyColumns);
  }

  cudf::groupby::groupby groupByOwner(
      inputView.select(keyIndices), cudf::null_policy::EXCLUDE);
  std::vector<cudf::groupby::aggregation_request> requests(2);
  requests[0].values = inputView.column(numKeyColumns);
  requests[0].aggregations.push_back(
      cudf::make_min_aggregation<cudf::groupby_aggregation>());
  requests[1].values = inputView.column(numKeyColumns + 1);
  requests[1].aggregations.push_back(
      cudf::make_max_aggregation<cudf::groupby_aggregation>());

  auto [summaryKeys, results] = groupByOwner.aggregate(requests, stream, mr);
  VELOX_CHECK_EQ(results.size(), 2);
  VELOX_CHECK_EQ(results[0].results.size(), 1);
  VELOX_CHECK_EQ(results[1].results.size(), 1);

  auto summaryColumns = summaryKeys->release();
  summaryColumns.push_back(std::move(results[0].results[0]));
  summaryColumns.push_back(std::move(results[1].results[0]));
  return std::make_unique<cudf::table>(std::move(summaryColumns));
}

size_t minMaxSummaryBatchRows() {
  const auto& cudfConfig = CudfConfig::getInstance();
  if (cudfConfig.batchSizeMaxThreshold.has_value()) {
    return static_cast<size_t>(cudfConfig.batchSizeMaxThreshold.value());
  }
  return kDefaultMinMaxSummaryBatchRows;
}

std::vector<std::unique_ptr<cudf::table>> compactMinMaxSummaryTables(
    std::vector<std::unique_ptr<cudf::table>> inputTables,
    cudf::size_type numKeyColumns,
    rmm::cuda_stream_view stream,
    int64_t& summaryRows) {
  std::vector<std::unique_ptr<cudf::table>> summaries;
  summaries.reserve(inputTables.size());
  summaryRows = 0;

  for (auto& inputTable : inputTables) {
    VELOX_CHECK_NOT_NULL(inputTable);
    auto summary = compactMinMaxSummaryTable(
        inputTable->view(), numKeyColumns, stream, get_output_mr());
    summaryRows += static_cast<int64_t>(summary->num_rows());
    summaries.push_back(std::move(summary));
    inputTable.reset();
  }

  return summaries;
}

std::vector<std::unique_ptr<cudf::table>> compactMinMaxSummaryVectorsBatched(
    std::vector<CudfVectorPtr> inputTables,
    const TypePtr& tableType,
    cudf::size_type numKeyColumns,
    rmm::cuda_stream_view stream,
    int64_t& summaryRows) {
  summaryRows = 0;
  if (inputTables.empty()) {
    auto emptyTables = getConcatenatedTableBatched(
        std::move(inputTables), tableType, stream, get_output_mr());
    return compactMinMaxSummaryTables(
        std::move(emptyTables), numKeyColumns, stream, summaryRows);
  }

  std::vector<std::unique_ptr<cudf::table>> summaries;
  const auto targetRows = minMaxSummaryBatchRows();
  size_t start = 0;
  size_t runningRows = 0;

  auto flush = [&](size_t end) {
    std::vector<CudfVectorPtr> batch;
    batch.reserve(end - start);
    for (auto i = start; i < end; ++i) {
      batch.push_back(std::move(inputTables[i]));
    }
    auto concatenated = getConcatenatedTableBatched(
        std::move(batch), tableType, stream, get_output_mr());
    int64_t batchSummaryRows = 0;
    auto batchSummaries = compactMinMaxSummaryTables(
        std::move(concatenated), numKeyColumns, stream, batchSummaryRows);
    summaryRows += batchSummaryRows;
    summaries.insert(
        summaries.end(),
        std::make_move_iterator(batchSummaries.begin()),
        std::make_move_iterator(batchSummaries.end()));
    start = end;
    runningRows = 0;
  };

  for (size_t i = 0; i < inputTables.size(); ++i) {
    VELOX_CHECK_NOT_NULL(inputTables[i]);
    const auto rows =
        static_cast<size_t>(inputTables[i]->getTableView().num_rows());
    if (runningRows > 0 && runningRows + rows > targetRows) {
      flush(i);
    }
    runningRows += rows;
  }
  if (start < inputTables.size()) {
    flush(inputTables.size());
  }

  return summaries;
}

} // namespace

void CudfHashJoinProbe::doClose() {
  Operator::close();
  input_.reset();
  inputs_.clear();
  hashObject_.reset();
  parkedHashObject_.reset();
  deferredHydration_ = false;
  deferredProbeBarrierEntered_ = false;
  nextBuildTableIndex_ = 0;
  rightMatchedFlags_.clear();
  cachedRightPrecomputed_.clear();
  cachedExtendedRightViews_.clear();
  buildReadyEvent_.reset();
  buildStream_.reset();
  filterEvaluator_.reset();
  simpleNotEqualLeftIndex_.reset();
  simpleNotEqualRightIndex_.reset();
  scalars_.clear();
  tree_ = {};
}

void CudfHashJoinBuild::doClose() {
  Operator::close();
  inputs_.clear();
  minMaxSummaryInputs_.clear();
}

void CudfHashJoinBridge::setHashTable(
    std::optional<CudfHashJoinBridge::hash_type> hashObject) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::setHashTable";
  }
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(started_);
    VELOX_CHECK(
        !hashObject_.has_value() && !parkedHashObject_.has_value() &&
            !probeFinished_,
        "CudfHashJoinBridge already has a hash table");
    hashObject_ = std::move(hashObject);
    promises = std::move(promises_);
  }
  notify(std::move(promises));
}

void CudfHashJoinBridge::setParkedHashTable(
    CudfHashJoinBridge::parked_hash_type hashObject) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(started_);
    VELOX_CHECK(
        !hashObject_.has_value() && !parkedHashObject_.has_value() &&
            !probeFinished_,
        "CudfHashJoinBridge already has build state");
    VELOX_CHECK(!hashObject.buildTables.empty());
    parkedHashObject_ = std::move(hashObject);
    promises = std::move(promises_);
  }
  notify(std::move(promises));
}

void CudfHashJoinBridge::setHydratedHashTable(
    CudfHashJoinBridge::hash_type hashObject) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(started_);
    VELOX_CHECK(hydrationStarted_);
    VELOX_CHECK(parkedHashObject_.has_value());
    VELOX_CHECK(!hashObject_.has_value());
    VELOX_CHECK(!probeFinished_);
    hashObject_ = std::move(hashObject);
    parkedHashObject_.reset();
    promises = std::move(promises_);
  }
  notify(std::move(promises));
}

void CudfHashJoinBridge::setHydrationError(std::exception_ptr error) {
  VELOX_CHECK(error != nullptr);
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(started_);
    VELOX_CHECK(hydrationStarted_);
    hydrationError_ = std::move(error);
    parkedHashObject_.reset();
    promises = std::move(promises_);
  }
  notify(std::move(promises));
}

CudfHashJoinBridge::HashOrFutureResult CudfHashJoinBridge::hashOrFuture(
    ContinueFuture* future,
    bool claimHydration) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::hashOrFuture";
  }
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK(started_);
  VELOX_CHECK(!cancelled_, "Getting hash table after join is aborted");
  VELOX_CHECK(!probeFinished_, "Getting hash table after probe finished");
  if (hydrationError_ != nullptr) {
    std::rethrow_exception(hydrationError_);
  }
  if (hashObject_.has_value()) {
    return {.hashObject = hashObject_};
  }
  if (parkedHashObject_.has_value() && !hydrationStarted_) {
    if (!claimHydration) {
      return {.buildParked = true};
    }
    hydrationStarted_ = true;
    return {
        .parkedHashObject = parkedHashObject_,
        .buildParked = true,
    };
  }
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::hashOrFuture constructing promise";
  }
  promises_.emplace_back("CudfHashJoinBridge::hashOrFuture");
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::hashOrFuture getSemiFuture";
  }
  *future = promises_.back().getSemiFuture();
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::hashOrFuture returning future";
  }
  return {};
}

void CudfHashJoinBridge::probeFinished() {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK(started_);
  VELOX_CHECK(!probeFinished_);
  probeFinished_ = true;
  hashObject_.reset();
  parkedHashObject_.reset();
  buildReadyEvent_.reset();
  buildStream_.reset();
}

void CudfHashJoinBridge::setBuildStream(rmm::cuda_stream_view buildStream) {
  std::lock_guard<std::mutex> l(mutex_);
  buildStream_ = buildStream;
}

std::optional<rmm::cuda_stream_view> CudfHashJoinBridge::getBuildStream() {
  std::lock_guard<std::mutex> l(mutex_);
  return buildStream_;
}

void CudfHashJoinBridge::setBuildReadyEvent(
    std::shared_ptr<CudaEvent> buildReadyEvent) {
  std::lock_guard<std::mutex> l(mutex_);
  buildReadyEvent_ = std::move(buildReadyEvent);
}

std::shared_ptr<CudaEvent> CudfHashJoinBridge::getBuildReadyEvent() {
  std::lock_guard<std::mutex> l(mutex_);
  return buildReadyEvent_;
}

CudfHashJoinBuild::CudfHashJoinBuild(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode)
    // TODO check outputType should be set or not?
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          nullptr, // outputType
          joinNode->id(),
          "CudfHashJoinBuild",
          nvtx3::rgb{65, 105, 225}, // Royal Blue
          NvtxMethodFlag::kAll,
          std::nullopt, // spillConfig
          joinNode),
      joinNode_(joinNode) {
  auto const& rightKeys = joinNode_->rightKeys();
  auto const buildType = joinNode_->sources()[1]->outputType();
  buildKeyIndices_.resize(rightKeys.size());
  for (size_t i = 0; i < buildKeyIndices_.size(); ++i) {
    buildKeyIndices_[i] = static_cast<cudf::size_type>(
        buildType->getChildIdx(rightKeys[i]->name()));
  }
  if (joinNode_->filter() && joinNode_->isLeftSemiProjectJoin() &&
      buildKeyIndices_.size() == 1) {
    simpleNotEqual_ = detectSimpleCrossSideNotEqual(
        joinNode_->filter(),
        asRowType(joinNode_->sources()[0]->outputType()),
        asRowType(buildType));
    if (simpleNotEqual_.has_value()) {
      minMaxSummaryType_ = makeMinMaxSummaryType(
          asRowType(buildType), buildKeyIndices_, simpleNotEqual_->second);
    }
  }
}

void CudfHashJoinBuild::doAddInput(RowVectorPtr input) {
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput);
    if (!joinNode_->rightKeysNonNull()) {
      auto const null_count = countNullJoinKeys(
          cudfInput->getTableView(), buildKeyIndices_, cudfInput->stream());
      {
        // Update statistics for null keys in join operator.
        auto lockedStats = stats_.wlock();
        lockedStats->numNullKeys += null_count;
      }
    }

    if (simpleNotEqual_.has_value()) {
      VELOX_CHECK_NOT_NULL(minMaxSummaryType_);
      const auto stream = cudfInput->stream();
      auto summary = buildMinMaxSummaryTable(
          cudfInput->getTableView(),
          buildKeyIndices_,
          simpleNotEqual_->second,
          stream,
          get_output_mr());
      const auto summaryRows = summary->num_rows();
      std::array<CudfVectorPtr, 1> inputVectors{cudfInput};
      std::array<rmm::cuda_stream_view, 1> inputStreams{stream};
      orderCudfVectorDeallocationsAfterStream(
          std::span<const CudfVectorPtr>(
              inputVectors.data(), inputVectors.size()),
          std::span<const rmm::cuda_stream_view>(
              inputStreams.data(), inputStreams.size()),
          stream);
      if (summaryRows > 0) {
        minMaxSummaryInputs_.push_back(
            std::make_shared<CudfVector>(
                pool(),
                minMaxSummaryType_,
                static_cast<vector_size_t>(summaryRows),
                std::move(summary),
                stream));
      }
      return;
    }

    // Queue regular build inputs, process all at once.
    inputs_.push_back(std::move(cudfInput));
  }
}

bool CudfHashJoinBuild::needsInput() const {
  return !noMoreInput_;
}

RowVectorPtr CudfHashJoinBuild::doGetOutput() {
  return nullptr;
}

void CudfHashJoinBuild::doNoMoreInput() {
  Operator::noMoreInput();
  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;
  // Only last driver collects all answers
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    return;
  }
  // Collect results from peers
  for (auto& peer : peers) {
    auto op = peer->findOperator(planNodeId());
    auto* build = dynamic_cast<CudfHashJoinBuild*>(op);
    VELOX_CHECK_NOT_NULL(build);
    if (simpleNotEqual_.has_value()) {
      minMaxSummaryInputs_.insert(
          minMaxSummaryInputs_.end(),
          std::make_move_iterator(build->minMaxSummaryInputs_.begin()),
          std::make_move_iterator(build->minMaxSummaryInputs_.end()));
      build->minMaxSummaryInputs_.clear();
    } else {
      inputs_.insert(
          inputs_.end(),
          std::make_move_iterator(build->inputs_.begin()),
          std::make_move_iterator(build->inputs_.end()));
      build->inputs_.clear();
    }
    auto retainedInputBatches = build->inputs_.size();
    common::testutil::TestValue::adjust(
        "facebook::velox::cudf_velox::CudfHashJoinBuild::doNoMoreInput::sourceDriverRetainedInputBatchesAfterTransfer",
        &retainedInputBatches);
  }

  SCOPE_EXIT {
    // Realize the promises so that the other Drivers (which were not
    // the last to finish) can continue from the barrier and finish.
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }
  };

  if (CudfConfig::getInstance().debugEnabled) {
    if (!inputs_.empty()) {
      VLOG(1) << "Build batches number of columns: "
              << inputs_[0]->getTableView().num_columns();
    }
    for (auto i = 0; i < inputs_.size(); i++) {
      VLOG(1) << "Build batch " << i
              << ": number of rows: " << inputs_[i]->getTableView().num_rows();
    }
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  auto hashBuildKeyIndices = buildKeyIndices_;

  // Construct hash_join object for join types that use hb->inner_join() or
  // hb->left_join(). Semi filter, anti joins, and left semi project joins use
  // standalone cudf functions or specialized marker paths (e.g.
  // mixed_left_semi_join, filtered_join, mark_join) that build their own
  // lookup state internally.
  const bool buildHashJoin = usesPrebuiltCudfHashJoin(*joinNode_);
  const bool preferDistinctHashJoin = buildHashJoin &&
      (joinNode_->rightKeysUnique() || simpleNotEqual_.has_value()) &&
      CudfConfig::getInstance().distinctHashJoinEnabled;
  const bool preferProbeUniqueInnerJoin =
      buildHashJoin && useProbeUniqueInnerJoin(*joinNode_);

  std::vector<std::shared_ptr<cudf::table>> sharedTables;
  std::vector<std::shared_ptr<cudf::hash_join>> hashObjects;
  std::vector<std::shared_ptr<cudf::distinct_hash_join>> distinctHashObjects;
  std::vector<RowVectorPtr> parkedBuildTables;
  uint64_t parkedBuildRows = 0;
  uint64_t parkedBuildDeviceBytes = 0;
  uint64_t parkedBuildHostBytes = 0;
  bool deferHashHydration = false;
  const auto buildType = asRowType(joinNode_->sources()[1]->outputType());

  const auto recordDeviceMemory = [&](const char* freeBytesName,
                                      const char* poolUsedBytesName,
                                      const char* poolReusableBytesName) {
    const auto info = currentDeviceMemoryInfo();
    if (!info.has_value()) {
      return;
    }
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        freeBytesName,
        RuntimeCounter(
            saturateRuntimeCounter(info->freeBytes),
            RuntimeCounter::Unit::kBytes));
    if (info->hasPoolStats) {
      lockedStats->addRuntimeStat(
          poolUsedBytesName,
          RuntimeCounter(
              saturateRuntimeCounter(info->poolUsedBytes),
              RuntimeCounter::Unit::kBytes));
      lockedStats->addRuntimeStat(
          poolReusableBytesName,
          RuntimeCounter(
              saturateRuntimeCounter(info->poolReusableBytes),
              RuntimeCounter::Unit::kBytes));
    }
  };

  const auto appendBuildTable = [&](std::unique_ptr<cudf::table> table) {
    VELOX_CHECK_NOT_NULL(table);
    const auto tableRows = static_cast<uint64_t>(table->num_rows());
    const auto tableDeviceBytes = static_cast<uint64_t>(table->alloc_size());

    if (deferHashHydration) {
      VELOX_CHECK(preferDistinctHashJoin);
      auto hostTable = with_arrow::toVeloxColumn(
          table->view(), pool(), buildType, "", stream, get_temp_mr());
      VELOX_CHECK_NOT_NULL(hostTable);
      hostTable->setType(buildType);
      const auto tableHostBytes = hostTable->estimateFlatSize();

      // The D2H conversion has completed. Drain the stream-ordered free before
      // materializing the next run so an idle future join cannot retain device
      // payload while an independent branch is still executing.
      table.reset();
      stream.synchronize();

      parkedBuildRows = saturatedAdd(parkedBuildRows, tableRows);
      parkedBuildDeviceBytes =
          saturatedAdd(parkedBuildDeviceBytes, tableDeviceBytes);
      parkedBuildHostBytes = saturatedAdd(parkedBuildHostBytes, tableHostBytes);
      parkedBuildTables.push_back(std::move(hostTable));

      {
        auto lockedStats = stats_.wlock();
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildTables", RuntimeCounter(1));
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildTableRows",
            RuntimeCounter(saturateRuntimeCounter(tableRows)));
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildHostParkedTables", RuntimeCounter(1));
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildHostParkedRows",
            RuntimeCounter(saturateRuntimeCounter(tableRows)));
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildHostParkedDeviceBytes",
            RuntimeCounter(
                saturateRuntimeCounter(tableDeviceBytes),
                RuntimeCounter::Unit::kBytes));
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildHostStorageBytes",
            RuntimeCounter(
                saturateRuntimeCounter(tableHostBytes),
                RuntimeCounter::Unit::kBytes));
      }
      recordDeviceMemory(
          "cudfHashJoinBuildPostParkDeviceFreeBytes",
          "cudfHashJoinBuildPostParkPoolUsedBytes",
          "cudfHashJoinBuildPostParkPoolReusableBytes");
      return;
    }

    const auto tableIndex = sharedTables.size();
    auto sharedTable = std::shared_ptr<cudf::table>(std::move(table));
    auto buildKeyView = sharedTable->view().select(hashBuildKeyIndices);
    bool buildKeysHaveNulls = false;
    if (preferDistinctHashJoin && !joinNode_->rightKeysNonNull()) {
      buildKeysHaveNulls = cudf::has_nulls(buildKeyView);
    }
    const bool useDistinctHashJoin =
        preferDistinctHashJoin && !buildKeysHaveNulls;

    sharedTables.push_back(std::move(sharedTable));
    if (useDistinctHashJoin) {
      hashObjects.push_back(nullptr);
      distinctHashObjects.push_back(
          std::make_shared<cudf::distinct_hash_join>(
              buildKeyView, cudf::null_equality::UNEQUAL, 0.5, stream));
      VELOX_CHECK_NOT_NULL(distinctHashObjects.back());
    } else if (preferProbeUniqueInnerJoin) {
      hashObjects.push_back(nullptr);
      distinctHashObjects.push_back(nullptr);
    } else if (buildHashJoin) {
      hashObjects.push_back(
          std::make_shared<cudf::hash_join>(
              buildKeyView, cudf::null_equality::UNEQUAL, stream));
      distinctHashObjects.push_back(nullptr);
      VELOX_CHECK_NOT_NULL(hashObjects.back());
    } else {
      hashObjects.push_back(nullptr);
      distinctHashObjects.push_back(nullptr);
    }

    {
      auto lockedStats = stats_.wlock();
      lockedStats->addRuntimeStat("cudfHashJoinBuildTables", RuntimeCounter(1));
      lockedStats->addRuntimeStat(
          "cudfHashJoinBuildTableRows",
          RuntimeCounter(saturateRuntimeCounter(tableRows)));
      if (distinctHashObjects.back() != nullptr) {
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildDistinctHashTables", RuntimeCounter(1));
      } else if (hashObjects.back() != nullptr) {
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildGenericHashTables", RuntimeCounter(1));
      }
    }

    recordDeviceMemory(
        "cudfHashJoinBuildPostHashDeviceFreeBytes",
        "cudfHashJoinBuildPostHashPoolUsedBytes",
        "cudfHashJoinBuildPostHashPoolReusableBytes");

    if (CudfConfig::getInstance().debugEnabled) {
      if (distinctHashObjects.back() != nullptr) {
        VLOG(2) << "distinctHashObject " << tableIndex << " is not nullptr "
                << distinctHashObjects.back().get() << "\n";
      } else if (hashObjects.back() != nullptr) {
        VLOG(2) << "hashObject " << tableIndex << " is not nullptr "
                << hashObjects.back().get() << "\n";
      } else {
        VLOG(2) << "hashObject " << tableIndex << " is *** nullptr\n";
      }
    }
  };

  if (simpleNotEqual_.has_value()) {
    VELOX_CHECK_NOT_NULL(minMaxSummaryType_);
    int64_t summaryRows = 0;
    auto tables = compactMinMaxSummaryVectorsBatched(
        std::exchange(minMaxSummaryInputs_, {}),
        minMaxSummaryType_,
        static_cast<cudf::size_type>(buildKeyIndices_.size()),
        stream,
        summaryRows);
    hashBuildKeyIndices =
        sequenceIndices(static_cast<cudf::size_type>(buildKeyIndices_.size()));
    for (auto& table : tables) {
      appendBuildTable(std::move(table));
    }
  } else {
    auto sourceInputs = std::exchange(inputs_, {});
    common::testutil::TestValue::adjust(
        "facebook::velox::cudf_velox::CudfHashJoinBuild::doNoMoreInput::collectedBuildInputs",
        &sourceInputs);

    uint64_t inputRows = 0;
    uint64_t inputBytes = 0;
    for (const auto& input : sourceInputs) {
      VELOX_CHECK_NOT_NULL(input);
      inputRows = saturatedAdd(inputRows, static_cast<uint64_t>(input->size()));
      inputBytes = saturatedAdd(inputBytes, input->estimateFlatSize());
    }

    const auto byteBoundedBuild = supportsByteBoundedHashBuild(*joinNode_);
    const auto targetBytes = byteBoundedBuild
        ? hashBuildBatchTargetBytes()
        : std::numeric_limits<uint64_t>::max();
    const auto maxRows = hashBuildBatchMaxRows();
    const auto deferredHydrationEligible =
        supportsDeferredHashHydration(*joinNode_);
    const auto deferredHydrationThreshold = deferredHydrationEligible
        ? deferredHashHydrationThresholdBytes()
        : std::numeric_limits<uint64_t>::max();
    deferHashHydration = deferredHydrationEligible &&
        hashBuildBatchWorkBytes(inputRows, inputBytes) >
            deferredHydrationThreshold;
    {
      auto lockedStats = stats_.wlock();
      lockedStats->addRuntimeStat(
          "cudfHashJoinBuildInputBatches",
          RuntimeCounter(saturateRuntimeCounter(sourceInputs.size())));
      lockedStats->addRuntimeStat(
          "cudfHashJoinBuildInputRows",
          RuntimeCounter(saturateRuntimeCounter(inputRows)));
      lockedStats->addRuntimeStat(
          "cudfHashJoinBuildInputBytes",
          RuntimeCounter(
              saturateRuntimeCounter(inputBytes),
              RuntimeCounter::Unit::kBytes));
      if (byteBoundedBuild) {
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildByteBoundedJoins", RuntimeCounter(1));
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildBatchTargetBytes",
            RuntimeCounter(
                saturateRuntimeCounter(targetBytes),
                RuntimeCounter::Unit::kBytes));
      }
      if (deferredHydrationEligible) {
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildDeferredHydrationThresholdBytes",
            RuntimeCounter(
                saturateRuntimeCounter(deferredHydrationThreshold),
                RuntimeCounter::Unit::kBytes));
      }
      if (deferHashHydration) {
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildDeferredBuilds", RuntimeCounter(1));
      }
    }

    std::vector<CudfVectorPtr> run;
    uint64_t runRows = 0;
    uint64_t runBytes = 0;
    size_t runSourceBatches = 0;
    size_t retainedSourceBatches = sourceInputs.size();

    const auto reportRetainedSourceBatches = [&]() {
      auto retained = retainedSourceBatches;
      common::testutil::TestValue::adjust(
          "facebook::velox::cudf_velox::CudfHashJoinBuild::doNoMoreInput::retainedInputBatchesAfterMaterializedRun",
          &retained);
    };

    const auto flushRun = [&]() {
      if (run.empty()) {
        return;
      }

      const auto materializedRows = runRows;
      const auto materializedBytes = runBytes;
      const auto materializedWorkBytes =
          hashBuildBatchWorkBytes(materializedRows, materializedBytes);
      {
        // Record the attempted envelope before allocating so failed builds
        // retain enough evidence to identify the next memory peak.
        auto lockedStats = stats_.wlock();
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildMaterializationAttempts", RuntimeCounter(1));
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildMaxMaterializationRows",
            RuntimeCounter(saturateRuntimeCounter(materializedRows)));
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildMaxMaterializationInputBytes",
            RuntimeCounter(
                saturateRuntimeCounter(materializedBytes),
                RuntimeCounter::Unit::kBytes));
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildMaxMaterializationWorkBytes",
            RuntimeCounter(
                saturateRuntimeCounter(materializedWorkBytes),
                RuntimeCounter::Unit::kBytes));
      }
      recordDeviceMemory(
          "cudfHashJoinBuildPreMaterializationDeviceFreeBytes",
          "cudfHashJoinBuildPreMaterializationPoolUsedBytes",
          "cudfHashJoinBuildPreMaterializationPoolReusableBytes");

      // getConcatenatedTable joins all producer streams, copies on the build
      // stream, and rebinds owned inputs so their frees are ordered after that
      // copy. Consuming one run at a time lets the async pool reuse those
      // source allocations before the next run is admitted.
      auto materialized = getConcatenatedTable(
          std::exchange(run, {}),
          joinNode_->sources()[1]->outputType(),
          stream,
          get_output_mr());
      VELOX_CHECK_NOT_NULL(materialized);
      VELOX_CHECK_EQ(
          static_cast<uint64_t>(materialized->num_rows()), materializedRows);
      recordDeviceMemory(
          "cudfHashJoinBuildPostMaterializationDeviceFreeBytes",
          "cudfHashJoinBuildPostMaterializationPoolUsedBytes",
          "cudfHashJoinBuildPostMaterializationPoolReusableBytes");

      VELOX_CHECK_GE(retainedSourceBatches, runSourceBatches);
      retainedSourceBatches -= runSourceBatches;
      {
        auto lockedStats = stats_.wlock();
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildMaterializedBatches", RuntimeCounter(1));
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildMaterializedRows",
            RuntimeCounter(saturateRuntimeCounter(materializedRows)));
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildMaterializedBytes",
            RuntimeCounter(
                saturateRuntimeCounter(materialized->alloc_size()),
                RuntimeCounter::Unit::kBytes));
        lockedStats->addRuntimeStat(
            "cudfHashJoinBuildSourceBatchesConsumed",
            RuntimeCounter(saturateRuntimeCounter(runSourceBatches)));
      }

      appendBuildTable(std::move(materialized));
      reportRetainedSourceBatches();
      runRows = 0;
      runBytes = 0;
      runSourceBatches = 0;
    };

    for (auto& input : sourceInputs) {
      VELOX_CHECK_NOT_NULL(input);
      const auto rows = static_cast<uint64_t>(input->size());
      const auto bytes = input->estimateFlatSize();
      const auto workBytes = hashBuildBatchWorkBytes(rows, bytes);

      if (rows > maxRows || workBytes > targetBytes) {
        flushRun();

        // Bound an individually oversized owner with one zero-copy slice at a
        // time. The original allocation cannot be released until its final
        // slice, but no individual concatenate or hash table can scale to the
        // full owner.
        auto owner = std::move(input);
        const auto ownerView = owner->getTableView();
        const auto ownerFlatSize = owner->estimateFlatSize();
        const auto workBytesPerRow = std::max(
            saturatedMultiply(
                ceilingDivide(std::max<uint64_t>(ownerFlatSize, 1), rows), 2),
            saturatedAdd(
                ceilingDivide(std::max<uint64_t>(ownerFlatSize, 1), rows),
                kHashBuildWorkspaceBytesPerRow));
        // Leave additional room for row-size skew in variable-width columns.
        // A single row can still exceed the target, but ordinary skew must not
        // turn one oversized owner into another target-sized allocation.
        const auto conservativeWorkBytesPerRow =
            saturatedMultiply(workBytesPerRow, 2);
        const auto rowsPerSlice = std::min<uint64_t>(
            maxRows,
            std::max<uint64_t>(targetBytes / conservativeWorkBytesPerRow, 1));
        const auto flatBytesPerRow = ownerFlatSize / rows;
        const auto flatByteRemainder = ownerFlatSize % rows;
        const auto flatSizeAtRow = [&](uint64_t row) {
          return saturatedAdd(
              saturatedMultiply(flatBytesPerRow, row),
              std::min<uint64_t>(row, flatByteRemainder));
        };

        for (uint64_t start = 0; start < rows;) {
          const auto end = std::min(start + rowsPerSlice, rows);
          const auto sliceRows = end - start;
          auto slices = cudf::slice(
              ownerView,
              {static_cast<cudf::size_type>(start),
               static_cast<cudf::size_type>(end)},
              owner->stream());
          VELOX_CHECK_EQ(slices.size(), 1);

          // cuDF slices of nested and variable-width columns retain unsliced
          // child buffers. Account the owner's logical flat size across its
          // slices explicitly; recursively measuring the view would count the
          // complete child allocation for every slice.
          const auto sliceBytes = flatSizeAtRow(end) - flatSizeAtRow(start);
          auto slice = std::make_shared<CudfVector>(
              pool(),
              owner->type(),
              static_cast<vector_size_t>(sliceRows),
              slices.front(),
              CudfVector::ViewOwner{owner},
              owner->stream(),
              sliceBytes);

          runRows = sliceRows;
          runBytes = sliceBytes;
          run.push_back(std::move(slice));
          runSourceBatches = 0;
          {
            auto lockedStats = stats_.wlock();
            lockedStats->addRuntimeStat(
                "cudfHashJoinBuildOversizedInputSlices", RuntimeCounter(1));
          }
          flushRun();
          start = end;
        }

        owner.reset();
        VELOX_CHECK_GT(retainedSourceBatches, 0);
        --retainedSourceBatches;
        {
          auto lockedStats = stats_.wlock();
          lockedStats->addRuntimeStat(
              "cudfHashJoinBuildSourceBatchesConsumed", RuntimeCounter(1));
        }
        reportRetainedSourceBatches();
        continue;
      }

      const auto candidateRows = saturatedAdd(runRows, rows);
      const auto candidateBytes = saturatedAdd(runBytes, bytes);
      if (!run.empty() &&
          (candidateRows > maxRows ||
           hashBuildBatchWorkBytes(candidateRows, candidateBytes) >
               targetBytes)) {
        flushRun();
      }

      run.push_back(std::move(input));
      runRows = saturatedAdd(runRows, rows);
      runBytes = saturatedAdd(runBytes, bytes);
      ++runSourceBatches;
    }
    flushRun();

    if (sharedTables.empty() && parkedBuildTables.empty()) {
      appendBuildTable(makeEmptyTable(joinNode_->sources()[1]->outputType()));
    }
    VELOX_CHECK_EQ(retainedSourceBatches, 0);
  }

  if (CudfConfig::getInstance().debugEnabled) {
    if (!sharedTables.empty()) {
      VLOG(1) << "Build table number of columns: "
              << sharedTables[0]->num_columns();
      for (auto i = 0; i < sharedTables.size(); i++) {
        VLOG(1) << "Build table " << i
                << ": number of rows: " << sharedTables[i]->num_rows();
      }
    } else if (!parkedBuildTables.empty()) {
      VLOG(1) << "Deferred " << parkedBuildTables.size()
              << " build tables on host with " << parkedBuildRows << " rows";
    }
  }

  // set hash table to CudfHashJoinBridge
  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto cudfHashJoinBridge =
      std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
  VELOX_CHECK_NOT_NULL(cudfHashJoinBridge);

  if (deferHashHydration) {
    VELOX_CHECK(sharedTables.empty());
    VELOX_CHECK(hashObjects.empty());
    VELOX_CHECK(distinctHashObjects.empty());
    cudfHashJoinBridge->setParkedHashTable(
        CudfHashJoinBridge::ParkedHashJoinState{
            std::move(parkedBuildTables),
            parkedBuildRows,
            parkedBuildDeviceBytes,
            parkedBuildHostBytes});
  } else {
    auto buildReadyEvent = std::make_shared<CudaEvent>(cudaEventDisableTiming);
    buildReadyEvent->recordFrom(stream);
    cudfHashJoinBridge->setBuildStream(stream);
    cudfHashJoinBridge->setBuildReadyEvent(std::move(buildReadyEvent));
    cudfHashJoinBridge->setHashTable(
        std::make_optional(
            CudfHashJoinBridge::HashJoinState{
                std::move(sharedTables),
                std::move(hashObjects),
                std::move(distinctHashObjects),
                false}));
  }
}

exec::BlockingReason CudfHashJoinBuild::isBlocked(ContinueFuture* future) {
  if (!future_.valid()) {
    return exec::BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return exec::BlockingReason::kWaitForJoinBuild;
}

bool CudfHashJoinBuild::isFinished() {
  return !future_.valid() && noMoreInput_;
}

CudfHashJoinProbe::CudfHashJoinProbe(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          joinNode->outputType(),
          joinNode->id(),
          "CudfHashJoinProbe",
          nvtx3::rgb{0, 128, 128}, // Teal
          NvtxMethodFlag::kAll,
          std::nullopt, // spillConfig
          joinNode),
      joinNode_(joinNode),
      probeType_(joinNode_->sources()[0]->outputType()),
      buildType_(joinNode_->sources()[1]->outputType()),
      cudaEvent_(std::make_unique<CudaEvent>(cudaEventDisableTiming)) {
  auto const& leftKeys = joinNode_->leftKeys(); // probe keys
  auto const& rightKeys = joinNode_->rightKeys(); // build keys

  if (CudfConfig::getInstance().debugEnabled) {
    for (int i = 0; i < probeType_->names().size(); i++) {
      VLOG(1) << "Left column " << i << ": " << probeType_->names()[i];
    }

    for (int i = 0; i < buildType_->names().size(); i++) {
      VLOG(1) << "Right column " << i << ": " << buildType_->names()[i];
    }

    for (int i = 0; i < leftKeys.size(); i++) {
      VLOG(1) << "Left key " << i << ": " << leftKeys[i]->name() << " "
              << leftKeys[i]->type()->kind();
    }

    for (int i = 0; i < rightKeys.size(); i++) {
      VLOG(1) << "Right key " << i << ": " << rightKeys[i]->name() << " "
              << rightKeys[i]->type()->kind();
    }
  }

  auto const probeTableNumColumns = probeType_->size();
  leftKeyIndices_ = std::vector<cudf::size_type>(leftKeys.size());
  for (size_t i = 0; i < leftKeyIndices_.size(); i++) {
    leftKeyIndices_[i] = static_cast<cudf::size_type>(
        probeType_->getChildIdx(leftKeys[i]->name()));
    VELOX_CHECK_LT(leftKeyIndices_[i], probeTableNumColumns);
  }
  auto const buildTableNumColumns = buildType_->size();
  rightKeyIndices_ = std::vector<cudf::size_type>(rightKeys.size());
  for (size_t i = 0; i < rightKeyIndices_.size(); i++) {
    rightKeyIndices_[i] = static_cast<cudf::size_type>(
        buildType_->getChildIdx(rightKeys[i]->name()));
    VELOX_CHECK_LT(rightKeyIndices_[i], buildTableNumColumns);
  }

  auto outputType = joinNode_->outputType();
  leftColumnIndicesToGather_ = std::vector<cudf::size_type>();
  rightColumnIndicesToGather_ = std::vector<cudf::size_type>();
  leftColumnOutputIndices_ = std::vector<size_t>();
  rightColumnOutputIndices_ = std::vector<size_t>();
  for (int i = 0; i < outputType->names().size(); i++) {
    auto const outputName = outputType->names()[i];
    if (CudfConfig::getInstance().debugEnabled) {
      VLOG(1) << "Output column " << i << ": " << outputName;
    }
    auto channel = probeType_->getChildIdxIfExists(outputName);
    if (channel.has_value()) {
      leftColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(channel.value()));
      leftColumnOutputIndices_.push_back(i);
      continue;
    }
    channel = buildType_->getChildIdxIfExists(outputName);
    if (channel.has_value()) {
      rightColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(channel.value()));
      rightColumnOutputIndices_.push_back(i);
      continue;
    }
    // For LEFT SEMI PROJECT, the last column is the boolean "match" column
    // which is not in probe or build types - skip it here, handled separately
    if (isLeftSemiProjectJoin(joinNode_->joinType()) &&
        i == outputType->size() - 1 &&
        outputType->childAt(i)->kind() == TypeKind::BOOLEAN) {
      continue;
    }
    VELOX_FAIL(
        "Join field {} not in probe or build input", outputType->children()[i]);
  }

  if (CudfConfig::getInstance().debugEnabled) {
    for (int i = 0; i < leftColumnIndicesToGather_.size(); i++) {
      VLOG(1) << "Left index to gather " << i << ": "
              << leftColumnIndicesToGather_[i];
    }

    for (int i = 0; i < rightColumnIndicesToGather_.size(); i++) {
      VLOG(1) << "Right index to gather " << i << ": "
              << rightColumnIndicesToGather_[i];
    }
  }
}

void CudfHashJoinProbe::waitForBuildReady(rmm::cuda_stream_view stream) {
  if (buildReadyEvent_ != nullptr) {
    buildReadyEvent_->waitOn(stream);
  }
}

CudfHashJoinProbe::hash_type CudfHashJoinProbe::hydrateParkedHashTable(
    const CudfHashJoinBridge::parked_hash_type& parkedHashObject,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(supportsDeferredHashHydration(*joinNode_));
  VELOX_CHECK(!parkedHashObject.buildTables.empty());

  hash_type hydrated;
  hydrated.deferredHydration = true;
  hydrated.buildTables.reserve(parkedHashObject.buildTables.size());
  hydrated.hashJoins.reserve(parkedHashObject.buildTables.size());
  hydrated.distinctHashJoins.reserve(parkedHashObject.buildTables.size());

  uint64_t hydratedRows = 0;
  uint64_t hydratedBytes = 0;
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "cudfHashJoinProbeHydrationClaims", RuntimeCounter(1));
  }

  for (const auto& hostTable : parkedHashObject.buildTables) {
    VELOX_CHECK_NOT_NULL(hostTable);
    auto table =
        with_arrow::toCudfTable(hostTable, pool(), stream, get_output_mr());
    VELOX_CHECK_NOT_NULL(table);
    const auto rows = static_cast<uint64_t>(table->num_rows());
    const auto bytes = static_cast<uint64_t>(table->alloc_size());
    auto sharedTable = std::shared_ptr<cudf::table>(std::move(table));
    auto buildKeyView = sharedTable->view().select(rightKeyIndices_);
    auto distinctHash = std::make_shared<cudf::distinct_hash_join>(
        buildKeyView, cudf::null_equality::UNEQUAL, 0.5, stream);
    VELOX_CHECK_NOT_NULL(distinctHash);

    hydratedRows = saturatedAdd(hydratedRows, rows);
    hydratedBytes = saturatedAdd(hydratedBytes, bytes);
    hydrated.buildTables.push_back(std::move(sharedTable));
    hydrated.hashJoins.push_back(nullptr);
    hydrated.distinctHashJoins.push_back(std::move(distinctHash));

    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "cudfHashJoinProbeHydratedTables", RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        "cudfHashJoinProbeHydratedRows",
        RuntimeCounter(saturateRuntimeCounter(rows)));
    lockedStats->addRuntimeStat(
        "cudfHashJoinProbeHydratedBytes",
        RuntimeCounter(
            saturateRuntimeCounter(bytes), RuntimeCounter::Unit::kBytes));
  }

  VELOX_CHECK_EQ(hydratedRows, parkedHashObject.rows);
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "cudfHashJoinProbeHydratedTotalBytes",
        RuntimeCounter(
            saturateRuntimeCounter(hydratedBytes),
            RuntimeCounter::Unit::kBytes));
  }
  if (auto info = currentDeviceMemoryInfo(); info.has_value()) {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "cudfHashJoinProbePostHydrationDeviceFreeBytes",
        RuntimeCounter(
            saturateRuntimeCounter(info->freeBytes),
            RuntimeCounter::Unit::kBytes));
    if (info->hasPoolStats) {
      lockedStats->addRuntimeStat(
          "cudfHashJoinProbePostHydrationPoolUsedBytes",
          RuntimeCounter(
              saturateRuntimeCounter(info->poolUsedBytes),
              RuntimeCounter::Unit::kBytes));
      lockedStats->addRuntimeStat(
          "cudfHashJoinProbePostHydrationPoolReusableBytes",
          RuntimeCounter(
              saturateRuntimeCounter(info->poolReusableBytes),
              RuntimeCounter::Unit::kBytes));
    }
  }
  return hydrated;
}

void CudfHashJoinProbe::initialize() {
  Operator::initialize();

  if (!joinNode_->filter()) {
    return;
  }

  // simplify expression
  exec::ExprSet exprs({joinNode_->filter()}, operatorCtx_->execCtx());
  VELOX_CHECK_EQ(exprs.exprs().size(), 1);

  // Disable AST-based filtering (and force precomputation) if the filter
  // expression contains a type the AST/JIT evaluator can't handle, using the
  // same shallow check applied during regular expression evaluation.
  if (containsAstUnsupportedType(exprs.exprs()[0])) {
    useAstFilter_ = false;
  }

  // Validate AST filtering for this join type now to avoid run-time error.
  if (joinNode_->isRightSemiFilterJoin() || joinNode_->isLeftSemiFilterJoin() ||
      joinNode_->isAntiJoin()) {
    VELOX_CHECK(
        useAstFilter_,
        "AST expression evaluation must be enabled for semi-filter and anti joins.");
  }

  // Create a reusable evaluator for the filter column. This is expensive to
  // build, and the expression + input schema are stable for the lifetime of
  // the operator instance.
  std::vector<velox::RowTypePtr> filterRowTypes{probeType_, buildType_};
  filterEvaluator_ = createCudfExpression(
      exprs.exprs()[0], facebook::velox::type::concatRowTypes(filterRowTypes));

  if (joinNode_->isLeftSemiProjectJoin() && leftKeyIndices_.size() == 1 &&
      rightKeyIndices_.size() == 1) {
    auto simpleNotEqual = detectSimpleCrossSideNotEqual(
        joinNode_->filter(), probeType_, buildType_);
    if (simpleNotEqual.has_value()) {
      simpleNotEqualLeftIndex_ = simpleNotEqual->first;
      simpleNotEqualRightIndex_ = simpleNotEqual->second;
    }
  }

  // Check if the filter expression spans both join sides (e.g., switch
  // expressions referencing columns from both probe and build). If so, we
  // cannot use AST-based filtering and must fall back to filterEvaluator_.
  if (hasNonAstSubexprSpanningBothSides(
          exprs.exprs()[0], probeType_, buildType_)) {
    VLOG(1) << "Filter expression spans both join sides, using "
               "filterEvaluator_ instead of AST";
    useAstFilter_ = false;
    return;
  }

  // We don't need to get tables that contain conditional comparison columns
  // We'll pass the entire table. The ast will handle finding the required
  // columns. This is required because we build the ast with whole row schema
  // and the column locations in that schema translate to column locations
  // in whole tables

  if (useAstFilter_) {
    // create ast tree
    if (joinNode_->isRightJoin() || joinNode_->isRightSemiFilterJoin()) {
      createAstTree(
          exprs.exprs()[0],
          tree_,
          scalars_,
          buildType_,
          probeType_,
          rightPrecomputeInstructions_,
          leftPrecomputeInstructions_);
    } else {
      createAstTree(
          exprs.exprs()[0],
          tree_,
          scalars_,
          probeType_,
          buildType_,
          leftPrecomputeInstructions_,
          rightPrecomputeInstructions_);
    }
  }
}

bool CudfHashJoinProbe::needsInput() const {
  if (joinNode_->isRightSemiFilterJoin()) {
    return !noMoreInput_;
  }
  return !noMoreInput_ && !finished_ && input_ == nullptr;
}

void CudfHashJoinProbe::doAddInput(RowVectorPtr input) {
  VELOX_CHECK_EQ(
      nextBuildTableIndex_,
      0,
      "Cannot accept a new probe input while deferred build shards remain");
  if (skipInput_) {
    VELOX_CHECK_NULL(input_);
    return;
  }
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  auto const null_count = countNullJoinKeys(
      cudfInput->getTableView(), leftKeyIndices_, cudfInput->stream());
  {
    // Update statistics for null keys in join operator.
    auto lockedStats = stats_.wlock();
    lockedStats->numNullKeys += null_count;
  }
  if (joinNode_->isRightSemiFilterJoin()) {
    // Queue inputs and process all at once
    if (input->size() > 0) {
      inputs_.push_back(std::move(cudfInput));
    }
    return;
  }

  if (input->size() > 0) {
    input_ = std::move(input);
  }
}

void CudfHashJoinProbe::doNoMoreInput() {
  Operator::noMoreInput();
  const bool deferredInnerJoin = deferredHydration_ ||
      (hashObject_.has_value() && hashObject_->deferredHydration);
  if (!joinNode_->isRightJoin() && !joinNode_->isRightSemiFilterJoin() &&
      !joinNode_->isFullJoin() && !deferredInnerJoin) {
    return;
  }
  deferredProbeBarrierEntered_ = deferredInnerJoin;
  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;
  // Only last driver collects all answers
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    return;
  }

  SCOPE_EXIT {
    // Realize the promises so that the other Drivers (which were not
    // the last to finish) can continue from the barrier and finish.
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }
  };

  if (deferredInnerJoin) {
    auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId());
    auto cudfJoinBridge =
        std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
    VELOX_CHECK_NOT_NULL(cudfJoinBridge);
    cudfJoinBridge->probeFinished();
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "cudfHashJoinProbeBridgeReleases", RuntimeCounter(1));
    return;
  }

  if (joinNode_->isRightJoin() || joinNode_->isFullJoin()) {
    isLastDriver_ = true;
    if (hashObject_.has_value()) {
      auto stream = cudfGlobalStreamPool().get_stream();

      // The allPeersFinished barrier above synchronizes CPU threads, but not
      // GPU streams. A driver's CPU thread may return from getOutput() while
      // its GPU work (updating rightMatchedFlags_) is still in flight.
      // join_streams establishes GPU-side ordering so that all probe stream
      // operations complete before the BITWISE_OR reads below.
      // Drivers without lastProbeStream_ (no probe batches) are skipped:
      // their flags are all-false from host-synchronized init with no pending
      // GPU work.
      std::vector<rmm::cuda_stream_view> inputStreams;
      if (lastProbeStream_.has_value()) {
        inputStreams.push_back(lastProbeStream_.value());
      }
      for (auto& peer : peers) {
        if (peer.get() == operatorCtx_->driver()) {
          continue;
        }
        // CudfBatchConcat and CudfHashJoinProbe intentionally share the join
        // plan-node ID. Locate the peer by operator ID so we get the probe,
        // not the first operator with that plan-node ID.
        auto op = peer->findOperator(operatorId());
        auto* probe = dynamic_cast<CudfHashJoinProbe*>(op);
        VELOX_CHECK_NOT_NULL(probe);
        if (probe->lastProbeStream_.has_value()) {
          inputStreams.push_back(probe->lastProbeStream_.value());
        }
      }
      if (!inputStreams.empty()) {
        cudf::detail::join_streams(inputStreams, stream);
      }

      for (auto& peer : peers) {
        if (peer.get() == operatorCtx_->driver()) {
          continue;
        }
        auto op = peer->findOperator(operatorId());
        auto* probe = dynamic_cast<CudfHashJoinProbe*>(op);
        VELOX_CHECK_NOT_NULL(probe);
        // Combine flags per partition using cuDF bitwise OR
        // DM: This needs a relook. This is for when build side exceeds cudf
        // size_type limits. In case of multiple right side chunks, I'm not sure
        // if partitions to combine are in the same place p
        for (size_t p = 0; p < rightMatchedFlags_.size(); ++p) {
          auto or_result = cudf::binary_operation(
              rightMatchedFlags_[p]->view(),
              probe->rightMatchedFlags_[p]->view(),
              cudf::binary_operator::BITWISE_OR,
              cudf::data_type{cudf::type_id::BOOL8},
              stream,
              get_temp_mr());
          // binary_operation is async on `stream`; the old column destructs via
          // cudaFreeAsync on its allocation stream (not `stream`), so the free
          // can race the kernel. Drain `stream` before the move-assign.
          stream.synchronize();
          rightMatchedFlags_[p] = std::move(or_result);
        }
      }
      stream.synchronize();
    }
    return;
  }

  // Handling RightSemiFilterJoin
  // Collect results from peers
  for (auto& peer : peers) {
    auto op = peer->findOperator(operatorId());
    auto* probe = dynamic_cast<CudfHashJoinProbe*>(op);
    VELOX_CHECK_NOT_NULL(probe);
    for (auto& input : probe->inputs_) {
      inputs_.push_back(std::move(input));
    }
    probe->inputs_.clear();
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  // Using output_mr here to allow spilling queued up large tables
  auto tbl = getConcatenatedTable(
      std::exchange(inputs_, {}),
      joinNode_->sources()[1]->outputType(),
      stream,
      get_output_mr());

  VELOX_CHECK_NOT_NULL(tbl);

  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "Probe table number of columns: " << tbl->num_columns();
    VLOG(1) << "Probe table number of rows: " << tbl->num_rows();
  }

  // Store the concatenated table in input_
  input_ = std::make_shared<CudfVector>(
      operatorCtx_->pool(),
      joinNode_->outputType(),
      tbl->num_rows(),
      std::move(tbl),
      stream);
}

CudfHashJoinProbe::JoinOutput CudfHashJoinProbe::unfilteredOutput(
    cudf::table_view leftTableView,
    cudf::column_view leftIndicesCol,
    cudf::table_view rightTableView,
    cudf::column_view rightIndicesCol,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::column>> joinedCols;
  auto const numRows = static_cast<vector_size_t>(
      std::max(leftIndicesCol.size(), rightIndicesCol.size()));
  auto leftInput = leftTableView.select(leftColumnIndicesToGather_);
  auto rightInput = rightTableView.select(rightColumnIndicesToGather_);
  auto leftResult = cudf::gather(
      leftInput, leftIndicesCol, oobPolicy, stream, get_output_mr());
  auto rightResult = cudf::gather(
      rightInput, rightIndicesCol, oobPolicy, stream, get_output_mr());

  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "Left result number of columns: " << leftResult->num_columns();
    VLOG(1) << "Right result number of columns: " << rightResult->num_columns();
  }

  auto leftCols = leftResult->release();
  auto rightCols = rightResult->release();
  joinedCols.resize(outputType_->names().size());
  for (int i = 0; i < leftColumnOutputIndices_.size(); i++) {
    joinedCols[leftColumnOutputIndices_[i]] = std::move(leftCols[i]);
  }
  for (int i = 0; i < rightColumnOutputIndices_.size(); i++) {
    joinedCols[rightColumnOutputIndices_[i]] = std::move(rightCols[i]);
  }
  if (buildStream_.has_value()) {
    // Ensure deallocation of build table happens after probe gathers
    cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
  }
  stream.synchronize();
  return {std::make_unique<cudf::table>(std::move(joinedCols)), numRows};
}

CudfHashJoinProbe::JoinOutput
CudfHashJoinProbe::unfilteredOutputWithIdentityLeft(
    cudf::table_view leftTableView,
    cudf::table_view rightTableView,
    cudf::column_view rightIndicesCol,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::column>> joinedCols;
  joinedCols.resize(outputType_->names().size());

  auto leftInput = leftTableView.select(leftColumnIndicesToGather_);
  for (size_t i = 0; i < leftColumnIndicesToGather_.size(); ++i) {
    joinedCols[leftColumnOutputIndices_[i]] = std::make_unique<cudf::column>(
        leftInput.column(i), stream, get_output_mr());
  }

  auto rightInput = rightTableView.select(rightColumnIndicesToGather_);
  auto rightResult = cudf::gather(
      rightInput, rightIndicesCol, oobPolicy, stream, get_output_mr());
  auto rightCols = rightResult->release();
  for (size_t i = 0; i < rightColumnOutputIndices_.size(); ++i) {
    joinedCols[rightColumnOutputIndices_[i]] = std::move(rightCols[i]);
  }

  if (buildStream_.has_value()) {
    // Ensure deallocation of build table happens after probe output copies.
    cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
  }
  stream.synchronize();
  return {
      std::make_unique<cudf::table>(std::move(joinedCols)),
      static_cast<vector_size_t>(rightIndicesCol.size())};
}

CudfHashJoinProbe::JoinOutput CudfHashJoinProbe::filteredOutput(
    cudf::table_view leftTableView,
    cudf::column_view leftIndicesCol,
    cudf::table_view rightTableView,
    cudf::column_view rightIndicesCol,
    std::function<std::vector<std::unique_ptr<cudf::column>>(
        std::vector<std::unique_ptr<cudf::column>>&&,
        cudf::column_view)> func,
    rmm::cuda_stream_view stream) {
  auto leftResult = cudf::gather(
      leftTableView, leftIndicesCol, oobPolicy, stream, get_output_mr());
  auto rightResult = cudf::gather(
      rightTableView, rightIndicesCol, oobPolicy, stream, get_output_mr());
  auto leftColsSize = leftResult->num_columns();
  auto rightColsSize = rightResult->num_columns();

  std::vector<std::unique_ptr<cudf::column>> joinedCols = leftResult->release();
  auto rightCols = rightResult->release();
  joinedCols.insert(
      joinedCols.end(),
      std::make_move_iterator(rightCols.begin()),
      std::make_move_iterator(rightCols.end()));

  VELOX_CHECK_NOT_NULL(
      filterEvaluator_,
      "Join filter evaluator must be initialized before filteredOutput()");
  std::vector<cudf::column_view> joinedColViews;
  joinedColViews.reserve(joinedCols.size());
  for (const auto& col : joinedCols) {
    joinedColViews.push_back(col->view());
  }
  auto filterColumns =
      filterEvaluator_->eval(joinedColViews, stream, get_output_mr());
  auto filterColumn = asView(filterColumns);

  joinedCols = func(std::move(joinedCols), filterColumn);
  auto const numRows = filteredOutputNumRows(
      outputType_->size() == 0,
      filterColumn,
      joinedCols,
      stream,
      get_temp_mr());

  auto filteredjoinedCols =
      std::vector<std::unique_ptr<cudf::column>>(outputType_->names().size());
  for (int i = 0; i < leftColumnOutputIndices_.size(); i++) {
    filteredjoinedCols[leftColumnOutputIndices_[i]] =
        std::move(joinedCols[leftColumnIndicesToGather_[i]]);
  }
  for (int i = 0; i < rightColumnOutputIndices_.size(); i++) {
    filteredjoinedCols[rightColumnOutputIndices_[i]] =
        std::move(joinedCols[leftColsSize + rightColumnIndicesToGather_[i]]);
  }
  joinedCols = std::move(filteredjoinedCols);
  if (buildStream_.has_value()) {
    // Ensure any deallocation of join indices is ordered wrt probe gathers
    cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
  }
  stream.synchronize();
  return {std::make_unique<cudf::table>(std::move(joinedCols)), numRows};
}

CudfHashJoinProbe::JoinOutput CudfHashJoinProbe::filteredOutputIndices(
    cudf::table_view leftTableView,
    cudf::column_view leftIndicesCol,
    cudf::table_view rightTableView,
    cudf::column_view rightIndicesCol,
    cudf::table_view extendedLeftView,
    cudf::table_view extendedRightView,
    cudf::join_kind joinKind,
    rmm::cuda_stream_view stream) {
  // Use extended views (with precomputed columns) for filter evaluation
  auto [filteredLeftJoinIndices, filteredRightJoinIndices] =
      cudf::filter_join_indices(
          extendedLeftView,
          extendedRightView,
          leftIndicesCol,
          rightIndicesCol,
          tree_.back(),
          joinKind,
          stream,
          get_temp_mr());

  auto filteredLeftIndicesSpan =
      cudf::device_span<cudf::size_type const>{*filteredLeftJoinIndices};
  auto filteredRightIndicesSpan =
      cudf::device_span<cudf::size_type const>{*filteredRightJoinIndices};
  auto filteredLeftIndicesCol = cudf::column_view{filteredLeftIndicesSpan};
  auto filteredRightIndicesCol = cudf::column_view{filteredRightIndicesSpan};
  // Use original views (without precomputed columns) for gathering output
  return unfilteredOutput(
      leftTableView,
      filteredLeftIndicesCol,
      rightTableView,
      filteredRightIndicesCol,
      stream);
}

CudfHashJoinProbe::JoinIndices CudfHashJoinProbe::computeInnerJoinIndices(
    size_t buildTableIndex,
    cudf::table_view leftKeyView,
    rmm::cuda_stream_view stream) {
  auto& state = hashObject_.value();
  VELOX_CHECK_LT(buildTableIndex, state.hashJoins.size());
  VELOX_CHECK_LT(buildTableIndex, state.distinctHashJoins.size());

  JoinIndices indices;
  if (state.distinctHashJoins[buildTableIndex] != nullptr) {
    indices = state.distinctHashJoins[buildTableIndex]->inner_join(
        leftKeyView, stream, get_temp_mr());
  } else {
    VELOX_CHECK_NOT_NULL(state.hashJoins[buildTableIndex]);
    indices = state.hashJoins[buildTableIndex]->inner_join(
        leftKeyView, std::nullopt, stream, get_temp_mr());
  }
  return indices;
}

CudfHashJoinProbe::JoinIndices CudfHashJoinProbe::computeLeftJoinIndices(
    size_t buildTableIndex,
    cudf::table_view leftKeyView,
    rmm::cuda_stream_view stream) {
  auto& state = hashObject_.value();
  VELOX_CHECK_LT(buildTableIndex, state.hashJoins.size());
  VELOX_CHECK_LT(buildTableIndex, state.distinctHashJoins.size());

  JoinIndices indices;
  if (state.distinctHashJoins[buildTableIndex] != nullptr) {
    auto rightJoinIndices = state.distinctHashJoins[buildTableIndex]->left_join(
        leftKeyView, stream, get_temp_mr());
    auto leftJoinIndicesColumn = cudf::sequence(
        leftKeyView.num_rows(),
        cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
        cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
        stream,
        get_temp_mr());
    auto leftJoinIndices =
        std::make_unique<rmm::device_uvector<cudf::size_type>>(
            leftKeyView.num_rows(), stream, get_temp_mr());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        leftJoinIndices->data(),
        leftJoinIndicesColumn->view().data<cudf::size_type>(),
        leftJoinIndices->size() * sizeof(cudf::size_type),
        cudaMemcpyDeviceToDevice,
        stream.value()));
    indices =
        std::make_pair(std::move(leftJoinIndices), std::move(rightJoinIndices));
  } else {
    VELOX_CHECK_NOT_NULL(state.hashJoins[buildTableIndex]);
    indices = state.hashJoins[buildTableIndex]->left_join(
        leftKeyView, std::nullopt, stream, get_temp_mr());
  }
  return indices;
}

CudfHashJoinProbe::JoinIndices
CudfHashJoinProbe::computeProbeUniqueInnerJoinIndices(
    cudf::table_view leftKeyView,
    cudf::table_view rightKeyView,
    rmm::cuda_stream_view stream) {
  cudf::distinct_hash_join probeHashJoin{
      leftKeyView, cudf::null_equality::UNEQUAL, 0.5, stream};
  auto [rightJoinIndices, leftJoinIndices] =
      probeHashJoin.inner_join(rightKeyView, stream, get_temp_mr());
  return std::make_pair(
      std::move(leftJoinIndices), std::move(rightJoinIndices));
}

CudfHashJoinProbe::JoinOutput CudfHashJoinProbe::innerJoinShard(
    size_t buildTableIndex,
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(!joinNode_->filter());
  VELOX_CHECK(!useProbeUniqueInnerJoin(*joinNode_));
  auto& state = hashObject_.value();
  VELOX_CHECK_LT(buildTableIndex, state.buildTables.size());
  auto& rightTable = state.buildTables[buildTableIndex];
  VELOX_CHECK_NOT_NULL(rightTable);
  auto rightTableView = rightTable->view();

  if (joinNode_->leftKeysCoveredByRightKeys() &&
      state.buildTables.size() == 1 &&
      state.distinctHashJoins[buildTableIndex] != nullptr) {
    auto rightJoinIndices = state.distinctHashJoins[buildTableIndex]->left_join(
        leftTableView.select(leftKeyIndices_), stream, get_temp_mr());
    auto rightIndicesSpan =
        cudf::device_span<cudf::size_type const>{*rightJoinIndices};
    return unfilteredOutputWithIdentityLeft(
        leftTableView,
        rightTableView,
        cudf::column_view{rightIndicesSpan},
        stream);
  }

  auto [leftJoinIndices, rightJoinIndices] = computeInnerJoinIndices(
      buildTableIndex, leftTableView.select(leftKeyIndices_), stream);
  auto leftIndicesSpan =
      cudf::device_span<cudf::size_type const>{*leftJoinIndices};
  auto rightIndicesSpan =
      cudf::device_span<cudf::size_type const>{*rightJoinIndices};
  return unfilteredOutput(
      leftTableView,
      cudf::column_view{leftIndicesSpan},
      rightTableView,
      cudf::column_view{rightIndicesSpan},
      stream);
}

std::vector<CudfHashJoinProbe::JoinOutput> CudfHashJoinProbe::innerJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<JoinOutput> cudfOutputs;

  auto& rightTables = hashObject_.value().buildTables;

  // Precompute left (probe) table columns if needed (once, outside loop)
  std::vector<ColumnOrView> leftPrecomputed;
  cudf::table_view extendedLeftView = leftTableView;
  if (joinNode_->filter() && useAstFilter_ &&
      !leftPrecomputeInstructions_.empty()) {
    auto leftColumnViews = tableViewToColumnViews(leftTableView);
    leftPrecomputed = precomputeSubexpressions(
        leftColumnViews,
        leftPrecomputeInstructions_,
        scalars_,
        probeType_,
        stream);
    extendedLeftView = createExtendedTableView(leftTableView, leftPrecomputed);
  }

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();

    // Use cached precomputed columns for right (build) table
    cudf::table_view extendedRightView =
        (joinNode_->filter() && useAstFilter_ &&
         !rightPrecomputeInstructions_.empty())
        ? cachedExtendedRightViews_[i]
        : rightTableView;

    auto& state = hashObject_.value();
    if (!joinNode_->filter() && joinNode_->leftKeysCoveredByRightKeys() &&
        rightTables.size() == 1 && state.distinctHashJoins[i] != nullptr) {
      auto rightJoinIndices = state.distinctHashJoins[i]->left_join(
          leftTableView.select(leftKeyIndices_), stream, get_temp_mr());

      auto rightIndicesSpan =
          cudf::device_span<cudf::size_type const>{*rightJoinIndices};
      auto rightIndicesCol = cudf::column_view{rightIndicesSpan};
      cudfOutputs.push_back(unfilteredOutputWithIdentityLeft(
          leftTableView, rightTableView, rightIndicesCol, stream));
      continue;
    }

    // left = probe, right = build
    JoinIndices indices;
    if (useProbeUniqueInnerJoin(*joinNode_)) {
      indices = computeProbeUniqueInnerJoinIndices(
          leftTableView.select(leftKeyIndices_),
          rightTableView.select(rightKeyIndices_),
          stream);
    } else {
      indices = computeInnerJoinIndices(
          i, leftTableView.select(leftKeyIndices_), stream);
    }
    auto [leftJoinIndices, rightJoinIndices] = std::move(indices);

    auto leftIndicesSpan =
        cudf::device_span<cudf::size_type const>{*leftJoinIndices};
    auto rightIndicesSpan =
        cudf::device_span<cudf::size_type const>{*rightJoinIndices};
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};
    std::vector<std::unique_ptr<cudf::column>> joinedCols;

    if (joinNode_->filter()) {
      if (useAstFilter_) {
        cudfOutputs.push_back(filteredOutputIndices(
            leftTableView,
            leftIndicesCol,
            rightTableView,
            rightIndicesCol,
            extendedLeftView,
            extendedRightView,
            cudf::join_kind::INNER_JOIN,
            stream));
      } else {
        auto filterFunc =
            [stream](
                std::vector<std::unique_ptr<cudf::column>>&& joinedCols,
                cudf::column_view filterColumn) {
              auto filterTable =
                  std::make_unique<cudf::table>(std::move(joinedCols));
              auto filteredTable = cudf::apply_boolean_mask(
                  *filterTable, filterColumn, stream, get_output_mr());
              return filteredTable->release();
            };
        cudfOutputs.push_back(filteredOutput(
            leftTableView,
            leftIndicesCol,
            rightTableView,
            rightIndicesCol,
            filterFunc,
            stream));
      }
    } else {
      cudfOutputs.push_back(unfilteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          stream));
    }
  }
  return cudfOutputs;
}

std::vector<CudfHashJoinProbe::JoinOutput> CudfHashJoinProbe::leftJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<JoinOutput> cudfOutputs;

  auto& rightTables = hashObject_.value().buildTables;
  auto numProbeRows = leftTableView.num_rows();

  // Track which probe rows matched in any build batch so that unmatched probe
  // rows can be emitted with NULL build columns after the loop.
  ProbeMatchTracker probeTracker(numProbeRows, stream, get_temp_mr());

  // Precompute left (probe) table columns if needed (once, outside loop)
  std::vector<ColumnOrView> leftPrecomputed;
  cudf::table_view extendedLeftView = leftTableView;
  if (joinNode_->filter() && !leftPrecomputeInstructions_.empty()) {
    auto leftColumnViews = tableViewToColumnViews(leftTableView);
    leftPrecomputed = precomputeSubexpressions(
        leftColumnViews,
        leftPrecomputeInstructions_,
        scalars_,
        probeType_,
        stream);
    extendedLeftView = createExtendedTableView(leftTableView, leftPrecomputed);
  }

  // Processes a build batch of join indices: applies the filter (if any),
  // updates probeTracker from post-filter left indices, and appends the result
  // to cudfOutputs.
  auto processBatch = [&](cudf::column_view leftIndicesCol,
                          cudf::column_view rightIndicesCol,
                          cudf::table_view rightTableView,
                          size_t buildBatchIdx) {
    if (joinNode_->filter()) {
      cudf::table_view extendedRightView = !rightPrecomputeInstructions_.empty()
          ? cachedExtendedRightViews_[buildBatchIdx]
          : rightTableView;

      if (useAstFilter_) {
        // Inline filter_join_indices so we can access post-filter left indices
        // for match tracking.
        auto [filteredLeftJoinIndices, filteredRightJoinIndices] =
            cudf::filter_join_indices(
                extendedLeftView,
                extendedRightView,
                leftIndicesCol,
                rightIndicesCol,
                tree_.back(),
                cudf::join_kind::INNER_JOIN,
                stream,
                get_temp_mr());

        if (filteredLeftJoinIndices->size() > 0) {
          auto filteredLeftSpan = cudf::device_span<cudf::size_type const>{
              *filteredLeftJoinIndices};
          auto filteredRightSpan = cudf::device_span<cudf::size_type const>{
              *filteredRightJoinIndices};
          auto filteredLeftCol = cudf::column_view{filteredLeftSpan};
          auto filteredRightCol = cudf::column_view{filteredRightSpan};

          probeTracker.update(filteredLeftCol, stream, get_temp_mr());

          cudfOutputs.push_back(unfilteredOutput(
              leftTableView,
              filteredLeftCol,
              rightTableView,
              filteredRightCol,
              stream));
        }
      } else {
        auto leftIndicesSpanCopy =
            cudf::device_span<cudf::size_type const>(leftIndicesCol);
        auto filterFunc =
            [&probeTracker, leftIndicesSpanCopy, stream](
                std::vector<std::unique_ptr<cudf::column>>&& joinedCols,
                cudf::column_view filterColumn) {
              auto filterTable =
                  std::make_unique<cudf::table>(std::move(joinedCols));
              auto filteredTable = cudf::apply_boolean_mask(
                  *filterTable, filterColumn, stream, get_output_mr());
              joinedCols = filteredTable->release();

              // Filter left join indices with the same mask to track which
              // probe rows passed the filter.
              auto leftIdxCol = cudf::column_view{leftIndicesSpanCopy};
              auto filteredIdxTable = cudf::apply_boolean_mask(
                  cudf::table_view{std::vector<cudf::column_view>{leftIdxCol}},
                  filterColumn,
                  stream,
                  get_temp_mr());
              auto filteredLeftIdxCol =
                  std::move(filteredIdxTable->release()[0]);
              probeTracker.update(
                  filteredLeftIdxCol->view(), stream, get_temp_mr());

              return std::move(joinedCols);
            };
        cudfOutputs.push_back(filteredOutput(
            leftTableView,
            leftIndicesCol,
            rightTableView,
            rightIndicesCol,
            filterFunc,
            stream));
      }
    } else {
      probeTracker.update(leftIndicesCol, stream, get_temp_mr());
      cudfOutputs.push_back(unfilteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          stream));
    }
  };

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();

    // Use inner_join to get only real matched pairs. Unmatched probe rows are
    // emitted separately after the loop.
    auto [leftJoinIndices, rightJoinIndices] = computeInnerJoinIndices(
        i, leftTableView.select(leftKeyIndices_), stream);

    if (leftJoinIndices->size() == 0) {
      continue;
    }

    auto leftIndicesSpan =
        cudf::device_span<cudf::size_type const>{*leftJoinIndices};
    auto rightIndicesSpan =
        cudf::device_span<cudf::size_type const>{*rightJoinIndices};
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};

    processBatch(leftIndicesCol, rightIndicesCol, rightTableView, i);
  }

  // Emit unmatched probe rows with JoinNoMatch right indices so that gather
  // with NULLIFY produces NULL build columns.
  auto unmatchedIndices =
      probeTracker.getUnmatchedIndices(stream, get_temp_mr());

  if (unmatchedIndices->size() > 0) {
    auto unmatchedLeftCol = unmatchedIndices->view();
    auto sentinelScalar = cudf::numeric_scalar<cudf::size_type>(
        cudf::JoinNoMatch, true, stream, get_temp_mr());
    auto unmatchedRightIndices = cudf::make_column_from_scalar(
        sentinelScalar, unmatchedIndices->size(), stream, get_temp_mr());
    auto unmatchedRightCol = unmatchedRightIndices->view();

    // Emit unmatched rows directly via unfilteredOutput for two reasons:
    // (1) We cannot use filteredOutputIndices with LEFT_JOIN here because
    //     filter_join_indices(LEFT_JOIN) ensures every row in the left *table*
    //     appears at least once. Since our left indices are a subset of probe
    //     rows (only the unmatched ones), LEFT_JOIN would re-add all the
    //     matched probe rows that are absent from this subset.
    // (2) Using unfilteredOutput is safe because all right indices are
    //     JoinNoMatch. Per filter_join_indices semantics, input pairs with
    //     JoinNoMatch in either position pass through unchanged (the predicate
    //     cannot be evaluated), so filtering would be a no-op anyway.
    cudfOutputs.push_back(unfilteredOutput(
        leftTableView,
        unmatchedLeftCol,
        rightTables[0]->view(),
        unmatchedRightCol,
        stream));
  }

  return cudfOutputs;
}

std::vector<CudfHashJoinProbe::JoinOutput> CudfHashJoinProbe::rightJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<JoinOutput> cudfOutputs;

  auto& rightTables = hashObject_.value().buildTables;

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();

    auto [leftJoinIndices, rightJoinIndices] = computeInnerJoinIndices(
        i, leftTableView.select(leftKeyIndices_), stream);
    if (!joinNode_->filter()) {
      // Mark matched build rows by checking which row indices appear in
      // rightJoinIndices. Use contains to avoid scatter with duplicate indices.
      auto rightIdxCol = cudf::column_view{
          cudf::device_span<cudf::size_type const>{*rightJoinIndices}};

      // Create sequence [0, 1, ..., n-1] for build table row indices
      auto n = rightTableView.num_rows();
      auto rowIndices = cudf::sequence(
          n,
          cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
          cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
          stream,
          get_temp_mr());

      // Check which build row indices are present in the join result
      auto matchedInBatch = cudf::contains(
          rightIdxCol, rowIndices->view(), stream, get_temp_mr());

      // OR with existing flags to accumulate matches across batches
      auto updatedFlags = cudf::binary_operation(
          rightMatchedFlags_[i]->view(),
          matchedInBatch->view(),
          cudf::binary_operator::BITWISE_OR,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          get_temp_mr());
      // binary_operation is async on `stream`; the old column destructs via
      // cudaFreeAsync on its allocation stream (not `stream`), so the free
      // can race the kernel. Drain `stream` before the move-assign.
      stream.synchronize();
      rightMatchedFlags_[i] = std::move(updatedFlags);
    }

    auto leftIndicesSpan =
        cudf::device_span<cudf::size_type const>{*leftJoinIndices};
    auto rightIndicesSpan =
        cudf::device_span<cudf::size_type const>{*rightJoinIndices};
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};
    std::vector<std::unique_ptr<cudf::column>> joinedCols;

    if (joinNode_->filter()) {
      auto& rightMatchedFlags = rightMatchedFlags_[i];
      auto numBuildRows = rightTableView.num_rows();
      auto filterFunc =
          [&rightMatchedFlags, rightIndicesSpan, numBuildRows, stream](
              std::vector<std::unique_ptr<cudf::column>>&& joinedCols,
              cudf::column_view filterColumn) {
            // apply the filter
            auto filterTable =
                std::make_unique<cudf::table>(std::move(joinedCols));
            auto filteredTable = cudf::apply_boolean_mask(
                *filterTable, filterColumn, stream, get_output_mr());
            joinedCols = filteredTable->release();

            // For streaming right join, after applying filter, we record
            // matched right indices filter rightJoinIndices with the same mask
            // to update matched flags
            auto rightIdxCol = cudf::column_view{rightIndicesSpan};
            auto filteredIdxTable = cudf::apply_boolean_mask(
                cudf::table_view{std::vector<cudf::column_view>{rightIdxCol}},
                filterColumn,
                stream,
                get_temp_mr());
            auto filteredCols = filteredIdxTable->release();
            auto filteredRightIdxCol = std::move(filteredCols[0]);

            // Use contains to check which build row indices passed the filter
            auto rowIndices = cudf::sequence(
                numBuildRows,
                cudf::numeric_scalar<cudf::size_type>(
                    0, true, stream, get_temp_mr()),
                cudf::numeric_scalar<cudf::size_type>(
                    1, true, stream, get_temp_mr()),
                stream,
                get_temp_mr());

            auto matchedInBatch = cudf::contains(
                filteredRightIdxCol->view(),
                rowIndices->view(),
                stream,
                get_temp_mr());

            // OR with existing flags to accumulate matches across batches
            auto updatedFlags = cudf::binary_operation(
                rightMatchedFlags->view(),
                matchedInBatch->view(),
                cudf::binary_operator::BITWISE_OR,
                cudf::data_type{cudf::type_id::BOOL8},
                stream,
                get_temp_mr());
            // binary_operation is async on `stream`; the old column destructs
            // via cudaFreeAsync on its allocation stream (not `stream`), so the
            // free can race the kernel. Drain `stream` before the move-assign.
            stream.synchronize();
            rightMatchedFlags = std::move(updatedFlags);
            return std::move(joinedCols);
          };
      cudfOutputs.push_back(filteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          filterFunc,
          stream));
    } else {
      cudfOutputs.push_back(unfilteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          stream));
    }
  }
  return cudfOutputs;
}

std::vector<CudfHashJoinProbe::JoinOutput> CudfHashJoinProbe::fullJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<JoinOutput> cudfOutputs;

  auto& rightTables = hashObject_.value().buildTables;
  auto numProbeRows = leftTableView.num_rows();

  // For now, AST support is necessary to filter join output
  if (joinNode_->filter() && !useAstFilter_) {
    VELOX_NYI("Full join requires AST support for filtering");
  }

  // Track which probe rows matched in any build batch so that unmatched probe
  // rows can be emitted with NULL build columns after the loop.
  ProbeMatchTracker probeTracker(numProbeRows, stream, get_temp_mr());

  // Helper to accumulate build-side (right) match flags for a build batch.
  auto updateRightMatchedFlags = [&](size_t batchIdx,
                                     cudf::column_view matchedRightIndices,
                                     cudf::size_type numBuildRows) {
    auto buildRowIndices = cudf::sequence(
        numBuildRows,
        cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
        cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
        stream,
        get_temp_mr());
    auto matchedInBatch = cudf::contains(
        matchedRightIndices, buildRowIndices->view(), stream, get_temp_mr());
    auto updatedFlags = cudf::binary_operation(
        rightMatchedFlags_[batchIdx]->view(),
        matchedInBatch->view(),
        cudf::binary_operator::BITWISE_OR,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        get_temp_mr());
    stream.synchronize();
    rightMatchedFlags_[batchIdx] = std::move(updatedFlags);
  };

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();

    // Use inner_join to get only real matched pairs. Unmatched probe rows are
    // emitted separately after the loop. Unmatched build rows are emitted in
    // doGetOutput via rightMatchedFlags_.
    auto [leftJoinIndices, rightJoinIndices] = computeInnerJoinIndices(
        i, leftTableView.select(leftKeyIndices_), stream);

    if (leftJoinIndices->size() == 0) {
      continue;
    }

    auto leftIndicesSpan =
        cudf::device_span<cudf::size_type const>{*leftJoinIndices};
    auto rightIndicesSpan =
        cudf::device_span<cudf::size_type const>{*rightJoinIndices};
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};

    if (joinNode_->filter()) {
      // Apply filter and keep only pairs where the predicate passes.
      auto [filteredLeftJoinIndices, filteredRightJoinIndices] =
          cudf::filter_join_indices(
              leftTableView,
              rightTableView,
              leftIndicesCol,
              rightIndicesCol,
              tree_.back(),
              cudf::join_kind::INNER_JOIN,
              stream,
              get_temp_mr());

      if (filteredLeftJoinIndices->size() > 0) {
        auto filteredLeftSpan =
            cudf::device_span<cudf::size_type const>{*filteredLeftJoinIndices};
        auto filteredRightSpan =
            cudf::device_span<cudf::size_type const>{*filteredRightJoinIndices};
        auto filteredLeftCol = cudf::column_view{filteredLeftSpan};
        auto filteredRightCol = cudf::column_view{filteredRightSpan};

        probeTracker.update(filteredLeftCol, stream, get_temp_mr());
        updateRightMatchedFlags(i, filteredRightCol, rightTableView.num_rows());

        cudfOutputs.push_back(unfilteredOutput(
            leftTableView,
            filteredLeftCol,
            rightTableView,
            filteredRightCol,
            stream));
      }
    } else {
      probeTracker.update(leftIndicesCol, stream, get_temp_mr());
      updateRightMatchedFlags(i, rightIndicesCol, rightTableView.num_rows());

      cudfOutputs.push_back(unfilteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          stream));
    }
  }

  // Emit unmatched probe rows with JoinNoMatch right indices so that gather
  // with NULLIFY produces NULL build columns.
  auto unmatchedIndices =
      probeTracker.getUnmatchedIndices(stream, get_temp_mr());

  if (unmatchedIndices->size() > 0) {
    auto unmatchedLeftCol = unmatchedIndices->view();
    auto sentinelScalar = cudf::numeric_scalar<cudf::size_type>(
        cudf::JoinNoMatch, true, stream, get_temp_mr());
    auto unmatchedRightIndices = cudf::make_column_from_scalar(
        sentinelScalar, unmatchedIndices->size(), stream, get_temp_mr());
    auto unmatchedRightCol = unmatchedRightIndices->view();

    // Use unfilteredOutput directly — see the matching comment in leftJoin()
    // for why filteredOutputIndices cannot be used here.
    cudfOutputs.push_back(unfilteredOutput(
        leftTableView,
        unmatchedLeftCol,
        rightTables[0]->view(),
        unmatchedRightCol,
        stream));
  }

  return cudfOutputs;
}

std::vector<CudfHashJoinProbe::JoinOutput>
CudfHashJoinProbe::leftSemiFilterJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<JoinOutput> cudfOutputs;

  auto& rightTables = hashObject_.value().buildTables;

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> leftJoinIndices;

    if (joinNode_->filter()) {
      leftJoinIndices = cudf::mixed_left_semi_join(
          leftTableView.select(leftKeyIndices_),
          rightTableView.select(rightKeyIndices_),
          leftTableView,
          rightTableView,
          tree_.back(),
          cudf::null_equality::UNEQUAL,
          stream,
          get_temp_mr());
    } else {
      cudf::filtered_join filter_join(
          rightTableView.select(rightKeyIndices_),
          cudf::null_equality::UNEQUAL,
          stream);
      leftJoinIndices = filter_join.semi_join(
          leftTableView.select(leftKeyIndices_), stream, get_temp_mr());
    }

    auto leftIndicesSpan =
        cudf::device_span<cudf::size_type const>{*leftJoinIndices};
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::empty_like(leftIndicesCol);

    cudfOutputs.push_back(unfilteredOutput(
        leftTableView,
        leftIndicesCol,
        rightTableView,
        rightIndicesCol->view(),
        stream));
  }
  return cudfOutputs;
}

namespace {
/// Creates a boolean column indicating which rows have NULL in ANY key column.
/// Returns a column where row[i] = true if ANY key column is NULL at row i.
std::unique_ptr<cudf::column> createProbeKeyNullMask(
    cudf::table_view keyView,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto numRows = keyView.num_rows();

  if (keyView.num_columns() == 0 || numRows == 0) {
    auto falseScalar = cudf::numeric_scalar<bool>(false, true, stream, mr);
    return cudf::make_column_from_scalar(falseScalar, numRows, stream, mr);
  }

  // Start with first column's null mask
  auto result = cudf::is_null(keyView.column(0), stream, mr);

  // OR with other columns' null masks
  for (cudf::size_type i = 1; i < keyView.num_columns(); i++) {
    auto colIsNull = cudf::is_null(keyView.column(i), stream, mr);
    result = cudf::binary_operation(
        result->view(),
        colIsNull->view(),
        cudf::binary_operator::BITWISE_OR,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
  }
  return result;
}

/// Applies a null mask to a boolean column.
/// Where nullMask[i] is true, result[i] becomes NULL.
/// Where nullMask[i] is false, result[i] keeps its original value from col.
std::unique_ptr<cudf::column> applyNullMask(
    cudf::column_view col,
    cudf::column_view nullMask,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Create a null scalar (valid=false means NULL)
  auto nullScalar = cudf::numeric_scalar<bool>(false, false, stream, mr);

  // copy_if_else: where nullMask is true, use nullScalar (NULL); else use col
  // value
  return cudf::copy_if_else(nullScalar, col, nullMask, stream, mr);
}

/// Get row indices where mask is true.
/// Returns a column of size_type indices.
std::unique_ptr<cudf::column> getIndicesWhere(
    cudf::column_view mask,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Create sequence [0, 1, 2, ..., mask.size()-1]
  auto seq = cudf::sequence(
      mask.size(),
      cudf::numeric_scalar<cudf::size_type>(0, true, stream, mr),
      cudf::numeric_scalar<cudf::size_type>(1, true, stream, mr),
      stream,
      mr);

  // Filter to keep only indices where mask is true
  auto indicesTable = cudf::apply_boolean_mask(
      cudf::table_view{{seq->view()}}, mask, stream, mr);

  return std::move(indicesTable->release()[0]);
}

std::unique_ptr<cudf::column> scatterTrueAtIndices(
    std::unique_ptr<cudf::column> target,
    cudf::column_view indices,
    std::vector<std::unique_ptr<cudf::column>>& retainedTargets,
    std::vector<std::unique_ptr<cudf::scalar>>& retainedScalars,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (indices.size() == 0) {
    return target;
  }

  auto trueScalar =
      std::make_unique<cudf::numeric_scalar<bool>>(true, true, stream, mr);
  std::vector<std::reference_wrapper<const cudf::scalar>> sources = {
      *trueScalar};
  auto scattered = cudf::scatter(
      sources, indices, cudf::table_view{{target->view()}}, stream, mr);
  retainedScalars.push_back(std::move(trueScalar));
  retainedTargets.push_back(std::move(target));
  return std::move(scattered->release()[0]);
}

/// Create cross-product of two index columns.
/// Given left = [a, b, c] and right = [x, y], produces:
///   leftOut = [a, a, b, b, c, c]
///   rightOut = [x, y, x, y, x, y]
/// Uses cudf::repeat for left (repeat each element) and cudf::tile for right.
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
createCrossProductIndices(
    cudf::column_view leftIndices,
    cudf::column_view rightIndices,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto numLeft = leftIndices.size();
  auto numRight = rightIndices.size();

  if (numLeft == 0 || numRight == 0) {
    // Return empty columns
    auto emptyLeft = cudf::make_empty_column(cudf::type_id::INT32);
    auto emptyRight = cudf::make_empty_column(cudf::type_id::INT32);
    return {std::move(emptyLeft), std::move(emptyRight)};
  }

  // Repeat each left element numRight times: [a,a,b,b,c,c]
  auto leftRepeated =
      cudf::repeat(cudf::table_view{{leftIndices}}, numRight, stream, mr);

  // Tile the right indices numLeft times: [x,y,x,y,x,y]
  auto rightTiled =
      cudf::tile(cudf::table_view{{rightIndices}}, numLeft, stream, mr);

  return {
      std::move(leftRepeated->release()[0]),
      std::move(rightTiled->release()[0])};
}

} // namespace

std::vector<CudfHashJoinProbe::JoinOutput>
CudfHashJoinProbe::leftSemiProjectMinMaxJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(simpleNotEqualLeftIndex_.has_value());
  VELOX_CHECK(simpleNotEqualRightIndex_.has_value());
  VELOX_CHECK_EQ(leftKeyIndices_.size(), 1);
  VELOX_CHECK_EQ(rightKeyIndices_.size(), 1);

  auto& state = hashObject_.value();
  auto& rightTables = state.buildTables;
  auto numProbeRows = leftTableView.num_rows();

  auto falseScalar =
      cudf::numeric_scalar<bool>(false, true, stream, get_output_mr());
  auto matchCol = cudf::make_column_from_scalar(
      falseScalar, numProbeRows, stream, get_output_mr());

  auto const leftKeyView = leftTableView.select(leftKeyIndices_);
  auto const& leftValue =
      leftTableView.column(simpleNotEqualLeftIndex_.value());
  auto const summaryKeyIndices =
      sequenceIndices(static_cast<cudf::size_type>(rightKeyIndices_.size()));

  for (auto i = 0; i < rightTables.size(); i++) {
    auto summaryTableView = rightTables[i]->view();
    if (summaryTableView.num_rows() == 0) {
      continue;
    }

    VELOX_CHECK_NOT_NULL(
        state.distinctHashJoins[i],
        "Expected prebuilt distinct hash join for Q21 min/max summary table");
    auto rightJoinIndices = state.distinctHashJoins[i]->left_join(
        leftKeyView, stream, get_temp_mr());
    auto rightIndicesSpan =
        cudf::device_span<cudf::size_type const>{*rightJoinIndices};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};
    auto const minValue = summaryTableView.column(summaryKeyIndices.size());
    auto const maxValue = summaryTableView.column(summaryKeyIndices.size() + 1);

    auto gatheredMinTable = cudf::gather(
        cudf::table_view{{minValue}},
        rightIndicesCol,
        oobPolicy,
        stream,
        get_temp_mr());
    auto gatheredMaxTable = cudf::gather(
        cudf::table_view{{maxValue}},
        rightIndicesCol,
        oobPolicy,
        stream,
        get_temp_mr());
    auto gatheredMinColumns = gatheredMinTable->release();
    auto gatheredMaxColumns = gatheredMaxTable->release();
    VELOX_CHECK_EQ(gatheredMinColumns.size(), 1);
    VELOX_CHECK_EQ(gatheredMaxColumns.size(), 1);

    auto minDiffers = cudf::binary_operation(
        leftValue,
        gatheredMinColumns[0]->view(),
        cudf::binary_operator::NOT_EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        get_temp_mr());
    auto maxDiffers = cudf::binary_operation(
        leftValue,
        gatheredMaxColumns[0]->view(),
        cudf::binary_operator::NOT_EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        get_temp_mr());
    auto batchMatchNullable = cudf::binary_operation(
        minDiffers->view(),
        maxDiffers->view(),
        cudf::binary_operator::BITWISE_OR,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        get_temp_mr());
    auto batchMatch = cudf::replace_nulls(
        batchMatchNullable->view(), falseScalar, stream, get_temp_mr());
    auto updatedMatch = cudf::binary_operation(
        matchCol->view(),
        batchMatch->view(),
        cudf::binary_operator::BITWISE_OR,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        get_output_mr());

    matchCol = std::move(updatedMatch);
  }

  std::vector<std::unique_ptr<cudf::column>> outputCols;
  outputCols.resize(outputType_->names().size());

  auto leftInput = leftTableView.select(leftColumnIndicesToGather_);
  for (size_t i = 0; i < leftColumnIndicesToGather_.size(); i++) {
    outputCols[leftColumnOutputIndices_[i]] = std::make_unique<cudf::column>(
        leftInput.column(i), stream, get_output_mr());
  }
  outputCols.back() = std::move(matchCol);

  if (buildStream_.has_value()) {
    cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
  }
  stream.synchronize();

  std::vector<JoinOutput> cudfOutputs;
  cudfOutputs.push_back(
      {std::make_unique<cudf::table>(std::move(outputCols)),
       static_cast<vector_size_t>(numProbeRows)});
  return cudfOutputs;
}

// LEFT SEMI PROJECT returns all probe rows with a boolean "match" column
// indicating whether each probe row has at least one matching build row
// (that also passes the filter, if specified). Unlike LEFT SEMI FILTER
// which filters out non-matching rows, this preserves all probe rows.
// Output cardinality always equals probe side cardinality.
//
// Implementation approach:
// 1. Without a join filter, build cudf::mark_join on the current probe batch
//    keys and probe it with each build-side batch to get matching probe rows.
// 2. With a simple cross-side non-equality filter, use a per-key min/max
//    summary of the build side to mark matching probe rows.
// 3. With other join filters, use the pair-index path because filter
//    evaluation needs both probe and build row indices.
// 4. Accumulate matches across build table batches by setting TRUE for every
//    marked probe row.
// 5. For null-aware mode (without filter): apply null mask based on probe key
//    nullity and build side null keys presence.
// 6. Output: all probe columns + match column.
std::vector<CudfHashJoinProbe::JoinOutput>
CudfHashJoinProbe::leftSemiProjectJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<JoinOutput> cudfOutputs;

  // For now, AST support is necessary to filter join output
  if (joinNode_->filter() && !useAstFilter_) {
    VELOX_NYI("Left semi project join requires AST support for filtering");
  }

  auto& rightTables = hashObject_.value().buildTables;
  auto numProbeRows = leftTableView.num_rows();

  const bool isNullAware = joinNode_->isNullAware();
  const bool hasFilter = joinNode_->filter() != nullptr;
  // For null-aware without filter, we use a different code path (existing)
  // For null-aware with filter, we need to compute indeterminate cases
  const bool isNullAwareWithFilter = isNullAware && hasFilter;
  const bool isNullAwareWithoutFilter = isNullAware && !hasFilter;

  // Q21 EXISTS semijoins arrive as null-aware left semi project joins. The
  // simple cross-side NOT_EQUAL filter still produces a boolean EXISTS marker:
  // NULL comparisons do not match and are folded to false in the min/max path.
  if (hasFilter && simpleNotEqualLeftIndex_.has_value() &&
      simpleNotEqualRightIndex_.has_value()) {
    return leftSemiProjectMinMaxJoin(leftTableView, stream);
  }

  // Initialize match column to all false
  auto falseScalar =
      cudf::numeric_scalar<bool>(false, true, stream, get_output_mr());
  auto matchCol = cudf::make_column_from_scalar(
      falseScalar, numProbeRows, stream, get_output_mr());

  // Create probe row indices sequence: [0, 1, 2, ..., numProbeRows-1].
  // This is only needed by the filtered path and null-aware filtered logic.
  std::unique_ptr<cudf::column> probeRowIndices;
  if (hasFilter) {
    probeRowIndices = cudf::sequence(
        numProbeRows,
        cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
        cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
        stream,
        get_temp_mr());
  }

  // Precompute left (probe) table columns if needed for filter
  std::vector<ColumnOrView> leftPrecomputed;
  cudf::table_view extendedLeftView = leftTableView;
  if (joinNode_->filter() && !leftPrecomputeInstructions_.empty()) {
    auto leftColumnViews = tableViewToColumnViews(leftTableView);
    leftPrecomputed = precomputeSubexpressions(
        leftColumnViews,
        leftPrecomputeInstructions_,
        scalars_,
        probeType_,
        stream);
    extendedLeftView = createExtendedTableView(leftTableView, leftPrecomputed);
  }

  auto const leftKeyView = leftTableView.select(leftKeyIndices_);
  std::unique_ptr<cudf::mark_join> markJoin;
  std::vector<std::unique_ptr<cudf::column>> retainedMatchColumns;
  std::vector<std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
      retainedMatchIndices;
  std::vector<std::unique_ptr<cudf::scalar>> retainedScatterScalars;
  if (!hasFilter) {
    markJoin = std::make_unique<cudf::mark_join>(
        leftKeyView,
        cudf::null_equality::UNEQUAL,
        cudf::join_prefilter::NO,
        stream);
    // Scatter reads all inputs asynchronously. Keep old match columns, scatter
    // maps, and scalar sources alive until the final stream synchronization
    // below.
    retainedMatchColumns.reserve(rightTables.size());
    retainedMatchIndices.reserve(rightTables.size());
    retainedScatterScalars.reserve(rightTables.size());
  } else {
    retainedMatchIndices.reserve(rightTables.size());
    retainedMatchColumns.reserve(rightTables.size());
    retainedScatterScalars.reserve(rightTables.size());
  }

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();

    if (!hasFilter) {
      auto matchedProbeIndices = markJoin->semi_join(
          rightTableView.select(rightKeyIndices_), stream, get_temp_mr());
      if (matchedProbeIndices->size() == 0) {
        continue;
      }
      retainedMatchIndices.push_back(std::move(matchedProbeIndices));
      auto matchedProbeIndicesSpan = cudf::device_span<cudf::size_type const>{
          *retainedMatchIndices.back()};
      auto matchedProbeIndicesCol = cudf::column_view{matchedProbeIndicesSpan};
      matchCol = scatterTrueAtIndices(
          std::move(matchCol),
          matchedProbeIndicesCol,
          retainedMatchColumns,
          retainedScatterScalars,
          stream,
          get_output_mr());
      continue;
    }

    // Use cached precomputed columns for right (build) table
    cudf::table_view extendedRightView = (!rightPrecomputeInstructions_.empty())
        ? cachedExtendedRightViews_[i]
        : rightTableView;

    // Step 1: Inner join to get (probe_idx, build_idx) pairs where keys match.
    // Unlike left_join, inner_join only returns valid pairs (no JoinNoMatch).
    auto [leftJoinIndices, rightJoinIndices] = computeInnerJoinIndices(
        i, leftTableView.select(leftKeyIndices_), stream);

    if (leftJoinIndices->size() == 0) {
      continue; // No matches from this build table
    }

    auto leftIndicesSpan =
        cudf::device_span<cudf::size_type const>{*leftJoinIndices};
    auto rightIndicesSpan =
        cudf::device_span<cudf::size_type const>{*rightJoinIndices};

    // Step 2: Apply filter to the join pairs. INNER_JOIN mode keeps only
    // pairs where the predicate evaluates to true.
    auto filteredJoinIndices = cudf::filter_join_indices(
        extendedLeftView,
        extendedRightView,
        leftIndicesSpan,
        rightIndicesSpan,
        tree_.back(),
        cudf::join_kind::INNER_JOIN,
        stream,
        get_temp_mr());

    auto filteredLeftIndices = std::move(filteredJoinIndices.first);
    if (filteredLeftIndices->size() == 0) {
      continue; // No matches passed filter
    }
    auto filteredLeftSpan =
        cudf::device_span<cudf::size_type const>{*filteredLeftIndices};
    auto matchedProbeIndices = cudf::column_view{filteredLeftSpan};

    // Step 3: Create match flags using cudf::contains. For each probe row index
    // in [0, numProbeRows), check if it appears in matchedProbeIndices.
    // This handles duplicates correctly - if a probe row matches multiple build
    // rows, it appears multiple times in matchedProbeIndices, but contains()
    // returns true if it appears at least once.
    auto matchedInBatch = cudf::contains(
        matchedProbeIndices, probeRowIndices->view(), stream, get_temp_mr());

    // Step 4: Accumulate matches across build table batches using OR.
    // A probe row's final match value is true if it matched in ANY batch.
    auto updatedMatch = cudf::binary_operation(
        matchCol->view(),
        matchedInBatch->view(),
        cudf::binary_operator::BITWISE_OR,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        get_output_mr());
    stream.synchronize();
    matchCol = std::move(updatedMatch);
  }

  // Step 5: Handle null-aware semantics (IN vs EXISTS).
  // For null-aware mode, we need to compute three-valued logic:
  // - TRUE: at least one match passes filter
  // - FALSE: no match passes filter AND no indeterminate cases
  // - NULL: probe key is NULL, OR (no match AND build has null keys that might
  // match)

  if (isNullAwareWithFilter) {
    // Null-aware LEFT SEMI PROJECT with filter implements SQL IN semantics:
    //   SELECT t0 IN (SELECT u0 FROM u WHERE filter) FROM t
    //
    // "Indeterminate" means the result should be NULL (unknown) rather than
    // FALSE. This happens when we cannot definitively say the probe value is
    // NOT IN the subquery because NULL comparisons are involved:
    //
    // - Type B (null probe key): When probe key is NULL, we can't determine
    //   if NULL equals any subquery value. If the filter passes for ANY build
    //   row, result is NULL (might match). If filter fails for ALL build rows,
    //   the subquery is empty, so result is FALSE.
    //
    // - Type A (non-null probe key, no match): When probe key doesn't match
    //   any non-NULL build key, but build has NULL keys where filter passes,
    //   we can't rule out a match (NULL might equal our probe key), so result
    //   is NULL.
    //
    // We evaluate these by creating synthetic (probe, build) index pairs and
    // running the filter to see if any pair passes.

    // Lambda to create device_span from a column
    auto toSpan = [](cudf::column_view col) {
      return cudf::device_span<cudf::size_type const>{
          static_cast<cudf::size_type const*>(col.head()),
          static_cast<size_t>(col.size())};
    };

    // Lambda to run filter on synthetic pairs and accumulate indeterminate
    // flags. Creates cross-product of probeIndices × buildIndices, runs the
    // filter, and ORs any passing probe rows into indeterminateCol.
    auto accumulateIndeterminate =
        [&](cudf::column_view probeIndices,
            cudf::column_view buildIndices,
            cudf::table_view extendedRight,
            std::unique_ptr<cudf::column>& indeterminateCol) {
          if (probeIndices.size() == 0 || buildIndices.size() == 0) {
            return;
          }

          auto [syntheticLeft, syntheticRight] = createCrossProductIndices(
              probeIndices, buildIndices, stream, get_temp_mr());

          if (syntheticLeft->size() == 0) {
            return;
          }

          auto [filteredLeft, filteredRight] = cudf::filter_join_indices(
              extendedLeftView,
              extendedRight,
              toSpan(syntheticLeft->view()),
              toSpan(syntheticRight->view()),
              tree_.back(),
              cudf::join_kind::INNER_JOIN,
              stream,
              get_temp_mr());

          if (filteredLeft->size() == 0) {
            return;
          }

          auto filteredLeftSpan =
              cudf::device_span<cudf::size_type const>{*filteredLeft};
          auto filteredLeftCol = cudf::column_view{filteredLeftSpan};
          auto indeterminate = cudf::contains(
              filteredLeftCol, probeRowIndices->view(), stream, get_temp_mr());

          indeterminateCol = cudf::binary_operation(
              indeterminateCol->view(),
              indeterminate->view(),
              cudf::binary_operator::BITWISE_OR,
              cudf::data_type{cudf::type_id::BOOL8},
              stream,
              get_temp_mr());
        };

    bool buildSideEmpty = true;
    for (const auto& rt : rightTables) {
      if (rt->num_rows() > 0) {
        buildSideEmpty = false;
        break;
      }
    }

    // For empty build side, IN returns FALSE (already set in matchCol).
    if (!buildSideEmpty) {
      auto probeKeyView = leftTableView.select(leftKeyIndices_);
      bool probeHasNulls = cudf::has_nulls(probeKeyView);

      // Compute probe key null mask upfront
      auto probeKeyNullMask =
          createProbeKeyNullMask(probeKeyView, stream, get_temp_mr());

      // Initialize indeterminate column to all false
      auto falseScalar =
          cudf::numeric_scalar<bool>(false, true, stream, get_temp_mr());
      auto indeterminateCol = cudf::make_column_from_scalar(
          falseScalar, numProbeRows, stream, get_temp_mr());

      // Process each build batch for indeterminate cases
      for (size_t i = 0; i < rightTables.size(); i++) {
        auto rightTableView = rightTables[i]->view();
        auto buildKeyView = rightTableView.select(rightKeyIndices_);
        bool buildBatchHasNullKeys = cudf::has_nulls(buildKeyView);
        auto numBuildRows = rightTableView.num_rows();

        if (numBuildRows == 0) {
          continue;
        }

        // Get extended views for filter evaluation
        cudf::table_view extendedRightView =
            (!rightPrecomputeInstructions_.empty())
            ? cachedExtendedRightViews_[i]
            : rightTableView;

        // Type B: Null probe keys × all build rows
        if (probeHasNulls) {
          auto nullProbeIndices = getRetainedIndices(
              probeKeyNullMask->view(), stream, get_temp_mr());
          auto allBuildIndices = cudf::sequence(
              numBuildRows,
              cudf::numeric_scalar<cudf::size_type>(
                  0, true, stream, get_temp_mr()),
              cudf::numeric_scalar<cudf::size_type>(
                  1, true, stream, get_temp_mr()),
              stream,
              get_temp_mr());

          accumulateIndeterminate(
              nullProbeIndices->view(),
              allBuildIndices->view(),
              extendedRightView,
              indeterminateCol);
        }

        // Type A: Non-null, non-matching probe keys × null-key build rows
        if (buildBatchHasNullKeys) {
          auto notProbeNull = cudf::unary_operation(
              probeKeyNullMask->view(),
              cudf::unary_operator::NOT,
              stream,
              get_temp_mr());
          auto noMatch = cudf::unary_operation(
              matchCol->view(),
              cudf::unary_operator::NOT,
              stream,
              get_temp_mr());
          auto typeAMask = cudf::binary_operation(
              notProbeNull->view(),
              noMatch->view(),
              cudf::binary_operator::BITWISE_AND,
              cudf::data_type{cudf::type_id::BOOL8},
              stream,
              get_temp_mr());

          auto typeAProbeIndices =
              getRetainedIndices(typeAMask->view(), stream, get_temp_mr());
          auto buildKeyNullMask =
              createProbeKeyNullMask(buildKeyView, stream, get_temp_mr());
          auto nullBuildIndices = getRetainedIndices(
              buildKeyNullMask->view(), stream, get_temp_mr());

          accumulateIndeterminate(
              typeAProbeIndices->view(),
              nullBuildIndices->view(),
              extendedRightView,
              indeterminateCol);
        }
      }

      // Apply three-valued logic:
      // - Where matchCol is TRUE → keep TRUE (takes precedence)
      // - Where matchCol is FALSE and indeterminateCol is TRUE → set NULL
      // - Where matchCol is FALSE and indeterminateCol is FALSE → keep FALSE
      auto notMatch = cudf::unary_operation(
          matchCol->view(), cudf::unary_operator::NOT, stream, get_temp_mr());
      auto shouldBeNull = cudf::binary_operation(
          notMatch->view(),
          indeterminateCol->view(),
          cudf::binary_operator::BITWISE_AND,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          get_temp_mr());

      matchCol = applyNullMask(
          matchCol->view(), shouldBeNull->view(), stream, get_output_mr());
    }
  } else if (isNullAwareWithoutFilter) {
    // Original null-aware without filter logic
    bool buildSideEmpty = true;
    for (const auto& rt : rightTables) {
      if (rt->num_rows() > 0) {
        buildSideEmpty = false;
        break;
      }
    }

    // For empty build side, IN returns FALSE (already set in matchCol).
    if (!buildSideEmpty) {
      auto probeKeyView = leftTableView.select(leftKeyIndices_);
      bool probeHasNulls = cudf::has_nulls(probeKeyView);

      if (probeHasNulls || buildSideHasNullKeys_) {
        // Compute null mask: true where result should be NULL
        auto probeKeyNullMask =
            createProbeKeyNullMask(probeKeyView, stream, get_temp_mr());

        std::unique_ptr<cudf::column> nullMask;
        if (buildSideHasNullKeys_) {
          // NULL where: probe key is NULL OR no match
          auto noMatchMask = cudf::unary_operation(
              matchCol->view(),
              cudf::unary_operator::NOT,
              stream,
              get_temp_mr());
          nullMask = cudf::binary_operation(
              probeKeyNullMask->view(),
              noMatchMask->view(),
              cudf::binary_operator::BITWISE_OR,
              cudf::data_type{cudf::type_id::BOOL8},
              stream,
              get_temp_mr());
        } else {
          // NULL only where probe key is NULL
          nullMask = std::move(probeKeyNullMask);
        }

        matchCol = applyNullMask(
            matchCol->view(), nullMask->view(), stream, get_output_mr());
      }
    }
  }

  // Step 6: Build output table with all probe columns + match column
  std::vector<std::unique_ptr<cudf::column>> outputCols;
  outputCols.resize(outputType_->names().size());

  // Copy probe columns
  auto leftInput = leftTableView.select(leftColumnIndicesToGather_);
  for (size_t i = 0; i < leftColumnIndicesToGather_.size(); i++) {
    outputCols[leftColumnOutputIndices_[i]] = std::make_unique<cudf::column>(
        leftInput.column(i), stream, get_output_mr());
  }

  // Add match column as the last column
  outputCols.back() = std::move(matchCol);

  if (buildStream_.has_value()) {
    cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
  }
  stream.synchronize();

  auto output = std::make_unique<cudf::table>(std::move(outputCols));
  cudfOutputs.push_back(
      {std::move(output), static_cast<vector_size_t>(numProbeRows)});
  return cudfOutputs;
}

std::vector<CudfHashJoinProbe::JoinOutput>
CudfHashJoinProbe::rightSemiFilterJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<JoinOutput> cudfOutputs;

  auto& rightTables = hashObject_.value().buildTables;
  auto rightTableView = rightTables[0]->view();

  VELOX_CHECK_EQ(
      rightTables.size(),
      1,
      "Multiple right tables not yet supported for rightSemiFilterJoin");

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> rightJoinIndices;
  if (joinNode_->filter()) {
    rightJoinIndices = cudf::mixed_left_semi_join(
        rightTableView.select(rightKeyIndices_),
        leftTableView.select(leftKeyIndices_),
        rightTableView,
        leftTableView,
        tree_.back(),
        cudf::null_equality::UNEQUAL,
        stream,
        get_temp_mr());
  } else {
    cudf::filtered_join filter_join(
        leftTableView.select(leftKeyIndices_),
        cudf::null_equality::UNEQUAL,
        stream);
    rightJoinIndices = filter_join.semi_join(
        rightTableView.select(rightKeyIndices_), stream, get_temp_mr());
  }

  auto rightIndicesSpan =
      cudf::device_span<cudf::size_type const>{*rightJoinIndices};
  auto rightIndicesCol = cudf::column_view{rightIndicesSpan};
  auto leftIndicesCol = cudf::empty_like(rightIndicesCol);
  cudfOutputs.push_back(unfilteredOutput(
      leftTableView,
      leftIndicesCol->view(),
      rightTableView,
      rightIndicesCol,
      stream));

  return cudfOutputs;
}

std::vector<CudfHashJoinProbe::JoinOutput> CudfHashJoinProbe::antiJoin(
    cudf::table_view leftTableViewParam,
    rmm::cuda_stream_view stream) {
  std::vector<JoinOutput> cudfOutputs;
  auto& rightTables = hashObject_.value().buildTables;

  VELOX_CHECK_EQ(
      rightTables.size(),
      1,
      "Multiple right tables not yet supported for antiJoin");

  auto rightTableView = rightTables[0]->view();

  // For the special case where we need to drop nulls, we create a local table.
  // Otherwise, we use the input view directly.
  std::unique_ptr<cudf::table> modifiedLeftTable;
  cudf::table_view leftTableView = leftTableViewParam;

  // Special case for null-aware anti join where
  // build table is not empty, no nulls, and probe table has nulls
  if (joinNode_->isNullAware() and !joinNode_->filter()) {
    auto const leftTableHasNulls =
        cudf::has_nulls(leftTableViewParam.select(leftKeyIndices_));
    auto const rightTableHasNulls =
        cudf::has_nulls(rightTableView.select(rightKeyIndices_));
    if (rightTables[0]->num_rows() > 0 and !rightTableHasNulls and
        leftTableHasNulls) {
      // drop nulls on probe table - creates a new table
      modifiedLeftTable = cudf::drop_nulls(
          leftTableViewParam, leftKeyIndices_, stream, get_temp_mr());
      leftTableView = modifiedLeftTable->view();
    }
  }

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> leftJoinIndices;
  if (joinNode_->filter()) {
    leftJoinIndices = cudf::mixed_left_anti_join(
        leftTableView.select(leftKeyIndices_),
        rightTableView.select(rightKeyIndices_),
        leftTableView,
        rightTableView,
        tree_.back(),
        cudf::null_equality::UNEQUAL,
        stream,
        get_temp_mr());
  } else {
    auto const rightTableHasNulls =
        cudf::has_nulls(rightTableView.select(rightKeyIndices_));
    if (joinNode_->isNullAware() and rightTableHasNulls) {
      // empty result
      leftJoinIndices = std::make_unique<rmm::device_uvector<cudf::size_type>>(
          0, stream, get_temp_mr());
    } else {
      cudf::filtered_join filter_join(
          rightTableView.select(rightKeyIndices_),
          cudf::null_equality::UNEQUAL,
          stream);
      leftJoinIndices = filter_join.anti_join(
          leftTableView.select(leftKeyIndices_), stream, get_temp_mr());
    }
  }

  auto leftIndicesSpan =
      cudf::device_span<cudf::size_type const>{*leftJoinIndices};
  auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
  auto rightIndicesCol = cudf::empty_like(leftIndicesCol);
  cudfOutputs.push_back(unfilteredOutput(
      leftTableView,
      leftIndicesCol,
      rightTableView,
      rightIndicesCol->view(),
      stream));

  return cudfOutputs;
}

RowVectorPtr CudfHashJoinProbe::doGetOutput() {
  if (finished_) {
    return nullptr;
  }
  if (parkedHashObject_.has_value()) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
    VELOX_CHECK_NOT_NULL(cudfInput);
    auto hydrationStream = cudfInput->stream();
    auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId());
    auto cudfJoinBridge =
        std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
    VELOX_CHECK_NOT_NULL(cudfJoinBridge);
    try {
      auto hydrated =
          hydrateParkedHashTable(parkedHashObject_.value(), hydrationStream);
      auto buildReadyEvent =
          std::make_shared<CudaEvent>(cudaEventDisableTiming);
      buildReadyEvent->recordFrom(hydrationStream);
      cudfJoinBridge->setBuildStream(hydrationStream);
      cudfJoinBridge->setBuildReadyEvent(buildReadyEvent);
      cudfJoinBridge->setHydratedHashTable(hydrated);
      hashObject_ = std::move(hydrated);
      buildStream_ = hydrationStream;
      buildReadyEvent_ = std::move(buildReadyEvent);
      parkedHashObject_.reset();
    } catch (...) {
      cudfJoinBridge->setHydrationError(std::current_exception());
      throw;
    }
  }
  if (!hashObject_.has_value()) {
    return nullptr;
  }
  if (!input_) {
    // If no more input, emit unmatched-right rows if needed.
    if ((joinNode_->isRightJoin() || joinNode_->isFullJoin()) && noMoreInput_ &&
        !finished_ && isLastDriver_) {
      auto& rightTables = hashObject_.value().buildTables;
      auto stream = cudfGlobalStreamPool().get_stream();
      std::vector<std::unique_ptr<cudf::table>> toConcat;
      vector_size_t unmatchedRows = 0;
      for (size_t i = 0; i < rightTables.size(); ++i) {
        auto& rightTable = rightTables[i];
        auto n = rightTable->num_rows();
        if (n == 0) {
          continue;
        }
        auto& flags = rightMatchedFlags_[i];
        // Build a boolean mask: unmatched = NOT(flags)
        auto boolMask = cudf::unary_operation(
            flags->view(), cudf::unary_operator::NOT, stream, get_temp_mr());

        // Count unmatched rows by summing the boolean mask
        auto unmatchedCountScalar = cudf::reduce(
            boolMask->view(),
            *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
            cudf::data_type{cudf::type_id::INT32},
            stream,
            get_temp_mr());
        auto m = static_cast<cudf::numeric_scalar<int32_t>*>(
                     unmatchedCountScalar.get())
                     ->value(stream);
        if (m == 0) {
          continue;
        }
        unmatchedRows += m;

        // Build left null columns
        std::vector<std::unique_ptr<cudf::column>> outCols(outputType_->size());
        // Left side nulls (types derive from probe schema at the matching
        // channel indices)
        for (size_t li = 0; li < leftColumnOutputIndices_.size(); ++li) {
          auto outIdx = leftColumnOutputIndices_[li];
          auto probeChannel = leftColumnIndicesToGather_[li];
          auto leftCudfDataType =
              veloxToCudfDataType(probeType_->childAt(probeChannel));
          auto nullScalar = cudf::make_default_constructed_scalar(
              leftCudfDataType, stream, get_temp_mr());
          outCols[outIdx] = cudf::make_column_from_scalar(
              *nullScalar, m, stream, get_output_mr());
        }
        // Right side - gather unmatched build columns if any
        if (!rightColumnIndicesToGather_.empty()) {
          auto rightInput =
              rightTable->view().select(rightColumnIndicesToGather_);
          auto unmatchedRight = cudf::apply_boolean_mask(
              rightInput, boolMask->view(), stream, get_output_mr());
          auto rightCols = unmatchedRight->release();
          for (size_t ri = 0; ri < rightColumnOutputIndices_.size(); ++ri) {
            auto outIdx = rightColumnOutputIndices_[ri];
            outCols[outIdx] = std::move(rightCols[ri]);
          }
        }
        toConcat.push_back(std::make_unique<cudf::table>(std::move(outCols)));
      }
      // TODO (dm): We build multiple right chunks only when they are too large
      // to fit in cudf::size_type. In case of a right join which doesn't have a
      // lot of matches we'll get outCols of similar size. This concatenation
      // will overflow. Try emitting result of one right chunk at a time.
      if (!toConcat.empty()) {
        auto out =
            concatenateTables(std::move(toConcat), stream, get_output_mr());
        finished_ = true;
        auto size = outputType_->size() == 0 ? unmatchedRows : out->num_rows();
        if (size == 0) {
          return nullptr;
        }
        return std::make_shared<CudfVector>(
            pool(), outputType_, size, std::move(out), stream);
      }
      finished_ = true;
    }
    return nullptr;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(cudfInput);
  auto stream = cudfInput->stream();
  waitForBuildReady(stream);
  // Use getTableView() to avoid expensive materialization for packed_table.
  // cudfInput is staying alive until the table view is no longer needed.
  auto leftTableView = cudfInput->getTableView();
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "Probe table number of columns: " << leftTableView.num_columns();
    VLOG(1) << "Probe table number of rows: " << leftTableView.num_rows();
  }

  if (hashObject_->deferredHydration) {
    VELOX_CHECK(joinNode_->isInnerJoin());
    auto& buildTables = hashObject_->buildTables;
    VELOX_CHECK(!buildTables.empty());
    while (nextBuildTableIndex_ < buildTables.size()) {
      const auto buildTableIndex = nextBuildTableIndex_++;
      auto output = innerJoinShard(buildTableIndex, leftTableView, stream);
      const auto outputRows = output.numRows;
      const bool inputComplete = nextBuildTableIndex_ == buildTables.size();
      {
        auto lockedStats = stats_.wlock();
        lockedStats->addRuntimeStat(
            "cudfHashJoinProbeBuildShardProbes", RuntimeCounter(1));
        lockedStats->addRuntimeStat(
            "cudfHashJoinProbeStreamedOutputRows", RuntimeCounter(outputRows));
      }

      if (inputComplete) {
        // unfilteredOutput synchronizes its gathered result, so the returned
        // table is independent of both the probe input and build shard.
        cudfInput.reset();
        input_.reset();
        nextBuildTableIndex_ = 0;
        finished_ = noMoreInput_;
      }

      const auto size =
          outputType_->size() == 0 ? outputRows : output.table->num_rows();
      if (size == 0) {
        if (inputComplete) {
          return nullptr;
        }
        continue;
      }
      return std::make_shared<CudfVector>(
          pool(), outputType_, size, std::move(output.table), stream);
    }
    VELOX_UNREACHABLE();
  }

  auto& rightTables = hashObject_.value().buildTables;
  auto& hashJoins = hashObject_.value().hashJoins;
  auto& distinctHashJoins = hashObject_.value().distinctHashJoins;
  for (auto i = 0; i < rightTables.size(); i++) {
    auto& rightTable = rightTables[i];
    VELOX_CHECK_NOT_NULL(rightTable);
    if (CudfConfig::getInstance().debugEnabled) {
      if (rightTable != nullptr) {
        VLOG(2) << "right_table is not nullptr " << rightTable.get()
                << " hasValue(" << hashObject_.has_value() << ")\n";
      }
      if (hashJoins[i] != nullptr) {
        VLOG(2) << "hash join is not nullptr " << hashJoins[i].get()
                << " hasValue(" << hashObject_.has_value() << ")\n";
      }
      if (distinctHashJoins[i] != nullptr) {
        VLOG(2) << "distinct hash join is not nullptr "
                << distinctHashJoins[i].get() << " hasValue("
                << hashObject_.has_value() << ")\n";
      }
    }
  }

  std::vector<JoinOutput> cudfOutputs;
  switch (joinNode_->joinType()) {
    case core::JoinType::kInner:
      cudfOutputs = innerJoin(leftTableView, stream);
      break;
    case core::JoinType::kLeft:
      cudfOutputs = leftJoin(leftTableView, stream);
      break;
    case core::JoinType::kRight:
      cudfOutputs = rightJoin(leftTableView, stream);
      break;
    case core::JoinType::kLeftSemiFilter:
      cudfOutputs = leftSemiFilterJoin(leftTableView, stream);
      break;
    case core::JoinType::kLeftSemiProject:
      cudfOutputs = leftSemiProjectJoin(leftTableView, stream);
      break;
    case core::JoinType::kRightSemiFilter:
      cudfOutputs = rightSemiFilterJoin(leftTableView, stream);
      break;
    case core::JoinType::kAnti:
      cudfOutputs = antiJoin(leftTableView, stream);
      break;
    case core::JoinType::kFull:
      cudfOutputs = fullJoin(leftTableView, stream);
      break;
    default:
      VELOX_FAIL("Unsupported join type: ", joinNode_->joinType());
  }

  // Record probe stream for cross-driver synchronization in noMoreInput().
  if (joinNode_->isRightJoin() || joinNode_->isFullJoin()) {
    lastProbeStream_ = stream;
  }

  // Release input CudfVector to free GPU memory before creating output.
  // This reduces peak memory from (input + output) to max(input, output).
  // cudfInput must be released first since input_.reset() only decrements
  // the refcount while cudfInput still holds a reference.
  cudfInput.reset();
  input_.reset();
  finished_ =
      noMoreInput_ && !joinNode_->isRightJoin() && !joinNode_->isFullJoin();

  vector_size_t zeroColumnOutputRows = 0;
  std::vector<std::unique_ptr<cudf::table>> cudfOutputTables;
  cudfOutputTables.reserve(cudfOutputs.size());
  for (auto& output : cudfOutputs) {
    zeroColumnOutputRows += output.numRows;
    cudfOutputTables.push_back(std::move(output.table));
  }

  auto cudfOutput =
      concatenateTables(std::move(cudfOutputTables), stream, get_output_mr());
  auto const size =
      outputType_->size() == 0 ? zeroColumnOutputRows : cudfOutput->num_rows();
  if (size == 0) {
    return nullptr;
  }
  return std::make_shared<CudfVector>(
      pool(), outputType_, size, std::move(cudfOutput), stream);
}

bool CudfHashJoinProbe::skipProbeOnEmptyBuild() const {
  auto const joinType = joinNode_->joinType();
  return isInnerJoin(joinType) || isLeftSemiFilterJoin(joinType) ||
      isRightJoin(joinType) || isRightSemiFilterJoin(joinType) ||
      isRightSemiProjectJoin(joinType);
}

exec::BlockingReason CudfHashJoinProbe::isBlocked(ContinueFuture* future) {
  if (deferredHydration_ && future_.valid()) {
    *future = std::move(future_);
    return exec::BlockingReason::kWaitForJoinProbe;
  }
  if (deferredProbeBarrierEntered_) {
    // The last probe releases the bridge before waking its peers. A peer that
    // received no probe rows has no local hash object, but is nevertheless
    // terminal after crossing this barrier and must not query the released
    // bridge when the Driver polls isBlocked() before isFinished().
    return exec::BlockingReason::kNotBlocked;
  }
  if (parkedHashObject_.has_value()) {
    return exec::BlockingReason::kNotBlocked;
  }
  if ((joinNode_->isRightJoin() || joinNode_->isRightSemiFilterJoin() ||
       joinNode_->isFullJoin()) &&
      hashObject_.has_value()) {
    if (!future_.valid()) {
      return exec::BlockingReason::kNotBlocked;
    }
    *future = std::move(future_);
    return exec::BlockingReason::kWaitForJoinProbe;
  }

  if (hashObject_.has_value()) {
    return exec::BlockingReason::kNotBlocked;
  }

  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto cudfJoinBridge =
      std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
  VELOX_CHECK_NOT_NULL(cudfJoinBridge);
  VELOX_CHECK_NOT_NULL(future);
  auto hashResult = cudfJoinBridge->hashOrFuture(future, input_ != nullptr);

  if (hashResult.parkedHashObject.has_value()) {
    deferredHydration_ = true;
    parkedHashObject_ = std::move(hashResult.parkedHashObject);
    return exec::BlockingReason::kNotBlocked;
  } else if (hashResult.hashObject.has_value()) {
    hashObject_ = std::move(hashResult.hashObject);
    deferredHydration_ = hashObject_->deferredHydration;
    buildStream_ = cudfJoinBridge->getBuildStream();
    buildReadyEvent_ = cudfJoinBridge->getBuildReadyEvent();
  } else if (hashResult.buildParked) {
    // Do not hydrate merely because the driver polls isBlocked(). Accept one
    // real probe input first; the next poll atomically elects its hydrator.
    deferredHydration_ = true;
    return exec::BlockingReason::kNotBlocked;
  } else {
    if (CudfConfig::getInstance().debugEnabled) {
      VLOG(2) << "CudfHashJoinProbe is blocked, waiting for join build or "
                 "deferred hydration";
    }
    return exec::BlockingReason::kWaitForJoinBuild;
  }

  // Lazy initialize matched flags only when build side is done
  if (joinNode_->isRightJoin() || joinNode_->isFullJoin()) {
    auto& rightTablesInit = hashObject_.value().buildTables;
    rightMatchedFlags_.clear();
    rightMatchedFlags_.reserve(rightTablesInit.size());
    auto initStream = cudfGlobalStreamPool().get_stream();
    waitForBuildReady(initStream);
    for (auto& rt : rightTablesInit) {
      auto n = rt->num_rows();
      auto false_scalar =
          cudf::numeric_scalar<bool>(false, true, initStream, get_temp_mr());
      auto flags_col = cudf::make_column_from_scalar(
          false_scalar, n, initStream, get_temp_mr());
      rightMatchedFlags_.push_back(std::move(flags_col));
    }
    initStream.synchronize();
  }

  // Precompute right table columns if filter exists (once when build is done)
  if (joinNode_->filter() && !rightPrecomputeInstructions_.empty()) {
    auto& rightTablesInit = hashObject_.value().buildTables;
    cachedRightPrecomputed_.clear();
    cachedExtendedRightViews_.clear();
    cachedRightPrecomputed_.reserve(rightTablesInit.size());
    cachedExtendedRightViews_.reserve(rightTablesInit.size());

    auto initStream = cudfGlobalStreamPool().get_stream();
    waitForBuildReady(initStream);
    for (auto& rt : rightTablesInit) {
      auto rightTableView = rt->view();
      auto rightColumnViews = tableViewToColumnViews(rightTableView);
      auto rightPrecomputed = precomputeSubexpressions(
          rightColumnViews,
          rightPrecomputeInstructions_,
          scalars_,
          buildType_,
          initStream);
      auto extendedView =
          createExtendedTableView(rightTableView, rightPrecomputed);
      cachedRightPrecomputed_.push_back(std::move(rightPrecomputed));
      cachedExtendedRightViews_.push_back(extendedView);
    }
    initStream.synchronize();
  }

  // Check if build side has any null keys (needed for null-aware left semi
  // project)
  if (joinNode_->isLeftSemiProjectJoin() && joinNode_->isNullAware()) {
    auto& rightTablesInit = hashObject_.value().buildTables;
    buildSideHasNullKeys_ = false;
    for (auto& rt : rightTablesInit) {
      auto keyView = rt->view().select(rightKeyIndices_);
      for (cudf::size_type k = 0; k < keyView.num_columns(); k++) {
        if (keyView.column(k).has_nulls()) {
          buildSideHasNullKeys_ = true;
          break;
        }
      }
      if (buildSideHasNullKeys_) {
        break;
      }
    }
  }

  auto& rightTables = hashObject_.value().buildTables;
  // should be rightTable->numDistinct() but it needs compute,
  // so we use num_rows()
  if (rightTables[0]->num_rows() == 0) {
    if (skipProbeOnEmptyBuild()) {
      if (operatorCtx_->driverCtx()
              ->queryConfig()
              .hashProbeFinishEarlyOnEmptyBuild()) {
        noMoreInput();
      } else {
        skipInput_ = true;
      }
    }
  }
  if ((joinNode_->isRightJoin() || joinNode_->isRightSemiFilterJoin() ||
       joinNode_->isFullJoin()) &&
      future_.valid()) {
    *future = std::move(future_);
    return exec::BlockingReason::kWaitForJoinProbe;
  }
  return exec::BlockingReason::kNotBlocked;
}

bool CudfHashJoinProbe::isFinished() {
  const auto outputDrained = finished_ ||
      (noMoreInput_ && input_ == nullptr && nextBuildTableIndex_ == 0);
  const auto isFinished =
      deferredHydration_ ? outputDrained && !future_.valid() : outputDrained;

  // Release hashObject_ if finished
  if (isFinished) {
    hashObject_.reset();
    buildReadyEvent_.reset();
    buildStream_.reset();
  }
  return isFinished;
}

std::unique_ptr<exec::Operator> CudfHashJoinBridgeTranslator::toOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridgeTranslator::toOperator";
  }
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
    return std::make_unique<CudfHashJoinProbe>(id, ctx, joinNode);
  }
  return nullptr;
}

std::unique_ptr<exec::JoinBridge> CudfHashJoinBridgeTranslator::toJoinBridge(
    const core::PlanNodePtr& node) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridgeTranslator::toJoinBridge";
  }
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
    auto joinBridge = std::make_unique<CudfHashJoinBridge>();
    return joinBridge;
  }
  return nullptr;
}

exec::OperatorSupplier CudfHashJoinBridgeTranslator::toOperatorSupplier(
    const core::PlanNodePtr& node) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridgeTranslator::toOperatorSupplier";
  }
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
    return [joinNode](int32_t operatorId, exec::DriverCtx* ctx) {
      return std::make_unique<CudfHashJoinBuild>(operatorId, ctx, joinNode);
    };
  }
  return nullptr;
}

} // namespace facebook::velox::cudf_velox
