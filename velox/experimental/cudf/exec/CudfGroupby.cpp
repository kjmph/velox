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
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/CudfGroupby.h"
#include "velox/experimental/cudf/exec/DecimalAggregationHostOps.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/HashAggregation.h"
#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/reduction.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <limits>

namespace {

using namespace facebook::velox;
using cudf_velox::castDecimalInputToDecimal128;
using cudf_velox::CountInputKind;
using cudf_velox::finalizeDecimalAverage;
using cudf_velox::get_output_mr;
using cudf_velox::get_temp_mr;
using cudf_velox::GroupbyAggregationRequestBuilder;
using cudf_velox::GroupbyAggregationResultRef;
using cudf_velox::GroupbyAggregator;
using cudf_velox::GroupbyPartialMeanResultRefs;
using cudf_velox::ResolvedAggregateInfo;
using cudf_velox::serializeDecimalPartialOrIntermediateState;
using cudf_velox::validateIntermediateColumnType;

constexpr uint64_t kMinFinalAggregationBucketBytes = 128ULL << 20;
constexpr uint64_t kMaxFinalAggregationBucketBytes = 1ULL << 30;
constexpr uint32_t kMaxFinalAggregationPartitionFanout = 256;
constexpr uint32_t kMaxFinalAggregationHashDepth = 8;
constexpr uint64_t kDecimalAggregateStateWorkBytes = 64;
// cuDF's hash group-by needs storage beyond the flat input and output tables.
// Reserve room for its hash set, row mapping, and associated index scratch so
// narrow inputs cannot turn a nominally small byte bucket into a multi-GiB
// workspace allocation.
constexpr uint64_t kHashGroupbyWorkspaceBytesPerRow = 16;

uint64_t ceilingDivide(uint64_t dividend, uint64_t divisor) {
  VELOX_CHECK_GT(divisor, 0);
  return dividend / divisor + (dividend % divisor != 0);
}

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

uint32_t roundUpToPowerOfTwo(uint64_t value, uint32_t maximum) {
  VELOX_CHECK_GT(maximum, 0);
  uint32_t result = 1;
  while (result < value && result < maximum) {
    result <<= 1;
  }
  return std::min(result, maximum);
}

uint32_t advanceHashSeed(uint32_t& seed) {
  // A deterministic full-period LCG gives recursive partitions independent
  // hash functions without making output depend on process-global state.
  seed = seed * 1664525U + 1013904223U;
  return seed;
}

uint64_t conservativePhysicalBytesPerRow(const TypePtr& type) {
  // Account for one byte of nullability per logical column. This deliberately
  // overestimates cuDF's bit mask so the row budget remains conservative for
  // small slices as well as large ones.
  constexpr uint64_t kNullByte = 1;
  if (type->isFixedWidth()) {
    return saturatedAdd(type->cppSizeInBytes(), kNullByte);
  }

  // Variable-width columns need at least an offset and a null bit per row.
  // Their payload is covered by the actual input bytes-per-row estimate used
  // alongside this projected intermediate-state estimate.
  uint64_t bytes = saturatedAdd(sizeof(cudf::size_type), kNullByte);
  for (uint32_t i = 0; i < type->size(); ++i) {
    bytes =
        saturatedAdd(bytes, conservativePhysicalBytesPerRow(type->childAt(i)));
  }
  return bytes;
}

std::unique_ptr<cudf::column> takeGroupbyResultColumn(
    std::vector<cudf::groupby::aggregation_result>& results,
    const GroupbyAggregationResultRef& ref,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK_LT(ref.requestIndex, results.size());
  VELOX_CHECK_LT(ref.resultIndex, results[ref.requestIndex].results.size());

  auto& result = results[ref.requestIndex].results[ref.resultIndex];
  VELOX_CHECK_NOT_NULL(result);
  if (ref.shared != nullptr && *ref.shared) {
    return std::make_unique<cudf::column>(
        result->view(), stream, get_output_mr());
  }
  return std::move(result);
}

#define DEFINE_SIMPLE_GROUPBY_AGGREGATOR(Name, name, KIND)               \
  struct Groupby##Name##Aggregator : GroupbyAggregator {                 \
    Groupby##Name##Aggregator(                                           \
        core::AggregationNode::Step step,                                \
        uint32_t inputIndex,                                             \
        uint32_t reusableInputKey,                                       \
        VectorPtr constant,                                              \
        const TypePtr& resultType)                                       \
        : GroupbyAggregator(                                             \
              step,                                                      \
              inputIndex,                                                \
              reusableInputKey,                                          \
              constant,                                                  \
              resultType) {}                                             \
                                                                         \
    void addGroupbyRequest(                                              \
        GroupbyAggregationRequestBuilder& builder) override {            \
      VELOX_CHECK(                                                       \
          constant == nullptr,                                           \
          #Name "Aggregator does not yet support constant input");       \
      outputRef_ = builder.addAggregationForColumn(                      \
          inputIndex,                                                    \
          cudf::make_##name##_aggregation<cudf::groupby_aggregation>()); \
    }                                                                    \
                                                                         \
    std::unique_ptr<cudf::column> makeOutputColumn(                      \
        std::vector<cudf::groupby::aggregation_result>& results,         \
        rmm::cuda_stream_view stream) override {                         \
      auto col = takeGroupbyResultColumn(results, outputRef_, stream);   \
      const auto cudfType = cudf_velox::veloxToCudfDataType(resultType); \
      if (col->type() != cudfType) {                                     \
        col = cudf::cast(*col, cudfType, stream, get_output_mr());       \
      }                                                                  \
      return col;                                                        \
    }                                                                    \
                                                                         \
   private:                                                              \
    GroupbyAggregationResultRef outputRef_;                              \
  };

DEFINE_SIMPLE_GROUPBY_AGGREGATOR(Min, min, MIN)
DEFINE_SIMPLE_GROUPBY_AGGREGATOR(Max, max, MAX)

struct GroupbySumAggregator : GroupbyAggregator {
  GroupbySumAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      uint32_t reusableInputKey,
      VectorPtr constant,
      const TypePtr& resultType)
      : GroupbyAggregator(
            step,
            inputIndex,
            reusableInputKey,
            constant,
            resultType) {}

  void addGroupbyRequest(GroupbyAggregationRequestBuilder& builder) override {
    VELOX_CHECK(
        constant == nullptr,
        "SumAggregator does not yet support constant input");
    outputRef_ = builder.addSum(inputIndex, reusableInputKey, step);
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    auto col = takeGroupbyResultColumn(results, outputRef_, stream);
    const auto cudfType = cudf_velox::veloxToCudfDataType(resultType);
    if (col->type() != cudfType) {
      col = cudf::cast(*col, cudfType, stream, get_output_mr());
    }
    return col;
  }

 private:
  GroupbyAggregationResultRef outputRef_;
};

// Decimal SUM and AVG need to serialize their intermediate sum/count state
// into Velox VARBINARY and accumulate narrow decimal input in DECIMAL128. They
// therefore cannot use the generic sum/mean request paths. These aggregators
// retain decoded and casted columns because the cuDF aggregation requests hold
// non-owning views of those columns until groupby execution completes.
void addDecimalDecodedSumCountRequests(
    GroupbyAggregationRequestBuilder& builder,
    uint32_t inputIndex,
    const TypePtr& resultType,
    rmm::cuda_stream_view stream,
    GroupbyAggregationResultRef& sumRef,
    GroupbyAggregationResultRef& countRef,
    std::unique_ptr<cudf::column>& decodedSum,
    std::unique_ptr<cudf::column>& decodedCount) {
  const auto encodedColumn = builder.tableView().column(inputIndex);
  validateIntermediateColumnType(encodedColumn);
  const auto scale = resultType->isDecimal()
      ? getDecimalPrecisionScale(*resultType).second
      : 0;
  auto sumAndCount =
      cudf_velox::deserializeDecimalSumState(encodedColumn, scale, stream);
  decodedSum.swap(sumAndCount.sum);
  decodedCount.swap(sumAndCount.count);

  sumRef = builder.addAggregationForValues(
      decodedSum->view(),
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  countRef = builder.addAggregationForValues(
      decodedCount->view(),
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());
}

void addDecimalFinalSumOnlyRequest(
    GroupbyAggregationRequestBuilder& builder,
    uint32_t inputIndex,
    const TypePtr& resultType,
    rmm::cuda_stream_view stream,
    GroupbyAggregationResultRef& sumRef,
    std::unique_ptr<cudf::column>& decodedSum) {
  const auto encodedColumn = builder.tableView().column(inputIndex);
  validateIntermediateColumnType(encodedColumn);
  const auto scale = getDecimalPrecisionScale(*resultType).second;
  auto sumAndCount =
      cudf_velox::deserializeDecimalSumState(encodedColumn, scale, stream);
  decodedSum.swap(sumAndCount.sum);
  sumRef = builder.addAggregationForValues(
      decodedSum->view(),
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());
}

void addDecimalRawPartialSingleSumRequest(
    GroupbyAggregationRequestBuilder& builder,
    uint32_t inputIndex,
    bool includeCountAggregation,
    rmm::cuda_stream_view stream,
    GroupbyAggregationResultRef& sumRef,
    GroupbyAggregationResultRef& countRef,
    std::unique_ptr<cudf::column>& castedInput) {
  auto inputView = castDecimalInputToDecimal128(
      builder.tableView().column(inputIndex), castedInput, stream);
  sumRef = builder.addAggregationForValues(
      inputView, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  if (includeCountAggregation) {
    builder.aggregationRequest(sumRef).aggregations.push_back(
        cudf::make_count_aggregation<cudf::groupby_aggregation>(
            cudf::null_policy::EXCLUDE));
    countRef = {sumRef.requestIndex, 1, std::make_shared<bool>(false)};
  }
}

struct GroupbyDecimalSumAggregator : GroupbyAggregator {
  GroupbyDecimalSumAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      uint32_t reusableInputKey,
      VectorPtr constant,
      const TypePtr& resultType)
      : GroupbyAggregator(
            step,
            inputIndex,
            reusableInputKey,
            constant,
            resultType) {}

  void addGroupbyRequest(GroupbyAggregationRequestBuilder& builder) override {
    const auto stream = builder.stream();
    if (step == core::AggregationNode::Step::kIntermediate) {
      addDecimalDecodedSumCountRequests(
          builder,
          inputIndex,
          resultType,
          stream,
          sumRef_,
          countRef_,
          decodedSum_,
          decodedCount_);
    } else if (step == core::AggregationNode::Step::kFinal) {
      addDecimalFinalSumOnlyRequest(
          builder, inputIndex, resultType, stream, sumRef_, decodedSum_);
    } else {
      addDecimalRawPartialSingleSumRequest(
          builder,
          inputIndex,
          step == core::AggregationNode::Step::kPartial,
          stream,
          sumRef_,
          countRef_,
          castedInput_);
    }
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    auto col = takeGroupbyResultColumn(results, sumRef_, stream);
    if (step == core::AggregationNode::Step::kPartial) {
      auto count = takeGroupbyResultColumn(results, countRef_, stream);
      return serializeDecimalPartialOrIntermediateState(
          std::move(col), std::move(count), stream, get_output_mr());
    }
    if (step == core::AggregationNode::Step::kIntermediate) {
      auto count = takeGroupbyResultColumn(results, countRef_, stream);
      return serializeDecimalPartialOrIntermediateState(
          std::move(col), std::move(count), stream, get_output_mr());
    }
    const auto cudfResultType = cudf_velox::veloxToCudfDataType(resultType);
    if (col->type() != cudfResultType) {
      col = cudf::cast(*col, cudfResultType, stream, get_output_mr());
    }
    return col;
  }

  void releaseRequestInputs() override {
    decodedSum_.reset();
    decodedCount_.reset();
    castedInput_.reset();
  }

 private:
  GroupbyAggregationResultRef sumRef_;
  GroupbyAggregationResultRef countRef_;
  std::unique_ptr<cudf::column> decodedSum_;
  std::unique_ptr<cudf::column> decodedCount_;
  std::unique_ptr<cudf::column> castedInput_;
};

struct GroupbyDecimalAvgAggregator : GroupbyAggregator {
  GroupbyDecimalAvgAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      uint32_t reusableInputKey,
      VectorPtr constant,
      const TypePtr& resultType)
      : GroupbyAggregator(
            step,
            inputIndex,
            reusableInputKey,
            constant,
            resultType) {}

  void addGroupbyRequest(GroupbyAggregationRequestBuilder& builder) override {
    const auto stream = builder.stream();
    if (step == core::AggregationNode::Step::kIntermediate ||
        step == core::AggregationNode::Step::kFinal) {
      addDecimalDecodedSumCountRequests(
          builder,
          inputIndex,
          resultType,
          stream,
          sumRef_,
          countRef_,
          decodedSum_,
          decodedCount_);
    } else {
      addDecimalRawPartialSingleSumRequest(
          builder,
          inputIndex,
          step == core::AggregationNode::Step::kPartial ||
              step == core::AggregationNode::Step::kSingle,
          stream,
          sumRef_,
          countRef_,
          castedInput_);
    }
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    auto sum = takeGroupbyResultColumn(results, sumRef_, stream);
    if (step == core::AggregationNode::Step::kSingle) {
      auto count = takeGroupbyResultColumn(results, countRef_, stream);
      return finalizeDecimalAverage(
          std::move(sum),
          std::move(count),
          resultType,
          stream,
          get_output_mr());
    }
    if (step == core::AggregationNode::Step::kPartial) {
      auto count = takeGroupbyResultColumn(results, countRef_, stream);
      return serializeDecimalPartialOrIntermediateState(
          std::move(sum), std::move(count), stream, get_output_mr());
    }
    if (step == core::AggregationNode::Step::kIntermediate) {
      auto count = takeGroupbyResultColumn(results, countRef_, stream);
      return serializeDecimalPartialOrIntermediateState(
          std::move(sum), std::move(count), stream, get_output_mr());
    }
    if (step == core::AggregationNode::Step::kFinal) {
      auto count = takeGroupbyResultColumn(results, countRef_, stream);
      return finalizeDecimalAverage(
          std::move(sum),
          std::move(count),
          resultType,
          stream,
          get_output_mr());
    }
    VELOX_UNREACHABLE();
  }

  void releaseRequestInputs() override {
    decodedSum_.reset();
    decodedCount_.reset();
    castedInput_.reset();
  }

 private:
  GroupbyAggregationResultRef sumRef_;
  GroupbyAggregationResultRef countRef_;
  std::unique_ptr<cudf::column> decodedSum_;
  std::unique_ptr<cudf::column> decodedCount_;
  std::unique_ptr<cudf::column> castedInput_;
};

struct GroupbyCountAggregator : GroupbyAggregator {
  GroupbyCountAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      uint32_t reusableInputKey,
      CountInputKind inputKind,
      const TypePtr& resultType)
      : GroupbyAggregator(
            step,
            inputIndex,
            reusableInputKey,
            nullptr,
            resultType),
        inputKind_(inputKind) {}

  void addGroupbyRequest(GroupbyAggregationRequestBuilder& builder) override {
    // kCountAll and kNullConstant both submit a count-all-rows request;
    // kNullConstant overrides the result with zeros in makeOutputColumn.
    const bool countAll = (inputKind_ != CountInputKind::kColumn);
    // For raw input, count(*) can use any column (column 0) since we just
    // need a row count. For non-raw input (intermediate/final in streaming),
    // the input is partial results where column 0 is the grouping key;
    // we must use inputIndex to access the partial count column.
    std::unique_ptr<cudf::groupby_aggregation> aggRequest =
        exec::isRawInput(step)
        ? cudf::make_count_aggregation<cudf::groupby_aggregation>(
              countAll ? cudf::null_policy::INCLUDE
                       : cudf::null_policy::EXCLUDE)
        : cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    outputRef_ = builder.addAggregationForColumn(
        (countAll && exec::isRawInput(step)) ? 0 : inputIndex,
        std::move(aggRequest));
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    auto col = takeGroupbyResultColumn(results, outputRef_, stream);
    if (inputKind_ == CountInputKind::kNullConstant) {
      auto zero = cudf::numeric_scalar<int64_t>(0, true, stream, get_temp_mr());
      col = cudf::make_column_from_scalar(
          zero, col->size(), stream, get_output_mr());
    }
    // cudf produces int32 for count but velox expects int64.
    const auto cudfOutputType = cudf_velox::veloxToCudfDataType(resultType);
    if (col->type() != cudfOutputType) {
      col = cudf::cast(*col, cudfOutputType, stream, get_output_mr());
    }
    return col;
  }

 private:
  CountInputKind inputKind_;
  GroupbyAggregationResultRef outputRef_;
};

struct GroupbyMeanAggregator : GroupbyAggregator {
  GroupbyMeanAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      uint32_t reusableInputKey,
      VectorPtr constant,
      const TypePtr& resultType)
      : GroupbyAggregator(
            step,
            inputIndex,
            reusableInputKey,
            constant,
            resultType) {}

  void addGroupbyRequest(GroupbyAggregationRequestBuilder& builder) override {
    switch (step) {
      case core::AggregationNode::Step::kSingle: {
        meanRef_ = builder.addAggregationForColumn(
            inputIndex,
            cudf::make_mean_aggregation<cudf::groupby_aggregation>());
        break;
      }
      case core::AggregationNode::Step::kPartial: {
        auto refs = builder.addPartialMean(inputIndex, reusableInputKey);
        sumRef_ = refs.sum;
        countRef_ = refs.count;
        break;
      }
      case core::AggregationNode::Step::kIntermediate:
      case core::AggregationNode::Step::kFinal: {
        // In intermediate and final aggregation, the previously computed sum
        // and count are in the child columns of the input column.
        sumRef_ = builder.addAggregationForValues(
            builder.tableView().column(inputIndex).child(0),
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        // The counts are already computed in partial aggregation, so we just
        // need to sum them up again.
        countRef_ = builder.addAggregationForValues(
            builder.tableView().column(inputIndex).child(1),
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        break;
      }
      default:
        VELOX_NYI("Unsupported aggregation step for mean");
    }
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    const auto& outputType = asRowType(resultType);
    switch (step) {
      case core::AggregationNode::Step::kSingle:
        return takeGroupbyResultColumn(results, meanRef_, stream);
      case core::AggregationNode::Step::kPartial: {
        auto sum = takeGroupbyResultColumn(results, sumRef_, stream);
        auto count = takeGroupbyResultColumn(results, countRef_, stream);

        auto const size = sum->size();
        auto const cudfSumType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(0));
        auto const cudfCountType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(1));
        if (sum->type() != cudfSumType) {
          sum = cudf::cast(*sum, cudfSumType, stream, get_output_mr());
        }
        if (count->type() != cudfCountType) {
          count = cudf::cast(*count, cudfCountType, stream, get_output_mr());
        }

        auto children = std::vector<std::unique_ptr<cudf::column>>();
        children.push_back(std::move(sum));
        children.push_back(std::move(count));

        // TODO: Handle nulls. This can happen if all values are null in a
        // group.
        return std::make_unique<cudf::column>(
            cudf::data_type(cudf::type_id::STRUCT),
            size,
            rmm::device_buffer{},
            rmm::device_buffer{},
            0,
            std::move(children));
      }
      case core::AggregationNode::Step::kIntermediate: {
        // The difference between intermediate and partial is in where the
        // sum and count are coming from. In partial, since the input column is
        // the same, the sum and count are in the same agg result. In
        // intermediate, the input columns are different (it's the child
        // columns of the input column) and so the sum and count are in
        // different agg results.
        auto sum = takeGroupbyResultColumn(results, sumRef_, stream);
        auto count = takeGroupbyResultColumn(results, countRef_, stream);

        auto size = sum->size();
        auto const cudfSumType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(0));
        auto const cudfCountType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(1));
        if (sum->type() != cudfSumType) {
          sum = cudf::cast(*sum, cudfSumType, stream, get_output_mr());
        }
        if (count->type() != cudfCountType) {
          count = cudf::cast(*count, cudfCountType, stream, get_output_mr());
        }

        auto children = std::vector<std::unique_ptr<cudf::column>>();
        children.push_back(std::move(sum));
        children.push_back(std::move(count));

        return std::make_unique<cudf::column>(
            cudf::data_type(cudf::type_id::STRUCT),
            size,
            rmm::device_buffer{},
            rmm::device_buffer{},
            0,
            std::move(children));
      }
      case core::AggregationNode::Step::kFinal: {
        auto sum = takeGroupbyResultColumn(results, sumRef_, stream);
        auto count = takeGroupbyResultColumn(results, countRef_, stream);
        auto avg = cudf::binary_operation(
            *sum,
            *count,
            cudf::binary_operator::DIV,
            cudf_velox::veloxToCudfDataType(resultType),
            stream,
            get_output_mr());
        // Null out groups where count == 0 (empty groups). Masking on count
        // preserves legitimate NaN values from non-empty input groups.
        cudf::numeric_scalar<int64_t> zero(0, true, stream, get_temp_mr());
        auto validMask = cudf::binary_operation(
            *count,
            zero,
            cudf::binary_operator::GREATER,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            get_temp_mr());
        auto [mask, nullCount] =
            cudf::bools_to_mask(*validMask, stream, get_temp_mr());
        avg->set_null_mask(std::move(*mask), nullCount);
        return avg;
      }
      default:
        VELOX_NYI("Unsupported aggregation step for mean");
    }
  }

 private:
  // These indices are used to track where the desired result columns
  // (mean/<sum, count>) are in the output of cudf::groupby::aggregate().
  GroupbyAggregationResultRef meanRef_;
  GroupbyAggregationResultRef sumRef_;
  GroupbyAggregationResultRef countRef_;
};

struct GroupbyStddevSampAggregator : GroupbyAggregator {
  GroupbyStddevSampAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      uint32_t reusableInputKey,
      VectorPtr constant,
      const TypePtr& resultType)
      : GroupbyAggregator(
            step,
            inputIndex,
            reusableInputKey,
            constant,
            resultType) {}

  void addGroupbyRequest(GroupbyAggregationRequestBuilder& builder) override {
    switch (step) {
      case core::AggregationNode::Step::kSingle:
        // Use cuDF's built-in std aggregation with ddof=1 (sample stddev)
        outputRef_ = builder.addAggregationForColumn(
            inputIndex,
            cudf::make_std_aggregation<cudf::groupby_aggregation>(1));
        break;
      case core::AggregationNode::Step::kPartial:
        // Compute count, mean, m2 from raw values
        outputRef_ = builder.addAggregationForColumn(
            inputIndex,
            cudf::make_count_aggregation<cudf::groupby_aggregation>(
                cudf::null_policy::EXCLUDE));
        builder.aggregationRequest(outputRef_)
            .aggregations.push_back(
                cudf::make_mean_aggregation<cudf::groupby_aggregation>());
        builder.aggregationRequest(outputRef_)
            .aggregations.push_back(
                cudf::make_m2_aggregation<cudf::groupby_aggregation>());
        break;
      case core::AggregationNode::Step::kIntermediate:
      case core::AggregationNode::Step::kFinal:
        // Input is struct(count, mean, m2) - use MERGE_M2 to merge
        outputRef_ = builder.addAggregationForColumn(
            inputIndex,
            cudf::make_merge_m2_aggregation<cudf::groupby_aggregation>());
        break;
      default:
        VELOX_NYI("Unsupported aggregation step for stddev_samp");
    }
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    switch (step) {
      case core::AggregationNode::Step::kSingle:
        return takeGroupbyResultColumn(results, outputRef_, stream);
      case core::AggregationNode::Step::kPartial: {
        auto count = takeGroupbyResultColumn(results, outputRef_, stream);
        auto mean = takeGroupbyResultColumn(
            results, {outputRef_.requestIndex, 1, outputRef_.shared}, stream);
        auto m2 = takeGroupbyResultColumn(
            results, {outputRef_.requestIndex, 2, outputRef_.shared}, stream);
        return makeM2StructColumn(
            std::move(count), std::move(mean), std::move(m2), stream);
      }
      case core::AggregationNode::Step::kIntermediate: {
        auto merged = takeGroupbyResultColumn(results, outputRef_, stream);

        // Check if types already match expected output - avoid copies if so
        const auto& outputType = asRowType(resultType);
        auto const cudfCountType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(0));
        auto const cudfMeanType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(1));
        auto const cudfM2Type =
            cudf_velox::veloxToCudfDataType(outputType->childAt(2));

        auto mergedView = merged->view();
        bool typesMatch = mergedView.child(0).type() == cudfCountType &&
            mergedView.child(1).type() == cudfMeanType &&
            mergedView.child(2).type() == cudfM2Type;

        if (typesMatch) {
          // Types match - return merged directly to avoid device copies
          return merged;
        }

        // Types don't match - need to copy and cast (use output_mr since
        // these become part of the output)
        auto count = std::make_unique<cudf::column>(
            mergedView.child(0), stream, get_output_mr());
        auto mean = std::make_unique<cudf::column>(
            mergedView.child(1), stream, get_output_mr());
        auto m2 = std::make_unique<cudf::column>(
            mergedView.child(2), stream, get_output_mr());
        return makeM2StructColumn(
            std::move(count), std::move(mean), std::move(m2), stream);
      }
      case core::AggregationNode::Step::kFinal: {
        // MERGE_M2 returns struct(count, mean, m2)
        // Compute sqrt(m2 / (count - 1)) with NULL where count < 2
        auto merged = takeGroupbyResultColumn(results, outputRef_, stream);
        auto mergedView = merged->view();
        auto countView = mergedView.child(0);
        auto m2View = mergedView.child(2);

        // count - 1 (binary_operation handles type promotion)
        cudf::numeric_scalar<double> one(1.0, true, stream, get_temp_mr());
        auto countMinus1 = cudf::binary_operation(
            countView,
            one,
            cudf::binary_operator::SUB,
            cudf::data_type{cudf::type_id::FLOAT64},
            stream,
            get_temp_mr());

        // m2 / (count - 1)
        auto variance = cudf::binary_operation(
            m2View,
            *countMinus1,
            cudf::binary_operator::DIV,
            cudf::data_type{cudf::type_id::FLOAT64},
            stream,
            get_temp_mr());

        // sqrt(variance)
        auto stddev = cudf::unary_operation(
            *variance, cudf::unary_operator::SQRT, stream, get_temp_mr());

        // count >= 2
        cudf::numeric_scalar<int64_t> two(2, true, stream, get_temp_mr());
        auto validMask = cudf::binary_operation(
            countView,
            two,
            cudf::binary_operator::GREATER_EQUAL,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            get_temp_mr());

        // Apply mask: where count < 2, result is NULL
        cudf::numeric_scalar<double> nullDouble(
            0.0, false, stream, get_temp_mr());
        return cudf::copy_if_else(
            *stddev, nullDouble, *validMask, stream, get_output_mr());
      }
      default:
        VELOX_NYI("Unsupported aggregation step for stddev_samp");
    }
  }

 private:
  // Build a struct column with (count, mean, m2), casting to expected types.
  std::unique_ptr<cudf::column> makeM2StructColumn(
      std::unique_ptr<cudf::column> count,
      std::unique_ptr<cudf::column> mean,
      std::unique_ptr<cudf::column> m2,
      rmm::cuda_stream_view stream) {
    const auto& outputType = asRowType(resultType);
    auto const cudfCountType =
        cudf_velox::veloxToCudfDataType(outputType->childAt(0));
    auto const cudfMeanType =
        cudf_velox::veloxToCudfDataType(outputType->childAt(1));
    auto const cudfM2Type =
        cudf_velox::veloxToCudfDataType(outputType->childAt(2));

    if (count->type() != cudfCountType) {
      count = cudf::cast(*count, cudfCountType, stream, get_output_mr());
    }
    if (mean->type() != cudfMeanType) {
      mean = cudf::cast(*mean, cudfMeanType, stream, get_output_mr());
    }
    if (m2->type() != cudfM2Type) {
      m2 = cudf::cast(*m2, cudfM2Type, stream, get_output_mr());
    }

    auto const size = count->size();
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(std::move(count));
    children.push_back(std::move(mean));
    children.push_back(std::move(m2));

    return std::make_unique<cudf::column>(
        cudf::data_type(cudf::type_id::STRUCT),
        size,
        rmm::device_buffer{},
        rmm::device_buffer{},
        0,
        std::move(children));
  }

  GroupbyAggregationResultRef outputRef_;
};

std::unique_ptr<GroupbyAggregator> createGroupbyAggregator(
    const ResolvedAggregateInfo& p,
    uint32_t reusableInputKey) {
  auto const& kind = p.kind;
  auto prefix = cudf_velox::CudfConfig::getInstance().functionNamePrefix;
  if (kind.rfind(prefix + "sum", 0) == 0) {
    if (p.isDecimalAggregate) {
      return std::make_unique<GroupbyDecimalSumAggregator>(
          p.companionStep,
          p.inputIndex,
          reusableInputKey,
          p.constant,
          p.resultType);
    }
    return std::make_unique<GroupbySumAggregator>(
        p.companionStep,
        p.inputIndex,
        reusableInputKey,
        p.constant,
        p.resultType);
  } else if (kind.rfind(prefix + "count", 0) == 0) {
    VELOX_CHECK(p.countInputKind.has_value());
    return std::make_unique<GroupbyCountAggregator>(
        p.companionStep,
        p.inputIndex,
        reusableInputKey,
        *p.countInputKind,
        p.resultType);
  } else if (kind.rfind(prefix + "min", 0) == 0) {
    return std::make_unique<GroupbyMinAggregator>(
        p.companionStep,
        p.inputIndex,
        reusableInputKey,
        p.constant,
        p.resultType);
  } else if (kind.rfind(prefix + "max", 0) == 0) {
    return std::make_unique<GroupbyMaxAggregator>(
        p.companionStep,
        p.inputIndex,
        reusableInputKey,
        p.constant,
        p.resultType);
  } else if (kind.rfind(prefix + "avg", 0) == 0) {
    if (p.isDecimalAggregate) {
      return std::make_unique<GroupbyDecimalAvgAggregator>(
          p.companionStep,
          p.inputIndex,
          reusableInputKey,
          p.constant,
          p.resultType);
    }
    return std::make_unique<GroupbyMeanAggregator>(
        p.companionStep,
        p.inputIndex,
        reusableInputKey,
        p.constant,
        p.resultType);
  } else if (kind.rfind(prefix + "stddev_samp", 0) == 0) {
    return std::make_unique<GroupbyStddevSampAggregator>(
        p.companionStep,
        p.inputIndex,
        reusableInputKey,
        p.constant,
        p.resultType);
  } else if (kind.rfind(prefix + "stddev", 0) == 0) {
    // stddev is an alias for stddev_samp
    return std::make_unique<GroupbyStddevSampAggregator>(
        p.companionStep,
        p.inputIndex,
        reusableInputKey,
        p.constant,
        p.resultType);
  } else {
    VELOX_NYI("Aggregation not yet supported, kind: {}", kind);
  }
}

} // namespace

namespace facebook::velox::cudf_velox {

GroupbyAggregationRequestBuilder::GroupbyAggregationRequestBuilder(
    cudf::table_view const& tbl,
    std::vector<cudf::groupby::aggregation_request>& requests,
    rmm::cuda_stream_view stream)
    : tableView_(tbl), requests_(requests), stream_(stream) {}

GroupbyAggregationResultRef
GroupbyAggregationRequestBuilder::addAggregationForValues(
    cudf::column_view values,
    std::unique_ptr<cudf::groupby_aggregation> aggregation) {
  auto& request = requests_.emplace_back();
  request.values = values;
  request.aggregations.push_back(std::move(aggregation));
  return {
      static_cast<uint32_t>(requests_.size() - 1),
      0,
      std::make_shared<bool>(false)};
}

GroupbyAggregationResultRef
GroupbyAggregationRequestBuilder::addAggregationForColumn(
    uint32_t inputIndex,
    std::unique_ptr<cudf::groupby_aggregation> aggregation) {
  return addAggregationForValues(
      tableView_.column(inputIndex), std::move(aggregation));
}

GroupbyAggregationResultRef
GroupbyAggregationRequestBuilder::addReusablePartialSum(
    uint32_t inputIndex,
    uint32_t reusableInputKey) {
  auto it = reusablePartialSums_.find(reusableInputKey);
  if (it != reusablePartialSums_.end()) {
    *it->second.shared = true;
    return it->second;
  }

  auto ref = addAggregationForColumn(
      inputIndex, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  reusablePartialSums_.emplace(reusableInputKey, ref);
  return ref;
}

GroupbyAggregationResultRef GroupbyAggregationRequestBuilder::addSum(
    uint32_t inputIndex,
    uint32_t reusableInputKey,
    core::AggregationNode::Step step) {
  if (step == core::AggregationNode::Step::kPartial) {
    return addReusablePartialSum(inputIndex, reusableInputKey);
  }
  return addAggregationForColumn(
      inputIndex, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
}

GroupbyPartialMeanResultRefs GroupbyAggregationRequestBuilder::addPartialMean(
    uint32_t inputIndex,
    uint32_t reusableInputKey) {
  auto it = reusablePartialSums_.find(reusableInputKey);
  if (it != reusablePartialSums_.end()) {
    *it->second.shared = true;
    return {
        it->second,
        addAggregationForColumn(
            inputIndex,
            cudf::make_count_aggregation<cudf::groupby_aggregation>(
                cudf::null_policy::EXCLUDE))};
  }

  auto& request = requests_.emplace_back();
  request.values = tableView_.column(inputIndex);
  request.aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  GroupbyAggregationResultRef sumRef{
      static_cast<uint32_t>(requests_.size() - 1),
      0,
      std::make_shared<bool>(false)};
  request.aggregations.push_back(
      cudf::make_count_aggregation<cudf::groupby_aggregation>(
          cudf::null_policy::EXCLUDE));
  GroupbyAggregationResultRef countRef{
      static_cast<uint32_t>(requests_.size() - 1),
      1,
      std::make_shared<bool>(false)};
  reusablePartialSums_.emplace(reusableInputKey, sumRef);
  return {sumRef, countRef};
}

cudf::groupby::aggregation_request&
GroupbyAggregationRequestBuilder::aggregationRequest(
    const GroupbyAggregationResultRef& ref) {
  VELOX_CHECK_LT(ref.requestIndex, requests_.size());
  return requests_[ref.requestIndex];
}

std::vector<std::unique_ptr<GroupbyAggregator>> toGroupbyAggregators(
    core::AggregationNode const& aggregationNode,
    core::AggregationNode::Step step,
    TypePtr const& outputType,
    std::vector<VectorPtr> const& constants,
    const std::vector<column_index_t>* aggregationInputChannels) {
  auto params =
      resolveAggregateInfos(aggregationNode, step, outputType, constants);

  std::vector<std::unique_ptr<GroupbyAggregator>> aggregators;
  aggregators.reserve(params.size());
  for (const auto& p : params) {
    auto reusableInputKey = p.inputIndex;
    if (aggregationInputChannels != nullptr) {
      VELOX_CHECK_LT(p.inputIndex, aggregationInputChannels->size());
      reusableInputKey = (*aggregationInputChannels)[p.inputIndex];
    }
    aggregators.push_back(createGroupbyAggregator(p, reusableInputKey));
  }
  return aggregators;
}

bool canGroupbyAggregationBeEvaluatedByCudf(
    const core::CallTypedExpr& call,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& rawInputTypes,
    core::QueryCtx* queryCtx) {
  return canAggregationBeEvaluatedByRegistry(
      getGroupbyAggregationRegistry(), call, step, rawInputTypes, queryCtx);
}

bool canGroupbyBeEvaluatedByCudf(
    const core::AggregationNode& aggregationNode,
    core::QueryCtx* queryCtx) {
  const core::PlanNode* sourceNode = aggregationNode.sources().empty()
      ? nullptr
      : aggregationNode.sources()[0].get();

  // Get the aggregation step from the node
  auto step = aggregationNode.step();

  // Check supported aggregation functions using step-aware aggregation registry
  for (const auto& aggregate : aggregationNode.aggregates()) {
    // Use step-aware validation that handles partial/final/intermediate steps
    if (!canGroupbyAggregationBeEvaluatedByCudf(
            *aggregate.call, step, aggregate.rawInputTypes, queryCtx)) {
      return false;
    }

    // `distinct` aggregations are not supported, in testing fails with "De-dup
    // before aggregation is not yet supported"
    if (aggregate.distinct) {
      return false;
    }

    // `mask` is NOT supported (in testing do not appear to be applied and
    // return incorrect results )
    if (aggregate.mask) {
      return false;
    }

    if (isCountFunctionName(aggregate.call->name())) {
      continue;
    }

    // Check input expressions can be evaluated by cuDF, expand the input first.
    for (const auto& input : aggregate.call->inputs()) {
      auto expandedInput = expandFieldReference(input, sourceNode);
      std::vector<core::TypedExprPtr> exprs = {expandedInput};
      if (!canBeEvaluatedByCudf(exprs, queryCtx)) {
        return false;
      }
    }
  }

  // Check grouping key expressions
  if (!canGroupingKeysBeEvaluatedByCudf(
          aggregationNode.groupingKeys(), sourceNode, queryCtx)) {
    return false;
  }

  return true;
}

CudfGroupby::CudfGroupby(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<core::AggregationNode const> const& aggregationNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          aggregationNode->outputType(),
          aggregationNode->id(),
          std::string{"CudfGroupby"} +
              std::string{
                  core::AggregationNode::toName(aggregationNode->step())},
          nvtx3::rgb{34, 139, 34}, // Forest Green
          NvtxMethodFlag::kAddInput | NvtxMethodFlag::kGetOutput,
          std::nullopt,
          aggregationNode),
      aggregationNode_(aggregationNode),
      isPartialOutput_(
          exec::isPartialOutput(aggregationNode->step()) &&
          !hasFinalAggs(aggregationNode->aggregates())),
      isSingleStep_(
          aggregationNode->step() == core::AggregationNode::Step::kSingle),
      flushGroupIdPartialInput_(
          isPartialOutput_ &&
          !hasCompanionAggregates(aggregationNode->aggregates()) &&
          aggregationNode->sources()[0]->is<core::GroupIdNode>()),
      maxPartialAggregationMemoryUsage_(
          driverCtx->queryConfig().maxPartialAggregationMemoryUsage()) {}

void CudfGroupby::initialize() {
  Operator::initialize();

  inputType_ = aggregationNode_->sources()[0]->outputType();
  ignoreNullKeys_ = aggregationNode_->ignoreNullKeys();
  setupGroupingKeyChannelProjections(
      *aggregationNode_, groupingKeyInputChannels_, groupingKeyOutputChannels_);

  // Velox CPU does optimizations related to pre-grouped keys. This can be
  // done in cudf by passing sort information to cudf::groupby() constructor.
  // We're postponing this for now.

  numAggregates_ = aggregationNode_->aggregates().size();
  const auto inputRowSchema = asRowType(inputType_);
  auto aggregationInput = buildAggregationInputChannels(
      *aggregationNode_,
      *operatorCtx_,
      inputRowSchema,
      groupingKeyInputChannels_);
  aggregationInputChannels_ = std::move(aggregationInput.channels);
  aggregators_ = toGroupbyAggregators(
      *aggregationNode_,
      aggregationNode_->step(),
      outputType_,
      aggregationInput.constants,
      &aggregationInputChannels_);
  nextFinalAggregationHashSeed_ = cudf::DEFAULT_HASH_SEED;
  streamingEnabled_ = !hasCompanionAggregates(aggregationNode_->aggregates());

  // Make aggregators for intermediate step when streaming is enabled.
  if (streamingEnabled_) {
    const bool isFinalOrSingle =
        aggregationNode_->step() == core::AggregationNode::Step::kFinal ||
        aggregationNode_->step() == core::AggregationNode::Step::kSingle;
    bufferedResultType_ = isFinalOrSingle
        ? getBufferedResultType(*aggregationNode_)
        : outputType_;

    std::vector<VectorPtr> nullConstants(numAggregates_);
    intermediateAggregators_ = toGroupbyAggregators(
        *aggregationNode_,
        core::AggregationNode::Step::kIntermediate,
        bufferedResultType_,
        nullConstants);

    if (isSingleStep_) {
      partialAggregators_ = toGroupbyAggregators(
          *aggregationNode_,
          core::AggregationNode::Step::kPartial,
          bufferedResultType_,
          aggregationInput.constants,
          &aggregationInputChannels_);
      finalAggregators_ = toGroupbyAggregators(
          *aggregationNode_,
          core::AggregationNode::Step::kFinal,
          outputType_,
          nullConstants);
    }
  }

  partialAggregationFlushThresholdBytes_ =
      partialAggregationFlushThresholdBytes();
  if (streamingEnabled_) {
    projectedIntermediateBytesPerRow_ =
        projectedIntermediateBytesPerRow(*aggregationNode_);
  }
  if (streamingEnabled_ && isPartialOutput_) {
    // A partial group-by can expand narrow raw input into wider serialized
    // intermediate state. Use the same finite byte target for input work and
    // retained partial output so neither the first aggregation nor the
    // pre-compaction concatenate can grow to an unbounded input batch.
    groupbyInputSliceTargetBytes_ = std::min<uint64_t>(
        finalAggregationBucketTargetBytes(),
        static_cast<uint64_t>(
            std::max<int64_t>(partialAggregationFlushThresholdBytes_, 1)));
    partialAggregationFlushThresholdBytes_ =
        saturateCast(groupbyInputSliceTargetBytes_);
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "cudfGroupbyInputSliceTargetBytes",
        RuntimeCounter(
            saturateCast(groupbyInputSliceTargetBytes_),
            RuntimeCounter::Unit::kBytes));
    lockedStats->addRuntimeStat(
        "cudfGroupbyProjectedIntermediateBytesPerRow",
        RuntimeCounter(
            saturateCast(projectedIntermediateBytesPerRow_),
            RuntimeCounter::Unit::kBytes));
  }

  // Check that aggregate result type match the output type.
  // TODO: This is output schema validation. In velox CPU, it's done using
  // output types reported by aggregation functions. We can't do that in cudf
  // groupby.

  // TODO: Set identity projections used by HashProbe to pushdown dynamic
  // filters to table scan.

  // TODO: Add support for grouping sets and group ids.

  aggregationNode_.reset();
}

uint64_t CudfGroupby::projectedIntermediateBytesPerRow(
    const core::AggregationNode& aggregationNode) const {
  VELOX_CHECK_NOT_NULL(bufferedResultType_);
  const auto numKeys = aggregationNode.groupingKeys().size();
  VELOX_CHECK_GE(bufferedResultType_->size(), numKeys + numAggregates_);

  uint64_t projectedBytes = 0;
  for (size_t i = 0; i < numKeys; ++i) {
    projectedBytes = saturatedAdd(
        projectedBytes,
        conservativePhysicalBytesPerRow(bufferedResultType_->childAt(i)));
  }

  const auto& functionPrefix = CudfConfig::getInstance().functionNamePrefix;
  for (size_t i = 0; i < numAggregates_; ++i) {
    const auto& aggregate = aggregationNode.aggregates()[i];
    const auto functionName = getOriginalName(aggregate.call->name());
    const bool isDecimalState = aggregate.rawInputTypes.size() == 1 &&
        aggregate.rawInputTypes.front()->isDecimal() &&
        (functionName.rfind(functionPrefix + "sum", 0) == 0 ||
         functionName.rfind(functionPrefix + "avg", 0) == 0);
    projectedBytes = saturatedAdd(
        projectedBytes,
        isDecimalState ? kDecimalAggregateStateWorkBytes
                       : conservativePhysicalBytesPerRow(
                             bufferedResultType_->childAt(numKeys + i)));
  }
  return std::max<uint64_t>(projectedBytes, 1);
}

uint64_t CudfGroupby::inputSliceTargetRows(const CudfVector& input) const {
  VELOX_CHECK_GT(input.size(), 0);
  VELOX_CHECK_GT(groupbyInputSliceTargetBytes_, 0);
  VELOX_CHECK_GT(projectedIntermediateBytesPerRow_, 0);

  const auto inputRows = static_cast<uint64_t>(input.size());
  const auto actualInputBytesPerRow = std::max<uint64_t>(
      ceilingDivide(std::max<uint64_t>(input.estimateFlatSize(), 1), inputRows),
      1);
  // The source owner remains resident while its zero-copy views drain. Sum
  // the input and projected state widths anyway to leave room for cuDF's
  // casts, hash table, and result allocations. This is a conservative work
  // budget for fixed-width aggregates, not worker-wide admission control.
  const auto workBytesPerRow =
      saturatedAdd(actualInputBytesPerRow, projectedIntermediateBytesPerRow_);
  uint64_t targetRows =
      std::max<uint64_t>(groupbyInputSliceTargetBytes_ / workBytesPerRow, 1);

  const auto& config = CudfConfig::getInstance();
  if (config.batchSizeMaxThreshold.has_value()) {
    VELOX_CHECK_GT(
        config.batchSizeMaxThreshold.value(),
        0,
        "cuDF max batch size must be positive");
    targetRows = std::min<uint64_t>(
        targetRows,
        static_cast<uint64_t>(config.batchSizeMaxThreshold.value()));
  }
  return std::min<uint64_t>(
      targetRows,
      static_cast<uint64_t>(std::numeric_limits<cudf::size_type>::max()));
}

void CudfGroupby::installPendingInput(CudfVectorPtr input) {
  VELOX_CHECK_NOT_NULL(input);
  VELOX_CHECK_GT(input->size(), 0);
  VELOX_CHECK_NULL(pendingInput_);
  VELOX_CHECK_EQ(pendingInputOffset_, 0);
  VELOX_CHECK_EQ(pendingInputSliceRows_, 0);

  const auto targetRows = inputSliceTargetRows(*input);
  VELOX_CHECK_GT(targetRows, 0);
  pendingInputSliceRows_ = static_cast<vector_size_t>(
      std::min<uint64_t>(targetRows, static_cast<uint64_t>(input->size())));
  pendingInput_ = std::move(input);
}

bool CudfGroupby::shouldFlushBeforeNextPendingInputSlice() const {
  if (!bufferedResult_ || !pendingInput_) {
    return false;
  }

  const auto remainingRows =
      static_cast<uint64_t>(pendingInput_->size() - pendingInputOffset_);
  const auto nextSliceRows = std::min<uint64_t>(
      remainingRows, static_cast<uint64_t>(pendingInputSliceRows_));
  const auto projectedNextStateBytes =
      saturatedMultiply(nextSliceRows, projectedIntermediateBytesPerRow_);
  return saturatedAdd(
             bufferedResult_->estimateFlatSize(), projectedNextStateBytes) >
      static_cast<uint64_t>(partialAggregationFlushThresholdBytes_);
}

void CudfGroupby::processNextPendingInputSlice() {
  VELOX_CHECK_NOT_NULL(pendingInput_);
  VELOX_CHECK_GT(pendingInputSliceRows_, 0);

  const auto ownerRows = pendingInput_->size();
  VELOX_CHECK_LT(pendingInputOffset_, ownerRows);
  const auto begin = pendingInputOffset_;
  const auto end = static_cast<vector_size_t>(std::min<int64_t>(
      static_cast<int64_t>(ownerRows),
      static_cast<int64_t>(begin) + pendingInputSliceRows_));
  const auto sliceRows = end - begin;
  VELOX_CHECK_GT(sliceRows, 0);

  CudfVectorPtr slice;
  if (begin == 0 && end == ownerRows) {
    slice = std::move(pendingInput_);
  } else {
    const auto ownerFlatSize = pendingInput_->estimateFlatSize();
    const auto flatBytesPerRow = ownerFlatSize / ownerRows;
    const auto flatByteRemainder = ownerFlatSize % ownerRows;
    const auto flatSizeAtRow = [&](uint64_t row) {
      return flatBytesPerRow * row + std::min(row, flatByteRemainder);
    };
    auto views = cudf::slice(
        pendingInput_->getTableView(),
        {static_cast<cudf::size_type>(begin),
         static_cast<cudf::size_type>(end)},
        pendingInput_->stream());
    VELOX_CHECK_EQ(views.size(), 1);
    slice = std::make_shared<CudfVector>(
        pool(),
        pendingInput_->type(),
        sliceRows,
        views.front(),
        CudfVector::ViewOwner{pendingInput_},
        pendingInput_->stream(),
        flatSizeAtRow(end) - flatSizeAtRow(begin));
  }

  pendingInputOffset_ = end;
  if (end == ownerRows) {
    pendingInput_.reset();
    pendingInputOffset_ = 0;
    pendingInputSliceRows_ = 0;
  }

  numInputRows_ += sliceRows;
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat("cudfGroupbyInputSlices", RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        "cudfGroupbyInputSliceRows", RuntimeCounter(sliceRows));
    lockedStats->addRuntimeStat(
        "cudfGroupbyMaxInputSliceRows", RuntimeCounter(sliceRows));
  }

  VELOX_CHECK(isPartialOutput_);
  computePartialGroupbyStreaming(std::move(slice));
}

void CudfGroupby::computePartialGroupbyStreaming(CudfVectorPtr tbl) {
  // For every input, we'll do a groupby and compact results with the existing
  // intermediate groupby results.

  auto inputTableStream = tbl->stream();
  // Use getTableView() to avoid expensive materialization for packed_table.
  // tbl stays alive during this function call, keeping the view valid.
  auto permutedInputView = tbl->getTableView().select(
      aggregationInputChannels_.begin(), aggregationInputChannels_.end());
  auto groupbyOnInput = doGroupByAggregation(
      permutedInputView,
      groupingKeyOutputChannels_,
      aggregators_,
      bufferedResultType_,
      inputTableStream);

  if (!groupbyOnInput) {
    return;
  }

  if (flushGroupIdPartialInput_) {
    VELOX_CHECK_NULL(bufferedResult_);
    bufferedResult_ = std::move(groupbyOnInput);
    return;
  }

  // If we already have partial output, concatenate the new results with it.
  if (bufferedResult_) {
    const uint64_t bufferedBytes = bufferedResult_->estimateFlatSize();
    const uint64_t newBytes = groupbyOnInput->estimateFlatSize();
    const uint64_t flushThreshold =
        std::max<int64_t>(partialAggregationFlushThresholdBytes_, 0);
    const auto wouldCrossFlushThreshold = bufferedBytes > flushThreshold ||
        newBytes > flushThreshold - std::min(bufferedBytes, flushThreshold);
    if (wouldCrossFlushThreshold) {
      // addInput() cannot return the existing result. Retain the new result
      // separately and stop accepting input until getOutput() flushes the old
      // result. Checking before concatenate avoids a transient allocation
      // proportional to both states merely to discover after compaction that
      // the result exceeded the limit.
      VELOX_CHECK_NULL(pendingPartialResult_);
      pendingPartialResult_ = std::move(groupbyOnInput);
      pendingPartialInputRows_ = tbl->size();
      auto lockedStats = stats_.wlock();
      lockedStats->addRuntimeStat(
          "cudfPartialAggregationPreemptiveFlush", RuntimeCounter(1));
      return;
    }

    auto partialOutputStream = bufferedResult_->stream();
    std::vector<CudfVectorPtr> tablesToConcat;
    tablesToConcat.push_back(std::move(bufferedResult_));
    tablesToConcat.push_back(std::move(groupbyOnInput));
    auto concatenatedTable = getConcatenatedTable(
        std::exchange(tablesToConcat, {}),
        bufferedResultType_,
        partialOutputStream,
        get_output_mr());

    // Now we have to groupby again but this time with intermediate aggregators.
    // Keep concatenatedTable alive while we use its view.
    auto compactedOutput = doGroupByAggregation(
        concatenatedTable->view(),
        groupingKeyOutputChannels_,
        intermediateAggregators_,
        bufferedResultType_,
        partialOutputStream);
    bufferedResult_ = compactedOutput;
  } else {
    // First time processing, just store the result of the input batch's groupby
    // This means we're storing the stream from the first batch.
    bufferedResult_ = groupbyOnInput;
  }
}

CudfVectorPtr CudfGroupby::compactGroupbyStates(
    std::vector<CudfVectorPtr>&& states,
    bool projectAggregationInputs,
    CudfVectorPtr existingState) {
  if (!existingState && states.empty()) {
    return nullptr;
  }

  if (!projectAggregationInputs && !existingState && states.size() == 1) {
    return std::move(states.front());
  }

  if (existingState && states.empty()) {
    return existingState;
  }

  if (projectAggregationInputs && !existingState && states.size() == 1) {
    auto input = std::move(states.front());
    auto stream = input->stream();
    auto projectedInput = input->getTableView().select(
        aggregationInputChannels_.begin(), aggregationInputChannels_.end());
    return doGroupByAggregation(
        projectedInput,
        groupingKeyOutputChannels_,
        intermediateAggregators_,
        bufferedResultType_,
        stream);
  }

  return aggregateGroupbyStates(
      std::move(states),
      projectAggregationInputs,
      std::move(existingState),
      intermediateAggregators_,
      bufferedResultType_);
}

CudfVectorPtr CudfGroupby::aggregateGroupbyStates(
    std::vector<CudfVectorPtr>&& states,
    bool projectAggregationInputs,
    CudfVectorPtr existingState,
    std::vector<std::unique_ptr<GroupbyAggregator>>& aggregators,
    const TypePtr& outputType) {
  VELOX_CHECK(existingState || !states.empty());

  const bool hasExistingState = existingState != nullptr;
  std::vector<CudfVectorPtr> allVectors;
  allVectors.reserve(states.size() + (hasExistingState ? 1 : 0));
  if (existingState) {
    allVectors.push_back(std::move(existingState));
  }
  for (auto& state : states) {
    allVectors.push_back(std::move(state));
  }

  if (allVectors.size() == 1) {
    auto input = std::move(allVectors.front());
    auto stream = input->stream();
    auto inputView = input->getTableView();
    if (!hasExistingState && projectAggregationInputs) {
      inputView = inputView.select(
          aggregationInputChannels_.begin(), aggregationInputChannels_.end());
    }
    return doGroupByAggregation(
        inputView, groupingKeyOutputChannels_, aggregators, outputType, stream);
  }

  std::vector<cudf::table_view> tableViews;
  std::vector<rmm::cuda_stream_view> inputStreams;
  tableViews.reserve(allVectors.size());
  inputStreams.reserve(allVectors.size());

  for (size_t i = 0; i < allVectors.size(); ++i) {
    VELOX_CHECK_NOT_NULL(allVectors[i]);
    if (i == 0 && hasExistingState) {
      tableViews.push_back(allVectors[i]->getTableView());
    } else if (projectAggregationInputs) {
      tableViews.push_back(
          allVectors[i]->getTableView().select(
              aggregationInputChannels_.begin(),
              aggregationInputChannels_.end()));
    } else {
      tableViews.push_back(allVectors[i]->getTableView());
    }
    inputStreams.push_back(allVectors[i]->stream());
  }

  auto stream = inputStreams.front();
  cudf::detail::join_streams(inputStreams, stream);
  auto concatenatedTable = cudf::concatenate(tableViews, stream, get_temp_mr());
  orderCudfVectorDeallocationsAfterStream(
      std::span<const CudfVectorPtr>(allVectors.data(), allVectors.size()),
      std::span<const rmm::cuda_stream_view>(
          inputStreams.data(), inputStreams.size()),
      stream);

  // concatenate has copied all input columns and the deallocation ordering is
  // established. Release the source states before groupby allocates decoded
  // decimal columns, hash tables, and result columns. With an async memory
  // resource this queues the frees on 'stream', allowing those allocations to
  // reuse the source storage instead of overlapping it for the entire
  // aggregation.
  tableViews.clear();
  allVectors.clear();

  return doGroupByAggregation(
      concatenatedTable->view(),
      groupingKeyOutputChannels_,
      aggregators,
      outputType,
      stream);
}

void CudfGroupby::compactPendingGroupbyStates(bool projectAggregationInputs) {
  auto states = std::move(pendingGroupbyStates_);
  pendingGroupbyStates_.clear();
  pendingGroupbyStateBytes_ = 0;

  int64_t inputRows = bufferedResult_ ? bufferedResult_->size() : 0;
  uint64_t inputBytes =
      bufferedResult_ ? bufferedResult_->estimateFlatSize() : 0;
  for (const auto& state : states) {
    VELOX_CHECK_NOT_NULL(state);
    inputRows += state->size();
    inputBytes = saturatedAdd(inputBytes, state->estimateFlatSize());
  }

  recordFinalAggregationCompactionInput(
      static_cast<uint64_t>(inputRows), inputBytes);
  bufferedResult_ = compactGroupbyStates(
      std::move(states), projectAggregationInputs, std::move(bufferedResult_));
  if (bufferedResult_) {
    pendingGroupbyStateBytes_ = bufferedResult_->estimateFlatSize();
    if (inputRows > 0 && bufferedResult_->size() == inputRows) {
      // There were no duplicate grouping keys to eliminate. Repeating this
      // intermediate compaction against an ever-growing state only adds a
      // concatenate, hash table, and result allocation on each input batch.
      // Keep subsequent states separate and combine them directly in the one
      // final aggregation at noMoreInput().
      intermediateCompactionAbandoned_ = true;
      auto lockedStats = stats_.wlock();
      lockedStats->addRuntimeStat(
          "cudfIntermediateCompactionNoReduction", RuntimeCounter(1));
    }
  }
}

void CudfGroupby::retainPendingGroupbyState(CudfVectorPtr state) {
  if (!state) {
    return;
  }

  pendingGroupbyStateBytes_ += state->estimateFlatSize();
  pendingGroupbyStates_.push_back(std::move(state));
}

void CudfGroupby::addPendingGroupbyState(
    CudfVectorPtr state,
    bool projectAggregationInputs) {
  if (!state) {
    return;
  }
  ++finalAggregationReceivedInputBatches_;
  finalAggregationReceivedInputRows_ = saturatedAdd(
      finalAggregationReceivedInputRows_, static_cast<uint64_t>(state->size()));
  finalAggregationReceivedInputBytes_ = saturatedAdd(
      finalAggregationReceivedInputBytes_, state->estimateFlatSize());

  if (finalAggregationCollecting_) {
    {
      auto lockedStats = stats_.wlock();
      lockedStats->addRuntimeStat(
          "cudfFinalAggregationCollectionInputBatches", RuntimeCounter(1));
    }
    collectFinalAggregationState(std::move(state), projectAggregationInputs);
    return;
  }

  retainPendingGroupbyState(std::move(state));

  // This check is deliberately independent of the ordinary compaction
  // threshold and of intermediateCompactionAbandoned_. A single narrow state
  // can fit the flat-byte threshold while requiring a multi-GiB group-by hash
  // workspace, and a previously observed no-reduction merge can later grow
  // beyond the safe envelope.
  if (!groupingKeyOutputChannels_.empty() &&
      pendingGroupbyStatesExceedFinalBucketEnvelope()) {
    startFinalAggregationCollection(projectAggregationInputs);
    return;
  }

  if (!intermediateCompactionAbandoned_ &&
      (bufferedResult_ || pendingGroupbyStates_.size() > 1) &&
      pendingGroupbyStateBytes_ >= maxPartialAggregationMemoryUsage_) {
    compactPendingGroupbyStates(projectAggregationInputs);
  }
}

bool CudfGroupby::pendingGroupbyStatesExceedFinalBucketEnvelope() const {
  uint64_t rows =
      bufferedResult_ ? static_cast<uint64_t>(bufferedResult_->size()) : 0;
  uint64_t bytes = bufferedResult_ ? bufferedResult_->estimateFlatSize() : 0;
  for (const auto& state : pendingGroupbyStates_) {
    VELOX_CHECK_NOT_NULL(state);
    rows = saturatedAdd(rows, static_cast<uint64_t>(state->size()));
    bytes = saturatedAdd(bytes, state->estimateFlatSize());
  }

  const auto targetBytes = finalAggregationBucketTargetBytes();
  const auto targetRows =
      finalAggregationBucketTargetRows(rows, bytes, targetBytes);
  return rows > targetRows ||
      finalAggregationWorkBytes(rows, bytes) > targetBytes;
}

void CudfGroupby::initializeFinalAggregationBucketEnvelope(
    uint64_t inputRows,
    uint64_t inputBytes) {
  VELOX_CHECK_EQ(finalAggregationBucketTargetRows_, 0);
  VELOX_CHECK_EQ(finalAggregationBucketTargetBytes_, 0);
  finalAggregationBucketTargetBytes_ = finalAggregationBucketTargetBytes();
  finalAggregationBucketTargetRows_ = finalAggregationBucketTargetRows(
      inputRows, inputBytes, finalAggregationBucketTargetBytes_);

  auto lockedStats = stats_.wlock();
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationBucketTargetRows",
      RuntimeCounter(finalAggregationBucketTargetRows_));
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationBucketTargetBytes",
      RuntimeCounter(
          finalAggregationBucketTargetBytes_, RuntimeCounter::Unit::kBytes));
}

void CudfGroupby::recordFinalAggregationCompactionInput(
    uint64_t inputRows,
    uint64_t inputBytes) {
  const auto workBytes = finalAggregationWorkBytes(inputRows, inputBytes);
  auto lockedStats = stats_.wlock();
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationMaxCompactionInputRows", RuntimeCounter(inputRows));
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationMaxCompactionInputBytes",
      RuntimeCounter(inputBytes, RuntimeCounter::Unit::kBytes));
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationMaxCompactionWorkBytes",
      RuntimeCounter(workBytes, RuntimeCounter::Unit::kBytes));
}

void CudfGroupby::recordFinalAggregationRetainedState() {
  uint64_t rows = 0;
  uint64_t bytes = 0;
  uint64_t runs = 0;
  for (const auto& bucket : finalAggregationCollectionBuckets_) {
    rows = saturatedAdd(rows, bucket.rows);
    bytes = saturatedAdd(bytes, bucket.bytes);
    runs = saturatedAdd(runs, bucket.numStates());
  }

  auto lockedStats = stats_.wlock();
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationMaxRetainedRows", RuntimeCounter(rows));
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationMaxRetainedBytes",
      RuntimeCounter(bytes, RuntimeCounter::Unit::kBytes));
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationMaxRetainedRuns", RuntimeCounter(runs));
}

CudfGroupby::ParkedFinalAggregationState CudfGroupby::parkFinalAggregationState(
    CudfVectorPtr state) {
  VELOX_CHECK_NOT_NULL(state);
  const auto rows = static_cast<uint64_t>(state->size());
  const auto deviceBytes = state->estimateFlatSize();
  auto stream = state->stream();
  auto hostState = with_arrow::toVeloxColumn(
      state->getTableView(),
      pool(),
      bufferedResultType_,
      "",
      stream,
      get_temp_mr());

  // toVeloxColumn returns after its device-to-host copies complete. Publish
  // the device owner's stream-ordered frees before another driver asks the
  // shared device pool for that memory.
  hostState->setType(bufferedResultType_);
  state.reset();
  stream.synchronize();

  finalAggregationHostParkedRows_ =
      saturatedAdd(finalAggregationHostParkedRows_, rows);
  finalAggregationHostParkedBytes_ =
      saturatedAdd(finalAggregationHostParkedBytes_, deviceBytes);
  auto lockedStats = stats_.wlock();
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationHostParkedRuns", RuntimeCounter(1));
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationHostParkedRows", RuntimeCounter(rows));
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationHostParkedBytes",
      RuntimeCounter(deviceBytes, RuntimeCounter::Unit::kBytes));
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationMaxHostParkedRows",
      RuntimeCounter(finalAggregationHostParkedRows_));
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationMaxHostParkedBytes",
      RuntimeCounter(
          finalAggregationHostParkedBytes_, RuntimeCounter::Unit::kBytes));
  return {std::move(hostState), deviceBytes};
}

CudfVectorPtr CudfGroupby::restoreFinalAggregationState(
    ParkedFinalAggregationState state,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK_NOT_NULL(state.state);
  const auto rows = static_cast<uint64_t>(state.state->size());
  VELOX_CHECK_GT(finalAggregationBucketTargetRows_, 0);
  VELOX_CHECK_GT(finalAggregationBucketTargetBytes_, 0);
  VELOX_CHECK_LE(
      rows,
      finalAggregationBucketTargetRows_,
      "A single parked final-aggregation run exceeds the row envelope");
  VELOX_CHECK_LE(
      finalAggregationWorkBytes(rows, state.deviceBytes),
      finalAggregationBucketTargetBytes_,
      "A single parked final-aggregation run exceeds the work envelope");
  auto table =
      with_arrow::toCudfTable(state.state, pool(), stream, get_temp_mr());
  state.state.reset();

  VELOX_CHECK_GE(finalAggregationHostParkedRows_, rows);
  VELOX_CHECK_GE(finalAggregationHostParkedBytes_, state.deviceBytes);
  finalAggregationHostParkedRows_ -= rows;
  finalAggregationHostParkedBytes_ -= state.deviceBytes;
  auto lockedStats = stats_.wlock();
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationHostRestoredRuns", RuntimeCounter(1));
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationHostRestoredRows", RuntimeCounter(rows));
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationHostRestoredBytes",
      RuntimeCounter(state.deviceBytes, RuntimeCounter::Unit::kBytes));

  return std::make_shared<CudfVector>(
      pool(),
      bufferedResultType_,
      static_cast<vector_size_t>(rows),
      std::move(table),
      stream);
}

void CudfGroupby::parkFinalAggregationBucket(FinalAggregationBucket& bucket) {
  auto states = std::move(bucket.states);
  bucket.states.clear();
  bucket.parkedStates.reserve(bucket.parkedStates.size() + states.size());
  for (auto& state : states) {
    bucket.parkedStates.push_back(parkFinalAggregationState(std::move(state)));
  }
}

void CudfGroupby::restoreFinalAggregationBucket(
    FinalAggregationBucket& bucket) {
  auto parkedStates = std::move(bucket.parkedStates);
  bucket.parkedStates.clear();
  bucket.states.reserve(bucket.states.size() + parkedStates.size());
  auto stream = cudfGlobalStreamPool().get_stream();
  for (auto& state : parkedStates) {
    bucket.states.push_back(
        restoreFinalAggregationState(std::move(state), stream));
  }
}

void CudfGroupby::startFinalAggregationCollection(
    bool projectAggregationInputs) {
  VELOX_CHECK(!finalAggregationCollecting_);
  VELOX_CHECK(finalAggregationCollectionBuckets_.empty());
  VELOX_CHECK(!groupingKeyOutputChannels_.empty());

  uint64_t inputRows =
      bufferedResult_ ? static_cast<uint64_t>(bufferedResult_->size()) : 0;
  uint64_t inputBytes =
      bufferedResult_ ? bufferedResult_->estimateFlatSize() : 0;
  for (const auto& state : pendingGroupbyStates_) {
    VELOX_CHECK_NOT_NULL(state);
    inputRows = saturatedAdd(inputRows, static_cast<uint64_t>(state->size()));
    inputBytes = saturatedAdd(inputBytes, state->estimateFlatSize());
  }
  initializeFinalAggregationBucketEnvelope(inputRows, inputBytes);

  // Aim for half-full leaves so later fragments can be admitted without
  // immediately growing the fanout. The hard envelope is still checked
  // before every admission and again when each bucket drains.
  const auto leafTargetRows =
      std::max<uint64_t>(finalAggregationBucketTargetRows_ / 2, 1);
  const auto leafTargetWorkBytes =
      std::max<uint64_t>(finalAggregationBucketTargetBytes_ / 2, 1);
  const auto requiredFanout = std::max(
      ceilingDivide(inputRows, leafTargetRows),
      ceilingDivide(
          finalAggregationWorkBytes(inputRows, inputBytes),
          leafTargetWorkBytes));
  finalAggregationCollectionFanout_ = roundUpToPowerOfTwo(
      std::max<uint64_t>(requiredFanout, 2),
      kMaxFinalAggregationPartitionFanout);
  finalAggregationCollectionHashSeed_ =
      advanceHashSeed(nextFinalAggregationHashSeed_);
  finalAggregationCollectionBuckets_.resize(finalAggregationCollectionFanout_);
  finalAggregationCollecting_ = true;

  auto existingState = std::move(bufferedResult_);
  auto states = std::move(pendingGroupbyStates_);
  pendingGroupbyStates_.clear();
  pendingGroupbyStateBytes_ = 0;

  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationCollectionTransitions", RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationCollectionInputBatches",
        RuntimeCounter(finalAggregationReceivedInputBatches_));
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationCollectionFanout",
        RuntimeCounter(finalAggregationCollectionFanout_));
    if (inputBytes > finalAggregationBucketTargetBytes_) {
      lockedStats->addRuntimeStat(
          "cudfFinalAggregationCollectionByteTransitions", RuntimeCounter(1));
    }
    if (finalAggregationWorkBytes(inputRows, inputBytes) >
        finalAggregationBucketTargetBytes_) {
      lockedStats->addRuntimeStat(
          "cudfFinalAggregationCollectionWorkTransitions", RuntimeCounter(1));
    }
    if (inputRows > finalAggregationBucketTargetRows_) {
      lockedStats->addRuntimeStat(
          "cudfFinalAggregationCollectionRowTransitions", RuntimeCounter(1));
    }
  }

  if (existingState) {
    collectFinalAggregationState(std::move(existingState), false);
  }
  for (auto& state : states) {
    collectFinalAggregationState(std::move(state), projectAggregationInputs);
  }
  recordFinalAggregationRetainedState();
}

void CudfGroupby::growFinalAggregationCollection() {
  VELOX_CHECK(finalAggregationCollecting_);
  VELOX_CHECK_LT(
      finalAggregationCollectionFanout_, kMaxFinalAggregationPartitionFanout);

  auto oldBuckets = std::move(finalAggregationCollectionBuckets_);
  finalAggregationCollectionFanout_ = std::min<uint32_t>(
      finalAggregationCollectionFanout_ * 2,
      kMaxFinalAggregationPartitionFanout);
  finalAggregationCollectionBuckets_.clear();
  finalAggregationCollectionBuckets_.resize(finalAggregationCollectionFanout_);

  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationCollectionGrows", RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationCollectionFanout",
        RuntimeCounter(finalAggregationCollectionFanout_));
  }

  // Rehash one owning state at a time. This bounds the transient copy during
  // fanout growth without rebuilding a complete parked bucket on the GPU.
  for (auto& bucket : oldBuckets) {
    routeFinalAggregationBucket(std::move(bucket));
  }
  recordFinalAggregationRetainedState();
}

void CudfGroupby::collectFinalAggregationSlice(
    CudfVectorPtr state,
    bool projectAggregationInputs) {
  VELOX_CHECK(finalAggregationCollecting_);
  VELOX_CHECK_NOT_NULL(state);
  VELOX_CHECK_LE(
      static_cast<uint64_t>(state->size()), finalAggregationBucketTargetRows_);
  VELOX_CHECK_LE(
      finalAggregationWorkBytes(
          static_cast<uint64_t>(state->size()), state->estimateFlatSize()),
      finalAggregationBucketTargetBytes_);

  std::vector<FinalAggregationBucket> incoming(
      finalAggregationCollectionFanout_);
  hashPartitionFinalState(
      std::move(state),
      projectAggregationInputs,
      finalAggregationCollectionFanout_,
      finalAggregationCollectionHashSeed_,
      0,
      incoming,
      true);

  bool needsGrowth = false;
  for (size_t i = 0; i < incoming.size(); ++i) {
    const auto combinedRows = saturatedAdd(
        finalAggregationCollectionBuckets_[i].rows, incoming[i].rows);
    const auto combinedBytes = saturatedAdd(
        finalAggregationCollectionBuckets_[i].bytes, incoming[i].bytes);
    if (combinedRows > finalAggregationBucketTargetRows_ ||
        finalAggregationWorkBytes(combinedRows, combinedBytes) >
            finalAggregationBucketTargetBytes_) {
      needsGrowth = true;
      break;
    }
  }

  if (needsGrowth &&
      finalAggregationCollectionFanout_ < kMaxFinalAggregationPartitionFanout) {
    growFinalAggregationCollection();
    for (auto& bucket : incoming) {
      routeFinalAggregationBucket(std::move(bucket));
    }
    return;
  }

  for (size_t i = 0; i < incoming.size(); ++i) {
    auto& incomingBucket = incoming[i];
    if (incomingBucket.empty()) {
      continue;
    }
    auto& resident = finalAggregationCollectionBuckets_[i];
    const auto combinedRows = saturatedAdd(resident.rows, incomingBucket.rows);
    const auto combinedBytes =
        saturatedAdd(resident.bytes, incomingBucket.bytes);

    if (combinedRows > finalAggregationBucketTargetRows_ ||
        finalAggregationWorkBytes(combinedRows, combinedBytes) >
            finalAggregationBucketTargetBytes_) {
      // At the flat fanout cap, retain bounded, independently reduced runs.
      // Drain-time recursive partitioning can refine this one key domain with
      // a new seed without ever launching the oversized merge here.
      resident.rows = combinedRows;
      resident.bytes = combinedBytes;
      for (auto& incomingState : incomingBucket.states) {
        resident.states.push_back(std::move(incomingState));
      }
      for (auto& incomingState : incomingBucket.parkedStates) {
        resident.parkedStates.push_back(std::move(incomingState));
      }
      auto lockedStats = stats_.wlock();
      lockedStats->addRuntimeStat(
          "cudfFinalAggregationCollectionSealedRuns", RuntimeCounter(1));
      continue;
    }

    resident.rows = combinedRows;
    resident.bytes = combinedBytes;
    resident.hashDepth = incomingBucket.hashDepth;
    for (auto& incomingState : incomingBucket.states) {
      resident.states.push_back(std::move(incomingState));
    }
    for (auto& incomingState : incomingBucket.parkedStates) {
      resident.parkedStates.push_back(std::move(incomingState));
    }

    // Do not compact an accumulating bucket here. Folding each later input
    // into a resident state repeatedly rewrites all previously collected rows.
    // Retain disjoint parked runs and aggregate them once when this bounded
    // bucket drains. Pre-admission fanout growth and recursive drain-time
    // partitioning continue to enforce the row and workspace envelope.
  }
  recordFinalAggregationRetainedState();
}

void CudfGroupby::collectFinalAggregationState(
    CudfVectorPtr state,
    bool projectAggregationInputs) {
  VELOX_CHECK(finalAggregationCollecting_);
  if (!state) {
    return;
  }
  auto slices = splitFinalAggregationState(std::move(state));
  for (auto& slice : slices) {
    collectFinalAggregationSlice(std::move(slice), projectAggregationInputs);
  }
}

void CudfGroupby::routeFinalAggregationState(CudfVectorPtr state) {
  collectFinalAggregationState(std::move(state), false);
}

void CudfGroupby::routeFinalAggregationBucket(FinalAggregationBucket&& bucket) {
  VELOX_CHECK(
      bucket.states.empty(),
      "Final-aggregation collection buckets must remain parked between "
      "admissions");
  auto parkedStates = std::move(bucket.parkedStates);
  auto restoreStream = cudfGlobalStreamPool().get_stream();
  for (auto& state : parkedStates) {
    routeFinalAggregationState(
        restoreFinalAggregationState(std::move(state), restoreStream));
  }
}

void CudfGroupby::computeFinalGroupbyStreaming(CudfVectorPtr tbl) {
  addPendingGroupbyState(std::move(tbl), true);
}

void CudfGroupby::computeSingleGroupbyStreaming(CudfVectorPtr tbl) {
  auto inputTableStream = tbl->stream();
  auto permutedInputView = tbl->getTableView().select(
      aggregationInputChannels_.begin(), aggregationInputChannels_.end());
  auto groupbyOnInput = doGroupByAggregation(
      permutedInputView,
      groupingKeyOutputChannels_,
      partialAggregators_,
      bufferedResultType_,
      inputTableStream);

  addPendingGroupbyState(std::move(groupbyOnInput), false);
}

uint64_t CudfGroupby::finalAggregationBucketTargetBytes() const {
  uint64_t targetBytes = 1ULL << 30;
  if (auto info = currentDeviceMemoryInfo(); info.has_value()) {
    // Final aggregation can transiently hold the retained intermediate states,
    // one concatenated bucket, decoded decimal request columns, the groupby
    // hash table, and its output. Limit each bucket to a small, fixed fraction
    // of device memory so those allocations cannot converge on the full input.
    targetBytes = info->totalBytes / 64;
  }
  return std::clamp(
      targetBytes,
      kMinFinalAggregationBucketBytes,
      kMaxFinalAggregationBucketBytes);
}

uint64_t CudfGroupby::finalAggregationBucketTargetRows(
    uint64_t inputRows,
    uint64_t inputBytes,
    uint64_t targetBytes) const {
  VELOX_CHECK_GT(targetBytes, 0);
  uint64_t bytesPerRow = 1;
  if (inputRows > 0 && inputBytes > 0) {
    bytesPerRow = std::max<uint64_t>(ceilingDivide(inputBytes, inputRows), 1);
  }
  bytesPerRow = saturatedAdd(bytesPerRow, projectedIntermediateBytesPerRow_);
  bytesPerRow = saturatedAdd(bytesPerRow, kHashGroupbyWorkspaceBytesPerRow);

  uint64_t targetRows = std::max<uint64_t>(targetBytes / bytesPerRow, 1);
  const auto& config = CudfConfig::getInstance();
  if (config.batchSizeMaxThreshold.has_value()) {
    VELOX_CHECK_GT(
        config.batchSizeMaxThreshold.value(),
        0,
        "cuDF max batch size must be positive");
    targetRows = std::min<uint64_t>(
        targetRows,
        static_cast<uint64_t>(config.batchSizeMaxThreshold.value()));
  }
  return std::min<uint64_t>(
      targetRows,
      static_cast<uint64_t>(std::numeric_limits<cudf::size_type>::max()));
}

uint64_t CudfGroupby::finalAggregationWorkBytes(
    uint64_t inputRows,
    uint64_t inputBytes) const {
  const auto perRowWorkBytes = saturatedAdd(
      projectedIntermediateBytesPerRow_, kHashGroupbyWorkspaceBytesPerRow);
  return saturatedAdd(
      inputBytes, saturatedMultiply(inputRows, perRowWorkBytes));
}

uint32_t CudfGroupby::finalAggregationPartitionCount(
    uint64_t rows,
    uint64_t bytes,
    bool requireSplit) const {
  VELOX_CHECK_GT(finalAggregationBucketTargetRows_, 0);
  VELOX_CHECK_GT(finalAggregationBucketTargetBytes_, 0);
  auto partitions = std::max(
      ceilingDivide(rows, finalAggregationBucketTargetRows_),
      ceilingDivide(
          finalAggregationWorkBytes(rows, bytes),
          finalAggregationBucketTargetBytes_));
  if (requireSplit) {
    partitions = std::max<uint64_t>(partitions, 2);
  }
  return roundUpToPowerOfTwo(
      std::max<uint64_t>(partitions, 1), kMaxFinalAggregationPartitionFanout);
}

bool CudfGroupby::finalAggregationBucketIsOversized(
    const FinalAggregationBucket& bucket) const {
  return bucket.rows > finalAggregationBucketTargetRows_ ||
      finalAggregationWorkBytes(bucket.rows, bucket.bytes) >
      finalAggregationBucketTargetBytes_;
}

std::vector<CudfVectorPtr> CudfGroupby::splitFinalAggregationState(
    CudfVectorPtr state) const {
  VELOX_CHECK_NOT_NULL(state);
  VELOX_CHECK_GT(finalAggregationBucketTargetRows_, 0);
  VELOX_CHECK_GT(finalAggregationBucketTargetBytes_, 0);

  const auto rows = static_cast<uint64_t>(state->size());
  if (rows == 0) {
    return {};
  }
  const auto bytes = state->estimateFlatSize();
  const auto estimatedFlatBytesPerRow =
      std::max<uint64_t>(ceilingDivide(std::max<uint64_t>(bytes, 1), rows), 1);
  const auto workBytesPerRow = saturatedAdd(
      estimatedFlatBytesPerRow,
      saturatedAdd(
          projectedIntermediateBytesPerRow_, kHashGroupbyWorkspaceBytesPerRow));
  const auto rowsByBytes = std::max<uint64_t>(
      finalAggregationBucketTargetBytes_ / workBytesPerRow, 1);
  const auto rowsPerSlice =
      std::min(finalAggregationBucketTargetRows_, rowsByBytes);
  if (rows <= rowsPerSlice) {
    VELOX_CHECK_LE(
        finalAggregationWorkBytes(rows, bytes),
        finalAggregationBucketTargetBytes_,
        "A single final-aggregation row exceeds the bounded work envelope");
    std::vector<CudfVectorPtr> result;
    result.push_back(std::move(state));
    return result;
  }

  const auto ownerView = state->getTableView();
  VELOX_CHECK_EQ(static_cast<uint64_t>(ownerView.num_rows()), rows);
  const auto flatBytesPerRow = bytes / rows;
  const auto flatByteRemainder = bytes % rows;
  const auto flatSizeAtRow = [&](uint64_t row) {
    return flatBytesPerRow * row + std::min(row, flatByteRemainder);
  };

  std::vector<CudfVectorPtr> slices;
  slices.reserve(ceilingDivide(rows, rowsPerSlice));
  for (uint64_t start = 0; start < rows; start += rowsPerSlice) {
    const auto end = std::min(start + rowsPerSlice, rows);
    auto views = cudf::slice(
        ownerView,
        {static_cast<cudf::size_type>(start),
         static_cast<cudf::size_type>(end)},
        state->stream());
    VELOX_CHECK_EQ(views.size(), 1);
    slices.push_back(
        std::make_shared<CudfVector>(
            pool(),
            state->type(),
            static_cast<vector_size_t>(end - start),
            views.front(),
            CudfVector::ViewOwner{state},
            state->stream(),
            flatSizeAtRow(end) - flatSizeAtRow(start)));
    VELOX_CHECK_LE(
        finalAggregationWorkBytes(
            end - start, flatSizeAtRow(end) - flatSizeAtRow(start)),
        finalAggregationBucketTargetBytes_,
        "A single final-aggregation row exceeds the bounded work envelope");
  }
  return slices;
}

void CudfGroupby::hashPartitionFinalState(
    CudfVectorPtr state,
    bool projectAggregationInputs,
    uint32_t numBuckets,
    uint32_t hashSeed,
    uint32_t hashDepth,
    std::vector<FinalAggregationBucket>& buckets,
    bool materializeBuckets) {
  VELOX_CHECK_NOT_NULL(state);
  VELOX_CHECK_GT(numBuckets, 1);
  VELOX_CHECK_EQ(buckets.size(), numBuckets);
  VELOX_CHECK(!groupingKeyOutputChannels_.empty());

  const auto inputRows = state->size();
  auto boundedStates = splitFinalAggregationState(std::move(state));
  std::vector<cudf::size_type> hashColumns(
      groupingKeyOutputChannels_.begin(), groupingKeyOutputChannels_.end());

  for (auto& boundedState : boundedStates) {
    auto stream = boundedState->stream();
    auto inputView = boundedState->getTableView();
    if (projectAggregationInputs) {
      inputView = inputView.select(
          aggregationInputChannels_.begin(), aggregationInputChannels_.end());
    }

    auto [partitionedTable, offsets] = cudf::hash_partition(
        inputView,
        hashColumns,
        static_cast<int>(numBuckets),
        cudf::hash_id::HASH_MURMUR3,
        hashSeed,
        stream,
        get_output_mr());
    VELOX_CHECK_NOT_NULL(partitionedTable);
    VELOX_CHECK_EQ(offsets.size(), numBuckets + 1);

    if (materializeBuckets) {
      // Persistent bucket views into one device table would keep every sibling
      // resident until the last bucket drains. Move the whole partitioned run
      // to host once, then make cheap host slices for its buckets. This avoids
      // one D2H synchronization per bucket while releasing the complete device
      // owner before another input is admitted.
      const auto partitionedInputRows = boundedState->size();
      boundedState.reset();
      auto partitioned = std::make_shared<CudfVector>(
          pool(),
          bufferedResultType_,
          partitionedInputRows,
          std::move(partitionedTable),
          stream);
      const auto partitionedRows = static_cast<uint64_t>(partitioned->size());
      const auto partitionedBytes = partitioned->estimateFlatSize();
      const auto flatBytesPerRow = partitionedBytes / partitionedRows;
      const auto flatByteRemainder = partitionedBytes % partitionedRows;
      const auto flatSizeAtRow = [&](uint64_t row) {
        return flatBytesPerRow * row + std::min(row, flatByteRemainder);
      };
      auto parkedPartitioned =
          parkFinalAggregationState(std::move(partitioned));
      auto hostRun = std::move(parkedPartitioned.state);

      for (uint32_t bucketIndex = 0; bucketIndex < numBuckets; ++bucketIndex) {
        const auto begin = static_cast<uint64_t>(offsets[bucketIndex]);
        const auto end = static_cast<uint64_t>(offsets[bucketIndex + 1]);
        if (begin == end) {
          continue;
        }
        const auto bucketRows = end - begin;
        const auto bucketBytes = flatSizeAtRow(end) - flatSizeAtRow(begin);
        auto hostSlice = std::dynamic_pointer_cast<RowVector>(hostRun->slice(
            static_cast<vector_size_t>(begin),
            static_cast<vector_size_t>(bucketRows)));
        VELOX_CHECK_NOT_NULL(hostSlice);
        auto& bucket = buckets[bucketIndex];
        bucket.rows = saturatedAdd(bucket.rows, bucketRows);
        bucket.bytes = saturatedAdd(bucket.bytes, bucketBytes);
        bucket.hashDepth = hashDepth;
        bucket.parkedStates.push_back({std::move(hostSlice), bucketBytes});
      }
      continue;
    }

    auto partitioned = std::make_shared<CudfVector>(
        pool(),
        bufferedResultType_,
        boundedState->size(),
        std::move(partitionedTable),
        stream);
    const auto partitionedRows = static_cast<uint64_t>(partitioned->size());
    const auto partitionedBytes = partitioned->estimateFlatSize();
    const auto flatBytesPerRow =
        partitionedRows == 0 ? 0 : partitionedBytes / partitionedRows;
    const auto flatByteRemainder =
        partitionedRows == 0 ? 0 : partitionedBytes % partitionedRows;
    const auto flatSizeAtRow = [&](uint64_t row) {
      return flatBytesPerRow * row + std::min(row, flatByteRemainder);
    };

    const auto partitionedView = partitioned->getTableView();
    for (uint32_t bucketIndex = 0; bucketIndex < numBuckets; ++bucketIndex) {
      const auto begin = static_cast<uint64_t>(offsets[bucketIndex]);
      const auto end = static_cast<uint64_t>(offsets[bucketIndex + 1]);
      if (begin == end) {
        continue;
      }
      auto views = cudf::slice(
          partitionedView,
          {offsets[bucketIndex], offsets[bucketIndex + 1]},
          stream);
      VELOX_CHECK_EQ(views.size(), 1);
      auto& bucket = buckets[bucketIndex];
      const auto bucketRows = end - begin;
      const auto bucketBytes = flatSizeAtRow(end) - flatSizeAtRow(begin);
      bucket.rows = saturatedAdd(bucket.rows, bucketRows);
      bucket.bytes = saturatedAdd(bucket.bytes, bucketBytes);
      bucket.hashDepth = hashDepth;
      bucket.states.push_back(
          std::make_shared<CudfVector>(
              pool(),
              bufferedResultType_,
              static_cast<vector_size_t>(bucketRows),
              views.front(),
              CudfVector::ViewOwner{partitioned},
              stream,
              bucketBytes));
    }
  }

  auto lockedStats = stats_.wlock();
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationHashPartitionedRows", RuntimeCounter(inputRows));
}

void CudfGroupby::initializeFinalAggregationBuckets(
    std::vector<CudfVectorPtr>&& states,
    CudfVectorPtr existingState,
    bool projectAggregationInputs,
    uint64_t inputRows,
    uint64_t inputBytes) {
  VELOX_CHECK(!groupingKeyOutputChannels_.empty());
  VELOX_CHECK(finalAggregationBuckets_.empty());
  hashBucketFinalization_ = true;

  const auto numBuckets =
      finalAggregationPartitionCount(inputRows, inputBytes, true);
  std::vector<FinalAggregationBucket> buckets(numBuckets);
  const auto hashSeed = advanceHashSeed(nextFinalAggregationHashSeed_);
  if (existingState) {
    hashPartitionFinalState(
        std::move(existingState),
        false,
        numBuckets,
        hashSeed,
        0,
        buckets,
        true);
  }
  for (auto& state : states) {
    hashPartitionFinalState(
        std::move(state),
        projectAggregationInputs,
        numBuckets,
        hashSeed,
        0,
        buckets,
        true);
  }

  uint64_t maxBucketRows = 0;
  uint64_t nonEmptyBuckets = 0;
  for (auto& bucket : buckets) {
    if (bucket.empty()) {
      continue;
    }
    maxBucketRows = std::max(maxBucketRows, bucket.rows);
    ++nonEmptyBuckets;
    finalAggregationBuckets_.push_back(std::move(bucket));
  }

  auto lockedStats = stats_.wlock();
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationHashBuckets", RuntimeCounter(nonEmptyBuckets));
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationMaxBucketRows", RuntimeCounter(maxBucketRows));
}

CudfGroupby::FinalAggregationBucket
CudfGroupby::compactSkewedFinalAggregationBucket(
    FinalAggregationBucket&& bucket) {
  VELOX_CHECK(
      bucket.states.empty(),
      "Oversized final-aggregation buckets must be parked before skew "
      "compaction");
  FinalAggregationBucket compacted;
  compacted.hashDepth = bucket.hashDepth;

  std::vector<CudfVectorPtr> chunk;
  uint64_t chunkRows = 0;
  uint64_t chunkBytes = 0;
  auto flushChunk = [&]() {
    if (chunk.empty()) {
      return;
    }
    recordFinalAggregationCompactionInput(chunkRows, chunkBytes);
    auto result = aggregateGroupbyStates(
        std::exchange(chunk, {}),
        false,
        nullptr,
        intermediateAggregators_,
        bufferedResultType_);
    chunkRows = 0;
    chunkBytes = 0;
    if (!result) {
      return;
    }
    compacted.rows =
        saturatedAdd(compacted.rows, static_cast<uint64_t>(result->size()));
    compacted.bytes = saturatedAdd(compacted.bytes, result->estimateFlatSize());
    compacted.states.push_back(std::move(result));
    parkFinalAggregationBucket(compacted);
  };

  auto consumeState = [&](CudfVectorPtr state) {
    auto slices = splitFinalAggregationState(std::move(state));
    for (auto& slice : slices) {
      const auto rows = static_cast<uint64_t>(slice->size());
      const auto bytes = slice->estimateFlatSize();
      const auto combinedRows = saturatedAdd(chunkRows, rows);
      const auto combinedBytes = saturatedAdd(chunkBytes, bytes);
      const bool crossesLimit = !chunk.empty() &&
          (combinedRows > finalAggregationBucketTargetRows_ ||
           finalAggregationWorkBytes(combinedRows, combinedBytes) >
               finalAggregationBucketTargetBytes_);
      if (crossesLimit) {
        flushChunk();
      }
      chunkRows = saturatedAdd(chunkRows, rows);
      chunkBytes = saturatedAdd(chunkBytes, bytes);
      chunk.push_back(std::move(slice));
    }
  };

  // An oversized bucket can contain far more parked data than fits in the
  // device envelope. Restore and consume one owning run at a time instead of
  // reconstructing the complete bucket on the GPU before chunking it.
  auto parkedStates = std::move(bucket.parkedStates);
  auto restoreStream = cudfGlobalStreamPool().get_stream();
  for (auto& state : parkedStates) {
    VELOX_CHECK_NOT_NULL(state.state);
    const auto rows = static_cast<uint64_t>(state.state->size());
    VELOX_CHECK_LE(rows, finalAggregationBucketTargetRows_);
    VELOX_CHECK_LE(
        finalAggregationWorkBytes(rows, state.deviceBytes),
        finalAggregationBucketTargetBytes_,
        "A single parked final-aggregation run exceeds the bounded work "
        "envelope");
    if (!chunk.empty() &&
        (saturatedAdd(chunkRows, rows) > finalAggregationBucketTargetRows_ ||
         finalAggregationWorkBytes(
             saturatedAdd(chunkRows, rows),
             saturatedAdd(chunkBytes, state.deviceBytes)) >
             finalAggregationBucketTargetBytes_)) {
      // Flush before H2D restoration so a full parked run never overlaps a
      // nearly full compaction chunk on the device.
      flushChunk();
    }
    consumeState(restoreFinalAggregationState(std::move(state), restoreStream));
  }
  flushChunk();

  auto lockedStats = stats_.wlock();
  lockedStats->addRuntimeStat(
      "cudfFinalAggregationSkewCompactions", RuntimeCounter(1));
  return compacted;
}

void CudfGroupby::repartitionFinalAggregationBucket(
    FinalAggregationBucket&& bucket) {
  VELOX_CHECK(
      bucket.states.empty(),
      "Oversized final-aggregation buckets must be parked before "
      "repartitioning");
  const auto originalRows = bucket.rows;
  const auto originalBytes = bucket.bytes;
  const auto originalDepth = bucket.hashDepth;
  if (originalDepth >= kMaxFinalAggregationHashDepth) {
    auto compacted = compactSkewedFinalAggregationBucket(std::move(bucket));
    if (compacted.rows < originalRows) {
      finalAggregationBuckets_.push_front(std::move(compacted));
      return;
    }
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationSkewGuards", RuntimeCounter(1));
    VELOX_FAIL(
        "cuDF final aggregation exceeded its bounded hash depth without "
        "reducing the oversized bucket: depth={}, rows={}, bytes={}, "
        "targetRows={}, targetBytes={}",
        originalDepth,
        originalRows,
        originalBytes,
        finalAggregationBucketTargetRows_,
        finalAggregationBucketTargetBytes_);
  }
  const auto childDepth = originalDepth + 1;
  const auto numBuckets =
      finalAggregationPartitionCount(originalRows, originalBytes, true);
  std::vector<FinalAggregationBucket> children(numBuckets);
  const auto hashSeed = advanceHashSeed(nextFinalAggregationHashSeed_);
  auto partitionState = [&](CudfVectorPtr state) {
    hashPartitionFinalState(
        std::move(state),
        false,
        numBuckets,
        hashSeed,
        childDepth,
        children,
        true);
  };

  // Repartition oversized host state one owning run at a time. Each run is
  // released by hashPartitionFinalState before the next restore, bounding the
  // transient H2D allocation independently of the complete bucket size.
  auto parkedStates = std::move(bucket.parkedStates);
  auto restoreStream = cudfGlobalStreamPool().get_stream();
  for (auto& state : parkedStates) {
    partitionState(
        restoreFinalAggregationState(std::move(state), restoreStream));
  }

  uint64_t maxChildRows = 0;
  size_t largestChild = 0;
  for (size_t i = 0; i < children.size(); ++i) {
    if (children[i].rows > maxChildRows) {
      maxChildRows = children[i].rows;
      largestChild = i;
    }
  }

  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationHashRepartitions", RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationMaxBucketRows", RuntimeCounter(maxChildRows));
  }

  if (maxChildRows == originalRows) {
    // Ordinary key skew hashes all copies of a hot key to the same child. A
    // bounded intermediate aggregation can collapse those copies before the
    // child is finalized. If no rows collapse, retry with a new hash seed;
    // this covers an unlucky collision between otherwise distinct keys.
    auto compacted =
        compactSkewedFinalAggregationBucket(std::move(children[largestChild]));
    if (compacted.rows < originalRows) {
      finalAggregationBuckets_.push_front(std::move(compacted));
      return;
    }
    if (childDepth >= kMaxFinalAggregationHashDepth) {
      auto lockedStats = stats_.wlock();
      lockedStats->addRuntimeStat(
          "cudfFinalAggregationSkewGuards", RuntimeCounter(1));
      VELOX_FAIL(
          "cuDF final aggregation could not bound a hash bucket after {} "
          "independent partitions: rows={}, bytes={}, targetRows={}, "
          "targetBytes={}",
          childDepth,
          originalRows,
          originalBytes,
          finalAggregationBucketTargetRows_,
          finalAggregationBucketTargetBytes_);
    }
    finalAggregationBuckets_.push_front(std::move(compacted));
    return;
  }

  // Preserve deterministic bucket order while putting the newly refined
  // children ahead of unrelated work already in the queue.
  for (auto it = children.rbegin(); it != children.rend(); ++it) {
    if (!it->empty()) {
      finalAggregationBuckets_.push_front(std::move(*it));
    }
  }
}

CudfVectorPtr CudfGroupby::getNextFinalAggregationBucket() {
  VELOX_CHECK(hashBucketFinalization_);
  while (!finalAggregationBuckets_.empty()) {
    auto bucket = std::move(finalAggregationBuckets_.front());
    finalAggregationBuckets_.pop_front();
    if (finalAggregationBucketIsOversized(bucket)) {
      repartitionFinalAggregationBucket(std::move(bucket));
      continue;
    }

    auto& finalAggregators = isSingleStep_ ? finalAggregators_ : aggregators_;
    recordFinalAggregationCompactionInput(bucket.rows, bucket.bytes);
    restoreFinalAggregationBucket(bucket);
    auto result = aggregateGroupbyStates(
        std::move(bucket.states),
        false,
        nullptr,
        finalAggregators,
        outputType_);
    if (!result) {
      continue;
    }
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationBucketOutputs", RuntimeCounter(1));
    return result;
  }
  return nullptr;
}

CudfVectorPtr CudfGroupby::finalizeCollectedGroupbyStates() {
  VELOX_CHECK(finalAggregationCollecting_);
  VELOX_CHECK(finalAggregationBuckets_.empty());
  hashBucketFinalization_ = true;

  uint64_t inputBatches = 0;
  uint64_t inputRows = 0;
  uint64_t inputBytes = 0;
  uint64_t maxBucketRows = 0;
  uint64_t nonEmptyBuckets = 0;
  for (auto& bucket : finalAggregationCollectionBuckets_) {
    if (bucket.empty()) {
      continue;
    }
    inputBatches = saturatedAdd(inputBatches, bucket.numStates());
    inputRows = saturatedAdd(inputRows, bucket.rows);
    inputBytes = saturatedAdd(inputBytes, bucket.bytes);
    maxBucketRows = std::max(maxBucketRows, bucket.rows);
    ++nonEmptyBuckets;
    finalAggregationBuckets_.push_back(std::move(bucket));
  }
  finalAggregationCollectionBuckets_.clear();
  finalAggregationCollecting_ = false;

  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationDirectInputBatches",
        RuntimeCounter(finalAggregationReceivedInputBatches_));
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationDirectInputRows",
        RuntimeCounter(finalAggregationReceivedInputRows_));
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationDirectInputBytes",
        RuntimeCounter(
            finalAggregationReceivedInputBytes_, RuntimeCounter::Unit::kBytes));
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationRetainedInputBatches",
        RuntimeCounter(inputBatches));
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationRetainedInputRows", RuntimeCounter(inputRows));
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationRetainedInputBytes",
        RuntimeCounter(inputBytes, RuntimeCounter::Unit::kBytes));
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationHashBuckets", RuntimeCounter(nonEmptyBuckets));
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationMaxBucketRows", RuntimeCounter(maxBucketRows));
  }
  return getNextFinalAggregationBucket();
}

CudfVectorPtr CudfGroupby::finalizePendingGroupbyStates() {
  finalAggregationInitialized_ = true;
  if (finalAggregationCollecting_) {
    return finalizeCollectedGroupbyStates();
  }

  auto states = std::move(pendingGroupbyStates_);
  pendingGroupbyStates_.clear();
  pendingGroupbyStateBytes_ = 0;

  if (!bufferedResult_ && states.empty()) {
    return nullptr;
  }

  int64_t inputBatches = bufferedResult_ ? 1 : 0;
  int64_t inputRows = bufferedResult_ ? bufferedResult_->size() : 0;
  int64_t inputBytes = bufferedResult_
      ? static_cast<int64_t>(bufferedResult_->estimateFlatSize())
      : 0;
  for (const auto& state : states) {
    VELOX_CHECK_NOT_NULL(state);
    ++inputBatches;
    inputRows += state->size();
    inputBytes += static_cast<int64_t>(state->estimateFlatSize());
  }
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationDirectInputBatches", RuntimeCounter(inputBatches));
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationDirectInputRows", RuntimeCounter(inputRows));
    lockedStats->addRuntimeStat(
        "cudfFinalAggregationDirectInputBytes",
        RuntimeCounter(inputBytes, RuntimeCounter::Unit::kBytes));
  }

  initializeFinalAggregationBucketEnvelope(inputRows, inputBytes);
  const bool requiresHashBuckets = !groupingKeyOutputChannels_.empty() &&
      (static_cast<uint64_t>(inputRows) > finalAggregationBucketTargetRows_ ||
       finalAggregationWorkBytes(
           static_cast<uint64_t>(inputRows),
           static_cast<uint64_t>(inputBytes)) >
           finalAggregationBucketTargetBytes_);
  if (requiresHashBuckets) {
    initializeFinalAggregationBuckets(
        std::move(states),
        std::move(bufferedResult_),
        !isSingleStep_,
        inputRows,
        inputBytes);
    return getNextFinalAggregationBucket();
  }

  auto& finalAggregators = isSingleStep_ ? finalAggregators_ : aggregators_;
  recordFinalAggregationCompactionInput(
      static_cast<uint64_t>(inputRows), static_cast<uint64_t>(inputBytes));
  return aggregateGroupbyStates(
      std::move(states),
      !isSingleStep_,
      std::move(bufferedResult_),
      finalAggregators,
      outputType_);
}

int64_t CudfGroupby::partialAggregationFlushThresholdBytes() const {
  // Velox's default partial aggregation threshold is tuned for CPU hash tables.
  // cuDF grouped state already lives on the GPU and can be much larger; using
  // the CPU default causes high-cardinality partial aggregations to emit many
  // mostly-duplicate states that the final aggregation must merge again.
  constexpr int64_t kTinyExplicitThresholdBytes = 1 << 20;
  constexpr int64_t kFallbackGpuThresholdBytes = 4LL << 30;
  constexpr int64_t kMaxGpuThresholdBytes = 8LL << 30;

  if (maxPartialAggregationMemoryUsage_ < kTinyExplicitThresholdBytes) {
    return maxPartialAggregationMemoryUsage_;
  }

  int64_t gpuThreshold = kFallbackGpuThresholdBytes;
  if (auto info = currentDeviceMemoryInfo(); info.has_value()) {
    gpuThreshold = std::min<int64_t>(
        kMaxGpuThresholdBytes, static_cast<int64_t>(info->totalBytes / 16));
  }

  return std::max(maxPartialAggregationMemoryUsage_, gpuThreshold);
}

void CudfGroupby::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }

  auto cudfInput = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);

  if (streamingEnabled_) {
    if (isPartialOutput_) {
      // Install a one-owner cursor. Process its first slice now unless the
      // prior partial state must be emitted first; getOutput() advances the
      // remaining slices while needsInput() stays false.
      installPendingInput(std::move(cudfInput));
      if (shouldFlushBeforeNextPendingInputSlice()) {
        auto lockedStats = stats_.wlock();
        lockedStats->addRuntimeStat(
            "cudfPartialAggregationDeferredInput", RuntimeCounter(1));
      } else {
        processNextPendingInputSlice();
      }
      return;
    }
    if (isSingleStep_) {
      numInputRows_ += cudfInput->size();
      computeSingleGroupbyStreaming(std::move(cudfInput));
      return;
    }
    numInputRows_ += cudfInput->size();
    computeFinalGroupbyStreaming(std::move(cudfInput));
    return;
  }

  // Handle non-streaming cases.
  numInputRows_ += cudfInput->size();
  inputs_.push_back(std::move(cudfInput));
}

CudfVectorPtr CudfGroupby::doGroupByAggregation(
    cudf::table_view tableView,
    std::vector<column_index_t> const& groupByKeys,
    std::vector<std::unique_ptr<GroupbyAggregator>>& aggregators,
    TypePtr const& outputType,
    rmm::cuda_stream_view stream) {
  auto groupbyKeyView =
      tableView.select(groupByKeys.begin(), groupByKeys.end());

  // TODO: All other args to groupby are related to sort groupby. We don't
  // support optimizations related to it yet.
  cudf::groupby::groupby groupByOwner(
      groupbyKeyView,
      ignoreNullKeys_ ? cudf::null_policy::EXCLUDE
                      : cudf::null_policy::INCLUDE);

  std::vector<cudf::groupby::aggregation_request> requests;
  GroupbyAggregationRequestBuilder builder(tableView, requests, stream);
  for (auto& aggregator : aggregators) {
    aggregator->addGroupbyRequest(builder);
  }

  auto releaseRequestInputs = [&]() {
    for (auto& aggregator : aggregators) {
      aggregator->releaseRequestInputs();
    }
  };
  auto aggregateResult = [&]() {
    try {
      return groupByOwner.aggregate(requests, stream, get_output_mr());
    } catch (...) {
      releaseRequestInputs();
      throw;
    }
  }();
  releaseRequestInputs();
  auto [groupKeys, results] = std::move(aggregateResult);
  // flatten the results
  std::vector<std::unique_ptr<cudf::column>> resultColumns;

  // first fill the grouping keys
  auto groupKeysColumns = groupKeys->release();
  resultColumns.insert(
      resultColumns.begin(),
      std::make_move_iterator(groupKeysColumns.begin()),
      std::make_move_iterator(groupKeysColumns.end()));

  // then fill the aggregation results
  for (auto& aggregator : aggregators) {
    resultColumns.push_back(aggregator->makeOutputColumn(results, stream));
  }

  // make a cudf table out of columns
  auto resultTable = std::make_unique<cudf::table>(std::move(resultColumns));

  auto numRows = resultTable->num_rows();

  // velox expects nullptr instead of a table with 0 rows
  if (numRows == 0) {
    return nullptr;
  }

  return std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType, numRows, std::move(resultTable), stream);
}

CudfVectorPtr CudfGroupby::releaseAndResetBufferedResult() {
  auto numOutputRows = bufferedResult_->size();
  const double aggregationPct =
      numOutputRows == 0 ? 0 : (numOutputRows * 1.0) / numInputRows_ * 100;
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        std::string(exec::HashAggregation::kFlushRowCount),
        RuntimeCounter(numOutputRows));
    lockedStats->addRuntimeStat(
        std::string(exec::HashAggregation::kFlushTimes), RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        std::string(exec::HashAggregation::kPartialAggregationPct),
        RuntimeCounter(aggregationPct));
  }

  numInputRows_ = 0;
  // We're moving bufferedResult_ to the caller because we want it to be null
  // after this call.
  return std::move(bufferedResult_);
}

CudfVectorPtr CudfGroupby::getPartialAggregationOutput() {
  VELOX_CHECK(isPartialOutput_ && streamingEnabled_);

  const auto releaseBufferedResult = [&](bool lastOutput) {
    auto output = releaseAndResetBufferedResult();
    if (lastOutput) {
      finished_ = true;
    }
    return output;
  };
  const auto isLastBufferedOutput = [&]() {
    return noMoreInput_ && !pendingInput_ && !pendingPartialResult_;
  };

  // GroupId input views must be reduced and emitted independently. The next
  // cursor slice is not processed until this result has left the operator.
  if (flushGroupIdPartialInput_ && bufferedResult_) {
    return releaseBufferedResult(isLastBufferedOutput());
  }

  if (pendingPartialResult_) {
    VELOX_CHECK_NOT_NULL(bufferedResult_);
    VELOX_CHECK_GT(pendingPartialInputRows_, 0);
    VELOX_CHECK_GE(numInputRows_, pendingPartialInputRows_);
    numInputRows_ -= pendingPartialInputRows_;
    auto output = releaseAndResetBufferedResult();
    bufferedResult_ = std::move(pendingPartialResult_);
    numInputRows_ = std::exchange(pendingPartialInputRows_, 0);
    return output;
  }

  if (bufferedResult_ &&
      bufferedResult_->estimateFlatSize() >
          partialAggregationFlushThresholdBytes_) {
    return releaseBufferedResult(isLastBufferedOutput());
  }

  if (shouldFlushBeforeNextPendingInputSlice()) {
    // Do not retain the previous state while launching the next slice's
    // group-by when their projected outputs already exceed the budget.
    {
      auto lockedStats = stats_.wlock();
      lockedStats->addRuntimeStat(
          "cudfPartialAggregationPreSliceFlush", RuntimeCounter(1));
    }
    return releaseBufferedResult(false);
  }

  // noMoreInput() may be called directly while a sliced owner is pending.
  // Retain the current partial state until every slice has been processed.
  if (!noMoreInput_ || pendingInput_) {
    return nullptr;
  }

  if (!bufferedResult_) {
    finished_ = true;
    return nullptr;
  }
  return releaseBufferedResult(true);
}

RowVectorPtr CudfGroupby::doGetOutput() {
  // Partial output has priority over advancing the input cursor. This bounds
  // both retained state and GroupId expansion before another slice is reduced.
  if (isPartialOutput_ && streamingEnabled_) {
    if (auto output = getPartialAggregationOutput()) {
      return output;
    }
    if (finished_) {
      return nullptr;
    }
    if (pendingInput_) {
      processNextPendingInputSlice();
      return getPartialAggregationOutput();
    }
    return nullptr;
  }

  if (finished_) {
    return nullptr;
  }

  if (!isPartialOutput_ && !noMoreInput_) {
    // Final aggregation has to wait for all batches to arrive so we cannot
    // return any results here.
    return nullptr;
  }

  // Streaming finalization: single step uses finalAggregators_ to convert
  // intermediate results to final output; final step uses aggregators_.
  // At this point isPartialOutput_ is false (handled above) and noMoreInput_
  // is true (guarded by the check above).
  if (streamingEnabled_) {
    auto result = finalAggregationInitialized_ ? getNextFinalAggregationBucket()
                                               : finalizePendingGroupbyStates();
    if (!result) {
      finished_ = true;
      return nullptr;
    }
    if (!hashBucketFinalization_ || finalAggregationBuckets_.empty()) {
      finished_ = true;
    }
    auto stream = result->stream();
    stream.synchronize();
    return result;
  }

  if (inputs_.empty() && !noMoreInput_) {
    return nullptr;
  }

  auto stream = cudfGlobalStreamPool().get_stream();

  auto tbl = getConcatenatedTable(
      std::exchange(inputs_, {}), inputType_, stream, get_output_mr());

  // Release input data after synchronizing.
  stream.synchronize();
  inputs_.clear();

  if (noMoreInput_) {
    finished_ = true;
  }

  VELOX_CHECK_NOT_NULL(tbl);

  auto permutedInputView = tbl->view().select(
      aggregationInputChannels_.begin(), aggregationInputChannels_.end());
  return doGroupByAggregation(
      permutedInputView,
      groupingKeyOutputChannels_,
      aggregators_,
      outputType_,
      stream);
}

void CudfGroupby::doNoMoreInput() {
  Operator::noMoreInput();
  if (isPartialOutput_ && inputs_.empty() && !pendingInput_ &&
      !pendingPartialResult_ && !bufferedResult_) {
    finished_ = true;
  }
}

void CudfGroupby::doClose() {
  inputs_.clear();
  pendingInput_.reset();
  pendingInputOffset_ = 0;
  pendingInputSliceRows_ = 0;
  pendingGroupbyStates_.clear();
  finalAggregationCollectionBuckets_.clear();
  finalAggregationBuckets_.clear();
  pendingPartialResult_.reset();
  bufferedResult_.reset();
  aggregators_.clear();
  intermediateAggregators_.clear();
  partialAggregators_.clear();
  finalAggregators_.clear();
  Operator::close();
}

bool CudfGroupby::isFinished() {
  return finished_;
}

} // namespace facebook::velox::cudf_velox
