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
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/HashAggregation.h"
#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"

#include <algorithm>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/unary.hpp>

namespace {

using namespace facebook::velox;
using cudf_velox::CountInputKind;
using cudf_velox::get_output_mr;
using cudf_velox::get_temp_mr;
using cudf_velox::GroupbyAggregationRequestBuilder;
using cudf_velox::GroupbyAggregationResultRef;
using cudf_velox::GroupbyAggregator;
using cudf_velox::GroupbyPartialMeanResultRefs;
using cudf_velox::ResolvedAggregateInfo;

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
      const auto cudfType =                                              \
          cudf::data_type(cudf_velox::veloxToCudfTypeId(resultType));    \
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
    const auto cudfType =
        cudf::data_type(cudf_velox::veloxToCudfTypeId(resultType));
    if (col->type() != cudfType) {
      col = cudf::cast(*col, cudfType, stream, get_output_mr());
    }
    return col;
  }

 private:
  GroupbyAggregationResultRef outputRef_;
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
    const auto cudfOutputType =
        cudf::data_type(cudf_velox::veloxToCudfTypeId(resultType));
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
        auto const cudfSumType = cudf::data_type(
            cudf_velox::veloxToCudfTypeId(outputType->childAt(0)));
        auto const cudfCountType = cudf::data_type(
            cudf_velox::veloxToCudfTypeId(outputType->childAt(1)));
        if (sum->type() != cudf::data_type(cudfSumType)) {
          sum = cudf::cast(
              *sum, cudf::data_type(cudfSumType), stream, get_output_mr());
        }
        if (count->type() != cudf::data_type(cudfCountType)) {
          count = cudf::cast(
              *count, cudf::data_type(cudfCountType), stream, get_output_mr());
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
        auto const cudfSumType = cudf::data_type(
            cudf_velox::veloxToCudfTypeId(outputType->childAt(0)));
        auto const cudfCountType = cudf::data_type(
            cudf_velox::veloxToCudfTypeId(outputType->childAt(1)));
        if (sum->type() != cudf::data_type(cudfSumType)) {
          sum = cudf::cast(
              *sum, cudf::data_type(cudfSumType), stream, get_output_mr());
        }
        if (count->type() != cudf::data_type(cudfCountType)) {
          count = cudf::cast(
              *count, cudf::data_type(cudfCountType), stream, get_output_mr());
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
            cudf::data_type(cudf_velox::veloxToCudfTypeId(resultType)),
            stream,
            get_output_mr());
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
        auto const cudfCountType = cudf::data_type(
            cudf_velox::veloxToCudfTypeId(outputType->childAt(0)));
        auto const cudfMeanType = cudf::data_type(
            cudf_velox::veloxToCudfTypeId(outputType->childAt(1)));
        auto const cudfM2Type = cudf::data_type(
            cudf_velox::veloxToCudfTypeId(outputType->childAt(2)));

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
        cudf::data_type(cudf_velox::veloxToCudfTypeId(outputType->childAt(0)));
    auto const cudfMeanType =
        cudf::data_type(cudf_velox::veloxToCudfTypeId(outputType->childAt(1)));
    auto const cudfM2Type =
        cudf::data_type(cudf_velox::veloxToCudfTypeId(outputType->childAt(2)));

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
    std::vector<cudf::groupby::aggregation_request>& requests)
    : tableView_(tbl), requests_(requests) {}

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

    // `mask` is NOT supported (in testing do not appear to be be applied and
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
  partialAggregationFlushThresholdBytes_ =
      partialAggregationFlushThresholdBytes();
  streamingEnabled_ = !hasCompanionAggregates(aggregationNode_->aggregates());
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "cudfPartialAggregationFlushThresholdBytes",
        RuntimeCounter(
            partialAggregationFlushThresholdBytes_,
            RuntimeCounter::Unit::kBytes));
  }

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

  // Check that aggregate result type match the output type.
  // TODO: This is output schema validation. In velox CPU, it's done using
  // output types reported by aggregation functions. We can't do that in cudf
  // groupby.

  // TODO: Set identity projections used by HashProbe to pushdown dynamic
  // filters to table scan.

  // TODO: Add support for grouping sets and group ids.

  aggregationNode_.reset();
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

  // If we already have partial output, concatenate the new results with it.
  if (bufferedResult_) {
    auto partialOutputStream = bufferedResult_->stream();
    std::vector<CudfVectorPtr> tablesToConcat;
    tablesToConcat.push_back(bufferedResult_);
    tablesToConcat.push_back(groupbyOnInput);
    auto concatenatedTable = getConcatenatedTable(
        std::move(tablesToConcat),
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

  const bool hasExistingState = existingState != nullptr;
  std::vector<CudfVectorPtr> allVectors;
  allVectors.reserve(states.size() + (hasExistingState ? 1 : 0));
  if (existingState) {
    allVectors.push_back(std::move(existingState));
  }
  for (auto& state : states) {
    allVectors.push_back(std::move(state));
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
      tableViews.push_back(allVectors[i]->getTableView().select(
          aggregationInputChannels_.begin(), aggregationInputChannels_.end()));
    } else {
      tableViews.push_back(allVectors[i]->getTableView());
    }
    inputStreams.push_back(allVectors[i]->stream());
  }

  auto stream = inputStreams.front();
  cudf::detail::join_streams(inputStreams, stream);
  auto concatenatedTable =
      cudf::concatenate(tableViews, stream, get_temp_mr());
  orderCudfVectorDeallocationsAfterStream(
      std::span<const CudfVectorPtr>(allVectors.data(), allVectors.size()),
      std::span<const rmm::cuda_stream_view>(
          inputStreams.data(), inputStreams.size()),
      stream);

  return doGroupByAggregation(
      concatenatedTable->view(),
      groupingKeyOutputChannels_,
      intermediateAggregators_,
      bufferedResultType_,
      stream);
}

void CudfGroupby::compactPendingGroupbyStates(
    bool projectAggregationInputs) {
  auto states = std::move(pendingGroupbyStates_);
  pendingGroupbyStates_.clear();
  pendingGroupbyStateBytes_ = 0;

  bufferedResult_ = compactGroupbyStates(
      std::move(states), projectAggregationInputs, std::move(bufferedResult_));
  if (bufferedResult_) {
    pendingGroupbyStateBytes_ = bufferedResult_->estimateFlatSize();
  }
}

void CudfGroupby::addPendingGroupbyState(
    CudfVectorPtr state,
    bool projectAggregationInputs) {
  if (!state) {
    return;
  }

  pendingGroupbyStateBytes_ += state->estimateFlatSize();
  pendingGroupbyStates_.push_back(std::move(state));
  if ((bufferedResult_ || pendingGroupbyStates_.size() > 1) &&
      pendingGroupbyStateBytes_ >= maxPartialAggregationMemoryUsage_) {
    compactPendingGroupbyStates(projectAggregationInputs);
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
  numInputRows_ += input->size();

  auto cudfInput = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);

  if (streamingEnabled_) {
    if (isPartialOutput_) {
      computePartialGroupbyStreaming(cudfInput);
      return;
    } else if (isSingleStep_) {
      computeSingleGroupbyStreaming(cudfInput);
      return;
    } else {
      computeFinalGroupbyStreaming(cudfInput);
      return;
    }
  }

  // Handle non-streaming cases.
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
  GroupbyAggregationRequestBuilder builder(tableView, requests);
  for (auto& aggregator : aggregators) {
    aggregator->addGroupbyRequest(builder);
  }

  auto [groupKeys, results] =
      groupByOwner.aggregate(requests, stream, get_output_mr());
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

RowVectorPtr CudfGroupby::doGetOutput() {
  // Handle partial streaming groupby.
  if (isPartialOutput_ && streamingEnabled_) {
    if (bufferedResult_ &&
        bufferedResult_->estimateFlatSize() >
            partialAggregationFlushThresholdBytes_) {
      return releaseAndResetBufferedResult();
    }
    if (not noMoreInput_) {
      // Don't produce output if the partial output hasn't reached memory limit
      // and there's more batches to come.
      return nullptr;
    }
    if (!bufferedResult_ && finished_) {
      return nullptr;
    }
    return releaseAndResetBufferedResult();
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
    finished_ = true;
    if (!pendingGroupbyStates_.empty()) {
      compactPendingGroupbyStates(!isSingleStep_);
    }
    if (!bufferedResult_) {
      return nullptr;
    }
    auto& aggs = isSingleStep_ ? finalAggregators_ : aggregators_;
    auto stream = bufferedResult_->stream();
    auto result = doGroupByAggregation(
        bufferedResult_->getTableView(),
        groupingKeyOutputChannels_,
        aggs,
        outputType_,
        stream);
    stream.synchronize();
    bufferedResult_.reset();
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
  if (isPartialOutput_ && inputs_.empty()) {
    finished_ = true;
  }
}

bool CudfGroupby::isFinished() {
  return finished_;
}

} // namespace facebook::velox::cudf_velox
