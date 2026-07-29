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
#include "velox/experimental/cudf/exec/CudfGroupId.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>

#include <algorithm>

namespace facebook::velox::cudf_velox {
namespace {

/// Composite owner for a GroupId output table view. CudfVector retains this
/// object because the view references all three kinds of storage.
struct GroupIdOutputOwner {
  CudfVectorPtr input;
  std::shared_ptr<cudf::table> nullColumns;
  std::unique_ptr<cudf::table> generatedColumns;
};

} // namespace

CudfGroupId::CudfGroupId(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::GroupIdNode>& groupIdNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          groupIdNode->outputType(),
          groupIdNode->id(),
          "CudfGroupId",
          nvtx3::rgb{128, 0, 128}, // Purple
          NvtxMethodFlag::kAddInput | NvtxMethodFlag::kGetOutput,
          std::nullopt,
          groupIdNode) {
  const auto& inputType = groupIdNode->sources()[0]->outputType();

  // Build output-to-input mapping for grouping keys
  std::unordered_map<column_index_t, column_index_t>
      outputToInputGroupingKeyMapping;
  for (const auto& groupingKeyInfo : groupIdNode->groupingKeyInfos()) {
    outputToInputGroupingKeyMapping[outputType_->getChildIdx(
        groupingKeyInfo.output)] =
        inputType->getChildIdx(groupingKeyInfo.input->name());
  }

  numGroupingKeys_ = groupIdNode->numGroupingKeys();
  numGroupingSets_ = groupIdNode->groupingSets().size();

  // Build groupingKeyMappings_ - one vector per grouping set
  groupingKeyMappings_.reserve(numGroupingSets_);
  for (const auto& groupingSet : groupIdNode->groupingSets()) {
    std::vector<column_index_t> mappings(numGroupingKeys_, kMissingGroupingKey);
    for (const auto& groupingKey : groupingSet) {
      auto outputChannel = outputType_->getChildIdx(groupingKey);
      VELOX_USER_CHECK_NE(
          outputToInputGroupingKeyMapping.count(outputChannel),
          0,
          "GroupIdNode didn't map grouping key {} to input channel",
          groupingKey);
      VELOX_CHECK_LT(
          outputChannel,
          outputToInputGroupingKeyMapping.size(),
          "outputChannel out of bounds in outputToInputGroupingKeyMapping");
      auto inputChannel = outputToInputGroupingKeyMapping.at(outputChannel);
      VELOX_CHECK_LT(
          outputChannel,
          mappings.size(),
          "outputChannel out of bounds in mappings");
      mappings[outputChannel] = inputChannel;
    }
    groupingKeyMappings_.emplace_back(std::move(mappings));
  }

  // Build aggregationInputs_ - indices of aggregation input columns
  const auto& aggregationInputs = groupIdNode->aggregationInputs();
  aggregationInputs_.reserve(aggregationInputs.size());
  for (const auto& input : aggregationInputs) {
    aggregationInputs_.push_back(inputType->getChildIdx(input->name()));
  }

  // Precompute cudf data types for grouping key columns (used to create
  // all-null columns for keys not in a grouping set).
  groupingKeyCudfTypes_.reserve(numGroupingKeys_);
  for (size_t i = 0; i < numGroupingKeys_; ++i) {
    groupingKeyCudfTypes_.push_back(
        veloxToCudfDataType(outputType_->childAt(i)));
  }

  // Cache at most one all-null column per grouping key for each input batch.
  // A ROLLUP with N keys otherwise creates N * (N + 1) / 2 null columns.
  nullColumnMappings_.assign(numGroupingKeys_, kNoNullColumn);
  column_index_t nextNullColumn = 0;
  for (size_t key = 0; key < numGroupingKeys_; ++key) {
    const auto isMissing = std::any_of(
        groupingKeyMappings_.begin(),
        groupingKeyMappings_.end(),
        [key](const auto& mapping) {
          return mapping[key] == kMissingGroupingKey;
        });
    if (isMissing) {
      nullColumnMappings_[key] = nextNullColumn++;
    }
  }
}

bool CudfGroupId::needsInput() const {
  return !noMoreInput_ && !input_;
}

void CudfGroupId::doAddInput(RowVectorPtr input) {
  VELOX_CHECK_NULL(input_, "Previous input not fully consumed");

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput, "CudfGroupId expects CudfVector input");

  inputStream_ = cudfInput->stream();
  // Normalize packed input deallocation to the stream advertised by the
  // vector before outputs begin sharing its backing storage.
  VELOX_CHECK(
      cudfInput->rebindStream(inputStream_),
      "CudfGroupId input storage cannot be retained on its input stream");
  input_ = std::move(cudfInput);

  std::vector<std::unique_ptr<cudf::column>> nullColumns;
  nullColumns.reserve(std::count_if(
      nullColumnMappings_.begin(), nullColumnMappings_.end(), [](auto index) {
        return index != kNoNullColumn;
      }));
  for (size_t key = 0; key < numGroupingKeys_; ++key) {
    if (nullColumnMappings_[key] == kNoNullColumn) {
      continue;
    }
    auto nullScalar = cudf::make_default_constructed_scalar(
        groupingKeyCudfTypes_[key], inputStream_, get_temp_mr());
    nullColumns.push_back(cudf::make_column_from_scalar(
        *nullScalar, input_->size(), inputStream_, get_output_mr()));
  }
  nullColumns_ = std::make_shared<cudf::table>(std::move(nullColumns));
  groupingSetIndex_ = 0;
}

RowVectorPtr CudfGroupId::doGetOutput() {
  if (!input_) {
    return nullptr;
  }

  auto stream = inputStream_;
  auto outputMr = get_output_mr();
  auto tempMr = get_temp_mr();
  auto numRows = static_cast<cudf::size_type>(input_->size());
  const auto inputView = input_->getTableView();
  const auto nullColumnsView = nullColumns_->view();

  VELOX_CHECK_LT(
      groupingSetIndex_,
      groupingKeyMappings_.size(),
      "groupingSetIndex_ out of bounds");
  const auto& mapping = groupingKeyMappings_[groupingSetIndex_];
  std::vector<cudf::column_view> outputColumns(outputType_->size());

  // Fill in grouping keys
  for (size_t i = 0; i < numGroupingKeys_; ++i) {
    if (mapping[i] == kMissingGroupingKey) {
      VELOX_CHECK_NE(nullColumnMappings_[i], kNoNullColumn);
      outputColumns[i] = nullColumnsView.column(nullColumnMappings_[i]);
    } else {
      VELOX_CHECK_LT(mapping[i], inputView.num_columns());
      outputColumns[i] = inputView.column(mapping[i]);
    }
  }

  // Fill in aggregation inputs
  for (size_t i = 0; i < aggregationInputs_.size(); ++i) {
    auto inputIdx = aggregationInputs_[i];
    auto outputIdx = numGroupingKeys_ + i;
    VELOX_CHECK_LT(
        inputIdx,
        inputView.num_columns(),
        "inputIdx out of bounds in aggregation inputs");
    VELOX_CHECK_LT(
        outputIdx,
        outputColumns.size(),
        "outputIdx out of bounds in aggregation inputs");
    outputColumns[outputIdx] = inputView.column(inputIdx);
  }

  // Add group_id column (constant BIGINT with current grouping set index)
  auto groupIdScalar = cudf::numeric_scalar<int64_t>(
      static_cast<int64_t>(groupingSetIndex_), true, stream, tempMr);
  std::vector<std::unique_ptr<cudf::column>> generatedColumns;
  generatedColumns.push_back(
      cudf::make_column_from_scalar(groupIdScalar, numRows, stream, outputMr));
  auto generatedTable =
      std::make_unique<cudf::table>(std::move(generatedColumns));
  outputColumns.back() = generatedTable->view().column(0);

  auto outputView = cudf::table_view(std::move(outputColumns));
  auto outputOwner = std::make_shared<GroupIdOutputOwner>(
      GroupIdOutputOwner{input_, nullColumns_, std::move(generatedTable)});

  // Advance to next grouping set
  ++groupingSetIndex_;
  if (groupingSetIndex_ == numGroupingSets_) {
    // All grouping sets processed for this input
    input_.reset();
    nullColumns_.reset();
    groupingSetIndex_ = 0;
  }

  return std::make_shared<CudfVector>(
      pool(),
      outputType_,
      numRows,
      std::move(outputView),
      std::move(outputOwner),
      stream);
}

} // namespace facebook::velox::cudf_velox
