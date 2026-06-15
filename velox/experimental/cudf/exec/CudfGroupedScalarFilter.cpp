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

#include "velox/experimental/cudf/exec/CudfGroupedScalarFilter.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>

#include <utility>

namespace facebook::velox::cudf_velox {

CudfGroupedScalarFilter::CudfGroupedScalarFilter(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::GroupedScalarFilterNode> planNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          planNode->outputType(),
          planNode->id(),
          "CudfGroupedScalarFilter",
          nvtx3::rgb{199, 21, 133},
          NvtxMethodFlag::kAddInput | NvtxMethodFlag::kGetOutput,
          std::nullopt,
          planNode),
      planNode_(std::move(planNode)),
      augmentedInputType_(planNode_->augmentedInputType()),
      groupIdChannel_(
          planNode_->sources()[0]->outputType()->getChildIdx(
              planNode_->groupIdName())),
      scalarValueChannel_(
          planNode_->sources()[0]->outputType()->getChildIdx(
              planNode_->scalarValueName())) {}

void CudfGroupedScalarFilter::initialize() {
  Operator::initialize();

  std::vector<core::TypedExprPtr> exprs{planNode_->filter()};
  auto exprSet =
      exec::makeExprSetFromFlag(std::move(exprs), operatorCtx_->execCtx(), false);
  filterEvaluator_ =
      createCudfExpression(exprSet->exprs()[0], augmentedInputType_);
}

void CudfGroupedScalarFilter::doAddInput(RowVectorPtr input) {
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  inputs_.push_back(std::move(cudfInput));
}

RowVectorPtr CudfGroupedScalarFilter::doGetOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  auto table = getConcatenatedTable(
      std::exchange(inputs_, {}), outputType_, stream, get_output_mr());
  VELOX_CHECK_NOT_NULL(table);
  const auto inputRows = table->num_rows();
  if (inputRows == 0) {
    finished_ = true;
    return nullptr;
  }

  auto const tableView = table->view();
  cudf::numeric_scalar<int64_t> scalarGroupId(
      planNode_->scalarGroupId(), true, stream, get_temp_mr());
  auto scalarMask = cudf::binary_operation(
      tableView.column(groupIdChannel_),
      scalarGroupId,
      cudf::binary_operator::EQUAL,
      cudf::data_type(cudf::type_id::BOOL8),
      stream,
      get_temp_mr());

  auto scalarRows =
      cudf::apply_boolean_mask(tableView, scalarMask->view(), stream, get_temp_mr());
  if (scalarRows->num_rows() == 0) {
    addRuntimeStat("groupedScalarFilterScalarFound", RuntimeCounter(0));
    finished_ = true;
    return nullptr;
  }
  addRuntimeStat("groupedScalarFilterScalarFound", RuntimeCounter(1));

  auto scalarOneRow = cudf::slice(scalarRows->view(), {0, 1}, stream);
  VELOX_CHECK(scalarOneRow.size() == 1);
  std::vector<cudf::column_view> scalarColumns{
      scalarOneRow[0].column(scalarValueChannel_)};
  auto repeatedScalar = cudf::repeat(
      cudf::table_view(scalarColumns),
      inputRows,
      stream,
      get_temp_mr());

  std::vector<cudf::column_view> augmentedViews;
  augmentedViews.reserve(tableView.num_columns() + 1);
  for (auto i = 0; i < tableView.num_columns(); ++i) {
    augmentedViews.push_back(tableView.column(i));
  }
  augmentedViews.push_back(repeatedScalar->view().column(0));

  auto filterColumn =
      filterEvaluator_->eval(augmentedViews, stream, get_temp_mr(), true);
  auto filteredTable = cudf::apply_boolean_mask(
      tableView, asView(filterColumn), stream, get_output_mr());

  addRuntimeStat("groupedScalarFilterInputRows", RuntimeCounter(inputRows));
  addRuntimeStat(
      "groupedScalarFilterOutputRows",
      RuntimeCounter(filteredTable->num_rows()));

  finished_ = true;
  if (filteredTable->num_rows() == 0) {
    return nullptr;
  }

  return std::make_shared<CudfVector>(
      pool(),
      outputType_,
      filteredTable->num_rows(),
      std::move(filteredTable),
      stream);
}

} // namespace facebook::velox::cudf_velox
