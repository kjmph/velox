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

#include "velox/exec/GroupedScalarFilter.h"
#include "velox/exec/OperatorType.h"
#include "velox/vector/DecodedVector.h"

#include <folly/ScopeGuard.h>

namespace facebook::velox::exec {

GroupedScalarFilter::GroupedScalarFilter(
    int32_t operatorId,
    DriverCtx* driverCtx,
    std::shared_ptr<const core::GroupedScalarFilterNode> planNode)
    : Operator(
          driverCtx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "GroupedScalarFilter"),
      planNode_(std::move(planNode)),
      augmentedInputType_(planNode_->augmentedInputType()),
      groupIdChannel_(
          planNode_->sources()[0]->outputType()->getChildIdx(
              planNode_->groupIdName())),
      scalarValueChannel_(
          planNode_->sources()[0]->outputType()->getChildIdx(
              planNode_->scalarValueName())) {}

void GroupedScalarFilter::initialize() {
  Operator::initialize();

  identityProjections_.clear();
  for (column_index_t i = 0; i < outputType_->size(); ++i) {
    identityProjections_.emplace_back(i, i);
  }

  std::vector<core::TypedExprPtr> exprs{planNode_->filter()};
  exprs_ =
      makeExprSetFromFlag(std::move(exprs), operatorCtx_->execCtx(), false);
}

void GroupedScalarFilter::addInput(RowVectorPtr input) {
  bufferedInputs_.push_back(std::move(input));
}

bool GroupedScalarFilter::findScalar() {
  for (const auto& input : bufferedInputs_) {
    if (input->size() == 0) {
      continue;
    }

    LocalSelectivityVector localRows(*operatorCtx_->execCtx(), input->size());
    auto* rows = localRows.get();
    rows->setAll();

    DecodedVector groupIds(*input->childAt(groupIdChannel_), *rows);
    for (vector_size_t i = 0; i < input->size(); ++i) {
      if (!groupIds.isNullAt(i) &&
          groupIds.valueAt<int64_t>(i) == planNode_->scalarGroupId()) {
        scalarVector_ = input->childAt(scalarValueChannel_);
        scalarIndex_ = i;
        return true;
      }
    }
  }
  return false;
}

RowVectorPtr GroupedScalarFilter::makeAugmentedInput(const RowVectorPtr& input) {
  auto children = input->children();
  children.push_back(
      BaseVector::wrapInConstant(input->size(), scalarIndex_, scalarVector_));
  return std::make_shared<RowVector>(
      pool(), augmentedInputType_, nullptr, input->size(), std::move(children));
}

RowVectorPtr GroupedScalarFilter::getOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  if (!scalarScanned_) {
    scalarFound_ = findScalar();
    scalarScanned_ = true;
    addRuntimeStat(
        "groupedScalarFilterScalarFound", RuntimeCounter(scalarFound_ ? 1 : 0));
  }

  if (!scalarFound_) {
    finished_ = true;
    bufferedInputs_.clear();
    return nullptr;
  }

  while (outputInputIndex_ < bufferedInputs_.size()) {
    auto sourceInput = bufferedInputs_[outputInputIndex_++];
    if (sourceInput->size() == 0) {
      continue;
    }

    input_ = makeAugmentedInput(sourceInput);
    SCOPE_EXIT {
      input_.reset();
    };

    LocalSelectivityVector localRows(*operatorCtx_->execCtx(), input_->size());
    auto* rows = localRows.get();
    rows->setAll();
    EvalCtx evalCtx(operatorCtx_->execCtx(), exprs_.get(), input_.get());

    std::vector<VectorPtr> results;
    exprs_->eval(0, 1, true, *rows, evalCtx, results);
    const auto numOut =
        processFilterResults(results[0], *rows, filterEvalCtx_, pool());
    addRuntimeStat("groupedScalarFilterInputRows", RuntimeCounter(input_->size()));
    addRuntimeStat("groupedScalarFilterOutputRows", RuntimeCounter(numOut));

    if (numOut == 0) {
      continue;
    }

    const bool allRowsSelected = numOut == input_->size();
    return fillOutput(
        numOut, allRowsSelected ? nullptr : filterEvalCtx_.selectedIndices, {});
  }

  finished_ = true;
  bufferedInputs_.clear();
  scalarVector_.reset();
  return nullptr;
}

OperatorStats GroupedScalarFilter::stats(bool clear) {
  auto stats = Operator::stats(clear);
  if (operatorCtx()
          ->driverCtx()
          ->queryConfig()
          .operatorTrackExpressionStats() &&
      exprs_ != nullptr) {
    stats.expressionStats = exprs_->stats(true /*excludeSpecialForm*/);
  }
  return stats;
}

} // namespace facebook::velox::exec
