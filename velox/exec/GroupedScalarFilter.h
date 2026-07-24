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

#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::exec {

class GroupedScalarFilter : public Operator {
 public:
  GroupedScalarFilter(
      int32_t operatorId,
      DriverCtx* driverCtx,
      std::shared_ptr<const core::GroupedScalarFilterNode> planNode);

  bool needsInput() const override {
    return !noMoreInput_;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* /*unused*/) override {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

  void initialize() override;

  OperatorStats stats(bool clear) override;

 private:
  bool findScalar();

  RowVectorPtr makeAugmentedInput(const RowVectorPtr& input);

  const std::shared_ptr<const core::GroupedScalarFilterNode> planNode_;
  const RowTypePtr augmentedInputType_;
  const column_index_t groupIdChannel_;
  const column_index_t scalarValueChannel_;

  std::unique_ptr<ExprSet> exprs_;
  FilterEvalCtx filterEvalCtx_;

  std::vector<RowVectorPtr> bufferedInputs_;
  VectorPtr scalarVector_;
  vector_size_t scalarIndex_{0};
  bool scalarFound_{false};
  bool scalarScanned_{false};
  size_t outputInputIndex_{0};
  bool finished_{false};
};

} // namespace facebook::velox::exec
