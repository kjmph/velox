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

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

namespace facebook::velox::cudf_velox {

class CudfGroupedScalarFilter : public CudfOperatorBase {
 public:
  CudfGroupedScalarFilter(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::GroupedScalarFilterNode> planNode);

  void initialize() override;

  bool needsInput() const override {
    return !noMoreInput_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;

  RowVectorPtr doGetOutput() override;

  void doClose() override {
    Operator::close();
    filterEvaluator_.reset();
    inputs_.clear();
  }

 private:
  const std::shared_ptr<const core::GroupedScalarFilterNode> planNode_;
  const RowTypePtr augmentedInputType_;
  const column_index_t groupIdChannel_;
  const column_index_t scalarValueChannel_;

  CudfExpressionPtr filterEvaluator_;
  std::vector<CudfVectorPtr> inputs_;
  bool finished_{false};
};

} // namespace facebook::velox::cudf_velox
