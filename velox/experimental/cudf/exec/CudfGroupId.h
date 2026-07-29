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

#include "velox/exec/Operator.h"

#include <cudf/types.hpp>

namespace cudf {
class table;
}

namespace facebook::velox::cudf_velox {

class CudfVector;

/// GPU implementation of the GroupId operator for GROUPING SETS, CUBE, and
/// ROLLUP operations. Takes a single input batch and produces N output batches
/// (one per grouping set).
class CudfGroupId : public CudfOperatorBase {
 public:
  CudfGroupId(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::GroupIdNode>& groupIdNode);

  bool needsInput() const override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return noMoreInput_ && !input_;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;

 private:
  static constexpr column_index_t kMissingGroupingKey =
      std::numeric_limits<column_index_t>::max();

  static constexpr column_index_t kNoNullColumn =
      std::numeric_limits<column_index_t>::max();

  /// A grouping set contains a subset of all the grouping keys. This list
  /// contains one entry per grouping set and identifies the grouping keys that
  /// are part of the set as indices of the input columns. The position in the
  /// list identifies the grouping key column in the output. Positions with
  /// kMissingGroupingKey correspond to grouping keys which are not included in
  /// the set.
  std::vector<std::vector<column_index_t>> groupingKeyMappings_;

  /// A list of input column indices corresponding to aggregation inputs.
  std::vector<column_index_t> aggregationInputs_;

  /// Precomputed cudf data types for each grouping key output column, used to
  /// create all-null columns for keys not in a grouping set.
  std::vector<cudf::data_type> groupingKeyCudfTypes_;

  /// Maps each grouping key to its cached all-null column. Keys present in
  /// every grouping set use kNoNullColumn.
  std::vector<column_index_t> nullColumnMappings_;

  /// Input and synthesized null columns retained while cycling through
  /// grouping sets. Outputs share ownership of these buffers.
  std::shared_ptr<CudfVector> input_;
  std::shared_ptr<cudf::table> nullColumns_;

  /// Stream associated with the input data.
  cuda::stream_ref inputStream_{cudaStream_t{cudaStreamDefault}};

  /// Index of the grouping set to output in the next getOutput call.
  size_t groupingSetIndex_{0};

  /// Total number of grouping sets.
  size_t numGroupingSets_{0};

  /// Number of grouping key columns in output (first N output columns).
  size_t numGroupingKeys_{0};
};

} // namespace facebook::velox::cudf_velox
