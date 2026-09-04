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
#include "velox/experimental/cudf/exec/CudfBatchConcat.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <algorithm>
#include <array>
#include <functional>
#include <optional>
#include <utility>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox;

class CudfBatchConcatTest : public OperatorTestBase {
 protected:
  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
    common::testutil::TestValue::enable();
  }

  void SetUp() override {
    OperatorTestBase::SetUp();
    CudfConfig::getInstance().debugEnabled = true;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    CudfConfig::getInstance().concatOptimizationEnabled = false;
    CudfConfig::getInstance().finalAggregationBatchSizeMinThreshold =
        std::nullopt;
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  void updateCudfConfig(
      int32_t min,
      std::optional<int32_t> max,
      std::optional<int32_t> finalAggregationMin = std::nullopt) {
    auto& config = CudfConfig::getInstance();
    config.batchSizeMinThreshold = min;
    config.batchSizeMaxThreshold = max;
    config.finalAggregationBatchSizeMinThreshold = finalAggregationMin;
  }

  template <typename T>
  FlatVectorPtr<T> makeFlatSequence(T start, vector_size_t size) {
    return makeFlatVector<T>(size, [start](auto row) { return start + row; });
  }

  // Builds fragmented input via localPartitionRoundRobin to prevent Values
  // from coalescing small batches.
  core::PlanNodePtr createFragmentedSource(
      const std::vector<RowVectorPtr>& vectors,
      std::shared_ptr<core::PlanNodeIdGenerator> generator) {
    std::vector<core::PlanNodePtr> sources;
    for (const auto& vec : vectors) {
      sources.push_back(PlanBuilder(generator).values({vec}).planNode());
    }
    return PlanBuilder(generator).localPartitionRoundRobin(sources).planNode();
  }

  // Returns CudfBatchConcat's input and output batch counts for the given plan
  // node, or nullopt if CudfBatchConcat wasn't inserted for that node.
  std::optional<std::pair<vector_size_t, vector_size_t>> getConcatVectorStats(
      const std::shared_ptr<Task>& task,
      const core::PlanNodeId& aggNodeId) {
    auto planStats = toPlanStats(task->taskStats());
    auto nodeIt = planStats.find(aggNodeId);
    if (nodeIt == planStats.end()) {
      return std::nullopt;
    }
    auto opIt = nodeIt->second.operatorStats.find("CudfBatchConcat");
    if (opIt == nodeIt->second.operatorStats.end()) {
      return std::nullopt;
    }
    return std::pair{opIt->second->inputVectors, opIt->second->outputVectors};
  }
};

DEBUG_ONLY_TEST_F(CudfBatchConcatTest, releasesInputOwnersByOutputBatch) {
  constexpr vector_size_t kRowsPerInput = 10;
  constexpr vector_size_t kMaxRows = 10;
  updateCudfConfig(/*min=*/100, /*max=*/kMaxRows);

  auto stream = cudfGlobalStreamPool().get_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  auto type = ROW({BIGINT()});
  std::vector<CudfVectorPtr> inputs;
  std::vector<std::weak_ptr<CudfVector>> inputOwners;
  for (int32_t batch = 0; batch < 3; ++batch) {
    auto input = makeRowVector(
        {makeFlatSequence<int64_t>(batch * kRowsPerInput, kRowsPerInput)});
    auto table = with_arrow::toCudfTable(input, pool(), stream, mr);
    auto cudfInput = std::make_shared<CudfVector>(
        pool(), type, input->size(), std::move(table), stream);
    inputOwners.push_back(cudfInput);
    inputs.push_back(std::move(cudfInput));
  }

  std::vector<size_t> retainedInputs;
  std::vector<size_t> expiredInputs;
  SCOPED_TESTVALUE_SET(
      "facebook::velox::cudf_velox::getConcatenatedTableBatched::retainedInputBatchesAfterBatchRelease",
      std::function<void(size_t*)>([&](size_t* retained) {
        retainedInputs.push_back(*retained);
        expiredInputs.push_back(
            std::count_if(
                inputOwners.begin(), inputOwners.end(), [](const auto& owner) {
                  return owner.expired();
                }));
      }));

  auto outputs =
      getConcatenatedTableBatched(std::move(inputs), type, stream, mr);
  EXPECT_EQ(outputs.size(), 3);
  EXPECT_EQ(retainedInputs, std::vector<size_t>({2, 1, 0}));
  EXPECT_EQ(expiredInputs, std::vector<size_t>({1, 2, 3}));
}

DEBUG_ONLY_TEST_F(CudfBatchConcatTest, retainsOversizedOwnerUntilFinalSlice) {
  constexpr vector_size_t kInputRows = 10;
  constexpr vector_size_t kMaxRows = 4;
  updateCudfConfig(/*min=*/100, /*max=*/kMaxRows);

  auto input = makeRowVector({makeFlatSequence<int64_t>(0, kInputRows)});
  auto stream = cudfGlobalStreamPool().get_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  auto table = with_arrow::toCudfTable(input, pool(), stream, mr);
  auto cudfInput = std::make_shared<CudfVector>(
      pool(), input->type(), input->size(), std::move(table), stream);
  std::weak_ptr<CudfVector> inputOwner = cudfInput;
  std::vector<CudfVectorPtr> inputs;
  inputs.push_back(std::move(cudfInput));

  std::vector<size_t> retainedInputs;
  std::vector<bool> ownerExpired;
  SCOPED_TESTVALUE_SET(
      "facebook::velox::cudf_velox::getConcatenatedTableBatched::retainedInputBatchesAfterBatchRelease",
      std::function<void(size_t*)>([&](size_t* retained) {
        retainedInputs.push_back(*retained);
        ownerExpired.push_back(inputOwner.expired());
      }));

  auto outputs =
      getConcatenatedTableBatched(std::move(inputs), input->type(), stream, mr);
  ASSERT_EQ(outputs.size(), 3);
  EXPECT_EQ(outputs[0]->num_rows(), 4);
  EXPECT_EQ(outputs[1]->num_rows(), 4);
  EXPECT_EQ(outputs[2]->num_rows(), 2);
  EXPECT_EQ(retainedInputs, std::vector<size_t>({1, 1, 0}));
  EXPECT_EQ(ownerExpired, std::vector<bool>({false, false, true}));
}

TEST_F(CudfBatchConcatTest, rejectsZeroRowCpuInput) {
  auto input = makeRowVector({makeFlatVector<int32_t>(std::vector<int32_t>{})});
  auto planNode = PlanBuilder().values({input}).project({"c0"}).planNode();
  core::PlanFragment planFragment;
  planFragment.planNode = planNode;
  auto task = Task::create(
      "CudfBatchConcatTest",
      std::move(planFragment),
      0,
      core::QueryCtx::create(driverExecutor_.get()),
      Task::ExecutionMode::kParallel);
  DriverCtx driverCtx(task, 0, 0, 0, 0);
  CudfBatchConcat concat(0, &driverCtx, planNode, 10);

  VELOX_ASSERT_THROW(
      concat.addInput(input), "CudfBatchConcat expects CudfVector input");
}

TEST_F(CudfBatchConcatTest, splitsSingleOversizedInputAtMaxThreshold) {
  constexpr vector_size_t kInputRows = 10;
  constexpr vector_size_t kMaxRows = 4;
  const std::array<vector_size_t, 3> expectedOutputRows{4, 4, 2};

  updateCudfConfig(/*min=*/kMaxRows, /*max=*/kMaxRows);

  auto input = makeRowVector(
      {makeFlatSequence<int64_t>(0, kInputRows),
       makeFlatVector<std::string>({
           "zero",
           "one",
           "two",
           "three",
           "four",
           "five",
           "six",
           "seven",
           "eight",
           "nine",
       })});
  auto planNode = PlanBuilder().values({input}).project({"c0"}).planNode();
  core::PlanFragment planFragment;
  planFragment.planNode = planNode;
  auto task = Task::create(
      "CudfBatchConcatTest.singleOversizedInput",
      std::move(planFragment),
      0,
      core::QueryCtx::create(driverExecutor_.get()),
      Task::ExecutionMode::kParallel);
  DriverCtx driverCtx(task, 0, 0, 0, 0);
  CudfBatchConcat concat(0, &driverCtx, planNode, kMaxRows);

  auto stream = cudfGlobalStreamPool().get_stream();
  auto inputTable = with_arrow::toCudfTable(
      input, pool(), stream, cudf::get_current_device_resource_ref());
  auto cudfInput = std::make_shared<CudfVector>(
      pool(), input->type(), input->size(), std::move(inputTable), stream);
  const auto inputColumn = cudfInput->getTableView().column(0);
  const auto* inputHead = inputColumn.head<int64_t>();
  const auto inputOffset = inputColumn.offset();
  const auto inputFlatSize = cudfInput->estimateFlatSize();

  concat.addInput(cudfInput);
  cudfInput.reset();
  concat.noMoreInput();

  vector_size_t outputOffset = 0;
  uint64_t outputFlatSize = 0;
  for (const auto expectedRows : expectedOutputRows) {
    auto output = std::dynamic_pointer_cast<CudfVector>(concat.getOutput());
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->size(), expectedRows);
    EXPECT_LE(output->size(), kMaxRows);
    EXPECT_EQ(output->stream().value(), stream.value());

    const auto outputColumn = output->getTableView().column(0);
    EXPECT_EQ(outputColumn.head<int64_t>(), inputHead);
    EXPECT_EQ(outputColumn.offset(), inputOffset + outputOffset);
    outputFlatSize += output->estimateFlatSize();
    outputOffset += expectedRows;
  }

  EXPECT_EQ(outputOffset, kInputRows);
  EXPECT_EQ(outputFlatSize, inputFlatSize)
      << "Slice estimates must apportion variable-width storage instead of "
         "counting the oversized owner's full backing data for every view";
  EXPECT_TRUE(concat.isFinished());
  EXPECT_EQ(concat.getOutput(), nullptr);
}

TEST_F(CudfBatchConcatTest, preservesOrderAroundOversizedInput) {
  constexpr vector_size_t kMaxRows = 4;
  const std::array<vector_size_t, 5> expectedOutputRows{2, 4, 4, 2, 2};

  updateCudfConfig(/*min=*/100, /*max=*/kMaxRows);

  auto prefix = makeRowVector({makeFlatSequence<int64_t>(0, 2)});
  auto oversized = makeRowVector({makeFlatSequence<int64_t>(2, 10)});
  auto suffix = makeRowVector({makeFlatSequence<int64_t>(12, 2)});
  auto planNode = PlanBuilder().values({prefix}).project({"c0"}).planNode();
  core::PlanFragment planFragment;
  planFragment.planNode = planNode;
  auto task = Task::create(
      "CudfBatchConcatTest.oversizedInputWithNeighbors",
      std::move(planFragment),
      0,
      core::QueryCtx::create(driverExecutor_.get()),
      Task::ExecutionMode::kParallel);
  DriverCtx driverCtx(task, 0, 0, 0, 0);
  CudfBatchConcat concat(0, &driverCtx, planNode, 100);

  auto stream = cudfGlobalStreamPool().get_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  const auto toCudfVector = [&](const RowVectorPtr& input) {
    auto table = with_arrow::toCudfTable(input, pool(), stream, mr);
    return std::make_shared<CudfVector>(
        pool(), input->type(), input->size(), std::move(table), stream);
  };

  auto prefixCudf = toCudfVector(prefix);
  auto oversizedCudf = toCudfVector(oversized);
  auto suffixCudf = toCudfVector(suffix);
  const auto oversizedColumn = oversizedCudf->getTableView().column(0);
  const auto* oversizedHead = oversizedColumn.head<int64_t>();
  const auto oversizedOffset = oversizedColumn.offset();

  concat.addInput(std::move(prefixCudf));
  concat.addInput(std::move(oversizedCudf));
  concat.addInput(std::move(suffixCudf));
  concat.noMoreInput();

  std::vector<int64_t> actualValues;
  for (size_t outputIndex = 0; outputIndex < expectedOutputRows.size();
       ++outputIndex) {
    auto output = std::dynamic_pointer_cast<CudfVector>(concat.getOutput());
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->size(), expectedOutputRows[outputIndex]);
    EXPECT_LE(output->size(), kMaxRows);

    if (outputIndex >= 1 && outputIndex <= 3) {
      const auto outputColumn = output->getTableView().column(0);
      EXPECT_EQ(outputColumn.head<int64_t>(), oversizedHead);
      EXPECT_EQ(
          outputColumn.offset(),
          oversizedOffset + static_cast<vector_size_t>((outputIndex - 1) * 4));
    }

    auto hostOutput = with_arrow::toVeloxColumn(
        output->getTableView(), pool(), output->type(), output->stream(), mr);
    output->stream().synchronize();
    auto values = hostOutput->childAt(0)->asFlatVector<int64_t>();
    for (vector_size_t row = 0; row < values->size(); ++row) {
      actualValues.push_back(values->valueAt(row));
    }
  }

  ASSERT_EQ(actualValues.size(), 14);
  for (size_t row = 0; row < actualValues.size(); ++row) {
    EXPECT_EQ(actualValues[row], static_cast<int64_t>(row));
  }
  EXPECT_TRUE(concat.isFinished());
  EXPECT_EQ(concat.getOutput(), nullptr);
}

// Verifies that CudfBatchConcat is inserted before aggregation and reduces
// the number of batches reaching the aggregation operator.
TEST_F(CudfBatchConcatTest, concatReducesBatchesBeforeAggregation) {
  // 6 batches of 10 rows each = 60 rows total.
  // With min threshold 30, concat should accumulate ~3 batches before flushing,
  // producing fewer output batches than the 6 it received.
  updateCudfConfig(/*min=*/30, /*max=*/std::nullopt);
  CudfConfig::getInstance().concatOptimizationEnabled = true;

  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 6; ++i) {
    vectors.push_back(makeRowVector({makeFlatSequence<int64_t>(i * 10, 10)}));
  }
  createDuckDbTable(vectors);

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId aggNodeId;

  auto plan = PlanBuilder(generator)
                  .addNode([&](auto id, auto pool) {
                    return createFragmentedSource(vectors, generator);
                  })
                  .singleAggregation({}, {"sum(c0)"})
                  .capturePlanNodeId(aggNodeId)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .maxDrivers(1)
                  .assertResults("SELECT sum(c0) FROM tmp");

  auto planStats = toPlanStats(task->taskStats());
  auto& nodeStats = planStats.at(aggNodeId);
  auto concatIt = nodeStats.operatorStats.find("CudfBatchConcat");
  ASSERT_NE(concatIt, nodeStats.operatorStats.end())
      << "CudfBatchConcat should be present in operator stats";

  auto& concatStats = *concatIt->second;
  EXPECT_EQ(concatStats.inputVectors, 6)
      << "CudfBatchConcat should have received all 6 input batches";
  EXPECT_LT(concatStats.outputVectors, concatStats.inputVectors)
      << "CudfBatchConcat should produce fewer output batches than input";
}

// Verifies that CudfBatchConcat is not inserted when the optimization is
// disabled, even when aggregation is present.
TEST_F(CudfBatchConcatTest, concatNotInsertedWhenDisabled) {
  updateCudfConfig(/*min=*/30, /*max=*/std::nullopt);
  CudfConfig::getInstance().concatOptimizationEnabled = false;

  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 6; ++i) {
    vectors.push_back(makeRowVector({makeFlatSequence<int64_t>(i * 10, 10)}));
  }
  createDuckDbTable(vectors);

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId aggNodeId;

  auto plan = PlanBuilder(generator)
                  .addNode([&](auto id, auto pool) {
                    return createFragmentedSource(vectors, generator);
                  })
                  .singleAggregation({}, {"sum(c0)"})
                  .capturePlanNodeId(aggNodeId)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .maxDrivers(1)
                  .assertResults("SELECT sum(c0) FROM tmp");

  auto planStats = toPlanStats(task->taskStats());
  auto& nodeStats = planStats.at(aggNodeId);
  EXPECT_EQ(nodeStats.operatorStats.count("CudfBatchConcat"), 0)
      << "CudfBatchConcat should not be present when optimization is disabled";
}

// When the threshold exceeds total input rows, concat accumulates all batches
// and flushes them as a single merged batch on noMoreInput.
TEST_F(CudfBatchConcatTest, concatMergesAllOnFlushWithHighThreshold) {
  updateCudfConfig(/*min=*/100000, /*max=*/std::nullopt);
  CudfConfig::getInstance().concatOptimizationEnabled = true;

  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 6; ++i) {
    vectors.push_back(makeRowVector({makeFlatSequence<int64_t>(i * 10, 10)}));
  }
  createDuckDbTable(vectors);

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId aggNodeId;

  auto plan = PlanBuilder(generator)
                  .addNode([&](auto id, auto pool) {
                    return createFragmentedSource(vectors, generator);
                  })
                  .singleAggregation({}, {"sum(c0)"})
                  .capturePlanNodeId(aggNodeId)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .maxDrivers(1)
                  .assertResults("SELECT sum(c0) FROM tmp");

  auto planStats = toPlanStats(task->taskStats());
  auto& nodeStats = planStats.at(aggNodeId);
  auto concatIt = nodeStats.operatorStats.find("CudfBatchConcat");
  ASSERT_NE(concatIt, nodeStats.operatorStats.end())
      << "CudfBatchConcat should still be inserted even with high threshold";

  auto& concatStats = *concatIt->second;
  EXPECT_EQ(concatStats.inputVectors, 6);
  EXPECT_EQ(concatStats.outputVectors, 1)
      << "All batches should be merged into one on noMoreInput flush";
}

// Verifies correctness with grouped aggregation (non-global) and concat.
TEST_F(CudfBatchConcatTest, concatWithGroupedAggregation) {
  updateCudfConfig(/*min=*/30, /*max=*/std::nullopt);
  CudfConfig::getInstance().concatOptimizationEnabled = true;

  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 6; ++i) {
    vectors.push_back(makeRowVector(
        {makeFlatVector<int64_t>(10, [](auto row) { return row % 3; }),
         makeFlatSequence<int64_t>(i * 10, 10)}));
  }
  createDuckDbTable(vectors);

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId aggNodeId;

  auto plan = PlanBuilder(generator)
                  .addNode([&](auto id, auto pool) {
                    return createFragmentedSource(vectors, generator);
                  })
                  .singleAggregation({"c0"}, {"sum(c1)"})
                  .capturePlanNodeId(aggNodeId)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .maxDrivers(1)
                  .assertResults("SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  auto planStats = toPlanStats(task->taskStats());
  auto& nodeStats = planStats.at(aggNodeId);
  auto concatIt = nodeStats.operatorStats.find("CudfBatchConcat");
  ASSERT_NE(concatIt, nodeStats.operatorStats.end());
  EXPECT_EQ(concatIt->second->inputVectors, 6);
  EXPECT_LT(concatIt->second->outputVectors, 6);
}

TEST_F(CudfBatchConcatTest, finalAggregationUsesDedicatedConcatThreshold) {
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 6; ++i) {
    vectors.push_back(makeRowVector(
        {makeFlatVector<int64_t>(10, [](auto row) { return row % 3; }),
         makeFlatSequence<int64_t>(i * 10, 10)}));
  }
  createDuckDbTable(vectors);

  auto runWithFinalAggregationThreshold =
      [&](std::optional<int32_t> finalAggregationMin) {
        updateCudfConfig(/*min=*/5, /*max=*/std::nullopt, finalAggregationMin);
        CudfConfig::getInstance().concatOptimizationEnabled = true;

        auto generator = std::make_shared<core::PlanNodeIdGenerator>();
        std::vector<core::PlanNodePtr> partialSources;
        partialSources.reserve(vectors.size());
        for (const auto& vector : vectors) {
          partialSources.push_back(PlanBuilder(generator)
                                       .values({vector})
                                       .partialAggregation({"c0"}, {"sum(c1)"})
                                       .planNode());
        }

        core::PlanNodeId finalAggNodeId;
        auto plan = PlanBuilder(generator)
                        .localPartition({"c0"}, partialSources)
                        .finalAggregation()
                        .capturePlanNodeId(finalAggNodeId)
                        .planNode();

        auto task =
            AssertQueryBuilder(duckDbQueryRunner_)
                .plan(plan)
                .maxDrivers(1)
                .config(core::QueryConfig::kMaxLocalExchangePartitionCount, "1")
                .assertResults("SELECT c0, sum(c1) FROM tmp GROUP BY c0");

        const auto concatStats = getConcatVectorStats(task, finalAggNodeId);
        VELOX_CHECK(concatStats.has_value());
        return concatStats.value();
      };

  const auto genericThresholdStats =
      runWithFinalAggregationThreshold(std::nullopt);
  const auto finalThresholdStats =
      runWithFinalAggregationThreshold(/*finalAggregationMin=*/100000);

  EXPECT_GT(genericThresholdStats.first, 1);
  EXPECT_GT(genericThresholdStats.second, finalThresholdStats.second);
  EXPECT_EQ(finalThresholdStats.first, genericThresholdStats.first);
  EXPECT_EQ(finalThresholdStats.second, 1)
      << "A high final aggregation threshold should merge compact partial "
         "aggregation inputs into a single final aggregation batch.";
}

TEST_F(CudfBatchConcatTest, concatPreservesZeroColumnRowCountForCountStar) {
  updateCudfConfig(/*min=*/30, /*max=*/std::nullopt);
  CudfConfig::getInstance().concatOptimizationEnabled = true;

  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
  });
  createDuckDbTable({data});

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId aggNodeId;

  auto plan = PlanBuilder(generator)
                  .values({data})
                  .filter("c0 > 0")
                  .project({})
                  .singleAggregation({}, {"count(*)"})
                  .capturePlanNodeId(aggNodeId)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .maxDrivers(1)
                  .assertResults("SELECT count(*) FROM tmp WHERE c0 > 0");

  auto planStats = toPlanStats(task->taskStats());
  auto& nodeStats = planStats.at(aggNodeId);
  auto concatIt = nodeStats.operatorStats.find("CudfBatchConcat");
  ASSERT_NE(concatIt, nodeStats.operatorStats.end());
  EXPECT_EQ(concatIt->second->inputVectors, 1);
  EXPECT_EQ(concatIt->second->outputVectors, 1);
}
// Verifies that CudfBatchConcat is inserted before the hash join probe and
// correctly handles the 2-source HashJoinNode plan node.
TEST_F(CudfBatchConcatTest, concatBeforeHashJoinProbe) {
  updateCudfConfig(/*min=*/30, /*max=*/std::nullopt);
  CudfConfig::getInstance().concatOptimizationEnabled = true;

  // Probe side: 6 batches of 10 rows each.
  std::vector<RowVectorPtr> probeVectors;
  for (int i = 0; i < 6; ++i) {
    probeVectors.push_back(makeRowVector(
        {"c0", "c1"},
        {makeFlatVector<int64_t>(10, [i](auto row) { return row % 3; }),
         makeFlatSequence<int64_t>(i * 10, 10)}));
  }

  // Build side: small dimension table.
  auto buildVector =
      makeRowVector({"u_c0"}, {makeFlatVector<int64_t>({0, 1, 2})});

  createDuckDbTable("probe", probeVectors);
  createDuckDbTable("build", {buildVector});

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId joinNodeId;

  auto plan = PlanBuilder(generator)
                  .addNode([&](auto id, auto pool) {
                    return createFragmentedSource(probeVectors, generator);
                  })
                  .hashJoin(
                      {"c0"},
                      {"u_c0"},
                      PlanBuilder(generator).values({buildVector}).planNode(),
                      "",
                      {"c0", "c1"},
                      core::JoinType::kInner)
                  .capturePlanNodeId(joinNodeId)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .plan(plan)
          .maxDrivers(1)
          .assertResults(
              "SELECT p.c0, p.c1 FROM probe p INNER JOIN build b ON p.c0 = b.u_c0");

  auto planStats = toPlanStats(task->taskStats());
  auto& nodeStats = planStats.at(joinNodeId);
  auto concatIt = nodeStats.operatorStats.find("CudfBatchConcat");
  ASSERT_NE(concatIt, nodeStats.operatorStats.end())
      << "CudfBatchConcat should be present before hash join probe";

  auto& concatStats = *concatIt->second;
  EXPECT_EQ(concatStats.inputVectors, 6)
      << "CudfBatchConcat should have received all 6 probe batches";
  EXPECT_LT(concatStats.outputVectors, concatStats.inputVectors)
      << "CudfBatchConcat should produce fewer output batches than input";
}

TEST_F(CudfBatchConcatTest, concatSplitsZeroColumnBatchesAtMaxThreshold) {
  updateCudfConfig(/*min=*/30, /*max=*/20);
  CudfConfig::getInstance().concatOptimizationEnabled = true;

  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 3; ++i) {
    vectors.push_back(makeRowVector({makeFlatSequence<int64_t>(i * 10, 10)}));
  }
  createDuckDbTable(vectors);

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId aggNodeId;

  auto plan = PlanBuilder(generator)
                  .addNode([&](auto id, auto pool) {
                    return createFragmentedSource(vectors, generator);
                  })
                  .filter("c0 >= 0")
                  .project({})
                  .singleAggregation({}, {"count(*)"})
                  .capturePlanNodeId(aggNodeId)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .maxDrivers(1)
                  .assertResults("SELECT count(*) FROM tmp WHERE c0 >= 0");

  auto planStats = toPlanStats(task->taskStats());
  auto& nodeStats = planStats.at(aggNodeId);
  auto concatIt = nodeStats.operatorStats.find("CudfBatchConcat");
  ASSERT_NE(concatIt, nodeStats.operatorStats.end());
  EXPECT_EQ(concatIt->second->inputVectors, 3);
  EXPECT_EQ(concatIt->second->outputVectors, 2)
      << "30 zero-column rows should be split into 20-row and 10-row batches";
}
