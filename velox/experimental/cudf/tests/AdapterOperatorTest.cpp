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
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/exec/CudfBatchConcat.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"
#include "velox/experimental/cudf/tests/utils/ExpressionTestUtil.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/FilterProject.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

#include <algorithm>
#include <string_view>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace {

constexpr std::string_view kPreCudfAdapterName = "pre-cuDF-test";

bool isIdentityProject(const core::ProjectNode& project) {
  const auto& sourceType = project.sources()[0]->outputType();
  if (project.projections().size() != sourceType->size() ||
      project.names().size() != sourceType->size()) {
    return false;
  }

  for (auto i = 0; i < sourceType->size(); ++i) {
    const auto field =
        std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
            project.projections()[i]);
    if (!field || !field->isInputColumn() ||
        field->name() != sourceType->nameOf(i) ||
        project.names()[i] != sourceType->nameOf(i)) {
      return false;
    }
  }
  return true;
}

bool replaceIdentityProjectWithCudfBatchConcat(
    const DriverFactory& factory,
    Driver& driver) {
  const auto operators = driver.operators();
  for (auto index = 0; index < operators.size(); ++index) {
    const auto* filterProject =
        dynamic_cast<const FilterProject*>(operators[index]);
    if (!filterProject) {
      continue;
    }

    const auto planNodeIt = std::find_if(
        factory.planNodes.begin(),
        factory.planNodes.end(),
        [&](const auto& node) {
          return node->id() == filterProject->planNodeId();
        });
    if (planNodeIt == factory.planNodes.end()) {
      continue;
    }

    const auto project =
        std::dynamic_pointer_cast<const core::ProjectNode>(*planNodeIt);
    if (!project || !isIdentityProject(*project)) {
      continue;
    }

    std::vector<std::unique_ptr<Operator>> replacement;
    replacement.push_back(
        std::make_unique<cudf_velox::CudfBatchConcat>(
            filterProject->operatorId(), driver.driverCtx(), project));
    factory.replaceOperators(driver, index, index + 1, std::move(replacement));
  }

  // Continue to the cuDF adapter so it can add conversion boundaries.
  return false;
}

void registerPreCudfAdapter() {
  DriverFactory::registerAdapter(
      {std::string{kPreCudfAdapterName},
       {},
       replaceIdentityProjectWithCudfBatchConcat});
}

void unregisterPreCudfAdapter() {
  auto& adapters = DriverFactory::adapters;
  adapters.erase(
      std::remove_if(
          adapters.begin(),
          adapters.end(),
          [](const DriverAdapter& adapter) {
            return adapter.label == kPreCudfAdapterName;
          }),
      adapters.end());
}

} // namespace

class AdapterOperatorTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    savedCpuFallback_ = cudf_velox::CudfConfig::getInstance().allowCpuFallback;
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    registerPreCudfAdapter();
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = savedCpuFallback_;
    unregisterPreCudfAdapter();
    OperatorTestBase::TearDown();
  }

  // Gives the test adapter priority while preserving built-ins for other
  // operators.
  void registerAdapterFirst(
      std::unique_ptr<cudf_velox::OperatorAdapter> adapter) {
    cudf_velox::OperatorAdapterRegistry::getInstance().registerAdapterFront(
        std::move(adapter));
  }

  // Re-registers the driver adapter to capture the updated CPU fallback
  // setting.
  void enableCpuFallback() {
    cudf_velox::unregisterCudf();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = true;
    cudf_velox::registerCudf();
  }

  bool savedCpuFallback_{true};
};

namespace {
// Returns no replacements to exercise both keep and replace contracts.
class EmptyReplacementAdapter : public cudf_velox::OperatorAdapter {
 public:
  EmptyReplacementAdapter(bool keepOperator, bool producesGpuOutput = false)
      : cudf_velox::OperatorAdapter("EmptyReplacement"),
        keepOperator_{keepOperator},
        producesGpuOutput_{producesGpuOutput} {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::FilterProject*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/) const override {
    return true;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return producesGpuOutput_;
  }

  bool keepOperator() const override {
    return keepOperator_;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {};
  }

 private:
  const bool keepOperator_;
  const bool producesGpuOutput_;
};

// Keeps FilterProject and appends a GPU round trip. The standard conversion
// suffixes preserve plan-node stat attribution.
class AppendingAdapter : public cudf_velox::OperatorAdapter {
 public:
  AppendingAdapter() : cudf_velox::OperatorAdapter("Appending") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::FilterProject*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/) const override {
    return true;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  bool keepOperator() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    std::vector<std::unique_ptr<exec::Operator>> appended;
    appended.push_back(
        std::make_unique<cudf_velox::CudfFromVelox>(
            operatorId,
            planNode->outputType(),
            ctx,
            planNode->id() + "-from-velox"));
    appended.push_back(
        std::make_unique<cudf_velox::CudfToVelox>(
            operatorId,
            planNode->outputType(),
            ctx,
            planNode->id() + "-to-velox"));
    return appended;
  }
};
// Declines FilterProject; createReplacements() must not be called.
class DecliningAdapter : public cudf_velox::OperatorAdapter {
 public:
  DecliningAdapter() : cudf_velox::OperatorAdapter("Declining") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::FilterProject*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/) const override {
    return false;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  bool keepOperator() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    VELOX_FAIL(
        "createReplacements() must not be called for a declined operator");
  }
};
} // namespace

TEST_F(AdapterOperatorTest, adapterStatsMergedIntoPlanNode) {
  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});

  core::PlanNodeId projNodeId;
  auto plan = PlanBuilder()
                  .values({data})
                  .project({"c0 * 2 as x"})
                  .capturePlanNodeId(projNodeId)
                  .planNode();

  std::shared_ptr<exec::Task> task;
  AssertQueryBuilder(plan).copyResults(pool(), task);

  auto stats = toPlanStats(task->taskStats());
  auto& projStats = stats.at(projNodeId);

  EXPECT_TRUE(projStats.isMultiOperatorTypeNode());
  EXPECT_TRUE(projStats.operatorStats.count("CudfToVelox"));
}

// An empty replacement is an adapter error, not CPU fallback.
TEST_F(AdapterOperatorTest, emptyReplacementIsRejectedWithoutFallback) {
  registerAdapterFirst(
      std::make_unique<EmptyReplacementAdapter>(/*keepOperator=*/false));

  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  auto plan = PlanBuilder().values({data}).project({"c0 * 2 as x"}).planNode();

  std::shared_ptr<exec::Task> task;
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool(), task),
      "Adapter replaced an operator with nothing");
}

// Empty additions are valid when the original operator is kept.
TEST_F(AdapterOperatorTest, keptOperatorNeedsNoAppendedOperators) {
  registerAdapterFirst(
      std::make_unique<EmptyReplacementAdapter>(/*keepOperator=*/true));

  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  auto plan = PlanBuilder().values({data}).project({"c0 * 2 as x"}).planNode();

  std::shared_ptr<exec::Task> task;
  auto results = AssertQueryBuilder(plan).copyResults(pool(), task);
  EXPECT_EQ(results->size(), 5);
}

// Reject before CudfToVelox can make an empty replacement appear non-empty.
TEST_F(
    AdapterOperatorTest,
    emptyReplacementIsRejectedDespiteConversionOperator) {
  registerAdapterFirst(
      std::make_unique<EmptyReplacementAdapter>(
          /*keepOperator=*/false, /*producesGpuOutput=*/true));

  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  auto plan = PlanBuilder().values({data}).project({"c0 * 2 as x"}).planNode();

  std::shared_ptr<exec::Task> task;
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool(), task),
      "Adapter replaced an operator with nothing");
}

// Adapter errors are rejected even when CPU fallback is enabled.
TEST_F(AdapterOperatorTest, emptyReplacementIsRejectedWithCpuFallbackEnabled) {
  enableCpuFallback();
  registerAdapterFirst(
      std::make_unique<EmptyReplacementAdapter>(
          /*keepOperator=*/false, /*producesGpuOutput=*/true));

  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  auto plan = PlanBuilder().values({data}).project({"c0 * 2 as x"}).planNode();

  std::shared_ptr<exec::Task> task;
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool(), task),
      "Adapter replaced an operator with nothing");
}

// A declined GPU path requires CPU fallback.
TEST_F(AdapterOperatorTest, declinedOperatorIsRejectedWithoutFallback) {
  registerAdapterFirst(std::make_unique<DecliningAdapter>());

  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  auto plan = PlanBuilder().values({data}).project({"c0 * 2 as x"}).planNode();

  std::shared_ptr<exec::Task> task;
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool(), task),
      "Replacement with cuDF operator failed");
}

// Verify that fallback keeps FilterProject on CPU.
TEST_F(AdapterOperatorTest, declinedOperatorRunsOnCpuWithFallback) {
  enableCpuFallback();
  registerAdapterFirst(std::make_unique<DecliningAdapter>());

  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  core::PlanNodeId projNodeId;
  auto plan = PlanBuilder()
                  .values({data})
                  .project({"c0 * 2 as x"})
                  .capturePlanNodeId(projNodeId)
                  .planNode();

  std::shared_ptr<exec::Task> task;
  auto results = AssertQueryBuilder(plan).copyResults(pool(), task);
  facebook::velox::test::assertEqualVectors(
      makeRowVector({"x"}, {makeFlatVector<int64_t>({2, 4, 6, 8, 10})}),
      results);

  auto stats = toPlanStats(task->taskStats());
  auto& projStats = stats.at(projNodeId);
  EXPECT_EQ(projStats.operatorStats.count("FilterProject"), 1);
  EXPECT_EQ(projStats.operatorStats.count("CudfFilterProject"), 0);
}

// Exercises appending after a kept operator and operator-ID renumbering.
TEST_F(AdapterOperatorTest, keptOperatorGetsAppendedOperators) {
  registerAdapterFirst(std::make_unique<AppendingAdapter>());

  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  core::PlanNodeId projNodeId;
  auto plan = PlanBuilder()
                  .values({data})
                  .project({"c0 * 2 as x"})
                  .capturePlanNodeId(projNodeId)
                  .planNode();

  std::shared_ptr<exec::Task> task;
  auto results = AssertQueryBuilder(plan).copyResults(pool(), task);

  facebook::velox::test::assertEqualVectors(
      makeRowVector({"x"}, {makeFlatVector<int64_t>({2, 4, 6, 8, 10})}),
      results);

  // All three operators must report under the original plan node.
  auto stats = toPlanStats(task->taskStats());
  auto& projStats = stats.at(projNodeId);
  EXPECT_TRUE(projStats.isMultiOperatorTypeNode());
  EXPECT_EQ(projStats.operatorStats.count("FilterProject"), 1);
  EXPECT_EQ(projStats.operatorStats.count("CudfFromVelox"), 1);
  EXPECT_EQ(projStats.operatorStats.count("CudfToVelox"), 1);
}

TEST_F(AdapterOperatorTest, fromVeloxRejectsDeviceInput) {
  // Values is classified as a CPU operator, so compile() places CudfFromVelox
  // after it. A device-resident batch arriving there means the upstream
  // operator was misclassified.
  auto uploadToDevice = [this](const RowVectorPtr& input) -> RowVectorPtr {
    auto stream = cudf::get_default_stream();
    auto table = cudf_velox::with_arrow::toCudfTable(
        input, pool(), stream, cudf::get_current_device_resource_ref());
    const auto numRows = table->num_rows();
    return std::make_shared<cudf_velox::CudfVector>(
        pool(), asRowType(input->type()), numRows, std::move(table), stream);
  };
  auto deviceBatch = uploadToDevice(
      makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 2, 3})}));
  auto plan =
      PlanBuilder().values({deviceBatch}).project({"c0 + 1 as x"}).planNode();
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "CudfFromVelox received a device-resident CudfVector");

  // The check applies to every input, not only the first one.
  auto hostBatch = makeRowVector({"c0"}, {makeFlatVector<int64_t>({4, 5, 6})});
  plan = PlanBuilder()
             .values({hostBatch, deviceBatch})
             .project({"c0 + 1 as x"})
             .planNode();
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "CudfFromVelox received a device-resident CudfVector");
}

TEST_F(AdapterOperatorTest, adaptsCudfOperatorInsertedByEarlierAdapter) {
  auto data = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
       makeFlatVector<int64_t>({10, 20, 30, 40, 50})});

  core::PlanNodeId projNodeId;
  auto plan = PlanBuilder()
                  .values({data})
                  .project({"c0", "c1"})
                  .capturePlanNodeId(projNodeId)
                  .planNode();

  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan).copyResults(pool(), task);
  facebook::velox::test::assertEqualVectors(data, result);

  auto stats = toPlanStats(task->taskStats());
  const auto& projStats = stats.at(projNodeId);
  EXPECT_EQ(projStats.operatorStats.size(), 3);
  EXPECT_TRUE(projStats.operatorStats.count("CudfFromVelox"));
  EXPECT_TRUE(projStats.operatorStats.count("CudfBatchConcat"));
  EXPECT_TRUE(projStats.operatorStats.count("CudfToVelox"));
  EXPECT_FALSE(projStats.operatorStats.count("FilterProject"));
}

TEST_F(AdapterOperatorTest, groupedScalarFilterUsesTypedExpressions) {
  auto queryCtx = core::QueryCtx::create();
  core::ExecCtx execCtx(pool(), queryCtx.get());
  const auto augmentedType = ROW(
      {"group_id", "value", "scalar_value"}, {BIGINT(), BIGINT(), BIGINT()});
  auto predicate = cudf_velox::test_utils::parseAndInferTypedExpr(
      "group_id = 0 AND value > scalar_value + (1 + 1)",
      augmentedType,
      &execCtx);

  auto grouped = makeRowVector(
      {"group_id", "value"},
      {makeFlatVector<int64_t>({0, 0, 0}),
       makeFlatVector<int64_t>({5, 12, 20})});
  auto scalar = makeRowVector(
      {"group_id", "value"},
      {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({10})});
  auto expected = makeRowVector(
      {"group_id", "value"},
      {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({20})});

  const auto run = [&](const std::vector<RowVectorPtr>& batches,
                       const RowVectorPtr& expectedRows) {
    auto plan = PlanBuilder()
                    .values(batches)
                    .addNode([&](core::PlanNodeId id, core::PlanNodePtr source) {
                      return std::make_shared<core::GroupedScalarFilterNode>(
                          id,
                          std::move(source),
                          "group_id",
                          0,
                          1,
                          "value",
                          "scalar_value",
                          predicate);
                    })
                    .planNode();
    std::shared_ptr<Task> task;
    auto result = AssertQueryBuilder(plan).copyResults(pool(), task);
    facebook::velox::test::assertEqualVectors(expectedRows, result);
    const auto stats = toPlanStats(task->taskStats());
    EXPECT_TRUE(
        stats.at(plan->id()).operatorStats.count("CudfGroupedScalarFilter"));
  };
  run({grouped, scalar}, expected);
  auto empty = makeRowVector(
      {"group_id", "value"},
      {makeFlatVector<int64_t>(std::vector<int64_t>{}),
       makeFlatVector<int64_t>(std::vector<int64_t>{})});
  run({grouped}, empty);
  run({empty}, empty);
}
