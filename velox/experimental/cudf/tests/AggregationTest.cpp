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
#include "velox/experimental/cudf/exec/AggregationRegistry.h"
#include "velox/experimental/cudf/exec/CudfGroupby.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/PrestoAggregateFunctions.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/type/Timestamp.h"

#include <folly/ScopeGuard.h>

#include <cmath>

namespace facebook::velox::exec::test {

using core::QueryConfig;
using facebook::velox::test::BatchMaker;
using namespace common::testutil;

class AggregationTest : public OperatorTestBase {
 public:
  enum class AggSteps { kSingle, kPartialFinal, kPartialIntermediateFinal };

 protected:
  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
    TestValue::enable();
  }

  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
    cudf_velox::registerPrestoAggregateFunctions("");
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    cudf_velox::unregisterAggregateFunctions();
    OperatorTestBase::TearDown();
  }

  std::vector<RowVectorPtr>
  makeVectors(const RowTypePtr& rowType, size_t size, int numVectors) {
    std::vector<RowVectorPtr> vectors;
    VectorFuzzer fuzzer({.vectorSize = size}, pool());
    for (int32_t i = 0; i < numVectors; ++i) {
      vectors.push_back(fuzzer.fuzzInputRow(rowType));
    }
    return vectors;
  }

  template <typename T>
  void testSingleKey(
      const std::vector<RowVectorPtr>& vectors,
      const std::string& keyName,
      bool ignoreNullKeys,
      bool distinct) {
    std::vector<std::string> aggregates;
    if (!distinct) {
      // TODO (dm): "sum(15)", "sum(0.1)",  "min(15)",  "min(0.1)", "max(15)",
      // "max(0.1)",
      aggregates = {
          "sum(c1)",
          "sum(c2)",
          "sum(c4)",
          "sum(c5)",
          "min(c1)",
          "min(c2)",
          "min(c3)",
          "min(c4)",
          "min(c5)",
          "max(c1)",
          "max(c2)",
          "max(c3)",
          "max(c4)",
          "max(c5)"};
    }

    auto op = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {keyName},
                      aggregates,
                      {},
                      core::AggregationNode::Step::kPartial,
                      ignoreNullKeys)
                  .planNode();

    std::string fromClause = "FROM tmp";
    if (ignoreNullKeys) {
      fromClause += " WHERE " + keyName + " IS NOT NULL";
    }
    if (distinct) {
      assertQuery(op, "SELECT distinct " + keyName + " " + fromClause);
    } else {
      // TODO (dm): sum(15), sum(cast(0.1 as double)), min(15), min(0.1),
      // max(15), max(0.1),
      assertQuery(
          op,
          "SELECT " + keyName +
              ", sum(c1), sum(c2), sum(c4), sum(c5) , min(c1), min(c2), min(c3), min(c4), min(c5), max(c1), max(c2), max(c3), max(c4), max(c5) " +
              fromClause + " GROUP BY " + keyName);
    }
  }

  void testMultiKey(
      const std::vector<RowVectorPtr>& vectors,
      bool ignoreNullKeys,
      bool distinct) {
    std::vector<std::string> aggregates;
    // TODO (dm): "sum(15)", "sum(0.1)",  "min(15)",  "min(0.1)", "max(15)",
    // "max(0.1)"
    if (!distinct) {
      aggregates = {
          "sum(c4)",
          "sum(c5)",
          "min(c3)",
          "min(c4)",
          "min(c5)",
          "max(c3)",
          "max(c4)",
          "max(c5)"};
    }
    auto op = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0", "c1", "c6"},
                      aggregates,
                      {},
                      core::AggregationNode::Step::kPartial,
                      ignoreNullKeys)
                  .planNode();

    std::string fromClause = "FROM tmp";
    if (ignoreNullKeys) {
      fromClause +=
          " WHERE c0 IS NOT NULL AND c1 IS NOT NULL AND c6 IS NOT NULL";
    }
    if (distinct) {
      assertQuery(op, "SELECT distinct c0, c1, c6 " + fromClause);
    } else {
      // TODO (dm): sum(15), sum(cast(0.1 as double)), min(15), min(0.1),
      // max(15), max(0.1),, sum(1)
      assertQuery(
          op,
          "SELECT c0, c1, c6, sum(c4), sum(c5), min(c3), min(c4), min(c5),  max(c3), max(c4), max(c5) " +
              fromClause + " GROUP BY c0, c1, c6");
    }
  }

  void testAggregation(
      const std::vector<RowVectorPtr>& data,
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates,
      const std::string& expectedSql,
      AggSteps steps) {
    auto builder = PlanBuilder().values(data);
    switch (steps) {
      case AggSteps::kSingle:
        builder.singleAggregation(groupingKeys, aggregates);
        break;
      case AggSteps::kPartialFinal:
        builder.partialAggregation(groupingKeys, aggregates).finalAggregation();
        break;
      case AggSteps::kPartialIntermediateFinal:
        builder.partialAggregation(groupingKeys, aggregates)
            .intermediateAggregation()
            .finalAggregation();
        break;
    }
    assertQuery(builder.planNode(), expectedSql);
  }

  void testGlobalCountStarZeroColumns(AggSteps steps) {
    auto data = makeRowVector({
        makeFlatVector<int64_t>({1, 2, 3, 4}),
    });
    createDuckDbTable({data});

    auto builder = PlanBuilder().values({data}).filter("c0 > 0").project({});
    switch (steps) {
      case AggSteps::kSingle:
        builder.singleAggregation({}, {"count(*)"});
        break;
      case AggSteps::kPartialFinal:
        builder.partialAggregation({}, {"count(*)"}).finalAggregation();
        break;
      case AggSteps::kPartialIntermediateFinal:
        builder.partialAggregation({}, {"count(*)"})
            .intermediateAggregation()
            .finalAggregation();
        break;
    }
    assertQuery(builder.planNode(), "SELECT count(*) FROM tmp WHERE c0 > 0");
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
          {BIGINT(),
           SMALLINT(),
           INTEGER(),
           BIGINT(),
           DOUBLE(), // DM: This used to be REAL() but we don't support that
           DOUBLE(),
           VARCHAR()})};
};

TEST_F(AggregationTest, global) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  // DM: removed "sum(15)","min(15)","max(15)",
  auto op = PlanBuilder()
                .values(vectors)
                .aggregation(
                    {},
                    {"sum(c1)",
                     "sum(c2)",
                     "sum(c4)",
                     "sum(c5)",

                     "min(c1)",
                     "min(c2)",
                     "min(c3)",
                     "min(c4)",
                     "min(c5)",

                     "max(c1)",
                     "max(c2)",
                     "max(c3)",
                     "max(c4)",
                     "max(c5)"},
                    {},
                    core::AggregationNode::Step::kPartial,
                    false)
                .planNode();

  // DM: removed sum(15), min(15), max(15),
  assertQuery(
      op,
      "SELECT sum(c1), sum(c2), sum(c4), sum(c5), "
      "min(c1), min(c2), min(c3), min(c4), min(c5), "
      "max(c1), max(c2), max(c3), max(c4), max(c5) FROM tmp");
}

TEST_F(AggregationTest, minMaxTimestampGlobal) {
  std::vector<std::optional<Timestamp>> timestamps = {
      Timestamp(1609459200, 0), // 2021-01-01 00:00:00
      Timestamp(1609459200, 500000000), // 2021-01-01 00:00:00.500
      Timestamp(1609545600, 0), // 2021-01-02 00:00:00
      std::nullopt,
      Timestamp(1609459199, 900000000) // 2020-12-31 23:59:59.900
  };

  auto data = makeRowVector(
      {makeNullableFlatVector<Timestamp>(timestamps, TIMESTAMP())});
  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"min(c0)", "max(c0)"})
                  .planNode();

  assertQuery(plan, "SELECT min(c0), max(c0) FROM tmp");
}

TEST_F(AggregationTest, minMaxTimestampGroupBy) {
  std::vector<std::optional<Timestamp>> timestamps = {
      Timestamp(1609459200, 0), // 2021-01-01 00:00:00
      std::nullopt,
      Timestamp(1609545600, 0), // 2021-01-02 00:00:00
      Timestamp(1609459199, 0), // 2020-12-31 23:59:59
      Timestamp(1609632000, 0) // 2021-01-03 00:00:00
  };

  auto data = makeRowVector(
      {makeFlatVector<int32_t>({1, 1, 2, 2, 2}),
       makeNullableFlatVector<Timestamp>(timestamps, TIMESTAMP())});
  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({"c0"}, {"min(c1)", "max(c1)"})
                  .planNode();

  assertQuery(plan, "SELECT c0, min(c1), max(c1) FROM tmp GROUP BY c0");
}

TEST_F(AggregationTest, minMaxDateGlobal) {
  // cuDF represents DATE as TIMESTAMP_DAYS, a distinct type from TIMESTAMP, so
  // exercise min/max on it directly.
  std::vector<std::optional<int32_t>> dates = {
      DATE()->toDays("2021-01-01"),
      DATE()->toDays("2021-01-02"),
      std::nullopt,
      DATE()->toDays("2020-12-31"),
      DATE()->toDays("2021-01-03"),
  };

  auto data = makeRowVector({makeNullableFlatVector<int32_t>(dates, DATE())});
  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"min(c0)", "max(c0)"})
                  .planNode();

  assertQuery(plan, "SELECT min(c0), max(c0) FROM tmp");
}

TEST_F(AggregationTest, minMaxDateGroupBy) {
  std::vector<std::optional<int32_t>> dates = {
      DATE()->toDays("2021-01-01"),
      std::nullopt,
      DATE()->toDays("2021-01-02"),
      DATE()->toDays("2020-12-31"),
      DATE()->toDays("2021-01-03"),
  };

  auto data = makeRowVector(
      {makeFlatVector<int32_t>({1, 1, 2, 2, 2}),
       makeNullableFlatVector<int32_t>(dates, DATE())});
  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({"c0"}, {"min(c1)", "max(c1)"})
                  .planNode();

  assertQuery(plan, "SELECT c0, min(c1), max(c1) FROM tmp GROUP BY c0");
}

TEST_F(AggregationTest, singleBigintKey) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testSingleKey<int64_t>(vectors, "c0", false, false);
  testSingleKey<int64_t>(vectors, "c0", true, false);
}

TEST_F(AggregationTest, singleBigintKeyDistinct) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testSingleKey<int64_t>(vectors, "c0", false, true);
  testSingleKey<int64_t>(vectors, "c0", true, true);
}

TEST_F(AggregationTest, singleStringKey) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testSingleKey<StringView>(vectors, "c6", false, false);
  testSingleKey<StringView>(vectors, "c6", true, false);
}

TEST_F(AggregationTest, singleStringKeyDistinct) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testSingleKey<StringView>(vectors, "c6", false, true);
  testSingleKey<StringView>(vectors, "c6", true, true);
}

TEST_F(AggregationTest, multiKey) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testMultiKey(vectors, false, false);
  testMultiKey(vectors, true, false);
}

TEST_F(AggregationTest, multiKeyDistinct) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testMultiKey(vectors, false, true);
  testMultiKey(vectors, true, true);
}

TEST_F(AggregationTest, aggregateOfNulls) {
  auto rowVector = makeRowVector({
      BatchMaker::createVector<TypeKind::BIGINT>(
          rowType_->childAt(0), 100, *pool_),
      makeNullConstant(TypeKind::SMALLINT, 100),
  });

  auto vectors = {rowVector};
  createDuckDbTable(vectors);

  auto op = PlanBuilder()
                .values(vectors)
                .aggregation(
                    {"c0"},
                    {"sum(c1)", "min(c1)", "max(c1)"},
                    {},
                    core::AggregationNode::Step::kPartial,
                    false)
                .planNode();

  assertQuery(op, "SELECT c0, sum(c1), min(c1), max(c1) FROM tmp GROUP BY c0");

  // global aggregation
  op = PlanBuilder()
           .values(vectors)
           .aggregation(
               {},
               {"sum(c1)", "min(c1)", "max(c1)"},
               {},
               core::AggregationNode::Step::kPartial,
               false)
           .planNode();

  assertQuery(op, "SELECT sum(c1), min(c1), max(c1) FROM tmp");
}

TEST_F(AggregationTest, varcharMinMax) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  // Groupby with varchar min/max.
  auto op = PlanBuilder()
                .values(vectors)
                .aggregation(
                    {"c0"},
                    {"min(c6)", "max(c6)"},
                    {},
                    core::AggregationNode::Step::kPartial,
                    false)
                .planNode();

  assertQuery(op, "SELECT c0, min(c6), max(c6) FROM tmp GROUP BY c0");

  // Global aggregation with varchar min/max.
  op = PlanBuilder()
           .values(vectors)
           .aggregation(
               {},
               {"min(c6)", "max(c6)"},
               {},
               core::AggregationNode::Step::kPartial,
               false)
           .planNode();

  assertQuery(op, "SELECT min(c6), max(c6) FROM tmp");
}

TEST_F(AggregationTest, allKeyTypes) {
  // Covers different key types. Unlike the integer/string tests, the
  // hash table begins life in the generic mode, not array or
  // normalized key. Add types here as they become supported.
  auto rowType = ROW(
      {"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
      {DOUBLE(), REAL(), BIGINT(), INTEGER(), BOOLEAN(), VARCHAR(), DOUBLE()});

  std::vector<RowVectorPtr> batches;
  for (auto i = 0; i < 10; ++i) {
    batches.push_back(
        std::static_pointer_cast<RowVector>(
            BatchMaker::createBatch(rowType, 100, *pool_)));
  }
  createDuckDbTable(batches);
  auto op =
      PlanBuilder()
          .values(batches)
          .singleAggregation({"c0", "c1", "c2", "c3", "c4", "c5"}, {"sum(c6)"})
          .planNode();

  // DM: Instead of sum(c6), this was sum(1) but we don't yet support constants
  assertQuery(
      op,
      "SELECT c0, c1, c2, c3, c4, c5, sum(c6) FROM tmp "
      " GROUP BY c0, c1, c2, c3, c4, c5");
}

TEST_F(AggregationTest, ignoreNullKeys) {
  // Some keys are null.
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>(
          {std::nullopt, 1, std::nullopt, 2, std::nullopt, 1, 2}),
      makeFlatVector<int32_t>({-1, 1, -2, 2, -3, 3, 4}),
  });

  auto makePlan = [&](bool ignoreNullKeys) {
    return PlanBuilder()
        .values({data})
        .aggregation(
            {"c0"},
            {"sum(c1)"},
            {},
            core::AggregationNode::Step::kPartial,
            ignoreNullKeys)
        .planNode();
  };

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<int64_t>({4, 6}),
  });
  AssertQueryBuilder(makePlan(true)).assertResults(expected);

  expected = makeRowVector({
      makeNullableFlatVector<int32_t>({std::nullopt, 1, 2}),
      makeFlatVector<int64_t>({-6, 4, 6}),
  });
  AssertQueryBuilder(makePlan(false)).assertResults(expected);

  // All keys are null.
  data = makeRowVector({
      makeAllNullFlatVector<int32_t>(3),
      makeFlatVector<int32_t>({1, 2, 3}),
  });

  AssertQueryBuilder(makePlan(true)).assertEmptyResults();
}

TEST_F(AggregationTest, avgSingleGrouped) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  // DM: removed avg(c3). We're having overflow issues with int64_t.
  std::vector<std::string> aggregates = {
      "avg(c1)", "avg(c2)", "avg(c4)", "avg(c5)"};

  std::string keyName = "c0";
  auto op = PlanBuilder()
                .values(vectors)
                .singleAggregation({keyName}, aggregates)
                .planNode();

  assertQuery(
      op,
      "SELECT " + keyName + ", avg(c1), avg(c2), avg(c4), avg(c5) " +
          "FROM tmp GROUP BY " + keyName);
}

TEST_F(AggregationTest, avgPartialFinalGrouped) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  // DM: removed avg(c3). We're having overflow issues with int64_t.
  std::vector<std::string> aggregates = {
      "avg(c1)", "avg(c2)", "avg(c4)", "avg(c5)"};

  std::string keyName = "c0";
  auto op = PlanBuilder()
                .values(vectors)
                .partialAggregation({keyName}, aggregates)
                .finalAggregation()
                .planNode();

  assertQuery(
      op,
      "SELECT " + keyName + ", avg(c1), avg(c2), avg(c4), avg(c5) " +
          "FROM tmp GROUP BY " + keyName);
}

TEST_F(AggregationTest, avgSingleGlobal) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  std::vector<std::string> aggregates = {
      "avg(c1)", "avg(c2)", "avg(c4)", "avg(c5)"};
  auto op = PlanBuilder()
                .values(vectors)
                .singleAggregation({}, aggregates)
                .planNode();

  assertQuery(op, "SELECT avg(c1), avg(c2), avg(c4), avg(c5) FROM tmp");
}

TEST_F(AggregationTest, avgPartialFinalGlobal) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  std::vector<std::string> aggregates = {
      "avg(c1)", "avg(c2)", "avg(c4)", "avg(c5)"};

  auto op = PlanBuilder()
                .values(vectors)
                .partialAggregation({}, aggregates)
                .finalAggregation()
                .planNode();

  assertQuery(op, "SELECT avg(c1), avg(c2), avg(c4), avg(c5) FROM tmp");
}

TEST_F(AggregationTest, countStarGlobal) {
  auto vectors = makeVectors(rowType_, 10, 100);

  createDuckDbTable(vectors);

  auto op = PlanBuilder()
                .values(vectors)
                .filter("c0 > 10")
                .project({})
                .partialAggregation({}, {"count(*)"})
                .finalAggregation()
                .planNode();

  assertQuery(op, "SELECT count(*) FROM tmp WHERE c0 > 10");
}

TEST_F(AggregationTest, countStarGlobalNonZeroRowsColumns) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
  });
  createDuckDbTable({data});

  auto op = PlanBuilder()
                .values({data})
                .partialAggregation({}, {"count(*)"})
                .finalAggregation()
                .planNode();

  assertQuery(op, "SELECT count(*) FROM tmp");
}

TEST_F(AggregationTest, countStarGlobalZeroRows) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
  });
  createDuckDbTable({data});

  auto op = PlanBuilder()
                .values({data})
                .filter("c0 > 10")
                .partialAggregation({}, {"count(*)"})
                .finalAggregation()
                .planNode();

  assertQuery(op, "SELECT count(*) FROM tmp WHERE c0 > 10");
}

TEST_F(AggregationTest, countStarGlobalPartialFinalZeroColumnsLocalPartition) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
  });
  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .values({data})
                  .filter("c0 > 0")
                  .project({})
                  .partialAggregation({}, {"count(*)"})
                  .localPartitionRoundRobin()
                  .finalAggregation()
                  .planNode();

  AssertQueryBuilder(duckDbQueryRunner_)
      .config(core::QueryConfig::kMaxLocalExchangePartitionCount, "2")
      .plan(plan)
      .assertResults("SELECT count(*) FROM tmp WHERE c0 > 0");
}

TEST_F(AggregationTest, countConstantSingleGroupByNonZeroKey) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testAggregation(
      vectors,
      {"c2"},
      {"count(1)"},
      "SELECT c2, count(1) FROM tmp GROUP BY c2",
      AggSteps::kSingle);
}

TEST_F(AggregationTest, countConstantPartialFinalGroupByNonZeroKey) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testAggregation(
      vectors,
      {"c2"},
      {"count(1)"},
      "SELECT c2, count(1) FROM tmp GROUP BY c2",
      AggSteps::kPartialFinal);
}

// Parameterized fixture that runs each count-aggregation scenario across
// single, partial+final, and partial+intermediate+final steps.
class CountAggregationStepsTest
    : public AggregationTest,
      public testing::WithParamInterface<AggregationTest::AggSteps> {};

TEST_P(CountAggregationStepsTest, countStarGlobalZeroColumns) {
  testGlobalCountStarZeroColumns(GetParam());
}

TEST_P(CountAggregationStepsTest, countStarVsCountColumnGlobalNulls) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({1, std::nullopt, 2, std::nullopt}),
  });
  createDuckDbTable({data});
  testAggregation(
      {data},
      {},
      {"count(*)", "count(c0)"},
      "SELECT count(*), count(c0) FROM tmp",
      GetParam());
}

TEST_P(CountAggregationStepsTest, countGroupBy) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testAggregation(
      vectors,
      {"c0"},
      {"count(0)"},
      "SELECT c0, count(*) FROM tmp GROUP BY c0",
      GetParam());
}

TEST_P(CountAggregationStepsTest, countConstantGroupBy) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testAggregation(
      vectors,
      {"c0"},
      {"count(1)"},
      "SELECT c0, count(1) FROM tmp GROUP BY c0",
      GetParam());
}

TEST_P(CountAggregationStepsTest, countGlobal) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testAggregation(
      vectors, {}, {"count(0)"}, "SELECT count(*) FROM tmp", GetParam());
}

TEST_P(CountAggregationStepsTest, countStarGlobal) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testAggregation(
      vectors, {}, {"count(*)"}, "SELECT count(*) FROM tmp", GetParam());
}

TEST_P(CountAggregationStepsTest, countStarGroupBy) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testAggregation(
      vectors,
      {"c0"},
      {"count(*)"},
      "SELECT c0, count(*) FROM tmp GROUP BY c0",
      GetParam());
}

TEST_P(CountAggregationStepsTest, countColumnGlobal) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testAggregation(
      vectors, {}, {"count(c0)"}, "SELECT count(c0) FROM tmp", GetParam());
}

TEST_P(CountAggregationStepsTest, countColumnGlobalNulls) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({1, std::nullopt, 2, std::nullopt}),
  });
  createDuckDbTable({data});
  testAggregation(
      {data}, {}, {"count(c0)"}, "SELECT count(c0) FROM tmp", GetParam());
}

TEST_P(CountAggregationStepsTest, countColumnGroupBy) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testAggregation(
      vectors,
      {"c0"},
      {"count(c3)"},
      "SELECT c0, count(c3) FROM tmp GROUP BY c0",
      GetParam());
}

TEST_P(CountAggregationStepsTest, countColumnGroupByNulls) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3}),
      makeNullableFlatVector<int64_t>(
          {10, std::nullopt, 20, std::nullopt, std::nullopt, std::nullopt}),
  });
  createDuckDbTable({data});
  testAggregation(
      {data},
      {"c0"},
      {"count(c1)"},
      "SELECT c0, count(c1) FROM tmp GROUP BY c0",
      GetParam());
}

TEST_P(CountAggregationStepsTest, countStarVsCountColumnGroupByNulls) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3}),
      makeNullableFlatVector<int64_t>(
          {10, std::nullopt, 20, std::nullopt, std::nullopt, std::nullopt}),
  });
  createDuckDbTable({data});
  testAggregation(
      {data},
      {"c0"},
      {"count(*)", "count(c1)"},
      "SELECT c0, count(*), count(c1) FROM tmp GROUP BY c0",
      GetParam());
}

TEST_P(CountAggregationStepsTest, countNullConstantMarkerForIntersectShape) {
  auto data = makeRowVector({
      makeFlatVector<StringView>({"left_only", "left_only", "both"}),
  });

  auto plan = PlanBuilder()
                  .values({data})
                  .project({
                      "true AS left_marker",
                      "cast(null AS boolean) AS right_marker",
                      "c0 AS key",
                  })
                  .partialAggregation(
                      {"key"}, {"count(left_marker)", "count(right_marker)"})
                  .finalAggregation()
                  .filter("a0 >= 1 AND a1 = 0")
                  .project({"key", "a0"})
                  .planNode();

  auto expected = makeRowVector({
      makeFlatVector<StringView>({"left_only", "both"}),
      makeFlatVector<int64_t>({2, 1}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_P(CountAggregationStepsTest, countConstantGlobalNulls) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({1, std::nullopt, 2, std::nullopt}),
  });
  createDuckDbTable({data});
  testAggregation(
      {data}, {}, {"count(1)"}, "SELECT count(1) FROM tmp", GetParam());
}

TEST_P(CountAggregationStepsTest, countNullGlobal) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
  });
  createDuckDbTable({data});
  testAggregation(
      {data}, {}, {"count(null)"}, "SELECT count(null) FROM tmp", GetParam());
}

TEST_P(CountAggregationStepsTest, countNullGroupBy) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
  });
  createDuckDbTable({data});
  testAggregation(
      {data},
      {"c0"},
      {"count(null)"},
      "SELECT c0, count(null) FROM tmp GROUP BY c0",
      GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    CountAggregation,
    CountAggregationStepsTest,
    testing::Values(
        AggregationTest::AggSteps::kSingle,
        AggregationTest::AggSteps::kPartialFinal,
        AggregationTest::AggSteps::kPartialIntermediateFinal),
    [](const testing::TestParamInfo<AggregationTest::AggSteps>& info)
        -> std::string {
      switch (info.param) {
        case AggregationTest::AggSteps::kSingle:
          return "Single";
        case AggregationTest::AggSteps::kPartialFinal:
          return "PartialFinal";
        case AggregationTest::AggSteps::kPartialIntermediateFinal:
          return "PartialIntermediateFinal";
      }
      return "Unknown";
    });

/// Tests the spark scenario of having different types of aggs in the same
/// planNode Specific example being tested is
/// https://github.com/facebookincubator/velox/issues/12830#issuecomment-2783340233
TEST_F(AggregationTest, CompanionAggs) {
  std::vector<int64_t> keys0{1, 1, 1, 2, 1, 1, 2, 2};
  std::vector<int64_t> keys1{1, 2, 1, 2, 1, 2, 1, 2};
  std::vector<int64_t> values{1, 2, 3, 4, 5, 6, 7, 8};
  auto rowVector = makeRowVector(
      {makeFlatVector<int64_t>(keys0),
       makeFlatVector<int64_t>(keys1),
       makeFlatVector<int64_t>(values)});

  createDuckDbTable({rowVector});

  auto op =
      PlanBuilder()
          .values({rowVector})
          .singleAggregation({"c2", "c0"}, {"count_partial(c1)"})
          .localPartition({"c2", "c0"})
          .singleAggregation({"c0"}, {"count_merge(a0)", "count_partial(c2)"})
          .localPartition({"c0"})
          .singleAggregation({"c0"}, {"count_merge(a0)", "count_merge(a1)"})
          .planNode();
  assertQuery(
      op, "SELECT c0, count(c1), count(distinct c2) FROM tmp GROUP BY c0");
}

TEST_F(AggregationTest, partialAggregationMemoryLimit) {
  auto vectors = {
      makeRowVector({makeFlatVector<int32_t>(
          100, [](auto row) { return row; }, nullEvery(5))}),
      makeRowVector({makeFlatVector<int32_t>(
          110, [](auto row) { return row + 29; }, nullEvery(7))}),
      makeRowVector({makeFlatVector<int32_t>(
          90, [](auto row) { return row - 71; }, nullEvery(7))}),
  };

  createDuckDbTable(vectors);

  // Set an artificially low limit on the amount of data to accumulate in
  // the partial aggregation.

  // Distinct aggregation.
  core::PlanNodeId aggNodeId;
  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config(QueryConfig::kMaxPartialAggregationMemory, 100)
                  .plan(
                      PlanBuilder()
                          .values(vectors)
                          .partialAggregation({"c0"}, {})
                          .capturePlanNodeId(aggNodeId)
                          .finalAggregation()
                          .planNode())
                  .assertResults("SELECT distinct c0 FROM tmp");

  auto rowFlushStats = toPlanStats(task->taskStats())
                           .at(aggNodeId)
                           .customStats.at("flushRowCount");
  EXPECT_GT(rowFlushStats.sum, 0);
  EXPECT_GT(rowFlushStats.max, 0);

  // Count aggregation.
  task = AssertQueryBuilder(duckDbQueryRunner_)
             .config(QueryConfig::kMaxPartialAggregationMemory, 1)
             .plan(
                 PlanBuilder()
                     .values(vectors)
                     .partialAggregation({"c0"}, {"count(1)"})
                     .capturePlanNodeId(aggNodeId)
                     .finalAggregation()
                     .planNode())
             .assertResults("SELECT c0, count(1) FROM tmp GROUP BY 1");

  rowFlushStats = toPlanStats(task->taskStats())
                      .at(aggNodeId)
                      .customStats.at("flushRowCount");
  EXPECT_GT(rowFlushStats.sum, 0);
  EXPECT_GT(rowFlushStats.max, 0);

  // Global aggregation.
  task = AssertQueryBuilder(duckDbQueryRunner_)
             .config(QueryConfig::kMaxPartialAggregationMemory, 1)
             .plan(
                 PlanBuilder()
                     .values(vectors)
                     .partialAggregation({}, {"sum(c0)"})
                     .capturePlanNodeId(aggNodeId)
                     .finalAggregation()
                     .planNode())
             .assertResults("SELECT sum(c0) FROM tmp");
  EXPECT_EQ(
      0,
      toPlanStats(task->taskStats())
          .at(aggNodeId)
          .customStats.count("flushRowCount"));
}

TEST_F(AggregationTest, partialAggregationOmitsDiagnosticFlushThresholdStat) {
  auto vectors = {
      makeRowVector({makeFlatVector<int32_t>(
          100, [](auto row) { return row % 10; }, nullEvery(5))}),
      makeRowVector({makeFlatVector<int32_t>(
          110, [](auto row) { return row % 10; }, nullEvery(7))}),
      makeRowVector({makeFlatVector<int32_t>(
          90, [](auto row) { return row % 10; }, nullEvery(7))}),
  };

  createDuckDbTable(vectors);

  core::PlanNodeId aggNodeId;
  constexpr int64_t maxPartialAggregationMemory = 2 << 20;
  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config(
                      QueryConfig::kMaxPartialAggregationMemory,
                      maxPartialAggregationMemory)
                  .plan(
                      PlanBuilder()
                          .values(vectors)
                          .partialAggregation({"c0"}, {"count(1)"})
                          .capturePlanNodeId(aggNodeId)
                          .finalAggregation()
                          .planNode())
                  .assertResults("SELECT c0, count(1) FROM tmp GROUP BY 1");

  EXPECT_EQ(
      0,
      toPlanStats(task->taskStats())
          .at(aggNodeId)
          .customStats.count("cudfPartialAggregationFlushThresholdBytes"));
}

TEST_F(AggregationTest, partialAggregationSlicesOversizedInput) {
  auto& cudfConfig = cudf_velox::CudfConfig::getInstance();
  const auto savedMaxBatchRows = cudfConfig.batchSizeMaxThreshold;
  constexpr int32_t kMaxSliceRows = 7;
  cudfConfig.batchSizeMaxThreshold = kMaxSliceRows;
  SCOPE_EXIT {
    cudfConfig.batchSizeMaxThreshold = savedMaxBatchRows;
  };

  constexpr vector_size_t kRows = 33;
  auto input = makeRowVector({
      makeFlatVector<int32_t>(kRows, [](auto row) { return row % 9; }),
      makeFlatVector<int64_t>(
          kRows, [](auto row) { return static_cast<int64_t>(row + 1); }),
  });
  createDuckDbTable({input});

  core::PlanNodeId partialAggId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .plan(
              PlanBuilder()
                  .values({input})
                  .partialAggregation(
                      {"c0"}, {"sum(c1)", "count(c1)", "min(c1)", "max(c1)"})
                  .capturePlanNodeId(partialAggId)
                  .finalAggregation()
                  .planNode())
          .maxDrivers(1)
          .assertResults(
              "SELECT c0, sum(c1), count(c1), min(c1), max(c1) "
              "FROM tmp GROUP BY c0");

  const auto planStats = toPlanStats(task->taskStats());
  const auto& partialStats = planStats.at(partialAggId);
  EXPECT_GT(
      partialStats.customStats.at("cudfGroupbyInputSliceTargetBytes").sum, 0);
  EXPECT_EQ(
      partialStats.customStats.at("cudfGroupbyInputSlices").sum,
      (kRows + kMaxSliceRows - 1) / kMaxSliceRows);
  EXPECT_EQ(
      partialStats.customStats.at("cudfGroupbyInputSliceRows").sum, kRows);
  EXPECT_EQ(
      partialStats.customStats.at("cudfGroupbyMaxInputSliceRows").max,
      kMaxSliceRows);
}

TEST_F(AggregationTest, partialAggregationSlicesByByteBudgetWithoutRowLimit) {
  auto& cudfConfig = cudf_velox::CudfConfig::getInstance();
  const auto savedMaxBatchRows = cudfConfig.batchSizeMaxThreshold;
  cudfConfig.batchSizeMaxThreshold.reset();
  SCOPE_EXIT {
    cudfConfig.batchSizeMaxThreshold = savedMaxBatchRows;
  };

  constexpr vector_size_t kRows = 41;
  constexpr int64_t kInputWorkBytes = 64;
  const auto makeInput = [&](vector_size_t begin, vector_size_t size) {
    return makeRowVector({
        makeFlatVector<int32_t>(
            size, [begin](auto row) { return begin + row; }),
        makeFlatVector<int64_t>(
            size,
            [begin](auto row) {
              return static_cast<int64_t>(begin + row + 1);
            }),
    });
  };
  auto firstInput = makeInput(0, 4);
  auto secondInput = makeInput(4, kRows - 4);
  createDuckDbTable({firstInput, secondInput});

  core::PlanNodeId partialAggId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config(QueryConfig::kMaxPartialAggregationMemory, kInputWorkBytes)
          .plan(
              PlanBuilder()
                  .values({firstInput, secondInput})
                  .partialAggregation({"c0"}, {"sum(c1)"})
                  .capturePlanNodeId(partialAggId)
                  .finalAggregation()
                  .planNode())
          .maxDrivers(1)
          .assertResults("SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  const auto planStats = toPlanStats(task->taskStats());
  const auto& partialStats = planStats.at(partialAggId);
  EXPECT_LE(
      partialStats.customStats.at("cudfGroupbyInputSliceTargetBytes").max,
      kInputWorkBytes);
  EXPECT_GT(partialStats.customStats.at("cudfGroupbyInputSlices").sum, 1);
  EXPECT_EQ(
      partialStats.customStats.at("cudfGroupbyInputSliceRows").sum, kRows);
  EXPECT_LT(
      partialStats.customStats.at("cudfGroupbyMaxInputSliceRows").max, kRows);
}

TEST_F(AggregationTest, closeReleasesPendingPartialInput) {
  auto& cudfConfig = cudf_velox::CudfConfig::getInstance();
  const auto savedMaxBatchRows = cudfConfig.batchSizeMaxThreshold;
  cudfConfig.batchSizeMaxThreshold = 2;
  SCOPE_EXIT {
    cudfConfig.batchSizeMaxThreshold = savedMaxBatchRows;
  };

  auto rawInput = makeRowVector({
      makeFlatVector<int32_t>({0, 1, 2, 3, 4}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
  });
  auto planNode = PlanBuilder()
                      .values({rawInput})
                      .partialAggregation({"c0"}, {"sum(c1)"})
                      .planNode();
  auto partialAggregation =
      std::dynamic_pointer_cast<const core::AggregationNode>(planNode);
  ASSERT_NE(partialAggregation, nullptr);

  core::PlanFragment planFragment;
  planFragment.planNode = planNode;
  auto task = Task::create(
      "AggregationTest.closePendingPartialInput",
      std::move(planFragment),
      0,
      core::QueryCtx::create(driverExecutor_.get()),
      Task::ExecutionMode::kParallel);
  auto driverCtx = std::make_unique<DriverCtx>(task, 0, 0, 0, 0);
  auto* driverCtxPtr = driverCtx.get();
  auto driver = Driver::testingCreate(std::move(driverCtx));
  cudf_velox::CudfGroupby groupby(0, driverCtxPtr, partialAggregation);
  groupby.initialize();

  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  auto ownedTable =
      std::shared_ptr<cudf::table>(cudf_velox::with_arrow::toCudfTable(
          rawInput, pool(), stream, cudf::get_current_device_resource_ref()));
  std::weak_ptr<cudf::table> weakOwner = ownedTable;
  auto cudfInput = std::make_shared<cudf_velox::CudfVector>(
      pool(),
      partialAggregation->sources()[0]->outputType(),
      rawInput->size(),
      ownedTable->view(),
      cudf_velox::CudfVector::ViewOwner{ownedTable},
      stream);
  ownedTable.reset();

  groupby.addInput(std::move(cudfInput));
  EXPECT_FALSE(groupby.needsInput());
  EXPECT_FALSE(weakOwner.expired());
  groupby.close();
  EXPECT_TRUE(weakOwner.expired())
      << "close() must release an oversized input with unprocessed slices";
}

TEST_F(AggregationTest, noMoreInputDrainsPendingPartialSlices) {
  auto& cudfConfig = cudf_velox::CudfConfig::getInstance();
  const auto savedMaxBatchRows = cudfConfig.batchSizeMaxThreshold;
  cudfConfig.batchSizeMaxThreshold = 2;
  SCOPE_EXIT {
    cudfConfig.batchSizeMaxThreshold = savedMaxBatchRows;
  };

  auto rawInput = makeRowVector({
      makeFlatVector<int32_t>({0, 1, 2, 3, 4}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
  });
  auto planNode = PlanBuilder()
                      .values({rawInput})
                      .partialAggregation({"c0"}, {"sum(c1)"})
                      .planNode();
  auto partialAggregation =
      std::dynamic_pointer_cast<const core::AggregationNode>(planNode);
  ASSERT_NE(partialAggregation, nullptr);

  core::PlanFragment planFragment;
  planFragment.planNode = planNode;
  auto task = Task::create(
      "AggregationTest.drainPendingPartialInput",
      std::move(planFragment),
      0,
      core::QueryCtx::create(driverExecutor_.get()),
      Task::ExecutionMode::kParallel);
  auto driverCtx = std::make_unique<DriverCtx>(task, 0, 0, 0, 0);
  auto* driverCtxPtr = driverCtx.get();
  auto driver = Driver::testingCreate(std::move(driverCtx));
  cudf_velox::CudfGroupby groupby(0, driverCtxPtr, partialAggregation);
  groupby.initialize();

  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  auto ownedTable =
      std::shared_ptr<cudf::table>(cudf_velox::with_arrow::toCudfTable(
          rawInput, pool(), stream, cudf::get_current_device_resource_ref()));
  auto cudfInput = std::make_shared<cudf_velox::CudfVector>(
      pool(),
      partialAggregation->sources()[0]->outputType(),
      rawInput->size(),
      ownedTable->view(),
      cudf_velox::CudfVector::ViewOwner{ownedTable},
      stream);

  groupby.addInput(std::move(cudfInput));
  groupby.noMoreInput();

  vector_size_t outputRows = 0;
  for (int32_t attempts = 0; attempts < 10 && !groupby.isFinished();
       ++attempts) {
    if (auto output = groupby.getOutput()) {
      outputRows += output->size();
    }
  }
  EXPECT_TRUE(groupby.isFinished());
  EXPECT_EQ(outputRows, rawInput->size());
  groupby.close();
}

TEST_F(AggregationTest, finalAggregationStreamsOnAddInput) {
  auto vectors = {
      makeRowVector({makeFlatVector<int32_t>(
          100, [](auto row) { return row; }, nullEvery(5))}),
      makeRowVector({makeFlatVector<int32_t>(
          110, [](auto row) { return row + 29; }, nullEvery(7))}),
      makeRowVector({makeFlatVector<int32_t>(
          90, [](auto row) { return row - 71; }, nullEvery(7))}),
  };

  createDuckDbTable(vectors);

  // Force the final aggregation to see multiple addInput() calls by setting an
  // artificially low limit on the amount of data to accumulate in the partial
  // aggregation.
  core::PlanNodeId partialAggId;
  core::PlanNodeId finalAggId;
  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config(QueryConfig::kMaxPartialAggregationMemory, 1)
                  .plan(
                      PlanBuilder()
                          .values(vectors)
                          .partialAggregation({"c0"}, {"sum(c0)"})
                          .capturePlanNodeId(partialAggId)
                          .finalAggregation()
                          .capturePlanNodeId(finalAggId)
                          .planNode())
                  .assertResults("SELECT c0, sum(c0) FROM tmp GROUP BY 1");

  const auto planStats = toPlanStats(task->taskStats());
  EXPECT_GT(planStats.at(partialAggId).customStats.at("flushRowCount").sum, 0);
  EXPECT_GT(planStats.at(finalAggId).outputRows, 0);
  EXPECT_GT(
      planStats.at(finalAggId)
          .customStats.at("cudfFinalAggregationDirectInputBatches")
          .sum,
      0);
}

TEST_F(AggregationTest, boundedPartialRolloverAndNoReductionFinalMerge) {
  std::vector<RowVectorPtr> vectors;
  for (int32_t batch = 0; batch < 3; ++batch) {
    vectors.push_back(makeRowVector({
        makeFlatVector<int32_t>(
            100, [batch](auto row) { return batch * 100 + row; }),
        makeFlatVector<int64_t>(100, [](auto /*row*/) { return 1; }),
    }));
  }
  createDuckDbTable(vectors);

  // The small partial-state limit forces rollover before concatenating an
  // unbounded state. Depending on the physical state width, this can happen at
  // an input-slice boundary or when admitting the next partial result. The
  // disjoint keys then make the first final intermediate compaction reduce zero
  // rows, so later inputs must not repeatedly rewrite a growing state.
  constexpr int64_t kPartialStateBytes = 1'500;
  core::PlanNodeId partialAggId;
  core::PlanNodeId finalAggId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config(QueryConfig::kMaxPartialAggregationMemory, kPartialStateBytes)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"c0"}, {"sum(c1)"})
                  .capturePlanNodeId(partialAggId)
                  .finalAggregation()
                  .capturePlanNodeId(finalAggId)
                  .planNode())
          .maxDrivers(1)
          .assertResults("SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  const auto planStats = toPlanStats(task->taskStats());
  const auto statSum = [](const auto& stats, std::string_view name) {
    const auto it = stats.customStats.find(std::string(name));
    return it == stats.customStats.end() ? 0 : it->second.sum;
  };
  const auto& partialStats = planStats.at(partialAggId);
  const auto& finalStats = planStats.at(finalAggId);
  EXPECT_GT(
      statSum(partialStats, "cudfPartialAggregationPreemptiveFlush") +
          statSum(partialStats, "cudfPartialAggregationPreSliceFlush") +
          statSum(partialStats, "cudfPartialAggregationDeferredInput"),
      0);
  EXPECT_EQ(statSum(finalStats, "cudfIntermediateCompactionNoReduction"), 1);
  EXPECT_GE(statSum(finalStats, "cudfFinalAggregationDirectInputBatches"), 2);
}

TEST_F(AggregationTest, hashBucketedFinalAggregationAcrossInputBatches) {
  auto& cudfConfig = cudf_velox::CudfConfig::getInstance();
  const auto savedMaxBatchRows = cudfConfig.batchSizeMaxThreshold;
  constexpr int32_t kMaxBucketRows = 16;
  cudfConfig.batchSizeMaxThreshold = kMaxBucketRows;
  SCOPE_EXIT {
    cudfConfig.batchSizeMaxThreshold = savedMaxBatchRows;
  };

  constexpr vector_size_t kRowsPerBatch = 64;
  constexpr vector_size_t kUniqueKeys = 2 * kRowsPerBatch;
  std::vector<RowVectorPtr> vectors;
  for (int32_t batch = 0; batch < 4; ++batch) {
    vectors.push_back(makeRowVector({
        makeFlatVector<int32_t>(
            kRowsPerBatch,
            [batch](auto row) {
              // The first two batches establish disjoint key domains. Each
              // later batch repeats one domain. A persistent collector must
              // route equal keys to the same bucket across admissions and
              // fanout growth or the DuckDB comparison will find duplicates.
              return (batch % 2) * kRowsPerBatch + row;
            }),
        makeFlatVector<int64_t>(
            kRowsPerBatch, [batch](auto /*row*/) { return batch + 1; }),
    }));
  }
  createDuckDbTable(vectors);

  core::PlanNodeId finalAggId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config(QueryConfig::kMaxPartialAggregationMemory, 1)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .partialAggregation(
                      {"c0"}, {"sum(c1)", "count(c1)", "min(c1)", "max(c1)"})
                  .finalAggregation()
                  .capturePlanNodeId(finalAggId)
                  .planNode())
          .maxDrivers(1)
          .assertResults(
              "SELECT c0, sum(c1), count(c1), min(c1), max(c1) "
              "FROM tmp GROUP BY c0");

  const auto planStats = toPlanStats(task->taskStats());
  const auto& finalStats = planStats.at(finalAggId);
  const auto groupbyStats = finalStats.operatorStats.find("CudfGroupbyFINAL");
  ASSERT_NE(groupbyStats, finalStats.operatorStats.end());
  EXPECT_EQ(groupbyStats->second->outputRows, kUniqueKeys);
  EXPECT_GT(
      finalStats.customStats.at("cudfFinalAggregationHashBuckets").sum, 1);
  EXPECT_GE(
      finalStats.customStats.at("cudfFinalAggregationHashPartitionedRows").sum,
      4 * kRowsPerBatch);
  EXPECT_GT(
      finalStats.customStats.at("cudfFinalAggregationMaxBucketRows").max, 0);
  EXPECT_GT(
      finalStats.customStats.at("cudfFinalAggregationBucketOutputs").sum, 1);
  EXPECT_EQ(
      finalStats.customStats.at("cudfFinalAggregationCollectionTransitions")
          .sum,
      1);
  EXPECT_GE(
      finalStats.customStats.at("cudfFinalAggregationCollectionInputBatches")
          .sum,
      4);
  EXPECT_GT(
      finalStats.customStats.at("cudfFinalAggregationCollectionGrows").sum, 0);
  EXPECT_LE(
      finalStats.customStats.at("cudfFinalAggregationMaxCompactionInputRows")
          .max,
      kMaxBucketRows);
  EXPECT_LE(
      finalStats.customStats.at("cudfFinalAggregationMaxCompactionWorkBytes")
          .max,
      finalStats.customStats.at("cudfFinalAggregationBucketTargetBytes").max);
  EXPECT_EQ(
      finalStats.customStats.at("cudfFinalAggregationDirectInputRows").sum,
      4 * kRowsPerBatch);
  EXPECT_GT(
      finalStats.customStats.at("cudfFinalAggregationHostParkedRuns").sum, 0);
  EXPECT_GT(
      finalStats.customStats.at("cudfFinalAggregationCollectionCompactions")
          .sum,
      0);
  EXPECT_GE(
      finalStats.customStats
          .at("cudfFinalAggregationMaxCollectionCompactionLevel")
          .max,
      2)
      << "A four-batch carry must promote reduced output beyond level one";
  EXPECT_LE(
      finalStats.customStats
          .at("cudfFinalAggregationCollectionCompactionInputRows")
          .sum,
      4 * finalStats.customStats.at("cudfFinalAggregationDirectInputRows").sum)
      << "Leveled carries must not repeatedly fold every new batch into "
         "complete collection history";
  EXPECT_LE(
      finalStats.customStats
          .at("cudfFinalAggregationCollectionCompactionOutputRows")
          .sum,
      finalStats.customStats
          .at("cudfFinalAggregationCollectionCompactionInputRows")
          .sum);
  EXPECT_EQ(
      finalStats.customStats.at("cudfFinalAggregationHostParkedRows").sum,
      finalStats.customStats.at("cudfFinalAggregationHostRestoredRows").sum);
  EXPECT_EQ(
      finalStats.customStats.at("cudfFinalAggregationHostParkedBytes").sum,
      finalStats.customStats.at("cudfFinalAggregationHostRestoredBytes").sum);
  EXPECT_GE(
      finalStats.customStats.at("cudfFinalAggregationHostParkedRows").sum,
      finalStats.customStats.at("cudfFinalAggregationHashPartitionedRows").sum)
      << "Collection compaction parks promoted outputs in addition to hash "
         "partition outputs";
  EXPECT_GT(
      finalStats.customStats.at("cudfFinalAggregationMaxHostParkedBytes").max,
      0);
  EXPECT_EQ(
      finalStats.customStats.at("cudfFinalAggregationCurrentHostPhysicalBytes")
          .sum,
      0);
  EXPECT_GT(
      finalStats.customStats.at("cudfFinalAggregationMaxHostPhysicalBytes").max,
      0);
  EXPECT_EQ(
      finalStats.customStats
          .at("cudfFinalAggregationHostPhysicalAllocatedBytes")
          .sum,
      finalStats.customStats.at("cudfFinalAggregationHostPhysicalReleasedBytes")
          .sum);

  EXPECT_GT(groupbyStats->second->outputVectors, 1)
      << "The cursor must drain more than one hash-bucket output batch";
}

TEST_F(AggregationTest, skewedFinalAggregationCompactsParkedRuns) {
  auto& cudfConfig = cudf_velox::CudfConfig::getInstance();
  const auto savedMaxBatchRows = cudfConfig.batchSizeMaxThreshold;
  constexpr int32_t kMaxBucketRows = 4;
  cudfConfig.batchSizeMaxThreshold = kMaxBucketRows;
  SCOPE_EXIT {
    cudfConfig.batchSizeMaxThreshold = savedMaxBatchRows;
  };

  auto rawInput = makeRowVector({
      makeFlatVector<int32_t>({7}),
      makeFlatVector<int64_t>({1}),
  });
  auto planNode = PlanBuilder()
                      .values({rawInput})
                      .partialAggregation({"c0"}, {"sum(c1)", "count(c1)"})
                      .finalAggregation()
                      .planNode();
  auto finalAggregation =
      std::dynamic_pointer_cast<const core::AggregationNode>(planNode);
  ASSERT_NE(finalAggregation, nullptr);

  core::PlanFragment planFragment;
  planFragment.planNode = planNode;
  auto task = Task::create(
      "AggregationTest.skewedFinalAggregationCompactsParkedRuns",
      std::move(planFragment),
      0,
      core::QueryCtx::create(driverExecutor_.get()),
      Task::ExecutionMode::kParallel);
  auto driverCtx = std::make_unique<DriverCtx>(task, 0, 0, 0, 0);
  auto* driverCtxPtr = driverCtx.get();
  auto driver = Driver::testingCreate(std::move(driverCtx));
  cudf_velox::CudfGroupby groupby(0, driverCtxPtr, finalAggregation);
  groupby.initialize();

#ifndef NDEBUG
  uint64_t observedHostPrecharges = 0;
  SCOPED_TESTVALUE_SET(
      "facebook::velox::cudf_velox::CudfGroupby::"
      "parkFinalAggregationState::afterHostPrecharge",
      std::function<void(memory::MemoryPool*)>([&](auto* chargedPool) {
        EXPECT_EQ(chargedPool, groupby.pool());
        EXPECT_GT(chargedPool->usedBytes(), 0)
            << "Arrow host allocation must happen only after pool admission";
        ++observedHostPrecharges;
      }));
#endif

  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  auto makeState = [&](vector_size_t rows) {
    auto hostState = makeRowVector({
        makeFlatVector<int32_t>(rows, [](auto /*row*/) { return 7; }),
        makeFlatVector<int64_t>(rows, [](auto /*row*/) { return 1; }),
        makeFlatVector<int64_t>(rows, [](auto /*row*/) { return 1; }),
    });
    auto table =
        std::shared_ptr<cudf::table>(cudf_velox::with_arrow::toCudfTable(
            hostState,
            pool(),
            stream,
            cudf::get_current_device_resource_ref()));
    const auto tableView = table->view();
    return std::make_shared<cudf_velox::CudfVector>(
        pool(),
        finalAggregation->sources()[0]->outputType(),
        rows,
        tableView,
        cudf_velox::CudfVector::ViewOwner{std::move(table)},
        stream);
  };

  // The fifth one-row state starts collection and grows a maximally skewed
  // bucket. The wider state is split into four bounded runs, forcing recursive
  // repartition and two skew-compaction passes during drain.
  for (int32_t i = 0; i < 5; ++i) {
    groupby.addInput(makeState(1));
  }
  groupby.addInput(makeState(16));
  groupby.noMoreInput();

  vector_size_t outputRows = 0;
  int64_t outputSum = 0;
  int64_t outputCount = 0;
  for (int32_t attempts = 0; attempts < 1'024 && !groupby.isFinished();
       ++attempts) {
    auto output =
        std::dynamic_pointer_cast<cudf_velox::CudfVector>(groupby.getOutput());
    if (!output) {
      continue;
    }
    auto hostOutput = cudf_velox::with_arrow::toVeloxColumn(
        output->getTableView(),
        pool(),
        output->type(),
        output->stream(),
        cudf::get_current_device_resource_ref());
    output->stream().synchronize();
    outputRows += hostOutput->size();
    for (vector_size_t row = 0; row < hostOutput->size(); ++row) {
      EXPECT_EQ(
          hostOutput->childAt(0)->asFlatVector<int32_t>()->valueAt(row), 7);
      outputSum +=
          hostOutput->childAt(1)->asFlatVector<int64_t>()->valueAt(row);
      outputCount +=
          hostOutput->childAt(2)->asFlatVector<int64_t>()->valueAt(row);
    }
  }
  EXPECT_TRUE(groupby.isFinished());
  EXPECT_EQ(outputRows, 1);
  EXPECT_EQ(outputSum, 21);
  EXPECT_EQ(outputCount, 21);

  const auto opStats = groupby.stats(false);
  const auto statSum = [&](std::string_view name) {
    const auto it = opStats.runtimeStats.find(std::string(name));
    return it == opStats.runtimeStats.end() ? 0 : it->second.sum;
  };
  EXPECT_GT(statSum("cudfFinalAggregationCollectionGrows"), 0);
  EXPECT_GT(statSum("cudfFinalAggregationCollectionCompactions"), 0);
  EXPECT_GT(
      statSum("cudfFinalAggregationCollectionCompactionInputRows"),
      statSum("cudfFinalAggregationCollectionCompactionOutputRows"));
  EXPECT_GT(statSum("cudfFinalAggregationCollectionSealedRuns"), 0);
  EXPECT_GT(
      opStats.runtimeStats
          .at("cudfFinalAggregationMaxCollectionCompactionLevel")
          .max,
      0)
      << "Collection compaction must promote its output to a persistent "
         "generation";
  EXPECT_GT(statSum("cudfFinalAggregationHashRepartitions"), 0);
  EXPECT_GE(statSum("cudfFinalAggregationSkewCompactions"), 2);
  EXPECT_EQ(statSum("cudfFinalAggregationSkewGuards"), 0);
  EXPECT_LE(
      opStats.runtimeStats.at("cudfFinalAggregationMaxCompactionInputRows").max,
      kMaxBucketRows);
  EXPECT_LE(
      opStats.runtimeStats.at("cudfFinalAggregationMaxCompactionWorkBytes").max,
      opStats.runtimeStats.at("cudfFinalAggregationBucketTargetBytes").max);
  EXPECT_EQ(
      statSum("cudfFinalAggregationHostParkedRows"),
      statSum("cudfFinalAggregationHostRestoredRows"));
  EXPECT_EQ(
      statSum("cudfFinalAggregationHostParkedBytes"),
      statSum("cudfFinalAggregationHostRestoredBytes"));
  EXPECT_EQ(statSum("cudfFinalAggregationCurrentHostPhysicalBytes"), 0);
  EXPECT_GT(statSum("cudfFinalAggregationMaxHostPhysicalBytes"), 0);
  EXPECT_EQ(
      statSum("cudfFinalAggregationHostPhysicalAllocatedBytes"),
      statSum("cudfFinalAggregationHostPhysicalReleasedBytes"));
  const auto poolStats = groupby.pool()->stats();
  EXPECT_GT(poolStats.numExternalAllocs, 0);
  EXPECT_EQ(poolStats.numExternalAllocs, poolStats.numExternalFrees);
  EXPECT_EQ(
      poolStats.numExternalAllocs,
      statSum("cudfFinalAggregationHostParkedRuns"))
      << "Hash-bucket slices sharing one Arrow owner must share one external "
         "allocation charge";
#ifndef NDEBUG
  EXPECT_EQ(observedHostPrecharges, poolStats.numExternalAllocs);
#endif
  EXPECT_GT(poolStats.cumulativeExternalBytes, 0);
  EXPECT_EQ(
      poolStats.cumulativeExternalBytes,
      statSum("cudfFinalAggregationHostParkedBytes"));
  groupby.close();
}

TEST_F(AggregationTest, closeReleasesPendingFinalAggregationState) {
  auto rawInput = makeRowVector({
      makeFlatVector<int32_t>({0, 1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30, 40}),
  });
  auto planNode = PlanBuilder()
                      .values({rawInput})
                      .partialAggregation({"c0"}, {"sum(c1)"})
                      .finalAggregation()
                      .planNode();
  auto finalAggregation =
      std::dynamic_pointer_cast<const core::AggregationNode>(planNode);
  ASSERT_NE(finalAggregation, nullptr);

  core::PlanFragment planFragment;
  planFragment.planNode = planNode;
  auto task = Task::create(
      "AggregationTest.closePendingFinalState",
      std::move(planFragment),
      0,
      core::QueryCtx::create(driverExecutor_.get()),
      Task::ExecutionMode::kParallel);
  auto driverCtx = std::make_unique<DriverCtx>(task, 0, 0, 0, 0);
  auto* driverCtxPtr = driverCtx.get();
  auto driver = Driver::testingCreate(std::move(driverCtx));
  cudf_velox::CudfGroupby groupby(0, driverCtxPtr, finalAggregation);
  groupby.initialize();

  auto partialState = makeRowVector({
      makeFlatVector<int32_t>({0, 1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30, 40}),
  });
  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  auto ownedTable =
      std::shared_ptr<cudf::table>(cudf_velox::with_arrow::toCudfTable(
          partialState,
          pool(),
          stream,
          cudf::get_current_device_resource_ref()));
  std::weak_ptr<cudf::table> weakOwner = ownedTable;
  auto cudfState = std::make_shared<cudf_velox::CudfVector>(
      pool(),
      finalAggregation->sources()[0]->outputType(),
      partialState->size(),
      ownedTable->view(),
      cudf_velox::CudfVector::ViewOwner{ownedTable},
      stream);
  ownedTable.reset();

  groupby.addInput(std::move(cudfState));
  EXPECT_FALSE(weakOwner.expired());
  groupby.close();
  EXPECT_TRUE(weakOwner.expired())
      << "close() must release final states retained before noMoreInput()";
}

TEST_F(AggregationTest, finalCollectorConsumesOversizedInputOwners) {
  constexpr vector_size_t kRows = 1'024;
  auto partialState = makeRowVector({
      makeFlatVector<int32_t>(kRows, [](auto row) { return row; }),
      makeFlatVector<int64_t>(
          kRows, [](auto row) { return 10 * static_cast<int64_t>(row + 1); }),
  });
  auto planNode = PlanBuilder()
                      .values({partialState})
                      .partialAggregation({"c0"}, {"sum(c1)"})
                      .finalAggregation()
                      .planNode();
  auto finalAggregation =
      std::dynamic_pointer_cast<const core::AggregationNode>(planNode);
  ASSERT_NE(finalAggregation, nullptr);

  core::PlanFragment planFragment;
  planFragment.planNode = planNode;
  auto task = Task::create(
      "AggregationTest.finalCollectorConsumesOversizedInputOwners",
      std::move(planFragment),
      0,
      core::QueryCtx::create(driverExecutor_.get()),
      Task::ExecutionMode::kParallel);
  auto driverCtx = std::make_unique<DriverCtx>(task, 0, 0, 0, 0);
  auto* driverCtxPtr = driverCtx.get();
  auto driver = Driver::testingCreate(std::move(driverCtx));
  cudf_velox::CudfGroupby groupby(0, driverCtxPtr, finalAggregation);
  groupby.initialize();

  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  std::vector<std::weak_ptr<cudf::table>> weakOwners;
  auto makeState = [&](uint64_t logicalBytes) {
    auto owner =
        std::shared_ptr<cudf::table>(cudf_velox::with_arrow::toCudfTable(
            partialState,
            pool(),
            stream,
            cudf::get_current_device_resource_ref()));
    weakOwners.push_back(owner);
    const auto tableView = owner->view();
    return std::make_shared<cudf_velox::CudfVector>(
        pool(),
        finalAggregation->sources()[0]->outputType(),
        partialState->size(),
        tableView,
        cudf_velox::CudfVector::ViewOwner{std::move(owner)},
        stream,
        logicalBytes);
  };

  // One logical 2 GiB input is above the hard 1 GiB envelope on every GPU,
  // while each logical row is well below the 128 MiB minimum envelope.
  // It must enter the collector immediately, and admission must materialize
  // bounded bucket states instead of retaining a view of the source owner.
  groupby.addInput(makeState(2ULL << 30));
  stream.synchronize();
  ASSERT_EQ(weakOwners.size(), 1);
  EXPECT_TRUE(weakOwners[0].expired());
  // The second state is twice as wide as the state that initialized the
  // collector. Slicing and bucket admission must use its current width, not
  // the first state's average-width estimate.
  groupby.addInput(makeState(4ULL << 30));
  stream.synchronize();
  ASSERT_EQ(weakOwners.size(), 2);
  EXPECT_TRUE(weakOwners[1].expired());

  groupby.noMoreInput();
  vector_size_t outputRows = 0;
  for (int32_t attempts = 0; attempts < 1'024 && !groupby.isFinished();
       ++attempts) {
    if (auto output = groupby.getOutput()) {
      outputRows += output->size();
    }
  }
  EXPECT_TRUE(groupby.isFinished());
  EXPECT_EQ(outputRows, kRows);

  const auto opStats = groupby.stats(false);
  EXPECT_EQ(
      opStats.runtimeStats.at("cudfFinalAggregationCollectionTransitions").sum,
      1);
  EXPECT_EQ(
      opStats.runtimeStats.at("cudfFinalAggregationCollectionByteTransitions")
          .sum,
      1);
  EXPECT_EQ(
      opStats.runtimeStats.at("cudfFinalAggregationCollectionInputBatches").sum,
      2);
  EXPECT_LE(
      opStats.runtimeStats.at("cudfFinalAggregationMaxCompactionWorkBytes").max,
      opStats.runtimeStats.at("cudfFinalAggregationBucketTargetBytes").max);
  EXPECT_EQ(
      opStats.runtimeStats.at("cudfFinalAggregationDirectInputRows").sum,
      2 * kRows);
  EXPECT_EQ(
      opStats.runtimeStats.at("cudfFinalAggregationDirectInputBytes").sum,
      6ULL << 30);
  EXPECT_GT(
      opStats.runtimeStats.at("cudfFinalAggregationHostParkedRuns").sum, 0);
  EXPECT_EQ(
      opStats.runtimeStats.at("cudfFinalAggregationHostParkedRows").sum,
      opStats.runtimeStats.at("cudfFinalAggregationHostRestoredRows").sum);
  EXPECT_EQ(
      opStats.runtimeStats.at("cudfFinalAggregationHostParkedBytes").sum,
      opStats.runtimeStats.at("cudfFinalAggregationHostRestoredBytes").sum);
  groupby.close();
}

TEST_F(AggregationTest, finalAggregationStreamingMixedAggs) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  core::PlanNodeId finalAggId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config(QueryConfig::kMaxPartialAggregationMemory, 1)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .partialAggregation(
                      {"c0"},
                      {"sum(c2)", "count(0)", "min(c3)", "max(c5)", "avg(c4)"})
                  .finalAggregation()
                  .capturePlanNodeId(finalAggId)
                  .planNode())
          .assertResults(
              "SELECT c0, sum(c2), count(*), min(c3), max(c5), avg(c4) FROM tmp GROUP BY c0");

  const auto planStats = toPlanStats(task->taskStats());
  EXPECT_GT(planStats.at(finalAggId).outputRows, 0);
}

TEST_F(AggregationTest, finalAggregationStreamingMultiKey) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  core::PlanNodeId finalAggId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config(QueryConfig::kMaxPartialAggregationMemory, 1)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .partialAggregation(
                      {"c0", "c1", "c6"},
                      {"sum(c4)", "count(0)", "avg(c5)", "max(c3)"})
                  .finalAggregation()
                  .capturePlanNodeId(finalAggId)
                  .planNode())
          .assertResults(
              "SELECT c0, c1, c6, sum(c4), count(*), avg(c5), max(c3) FROM tmp GROUP BY c0, c1, c6");

  const auto planStats = toPlanStats(task->taskStats());
  EXPECT_GT(planStats.at(finalAggId).outputRows, 0);
}

class EmptyInputAggregationTest : public AggregationTest {
 protected:
  void SetUp() override {
    AggregationTest::SetUp();

    // Common test data setup
    data_ = makeRowVector({
        makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
        makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
        makeFlatVector<std::string>({"a", "b", "c", "d", "e"}),
    });

    createDuckDbTable({data_});
    filter_ = "c0 > 10"; // This filter eliminates all rows
  }

  void TearDown() override {
    // Need to clear data before plan destruction to keep memory pools happy
    data_.reset();
    plan_.reset();
    filter_.clear();
    AggregationTest::TearDown();
  }

  RowVectorPtr data_;
  core::PlanNodePtr plan_;
  std::string filter_;
};

TEST_F(EmptyInputAggregationTest, groupedSingleAggregation) {
  // Test case where CUDF aggregation operator receives no input rows for
  // grouped aggregation
  plan_ = PlanBuilder()
              .values({data_})
              .filter(filter_)
              .singleAggregation(
                  {"c2"}, {"sum(c0)", "count(c1)", "max(c1)", "avg(c1)"})
              .planNode();

  // should return empty result for grouped aggregation
  assertQuery(
      plan_,
      "SELECT c2, sum(c0), count(c1), max(c1), avg(c1) FROM tmp WHERE c0 > 10 GROUP BY c2");
}

TEST_F(EmptyInputAggregationTest, globalSingleAggregation) {
  // Test case where CUDF aggregation operator receives no input rows for global
  // aggregation
  plan_ =
      PlanBuilder()
          .values({data_})
          .filter(filter_)
          .singleAggregation({}, {"sum(c0)", "count(c1)", "max(c1)", "avg(c1)"})
          .planNode();

  // global aggregation should return one row with null/zero values
  assertQuery(
      plan_,
      "SELECT sum(c0), count(c1), max(c1), avg(c1) FROM tmp WHERE c0 > 10");
}

TEST_F(EmptyInputAggregationTest, distinctSingleAggregation) {
  // Test case where CUDF aggregation operator receives no input rows for
  // distinct aggregation
  plan_ = PlanBuilder()
              .values({data_})
              .filter(filter_)
              .singleAggregation({"c2"}, {})
              .planNode();

  // should return empty result for distinct aggregation
  assertQuery(plan_, "SELECT DISTINCT c2 FROM tmp WHERE c0 > 10");
}

TEST_F(EmptyInputAggregationTest, distinctPartialFinalAggregation) {
  // Test case where CUDF aggregation operator receives no input rows for
  // distinct partial-final aggregation
  plan_ = PlanBuilder()
              .values({data_})
              .filter(filter_)
              .partialAggregation({"c2"}, {})
              .finalAggregation()
              .planNode();

  // should return empty result for distinct aggregation
  assertQuery(plan_, "SELECT DISTINCT c2 FROM tmp WHERE c0 > 10");
}

TEST_F(EmptyInputAggregationTest, groupedPartialFinalAggregation) {
  // Test case where CUDF aggregation operator receives no input rows for
  // partial-final aggregation
  plan_ = PlanBuilder()
              .values({data_})
              .filter(filter_)
              .partialAggregation(
                  {"c2"}, {"sum(c0)", "count(c1)", "max(c1)", "avg(c1)"})
              .finalAggregation()
              .planNode();

  // should return empty result for partial-final aggregation
  assertQuery(
      plan_,
      "SELECT c2, sum(c0), count(c1), max(c1), avg(c1) FROM tmp WHERE c0 > 10 GROUP BY c2");
}

TEST_F(EmptyInputAggregationTest, globalPartialFinalAggregation) {
  // Test case where CUDF aggregation operator receives no input rows for global
  // partial-final aggregation
  plan_ = PlanBuilder()
              .values({data_})
              .filter(filter_)
              .partialAggregation(
                  {}, {"sum(c0)", "count(c1)", "max(c1)", "avg(c1)"})
              .finalAggregation()
              .planNode();

  // global partial-final aggregation should return 1 row with null/zero values
  assertQuery(
      plan_,
      "SELECT sum(c0), count(c1), max(c1), avg(c1) FROM tmp WHERE c0 > 10");
}

TEST_F(AggregationTest, singleAggregationStreamingSumMinMax) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  std::string keyName = "c0";
  std::vector<std::string> aggregates = {
      "sum(c1)",
      "sum(c2)",
      "sum(c4)",
      "sum(c5)",
      "min(c1)",
      "min(c2)",
      "min(c3)",
      "min(c4)",
      "min(c5)",
      "max(c1)",
      "max(c2)",
      "max(c3)",
      "max(c4)",
      "max(c5)"};

  auto op = PlanBuilder()
                .values(vectors)
                .singleAggregation({keyName}, aggregates)
                .planNode();

  assertQuery(
      op,
      "SELECT " + keyName +
          ", sum(c1), sum(c2), sum(c4), sum(c5)"
          ", min(c1), min(c2), min(c3), min(c4), min(c5)"
          ", max(c1), max(c2), max(c3), max(c4), max(c5)"
          " FROM tmp GROUP BY " +
          keyName);
}

TEST_F(AggregationTest, singleAggregationStreamingAvg) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  std::string keyName = "c0";
  std::vector<std::string> aggregates = {
      "avg(c1)", "avg(c2)", "avg(c4)", "avg(c5)"};

  auto op = PlanBuilder()
                .values(vectors)
                .singleAggregation({keyName}, aggregates)
                .planNode();

  assertQuery(
      op,
      "SELECT " + keyName + ", avg(c1), avg(c2), avg(c4), avg(c5) " +
          "FROM tmp GROUP BY " + keyName);
}

TEST_F(AggregationTest, singleAggregationStreamingCount) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  std::string keyName = "c0";
  auto op = PlanBuilder()
                .values(vectors)
                .singleAggregation({keyName}, {"count(0)"})
                .planNode();

  assertQuery(
      op, "SELECT " + keyName + ", count(*) FROM tmp GROUP BY " + keyName);
}

TEST_F(AggregationTest, singleAggregationStreamingMultiKey) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  std::vector<std::string> aggregates = {
      "sum(c4)",
      "sum(c5)",
      "min(c3)",
      "min(c4)",
      "min(c5)",
      "max(c3)",
      "max(c4)",
      "max(c5)"};

  auto op = PlanBuilder()
                .values(vectors)
                .singleAggregation({"c0", "c1", "c6"}, aggregates)
                .planNode();

  assertQuery(
      op,
      "SELECT c0, c1, c6, sum(c4), sum(c5), min(c3), min(c4), min(c5),"
      " max(c3), max(c4), max(c5) FROM tmp GROUP BY c0, c1, c6");
}

TEST_F(AggregationTest, singleAggregationStreamingMixedAggs) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  std::string keyName = "c0";
  std::vector<std::string> aggregates = {
      "sum(c2)", "count(0)", "min(c3)", "max(c5)", "avg(c4)"};

  auto op = PlanBuilder()
                .values(vectors)
                .singleAggregation({keyName}, aggregates)
                .planNode();

  assertQuery(
      op,
      "SELECT " + keyName +
          ", sum(c2), count(*), min(c3), max(c5), avg(c4)"
          " FROM tmp GROUP BY " +
          keyName);
}

TEST_F(AggregationTest, singleAggregationStreamingWithNulls) {
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>(
          {std::nullopt, 1, std::nullopt, 2, std::nullopt, 1, 2}),
      makeFlatVector<int32_t>({-1, 1, -2, 2, -3, 3, 4}),
  });

  createDuckDbTable({data});

  auto op = PlanBuilder()
                .values({data})
                .singleAggregation({"c0"}, {"sum(c1)", "min(c1)", "max(c1)"})
                .planNode();

  assertQuery(op, "SELECT c0, sum(c1), min(c1), max(c1) FROM tmp GROUP BY c0");
}

TEST_F(AggregationTest, singleAggregationStreamingIgnoreNullKeys) {
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>(
          {std::nullopt, 1, std::nullopt, 2, std::nullopt, 1, 2}),
      makeFlatVector<int32_t>({-1, 1, -2, 2, -3, 3, 4}),
  });

  auto op = PlanBuilder()
                .values({data})
                .aggregation(
                    {"c0"},
                    {"sum(c1)"},
                    {},
                    core::AggregationNode::Step::kSingle,
                    true)
                .planNode();

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<int64_t>({4, 6}),
  });
  AssertQueryBuilder(op).assertResults(expected);
}

TEST_F(AggregationTest, singleAggregationStreamingIgnoreNullKeysAcrossBatches) {
  auto batch1 = makeRowVector({
      makeNullableFlatVector<int32_t>({1, 2, 1}),
      makeFlatVector<int32_t>({10, 20, 30}),
  });
  auto batch2 = makeRowVector({
      makeNullableFlatVector<int32_t>(
          {std::nullopt, std::nullopt, std::nullopt}),
      makeFlatVector<int32_t>({7, 8, 9}),
  });
  std::vector<RowVectorPtr> vectors{batch1, batch2};

  createDuckDbTable(vectors);

  auto op = PlanBuilder()
                .values(vectors)
                .aggregation(
                    {"c0"},
                    {"sum(c1)", "count(0)"},
                    {},
                    core::AggregationNode::Step::kSingle,
                    true)
                .planNode();

  assertQuery(
      op,
      "SELECT c0, sum(c1), count(*) FROM tmp WHERE c0 IS NOT NULL GROUP BY c0");
}

TEST_F(AggregationTest, globalApproxDistinct) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 1, 2, 3, 4, 5}),
      makeFlatVector<int32_t>({10, 20, 30, 40, 50, 10, 20, 30, 40, 50}),
  });

  auto plan = PlanBuilder()
                  .values({data})
                  .partialAggregation(
                      {}, {"approx_distinct(c0)", "approx_distinct(c1)"})
                  .finalAggregation()
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto c0_estimate = result->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0);
  auto c1_estimate = result->childAt(1)->as<FlatVector<int64_t>>()->valueAt(0);

  EXPECT_GE(c0_estimate, 4);
  EXPECT_LE(c0_estimate, 6);

  EXPECT_GE(c1_estimate, 4);
  EXPECT_LE(c1_estimate, 6);
}

TEST_F(AggregationTest, globalApproxDistinctWithNulls) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({1, 2, std::nullopt, 3, 4, 5, 1, 2, 3}),
  });

  auto plan = PlanBuilder()
                  .values({data})
                  .partialAggregation({}, {"approx_distinct(c0)"})
                  .finalAggregation()
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto estimate = result->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0);

  EXPECT_GE(estimate, 4);
  EXPECT_LE(estimate, 6);
}

TEST_F(AggregationTest, globalApproxDistinctHighCardinality) {
  std::vector<int64_t> values;
  for (int64_t i = 0; i < 10000; ++i) {
    values.push_back(i);
  }

  auto data = makeRowVector({
      makeFlatVector<int64_t>(values),
  });

  auto plan = PlanBuilder()
                  .values({data})
                  .partialAggregation({}, {"approx_distinct(c0)"})
                  .finalAggregation()
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto estimate = result->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0);

  double error_rate =
      std::abs(static_cast<double>(estimate) - 10000.0) / 10000.0;
  EXPECT_LT(error_rate, 0.05);
}

TEST_F(AggregationTest, globalApproxDistinctEmpty) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>(std::vector<int64_t>{}),
  });

  auto plan = PlanBuilder()
                  .values({data})
                  .partialAggregation({}, {"approx_distinct(c0)"})
                  .finalAggregation()
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto estimate = result->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0);

  EXPECT_EQ(estimate, 0);
}

TEST_F(AggregationTest, globalApproxDistinctPartialIntermediateFinal) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 1, 2, 3, 4, 5}),
      makeFlatVector<int32_t>({10, 20, 30, 40, 50, 10, 20, 30, 40, 50}),
  });

  auto plan = PlanBuilder()
                  .values({data})
                  .partialAggregation(
                      {}, {"approx_distinct(c0)", "approx_distinct(c1)"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto c0_estimate = result->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0);
  auto c1_estimate = result->childAt(1)->as<FlatVector<int64_t>>()->valueAt(0);

  EXPECT_GE(c0_estimate, 4);
  EXPECT_LE(c0_estimate, 6);

  EXPECT_GE(c1_estimate, 4);
  EXPECT_LE(c1_estimate, 6);
}

TEST_F(AggregationTest, globalApproxDistinctWithNaN) {
  auto data = makeRowVector({
      makeFlatVector<double>({1.0, 2.0, std::nan(""), 4.0, std::nan(""), 1.0}),
  });

  auto planCudf = PlanBuilder()
                      .values({data})
                      .partialAggregation({}, {"approx_distinct(c0)"})
                      .finalAggregation()
                      .planNode();

  auto cudfResult = AssertQueryBuilder(planCudf).copyResults(pool());
  ASSERT_EQ(cudfResult->size(), 1);
  auto cudfEstimate =
      cudfResult->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0);

  cudf_velox::unregisterCudf();
  auto planVelox = PlanBuilder()
                       .values({data})
                       .partialAggregation({}, {"approx_distinct(c0)"})
                       .finalAggregation()
                       .planNode();

  auto veloxResult = AssertQueryBuilder(planVelox).copyResults(pool());
  ASSERT_EQ(veloxResult->size(), 1);
  auto veloxEstimate =
      veloxResult->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0);
  cudf_velox::registerCudf();

  EXPECT_EQ(cudfEstimate, veloxEstimate)
      << "CUDF and Velox should produce the same result for NaN values. "
      << "Expected distinct count: 3 (1.0, 2.0, 4.0) plus NaN as distinct. "
      << "CUDF result: " << cudfEstimate << ", Velox result: " << veloxEstimate;

  EXPECT_GE(cudfEstimate, 3);
  EXPECT_LE(cudfEstimate, 5);
}

// Test stddev_samp with kSingle step and grouped aggregation
TEST_F(AggregationTest, stddevSampSingleGrouped) {
  // Hand-crafted data with known expected results
  // Group 0: [1, 2, 3] -> stddev_samp = 1.0
  // Group 1: [4, 6] -> stddev_samp = sqrt(2) ≈ 1.414
  // Group 2: [10, 20, 30, 40] -> stddev_samp = sqrt(500/3) ≈ 12.909
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 0, 1, 1, 2, 2, 2, 2}), // c0 - key
      makeFlatVector<int64_t>({1, 2, 3, 4, 6, 10, 20, 30, 40}), // c1 - bigint
      makeFlatVector<double>(
          {1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 30.0, 40.0}), // c2 - double
  });
  createDuckDbTable({data});

  // Test with bigint input
  auto op = PlanBuilder()
                .values({data})
                .singleAggregation({"c0"}, {"stddev_samp(c1)"})
                .planNode();

  assertQuery(op, "SELECT c0, stddev_samp(c1) FROM tmp GROUP BY c0");

  // Test with double input
  auto op2 = PlanBuilder()
                 .values({data})
                 .singleAggregation({"c0"}, {"stddev_samp(c2)"})
                 .planNode();

  assertQuery(op2, "SELECT c0, stddev_samp(c2) FROM tmp GROUP BY c0");
}

// Test stddev_samp with kPartial + kFinal (two-stage distributed)
TEST_F(AggregationTest, stddevSampPartialFinalGrouped) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 0, 1, 1, 2, 2, 2, 2}),
      makeFlatVector<int64_t>({1, 2, 3, 4, 6, 10, 20, 30, 40}),
      makeFlatVector<double>({1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 30.0, 40.0}),
  });
  createDuckDbTable({data});

  // Test with bigint input
  auto op = PlanBuilder()
                .values({data})
                .partialAggregation({"c0"}, {"stddev_samp(c1)"})
                .finalAggregation()
                .planNode();

  assertQuery(op, "SELECT c0, stddev_samp(c1) FROM tmp GROUP BY c0");

  // Test with double input
  auto op2 = PlanBuilder()
                 .values({data})
                 .partialAggregation({"c0"}, {"stddev_samp(c2)"})
                 .finalAggregation()
                 .planNode();

  assertQuery(op2, "SELECT c0, stddev_samp(c2) FROM tmp GROUP BY c0");
}

// Test stddev_samp with kPartial + kIntermediate + kFinal (three-stage)
TEST_F(AggregationTest, stddevSampPartialIntermediateFinalGrouped) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 0, 1, 1, 2, 2, 2, 2}),
      makeFlatVector<int64_t>({1, 2, 3, 4, 6, 10, 20, 30, 40}),
      makeFlatVector<double>({1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 30.0, 40.0}),
  });
  createDuckDbTable({data});

  // Test with bigint input
  auto op = PlanBuilder()
                .values({data})
                .partialAggregation({"c0"}, {"stddev_samp(c1)"})
                .intermediateAggregation()
                .finalAggregation()
                .planNode();

  assertQuery(op, "SELECT c0, stddev_samp(c1) FROM tmp GROUP BY c0");

  // Test with double input
  auto op2 = PlanBuilder()
                 .values({data})
                 .partialAggregation({"c0"}, {"stddev_samp(c2)"})
                 .intermediateAggregation()
                 .finalAggregation()
                 .planNode();

  assertQuery(op2, "SELECT c0, stddev_samp(c2) FROM tmp GROUP BY c0");
}

// Test stddev_samp with NULL values in input
TEST_F(AggregationTest, stddevSampWithNulls) {
  // Group 0: [1, NULL, 3] -> should compute stddev of [1, 3] = sqrt(2) ≈ 1.414
  // Group 1: [4, 6, NULL] -> should compute stddev of [4, 6] = sqrt(2) ≈ 1.414
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 0, 1, 1, 1}),
      makeNullableFlatVector<int64_t>({1, std::nullopt, 3, 4, 6, std::nullopt}),
      makeNullableFlatVector<double>(
          {1.0, std::nullopt, 3.0, 4.0, 6.0, std::nullopt}),
  });
  createDuckDbTable({data});

  auto op = PlanBuilder()
                .values({data})
                .singleAggregation({"c0"}, {"stddev_samp(c1)"})
                .planNode();

  assertQuery(op, "SELECT c0, stddev_samp(c1) FROM tmp GROUP BY c0");

  auto op2 = PlanBuilder()
                 .values({data})
                 .singleAggregation({"c0"}, {"stddev_samp(c2)"})
                 .planNode();

  assertQuery(op2, "SELECT c0, stddev_samp(c2) FROM tmp GROUP BY c0");
}

// Test stddev_samp with single value per group (should return NULL)
TEST_F(AggregationTest, stddevSampSingleValueGroup) {
  // Each group has only one value -> stddev_samp should return NULL
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 1, 2}),
      makeFlatVector<int64_t>({10, 20, 30}),
      makeFlatVector<double>({10.0, 20.0, 30.0}),
  });
  createDuckDbTable({data});

  auto op = PlanBuilder()
                .values({data})
                .singleAggregation({"c0"}, {"stddev_samp(c1)"})
                .planNode();

  assertQuery(op, "SELECT c0, stddev_samp(c1) FROM tmp GROUP BY c0");

  auto op2 = PlanBuilder()
                 .values({data})
                 .singleAggregation({"c0"}, {"stddev_samp(c2)"})
                 .planNode();

  assertQuery(op2, "SELECT c0, stddev_samp(c2) FROM tmp GROUP BY c0");
}

// Test stddev_samp with all NULL input (should return NULL)
TEST_F(AggregationTest, stddevSampAllNulls) {
  // Group 0: all NULLs -> stddev_samp should return NULL
  // Group 1: has values -> should compute normally
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 1, 1}),
      makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt, 1, 2}),
      makeNullableFlatVector<double>({std::nullopt, std::nullopt, 1.0, 2.0}),
  });
  createDuckDbTable({data});

  auto op = PlanBuilder()
                .values({data})
                .singleAggregation({"c0"}, {"stddev_samp(c1)"})
                .planNode();

  assertQuery(op, "SELECT c0, stddev_samp(c1) FROM tmp GROUP BY c0");

  auto op2 = PlanBuilder()
                 .values({data})
                 .singleAggregation({"c0"}, {"stddev_samp(c2)"})
                 .planNode();

  assertQuery(op2, "SELECT c0, stddev_samp(c2) FROM tmp GROUP BY c0");
}

// Test avg with all NULL input (should return NULL, not NaN)
TEST_F(AggregationTest, avgAllNulls) {
  // Group 0: all NULLs -> avg should return NULL
  // Group 1: has values -> should compute normally
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 1, 1}),
      makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt, 4, 6}),
      makeNullableFlatVector<double>({std::nullopt, std::nullopt, 4.0, 6.0}),
  });
  createDuckDbTable({data});

  auto op = PlanBuilder()
                .values({data})
                .singleAggregation({"c0"}, {"avg(c1)"})
                .planNode();

  assertQuery(op, "SELECT c0, avg(c1) FROM tmp GROUP BY c0");

  auto op2 = PlanBuilder()
                 .values({data})
                 .singleAggregation({"c0"}, {"avg(c2)"})
                 .planNode();

  assertQuery(op2, "SELECT c0, avg(c2) FROM tmp GROUP BY c0");
}

// Test avg with all NULL input using partial + final (distributed) aggregation
TEST_F(AggregationTest, avgAllNullsPartialFinal) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 1, 1}),
      makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt, 4, 6}),
      makeNullableFlatVector<double>({std::nullopt, std::nullopt, 4.0, 6.0}),
  });
  createDuckDbTable({data});

  auto op = PlanBuilder()
                .values({data})
                .partialAggregation({"c0"}, {"avg(c1)"})
                .finalAggregation()
                .planNode();

  assertQuery(op, "SELECT c0, avg(c1) FROM tmp GROUP BY c0");

  auto op2 = PlanBuilder()
                 .values({data})
                 .partialAggregation({"c0"}, {"avg(c2)"})
                 .finalAggregation()
                 .planNode();

  assertQuery(op2, "SELECT c0, avg(c2) FROM tmp GROUP BY c0");
}

TEST_F(AggregationTest, partialSumReuseWithAvg) {
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({
          makeFlatVector<int64_t>({0, 0, 1, 2, 3}),
          makeNullableFlatVector<double>(
              {1.0, std::nullopt, std::nullopt, 5.0, std::nullopt}),
          makeNullableFlatVector<double>(
              {std::nullopt, 2.0, 10.0, std::nullopt, std::nullopt}),
      }),
      makeRowVector({
          makeFlatVector<int64_t>({0, 1, 1, 2, 3}),
          makeNullableFlatVector<double>(
              {3.0, std::nullopt, std::nullopt, 7.0, std::nullopt}),
          makeNullableFlatVector<double>(
              {4.0, 14.0, std::nullopt, std::nullopt, std::nullopt}),
      }),
  };
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .partialAggregation(
                      {"c0"}, {"sum(c1)", "avg(c1)", "avg(c2)", "sum(c2)"})
                  .finalAggregation()
                  .planNode();

  assertQuery(
      plan,
      "SELECT c0, sum(c1), avg(c1), avg(c2), sum(c2) "
      "FROM tmp GROUP BY c0");
}

// Test avg with NaN inputs preserves NaN (does not convert to NULL)
TEST_F(AggregationTest, avgNaNInputs) {
  // Group 0: NaN only -> avg should be NaN
  // Group 1: normal values -> avg should compute normally
  // Group 2: all NULLs (count == 0) -> avg should be NULL
  // Group 3: NaN + NULL + normal -> avg should be NaN
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 1, 1, 2, 2, 3, 3, 3}),
      makeNullableFlatVector<double>(
          {std::nan(""),
           1.0,
           3.0,
           5.0,
           std::nullopt,
           std::nullopt,
           std::nan(""),
           std::nullopt,
           7.0}),
  });
  createDuckDbTable({data});

  auto op = PlanBuilder()
                .values({data})
                .singleAggregation({"c0"}, {"avg(c1)"})
                .planNode();

  assertQuery(op, "SELECT c0, avg(c1) FROM tmp GROUP BY c0");
}

// Test that zero-column rows flow correctly through CudfFromVelox.
// project({}) produces zero-column output; localPartitionRoundRobin is a CPU
// operator that forces CudfFromVelox insertion before the GPU aggregation.
// Without the zero-column fix in CudfFromVelox, this crashes with:
//   "Operator::getOutput() must return nullptr or a non-empty vector"
// because toCudfTable loses the row count for zero-column tables.
TEST_F(AggregationTest, zeroColumnThroughCudfFromVelox) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
  });
  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .values({data})
                  .filter("c0 > 0")
                  .project({})
                  .localPartitionRoundRobin()
                  .singleAggregation({}, {"count(*)"})
                  .planNode();

  AssertQueryBuilder(duckDbQueryRunner_)
      .config(core::QueryConfig::kMaxLocalExchangePartitionCount, "2")
      .plan(plan)
      .assertResults("SELECT count(*) FROM tmp WHERE c0 > 0");
}

} // namespace facebook::velox::exec::test
