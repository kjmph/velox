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

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

#include <gtest/gtest.h>

namespace facebook::velox::cudf_velox::test {

TEST(ConfigTest, batchConcatThresholdDefaults) {
  CudfConfig config;
  EXPECT_EQ(config.batchSizeMinThreshold, 100'000);
  EXPECT_FALSE(config.batchSizeMinBytes);
}

TEST(ConfigTest, Defaults) {
  CudfConfig config;
  ASSERT_EQ(config.distinctHashJoinEnabled, true);
  ASSERT_EQ(config.probeUniqueInnerJoinEnabled, false);
}

TEST(ConfigTest, DistinctHashJoinCanBeDisabled) {
  std::unordered_map<std::string, std::string> options = {
      {CudfConfig::kCudfDistinctHashJoinEnabled, "false"}};

  CudfConfig config;
  config.initialize(std::move(options));
  ASSERT_EQ(config.distinctHashJoinEnabled, false);
}

TEST(ConfigTest, CudfConfig) {
  std::unordered_map<std::string, std::string> options = {
      {CudfConfig::kCudfEnabled, "false"},
      {CudfConfig::kCudfDebugEnabled, "true"},
      {CudfConfig::kCudfMemoryResource, "arena"},
      {CudfConfig::kCudfMemoryPercent, "25"},
      {CudfConfig::kCudfFunctionNamePrefix, "presto"},
      {CudfConfig::kCudfStreamingGroupbyEnabled, "true"},
      {CudfConfig::kCudfStreamingGroupbyCapacityMultiplier, "3.5"},
      {CudfConfig::kCudfAllowCpuFallback, "false"},
      {CudfConfig::kCudfBatchSizeMinThreshold, "123456"},
      {CudfConfig::kCudfBatchSizeMinBytes, "2147483648"},
      {CudfConfig::kCudfFinalAggregationBatchSizeMinThreshold, "150000000"},
      {CudfConfig::kCudfDistinctHashJoinEnabled, "true"},
      {CudfConfig::kCudfProbeUniqueInnerJoinEnabled, "true"},
      {CudfConfig::kCudfExchange, "true"},
      {CudfConfig::kCudfExchangeServerPort, "12345"},
      {CudfConfig::kCudfIntraNodeExchange, "false"},
      {CudfConfig::kCudfPartitionedOutputBatchRows, "123456"},
      {CudfConfig::kCudfExchangeLogLevel, "7"},
      {CudfConfig::kCudfUcxxBlockingPolling, "true"},
      {CudfConfig::kCudfUcxxErrorHandling, "false"}};

  CudfConfig config;
  ASSERT_FALSE(config.streamingGroupbyEnabled);
  ASSERT_EQ(config.streamingGroupbyCapacityMultiplier, 2.0);
  config.initialize(std::move(options));
  ASSERT_EQ(config.enabled, false);
  ASSERT_EQ(config.debugEnabled, true);
  ASSERT_EQ(config.memoryResource, "arena");
  ASSERT_EQ(config.memoryPercent, 25);
  ASSERT_EQ(config.functionNamePrefix, "presto");
  ASSERT_EQ(config.streamingGroupbyEnabled, true);
  ASSERT_EQ(config.streamingGroupbyCapacityMultiplier, 3.5);
  ASSERT_EQ(config.allowCpuFallback, false);
  ASSERT_EQ(config.batchSizeMinThreshold, 123'456);
  ASSERT_EQ(config.batchSizeMinBytes.value(), 2'147'483'648);
  ASSERT_TRUE(config.finalAggregationBatchSizeMinThreshold.has_value());
  ASSERT_EQ(config.finalAggregationBatchSizeMinThreshold.value(), 150000000);
  ASSERT_EQ(config.distinctHashJoinEnabled, true);
  ASSERT_EQ(config.probeUniqueInnerJoinEnabled, true);
  ASSERT_EQ(config.exchange, true);
  ASSERT_EQ(config.exchangeServerPort, 12345);
  ASSERT_EQ(config.intraNodeExchange, false);
  ASSERT_EQ(config.partitionedOutputBatchRows, 123456);
  ASSERT_EQ(config.exchangeLogLevel, 7);
  ASSERT_EQ(config.ucxxBlockingPolling, true);
  ASSERT_EQ(config.ucxxErrorHandling, false);
}

TEST(ConfigTest, rejectsNonPositiveBatchConcatTargets) {
  auto initialize = [](const char* key, const char* value) {
    CudfConfig config;
    config.initialize({{key, value}});
  };

  VELOX_ASSERT_USER_THROW(
      initialize(CudfConfig::kCudfBatchSizeMinThreshold, "0"),
      "cuDF BatchConcat minimum row target must be positive");
  VELOX_ASSERT_USER_THROW(
      initialize(CudfConfig::kCudfBatchSizeMinThreshold, "-5"),
      "cuDF BatchConcat minimum row target must be positive");
  VELOX_ASSERT_USER_THROW(
      initialize(CudfConfig::kCudfBatchSizeMinBytes, "0"),
      "cuDF BatchConcat minimum byte target must be a positive integer: 0");
  VELOX_ASSERT_USER_THROW(
      initialize(CudfConfig::kCudfBatchSizeMinBytes, "-1"),
      "cuDF BatchConcat minimum byte target must be a positive integer: -1");
}

} // namespace facebook::velox::cudf_velox::test
