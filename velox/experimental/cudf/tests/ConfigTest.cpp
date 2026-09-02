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

#include <gtest/gtest.h>

namespace facebook::velox::cudf_velox::test {

TEST(ConfigTest, Defaults) {
  CudfConfig config;
  ASSERT_EQ(config.distinctHashJoinEnabled, true);
  ASSERT_EQ(config.probeUniqueInnerJoinEnabled, false);
  ASSERT_TRUE(config.hostToDeviceStagingEnabled);
  ASSERT_EQ(config.hostToDeviceStagingWindowBytes, 128ULL << 20);
  ASSERT_EQ(config.hostToDeviceStagingPackThreads, 4);
  ASSERT_EQ(config.hostToDeviceStagingWindowSets, 2);
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
      {CudfConfig::kCudfHostToDeviceStagingEnabled, "false"},
      {CudfConfig::kCudfHostToDeviceStagingWindowBytes, "67108864"},
      {CudfConfig::kCudfHostToDeviceStagingPackThreads, "3"},
      {CudfConfig::kCudfHostToDeviceStagingWindowSets, "5"},
      {CudfConfig::kCudfFunctionNamePrefix, "presto"},
      {CudfConfig::kCudfAllowCpuFallback, "false"},
      {CudfConfig::kCudfBatchSizeMinThreshold, "50000000"},
      {CudfConfig::kCudfFinalAggregationBatchSizeMinThreshold, "150000000"},
      {CudfConfig::kCudfDistinctHashJoinEnabled, "true"},
      {CudfConfig::kCudfProbeUniqueInnerJoinEnabled, "true"},
      {CudfConfig::kCudfExchange, "true"},
      {CudfConfig::kCudfExchangeServerPort, "12345"},
      {CudfConfig::kCudfIntraNodeExchange, "false"},
      {CudfConfig::kCudfPartitionedOutputBatchRows, "123456"},
      {CudfConfig::kCudfPartitionedOutputMaxBatchRows, "654321"},
      {CudfConfig::kCudfExchangeLogLevel, "7"},
      {CudfConfig::kCudfUcxxBlockingPolling, "true"},
      {CudfConfig::kCudfUcxxErrorHandling, "false"}};

  CudfConfig config;
  config.initialize(std::move(options));
  ASSERT_EQ(config.enabled, false);
  ASSERT_EQ(config.debugEnabled, true);
  ASSERT_EQ(config.memoryResource, "arena");
  ASSERT_EQ(config.memoryPercent, 25);
  ASSERT_FALSE(config.hostToDeviceStagingEnabled);
  ASSERT_EQ(config.hostToDeviceStagingWindowBytes, 67'108'864);
  ASSERT_EQ(config.hostToDeviceStagingPackThreads, 3);
  ASSERT_EQ(config.hostToDeviceStagingWindowSets, 5);
  ASSERT_EQ(config.functionNamePrefix, "presto");
  ASSERT_EQ(config.allowCpuFallback, false);
  ASSERT_EQ(config.batchSizeMinThreshold, 50000000);
  ASSERT_TRUE(config.finalAggregationBatchSizeMinThreshold.has_value());
  ASSERT_EQ(config.finalAggregationBatchSizeMinThreshold.value(), 150000000);
  ASSERT_EQ(config.distinctHashJoinEnabled, true);
  ASSERT_EQ(config.probeUniqueInnerJoinEnabled, true);
  ASSERT_EQ(config.exchange, true);
  ASSERT_EQ(config.exchangeServerPort, 12345);
  ASSERT_EQ(config.intraNodeExchange, false);
  ASSERT_EQ(config.partitionedOutputBatchRows, 123456);
  ASSERT_EQ(config.partitionedOutputMaxBatchRows, 654321);
  ASSERT_EQ(config.exchangeLogLevel, 7);
  ASSERT_EQ(config.ucxxBlockingPolling, true);
  ASSERT_EQ(config.ucxxErrorHandling, false);
}
} // namespace facebook::velox::cudf_velox::test
