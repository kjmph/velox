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

#include <cudf/types.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace facebook::velox::cudf_velox {

struct CudfConfig {
  /// Keys used by the initialize() method.
  static constexpr const char* kCudfEnabled{"cudf.enabled"};
  static constexpr const char* kCudfDebugEnabled{"cudf.debug_enabled"};
  static constexpr const char* kCudfMemoryResource{"cudf.memory_resource"};
  static constexpr const char* kCudfMemoryPercent{"cudf.memory_percent"};
  static constexpr const char* kCudfFunctionNamePrefix{
      "cudf.function_name_prefix"};
  static constexpr const char* kCudfAstExpressionEnabled{
      "cudf.ast_expression_enabled"};
  static constexpr const char* kCudfAstExpressionPriority{
      "cudf.ast_expression_priority"};
  static constexpr const char* kCudfJitExpressionEnabled{
      "cudf.jit_expression_enabled"};
  static constexpr const char* kCudfJitExpressionPriority{
      "cudf.jit_expression_priority"};
  static constexpr const char* kCudfOutputMr{"cudf.output_mr"};
  static constexpr const char* kCudfHostToDeviceStagingEnabled{
      "cudf.host_to_device_staging_enabled"};
  static constexpr const char* kCudfHostToDeviceStagingWindowBytes{
      "cudf.host_to_device_staging_window_bytes"};
  static constexpr const char* kCudfHostToDeviceStagingPackThreads{
      "cudf.host_to_device_staging_pack_threads"};
  static constexpr const char* kCudfHostToDeviceStagingWindowSets{
      "cudf.host_to_device_staging_window_sets"};
  static constexpr const char* kCudfAllowCpuFallback{"cudf.allow_cpu_fallback"};
  static constexpr const char* kCudfLogFallback{"cudf.log_fallback"};
  static constexpr const char* kCudfBatchSizeMinThreshold{
      "cudf.batch_size_min_threshold"};
  static constexpr const char* kCudfBatchSizeMaxThreshold{
      "cudf.batch_size_max_threshold"};
  static constexpr const char* kCudfFinalAggregationBatchSizeMinThreshold{
      "cudf.final_aggregation_batch_size_min_threshold"};
  static constexpr const char* kCudfConcatOptimizationEnabled{
      "cudf.concat_optimization_enabled"};
  // TODO: Remove this switch during the production config/default pass.
  static constexpr const char* kCudfDistinctHashJoinEnabled{
      "cudf.distinct_hash_join_enabled"};
  static constexpr const char* kCudfProbeUniqueInnerJoinEnabled{
      "cudf.probe_unique_inner_join_enabled"};
  static constexpr const char* kCudfExchange{"cudf.exchange"};
  static constexpr const char* kCudfExchangeServerPort{
      "cudf.exchange.server.port"};
  static constexpr const char* kCudfIntraNodeExchange{
      "cudf.intra_node_exchange"};
  static constexpr const char* kCudfPartitionedOutputBatchRows{
      "cudf.partitioned_output_batch_rows"};
  static constexpr const char* kCudfPartitionedOutputMaxBatchRows{
      "cudf.partitioned_output_max_batch_rows"};
  static constexpr const char* kCudfExchangeLogLevel{"cudf.exchange_log_level"};
  static constexpr const char* kCudfUcxxBlockingPolling{
      "cudf.ucxx_blocking_polling"};
  static constexpr const char* kCudfUcxxErrorHandling{
      "cudf.ucxx_error_handling"};
  static constexpr const char* kCudfTimestampUnit{"cudf.timestamp_unit"};
  /// Query session configs for the cuDF Operators.
  static constexpr const char* kCudfTopNBatchSize{"cudf.topk_batch_size"};

  /// Singleton CudfConfig instance.
  /// Clients must set the configs below before invoking registerCudf().
  static CudfConfig& getInstance();

  /// Initialize from a map with the above keys.
  void initialize(std::unordered_map<std::string, std::string>&&);

  /// Enable cudf by default.
  /// Clients can disable here and enable it via the QueryConfig as well.
  bool enabled{true};

  /// Enable debug printing.
  bool debugEnabled{false};

  /// Allow fallback to CPU operators if GPU operator replacement fails.
  bool allowCpuFallback{true};

  /// Memory resource for cuDF.
  /// Possible values are (cuda, pool, async, arena, managed, managed_pool).
  std::string memoryResource{"async"};

  /// The initial percent of GPU memory to allocate for pool or arena memory
  /// resources.
  int32_t memoryPercent{50};

  /// Memory resource for output vectors. When set to a value different from
  /// memoryResource, a separate MR is created for output allocations.
  /// When empty, the main memoryResource is used.
  std::string outputMemoryResource;

  /// Whether pageable or fragmented host input should use bounded pinned
  /// staging before asynchronous host-to-device copies.
  bool hostToDeviceStagingEnabled{true};

  /// Bytes in each process-global pinned staging window.
  uint64_t hostToDeviceStagingWindowBytes{128ULL << 20};

  /// Persistent host copy threads per staging window set. Total persistent
  /// pack threads are this value multiplied by hostToDeviceStagingWindowSets.
  uint32_t hostToDeviceStagingPackThreads{4};

  /// Concurrent two-window sets in the process-global staging arena.
  uint32_t hostToDeviceStagingWindowSets{2};

  /// Register all the functions with the functionNamePrefix.
  std::string functionNamePrefix;

  /// Enable AST in expression evaluation.
  bool astExpressionEnabled{true};

  /// Enable JIT in expression evaluation.
  bool jitExpressionEnabled{true};

  /// Priority of AST expression. Expression with higher priority is chosen for
  /// a given root expression.
  /// Example:
  /// Priority of expression that uses individual cuDF functions is 50.
  /// If AST priority is 100 then for a velox expression node that is supported
  /// by both, AST will be chosen as replacement for cudf execution, if AST
  /// priority is 25 then standalone cudf function is chosen.
  int astExpressionPriority{100};

  /// Priority of JIT expression.
  int jitExpressionPriority{101};

  /// Whether to log a reason for falling back to Velox CPU execution.
  bool logFallback{true};

  /// Whether to insert CudfBatchConcat operators before supported Cudf
  /// operators.
  /// This can improve performance by reducing the number of cuda kernel
  /// launches on addInput of certain operators by collecting a minimum number
  /// of rows before concatenating and passing on to the next operator.
  /// This batch size is determined by batchSizeMinThreshold and
  /// batchSizeMaxThreshold
  bool concatOptimizationEnabled{false};

  /// Whether joins with trusted unique build keys should use
  /// cudf::distinct_hash_join instead of cudf::hash_join.
  bool distinctHashJoinEnabled{true};

  /// Experimental: for inner joins with trusted unique, non-null probe keys,
  /// build a temporary distinct hash table on the probe batch and probe it with
  /// the build side.
  bool probeUniqueInnerJoinEnabled{false};

  /// Minimum rows to accumulate before GPU-side concatenation in
  /// `CudfBatchConcat` (default 100k).
  int32_t batchSizeMinThreshold{100000};

  /// Optional CudfBatchConcat target used only before final aggregations.
  std::optional<int32_t> finalAggregationBatchSizeMinThreshold;

  /// Maximum rows allowed in a concatenated batch (user configurable).
  /// When not set, cuDF's own `size_type::max()` is used.
  std::optional<int32_t> batchSizeMaxThreshold;

  /// Whether remote exchanges should use the cuDF UCX transport.
  bool exchange{false};

  /// Port used by the cuDF UCX communicator.
  int32_t exchangeServerPort{30050};

  /// Whether same-process UCX exchange should bypass UCX data transfer.
  bool intraNodeExchange{true};

  /// Minimum rows to accumulate before flushing cuDF partitioned output.
  int64_t partitionedOutputBatchRows{10'000};

  /// Maximum rows in one cuDF partitioned-output batch. Non-positive values
  /// preserve the legacy unbounded behavior.
  int64_t partitionedOutputMaxBatchRows{0};

  /// Verbosity for ucx-exchange VLOG modules.
  int32_t exchangeLogLevel{0};

  /// Whether UCXX should use blocking progress mode.
  bool ucxxBlockingPolling{false};

  /// Whether UCXX endpoints should use error handling callbacks.
  bool ucxxErrorHandling{true};

  // Query config key for the TopN batch size in the cuDF TopN operator.
  int32_t topNBatchSize{5};

  /// Timestamp unit for cuDF timestamp types.
  /// Can be configured via kCudfTimestampUnit with string values:
  /// "s" (seconds), "ms" (milliseconds), "us" (microseconds), "ns"
  /// (nanoseconds).
  cudf::type_id timestampUnit = cudf::type_id::TIMESTAMP_NANOSECONDS;
};

} // namespace facebook::velox::cudf_velox
