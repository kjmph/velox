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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/memory/Memory.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/storage_adapters/s3fs/RegisterS3FileSystem.h"
#include "velox/connectors/hive/storage_adapters/s3fs/tests/S3Test.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#ifdef VELOX_ENABLE_S3_DIRECT_RECEIVE
#include <kvikio/defaults.hpp>
#include <kvikio/remote_direct_receive.hpp>
#include <kvikio/remote_handle.hpp>
#endif

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <thread>
#include <tuple>
#include <utility>

using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox::exec::test;
namespace {

class S3ReadTest : public S3Test, public ::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    ASSERT_TRUE(filesystems::initializeS3("OFF"));
  }

  static void TearDownTestCase() {
    filesystems::finalizeS3FileSystem();
  }

  void SetUp() override {
    S3Test::SetUp();
    // Register cudf to enable the CudfDatasource creation from
    // CudfHiveConnector
    facebook::velox::cudf_velox::registerCudf();
    filesystems::registerS3FileSystem();

    // Register Hive connector
    facebook::velox::cudf_velox::connector::hive::CudfHiveConnectorFactory
        factory;
    auto hiveConnector = factory.newConnector(
        kCudfHiveConnectorId, minioServer_->hiveConfig(), ioExecutor_.get());
    facebook::velox::connector::ConnectorRegistry::global().insert(
        hiveConnector->connectorId(), hiveConnector);
    ASSERT_NO_FATAL_FAILURE(waitForMinio());
  }

  void TearDown() override {
    facebook::velox::connector::ConnectorRegistry::global().erase(
        kCudfHiveConnectorId);
    S3Test::TearDown();
  }

  void waitForMinio() {
    constexpr std::string_view kBucket = "readiness-check";
    constexpr std::string_view kKey = "ready.bin";
    minioServer_->addBucket(kBucket.data());
    {
      std::ofstream output(
          minioServer_->path() + "/" + std::string(kBucket) + "/" +
              std::string(kKey),
          std::ios::binary);
      ASSERT_TRUE(output.is_open());
      output.put('!');
      ASSERT_TRUE(output.good());
    }

    const auto readyPath =
        "s3://" + std::string(kBucket) + "/" + std::string(kKey);
    auto fileSystem =
        filesystems::getFileSystem(readyPath, minioServer_->hiveConfig());
    std::string lastError;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
      try {
        filesystems::FileOptions options;
        options.fileSize = 1;
        auto readFile = fileSystem->openFileForRead(readyPath, options);
        char byte;
        readFile->pread(0, 1, &byte);
        if (byte == '!') {
          return;
        }
        lastError = "readiness object contained an unexpected byte";
      } catch (const std::exception& error) {
        lastError = error.what();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    FAIL() << "MinIO did not become ready: " << lastError;
  }
};

// Resolves int.parquet, falling back to a working-directory lookup when the
// build-time path baked into the binary is unreachable (velox-cudf CI builds
// and runs the binaries on different hosts).
std::string resolveIntParquetPath() {
  const std::string relativePath =
      "../../../dwio/parquet/tests/examples/int.parquet";
  auto path =
      test::getDataFilePath("velox/experimental/cudf/tests", relativePath);
  if (std::filesystem::exists(path)) {
    return path;
  }
  return (std::filesystem::current_path() / relativePath)
      .lexically_normal()
      .string();
}

#ifdef VELOX_ENABLE_S3_DIRECT_RECEIVE
class ScopedEnvironmentVariable {
 public:
  ScopedEnvironmentVariable(std::string key, std::string value)
      : key_(std::move(key)) {
    savePreviousValue();
    VELOX_CHECK_EQ(setenv(key_.c_str(), value.c_str(), 1), 0);
  }

  ScopedEnvironmentVariable(std::string key, std::nullopt_t)
      : key_(std::move(key)) {
    savePreviousValue();
    VELOX_CHECK_EQ(unsetenv(key_.c_str()), 0);
  }

  ~ScopedEnvironmentVariable() {
    if (previous_.has_value()) {
      std::ignore = setenv(key_.c_str(), previous_->c_str(), 1);
    } else {
      std::ignore = unsetenv(key_.c_str());
    }
  }

  ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
  ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) =
      delete;

 private:
  void savePreviousValue() {
    if (const auto* previous = std::getenv(key_.c_str())) {
      previous_ = previous;
    }
  }

  std::string key_;
  std::optional<std::string> previous_;
};

class ScopedKvikioDirectReceive {
 public:
  ScopedKvikioDirectReceive()
      : backend_(kvikio::defaults::remote_io_backend()),
        mode_(kvikio::defaults::remote_direct_receive_mode()) {
    kvikio::defaults::set_remote_io_backend(
        kvikio::RemoteIOBackend::MULTI_POLL);
    kvikio::defaults::set_remote_direct_receive_mode(
        kvikio::RemoteDirectReceiveMode::PREFER);
    kvikio::reset_remote_direct_receive_stats();
  }

  ~ScopedKvikioDirectReceive() {
    kvikio::defaults::set_remote_direct_receive_mode(mode_);
    kvikio::defaults::set_remote_io_backend(backend_);
    kvikio::reset_remote_direct_receive_stats();
  }

  ScopedKvikioDirectReceive(const ScopedKvikioDirectReceive&) = delete;
  ScopedKvikioDirectReceive& operator=(const ScopedKvikioDirectReceive&) =
      delete;

 private:
  kvikio::RemoteIOBackend backend_;
  kvikio::RemoteDirectReceiveMode mode_;
};
#endif
} // namespace

TEST_F(S3ReadTest, s3ReadTest) {
  const auto sourceFile = resolveIntParquetPath();
  const char* bucketName = "data";
  const auto destinationFile = S3Test::localPath(bucketName) + "/int.parquet";
  minioServer_->addBucket(bucketName);
  std::ifstream src(sourceFile, std::ios::binary);
  std::ofstream dest(destinationFile, std::ios::binary);
  // Copy source file to destination bucket.
  dest << src.rdbuf();
  ASSERT_GT(dest.tellp(), 0) << "Unable to copy from source " << sourceFile;
  dest.close();

  // Read the parquet file via the S3 bucket.
  auto rowType = ROW({"int", "bigint"}, {INTEGER(), BIGINT()});
  auto tableHandle =
      std::make_shared<facebook::velox::connector::hive::HiveTableHandle>(
          kCudfHiveConnectorId,
          "int_table",
          common::SubfieldFilters{},
          nullptr);
  auto plan = PlanBuilder(pool())
                  .startTableScan()
                  .tableHandle(tableHandle)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();
  auto split = facebook::velox::connector::hive::HiveConnectorSplitBuilder(
                   filesystems::s3URI(bucketName, "int.parquet"))
                   .connectorId(kCudfHiveConnectorId)
                   .fileFormat(dwio::common::FileFormat::PARQUET)
                   .build();

  auto copy = AssertQueryBuilder(plan).split(split).copyResults(pool());

  // expectedResults is the data in int.parquet file.
  const int64_t kExpectedRows = 10;
  auto expectedResults = makeRowVector(
      {makeFlatVector<int32_t>(
           kExpectedRows, [](auto row) { return row + 100; }),
       makeFlatVector<int64_t>(
           kExpectedRows, [](auto row) { return row + 1000; })});
  assertEqualResults({expectedResults}, {copy});
}

#ifdef VELOX_ENABLE_S3_DIRECT_RECEIVE
#if defined(VELOX_ENABLE_S3_DIRECT_RECEIVE_KTLS)
TEST_F(S3ReadTest, kvikioCopiedStreamDirectReceiveRead) {
  const auto sourceFile = resolveIntParquetPath();
  constexpr const char* kBucket = "direct-data";
  const auto destinationFile = S3Test::localPath(kBucket) + "/int.parquet";
  minioServer_->addBucket(kBucket);
  std::ifstream src(sourceFile, std::ios::binary);
  std::ofstream dest(destinationFile, std::ios::binary);
  dest << src.rdbuf();
  ASSERT_GT(dest.tellp(), 0) << "Unable to copy from source " << sourceFile;
  dest.close();

  const auto config = minioServer_->hiveConfig();
  const auto& properties = config->rawConfigs();
  ScopedEnvironmentVariable endpoint(
      "AWS_ENDPOINT_URL", "http://" + properties.at("hive.s3.endpoint"));
  ScopedEnvironmentVariable accessKey(
      "AWS_ACCESS_KEY_ID", properties.at("hive.s3.aws-access-key"));
  ScopedEnvironmentVariable secretKey(
      "AWS_SECRET_ACCESS_KEY", properties.at("hive.s3.aws-secret-key"));
  ScopedEnvironmentVariable sessionToken("AWS_SESSION_TOKEN", std::nullopt);
  ScopedEnvironmentVariable region("AWS_REGION", "us-east-1");
  ScopedKvikioDirectReceive directReceive;

  auto rowType = ROW({"int", "bigint"}, {INTEGER(), BIGINT()});
  auto tableHandle =
      std::make_shared<facebook::velox::connector::hive::HiveTableHandle>(
          kCudfHiveConnectorId,
          "int_table",
          common::SubfieldFilters{},
          nullptr);
  auto plan = PlanBuilder(pool())
                  .startTableScan()
                  .tableHandle(tableHandle)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();
  constexpr int64_t kExpectedRows = 10;
  auto expectedResults = makeRowVector(
      {makeFlatVector<int32_t>(
           kExpectedRows, [](auto row) { return row + 100; }),
       makeFlatVector<int64_t>(
           kExpectedRows, [](auto row) { return row + 1000; })});

  for (int reader = 0; reader < 2; ++reader) {
    const bool experimentalReader = reader != 0;
    SCOPED_TRACE(experimentalReader ? "experimental reader" : "default reader");
    kvikio::reset_remote_direct_receive_stats();

    auto split = facebook::velox::connector::hive::HiveConnectorSplitBuilder(
                     filesystems::s3URI(kBucket, "int.parquet"))
                     .connectorId(kCudfHiveConnectorId)
                     .fileFormat(dwio::common::FileFormat::PARQUET)
                     .build();
    auto copy = AssertQueryBuilder(plan)
                    .connectorSessionProperty(
                        kCudfHiveConnectorId,
                        facebook::velox::cudf_velox::connector::hive::
                            CudfHiveConfig::kUseBufferedInputSession,
                        "false")
                    .connectorSessionProperty(
                        kCudfHiveConnectorId,
                        facebook::velox::cudf_velox::connector::hive::
                            CudfHiveConfig::kUseExperimentalCudfReaderSession,
                        experimentalReader ? "true" : "false")
                    .split(split)
                    .copyResults(pool());
    assertEqualResults({expectedResults}, {copy});

    const auto stats = kvikio::remote_direct_receive_stats();
    EXPECT_GT(stats.transfers_requested, 0);
    EXPECT_EQ(stats.transfers_failed, 0);
    EXPECT_EQ(stats.protocol_validation_failures, 0);
    EXPECT_EQ(stats.retries, 0);
    EXPECT_EQ(stats.cancellations, 0);
    EXPECT_GT(stats.pinned_slots_acquired, 0);
    EXPECT_EQ(stats.strict_rx_transfers_activated, 0);
    EXPECT_EQ(stats.strict_rx_transfers_completed, 0);
    EXPECT_EQ(stats.strict_rx_raw_received_bytes, 0);
    EXPECT_EQ(stats.strict_rx_body_bytes, 0);
    EXPECT_EQ(stats.strict_rx_h2d_bytes, 0);
    EXPECT_EQ(stats.transfers_fallback, stats.transfers_requested);
    EXPECT_EQ(stats.fallback_ineligible_request, stats.transfers_requested);
    EXPECT_EQ(stats.fallback_capability_unavailable, 0);
    EXPECT_EQ(
        stats.copied_stream_transfers_completed, stats.transfers_requested);
    EXPECT_GT(stats.copied_stream_body_bytes, 0);
    EXPECT_GT(stats.copied_stream_h2d_bytes, 0);
    EXPECT_LE(stats.copied_stream_h2d_bytes, stats.copied_stream_body_bytes);
  }
}
#endif
#endif
