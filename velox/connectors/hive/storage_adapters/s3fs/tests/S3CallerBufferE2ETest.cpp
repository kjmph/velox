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

#include <folly/init/Init.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include "velox/common/base/StatsReporter.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"
#include "velox/connectors/hive/storage_adapters/s3fs/RegisterS3FileSystem.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Counters.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3DirectReceive.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3FileSystem.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3ReadFile.h"
#include "velox/connectors/hive/storage_adapters/s3fs/tests/MinioServer.h"

#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/http/DirectResponseReceiveStream.h>
#include <aws/core/utils/memory/AWSMemory.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <gtest/gtest.h>

namespace facebook::velox::filesystems {
namespace {

class DirectReceiveStatsReporter final : public BaseStatsReporter {
 public:
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    counters_.clear();
  }

  size_t value(std::string_view key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = counters_.find(std::string(key));
    return it == counters_.end() ? 0 : it->second;
  }

  void addMetricValue(const std::string& key, size_t value) const override {
    add(key, value);
  }

  void addMetricValue(const char* key, size_t value) const override {
    add(key, value);
  }

  void addMetricValue(folly::StringPiece key, size_t value) const override {
    add(key.str(), value);
  }

 private:
  void add(std::string_view key, size_t value) const {
    std::lock_guard<std::mutex> lock(mutex_);
    counters_[std::string(key)] += value;
  }

  mutable std::mutex mutex_;
  mutable std::map<std::string, size_t> counters_;
};

folly::Singleton<BaseStatsReporter> reporter([]() {
  return new DirectReceiveStatsReporter();
});

class RejectedPartialResponseS3Client final : public Aws::S3::S3Client {
 public:
  RejectedPartialResponseS3Client()
      : Aws::S3::S3Client(Aws::Auth::AWSCredentials("access", "secret")) {}

  Aws::S3::Model::GetObjectOutcome GetObject(
      const Aws::S3::Model::GetObjectRequest& request) const override {
    using StreamPtr = std::unique_ptr<Aws::IOStream, void (*)(Aws::IOStream*)>;
    StreamPtr stream(
        request.GetResponseStreamFactory()(),
        [](Aws::IOStream* value) { Aws::Delete(value); });
    auto* direct =
        dynamic_cast<Aws::Http::DirectResponseReceiveStream*>(stream.get());
    VELOX_CHECK_NOT_NULL(direct);

    Aws::Http::HeaderValueCollection headers;
    headers.emplace("content-length", "64");
    VELOX_CHECK(!direct->AcceptDirectResponse(
        Aws::Http::HttpResponseCode::PARTIAL_CONTENT, headers));
    direct->CompleteDirectResponse(
        {Aws::Http::DirectResponseReceiveCompletionResult::
             RESPONSE_NOT_ACCEPTED,
         0,
         0});
    return Aws::S3::Model::GetObjectOutcome(Aws::S3::Model::GetObjectResult{});
  }
};

class S3CallerBufferE2ETest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    BaseStatsReporter::registered = true;
    registerS3Metrics();
    ASSERT_TRUE(initializeS3("OFF"));
  }

  static void TearDownTestSuite() {
    finalizeS3();
    BaseStatsReporter::registered = false;
  }

  void SetUp() override {
    minioServer_ = std::make_unique<MinioServer>();
    minioServer_->start();
    ASSERT_NO_FATAL_FAILURE(waitForMinio());
    stats_ = std::dynamic_pointer_cast<DirectReceiveStatsReporter>(
        folly::Singleton<BaseStatsReporter>::try_get());
    ASSERT_NE(stats_, nullptr);
    stats_->clear();
  }

  void TearDown() override {
    minioServer_->stop();
  }

  static std::vector<char> makeObject(size_t size) {
    std::vector<char> object(size);
    for (size_t i = 0; i < size; ++i) {
      object[i] = static_cast<char>((i * 193 + 29) & 0xff);
    }
    return object;
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

    S3FileSystem fileSystem(kBucket, minioServer_->hiveConfig());
    std::string lastError;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
      try {
        FileOptions options;
        options.fileSize = 1;
        auto readFile = fileSystem.openFileForRead(
            "s3://" + std::string(kBucket) + "/" + std::string(kKey), options);
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

  std::unique_ptr<MinioServer> minioServer_;
  std::shared_ptr<DirectReceiveStatsReporter> stats_;
};

TEST_F(S3CallerBufferE2ETest, denseAndSparseReadsUseCallerOwnedBuffers) {
  constexpr std::string_view kBucket = "caller-buffer";
  constexpr std::string_view kKey = "object.bin";
  constexpr size_t kObjectSize = 2 * 1024 * 1024 + 257;
  const auto object = makeObject(kObjectSize);

  minioServer_->addBucket(kBucket.data());
  const auto objectPath = minioServer_->path() + "/" + std::string(kBucket) +
      "/" + std::string(kKey);
  {
    std::ofstream output(objectPath, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output.write(object.data(), object.size());
    ASSERT_TRUE(output.good());
  }

  auto config = minioServer_->hiveConfig(
      {{"hive.s3.direct-receive-mode", "caller-buffer"}});
  S3FileSystem fileSystem(kBucket, config);
  auto readFile = fileSystem.openFileForRead(
      "s3://" + std::string(kBucket) + "/" + std::string(kKey));
  ASSERT_EQ(readFile->size(), object.size());

  constexpr size_t kDenseOffset = 37;
  constexpr size_t kDenseSize = 256 * 1024 + 123;
  std::vector<char> dense(kDenseSize, '?');
  EXPECT_EQ(
      readFile->pread(kDenseOffset, dense.size(), dense.data()),
      std::string_view(object.data() + kDenseOffset, kDenseSize));

  constexpr size_t kSparseOffset = 1024 * 1024 + 17;
  constexpr size_t kFirstSize = 4 * 1024;
  constexpr size_t kFirstGap = 128 * 1024 + 7;
  constexpr size_t kMiddleSize = 8 * 1024;
  constexpr size_t kSecondGap = 64 * 1024 + 3;
  constexpr size_t kLastSize = 16 * 1024;
  constexpr size_t kSparseSize =
      kFirstSize + kFirstGap + kMiddleSize + kSecondGap + kLastSize;

  std::vector<char> first(kFirstSize, '?');
  std::vector<char> middle(kMiddleSize, '?');
  std::vector<char> last(kLastSize, '?');
  const std::vector<folly::Range<char*>> ranges = {
      {first.data(), first.size()},
      {nullptr, kFirstGap},
      {middle.data(), middle.size()},
      {nullptr, kSecondGap},
      {last.data(), last.size()},
  };
  EXPECT_EQ(readFile->preadv(kSparseOffset, ranges), kSparseSize);

  size_t expectedOffset = kSparseOffset;
  EXPECT_EQ(
      std::string_view(first.data(), first.size()),
      std::string_view(object.data() + expectedOffset, first.size()));
  expectedOffset += first.size() + kFirstGap;
  EXPECT_EQ(
      std::string_view(middle.data(), middle.size()),
      std::string_view(object.data() + expectedOffset, middle.size()));
  expectedOffset += middle.size() + kSecondGap;
  EXPECT_EQ(
      std::string_view(last.data(), last.size()),
      std::string_view(object.data() + expectedOffset, last.size()));

  const auto expectedBodyBytes = kDenseSize + kSparseSize;
  EXPECT_EQ(readFile->bytesRead(), expectedBodyBytes);
  EXPECT_EQ(stats_->value(kMetricS3DirectReceiveCallerBufferAttempts), 2);
  EXPECT_EQ(stats_->value(kMetricS3DirectReceiveKernelTlsAttempts), 0);
  EXPECT_EQ(stats_->value(kMetricS3DirectReceiveMechanismFailures), 0);
  EXPECT_EQ(stats_->value(kMetricS3DirectReceiveResponseRejections), 0);
  EXPECT_EQ(
      stats_->value(kMetricS3DirectReceiveSuccessfulBodyBytes),
      expectedBodyBytes);
  EXPECT_GT(stats_->value(kMetricS3DirectReceiveSuccessfulDirectBodyBytes), 0);
  EXPECT_EQ(
      stats_->value(kMetricS3DirectReceiveSuccessfulCopiedBodyBytes) +
          stats_->value(kMetricS3DirectReceiveSuccessfulDirectBodyBytes),
      expectedBodyBytes);
  EXPECT_EQ(
      stats_->value(kMetricS3DirectReceiveSuccessfulDiscardedBodyBytes),
      kFirstGap + kSecondGap);
}

TEST_F(S3CallerBufferE2ETest, disabledModePreservesSparseReadSemantics) {
  constexpr std::string_view kBucket = "disabled-scatter";
  constexpr std::string_view kKey = "object.bin";
  constexpr size_t kObjectSize = 512 * 1024 + 19;
  const auto object = makeObject(kObjectSize);

  minioServer_->addBucket(kBucket.data());
  const auto objectPath = minioServer_->path() + "/" + std::string(kBucket) +
      "/" + std::string(kKey);
  {
    std::ofstream output(objectPath, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output.write(object.data(), object.size());
    ASSERT_TRUE(output.good());
  }

  S3FileSystem fileSystem(kBucket, minioServer_->hiveConfig());
  auto readFile = fileSystem.openFileForRead(
      "s3://" + std::string(kBucket) + "/" + std::string(kKey));

  constexpr size_t kOffset = 123;
  constexpr size_t kFirstSize = 4 * 1024;
  constexpr size_t kFirstGap = 32 * 1024 + 3;
  constexpr size_t kMiddleSize = 8 * 1024 + 1;
  constexpr size_t kSecondGap = 64 * 1024 + 5;
  constexpr size_t kLastSize = 16 * 1024 + 7;
  constexpr size_t kReadSize =
      kFirstSize + kFirstGap + kMiddleSize + kSecondGap + kLastSize;

  std::vector<char> first(kFirstSize, '?');
  std::vector<char> middle(kMiddleSize, '?');
  std::vector<char> last(kLastSize, '?');
  const std::vector<folly::Range<char*>> ranges = {
      {first.data(), first.size()},
      {nullptr, kFirstGap},
      {middle.data(), middle.size()},
      {nullptr, kSecondGap},
      {last.data(), last.size()},
  };
  EXPECT_EQ(readFile->preadv(kOffset, ranges), kReadSize);

  size_t expectedOffset = kOffset;
  EXPECT_EQ(
      std::string_view(first.data(), first.size()),
      std::string_view(object.data() + expectedOffset, first.size()));
  expectedOffset += first.size() + kFirstGap;
  EXPECT_EQ(
      std::string_view(middle.data(), middle.size()),
      std::string_view(object.data() + expectedOffset, middle.size()));
  expectedOffset += middle.size() + kSecondGap;
  EXPECT_EQ(
      std::string_view(last.data(), last.size()),
      std::string_view(object.data() + expectedOffset, last.size()));

  EXPECT_EQ(readFile->bytesRead(), kReadSize);
  EXPECT_EQ(stats_->value(kMetricS3GetObjectCalls), 1);
  EXPECT_EQ(stats_->value(kMetricS3GetObjectErrors), 0);
  EXPECT_EQ(stats_->value(kMetricS3DirectReceiveKernelTlsAttempts), 0);
  EXPECT_EQ(stats_->value(kMetricS3DirectReceiveCallerBufferAttempts), 0);
  EXPECT_EQ(stats_->value(kMetricS3DirectReceiveSuccessfulBodyBytes), 0);
}

TEST_F(S3CallerBufferE2ETest, serviceErrorDoesNotPublishCallerBuffer) {
  constexpr std::string_view kBucket = "caller-buffer-error";
  constexpr std::string_view kKey = "missing.bin";
  constexpr size_t kExpectedSize = 4 * 1024;

  minioServer_->addBucket(kBucket.data());
  auto config = minioServer_->hiveConfig(
      {{"hive.s3.direct-receive-mode", "caller-buffer"}});
  S3FileSystem fileSystem(kBucket, config);
  FileOptions options;
  // Skip HEAD so the missing object reaches the direct GetObject response
  // stream and exercises AWS's XML service-error marshalling path.
  options.fileSize = kExpectedSize;
  auto readFile = fileSystem.openFileForRead(
      "s3://" + std::string(kBucket) + "/" + std::string(kKey), options);

  std::vector<char> destination(kExpectedSize, '?');
  try {
    readFile->pread(0, destination.size(), destination.data());
    FAIL() << "Expected a missing-object read to fail";
  } catch (const VeloxException& error) {
    EXPECT_EQ(error.errorCode(), error_code::kFileNotFound);
  }

  EXPECT_TRUE(
      std::all_of(destination.begin(), destination.end(), [](char value) {
        return value == '?';
      }));
  EXPECT_EQ(readFile->bytesRead(), 0);
  EXPECT_EQ(stats_->value(kMetricS3GetObjectCalls), 1);
  EXPECT_EQ(stats_->value(kMetricS3GetObjectErrors), 1);
  EXPECT_EQ(stats_->value(kMetricS3DirectReceiveCallerBufferAttempts), 1);
  EXPECT_EQ(stats_->value(kMetricS3DirectReceiveResponseRejections), 1);
  EXPECT_EQ(stats_->value(kMetricS3DirectReceiveMechanismFailures), 0);
  EXPECT_GT(stats_->value(kMetricS3DirectReceiveReceivedBodyBytes), 0);
  EXPECT_EQ(stats_->value(kMetricS3DirectReceiveSuccessfulBodyBytes), 0);
}

TEST_F(S3CallerBufferE2ETest, rejectedSuccessfulResponseCountsAsGetError) {
  constexpr size_t kExpectedSize = 64;
  RejectedPartialResponseS3Client client;
  S3ReadFile readFile(
      "metric-test/object.bin",
      &client,
      S3DirectReceiveMode::CALLER_BUFFER,
      nullptr);
  FileOptions options;
  options.fileSize = kExpectedSize;
  readFile.initialize(options);

  std::vector<char> destination(kExpectedSize, '?');
  VELOX_ASSERT_THROW(
      readFile.pread(0, destination.size(), destination.data()),
      "missing Content-Range");

  EXPECT_EQ(readFile.bytesRead(), 0);
  EXPECT_EQ(stats_->value(kMetricS3GetObjectCalls), 1);
  EXPECT_EQ(stats_->value(kMetricS3GetObjectErrors), 1);
  EXPECT_EQ(stats_->value(kMetricS3DirectReceiveResponseRejections), 1);
  EXPECT_EQ(stats_->value(kMetricS3DirectReceiveSuccessfulBodyBytes), 0);
}

} // namespace
} // namespace facebook::velox::filesystems

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv, false};
  return RUN_ALL_TESTS();
}
