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

#include "velox/connectors/hive/storage_adapters/s3fs/S3ReadFile.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Counters.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3DirectReceive.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3DirectReceiveCapabilityCache.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Util.h"

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <folly/ScopeGuard.h>

#include <cstring>
#include <limits>
#include <optional>

namespace facebook::velox::filesystems {

namespace {

// By default, the AWS SDK reads object data into an auto-growing StringStream.
// To avoid copies, read directly into a pre-allocated buffer instead.
// See https://github.com/aws/aws-sdk-cpp/issues/64 for an alternative but
// functionally similar recipe.
Aws::IOStreamFactory AwsWriteableStreamFactory(void* data, int64_t nbytes) {
  return [=]() { return Aws::New<StringViewStream>("", data, nbytes); };
}

} // namespace

class S3ReadFile ::Impl {
 public:
  explicit Impl(
      std::string_view path,
      Aws::S3::S3Client* client,
      S3DirectReceiveMode directReceiveMode,
      std::shared_ptr<S3DirectReceiveCapabilityCache> capabilityCache)
      : client_(client),
        directReceiveMode_(directReceiveMode),
        capabilityCache_(std::move(capabilityCache)) {
    getBucketAndKeyFromPath(path, bucket_, key_);
    VELOX_USER_CHECK(
        directReceiveMode_ == S3DirectReceiveMode::DISABLED ||
            s3DirectReceiveBuildSupported(),
        "hive.s3.direct-receive-mode requires an AWS SDK and curl build "
        "with caller-owned receive-buffer support");
    VELOX_USER_CHECK(
        (directReceiveMode_ != S3DirectReceiveMode::PREFERRED &&
         directReceiveMode_ != S3DirectReceiveMode::REQUIRED) ||
            s3DirectReceiveKernelTlsBuildSupported(),
        "hive.s3.direct-receive-mode={} requires a build with strict RX "
        "kTLS support; use caller-buffer for CPU direct placement without "
        "kTLS",
        directReceiveMode_ == S3DirectReceiveMode::PREFERRED ? "preferred"
                                                             : "required");
    VELOX_CHECK(
        directReceiveMode_ != S3DirectReceiveMode::PREFERRED ||
            capabilityCache_ != nullptr,
        "Preferred S3 direct receive requires a shared capability cache");
  }

  // Gets the length of the file.
  // Checks if there are any issues reading the file.
  void initialize(const filesystems::FileOptions& options) {
    if (options.fileSize.has_value()) {
      VELOX_CHECK_GE(
          options.fileSize.value(), 0, "File size must be non-negative");
      length_ = options.fileSize.value();
    }

    // Make it a no-op if invoked twice.
    if (length_ != -1) {
      return;
    }

    Aws::S3::Model::HeadObjectRequest request;
    request.SetBucket(awsString(bucket_));
    request.SetKey(awsString(key_));

    RECORD_METRIC_VALUE(kMetricS3MetadataCalls);
    auto outcome = client_->HeadObject(request);
    if (!outcome.IsSuccess()) {
      RECORD_METRIC_VALUE(kMetricS3GetMetadataErrors);
    }
    RECORD_METRIC_VALUE(kMetricS3GetMetadataRetries, outcome.GetRetryCount());
    VELOX_CHECK_AWS_OUTCOME(
        outcome, "Failed to get metadata for S3 object", bucket_, key_);
    length_ = outcome.GetResult().GetContentLength();
    VELOX_CHECK_GE(length_, 0);
  }

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buffer,
      const FileIoContext& context) const {
    if (length > 0) {
      VELOX_CHECK_NOT_NULL(buffer, "S3 pread destination must not be null");
      preadInternal(
          offset, {folly::Range<char*>(static_cast<char*>(buffer), length)});
    }
    return {static_cast<char*>(buffer), length};
  }

  std::string
  pread(uint64_t offset, uint64_t length, const FileIoContext& context) const {
    std::string result(length, 0);
    if (length > 0) {
      preadInternal(
          offset, {folly::Range<char*>(result.data(), result.size())});
    }
    return result;
  }

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& context) const {
    // 'buffers' contains Ranges(data, size) with some gaps (data = nullptr) in
    // between. This call must populate the ranges (except gap ranges)
    // sequentially starting from 'offset'. AWS S3 GetObject does not support
    // multi-range. AWS S3 also charges by number of read requests and not size.
    // A copied response scatters directly from each AWS body callback. Direct
    // receive loans real ranges to curl and uses bounded scratch for gaps.
    const auto length = totalRangeSize(buffers);
    if (length > 0) {
      preadInternal(offset, buffers);
    }
    return length;
  }

  uint64_t size() const {
    return length_;
  }

  uint64_t memoryUsage() const {
    // TODO: Check if any buffers are being used by the S3 library
    return sizeof(Aws::S3::S3Client) + kS3MaxKeySize + 2 * sizeof(std::string) +
        sizeof(int64_t);
  }

  bool shouldCoalesce() const {
    return false;
  }

  std::string getName() const {
    return fmt::format("s3://{}/{}", bucket_, key_);
  }

 private:
  static uint64_t totalRangeSize(
      const std::vector<folly::Range<char*>>& buffers) {
    uint64_t length = 0;
    for (const auto& range : buffers) {
      VELOX_CHECK_LE(
          range.size(),
          std::numeric_limits<uint64_t>::max() - length,
          "S3 read range size overflow");
      length += range.size();
    }
    return length;
  }

  void preadInternal(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers) const {
    const auto length = totalRangeSize(buffers);
    VELOX_CHECK_GT(length, 0);
    VELOX_CHECK_GE(length_, 0);
    const auto objectSize = static_cast<uint64_t>(length_);
    VELOX_CHECK_LE(offset, objectSize, "S3 read offset exceeds object size");
    VELOX_CHECK_LE(
        length, objectSize - offset, "S3 read range exceeds object size");

    // Read the desired range of bytes.
    Aws::S3::Model::GetObjectRequest request;

    request.SetBucket(awsString(bucket_));
    request.SetKey(awsString(key_));
    std::stringstream ss;
    ss << "bytes=" << offset << "-" << offset + length - 1;
    request.SetRange(awsString(ss.str()));

    std::shared_ptr<S3DirectReceiveRequestState> directState;
    std::optional<S3DirectReceiveCapabilityCache::Attempt> capabilityAttempt;
    auto abandonCapabilityProbe = folly::makeGuard([&]() {
      if (capabilityAttempt.has_value()) {
        capabilityCache_->abandonAttempt(capabilityAttempt.value());
      }
    });
    std::string copiedScatterBuffer;
    if (directReceiveMode_ == S3DirectReceiveMode::DISABLED) {
      if (buffers.size() == 1 && buffers.front().data() != nullptr) {
        request.SetResponseStreamFactory(
            AwsWriteableStreamFactory(buffers.front().data(), length));
      } else {
        VELOX_CHECK_LE(
            length,
            copiedScatterBuffer.max_size(),
            "S3 scatter read exceeds the response buffer size limit");
        copiedScatterBuffer.resize(length);
        request.SetResponseStreamFactory(AwsWriteableStreamFactory(
            copiedScatterBuffer.data(), copiedScatterBuffer.size()));
      }
    } else {
      auto attemptMode = directReceiveMode_;
      if (directReceiveMode_ == S3DirectReceiveMode::PREFERRED) {
        capabilityAttempt = capabilityCache_->beginAttempt();
        if (!capabilityAttempt->requiresKernelTls()) {
          attemptMode = S3DirectReceiveMode::CALLER_BUFFER;
        }
      }
      directState =
          std::make_shared<S3DirectReceiveRequestState>(attemptMode, length);
      request.SetResponseStreamFactory(
          [directState, buffers, offset, objectSize]() {
            return makeS3DirectReceiveStream(
                "", directState, buffers, offset, objectSize);
          });
    }

    RECORD_METRIC_VALUE(kMetricS3ActiveConnections);
    auto decrementActiveConnections = folly::makeGuard(
        []() { RECORD_METRIC_VALUE(kMetricS3ActiveConnections, -1); });
    RECORD_METRIC_VALUE(kMetricS3GetObjectCalls);
    auto getObject = [&]() { return client_->GetObject(request); };
    auto outcome = getObject();
    auto retryCount = outcome.GetRetryCount();
    if (!outcome.IsSuccess() && directState &&
        directState->shouldRetryWithCallerBuffer()) {
      // Strict RX kTLS was unavailable before any body byte was delivered.
      // The curl client destroys that failed handle, so the fallback cannot
      // accidentally reuse a partially configured connection.
      outcome = getObject();
      retryCount += 1 + outcome.GetRetryCount();
    }
    if (!outcome.IsSuccess()) {
      RECORD_METRIC_VALUE(kMetricS3GetObjectErrors);
    }
    RECORD_METRIC_VALUE(kMetricS3GetObjectRetries, retryCount);
    std::optional<S3DirectReceiveSummary> directSummary;
    if (directState) {
      directSummary = directState->summary();
      const auto& summary = directSummary.value();
      if (capabilityAttempt.has_value()) {
        auto observation =
            S3DirectReceiveCapabilityCache::Observation::INDETERMINATE;
        uint64_t observedBodyBytes = summary.receivedBodyBytes;
        uint64_t observedDirectBodyBytes = summary.receivedDirectBodyBytes;
        if (summary.kernelTlsUnavailable) {
          // RequestState sets this bit only for a strict UNAVAILABLE attempt
          // that delivered no body bytes. A later caller-buffer fallback may
          // contribute to the logical read's cumulative counters.
          observation =
              S3DirectReceiveCapabilityCache::Observation::UNAVAILABLE;
          observedBodyBytes = 0;
          observedDirectBodyBytes = 0;
        } else if (
            outcome.IsSuccess() && summary.succeeded &&
            summary.callerBufferAttempts == 0) {
          observation = S3DirectReceiveCapabilityCache::Observation::AVAILABLE;
        }
        capabilityCache_->finishAttempt(
            capabilityAttempt.value(),
            observation,
            observedBodyBytes,
            observedDirectBodyBytes);
        abandonCapabilityProbe.dismiss();
      }
      RECORD_METRIC_VALUE(
          kMetricS3DirectReceiveKernelTlsAttempts, summary.kernelTlsAttempts);
      RECORD_METRIC_VALUE(
          kMetricS3DirectReceiveCallerBufferAttempts,
          summary.callerBufferAttempts);
      RECORD_METRIC_VALUE(
          kMetricS3DirectReceiveKernelTlsUnavailable,
          summary.kernelTlsUnavailableAttempts);
      RECORD_METRIC_VALUE(
          kMetricS3DirectReceiveFallbacks,
          summary.kernelTlsUnavailable && summary.callerBufferAttempts > 0 ? 1
                                                                           : 0);
      RECORD_METRIC_VALUE(
          kMetricS3DirectReceiveResponseRejections,
          summary.responseRejectedAttempts);
      RECORD_METRIC_VALUE(
          kMetricS3DirectReceiveMechanismFailures, summary.failedAttempts);
      RECORD_METRIC_VALUE(
          kMetricS3DirectReceiveReceivedBodyBytes, summary.receivedBodyBytes);
      RECORD_METRIC_VALUE(
          kMetricS3DirectReceiveReceivedDirectBodyBytes,
          summary.receivedDirectBodyBytes);
      RECORD_METRIC_VALUE(
          kMetricS3DirectReceiveReceivedCopiedBodyBytes,
          summary.receivedCopiedBodyBytes);
      RECORD_METRIC_VALUE(
          kMetricS3DirectReceiveReceivedDiscardedBodyBytes,
          summary.receivedDiscardedBodyBytes);
      const bool publishable = summary.succeeded && outcome.IsSuccess();
      RECORD_METRIC_VALUE(
          kMetricS3DirectReceiveSuccessfulBodyBytes,
          publishable ? summary.successfulBodyBytes : 0);
      RECORD_METRIC_VALUE(
          kMetricS3DirectReceiveSuccessfulDirectBodyBytes,
          publishable ? summary.successfulDirectBodyBytes : 0);
      RECORD_METRIC_VALUE(
          kMetricS3DirectReceiveSuccessfulCopiedBodyBytes,
          publishable ? summary.successfulCopiedBodyBytes : 0);
      RECORD_METRIC_VALUE(
          kMetricS3DirectReceiveSuccessfulDiscardedBodyBytes,
          publishable ? summary.successfulDiscardedBodyBytes : 0);
    }
    VELOX_CHECK_AWS_OUTCOME(outcome, "Failed to get S3 object", bucket_, key_);
    if (!copiedScatterBuffer.empty()) {
      size_t sourceOffset = 0;
      for (const auto& range : buffers) {
        if (range.data() != nullptr) {
          std::memcpy(
              range.data(),
              copiedScatterBuffer.data() + sourceOffset,
              range.size());
        }
        sourceOffset += range.size();
      }
    }
    if (directSummary) {
      const auto& summary = directSummary.value();
      const bool directReceiveValid =
          summary.succeeded && summary.successfulBodyBytes == length;
      if (!directReceiveValid) {
        // The SDK can report a successful HTTP operation even when the direct
        // stream rejects unsafe response framing. Count the resulting Velox
        // read failure as a GET error as well.
        RECORD_METRIC_VALUE(kMetricS3GetObjectErrors);
      }
      VELOX_CHECK(
          directReceiveValid,
          "S3 direct receive validation failed for s3://{}/{}: {}",
          bucket_,
          key_,
          summary.failure.empty() ? "incomplete direct response"
                                  : summary.failure);
    }
  }

  Aws::S3::S3Client* client_;
  std::string bucket_;
  std::string key_;
  int64_t length_ = -1;
  const S3DirectReceiveMode directReceiveMode_;
  const std::shared_ptr<S3DirectReceiveCapabilityCache> capabilityCache_;
};

S3ReadFile::S3ReadFile(std::string_view path, Aws::S3::S3Client* client)
    : S3ReadFile(path, client, S3DirectReceiveMode::DISABLED, nullptr) {}

S3ReadFile::S3ReadFile(
    std::string_view path,
    Aws::S3::S3Client* client,
    S3DirectReceiveMode directReceiveMode,
    std::shared_ptr<S3DirectReceiveCapabilityCache> capabilityCache) {
  impl_ = std::make_shared<Impl>(
      path, client, directReceiveMode, std::move(capabilityCache));
}

S3ReadFile::~S3ReadFile() = default;

void S3ReadFile::initialize(const filesystems::FileOptions& options) {
  return impl_->initialize(options);
}

std::string_view S3ReadFile::pread(
    uint64_t offset,
    uint64_t length,
    void* buf,
    const FileIoContext& context) const {
  auto result = impl_->pread(offset, length, buf, context);
  bytesRead_ += length;
  return result;
}

std::string S3ReadFile::pread(
    uint64_t offset,
    uint64_t length,
    const FileIoContext& context) const {
  auto result = impl_->pread(offset, length, context);
  bytesRead_ += length;
  return result;
}

uint64_t S3ReadFile::preadv(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers,
    const FileIoContext& context) const {
  const auto length = impl_->preadv(offset, buffers, context);
  bytesRead_ += length;
  return length;
}

uint64_t S3ReadFile::size() const {
  return impl_->size();
}

uint64_t S3ReadFile::memoryUsage() const {
  return impl_->memoryUsage();
}

bool S3ReadFile::shouldCoalesce() const {
  return impl_->shouldCoalesce();
}

std::string S3ReadFile::getName() const {
  return impl_->getName();
}

} // namespace facebook::velox::filesystems
