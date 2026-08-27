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

#include <aws/core/utils/memory/stl/AWSStreamFwd.h>
#include <folly/Range.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#if defined(VELOX_ENABLE_S3_DIRECT_RECEIVE) && \
    __has_include(<aws/core/http/DirectResponseReceiveStream.h>)
#define VELOX_S3_DIRECT_RECEIVE_AVAILABLE 1
#else
#define VELOX_S3_DIRECT_RECEIVE_AVAILABLE 0
#endif

#if VELOX_S3_DIRECT_RECEIVE_AVAILABLE && \
    defined(VELOX_ENABLE_S3_DIRECT_RECEIVE_KTLS)
#define VELOX_S3_DIRECT_RECEIVE_KTLS_AVAILABLE 1
#else
#define VELOX_S3_DIRECT_RECEIVE_KTLS_AVAILABLE 0
#endif

namespace facebook::velox::filesystems {

/// Controls placement of S3 response bytes into caller-owned host memory.
enum class S3DirectReceiveMode {
  /// Use the ordinary AWS SDK response stream path.
  DISABLED,
  /// Loan final CPU-addressable host destinations to curl. Directly placed
  /// body spans avoid the AWS response-stream copy; header/body tails may use
  /// bounded scratch, and sparse gaps use bounded discard storage. TLS, when
  /// enabled, remains in user space.
  CALLER_BUFFER,
  /// Try strict RX kTLS, then retry on a fresh connection with user-space TLS
  /// only after an authenticated capability rejection delivered no body bytes.
  PREFERRED,
  /// Fail the read unless strict RX kTLS direct receive is active.
  REQUIRED,
};

struct S3DirectReceiveSummary {
  size_t attempts{0};
  size_t kernelTlsAttempts{0};
  size_t callerBufferAttempts{0};
  size_t responseRejectedAttempts{0};
  size_t unavailableAttempts{0};
  size_t kernelTlsUnavailableAttempts{0};
  size_t failedAttempts{0};
  bool kernelTlsUnavailable{false};
  bool succeeded{false};
  /// Response-body bytes observed by curl across all attempts for this
  /// logical read, including rejected service responses and failed or retried
  /// transfers. This is not a wire-byte count or a unique-logical-byte count.
  uint64_t receivedBodyBytes{0};
  /// Observed bytes that curl placed in caller-owned loaned buffers and the
  /// stream successfully committed. This includes bytes from attempts that
  /// later failed and bytes placed in a discard buffer for sparse reads.
  uint64_t receivedDirectBodyBytes{0};
  /// Bytes from accepted responses that the stream copied into caller-owned
  /// destinations (or its discard buffer). Attempts that later fail are still
  /// included; rejected response bodies remain on the error-stream path and
  /// are therefore not included.
  uint64_t receivedCopiedBodyBytes{0};
  /// Subset of receivedDirectBodyBytes plus receivedCopiedBodyBytes consumed
  /// for null preadv ranges. Discarded bytes are not an additional placement
  /// category.
  uint64_t receivedDiscardedBodyBytes{0};
  /// Placement accounting for the final attempt, populated only when that
  /// attempt completed successfully at the HTTP transport layer. Direct plus
  /// copied equals body; discarded is a subset of those placement categories.
  /// The bytes become publishable only after the enclosing GetObject succeeds.
  uint64_t successfulBodyBytes{0};
  uint64_t successfulDirectBodyBytes{0};
  uint64_t successfulCopiedBodyBytes{0};
  uint64_t successfulDiscardedBodyBytes{0};
  std::string failure;
};

enum class S3DirectReceiveAttemptResult {
  SUCCESS,
  RESPONSE_REJECTED,
  UNAVAILABLE,
  FAILED,
};

/// State shared by all AWS SDK attempts for one logical range read.
///
/// AWS constructs a new response stream for every retry. Sharing this object
/// lets PREFERRED mode change only the retry after a capability failure to
/// caller-owned buffers with ordinary TLS. Network, certificate, protocol,
/// integrity, and partial-body failures never activate that fallback.
class S3DirectReceiveRequestState {
 public:
  S3DirectReceiveRequestState(
      S3DirectReceiveMode mode,
      uint64_t expectedBodyBytes);

  /// Starts an attempt and returns whether that attempt must use RX kTLS.
  bool beginAttempt();

  void finishAttempt(
      bool requiredKernelTls,
      S3DirectReceiveAttemptResult result,
      uint64_t bodyBytes,
      uint64_t directBodyBytes,
      uint64_t copiedBodyBytes,
      uint64_t discardedBodyBytes,
      std::string_view failure) noexcept;

  /// True when the SDK exhausted its retries immediately after discovering
  /// that kTLS was unavailable, before it constructed a caller-buffer attempt.
  bool shouldRetryWithCallerBuffer() const;

  S3DirectReceiveSummary summary() const;

  S3DirectReceiveMode mode() const {
    return mode_;
  }

  uint64_t expectedBodyBytes() const {
    return expectedBodyBytes_;
  }

 private:
  const S3DirectReceiveMode mode_;
  const uint64_t expectedBodyBytes_;
  mutable std::mutex mutex_;
  S3DirectReceiveSummary summary_;
};

bool s3DirectReceiveBuildSupported();

/// Returns whether this build supports the strict RX kTLS modes. CPU
/// caller-buffer placement intentionally does not depend on this capability.
bool s3DirectReceiveKernelTlsBuildSupported();

/// Creates one response stream for an AWS SDK attempt.
///
/// Non-null ranges are final destinations. Null ranges consume response bytes
/// through a bounded discard buffer, matching ReadFile::preadv gap semantics.
/// The pointed-to storage must outlive the returned stream.
Aws::IOStream* makeS3DirectReceiveStream(
    const char* allocationTag,
    std::shared_ptr<S3DirectReceiveRequestState> state,
    std::vector<folly::Range<char*>> ranges,
    uint64_t offset,
    uint64_t objectSize);

} // namespace facebook::velox::filesystems
