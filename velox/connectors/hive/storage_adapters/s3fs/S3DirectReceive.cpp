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

#include "velox/connectors/hive/storage_adapters/s3fs/S3DirectReceive.h"

#include "velox/common/base/Exceptions.h"

#include <aws/core/utils/memory/AWSMemory.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstring>
#include <ios>
#include <limits>
#include <sstream>
#include <streambuf>
#include <string_view>
#include <utility>

#if VELOX_S3_DIRECT_RECEIVE_AVAILABLE
#include <aws/core/http/DirectResponseReceiveStream.h>
#include <aws/core/http/HttpResponse.h>
#endif

namespace facebook::velox::filesystems {

S3DirectReceiveRequestState::S3DirectReceiveRequestState(
    S3DirectReceiveMode mode,
    uint64_t expectedBodyBytes)
    : mode_(mode), expectedBodyBytes_(expectedBodyBytes) {
  VELOX_CHECK(
      mode_ != S3DirectReceiveMode::DISABLED,
      "A disabled S3 read must not construct direct-receive state");
}

bool S3DirectReceiveRequestState::beginAttempt() {
  std::lock_guard<std::mutex> lock(mutex_);
  const bool requireKernelTls = mode_ == S3DirectReceiveMode::REQUIRED ||
      (mode_ == S3DirectReceiveMode::PREFERRED &&
       !summary_.kernelTlsUnavailable);
  ++summary_.attempts;
  if (requireKernelTls) {
    ++summary_.kernelTlsAttempts;
  } else {
    ++summary_.callerBufferAttempts;
  }
  return requireKernelTls;
}

void S3DirectReceiveRequestState::finishAttempt(
    bool requiredKernelTls,
    S3DirectReceiveAttemptResult result,
    uint64_t bodyBytes,
    uint64_t directBodyBytes,
    uint64_t copiedBodyBytes,
    uint64_t discardedBodyBytes,
    std::string_view failure) noexcept {
  try {
    std::lock_guard<std::mutex> lock(mutex_);
    if (requiredKernelTls &&
        result == S3DirectReceiveAttemptResult::UNAVAILABLE && bodyBytes == 0 &&
        directBodyBytes == 0) {
      summary_.kernelTlsUnavailable = true;
    }
    switch (result) {
      case S3DirectReceiveAttemptResult::SUCCESS:
        break;
      case S3DirectReceiveAttemptResult::RESPONSE_REJECTED:
        ++summary_.responseRejectedAttempts;
        break;
      case S3DirectReceiveAttemptResult::UNAVAILABLE:
        ++summary_.unavailableAttempts;
        if (requiredKernelTls) {
          ++summary_.kernelTlsUnavailableAttempts;
        }
        break;
      case S3DirectReceiveAttemptResult::FAILED:
        ++summary_.failedAttempts;
        break;
    }
    summary_.receivedBodyBytes += bodyBytes;
    summary_.receivedDirectBodyBytes += directBodyBytes;
    summary_.receivedCopiedBodyBytes += copiedBodyBytes;
    summary_.receivedDiscardedBodyBytes += discardedBodyBytes;
    summary_.succeeded = result == S3DirectReceiveAttemptResult::SUCCESS;
    if (summary_.succeeded) {
      summary_.successfulBodyBytes = bodyBytes;
      summary_.successfulDirectBodyBytes = directBodyBytes;
      summary_.successfulCopiedBodyBytes = copiedBodyBytes;
      summary_.successfulDiscardedBodyBytes = discardedBodyBytes;
      summary_.failure.clear();
    } else {
      summary_.successfulBodyBytes = 0;
      summary_.successfulDirectBodyBytes = 0;
      summary_.successfulCopiedBodyBytes = 0;
      summary_.successfulDiscardedBodyBytes = 0;
      summary_.failure.assign(failure.data(), failure.size());
    }
  } catch (...) {
    // This method is invoked from an HTTP completion callback. Never allow
    // diagnostics allocation or locking failures to unwind through curl.
  }
}

bool S3DirectReceiveRequestState::shouldRetryWithCallerBuffer() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_ == S3DirectReceiveMode::PREFERRED &&
      summary_.kernelTlsUnavailable && summary_.callerBufferAttempts == 0 &&
      !summary_.succeeded && summary_.receivedBodyBytes == 0;
}

S3DirectReceiveSummary S3DirectReceiveRequestState::summary() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return summary_;
}

bool s3DirectReceiveBuildSupported() {
  return VELOX_S3_DIRECT_RECEIVE_AVAILABLE;
}

bool s3DirectReceiveKernelTlsBuildSupported() {
  return VELOX_S3_DIRECT_RECEIVE_KTLS_AVAILABLE;
}

#if VELOX_S3_DIRECT_RECEIVE_AVAILABLE
namespace {

constexpr size_t kDiscardBufferSize = 64 * 1024;
// Keep enough of a rejected response for the AWS service error parser while
// preventing a proxy or service from growing the response stream without
// relation to the requested range. Bytes beyond the limit are consumed and
// accounted by curl, but are not retained.
constexpr size_t kMaxRetainedErrorBodyBytes = 1024 * 1024;

std::string_view trimOws(std::string_view value) {
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
    value.remove_prefix(1);
  }
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) {
    value.remove_suffix(1);
  }
  return value;
}

bool parseUint64(std::string_view value, uint64_t& result) {
  value = trimOws(value);
  const auto* begin = value.data();
  const auto* end = begin + value.size();
  const auto parsed = std::from_chars(begin, end, result);
  return parsed.ec == std::errc{} && parsed.ptr == end;
}

bool parseUint64(const Aws::String& value, uint64_t& result) {
  return parseUint64(std::string_view(value.data(), value.size()), result);
}

bool equalsIgnoreCase(std::string_view left, std::string_view right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (size_t i = 0; i < left.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(left[i])) !=
        std::tolower(static_cast<unsigned char>(right[i]))) {
      return false;
    }
  }
  return true;
}

bool parseContentRange(
    const Aws::String& value,
    uint64_t& first,
    uint64_t& last,
    uint64_t& completeLength) {
  auto range = trimOws(std::string_view(value.data(), value.size()));
  constexpr std::string_view kUnit{"bytes"};
  if (range.size() <= kUnit.size() ||
      !equalsIgnoreCase(range.substr(0, kUnit.size()), kUnit) ||
      (range[kUnit.size()] != ' ' && range[kUnit.size()] != '\t')) {
    return false;
  }
  range = trimOws(range.substr(kUnit.size()));
  const auto dash = range.find('-');
  const auto slash = range.find('/');
  if (dash == std::string_view::npos || slash == std::string_view::npos ||
      dash == 0 || slash <= dash + 1 || slash + 1 >= range.size() ||
      range.find('-', dash + 1) != std::string_view::npos ||
      range.find('/', slash + 1) != std::string_view::npos) {
    return false;
  }
  return parseUint64(range.substr(0, dash), first) &&
      parseUint64(range.substr(dash + 1, slash - dash - 1), last) &&
      parseUint64(range.substr(slash + 1), completeLength);
}

bool isIdentityEncoding(const Aws::String& value) {
  return equalsIgnoreCase(
      trimOws(std::string_view(value.data(), value.size())), "identity");
}

const Aws::String* findHeader(
    const Aws::Http::HeaderValueCollection& headers,
    std::string_view name) {
  for (const auto& [headerName, headerValue] : headers) {
    if (equalsIgnoreCase(
            std::string_view(headerName.data(), headerName.size()), name)) {
      return &headerValue;
    }
  }
  return nullptr;
}

class DirectS3ResponseStream final
    : private std::streambuf,
      public Aws::IOStream,
      public Aws::Http::DirectResponseReceiveStream {
 public:
  using int_type = std::streambuf::int_type;
  using off_type = std::streambuf::off_type;
  using pos_type = std::streambuf::pos_type;
  using traits_type = std::streambuf::traits_type;

  DirectS3ResponseStream(
      std::shared_ptr<S3DirectReceiveRequestState> state,
      std::vector<folly::Range<char*>> ranges,
      uint64_t offset,
      uint64_t objectSize)
      : std::streambuf(),
        Aws::IOStream(this),
        state_(std::move(state)),
        ranges_(std::move(ranges)),
        offset_(offset),
        objectSize_(objectSize),
        requireKernelTls_(state_->beginAttempt()),
        errorBody_(std::ios_base::in | std::ios_base::out) {
    for (const auto& range : ranges_) {
      VELOX_CHECK_LE(
          range.size(),
          std::numeric_limits<uint64_t>::max() - expectedBodyBytes_,
          "S3 direct receive range size overflow");
      expectedBodyBytes_ += range.size();
    }
    VELOX_CHECK_EQ(expectedBodyBytes_, state_->expectedBodyBytes());
    VELOX_CHECK_GT(expectedBodyBytes_, 0);
    VELOX_CHECK_LE(offset_, objectSize_);
    VELOX_CHECK_LE(expectedBodyBytes_, objectSize_ - offset_);
    skipExhaustedRanges();
  }

  bool RequireKernelTls() const noexcept override {
    return requireKernelTls_;
  }

  bool AcceptDirectResponse(
      Aws::Http::HttpResponseCode responseCode,
      const Aws::Http::HeaderValueCollection& headers) noexcept override {
    try {
      if (responseCode != Aws::Http::HttpResponseCode::PARTIAL_CONTENT) {
        reject("S3 direct receive requires HTTP 206");
        return false;
      }
      const auto* contentLengthHeader = findHeader(headers, "content-length");
      if (contentLengthHeader == nullptr) {
        reject("S3 direct receive response is missing Content-Length");
        return false;
      }
      uint64_t contentLength = 0;
      if (!parseUint64(*contentLengthHeader, contentLength) ||
          contentLength != expectedBodyBytes_) {
        reject("S3 direct receive response has an unexpected Content-Length");
        return false;
      }
      const auto* contentRangeHeader = findHeader(headers, "content-range");
      if (contentRangeHeader == nullptr) {
        reject("S3 direct receive response is missing Content-Range");
        return false;
      }
      uint64_t first = 0;
      uint64_t last = 0;
      uint64_t completeLength = 0;
      if (!parseContentRange(
              *contentRangeHeader, first, last, completeLength) ||
          first != offset_ || last != offset_ + expectedBodyBytes_ - 1 ||
          completeLength != objectSize_) {
        reject("S3 direct receive response has an unexpected Content-Range");
        return false;
      }
      if (findHeader(headers, "transfer-encoding") != nullptr) {
        reject("S3 direct receive does not accept Transfer-Encoding");
        return false;
      }
      if (findHeader(headers, "trailer") != nullptr) {
        reject("S3 direct receive does not accept trailers");
        return false;
      }
      const auto* contentEncodingHeader =
          findHeader(headers, "content-encoding");
      if (contentEncodingHeader != nullptr &&
          !isIdentityEncoding(*contentEncodingHeader)) {
        reject("S3 direct receive requires identity Content-Encoding");
        return false;
      }
      accepted_ = true;
      rejection_.clear();
      return true;
    } catch (...) {
      reject("S3 direct receive failed while validating response headers");
      return false;
    }
  }

  Aws::Http::DirectResponseReceiveBufferResult AcquireDirectResponseBuffer(
      size_t /* suggestedSize */,
      Aws::Http::DirectResponseReceiveBuffer& buffer) noexcept override {
    if (!accepted_ || completed_ || failed_ || loanOutstanding_) {
      fail("Invalid S3 direct receive buffer acquisition");
      return Aws::Http::DirectResponseReceiveBufferResult::ERROR;
    }
    skipExhaustedRanges();
    if (rangeIndex_ == ranges_.size()) {
      fail("S3 direct receive requested a buffer after the expected body");
      return Aws::Http::DirectResponseReceiveBufferResult::ERROR;
    }

    const auto& range = ranges_[rangeIndex_];
    const auto remaining = range.size() - rangeOffset_;
    loanIsDiscard_ = range.data() == nullptr;
    loanData_ = loanIsDiscard_
        ? discardBuffer_.data()
        : reinterpret_cast<unsigned char*>(range.data() + rangeOffset_);
    loanLength_ =
        loanIsDiscard_ ? std::min(remaining, discardBuffer_.size()) : remaining;
    loanCommitted_ = 0;
    loanOutstanding_ = true;
    buffer.data = loanData_;
    buffer.length = loanLength_;
    buffer.token = this;
    return Aws::Http::DirectResponseReceiveBufferResult::SUCCESS;
  }

  bool CommitDirectResponseBody(
      const Aws::Http::DirectResponseReceiveBuffer& buffer,
      const unsigned char* data,
      size_t length) noexcept override {
    if (!loanOutstanding_ || buffer.token != this || buffer.data != loanData_ ||
        buffer.length != loanLength_ || data != loanData_ + loanCommitted_ ||
        length > loanLength_ - loanCommitted_) {
      fail("Invalid S3 direct receive body commit");
      return false;
    }
    loanCommitted_ += length;
    directBodyBytes_ += length;
    if (loanIsDiscard_) {
      discardedBodyBytes_ += length;
    }
    return advance(length);
  }

  void ReleaseDirectResponseBuffer(
      const Aws::Http::DirectResponseReceiveBuffer& buffer,
      size_t used) noexcept override {
    if (!loanOutstanding_ || buffer.token != this || buffer.data != loanData_ ||
        buffer.length != loanLength_ || used != loanCommitted_) {
      fail("Invalid S3 direct receive buffer release");
    }
    loanOutstanding_ = false;
    loanData_ = nullptr;
    loanLength_ = 0;
    loanCommitted_ = 0;
    loanIsDiscard_ = false;
  }

  void CompleteDirectResponse(
      const Aws::Http::DirectResponseReceiveCompletion& completion) noexcept
      override {
    if (completed_) {
      fail("S3 direct receive completed more than once");
      return;
    }
    completed_ = true;
    if (loanOutstanding_) {
      fail("S3 direct receive completed with an outstanding buffer");
    }

    const bool unavailable = completion.result ==
            Aws::Http::DirectResponseReceiveCompletionResult::UNAVAILABLE &&
        completion.bodyBytes == 0 && completion.directBodyBytes == 0;
    bool success = completion.result ==
            Aws::Http::DirectResponseReceiveCompletionResult::SUCCESS &&
        accepted_ && !failed_ && position_ == expectedBodyBytes_ &&
        completion.bodyBytes == expectedBodyBytes_ &&
        completion.directBodyBytes == directBodyBytes_ &&
        directBodyBytes_ + copiedBodyBytes_ == expectedBodyBytes_;
    std::string_view failure = failure_;
    if (!success && !unavailable && failure.empty()) {
      failure = !rejection_.empty()
          ? std::string_view(rejection_)
          : std::string_view(
                "S3 direct receive did not complete the exact response body");
    }
    auto result = S3DirectReceiveAttemptResult::FAILED;
    if (success) {
      result = S3DirectReceiveAttemptResult::SUCCESS;
    } else if (unavailable) {
      result = S3DirectReceiveAttemptResult::UNAVAILABLE;
    } else if (
        completion.result ==
        Aws::Http::DirectResponseReceiveCompletionResult::
            RESPONSE_NOT_ACCEPTED) {
      result = S3DirectReceiveAttemptResult::RESPONSE_REJECTED;
    }
    state_->finishAttempt(
        requireKernelTls_,
        result,
        completion.bodyBytes,
        completion.directBodyBytes,
        copiedBodyBytes_,
        discardedBodyBytes_,
        failure);
  }

 protected:
  std::streamsize xsputn(const char* source, std::streamsize count) override {
    if (count <= 0) {
      return 0;
    }
    if (!accepted_) {
      const auto requested = static_cast<size_t>(count);
      const auto retained = std::min(
          requested, kMaxRetainedErrorBodyBytes - retainedErrorBodyBytes_);
      if (retained > 0) {
        const auto written =
            errorBody_.sputn(source, static_cast<std::streamsize>(retained));
        if (written != static_cast<std::streamsize>(retained)) {
          return written;
        }
        retainedErrorBodyBytes_ += retained;
      }
      return count;
    }

    const auto requested = static_cast<size_t>(count);
    const auto written = copyAndAdvance(source, requested);
    copiedBodyBytes_ += written;
    return static_cast<std::streamsize>(written);
  }

  int_type overflow(int_type value) override {
    if (traits_type::eq_int_type(value, traits_type::eof())) {
      return traits_type::not_eof(value);
    }
    const auto byte = traits_type::to_char_type(value);
    return xsputn(&byte, 1) == 1 ? value : traits_type::eof();
  }

  std::streamsize xsgetn(char* destination, std::streamsize count) override {
    return accepted_ ? 0 : errorBody_.sgetn(destination, count);
  }

  int_type underflow() override {
    return accepted_ ? traits_type::eof() : errorBody_.sgetc();
  }

  int_type uflow() override {
    return accepted_ ? traits_type::eof() : errorBody_.sbumpc();
  }

  int_type pbackfail(int_type value) override {
    if (accepted_) {
      return traits_type::eof();
    }
    return traits_type::eq_int_type(value, traits_type::eof())
        ? errorBody_.sungetc()
        : errorBody_.sputbackc(traits_type::to_char_type(value));
  }

  pos_type seekoff(
      off_type offset,
      std::ios_base::seekdir direction,
      std::ios_base::openmode mode) override {
    if (!accepted_) {
      return errorBody_.pubseekoff(offset, direction, mode);
    }
    if ((mode & std::ios_base::out) == 0) {
      return pos_type(off_type(-1));
    }
    return seekBody(offset, direction);
  }

  pos_type seekpos(pos_type position, std::ios_base::openmode mode) override {
    if (!accepted_) {
      return errorBody_.pubseekpos(position, mode);
    }
    if ((mode & std::ios_base::out) == 0) {
      return pos_type(off_type(-1));
    }
    const auto target = static_cast<off_type>(position);
    if (target < 0 || !setPosition(static_cast<uint64_t>(target))) {
      return pos_type(off_type(-1));
    }
    return position;
  }

  int sync() override {
    return accepted_ ? 0 : errorBody_.pubsync();
  }

 private:
  void reject(const char* message) noexcept {
    try {
      rejection_ = message;
    } catch (...) {
    }
  }

  void fail(const char* message) noexcept {
    failed_ = true;
    try {
      if (failure_.empty()) {
        failure_ = message;
      }
    } catch (...) {
    }
  }

  void skipExhaustedRanges() noexcept {
    while (rangeIndex_ < ranges_.size() &&
           rangeOffset_ == ranges_[rangeIndex_].size()) {
      ++rangeIndex_;
      rangeOffset_ = 0;
    }
  }

  bool advance(size_t length) noexcept {
    if (length > expectedBodyBytes_ - position_) {
      fail("S3 direct receive exceeded the expected body length");
      return false;
    }
    auto remaining = length;
    while (remaining > 0) {
      skipExhaustedRanges();
      if (rangeIndex_ == ranges_.size()) {
        fail("S3 direct receive exceeded its destination ranges");
        return false;
      }
      const auto available = ranges_[rangeIndex_].size() - rangeOffset_;
      const auto step = std::min(remaining, available);
      rangeOffset_ += step;
      position_ += step;
      remaining -= step;
    }
    skipExhaustedRanges();
    return true;
  }

  size_t copyAndAdvance(const char* source, size_t length) noexcept {
    const auto writable = static_cast<size_t>(
        std::min<uint64_t>(length, expectedBodyBytes_ - position_));
    auto remaining = writable;
    while (remaining > 0) {
      skipExhaustedRanges();
      if (rangeIndex_ == ranges_.size()) {
        fail("S3 response exceeded its destination ranges");
        break;
      }
      const auto& range = ranges_[rangeIndex_];
      const auto step = std::min(remaining, range.size() - rangeOffset_);
      if (range.data() != nullptr) {
        std::memcpy(range.data() + rangeOffset_, source, step);
      } else {
        discardedBodyBytes_ += step;
      }
      source += step;
      rangeOffset_ += step;
      position_ += step;
      remaining -= step;
    }
    skipExhaustedRanges();
    if (writable != length) {
      fail("S3 response exceeded the expected body length");
    }
    return writable - remaining;
  }

  bool setPosition(uint64_t position) noexcept {
    if (position > expectedBodyBytes_ || loanOutstanding_) {
      return false;
    }
    position_ = position;
    rangeIndex_ = 0;
    rangeOffset_ = static_cast<size_t>(position);
    while (rangeIndex_ < ranges_.size() &&
           rangeOffset_ >= ranges_[rangeIndex_].size()) {
      rangeOffset_ -= ranges_[rangeIndex_].size();
      ++rangeIndex_;
    }
    skipExhaustedRanges();
    return true;
  }

  pos_type seekBody(off_type offset, std::ios_base::seekdir direction) {
    uint64_t base = 0;
    if (direction == std::ios_base::beg) {
      base = 0;
    } else if (direction == std::ios_base::cur) {
      base = position_;
    } else if (direction == std::ios_base::end) {
      base = expectedBodyBytes_;
    } else {
      return pos_type(off_type(-1));
    }
    uint64_t target = 0;
    if (offset >= 0) {
      const auto forward = static_cast<uint64_t>(offset);
      if (forward > expectedBodyBytes_ - base) {
        return pos_type(off_type(-1));
      }
      target = base + forward;
    } else {
      const auto backward = static_cast<uint64_t>(-(offset + 1)) + uint64_t{1};
      if (backward > base) {
        return pos_type(off_type(-1));
      }
      target = base - backward;
    }
    if (!setPosition(target)) {
      return pos_type(off_type(-1));
    }
    return pos_type(static_cast<off_type>(target));
  }

  const std::shared_ptr<S3DirectReceiveRequestState> state_;
  const std::vector<folly::Range<char*>> ranges_;
  const uint64_t offset_;
  const uint64_t objectSize_;
  const bool requireKernelTls_;
  std::stringbuf errorBody_;
  size_t retainedErrorBodyBytes_{0};
  std::array<unsigned char, kDiscardBufferSize> discardBuffer_{};
  uint64_t expectedBodyBytes_{0};
  uint64_t position_{0};
  size_t rangeIndex_{0};
  size_t rangeOffset_{0};
  unsigned char* loanData_{nullptr};
  size_t loanLength_{0};
  size_t loanCommitted_{0};
  uint64_t directBodyBytes_{0};
  uint64_t copiedBodyBytes_{0};
  uint64_t discardedBodyBytes_{0};
  bool accepted_{false};
  bool completed_{false};
  bool failed_{false};
  bool loanOutstanding_{false};
  bool loanIsDiscard_{false};
  std::string rejection_;
  std::string failure_;
};

} // namespace
#endif

Aws::IOStream* makeS3DirectReceiveStream(
    const char* allocationTag,
    std::shared_ptr<S3DirectReceiveRequestState> state,
    std::vector<folly::Range<char*>> ranges,
    uint64_t offset,
    uint64_t objectSize) {
#if VELOX_S3_DIRECT_RECEIVE_AVAILABLE
  return Aws::New<DirectS3ResponseStream>(
      allocationTag, std::move(state), std::move(ranges), offset, objectSize);
#else
  VELOX_FAIL(
      "S3 direct receive requires an AWS SDK built with "
      "DirectResponseReceiveStream support");
#endif
}

} // namespace facebook::velox::filesystems
