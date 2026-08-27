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

#include <gtest/gtest.h>

#if VELOX_S3_DIRECT_RECEIVE_AVAILABLE
#include <aws/core/http/DirectResponseReceiveStream.h>
#include <aws/core/http/standard/StandardHttpRequest.h>
#include <aws/core/http/standard/StandardHttpResponse.h>
#include <aws/core/utils/memory/AWSMemory.h>
#include <aws/core/utils/stream/ResponseStream.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace facebook::velox::filesystems {
namespace {

using Aws::Http::DirectResponseReceiveBuffer;
using Aws::Http::DirectResponseReceiveBufferResult;
using Aws::Http::DirectResponseReceiveCompletion;
using Aws::Http::DirectResponseReceiveCompletionResult;
using Aws::Http::DirectResponseReceiveStream;

using AwsStreamPtr = std::unique_ptr<Aws::IOStream, void (*)(Aws::IOStream*)>;

AwsStreamPtr makeStream(
    const std::shared_ptr<S3DirectReceiveRequestState>& state,
    std::vector<folly::Range<char*>> ranges,
    uint64_t offset,
    uint64_t objectSize) {
  return AwsStreamPtr(
      makeS3DirectReceiveStream(
          "S3DirectReceiveTest", state, std::move(ranges), offset, objectSize),
      [](Aws::IOStream* stream) { Aws::Delete(stream); });
}

std::shared_ptr<Aws::Http::HttpRequest> makeRequest() {
  auto request = Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>(
      "S3DirectReceiveTest",
      Aws::Http::URI("http://127.0.0.1/object"),
      Aws::Http::HttpMethod::HTTP_GET);
  request->SetResponseStreamFactory(
      Aws::Utils::Stream::DefaultResponseStreamFactoryMethod);
  return request;
}

std::shared_ptr<Aws::Http::Standard::StandardHttpResponse>
makePartialResponse(uint64_t offset, uint64_t length, uint64_t objectSize) {
  auto response = std::make_shared<Aws::Http::Standard::StandardHttpResponse>(
      makeRequest());
  response->SetResponseCode(Aws::Http::HttpResponseCode::PARTIAL_CONTENT);
  response->AddHeader("content-length", std::to_string(length).c_str());
  response->AddHeader(
      "content-range",
      fmt::format("bytes {}-{}/{}", offset, offset + length - 1, objectSize)
          .c_str());
  return response;
}

bool acceptResponse(
    DirectResponseReceiveStream* direct,
    const Aws::Http::HttpResponse& response) {
  return direct->AcceptDirectResponse(
      response.GetResponseCode(), response.GetHeaders());
}

std::vector<unsigned char> makeBody(size_t size) {
  std::vector<unsigned char> body(size);
  for (size_t i = 0; i < size; ++i) {
    body[i] = static_cast<unsigned char>((i * 193 + 29) & 0xff);
  }
  return body;
}

TEST(S3DirectReceiveTest, reportsIndependentBuildCapabilities) {
  EXPECT_TRUE(s3DirectReceiveBuildSupported());
  EXPECT_EQ(
      s3DirectReceiveKernelTlsBuildSupported(),
      static_cast<bool>(VELOX_S3_DIRECT_RECEIVE_KTLS_AVAILABLE));
}

TEST(S3DirectReceiveTest, denseAndScatterPlacement) {
  constexpr uint64_t kOffset = 100;
  constexpr size_t kGap = 128 * 1024 + 17;
  std::array<char, 7> first{};
  std::array<char, 11> second{};
  const uint64_t length = first.size() + kGap + second.size();
  constexpr uint64_t kObjectSize = 1024 * 1024;
  auto state = std::make_shared<S3DirectReceiveRequestState>(
      S3DirectReceiveMode::CALLER_BUFFER, length);
  auto stream = makeStream(
      state,
      {
          folly::Range<char*>(first.data(), first.size()),
          folly::Range<char*>(static_cast<char*>(nullptr), kGap),
          folly::Range<char*>(second.data(), second.size()),
      },
      kOffset,
      kObjectSize);
  auto* direct = dynamic_cast<DirectResponseReceiveStream*>(stream.get());
  ASSERT_NE(direct, nullptr);
  EXPECT_FALSE(direct->RequireKernelTls());

  auto response = makePartialResponse(kOffset, length, kObjectSize);
  ASSERT_TRUE(acceptResponse(direct, *response));
  const auto body = makeBody(length);

  // Models body bytes sharing curl's final header scratch loan.
  constexpr size_t kCopiedPrefix = 13;
  stream->write(reinterpret_cast<const char*>(body.data()), kCopiedPrefix);
  ASSERT_TRUE(stream->good());

  size_t position = kCopiedPrefix;
  size_t directBytes = 0;
  while (position < body.size()) {
    DirectResponseReceiveBuffer buffer;
    ASSERT_EQ(
        DirectResponseReceiveBufferResult::SUCCESS,
        direct->AcquireDirectResponseBuffer(body.size() - position, buffer));
    const auto bytes = std::min(buffer.length, body.size() - position);
    std::memcpy(buffer.data, body.data() + position, bytes);
    ASSERT_TRUE(direct->CommitDirectResponseBody(buffer, buffer.data, bytes));
    direct->ReleaseDirectResponseBuffer(buffer, bytes);
    position += bytes;
    directBytes += bytes;
  }
  direct->CompleteDirectResponse(DirectResponseReceiveCompletion(
      DirectResponseReceiveCompletionResult::SUCCESS,
      body.size(),
      directBytes));

  EXPECT_TRUE(
      std::equal(
          first.begin(),
          first.end(),
          reinterpret_cast<const char*>(body.data())));
  EXPECT_TRUE(
      std::equal(
          second.begin(),
          second.end(),
          reinterpret_cast<const char*>(body.data() + first.size() + kGap)));
  const auto summary = state->summary();
  EXPECT_TRUE(summary.succeeded) << summary.failure;
  EXPECT_EQ(body.size(), summary.receivedBodyBytes);
  EXPECT_EQ(directBytes, summary.receivedDirectBodyBytes);
  EXPECT_EQ(kCopiedPrefix, summary.receivedCopiedBodyBytes);
  EXPECT_EQ(kGap, summary.receivedDiscardedBodyBytes);
  EXPECT_EQ(body.size(), summary.successfulBodyBytes);
  EXPECT_EQ(directBytes, summary.successfulDirectBodyBytes);
  EXPECT_EQ(kCopiedPrefix, summary.successfulCopiedBodyBytes);
  EXPECT_EQ(kGap, summary.successfulDiscardedBodyBytes);
}

TEST(S3DirectReceiveTest, rejectedResponseBodyRemainsReadable) {
  std::array<char, 64> destination;
  destination.fill('?');
  auto state = std::make_shared<S3DirectReceiveRequestState>(
      S3DirectReceiveMode::CALLER_BUFFER, destination.size());
  auto stream = makeStream(
      state,
      {folly::Range<char*>(destination.data(), destination.size())},
      0,
      destination.size());
  auto* direct = dynamic_cast<DirectResponseReceiveStream*>(stream.get());
  ASSERT_NE(direct, nullptr);

  Aws::Http::Standard::StandardHttpResponse response(makeRequest());
  response.SetResponseCode(Aws::Http::HttpResponseCode::NOT_FOUND);
  const std::string errorBody =
      "<Error><Code>NoSuchKey</Code><Message>missing</Message></Error>";
  response.AddHeader(
      "content-length", std::to_string(errorBody.size()).c_str());
  ASSERT_FALSE(acceptResponse(direct, response));
  stream->write(errorBody.data(), errorBody.size());
  ASSERT_TRUE(stream->good());
  direct->CompleteDirectResponse(DirectResponseReceiveCompletion(
      DirectResponseReceiveCompletionResult::RESPONSE_NOT_ACCEPTED,
      errorBody.size(),
      0));

  EXPECT_EQ(
      static_cast<std::streamoff>(errorBody.size()),
      static_cast<std::streamoff>(stream->tellp()));
  stream->seekg(0);
  const std::string recovered{
      std::istreambuf_iterator<char>(*stream),
      std::istreambuf_iterator<char>()};
  EXPECT_EQ(errorBody, recovered);
  EXPECT_TRUE(
      std::all_of(destination.begin(), destination.end(), [](char value) {
        return value == '?';
      }));
  const auto summary = state->summary();
  EXPECT_FALSE(summary.succeeded);
  EXPECT_EQ(1, summary.responseRejectedAttempts);
  EXPECT_EQ(errorBody.size(), summary.receivedBodyBytes);
  EXPECT_EQ(0, summary.successfulBodyBytes);
  EXPECT_NE(summary.failure.find("HTTP 206"), std::string::npos);
}

TEST(S3DirectReceiveTest, rejectedResponseBodyRetentionIsBounded) {
  constexpr size_t kRetainedLimit = 1024 * 1024;
  std::array<char, 64> destination;
  destination.fill('?');
  auto state = std::make_shared<S3DirectReceiveRequestState>(
      S3DirectReceiveMode::CALLER_BUFFER, destination.size());
  auto stream = makeStream(
      state,
      {folly::Range<char*>(destination.data(), destination.size())},
      0,
      destination.size());
  auto* direct = dynamic_cast<DirectResponseReceiveStream*>(stream.get());
  ASSERT_NE(direct, nullptr);

  Aws::Http::Standard::StandardHttpResponse response(makeRequest());
  response.SetResponseCode(Aws::Http::HttpResponseCode::BAD_REQUEST);
  const std::string errorBody(kRetainedLimit + 4096, 'x');
  response.AddHeader(
      "content-length", std::to_string(errorBody.size()).c_str());
  ASSERT_FALSE(acceptResponse(direct, response));
  stream->write(errorBody.data(), errorBody.size());
  ASSERT_TRUE(stream->good());
  direct->CompleteDirectResponse(DirectResponseReceiveCompletion(
      DirectResponseReceiveCompletionResult::RESPONSE_NOT_ACCEPTED,
      errorBody.size(),
      0));

  EXPECT_EQ(
      static_cast<std::streamoff>(kRetainedLimit),
      static_cast<std::streamoff>(stream->tellp()));
  stream->seekg(0);
  const std::string recovered{
      std::istreambuf_iterator<char>(*stream),
      std::istreambuf_iterator<char>()};
  EXPECT_EQ(kRetainedLimit, recovered.size());
  EXPECT_TRUE(
      std::all_of(destination.begin(), destination.end(), [](char value) {
        return value == '?';
      }));
  const auto summary = state->summary();
  EXPECT_FALSE(summary.succeeded);
  EXPECT_EQ(1, summary.responseRejectedAttempts);
  EXPECT_EQ(errorBody.size(), summary.receivedBodyBytes);
}

TEST(S3DirectReceiveTest, preferredFallsBackOnlyAfterEmptyUnavailableAttempt) {
  std::array<char, 4> destination{};
  auto state = std::make_shared<S3DirectReceiveRequestState>(
      S3DirectReceiveMode::PREFERRED, destination.size());
  {
    auto stream = makeStream(
        state,
        {folly::Range<char*>(destination.data(), destination.size())},
        0,
        destination.size());
    auto* direct = dynamic_cast<DirectResponseReceiveStream*>(stream.get());
    ASSERT_NE(direct, nullptr);
    EXPECT_TRUE(direct->RequireKernelTls());
    direct->CompleteDirectResponse(DirectResponseReceiveCompletion(
        DirectResponseReceiveCompletionResult::UNAVAILABLE, 0, 0));
  }
  ASSERT_TRUE(state->shouldRetryWithCallerBuffer());

  auto stream = makeStream(
      state,
      {folly::Range<char*>(destination.data(), destination.size())},
      0,
      destination.size());
  auto* direct = dynamic_cast<DirectResponseReceiveStream*>(stream.get());
  ASSERT_NE(direct, nullptr);
  EXPECT_FALSE(direct->RequireKernelTls());
  EXPECT_FALSE(state->shouldRetryWithCallerBuffer());
  direct->CompleteDirectResponse(DirectResponseReceiveCompletion(
      DirectResponseReceiveCompletionResult::FAILED, 0, 0));
  const auto summary = state->summary();
  EXPECT_EQ(2, summary.attempts);
  EXPECT_EQ(1, summary.kernelTlsAttempts);
  EXPECT_EQ(1, summary.callerBufferAttempts);
  EXPECT_EQ(1, summary.unavailableAttempts);
  EXPECT_EQ(1, summary.kernelTlsUnavailableAttempts);
  EXPECT_EQ(1, summary.failedAttempts);
}

TEST(
    S3DirectReceiveTest,
    doesNotMislabelCallerBufferUnavailabilityAsKernelTls) {
  std::array<char, 4> destination{};
  auto state = std::make_shared<S3DirectReceiveRequestState>(
      S3DirectReceiveMode::CALLER_BUFFER, destination.size());
  auto stream = makeStream(
      state,
      {folly::Range<char*>(destination.data(), destination.size())},
      0,
      destination.size());
  auto* direct = dynamic_cast<DirectResponseReceiveStream*>(stream.get());
  ASSERT_NE(direct, nullptr);
  EXPECT_FALSE(direct->RequireKernelTls());
  direct->CompleteDirectResponse(DirectResponseReceiveCompletion(
      DirectResponseReceiveCompletionResult::UNAVAILABLE, 0, 0));

  const auto summary = state->summary();
  EXPECT_EQ(1, summary.unavailableAttempts);
  EXPECT_EQ(0, summary.kernelTlsUnavailableAttempts);
}

TEST(S3DirectReceiveTest, accountsRetriedBytesAndFinalSuccessfulPlacement) {
  std::array<char, 4> destination{};
  auto state = std::make_shared<S3DirectReceiveRequestState>(
      S3DirectReceiveMode::CALLER_BUFFER, destination.size());
  {
    auto stream = makeStream(
        state,
        {folly::Range<char*>(destination.data(), destination.size())},
        0,
        destination.size());
    auto* direct = dynamic_cast<DirectResponseReceiveStream*>(stream.get());
    ASSERT_NE(direct, nullptr);
    ASSERT_TRUE(acceptResponse(direct, *makePartialResponse(0, 4, 4)));
    stream->write("ab", 2);
    direct->CompleteDirectResponse(DirectResponseReceiveCompletion(
        DirectResponseReceiveCompletionResult::FAILED, 2, 0));
  }
  {
    auto stream = makeStream(
        state,
        {folly::Range<char*>(destination.data(), destination.size())},
        0,
        destination.size());
    auto* direct = dynamic_cast<DirectResponseReceiveStream*>(stream.get());
    ASSERT_NE(direct, nullptr);
    ASSERT_TRUE(acceptResponse(direct, *makePartialResponse(0, 4, 4)));
    stream->write("wxyz", 4);
    direct->CompleteDirectResponse(DirectResponseReceiveCompletion(
        DirectResponseReceiveCompletionResult::SUCCESS, 4, 0));
  }

  const auto summary = state->summary();
  EXPECT_TRUE(summary.succeeded) << summary.failure;
  EXPECT_EQ(2, summary.attempts);
  EXPECT_EQ(1, summary.failedAttempts);
  EXPECT_EQ(6, summary.receivedBodyBytes);
  EXPECT_EQ(6, summary.receivedCopiedBodyBytes);
  EXPECT_EQ(4, summary.successfulBodyBytes);
  EXPECT_EQ(4, summary.successfulCopiedBodyBytes);
  EXPECT_EQ("wxyz", std::string(destination.data(), destination.size()));
}

TEST(S3DirectReceiveTest, preferredDoesNotFallbackAfterAnyBodyByte) {
  std::array<char, 4> destination{};
  auto state = std::make_shared<S3DirectReceiveRequestState>(
      S3DirectReceiveMode::PREFERRED, destination.size());
  auto stream = makeStream(
      state,
      {folly::Range<char*>(destination.data(), destination.size())},
      0,
      destination.size());
  auto* direct = dynamic_cast<DirectResponseReceiveStream*>(stream.get());
  ASSERT_NE(direct, nullptr);
  EXPECT_TRUE(direct->RequireKernelTls());
  direct->CompleteDirectResponse(DirectResponseReceiveCompletion(
      DirectResponseReceiveCompletionResult::UNAVAILABLE, 1, 0));

  EXPECT_FALSE(state->shouldRetryWithCallerBuffer());
  const auto summary = state->summary();
  EXPECT_FALSE(summary.kernelTlsUnavailable);
  EXPECT_EQ(0, summary.unavailableAttempts);
  EXPECT_EQ(1, summary.failedAttempts);
  EXPECT_EQ(1, summary.receivedBodyBytes);
}

TEST(S3DirectReceiveTest, acceptsEquivalentContentRangeFormatting) {
  std::array<char, 8> destination{};
  auto state = std::make_shared<S3DirectReceiveRequestState>(
      S3DirectReceiveMode::CALLER_BUFFER, destination.size());
  auto stream = makeStream(
      state,
      {folly::Range<char*>(destination.data(), destination.size())},
      10,
      100);
  auto* direct = dynamic_cast<DirectResponseReceiveStream*>(stream.get());
  ASSERT_NE(direct, nullptr);

  Aws::Http::Standard::StandardHttpResponse response(makeRequest());
  response.SetResponseCode(Aws::Http::HttpResponseCode::PARTIAL_CONTENT);
  response.AddHeader("content-length", "\t8 ");
  response.AddHeader("content-range", " BYTES\t10 - 17 / 100\t");
  EXPECT_TRUE(acceptResponse(direct, response));
  direct->CompleteDirectResponse(DirectResponseReceiveCompletion(
      DirectResponseReceiveCompletionResult::FAILED, 0, 0));
}

TEST(S3DirectReceiveTest, rejectsMismatchedRangeAndEncoding) {
  std::array<char, 8> destination{};
  auto state = std::make_shared<S3DirectReceiveRequestState>(
      S3DirectReceiveMode::REQUIRED, destination.size());
  auto stream = makeStream(
      state,
      {folly::Range<char*>(destination.data(), destination.size())},
      10,
      100);
  auto* direct = dynamic_cast<DirectResponseReceiveStream*>(stream.get());
  ASSERT_NE(direct, nullptr);

  auto response = makePartialResponse(10, destination.size(), 100);
  response->AddHeader("content-encoding", "gzip");
  EXPECT_FALSE(acceptResponse(direct, *response));
  DirectResponseReceiveBuffer buffer;
  EXPECT_EQ(
      DirectResponseReceiveBufferResult::ERROR,
      direct->AcquireDirectResponseBuffer(destination.size(), buffer));
  direct->CompleteDirectResponse(DirectResponseReceiveCompletion(
      DirectResponseReceiveCompletionResult::FAILED, 0, 0));
}

} // namespace
} // namespace facebook::velox::filesystems

#endif
