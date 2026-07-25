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

#include "velox/experimental/ucx-exchange/CpuRowPayloadCopy.h"

#include <gtest/gtest.h>
#include <array>
#include <stdexcept>
#include <string_view>

namespace facebook::velox::ucx_exchange {
namespace {

std::unique_ptr<folly::IOBuf> makeSeventeenByteChain() {
  auto payload = folly::IOBuf::copyBuffer("12345678");
  payload->appendToChain(folly::IOBuf::copyBuffer("abcdefghi"));
  return payload;
}

TEST(CpuRowPayloadCopyTest, copiesExactAdvertisedSize) {
  auto payload = makeSeventeenByteChain();
  std::array<uint8_t, 17> destination{};

  copyCpuRowPayloads({payload.get()}, destination);

  EXPECT_EQ(
      std::string_view(
          reinterpret_cast<const char*>(destination.data()),
          destination.size()),
      "12345678abcdefghi");
}

TEST(CpuRowPayloadCopyTest, rejectsPayloadLargerThanAdvertisedSize) {
  auto payload = makeSeventeenByteChain();
  std::array<uint8_t, 17> storage;
  storage.fill(0xa5);

  EXPECT_THROW(
      copyCpuRowPayloads(
          {payload.get()}, std::span<uint8_t>(storage.data(), 16)),
      std::length_error);
  EXPECT_EQ(storage.back(), 0xa5);
}

TEST(CpuRowPayloadCopyTest, rejectsPayloadSmallerThanAdvertisedSize) {
  auto payload = makeSeventeenByteChain();
  std::array<uint8_t, 18> destination{};

  EXPECT_THROW(
      copyCpuRowPayloads({payload.get()}, destination), std::length_error);
}

} // namespace
} // namespace facebook::velox::ucx_exchange
