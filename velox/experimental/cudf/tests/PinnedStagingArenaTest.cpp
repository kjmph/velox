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

#include "velox/experimental/cudf/connectors/hive/PinnedStagingArena.h"

#include "velox/common/base/Exceptions.h"

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstring>
#include <future>
#include <limits>
#include <thread>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive::test {
namespace {

using namespace std::chrono_literals;

class PinnedStagingArenaTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  }

  void TearDown() override {
    PinnedStagingArena::setAllocationFailureForTesting(false);
  }
};

TEST_F(PinnedStagingArenaTest, disabled) {
  PinnedStagingArena::configure(false, 0, 0);
  EXPECT_FALSE(PinnedStagingArena::enabled());
  EXPECT_FALSE(PinnedStagingArena::acquirePair().has_value());
}

TEST_F(PinnedStagingArenaTest, validatesEnabledConfiguration) {
  EXPECT_THROW(PinnedStagingArena::configure(true, 0, 1), VeloxException);
  EXPECT_THROW(PinnedStagingArena::configure(true, 4096, 0), VeloxException);
}

TEST_F(PinnedStagingArenaTest, acquiresBothWindowsAtomically) {
  constexpr uint64_t kWindowBytes = 4096;
  PinnedStagingArena::configure(true, kWindowBytes, 2);
  EXPECT_TRUE(PinnedStagingArena::enabled());

  auto first = PinnedStagingArena::acquirePair();
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->capacity(), kWindowBytes);
  EXPECT_NE(first->data(0), first->data(1));
  auto* firstWindow = first->data(0);
  auto* secondWindow = first->data(1);

  std::promise<void> waiterStarted;
  auto waiter = std::async(std::launch::async, [&waiterStarted]() {
    waiterStarted.set_value();
    return PinnedStagingArena::acquirePair();
  });
  waiterStarted.get_future().wait();
  EXPECT_EQ(waiter.wait_for(50ms), std::future_status::timeout);

  first->release();
  auto second = waiter.get();
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->data(0), firstWindow);
  EXPECT_EQ(second->data(1), secondWindow);
}

TEST_F(PinnedStagingArenaTest, moveAndReleaseWindowSet) {
  PinnedStagingArena::configure(true, 4096, 2);
  auto first = PinnedStagingArena::acquirePair();
  ASSERT_TRUE(first.has_value());

  auto* firstWindow = first->data(0);
  auto* secondWindow = first->data(1);
  PinnedStagingArena::WindowSetLease moved = std::move(*first);
  EXPECT_FALSE(static_cast<bool>(*first));
  ASSERT_TRUE(static_cast<bool>(moved));
  EXPECT_EQ(moved.data(0), firstWindow);
  EXPECT_EQ(moved.data(1), secondWindow);

  moved.release();
  auto replacement = PinnedStagingArena::acquirePair();
  ASSERT_TRUE(replacement.has_value());
  EXPECT_EQ(replacement->data(0), firstWindow);
  EXPECT_EQ(replacement->data(1), secondWindow);
}

TEST_F(PinnedStagingArenaTest, packsBothWindowsInParallel) {
  constexpr uint64_t kWindowBytes =
      2 * PinnedStagingArena::kPackQuantumBytes + 17;
  PinnedStagingArena::configure(true, kWindowBytes, 4);
  auto lease = PinnedStagingArena::acquirePair();
  ASSERT_TRUE(lease.has_value());

  std::vector<std::vector<uint8_t>> scatterSources;
  scatterSources.reserve(32);
  std::vector<PinnedStagingArena::Copy> scatterCopies;
  scatterCopies.reserve(32);
  constexpr uint64_t kCopyBytes = 128;
  for (uint64_t i = 0; i < 32; ++i) {
    scatterSources.emplace_back(kCopyBytes, static_cast<uint8_t>(i + 1));
    scatterCopies.push_back(
        PinnedStagingArena::Copy{
            scatterSources.back().data(), i * kCopyBytes, kCopyBytes});
  }

  std::vector<uint8_t> largeSource(kWindowBytes);
  for (uint64_t i = 0; i < largeSource.size(); ++i) {
    largeSource[i] = static_cast<uint8_t>(i);
  }
  const std::array largeCopy = {
      PinnedStagingArena::Copy{largeSource.data(), 0, largeSource.size()}};

  auto firstPack =
      std::async(std::launch::async, [&]() { lease->pack(0, scatterCopies); });
  auto secondPack =
      std::async(std::launch::async, [&]() { lease->pack(1, largeCopy); });
  firstPack.get();
  secondPack.get();

  for (uint64_t i = 0; i < scatterSources.size(); ++i) {
    EXPECT_EQ(
        std::memcmp(
            lease->data(0) + i * kCopyBytes,
            scatterSources[i].data(),
            kCopyBytes),
        0);
  }
  EXPECT_EQ(
      std::memcmp(lease->data(1), largeSource.data(), largeSource.size()), 0);
}

TEST_F(PinnedStagingArenaTest, validatesAllRangesBeforeWriting) {
  constexpr uint64_t kWindowBytes = 4096;
  PinnedStagingArena::configure(true, kWindowBytes, 2);
  auto lease = PinnedStagingArena::acquirePair();
  ASSERT_TRUE(lease.has_value());
  std::memset(lease->data(0), 0x5a, kWindowBytes);

  std::vector<uint8_t> source(128, 0xa5);
  const std::array<PinnedStagingArena::Copy, 2> outOfBounds = {
      PinnedStagingArena::Copy{source.data(), 0, source.size()},
      PinnedStagingArena::Copy{source.data(), kWindowBytes - 1, 2}};
  EXPECT_THROW(lease->pack(0, outOfBounds), VeloxException);
  EXPECT_EQ(lease->data(0)[0], 0x5a);

  const std::array<PinnedStagingArena::Copy, 2> overlapping = {
      PinnedStagingArena::Copy{source.data(), 0, source.size()},
      PinnedStagingArena::Copy{source.data(), 64, source.size()}};
  EXPECT_THROW(lease->pack(0, overlapping), VeloxException);
  EXPECT_EQ(lease->data(0)[0], 0x5a);

  const std::array<PinnedStagingArena::Copy, 1> overflow = {
      PinnedStagingArena::Copy{
          source.data(),
          kWindowBytes - 1,
          std::numeric_limits<uint64_t>::max()}};
  EXPECT_THROW(lease->pack(0, overflow), VeloxException);
  EXPECT_EQ(lease->data(0)[0], 0x5a);

  const std::array<PinnedStagingArena::Copy, 1> nullSource = {
      PinnedStagingArena::Copy{nullptr, 0, 1}};
  EXPECT_THROW(lease->pack(0, nullSource), VeloxException);
  EXPECT_EQ(lease->data(0)[0], 0x5a);

  EXPECT_THROW(lease->data(PinnedStagingArena::kWindowCount), VeloxException);
  EXPECT_THROW(
      lease->pack(
          PinnedStagingArena::kWindowCount,
          std::span<const PinnedStagingArena::Copy>{}),
      VeloxException);
}

TEST_F(PinnedStagingArenaTest, resetInterruptsWaiterButPreservesWindowSet) {
  PinnedStagingArena::configure(true, 4096, 2);
  auto first = PinnedStagingArena::acquirePair();
  ASSERT_TRUE(first.has_value());

  std::promise<void> waiterStarted;
  auto waiter = std::async(std::launch::async, [&waiterStarted]() {
    waiterStarted.set_value();
    return PinnedStagingArena::acquirePair();
  });
  waiterStarted.get_future().wait();
  EXPECT_EQ(waiter.wait_for(50ms), std::future_status::timeout);

  PinnedStagingArena::reset();
  EXPECT_FALSE(waiter.get().has_value());

  std::vector<uint8_t> source(16, 0xcc);
  first->pack(
      0, std::array{PinnedStagingArena::Copy{source.data(), 0, source.size()}});
  EXPECT_EQ(std::memcmp(first->data(0), source.data(), source.size()), 0);
  EXPECT_FALSE(PinnedStagingArena::acquirePair().has_value());
}

TEST_F(PinnedStagingArenaTest, allocationFailureDisablesGeneration) {
  PinnedStagingArena::setAllocationFailureForTesting(true);
  PinnedStagingArena::configure(true, 4096, 2);
  EXPECT_FALSE(PinnedStagingArena::acquirePair().has_value());
  EXPECT_FALSE(PinnedStagingArena::acquirePair().has_value());

  PinnedStagingArena::setAllocationFailureForTesting(false);
  PinnedStagingArena::configure(true, 4096, 2);
  EXPECT_TRUE(PinnedStagingArena::acquirePair().has_value());
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive::test
