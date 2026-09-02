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
#include <fstream>
#include <future>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive::test {
namespace {

using namespace std::chrono_literals;

bool waitForQueuedLeases(uint32_t expected) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  do {
    if (PinnedStagingArena::waitingLeaseCountForTesting() >= expected) {
      return true;
    }
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < deadline);
  return PinnedStagingArena::waitingLeaseCountForTesting() >= expected;
}

std::optional<uint32_t> firstAllowedNumaNode() {
#if defined(__linux__)
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    constexpr std::string_view kPrefix = "Mems_allowed_list:";
    if (!line.starts_with(kPrefix)) {
      continue;
    }
    const auto firstDigit = line.find_first_of("0123456789", kPrefix.size());
    if (firstDigit == std::string::npos) {
      return std::nullopt;
    }
    const auto end = line.find_first_not_of("0123456789", firstDigit);
    return static_cast<uint32_t>(
        std::stoul(line.substr(firstDigit, end - firstDigit)));
  }
#endif
  return std::nullopt;
}

class PinnedStagingArenaTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  }

  void SetUp() override {
    // Keep tests independent of a launcher environment inherited by the test
    // process. Individual strict-placement tests opt back in explicitly.
    PinnedStagingArena::setRequireNumaLocalForTesting(false);
  }

  void TearDown() override {
    PinnedStagingArena::setAllocationFailureForTesting(false);
    PinnedStagingArena::setRegistrationFailureAtForTesting(0);
    PinnedStagingArena::clearNumaNodeForTesting();
    PinnedStagingArena::clearRequireNumaLocalForTesting();
  }
};

TEST_F(PinnedStagingArenaTest, disabled) {
  PinnedStagingArena::configure(false, 0, 0, 0);
  EXPECT_FALSE(PinnedStagingArena::enabled());
  EXPECT_FALSE(PinnedStagingArena::acquirePair().has_value());
}

TEST_F(PinnedStagingArenaTest, validatesEnabledConfiguration) {
  EXPECT_THROW(PinnedStagingArena::configure(true, 0, 1, 1), VeloxException);
  EXPECT_THROW(PinnedStagingArena::configure(true, 4096, 0, 1), VeloxException);
  EXPECT_THROW(PinnedStagingArena::configure(true, 4096, 1, 0), VeloxException);
  EXPECT_THROW(
      PinnedStagingArena::configure(
          true,
          std::numeric_limits<size_t>::max() / PinnedStagingArena::kWindowCount,
          1,
          2),
      VeloxException);
}

TEST_F(PinnedStagingArenaTest, acquiresBothWindowsAtomically) {
  constexpr uint64_t kWindowBytes = 4096;
  PinnedStagingArena::configure(true, kWindowBytes, 2, 1);
  EXPECT_TRUE(PinnedStagingArena::enabled());

  auto first = PinnedStagingArena::acquirePair();
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->capacity(), kWindowBytes);
  EXPECT_EQ(first->windowSetCount(), 1);
  EXPECT_EQ(first->activeLeasesAtAcquire(), 1);
  EXPECT_FALSE(first->wasContended());
  EXPECT_NE(first->data(0), first->data(1));
  auto* firstWindow = first->data(0);
  auto* secondWindow = first->data(1);

  auto waiter = std::async(
      std::launch::async, []() { return PinnedStagingArena::acquirePair(); });
  EXPECT_TRUE(waitForQueuedLeases(1));
  EXPECT_EQ(waiter.wait_for(0ms), std::future_status::timeout);

  first->release();
  auto second = waiter.get();
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->data(0), firstWindow);
  EXPECT_EQ(second->data(1), secondWindow);
  EXPECT_TRUE(second->wasContended());
}

TEST_F(PinnedStagingArenaTest, leasesIndependentWindowSetsConcurrently) {
  constexpr uint64_t kWindowBytes = 4096;
  PinnedStagingArena::configure(true, kWindowBytes, 2, 2);

  auto first = PinnedStagingArena::acquirePair();
  ASSERT_TRUE(first.has_value());
  auto second = PinnedStagingArena::acquirePair();
  ASSERT_TRUE(second.has_value());

  EXPECT_EQ(first->windowSetCount(), 2);
  EXPECT_EQ(first->activeLeasesAtAcquire(), 1);
  EXPECT_EQ(second->activeLeasesAtAcquire(), 2);
  EXPECT_FALSE(first->wasContended());
  EXPECT_FALSE(second->wasContended());
  EXPECT_NE(first->data(0), second->data(0));
  EXPECT_NE(first->data(1), second->data(1));

  auto* firstWindow = first->data(0);
  auto waiter = std::async(
      std::launch::async, []() { return PinnedStagingArena::acquirePair(); });
  EXPECT_TRUE(waitForQueuedLeases(1));
  EXPECT_EQ(waiter.wait_for(0ms), std::future_status::timeout);

  first->release();
  auto third = waiter.get();
  ASSERT_TRUE(third.has_value());
  EXPECT_TRUE(third->wasContended());
  EXPECT_EQ(third->activeLeasesAtAcquire(), 2);
  EXPECT_EQ(third->data(0), firstWindow);
}

TEST_F(PinnedStagingArenaTest, packsIndependentWindowSetsConcurrently) {
  constexpr uint64_t kWindowBytes = PinnedStagingArena::kPackQuantumBytes + 17;
  PinnedStagingArena::configure(true, kWindowBytes, 2, 2);

  auto first = PinnedStagingArena::acquirePair();
  auto second = PinnedStagingArena::acquirePair();
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());

  std::vector<uint8_t> firstSource(kWindowBytes, 0x31);
  std::vector<uint8_t> secondSource(kWindowBytes, 0x72);
  const std::array firstCopy = {
      PinnedStagingArena::Copy{firstSource.data(), 0, firstSource.size()}};
  const std::array secondCopy = {
      PinnedStagingArena::Copy{secondSource.data(), 0, secondSource.size()}};

  auto firstPack =
      std::async(std::launch::async, [&]() { first->pack(0, firstCopy); });
  auto secondPack =
      std::async(std::launch::async, [&]() { second->pack(0, secondCopy); });
  firstPack.get();
  secondPack.get();

  EXPECT_EQ(
      std::memcmp(first->data(0), firstSource.data(), firstSource.size()), 0);
  EXPECT_EQ(
      std::memcmp(second->data(0), secondSource.data(), secondSource.size()),
      0);
}

TEST_F(PinnedStagingArenaTest, retainsCompletedSetAfterLaterAllocationFailure) {
  // Two allocations complete set zero. Allocation four fails after the first
  // window of set one was allocated, so that incomplete set must be discarded
  // while set zero remains usable.
  PinnedStagingArena::setAllocationFailureAtForTesting(4);
  PinnedStagingArena::configure(true, 4096, 2, 3);

  auto first = PinnedStagingArena::acquirePair();
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->windowSetCount(), 1);

  auto waiter = std::async(
      std::launch::async, []() { return PinnedStagingArena::acquirePair(); });
  EXPECT_TRUE(waitForQueuedLeases(1));
  EXPECT_EQ(waiter.wait_for(0ms), std::future_status::timeout);

  auto* retainedWindow = first->data(0);
  first->release();
  auto replacement = waiter.get();
  ASSERT_TRUE(replacement.has_value());
  EXPECT_EQ(replacement->windowSetCount(), 1);
  EXPECT_EQ(replacement->data(0), retainedWindow);
  EXPECT_TRUE(replacement->wasContended());
}

TEST_F(PinnedStagingArenaTest, moveAndReleaseWindowSet) {
  PinnedStagingArena::configure(true, 4096, 2, 1);
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
  PinnedStagingArena::configure(true, kWindowBytes, 4, 1);
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
  PinnedStagingArena::configure(true, kWindowBytes, 2, 1);
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
  PinnedStagingArena::configure(true, 4096, 2, 1);
  auto first = PinnedStagingArena::acquirePair();
  ASSERT_TRUE(first.has_value());

  auto waiter = std::async(
      std::launch::async, []() { return PinnedStagingArena::acquirePair(); });
  EXPECT_TRUE(waitForQueuedLeases(1));
  EXPECT_EQ(waiter.wait_for(0ms), std::future_status::timeout);

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
  PinnedStagingArena::configure(true, 4096, 2, 1);
  EXPECT_FALSE(PinnedStagingArena::acquirePair().has_value());
  EXPECT_FALSE(PinnedStagingArena::acquirePair().has_value());

  PinnedStagingArena::setAllocationFailureForTesting(false);
  PinnedStagingArena::configure(true, 4096, 2, 1);
  EXPECT_TRUE(PinnedStagingArena::acquirePair().has_value());
}

TEST_F(PinnedStagingArenaTest, portableAllocationCanBeSelectedExplicitly) {
  PinnedStagingArena::setNumaNodeForTesting(std::nullopt);
  PinnedStagingArena::configure(true, 4096, 2, 1);
  auto lease = PinnedStagingArena::acquirePair();
  ASSERT_TRUE(lease.has_value());
  EXPECT_EQ(
      PinnedStagingArena::hostAllocationStrategyForTesting(),
      PinnedStagingArena::HostAllocationStrategy::kCudaHostAlloc);
}

TEST_F(PinnedStagingArenaTest, requiredNumaBindingFailsClosed) {
  PinnedStagingArena::setNumaNodeForTesting(std::nullopt);
  PinnedStagingArena::setRequireNumaLocalForTesting(true);
  PinnedStagingArena::configure(true, 4096, 2, 1);
  EXPECT_FALSE(PinnedStagingArena::acquirePair().has_value());
  EXPECT_EQ(
      PinnedStagingArena::hostAllocationStrategyForTesting(),
      PinnedStagingArena::HostAllocationStrategy::kUninitialized);
}

TEST_F(PinnedStagingArenaTest, resolvesRelativeAndAbsolutePolicyNodes) {
  constexpr std::array<uint32_t, 1> kPolicyNodes{1};
  constexpr std::array<uint32_t, 3> kAllowedNodes{2, 5, 9};
  EXPECT_EQ(
      PinnedStagingArena::resolveNumaNodeForTesting(
          true, kPolicyNodes, kAllowedNodes),
      5);
  EXPECT_EQ(
      PinnedStagingArena::resolveNumaNodeForTesting(
          false, kPolicyNodes, kAllowedNodes),
      1);

  constexpr std::array<uint32_t, 2> kMultiplePolicyNodes{0, 1};
  EXPECT_FALSE(
      PinnedStagingArena::resolveNumaNodeForTesting(
          true, kMultiplePolicyNodes, kAllowedNodes)
          .has_value());
  constexpr std::array<uint32_t, 1> kOutOfRangePolicyNode{3};
  EXPECT_THROW(
      PinnedStagingArena::resolveNumaNodeForTesting(
          true, kOutOfRangePolicyNode, kAllowedNodes),
      std::runtime_error);
}

TEST_F(PinnedStagingArenaTest, usesStrictNumaLocalRegisteredAllocation) {
  const auto numaNode = firstAllowedNumaNode();
  if (!numaNode.has_value()) {
    GTEST_SKIP() << "No allowed NUMA node is visible";
  }

  PinnedStagingArena::setNumaNodeForTesting(numaNode);
  PinnedStagingArena::setRequireNumaLocalForTesting(true);
  PinnedStagingArena::configure(true, 4096, 2, 1);
  auto lease = PinnedStagingArena::acquirePair();
  if (!lease.has_value()) {
    GTEST_SKIP()
        << "The test environment does not permit mbind/cudaHostRegister";
  }
  EXPECT_EQ(
      PinnedStagingArena::hostAllocationStrategyForTesting(),
      PinnedStagingArena::HostAllocationStrategy::kMmapMbindCudaHostRegister);
}

TEST_F(
    PinnedStagingArenaTest,
    registrationFailureDoesNotFallBackFromStrictNumaAllocation) {
  const auto numaNode = firstAllowedNumaNode();
  if (!numaNode.has_value()) {
    GTEST_SKIP() << "No allowed NUMA node is visible";
  }

  // First prove that this environment can exercise the real strict path.
  PinnedStagingArena::setNumaNodeForTesting(numaNode);
  PinnedStagingArena::setRequireNumaLocalForTesting(true);
  PinnedStagingArena::configure(true, 4096, 2, 1);
  auto probe = PinnedStagingArena::acquirePair();
  if (!probe.has_value()) {
    GTEST_SKIP()
        << "The test environment does not permit mbind/cudaHostRegister";
  }
  probe->release();

  PinnedStagingArena::setRegistrationFailureAtForTesting(1);
  PinnedStagingArena::configure(true, 4096, 2, 1);
  EXPECT_FALSE(PinnedStagingArena::acquirePair().has_value());
  EXPECT_EQ(
      PinnedStagingArena::hostAllocationStrategyForTesting(),
      PinnedStagingArena::HostAllocationStrategy::kUninitialized);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive::test
