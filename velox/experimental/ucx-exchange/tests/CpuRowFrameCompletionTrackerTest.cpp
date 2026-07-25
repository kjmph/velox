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

#include "velox/experimental/ucx-exchange/CpuRowFrameCompletionTracker.h"

#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>

namespace facebook::velox::ucx_exchange {
namespace {

TEST(CpuRowCallbackOnceTest, claimsOnlyOnce) {
  CpuRowCallbackOnce once;

  EXPECT_TRUE(once.tryClaim());
  EXPECT_FALSE(once.tryClaim());
}

TEST(CpuRowCallbackOnceTest, concurrentClaimsHaveOneWinner) {
  constexpr size_t kNumThreads = 16;
  CpuRowCallbackOnce once;
  std::atomic<bool> start{false};
  std::atomic<size_t> winners{0};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (size_t i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      if (once.tryClaim()) {
        winners.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(winners.load(std::memory_order_relaxed), 1);
}

TEST(CpuRowFrameCompletionTrackerTest, duplicateDoesNotCompleteOrSetError) {
  using State = CpuRowFrameCompletionTracker::State;

  CpuRowFrameCompletionTracker tracker(3);
  auto result = tracker.complete(0, UCS_OK);
  EXPECT_EQ(result.state, State::kPending);
  EXPECT_EQ(result.finalStatus, UCS_OK);

  result = tracker.complete(0, UCS_ERR_IO_ERROR);
  EXPECT_EQ(result.state, State::kDuplicate);
  EXPECT_EQ(result.finalStatus, UCS_OK);

  result = tracker.complete(1, UCS_OK);
  EXPECT_EQ(result.state, State::kPending);
  EXPECT_EQ(result.finalStatus, UCS_OK);

  result = tracker.complete(2, UCS_OK);
  EXPECT_EQ(result.state, State::kComplete);
  EXPECT_EQ(result.finalStatus, UCS_OK);

  result = tracker.complete(2, UCS_ERR_CANCELED);
  EXPECT_EQ(result.state, State::kDuplicate);
  EXPECT_EQ(result.finalStatus, UCS_OK);
}

TEST(CpuRowFrameCompletionTrackerTest, concurrentDuplicatesCompleteOnce) {
  using State = CpuRowFrameCompletionTracker::State;

  constexpr size_t kNumFrames = 3;
  constexpr size_t kNumThreads = 48;
  CpuRowFrameCompletionTracker tracker(kNumFrames);
  std::atomic<bool> start{false};
  std::atomic<size_t> pending{0};
  std::atomic<size_t> complete{0};
  std::atomic<size_t> duplicate{0};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (size_t i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&, frameIndex = i % kNumFrames]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      const auto result = tracker.complete(frameIndex, UCS_OK);
      switch (result.state) {
        case State::kPending:
          pending.fetch_add(1, std::memory_order_relaxed);
          break;
        case State::kComplete:
          complete.fetch_add(1, std::memory_order_relaxed);
          break;
        case State::kDuplicate:
          duplicate.fetch_add(1, std::memory_order_relaxed);
          break;
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(pending.load(std::memory_order_relaxed), kNumFrames - 1);
  EXPECT_EQ(complete.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(
      duplicate.load(std::memory_order_relaxed), kNumThreads - kNumFrames);
}

TEST(CpuRowFrameCompletionTrackerTest, preservesFirstError) {
  using State = CpuRowFrameCompletionTracker::State;

  CpuRowFrameCompletionTracker tracker(3);
  auto result = tracker.complete(0, UCS_ERR_IO_ERROR);
  EXPECT_EQ(result.state, State::kPending);
  EXPECT_EQ(result.finalStatus, UCS_ERR_IO_ERROR);

  result = tracker.complete(1, UCS_ERR_CANCELED);
  EXPECT_EQ(result.state, State::kPending);
  EXPECT_EQ(result.finalStatus, UCS_ERR_IO_ERROR);

  result = tracker.complete(2, UCS_OK);
  EXPECT_EQ(result.state, State::kComplete);
  EXPECT_EQ(result.finalStatus, UCS_ERR_IO_ERROR);
}

TEST(CpuRowFrameCompletionTrackerTest, rejectsInvalidConstructionAndIndex) {
  EXPECT_THROW(CpuRowFrameCompletionTracker(0), std::invalid_argument);

  CpuRowFrameCompletionTracker tracker(1);
  EXPECT_THROW(tracker.complete(1, UCS_OK), std::out_of_range);
}

} // namespace
} // namespace facebook::velox::ucx_exchange
