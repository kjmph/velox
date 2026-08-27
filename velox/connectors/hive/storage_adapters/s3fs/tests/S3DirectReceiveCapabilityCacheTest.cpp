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

#include "velox/connectors/hive/storage_adapters/s3fs/S3DirectReceiveCapabilityCache.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

namespace facebook::velox::filesystems {
namespace {

using Cache = S3DirectReceiveCapabilityCache;
using namespace std::chrono_literals;

class ManualClock {
 public:
  Cache::TimePoint now() const {
    return Cache::TimePoint(Cache::Duration(ticks_.load()));
  }

  void advance(Cache::Duration duration) {
    ticks_.fetch_add(duration.count());
  }

 private:
  std::atomic<Cache::Duration::rep> ticks_{0};
};

TEST(S3DirectReceiveCapabilityCacheTest, electsExactlyOneConcurrentProber) {
  Cache cache;
  constexpr size_t kThreads = 32;
  std::atomic<bool> start{false};
  std::vector<Cache::Attempt> attempts(kThreads);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (size_t i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      attempts[i] = cache.beginAttempt();
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }

  size_t probes = 0;
  size_t callerBuffers = 0;
  for (const auto& attempt : attempts) {
    probes += attempt.isProbe();
    callerBuffers += attempt.path() == Cache::Path::CALLER_BUFFER;
  }
  EXPECT_EQ(probes, 1);
  EXPECT_EQ(callerBuffers, kThreads - 1);
  EXPECT_EQ(cache.snapshot().state, Cache::State::PROBING);
}

TEST(S3DirectReceiveCapabilityCacheTest, successfulProbeEnablesStrictReceive) {
  Cache cache;
  const auto probe = cache.beginAttempt();
  ASSERT_TRUE(probe.isProbe());
  cache.finishAttempt(probe, Cache::Observation::AVAILABLE, 1024, 1024);

  EXPECT_EQ(cache.snapshot().state, Cache::State::AVAILABLE);
  const auto attempt = cache.beginAttempt();
  EXPECT_TRUE(attempt.requiresKernelTls());
  EXPECT_FALSE(attempt.isProbe());
}

TEST(S3DirectReceiveCapabilityCacheTest, negativeEntryUsesTtlAndJitter) {
  ManualClock clock;
  Cache cache(
      {.unavailableTtl = 10s, .maxJitter = 5s},
      [&] { return clock.now(); },
      [](Cache::Duration maximum) {
        EXPECT_EQ(maximum, 5s);
        return 3s;
      });

  const auto probe = cache.beginAttempt();
  cache.finishAttempt(probe, Cache::Observation::UNAVAILABLE, 0, 0);
  auto snapshot = cache.snapshot();
  ASSERT_EQ(snapshot.state, Cache::State::UNAVAILABLE_UNTIL);
  EXPECT_EQ(snapshot.unavailableUntil, Cache::TimePoint(13s));
  EXPECT_EQ(cache.beginAttempt().path(), Cache::Path::CALLER_BUFFER);

  clock.advance(13s - 1ns);
  EXPECT_EQ(cache.beginAttempt().path(), Cache::Path::CALLER_BUFFER);
  clock.advance(1ns);
  const auto reprobe = cache.beginAttempt();
  EXPECT_TRUE(reprobe.requiresKernelTls());
  EXPECT_TRUE(reprobe.isProbe());
  EXPECT_EQ(cache.snapshot().state, Cache::State::PROBING);
}

TEST(
    S3DirectReceiveCapabilityCacheTest,
    expiredEntryElectsExactlyOneConcurrentReprober) {
  ManualClock clock;
  Cache cache(
      {.unavailableTtl = 1s, .maxJitter = Cache::Duration::zero()},
      [&] { return clock.now(); },
      [](Cache::Duration) { return Cache::Duration::zero(); });
  const auto firstProbe = cache.beginAttempt();
  cache.finishAttempt(firstProbe, Cache::Observation::UNAVAILABLE, 0, 0);
  clock.advance(1s);

  constexpr size_t kThreads = 32;
  std::atomic<bool> start{false};
  std::vector<Cache::Attempt> attempts(kThreads);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (size_t i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      attempts[i] = cache.beginAttempt();
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }

  size_t probes = 0;
  for (const auto& attempt : attempts) {
    probes += attempt.isProbe();
  }
  EXPECT_EQ(probes, 1);
  EXPECT_EQ(cache.snapshot().state, Cache::State::PROBING);
}

TEST(S3DirectReceiveCapabilityCacheTest, cachesOnlyStrictZeroBodyUnavailable) {
  Cache cache;
  auto probe = cache.beginAttempt();
  cache.finishAttempt(probe, Cache::Observation::UNAVAILABLE, 1, 0);
  EXPECT_EQ(cache.snapshot().state, Cache::State::UNKNOWN);

  probe = cache.beginAttempt();
  cache.finishAttempt(probe, Cache::Observation::UNAVAILABLE, 0, 1);
  EXPECT_EQ(cache.snapshot().state, Cache::State::UNKNOWN);

  probe = cache.beginAttempt();
  cache.finishAttempt(probe, Cache::Observation::INDETERMINATE, 0, 0);
  EXPECT_EQ(cache.snapshot().state, Cache::State::UNKNOWN);

  probe = cache.beginAttempt();
  const auto callerBuffer = cache.beginAttempt();
  ASSERT_EQ(callerBuffer.path(), Cache::Path::CALLER_BUFFER);
  cache.finishAttempt(callerBuffer, Cache::Observation::UNAVAILABLE, 0, 0);
  EXPECT_EQ(cache.snapshot().state, Cache::State::PROBING);
  cache.abandonAttempt(probe);
  EXPECT_EQ(cache.snapshot().state, Cache::State::UNKNOWN);
}

TEST(
    S3DirectReceiveCapabilityCacheTest,
    availableCapabilityCanBecomeTemporarilyUnavailable) {
  ManualClock clock;
  Cache cache(
      {.unavailableTtl = 1min, .maxJitter = Cache::Duration::zero()},
      [&] { return clock.now(); },
      [](Cache::Duration) { return Cache::Duration::zero(); });
  const auto probe = cache.beginAttempt();
  cache.finishAttempt(probe, Cache::Observation::AVAILABLE, 10, 10);

  const auto strict = cache.beginAttempt();
  ASSERT_FALSE(strict.isProbe());
  cache.finishAttempt(strict, Cache::Observation::UNAVAILABLE, 0, 0);
  EXPECT_EQ(cache.snapshot().state, Cache::State::UNAVAILABLE_UNTIL);
  EXPECT_EQ(cache.beginAttempt().path(), Cache::Path::CALLER_BUFFER);
}

TEST(S3DirectReceiveCapabilityCacheTest, abandonedProbeAllowsReprobe) {
  Cache cache;
  const auto abandoned = cache.beginAttempt();
  ASSERT_TRUE(abandoned.isProbe());
  const auto abandonedEpoch = cache.snapshot().epoch;

  cache.abandonAttempt(abandoned);
  EXPECT_EQ(cache.snapshot().state, Cache::State::UNKNOWN);
  const auto replacement = cache.beginAttempt();
  EXPECT_TRUE(replacement.isProbe());
  EXPECT_NE(abandonedEpoch, cache.snapshot().epoch);

  cache.abandonAttempt(abandoned);
  EXPECT_EQ(cache.snapshot().state, Cache::State::PROBING);
}

TEST(S3DirectReceiveCapabilityCacheTest, ignoresStaleCompletions) {
  Cache cache;
  const auto firstProbe = cache.beginAttempt();
  const auto firstEpoch = cache.snapshot().epoch;
  cache.finishAttempt(firstProbe, Cache::Observation::INDETERMINATE, 0, 0);
  const auto secondProbe = cache.beginAttempt();
  ASSERT_NE(firstEpoch, cache.snapshot().epoch);

  cache.finishAttempt(firstProbe, Cache::Observation::AVAILABLE, 100, 100);
  EXPECT_EQ(cache.snapshot().state, Cache::State::PROBING);
  cache.finishAttempt(secondProbe, Cache::Observation::AVAILABLE, 100, 100);
  EXPECT_EQ(cache.snapshot().state, Cache::State::AVAILABLE);
}

TEST(S3DirectReceiveCapabilityCacheTest, clampsInjectedJitter) {
  ManualClock clock;
  Cache cache(
      {.unavailableTtl = 2s, .maxJitter = 4s},
      [&] { return clock.now(); },
      [](Cache::Duration) { return 1h; });
  const auto probe = cache.beginAttempt();
  cache.finishAttempt(probe, Cache::Observation::UNAVAILABLE, 0, 0);
  EXPECT_EQ(cache.snapshot().unavailableUntil, Cache::TimePoint(6s));
}

TEST(S3DirectReceiveCapabilityCacheTest, callbackFailuresDoNotWedgeProbing) {
  ManualClock clock;
  Cache throwingJitter(
      {.unavailableTtl = 2s, .maxJitter = 1s},
      [&] { return clock.now(); },
      [](Cache::Duration) -> Cache::Duration {
        throw std::runtime_error("jitter failed");
      });
  auto probe = throwingJitter.beginAttempt();
  throwingJitter.finishAttempt(probe, Cache::Observation::UNAVAILABLE, 0, 0);
  EXPECT_EQ(throwingJitter.snapshot().state, Cache::State::UNKNOWN);
  EXPECT_TRUE(throwingJitter.beginAttempt().isProbe());

  std::atomic<bool> throwClock{false};
  Cache throwingClock(
      {.unavailableTtl = 2s, .maxJitter = Cache::Duration::zero()},
      [&] {
        if (throwClock.load()) {
          throw std::runtime_error("clock failed");
        }
        return clock.now();
      },
      [](Cache::Duration) { return Cache::Duration::zero(); });
  probe = throwingClock.beginAttempt();
  throwingClock.finishAttempt(probe, Cache::Observation::UNAVAILABLE, 0, 0);
  ASSERT_EQ(throwingClock.snapshot().state, Cache::State::UNAVAILABLE_UNTIL);
  throwClock = true;
  EXPECT_TRUE(throwingClock.beginAttempt().isProbe());
}

} // namespace
} // namespace facebook::velox::filesystems
