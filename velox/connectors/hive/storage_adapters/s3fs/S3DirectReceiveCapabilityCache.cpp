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

#include <algorithm>
#include <optional>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace facebook::velox::filesystems {
namespace {

S3DirectReceiveCapabilityCache::TimePoint saturatedDeadline(
    S3DirectReceiveCapabilityCache::TimePoint now,
    S3DirectReceiveCapabilityCache::Duration ttl,
    S3DirectReceiveCapabilityCache::Duration jitter) {
  using Duration = S3DirectReceiveCapabilityCache::Duration;
  using TimePoint = S3DirectReceiveCapabilityCache::TimePoint;

  auto delay = ttl;
  if (Duration::max() - delay < jitter) {
    delay = Duration::max();
  } else {
    delay += jitter;
  }
  if (TimePoint::max() - now < delay) {
    return TimePoint::max();
  }
  return now + delay;
}

} // namespace

S3DirectReceiveCapabilityCache::S3DirectReceiveCapabilityCache()
    : S3DirectReceiveCapabilityCache(Options{}) {}

S3DirectReceiveCapabilityCache::S3DirectReceiveCapabilityCache(Options options)
    : S3DirectReceiveCapabilityCache(options, systemNow, randomJitter) {}

S3DirectReceiveCapabilityCache::S3DirectReceiveCapabilityCache(
    Options options,
    Now now,
    Jitter jitter)
    : options_(options), now_(std::move(now)), jitter_(std::move(jitter)) {
  if (options_.unavailableTtl < Duration::zero()) {
    throw std::invalid_argument(
        "S3 direct receive unavailable TTL is negative");
  }
  if (options_.maxJitter < Duration::zero()) {
    throw std::invalid_argument("S3 direct receive maximum jitter is negative");
  }
  if (!now_) {
    throw std::invalid_argument("S3 direct receive clock is empty");
  }
  if (!jitter_) {
    throw std::invalid_argument("S3 direct receive jitter source is empty");
  }
}

S3DirectReceiveCapabilityCache::Attempt
S3DirectReceiveCapabilityCache::beginAttempt() {
  std::unique_lock<std::mutex> lock(mutex_);
  for (;;) {
    switch (state_) {
      case State::UNKNOWN:
        return beginProbeLocked();
      case State::PROBING:
        return Attempt(Path::CALLER_BUFFER, false, epoch_);
      case State::AVAILABLE:
        return Attempt(Path::STRICT_KERNEL_TLS, false, epoch_);
      case State::UNAVAILABLE_UNTIL: {
        // Do not invoke an injected clock while holding the cache mutex. Keep
        // the epoch so a concurrent transition can be detected after locking
        // again.
        const auto unavailableEpoch = epoch_;
        lock.unlock();
        TimePoint now;
        try {
          now = now_();
        } catch (...) {
          // A clock is diagnostic/test injection, not part of the read. Treat
          // failure as expiration instead of failing or wedging S3 reads.
          now = TimePoint::max();
        }
        lock.lock();
        if (state_ != State::UNAVAILABLE_UNTIL || epoch_ != unavailableEpoch) {
          continue;
        }
        if (now >= unavailableUntil_) {
          return beginProbeLocked();
        }
        return Attempt(Path::CALLER_BUFFER, false, epoch_);
      }
    }
  }
}

void S3DirectReceiveCapabilityCache::finishAttempt(
    const Attempt& attempt,
    Observation observation,
    uint64_t bodyBytes,
    uint64_t directBodyBytes) {
  if (!attempt.requiresKernelTls()) {
    return;
  }

  const bool strictZeroBodyUnavailable =
      observation == Observation::UNAVAILABLE && bodyBytes == 0 &&
      directBodyBytes == 0;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (attempt.epoch_ != epoch_) {
      return;
    }
    if (attempt.isProbe_) {
      if (state_ != State::PROBING) {
        return;
      }
      if (observation == Observation::AVAILABLE) {
        setStateLocked(State::AVAILABLE);
        return;
      }
      if (!strictZeroBodyUnavailable) {
        setStateLocked(State::UNKNOWN);
        return;
      }
    } else if (state_ != State::AVAILABLE || !strictZeroBodyUnavailable) {
      return;
    }
  }

  // A negative transition needs the clock and jitter source. Evaluate both
  // outside the cache mutex, then revalidate the attempt epoch before
  // publishing the deadline.
  std::optional<TimePoint> unavailableUntil;
  try {
    auto jitter = jitter_(options_.maxJitter);
    jitter = std::clamp(jitter, Duration::zero(), options_.maxJitter);
    unavailableUntil =
        saturatedDeadline(now_(), options_.unavailableTtl, jitter);
  } catch (...) {
    // Injection or entropy failure must not fail a read or leave PROBING
    // permanent. Falling back to UNKNOWN allows a later request to retry.
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (attempt.epoch_ != epoch_ ||
      (attempt.isProbe_ ? state_ != State::PROBING
                        : state_ != State::AVAILABLE)) {
    return;
  }
  if (unavailableUntil.has_value()) {
    cacheUnavailableLocked(unavailableUntil.value());
  } else {
    setStateLocked(State::UNKNOWN);
  }
}

void S3DirectReceiveCapabilityCache::abandonAttempt(const Attempt& attempt) {
  if (!attempt.isProbe_) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ == State::PROBING && attempt.epoch_ == epoch_) {
    setStateLocked(State::UNKNOWN);
  }
}

S3DirectReceiveCapabilityCache::Snapshot
S3DirectReceiveCapabilityCache::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return {state_, unavailableUntil_, epoch_};
}

S3DirectReceiveCapabilityCache::TimePoint
S3DirectReceiveCapabilityCache::systemNow() {
  return SteadyClock::now();
}

S3DirectReceiveCapabilityCache::Duration
S3DirectReceiveCapabilityCache::randomJitter(Duration maxJitter) {
  static_assert(std::is_integral_v<Duration::rep>);
  if (maxJitter <= Duration::zero()) {
    return Duration::zero();
  }

  static std::mutex randomMutex;
  static std::mt19937_64 random(std::random_device{}());
  std::lock_guard<std::mutex> lock(randomMutex);
  std::uniform_int_distribution<Duration::rep> distribution(
      0, maxJitter.count());
  return Duration(distribution(random));
}

S3DirectReceiveCapabilityCache::Attempt
S3DirectReceiveCapabilityCache::beginProbeLocked() {
  setStateLocked(State::PROBING);
  return Attempt(Path::STRICT_KERNEL_TLS, true, epoch_);
}

void S3DirectReceiveCapabilityCache::setStateLocked(State state) {
  state_ = state;
  unavailableUntil_ = TimePoint{};
  ++epoch_;
}

void S3DirectReceiveCapabilityCache::cacheUnavailableLocked(
    TimePoint unavailableUntil) {
  state_ = State::UNAVAILABLE_UNTIL;
  unavailableUntil_ = unavailableUntil;
  ++epoch_;
}

} // namespace facebook::velox::filesystems
