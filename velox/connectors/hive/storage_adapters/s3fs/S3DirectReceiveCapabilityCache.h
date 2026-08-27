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

#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>

namespace facebook::velox::filesystems {

/// Thread-safe capability state shared by S3 reads using preferred direct
/// receive mode.
///
/// One request probes strict RX kTLS while capability is unknown or an
/// unavailable entry has expired. Concurrent requests use caller-owned buffers
/// with ordinary TLS instead of repeating the probe. An unavailable result is
/// cached only when a strict request reports that no response body was
/// delivered.
class S3DirectReceiveCapabilityCache {
 public:
  using SteadyClock = std::chrono::steady_clock;
  using TimePoint = SteadyClock::time_point;
  using Duration = SteadyClock::duration;
  using Now = std::function<TimePoint()>;
  using Jitter = std::function<Duration(Duration maxJitter)>;

  enum class State {
    UNKNOWN,
    PROBING,
    AVAILABLE,
    UNAVAILABLE_UNTIL,
  };

  enum class Path {
    CALLER_BUFFER,
    STRICT_KERNEL_TLS,
  };

  /// Capability information learned by a strict attempt. INDETERMINATE covers
  /// request, authentication, protocol, integrity, and transport failures that
  /// must not be negatively cached.
  enum class Observation {
    AVAILABLE,
    UNAVAILABLE,
    INDETERMINATE,
  };

  struct Options {
    Duration unavailableTtl{std::chrono::minutes(5)};
    Duration maxJitter{std::chrono::seconds(30)};
  };

  class Attempt {
   public:
    Attempt() = default;

    Path path() const {
      return path_;
    }

    bool isProbe() const {
      return isProbe_;
    }

    bool requiresKernelTls() const {
      return path_ == Path::STRICT_KERNEL_TLS;
    }

   private:
    friend class S3DirectReceiveCapabilityCache;

    Attempt(Path path, bool isProbe, uint64_t epoch)
        : path_(path), isProbe_(isProbe), epoch_(epoch) {}

    Path path_{Path::CALLER_BUFFER};
    bool isProbe_{false};
    uint64_t epoch_{0};
  };

  struct Snapshot {
    State state{State::UNKNOWN};
    TimePoint unavailableUntil{};
    uint64_t epoch{0};
  };

  S3DirectReceiveCapabilityCache();
  explicit S3DirectReceiveCapabilityCache(Options options);
  S3DirectReceiveCapabilityCache(Options options, Now now, Jitter jitter);

  S3DirectReceiveCapabilityCache(const S3DirectReceiveCapabilityCache&) =
      delete;
  S3DirectReceiveCapabilityCache& operator=(
      const S3DirectReceiveCapabilityCache&) = delete;

  /// Chooses the receive path for one request. Exactly one concurrent caller
  /// receives a probe attempt for each UNKNOWN or expired state.
  Attempt beginAttempt();

  /// Reports the capability observation from a completed attempt.
  ///
  /// Only a STRICT_KERNEL_TLS attempt with observation UNAVAILABLE and both
  /// byte counts equal to zero creates a negative cache entry. Caller-buffer
  /// and stale attempts are ignored.
  void finishAttempt(
      const Attempt& attempt,
      Observation observation,
      uint64_t bodyBytes,
      uint64_t directBodyBytes);

  /// Releases an unfinished probe after an exception or cancellation. The
  /// next caller can probe again. This is a no-op for caller-buffer,
  /// non-probe, completed, and stale attempts.
  void abandonAttempt(const Attempt& attempt);

  Snapshot snapshot() const;

 private:
  static TimePoint systemNow();
  static Duration randomJitter(Duration maxJitter);

  Attempt beginProbeLocked();
  void setStateLocked(State state);
  void cacheUnavailableLocked(TimePoint unavailableUntil);

  const Options options_;
  const Now now_;
  const Jitter jitter_;

  mutable std::mutex mutex_;
  State state_{State::UNKNOWN};
  TimePoint unavailableUntil_{};
  uint64_t epoch_{0};
};

} // namespace facebook::velox::filesystems
