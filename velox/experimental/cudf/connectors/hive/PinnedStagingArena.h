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

#include <cstdint>
#include <memory>
#include <optional>
#include <span>

namespace facebook::velox::cudf_velox::connector::hive {

/// A fixed-size, process-wide pool of pinned host memory used to stage
/// host-to-device transfers. The arena owns exactly two windows and never
/// allocates from an upstream resource when they are busy. A transfer acquires
/// both windows atomically so two transfers cannot deadlock while rolling
/// between them. Callers retain the move-only lease until all asynchronous
/// CUDA access to both windows has completed.
class PinnedStagingArena {
 public:
  static constexpr uint32_t kWindowCount = 2;
  static constexpr uint64_t kPackQuantumBytes = 4ULL << 20;

  struct Copy {
    const void* source{nullptr};
    uint64_t destinationOffset{0};
    uint64_t size{0};
  };

 private:
  struct Impl;

 public:
  class WindowSetLease {
   public:
    WindowSetLease() = default;
    ~WindowSetLease();

    WindowSetLease(const WindowSetLease&) = delete;
    WindowSetLease& operator=(const WindowSetLease&) = delete;
    WindowSetLease(WindowSetLease&& other) noexcept;
    WindowSetLease& operator=(WindowSetLease&& other) noexcept;

    [[nodiscard]] explicit operator bool() const {
      return arena_ != nullptr;
    }

    [[nodiscard]] uint8_t* data(uint32_t windowIndex) const;
    [[nodiscard]] uint64_t capacity() const;

    /// Copies non-overlapping source ranges into one window. Copy work is
    /// distributed in at most kPackQuantumBytes work units over the arena's
    /// fixed pack thread pool; this call returns only after every source range
    /// has been consumed. Source memory must remain valid until this method
    /// returns. The two windows may be packed concurrently.
    void pack(uint32_t windowIndex, std::span<const Copy> copies) const;

    /// Returns both windows to the arena. Callers must first establish that no
    /// asynchronous CUDA operation can still access either one.
    void release();

   private:
    friend class PinnedStagingArena;
    explicit WindowSetLease(std::shared_ptr<Impl> arena);

    std::shared_ptr<Impl> arena_;
  };

  /// Configures the arena without allocating host memory or starting pack
  /// threads. The first acquirePair() performs initialization on the calling
  /// thread, after the process has initialized its CUDA context.
  static void
  configure(bool enabled, uint64_t windowBytes, uint32_t packThreads);

  /// Returns the configured policy without forcing lazy arena allocation.
  /// This lets callers distinguish an intentional disabled bypass from an
  /// enabled arena that could not be acquired.
  [[nodiscard]] static bool enabled();

  /// Acquires both windows atomically in FIFO ticket order. Returns nullopt if
  /// staging is disabled, initialization failed, or reset() interrupts a
  /// waiter. Otherwise this blocks until the current window-set owner releases
  /// it. There is deliberately no single-window blocking API: every transfer
  /// owns the complete rolling-buffer set or none of it.
  [[nodiscard]] static std::optional<WindowSetLease> acquirePair();

  /// Prevents new acquisitions and releases the process-global arena. Leases
  /// already held remain valid and keep their backing allocation alive until
  /// released.
  static void reset();

  /// Forces host-allocation failure for focused tests. This also resets the
  /// current arena and must not be used by production callers.
  static void setAllocationFailureForTesting(bool fail);
};

} // namespace facebook::velox::cudf_velox::connector::hive
