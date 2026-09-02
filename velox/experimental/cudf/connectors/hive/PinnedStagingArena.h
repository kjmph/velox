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
/// host-to-device transfers. The arena owns a configured number of window
/// sets, with exactly two rolling windows in each set, and never allocates from
/// an upstream resource when they are busy. A transfer acquires one complete
/// set atomically so concurrent transfers cannot deadlock while rolling between
/// their windows. Callers retain the move-only lease until all asynchronous
/// CUDA access to both windows has completed.
class PinnedStagingArena {
 public:
  static constexpr uint32_t kWindowCount = 2;
  static constexpr uint64_t kPackQuantumBytes = 4ULL << 20;

  enum class HostAllocationStrategy {
    kUninitialized,
    kCudaHostAlloc,
    kMmapMbindCudaHostRegister,
  };

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

    /// Number of independent two-window sets in this arena generation.
    [[nodiscard]] uint32_t windowSetCount() const;

    /// Number of leases active immediately after this lease was granted.
    /// Exposed for runtime observability; it is not a synchronization
    /// primitive.
    [[nodiscard]] uint32_t activeLeasesAtAcquire() const;

    /// True when this acquisition could not be granted immediately.
    [[nodiscard]] bool wasContended() const;

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
    WindowSetLease(
        std::shared_ptr<Impl> arena,
        uint32_t windowSetIndex,
        uint32_t activeLeasesAtAcquire,
        bool wasContended);

    std::shared_ptr<Impl> arena_;
    uint32_t windowSetIndex_{0};
    uint32_t activeLeasesAtAcquire_{0};
    bool wasContended_{false};
  };

  /// Configures the arena without allocating host memory or starting pack
  /// threads. The first acquirePair() performs initialization on the calling
  /// thread, after the process has initialized its CUDA context.
  static void configure(
      bool enabled,
      uint64_t windowBytes,
      uint32_t packThreads,
      uint32_t windowSetCount);

  /// Returns the configured policy without forcing lazy arena allocation.
  /// This lets callers distinguish an intentional disabled bypass from an
  /// enabled arena that could not be acquired.
  [[nodiscard]] static bool enabled();

  /// Acquires one two-window set atomically in FIFO ticket order. Returns
  /// nullopt if staging is disabled, initialization failed, or reset()
  /// interrupts a waiter. Otherwise this blocks until a complete set is free.
  /// There is deliberately no single-window blocking API: every transfer owns
  /// a complete rolling-buffer set or none of it.
  [[nodiscard]] static std::optional<WindowSetLease> acquirePair();

  /// Prevents new acquisitions and releases the process-global arena. Leases
  /// already held remain valid and keep their backing allocation alive until
  /// released.
  static void reset();

  /// Forces host-allocation failure for focused tests. This also resets the
  /// current arena and must not be used by production callers.
  static void setAllocationFailureForTesting(bool fail);

  /// Forces the one-based host allocation in a newly initialized arena to
  /// fail. A value of zero disables injection. This resets the current arena
  /// and must not be used by production callers.
  static void setAllocationFailureAtForTesting(uint32_t allocationOrdinal);

  /// Overrides automatic single-node MPOL_BIND discovery for focused tests.
  /// A node selects the Linux mmap -> mbind -> first-touch ->
  /// cudaHostRegister path; nullopt selects the portable cudaHostAlloc path.
  /// This resets the current arena and must not be used by production callers.
  static void setNumaNodeForTesting(std::optional<uint32_t> numaNode);

  /// Restores automatic NUMA-policy discovery after setNumaNodeForTesting().
  /// This resets the current arena and must not be used by production callers.
  static void clearNumaNodeForTesting();

  /// Overrides PRESTO_GPU_NUMA_BINDING=required handling for focused tests
  /// without mutating the process environment. This resets the current arena
  /// and must not be used by production callers.
  static void setRequireNumaLocalForTesting(bool required);

  /// Restores production NUMA-requirement discovery after
  /// setRequireNumaLocalForTesting(). This resets the current arena and must
  /// not be used by production callers.
  static void clearRequireNumaLocalForTesting();

  /// Resolves policy-mask nodes for focused tests. Relative policy nodes are
  /// ordinal indexes into allowedNodes; absolute policy nodes are returned as
  /// supplied. Returns nullopt unless the resolved mask selects exactly one
  /// physical node.
  [[nodiscard]] static std::optional<uint32_t> resolveNumaNodeForTesting(
      bool relativeNodes,
      std::span<const uint32_t> policyNodes,
      std::span<const uint32_t> allowedNodes);

  /// Forces the one-based cudaHostRegister operation in a newly initialized
  /// NUMA-local arena to fail. A value of zero disables injection. This resets
  /// the current arena and must not be used by production callers.
  static void setRegistrationFailureAtForTesting(uint32_t registrationOrdinal);

  /// Returns the backing strategy of the current arena. This is test-only
  /// introspection and does not force lazy initialization.
  [[nodiscard]] static HostAllocationStrategy
  hostAllocationStrategyForTesting();

  /// Returns the number of callers currently queued for a window set. This is
  /// test-only introspection used to synchronize waiter tests without sleeps.
  [[nodiscard]] static uint32_t waitingLeaseCountForTesting();
};

} // namespace facebook::velox::cudf_velox::connector::hive
