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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace facebook::velox::ucx_exchange {

struct UcxCpuRowShmSegment {
  UcxCpuRowShmSegment(
      std::string name,
      size_t size,
      void* address,
      int fd,
      bool unlinkOnDestroy);

  ~UcxCpuRowShmSegment();

  UcxCpuRowShmSegment(const UcxCpuRowShmSegment&) = delete;
  UcxCpuRowShmSegment& operator=(const UcxCpuRowShmSegment&) = delete;

  uint8_t* data() const {
    return static_cast<uint8_t*>(address);
  }

  std::string name;
  size_t size{0};
  void* address{nullptr};
  int fd{-1};
  bool unlinkOnDestroy{false};
};

bool ucxCpuRowShmEnabled();

bool ucxCpuRowShmDirectTxEnabled();

bool ucxCpuRowShmSlotPoolEnabled();

uint64_t ucxCpuRowShmSlotPoolSlotBytes();

uint32_t ucxCpuRowShmSlotPoolInitialSlots();

uint32_t ucxCpuRowShmSlotPoolNumSlots();

uint32_t ucxCpuRowShmSlotPoolMaxPools();

std::shared_ptr<UcxCpuRowShmSegment> createUcxCpuRowShmSegment(size_t size);

std::shared_ptr<UcxCpuRowShmSegment> openUcxCpuRowShmSegment(
    std::string_view name,
    size_t size,
    bool unlinkAfterOpen,
    bool writable = false);

class UcxCpuRowShmSlotPool {
 public:
  struct Slot {
    uint32_t id;
    uint8_t* data;
  };

  static std::shared_ptr<UcxCpuRowShmSlotPool>
  create(size_t slotSize, uint32_t numSlots, uint32_t expectedOpeners);

  static std::shared_ptr<UcxCpuRowShmSlotPool> open(
      std::string_view name,
      size_t expectedSize,
      bool unlinkAfterExpectedOpeners);

  const std::string& name() const;

  /// Transfers responsibility for unlinking the slot-pool name to receivers.
  /// Receivers atomically count opens in the shared header and the last
  /// expected opener unlinks the POSIX SHM name. This allows broadcast outputs
  /// to advertise one pool to multiple receivers without the first receiver
  /// making the name disappear before the others open it.
  void disableUnlinkOnDestroy();

  size_t totalSize() const;

  size_t slotSize() const;

  uint32_t numSlots() const;

  uint32_t expectedOpeners() const;

  std::optional<Slot> tryAcquire(size_t bytes);

  /// Publishes a written slot with one producer-side reference.
  void markReady(uint32_t slotId);

  bool isReady(uint32_t slotId) const;

  /// Adds one receiver reference before advertising slot metadata.
  /// Returns false if the slot is not in a ready state.
  bool addRef(uint32_t slotId);

  /// Drops one producer or receiver reference. The slot returns to the
  /// free list when the last reference is released.
  void release(uint32_t slotId);

  uint8_t* slotData(uint32_t slotId) const;

 private:
  explicit UcxCpuRowShmSlotPool(std::shared_ptr<UcxCpuRowShmSegment> segment);

  static size_t dataOffset(uint32_t numSlots);

  uint32_t* statePtr(uint32_t slotId) const;

  void recordOpenAndMaybeUnlink();

  std::shared_ptr<UcxCpuRowShmSegment> segment_;
  uint32_t numSlots_{0};
  uint32_t expectedOpeners_{0};
  size_t slotSize_{0};
  size_t dataOffset_{0};
  mutable uint32_t nextSlot_{0};
};

} // namespace facebook::velox::ucx_exchange
