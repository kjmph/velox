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
#include "velox/experimental/ucx-exchange/UcxCpuRowShm.h"

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::ucx_exchange {
namespace {

std::string errnoString() {
  return std::string(std::strerror(errno));
}

bool sizeFitsOffT(size_t size) {
  return size <= static_cast<size_t>(std::numeric_limits<off_t>::max());
}

std::string sanitizeForShmName(std::string value) {
  for (char& c : value) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-') {
      c = '_';
    }
  }
  if (value.size() > 96) {
    value.resize(96);
  }
  return value.empty() ? "unknown" : value;
}

const std::string& shmNamePrefix() {
  static const std::string kPrefix = [] {
    std::string identity;
    if (const char* port = std::getenv("VELOX_UCX_CPU_PORT");
        port != nullptr && *port != '\0') {
      identity += "p";
      identity += sanitizeForShmName(port);
    }
    if (const char* workerId = std::getenv("WORKER_ID");
        workerId != nullptr && *workerId != '\0') {
      identity += "_w";
      identity += sanitizeForShmName(workerId);
    }

    char hostname[128] = {0};
    if (::gethostname(hostname, sizeof(hostname) - 1) == 0 &&
        hostname[0] != '\0') {
      identity += "_h";
      identity += sanitizeForShmName(hostname);
    }

    if (identity.empty()) {
      identity = "unknown";
    }

    return "/velox_ucx_cpu_" + identity + "_" + std::to_string(::getpid()) +
        "_";
  }();
  return kPrefix;
}

std::string nextShmName() {
  static std::atomic<uint64_t> counter{0};
  return shmNamePrefix() +
      std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

uint64_t readUint64Env(
    const char* name,
    uint64_t defaultValue,
    uint64_t minValue,
    uint64_t maxValue) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return defaultValue;
  }

  errno = 0;
  char* end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || errno != 0) {
    VLOG(1) << "Ignoring invalid " << name << "=" << value;
    return defaultValue;
  }

  return std::clamp<uint64_t>(parsed, minValue, maxValue);
}

bool readBoolEnv(const char* name, bool defaultValue) {
  return readUint64Env(name, defaultValue ? 1 : 0, 0, 1) == 1;
}

void closeFd(int fd) {
  if (fd >= 0 && ::close(fd) != 0) {
    VLOG(1) << "Failed to close CPU SHM fd: " << errnoString();
  }
}

void unlinkShm(const std::string& name) {
  if (!name.empty() && ::shm_unlink(name.c_str()) != 0 && errno != ENOENT) {
    VLOG(1) << "Failed to unlink CPU SHM object " << name << ": "
            << errnoString();
  }
}

constexpr uint32_t kSlotPoolMagic = 0x56534350; // "VSCP"
constexpr uint32_t kSlotPoolVersion = 2;
constexpr uint32_t kSlotStateFree = 0;
constexpr uint32_t kSlotStateWriting = 1;
constexpr uint32_t kSlotStateReadyBase = 2;

struct UcxCpuRowShmSlotPoolHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t numSlots;
  uint32_t expectedOpeners;
  uint32_t openers;
  uint32_t reserved;
  uint64_t slotSize;
  uint64_t dataOffset;
  uint64_t totalSize;
};

size_t alignUp(size_t value, size_t alignment) {
  VELOX_CHECK_GT(alignment, 0);
  VELOX_CHECK_EQ(
      alignment & (alignment - 1),
      0,
      "CPU SHM alignment must be a power of two");
  VELOX_CHECK_LE(
      value,
      std::numeric_limits<size_t>::max() - (alignment - 1),
      "CPU SHM alignment overflow");
  return (value + alignment - 1) & ~(alignment - 1);
}

} // namespace

UcxCpuRowShmSegment::UcxCpuRowShmSegment(
    std::string _name,
    size_t _size,
    void* _address,
    int _fd,
    bool _unlinkOnDestroy)
    : name(std::move(_name)),
      size(_size),
      address(_address),
      fd(_fd),
      unlinkOnDestroy(_unlinkOnDestroy) {}

UcxCpuRowShmSegment::~UcxCpuRowShmSegment() {
  if (address != nullptr && address != MAP_FAILED && size > 0) {
    if (::munmap(address, size) != 0) {
      VLOG(1) << "Failed to unmap CPU SHM object " << name << ": "
              << errnoString();
    }
  }
  closeFd(fd);
  if (unlinkOnDestroy) {
    unlinkShm(name);
  }
}

bool ucxCpuRowShmEnabled() {
  static const bool kEnabled = readBoolEnv("VELOX_UCX_CPU_SHM", false);
  return kEnabled;
}

bool ucxCpuRowShmDirectTxEnabled() {
  static const bool kEnabled = [] {
    const char* value = std::getenv("VELOX_UCX_CPU_SHM_DIRECT_TX");
    if (value == nullptr || *value == '\0') {
      return ucxCpuRowShmEnabled();
    }
    return ucxCpuRowShmEnabled() &&
        readBoolEnv("VELOX_UCX_CPU_SHM_DIRECT_TX", false);
  }();
  return kEnabled;
}

bool ucxCpuRowShmSlotPoolEnabled() {
  static const bool kEnabled = ucxCpuRowShmEnabled() &&
      readBoolEnv("VELOX_UCX_CPU_SHM_SLOT_POOL", false);
  return kEnabled;
}

uint64_t ucxCpuRowShmSlotPoolSlotBytes() {
  static const uint64_t kBytes = [] {
    const uint64_t configuredPageBytes = readUint64Env(
        "VELOX_UCX_CPU_MAX_PAGE_BYTES", 16UL << 20, 64UL << 10, 1UL << 30);
    const uint64_t defaultSlotBytes = configuredPageBytes <= (1UL << 29)
        ? configuredPageBytes * 2
        : configuredPageBytes;
    return readUint64Env(
        "VELOX_UCX_CPU_SHM_SLOT_BYTES",
        defaultSlotBytes,
        64UL << 10,
        1UL << 30);
  }();
  return kBytes;
}

uint32_t ucxCpuRowShmSlotPoolInitialSlots() {
  static const uint32_t kNumSlots = static_cast<uint32_t>(
      readUint64Env("VELOX_UCX_CPU_SHM_SLOT_POOL_INITIAL_SLOTS", 8, 1, 1024));
  return kNumSlots;
}

uint32_t ucxCpuRowShmSlotPoolNumSlots() {
  static const uint32_t kNumSlots = static_cast<uint32_t>(
      readUint64Env("VELOX_UCX_CPU_SHM_SLOT_COUNT", 128, 1, 1024));
  return kNumSlots;
}

uint32_t ucxCpuRowShmSlotPoolMaxPools() {
  static const uint32_t kMaxPools = static_cast<uint32_t>(
      readUint64Env("VELOX_UCX_CPU_SHM_SLOT_POOL_MAX_POOLS", 5, 1, 1024));
  return kMaxPools;
}

std::shared_ptr<UcxCpuRowShmSegment> createUcxCpuRowShmSegment(size_t size) {
  if (size == 0) {
    return nullptr;
  }
  if (!sizeFitsOffT(size)) {
    VLOG(1) << "CPU SHM object size " << size << " exceeds off_t range";
    return nullptr;
  }

  for (int attempt = 0; attempt < 8; ++attempt) {
    auto name = nextShmName();
    int fd = ::shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
      if (errno == EEXIST) {
        continue;
      }
      VLOG(1) << "Failed to create CPU SHM object " << name << ": "
              << errnoString();
      return nullptr;
    }

    if (::ftruncate(fd, static_cast<off_t>(size)) != 0) {
      VLOG(1) << "Failed to resize CPU SHM object " << name << " to " << size
              << " bytes: " << errnoString();
      closeFd(fd);
      unlinkShm(name);
      return nullptr;
    }

    int reserveStatus = ::posix_fallocate(fd, 0, static_cast<off_t>(size));
    if (reserveStatus != 0) {
      VLOG(1) << "Failed to reserve CPU SHM object " << name << " to " << size
              << " bytes: " << std::strerror(reserveStatus);
      closeFd(fd);
      unlinkShm(name);
      return nullptr;
    }

    void* address =
        ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (address == MAP_FAILED) {
      VLOG(1) << "Failed to map CPU SHM object " << name << ": "
              << errnoString();
      closeFd(fd);
      unlinkShm(name);
      return nullptr;
    }

    return std::make_shared<UcxCpuRowShmSegment>(
        std::move(name), size, address, fd, false);
  }

  VLOG(1) << "Failed to allocate a unique CPU SHM object name";
  return nullptr;
}

std::shared_ptr<UcxCpuRowShmSegment> openUcxCpuRowShmSegment(
    std::string_view name,
    size_t size,
    bool unlinkAfterOpen,
    bool writable) {
  if (name.empty() || size == 0) {
    return nullptr;
  }

  std::string ownedName(name);
  int fd = ::shm_open(ownedName.c_str(), writable ? O_RDWR : O_RDONLY, 0600);
  if (fd < 0) {
    VLOG(1) << "Failed to open CPU SHM object " << ownedName << ": "
            << errnoString();
    return nullptr;
  }

  struct stat fileStat {};
  if (::fstat(fd, &fileStat) != 0) {
    VLOG(1) << "Failed to stat CPU SHM object " << ownedName << ": "
            << errnoString();
    closeFd(fd);
    return nullptr;
  }
  if (fileStat.st_size < 0 || static_cast<size_t>(fileStat.st_size) < size) {
    VLOG(1) << "CPU SHM object " << ownedName << " is too small: expected at "
            << "least " << size << " bytes, actual " << fileStat.st_size;
    closeFd(fd);
    return nullptr;
  }

  void* address = ::mmap(
      nullptr,
      size,
      writable ? (PROT_READ | PROT_WRITE) : PROT_READ,
      MAP_SHARED,
      fd,
      0);
  if (address == MAP_FAILED) {
    VLOG(1) << "Failed to map CPU SHM object " << ownedName << ": "
            << errnoString();
    closeFd(fd);
    return nullptr;
  }

  if (unlinkAfterOpen) {
    unlinkShm(ownedName);
  }

  return std::make_shared<UcxCpuRowShmSegment>(
      std::move(ownedName), size, address, fd, false);
}

std::shared_ptr<UcxCpuRowShmSlotPool> UcxCpuRowShmSlotPool::create(
    size_t slotSize,
    uint32_t numSlots,
    uint32_t expectedOpeners) {
  if (slotSize == 0 || numSlots == 0 || expectedOpeners == 0) {
    return nullptr;
  }

  const auto offset = dataOffset(numSlots);
  VELOX_CHECK_LE(
      slotSize,
      (std::numeric_limits<size_t>::max() - offset) / numSlots,
      "CPU SHM slot-pool size overflow");
  const size_t totalSize = offset + (slotSize * numSlots);
  auto segment = createUcxCpuRowShmSegment(totalSize);
  if (!segment) {
    return nullptr;
  }

  auto* header = reinterpret_cast<UcxCpuRowShmSlotPoolHeader*>(segment->data());
  *header = UcxCpuRowShmSlotPoolHeader{
      kSlotPoolMagic,
      kSlotPoolVersion,
      numSlots,
      expectedOpeners,
      0,
      0,
      static_cast<uint64_t>(slotSize),
      static_cast<uint64_t>(offset),
      static_cast<uint64_t>(totalSize)};

  auto* states = reinterpret_cast<uint32_t*>(
      segment->data() + sizeof(UcxCpuRowShmSlotPoolHeader));
  for (uint32_t i = 0; i < numSlots; ++i) {
    __atomic_store_n(&states[i], kSlotStateFree, __ATOMIC_RELEASE);
  }

  segment->unlinkOnDestroy = true;
  return std::shared_ptr<UcxCpuRowShmSlotPool>(
      new UcxCpuRowShmSlotPool(std::move(segment)));
}

std::shared_ptr<UcxCpuRowShmSlotPool> UcxCpuRowShmSlotPool::open(
    std::string_view name,
    size_t expectedSize,
    bool unlinkAfterExpectedOpeners) {
  auto segment = openUcxCpuRowShmSegment(name, expectedSize, false, true);
  if (!segment) {
    return nullptr;
  }

  if (segment->size < sizeof(UcxCpuRowShmSlotPoolHeader)) {
    VLOG(1) << "CPU SHM slot-pool " << segment->name
            << " is smaller than its header";
    return nullptr;
  }

  const auto* header =
      reinterpret_cast<const UcxCpuRowShmSlotPoolHeader*>(segment->data());
  if (header->magic != kSlotPoolMagic || header->version != kSlotPoolVersion ||
      header->numSlots == 0 || header->expectedOpeners == 0 ||
      header->openers > header->expectedOpeners || header->slotSize == 0 ||
      header->dataOffset == 0 || header->totalSize != segment->size) {
    VLOG(1) << "CPU SHM slot-pool " << segment->name << " has invalid header";
    return nullptr;
  }

  const auto expectedOffset = dataOffset(header->numSlots);
  if (header->dataOffset != expectedOffset) {
    VLOG(1) << "CPU SHM slot-pool " << segment->name
            << " has unexpected data offset";
    return nullptr;
  }

  VELOX_CHECK_LE(
      static_cast<size_t>(header->slotSize),
      (std::numeric_limits<size_t>::max() - expectedOffset) / header->numSlots,
      "CPU SHM slot-pool size overflow");
  const size_t computedSize = expectedOffset +
      (static_cast<size_t>(header->slotSize) *
       static_cast<size_t>(header->numSlots));
  if (computedSize != segment->size) {
    VLOG(1) << "CPU SHM slot-pool " << segment->name
            << " size does not match header";
    return nullptr;
  }

  auto pool = std::shared_ptr<UcxCpuRowShmSlotPool>(
      new UcxCpuRowShmSlotPool(std::move(segment)));
  if (unlinkAfterExpectedOpeners) {
    pool->recordOpenAndMaybeUnlink();
  }
  return pool;
}

UcxCpuRowShmSlotPool::UcxCpuRowShmSlotPool(
    std::shared_ptr<UcxCpuRowShmSegment> segment)
    : segment_(std::move(segment)) {
  const auto* header =
      reinterpret_cast<const UcxCpuRowShmSlotPoolHeader*>(segment_->data());
  numSlots_ = header->numSlots;
  expectedOpeners_ = header->expectedOpeners;
  slotSize_ = static_cast<size_t>(header->slotSize);
  dataOffset_ = static_cast<size_t>(header->dataOffset);
}

const std::string& UcxCpuRowShmSlotPool::name() const {
  return segment_->name;
}

void UcxCpuRowShmSlotPool::disableUnlinkOnDestroy() {
  segment_->unlinkOnDestroy = false;
}

size_t UcxCpuRowShmSlotPool::totalSize() const {
  return segment_->size;
}

size_t UcxCpuRowShmSlotPool::slotSize() const {
  return slotSize_;
}

uint32_t UcxCpuRowShmSlotPool::numSlots() const {
  return numSlots_;
}

uint32_t UcxCpuRowShmSlotPool::expectedOpeners() const {
  return expectedOpeners_;
}

std::optional<UcxCpuRowShmSlotPool::Slot> UcxCpuRowShmSlotPool::tryAcquire(
    size_t bytes) {
  if (bytes == 0 || bytes > slotSize_ || numSlots_ == 0) {
    return std::nullopt;
  }

  const uint32_t start = nextSlot_++ % numSlots_;
  for (uint32_t i = 0; i < numSlots_; ++i) {
    const uint32_t slotId = (start + i) % numSlots_;
    uint32_t expected = kSlotStateFree;
    if (__atomic_compare_exchange_n(
            statePtr(slotId),
            &expected,
            kSlotStateWriting,
            false,
            __ATOMIC_ACQ_REL,
            __ATOMIC_ACQUIRE)) {
      return Slot{slotId, slotData(slotId)};
    }
  }
  return std::nullopt;
}

void UcxCpuRowShmSlotPool::markReady(uint32_t slotId) {
  VELOX_CHECK_LT(slotId, numSlots_, "CPU SHM slot id is out of range");
  // The ready state stores a reference count biased by
  // kSlotStateReadyBase. The initial reference is the producer-side
  // payload/queue ownership; each advertised receiver gets an extra
  // reference before metadata is sent.
  __atomic_store_n(statePtr(slotId), kSlotStateReadyBase, __ATOMIC_RELEASE);
}

bool UcxCpuRowShmSlotPool::isReady(uint32_t slotId) const {
  if (slotId >= numSlots_) {
    return false;
  }
  return __atomic_load_n(statePtr(slotId), __ATOMIC_ACQUIRE) >=
      kSlotStateReadyBase;
}

bool UcxCpuRowShmSlotPool::addRef(uint32_t slotId) {
  if (slotId >= numSlots_) {
    return false;
  }

  auto* state = statePtr(slotId);
  uint32_t current = __atomic_load_n(state, __ATOMIC_ACQUIRE);
  while (current >= kSlotStateReadyBase) {
    VELOX_CHECK_LT(
        current,
        std::numeric_limits<uint32_t>::max(),
        "CPU SHM slot refcount overflow");
    const uint32_t desired = current + 1;
    if (__atomic_compare_exchange_n(
            state,
            &current,
            desired,
            false,
            __ATOMIC_ACQ_REL,
            __ATOMIC_ACQUIRE)) {
      return true;
    }
  }
  return false;
}

void UcxCpuRowShmSlotPool::release(uint32_t slotId) {
  if (slotId >= numSlots_) {
    return;
  }

  auto* state = statePtr(slotId);
  uint32_t current = __atomic_load_n(state, __ATOMIC_ACQUIRE);
  while (current >= kSlotStateReadyBase) {
    const uint32_t desired =
        current == kSlotStateReadyBase ? kSlotStateFree : current - 1;
    if (__atomic_compare_exchange_n(
            state,
            &current,
            desired,
            false,
            __ATOMIC_ACQ_REL,
            __ATOMIC_ACQUIRE)) {
      return;
    }
  }
}

uint8_t* UcxCpuRowShmSlotPool::slotData(uint32_t slotId) const {
  VELOX_CHECK_LT(slotId, numSlots_, "CPU SHM slot id is out of range");
  return segment_->data() + dataOffset_ +
      (static_cast<size_t>(slotId) * slotSize_);
}

size_t UcxCpuRowShmSlotPool::dataOffset(uint32_t numSlots) {
  return alignUp(
      sizeof(UcxCpuRowShmSlotPoolHeader) +
          (static_cast<size_t>(numSlots) * sizeof(uint32_t)),
      64);
}

uint32_t* UcxCpuRowShmSlotPool::statePtr(uint32_t slotId) const {
  VELOX_CHECK_LT(slotId, numSlots_, "CPU SHM slot id is out of range");
  return reinterpret_cast<uint32_t*>(
      segment_->data() + sizeof(UcxCpuRowShmSlotPoolHeader) +
      (static_cast<size_t>(slotId) * sizeof(uint32_t)));
}

void UcxCpuRowShmSlotPool::recordOpenAndMaybeUnlink() {
  auto* header =
      reinterpret_cast<UcxCpuRowShmSlotPoolHeader*>(segment_->data());
  const uint32_t opened =
      __atomic_add_fetch(&header->openers, 1, __ATOMIC_ACQ_REL);
  if (opened == expectedOpeners_) {
    unlinkShm(segment_->name);
  } else if (opened > expectedOpeners_) {
    VLOG(1) << "CPU SHM slot-pool " << segment_->name
            << " opened more times than expected: expected " << expectedOpeners_
            << ", opened " << opened;
  }
}

} // namespace facebook::velox::ucx_exchange
