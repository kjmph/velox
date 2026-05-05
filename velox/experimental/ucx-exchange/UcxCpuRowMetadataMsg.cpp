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
#include "velox/experimental/ucx-exchange/UcxCpuRowMetadataMsg.h"

#include <cstring>
#include <limits>
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::ucx_exchange {
namespace {

void checkRemaining(
    const uint8_t* ptr,
    const uint8_t* endPtr,
    size_t bytes,
    const char* field) {
  VELOX_CHECK(
      ptr <= endPtr,
      "CPU row metadata cursor advanced past end while reading {}",
      field);
  VELOX_CHECK_LE(
      bytes,
      static_cast<size_t>(endPtr - ptr),
      "Insufficient data for {}",
      field);
}

template <typename T>
T readScalar(const uint8_t*& ptr, const uint8_t* endPtr, const char* field) {
  checkRemaining(ptr, endPtr, sizeof(T), field);
  T value;
  std::memcpy(&value, ptr, sizeof(T));
  ptr += sizeof(T);
  return value;
}

void checkElementBytes(
    const uint8_t* ptr,
    const uint8_t* endPtr,
    WireLengthType count,
    size_t elementSize,
    const char* field) {
  VELOX_CHECK(
      ptr <= endPtr,
      "CPU row metadata cursor advanced past end while reading {}",
      field);
  VELOX_CHECK_LE(
      count,
      static_cast<WireLengthType>((endPtr - ptr) / elementSize),
      "CPU row metadata {} count {} exceeds remaining metadata bytes",
      field,
      count);
}

} // namespace

std::pair<std::shared_ptr<uint8_t>, size_t> UcxCpuRowMetadataMsg::serialize()
    const {
  uint32_t totalSize = getSerializedSize();
  VELOX_CHECK_LE(
      totalSize,
      kMaxMetaBufSize,
      "UcxCpuRowMetadataMsg size ({}) exceeds {}",
      totalSize,
      kMaxMetaBufSize);
  VELOX_CHECK_GE(dataSizeBytes, 0, "CPU row metadata data size is negative");

  auto deleter = [](uint8_t* p) { delete[] p; };
  std::shared_ptr<uint8_t> buffer(new uint8_t[totalSize], deleter);
  uint8_t* ptr = buffer.get();

  std::memcpy(ptr, &kMagicNumber, sizeof(kMagicNumber));
  ptr += sizeof(kMagicNumber);

  std::memcpy(ptr, &totalSize, sizeof(totalSize));
  ptr += sizeof(totalSize);

  std::memcpy(ptr, &dataSizeBytes, sizeof(dataSizeBytes));
  ptr += sizeof(dataSizeBytes);

  WireLengthType numFrames = frameSizes.size();
  std::memcpy(ptr, &numFrames, sizeof(numFrames));
  ptr += sizeof(numFrames);

  if (numFrames > 0) {
    auto bytesSize = numFrames * sizeof(frameSizes[0]);
    std::memcpy(ptr, frameSizes.data(), bytesSize);
    ptr += bytesSize;
  }

  WireLengthType numRemaining = remainingBytes.size();
  std::memcpy(ptr, &numRemaining, sizeof(numRemaining));
  ptr += sizeof(numRemaining);

  if (numRemaining > 0) {
    auto bytesSize = numRemaining * sizeof(remainingBytes[0]);
    std::memcpy(ptr, remainingBytes.data(), bytesSize);
    ptr += bytesSize;
  }

  *ptr = atEnd ? 1 : 0;
  ptr += sizeof(uint8_t);

  *ptr = static_cast<uint8_t>(transport);
  ptr += sizeof(uint8_t);

  VELOX_CHECK_LE(
      shmName.size(),
      std::numeric_limits<uint32_t>::max(),
      "CPU SHM name is too large");
  uint32_t shmNameSize = static_cast<uint32_t>(shmName.size());
  std::memcpy(ptr, &shmNameSize, sizeof(shmNameSize));
  ptr += sizeof(shmNameSize);

  if (shmNameSize > 0) {
    std::memcpy(ptr, shmName.data(), shmNameSize);
    ptr += shmNameSize;
  }

  WireLengthType numShmNames = shmNames.size();
  std::memcpy(ptr, &numShmNames, sizeof(numShmNames));
  ptr += sizeof(numShmNames);

  for (const auto& name : shmNames) {
    VELOX_CHECK_LE(
        name.size(),
        std::numeric_limits<uint32_t>::max(),
        "CPU SHM name is too large");
    uint32_t nameSize = static_cast<uint32_t>(name.size());
    std::memcpy(ptr, &nameSize, sizeof(nameSize));
    ptr += sizeof(nameSize);
    if (nameSize > 0) {
      std::memcpy(ptr, name.data(), nameSize);
      ptr += nameSize;
    }
  }

  if (!shmPoolName.empty() || !shmSlotIds.empty()) {
    VELOX_CHECK(
        transport == UcxCpuRowMetadataMsg::Transport::kShmSlot,
        "CPU SHM slot-pool metadata requires slot-pool transport");
    VELOX_CHECK_GT(shmPoolSize, 0, "CPU SHM slot-pool size must be positive");
    VELOX_CHECK_GT(shmSlotSize, 0, "CPU SHM slot size must be positive");
    VELOX_CHECK_LE(
        shmPoolName.size(),
        std::numeric_limits<uint32_t>::max(),
        "CPU SHM slot-pool name is too large");
    uint32_t poolNameSize = static_cast<uint32_t>(shmPoolName.size());
    std::memcpy(ptr, &poolNameSize, sizeof(poolNameSize));
    ptr += sizeof(poolNameSize);
    if (poolNameSize > 0) {
      std::memcpy(ptr, shmPoolName.data(), poolNameSize);
      ptr += poolNameSize;
    }

    std::memcpy(ptr, &shmPoolSize, sizeof(shmPoolSize));
    ptr += sizeof(shmPoolSize);
    std::memcpy(ptr, &shmSlotSize, sizeof(shmSlotSize));
    ptr += sizeof(shmSlotSize);

    WireLengthType numShmSlotIds = shmSlotIds.size();
    std::memcpy(ptr, &numShmSlotIds, sizeof(numShmSlotIds));
    ptr += sizeof(numShmSlotIds);
    if (numShmSlotIds > 0) {
      auto bytesSize = numShmSlotIds * sizeof(shmSlotIds[0]);
      std::memcpy(ptr, shmSlotIds.data(), bytesSize);
      ptr += bytesSize;
    }
  }

  return std::make_pair<std::shared_ptr<uint8_t>, size_t>(
      std::move(buffer), totalSize);
}

UcxCpuRowMetadataMsg UcxCpuRowMetadataMsg::deserialize(const uint8_t* buffer) {
  VELOX_CHECK_NOT_NULL(buffer, "Cannot deserialize CPU row metadata from null");
  const uint8_t* ptr = buffer;

  uint32_t magicNumber = 0;
  std::memcpy(&magicNumber, ptr, sizeof(magicNumber));
  VELOX_CHECK_EQ(magicNumber, kMagicNumber);
  ptr += sizeof(magicNumber);

  uint32_t totalSize = 0;
  std::memcpy(&totalSize, ptr, sizeof(totalSize));
  ptr += sizeof(totalSize);
  VELOX_CHECK_GE(
      totalSize,
      kMetaHeaderSize,
      "CPU row metadata total size {} is smaller than header size {}",
      totalSize,
      kMetaHeaderSize);
  VELOX_CHECK_LE(
      totalSize,
      kMaxMetaBufSize,
      "CPU row metadata total size {} exceeds receive buffer size {}",
      totalSize,
      kMaxMetaBufSize);

  const uint8_t* endPtr = buffer + totalSize;

  UcxCpuRowMetadataMsg msg;

  msg.dataSizeBytes =
      readScalar<WireDataSizeType>(ptr, endPtr, "dataSizeBytes");
  VELOX_CHECK_GE(
      msg.dataSizeBytes, 0, "CPU row metadata data size is negative");

  auto numFrames = readScalar<WireLengthType>(ptr, endPtr, "frameSizes count");
  checkElementBytes(
      ptr, endPtr, numFrames, sizeof(WireDataSizeType), "frameSizes");

  msg.frameSizes.resize(numFrames);
  if (numFrames > 0) {
    auto bytesSize = numFrames * sizeof(msg.frameSizes[0]);
    checkRemaining(ptr, endPtr, bytesSize, "frameSizes values");
    std::memcpy(msg.frameSizes.data(), ptr, bytesSize);
    ptr += bytesSize;
  }

  auto numRemaining =
      readScalar<WireLengthType>(ptr, endPtr, "remainingBytes count");
  checkElementBytes(
      ptr,
      endPtr,
      numRemaining,
      sizeof(WireRemainingElementType),
      "remainingBytes");

  msg.remainingBytes.resize(numRemaining);
  if (numRemaining > 0) {
    auto bytesSize = numRemaining * sizeof(msg.remainingBytes[0]);
    checkRemaining(ptr, endPtr, bytesSize, "remainingBytes values");
    std::memcpy(msg.remainingBytes.data(), ptr, bytesSize);
    ptr += bytesSize;
  }

  checkRemaining(ptr, endPtr, sizeof(uint8_t), "atEnd flag");
  msg.atEnd = (*ptr != 0);
  ptr += sizeof(uint8_t);

  // Older CPU metadata records end after atEnd. Treat them as UCX data.
  if (ptr == endPtr) {
    return msg;
  }

  checkRemaining(ptr, endPtr, sizeof(uint8_t), "transport");
  auto transport = static_cast<UcxCpuRowMetadataMsg::Transport>(*ptr);
  VELOX_CHECK(
      transport == UcxCpuRowMetadataMsg::Transport::kUcx ||
          transport == UcxCpuRowMetadataMsg::Transport::kShm ||
          transport == UcxCpuRowMetadataMsg::Transport::kShmSlot,
      "Unknown CPU row metadata transport {}",
      static_cast<uint8_t>(transport));
  msg.transport = transport;
  ptr += sizeof(uint8_t);

  if (ptr == endPtr) {
    return msg;
  }

  auto shmNameSize = readScalar<uint32_t>(ptr, endPtr, "shmName size");

  if (shmNameSize > 0) {
    checkRemaining(ptr, endPtr, shmNameSize, "shmName");
    msg.shmName.assign(reinterpret_cast<const char*>(ptr), shmNameSize);
    ptr += shmNameSize;
  }

  if (ptr == endPtr) {
    return msg;
  }

  auto numShmNames = readScalar<WireLengthType>(ptr, endPtr, "shmNames count");
  checkElementBytes(ptr, endPtr, numShmNames, sizeof(uint32_t), "shmNames");

  msg.shmNames.reserve(numShmNames);
  for (WireLengthType i = 0; i < numShmNames; ++i) {
    auto nameSize = readScalar<uint32_t>(ptr, endPtr, "shmName entry size");

    checkRemaining(ptr, endPtr, nameSize, "shmName entry");
    msg.shmNames.emplace_back(reinterpret_cast<const char*>(ptr), nameSize);
    ptr += nameSize;
  }

  if (ptr == endPtr) {
    return msg;
  }

  VELOX_CHECK(
      msg.transport == UcxCpuRowMetadataMsg::Transport::kShmSlot,
      "CPU row metadata has slot-pool fields without slot-pool transport");

  auto poolNameSize = readScalar<uint32_t>(ptr, endPtr, "shmPoolName size");
  if (poolNameSize > 0) {
    checkRemaining(ptr, endPtr, poolNameSize, "shmPoolName");
    msg.shmPoolName.assign(reinterpret_cast<const char*>(ptr), poolNameSize);
    ptr += poolNameSize;
  }

  msg.shmPoolSize = readScalar<WireDataSizeType>(ptr, endPtr, "shmPoolSize");
  VELOX_CHECK_GT(msg.shmPoolSize, 0, "CPU SHM slot-pool size must be positive");

  msg.shmSlotSize = readScalar<WireDataSizeType>(ptr, endPtr, "shmSlotSize");
  VELOX_CHECK_GT(msg.shmSlotSize, 0, "CPU SHM slot size must be positive");

  auto numShmSlotIds =
      readScalar<WireLengthType>(ptr, endPtr, "shmSlotIds count");
  checkElementBytes(
      ptr, endPtr, numShmSlotIds, sizeof(WireLengthType), "shmSlotIds");

  msg.shmSlotIds.resize(numShmSlotIds);
  if (numShmSlotIds > 0) {
    auto bytesSize = numShmSlotIds * sizeof(msg.shmSlotIds[0]);
    checkRemaining(ptr, endPtr, bytesSize, "shmSlotIds values");
    std::memcpy(msg.shmSlotIds.data(), ptr, bytesSize);
    ptr += bytesSize;
  }

  const auto trailingBytes = endPtr - ptr;
  VELOX_CHECK_EQ(
      trailingBytes,
      0,
      "CPU row metadata has {} trailing bytes",
      trailingBytes);

  return msg;
}

} // namespace facebook::velox::ucx_exchange
