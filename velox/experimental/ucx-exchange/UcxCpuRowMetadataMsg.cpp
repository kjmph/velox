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

  const auto trailingBytes = endPtr - ptr;
  VELOX_CHECK_EQ(
      trailingBytes,
      0,
      "CPU row metadata has {} trailing bytes",
      trailingBytes);

  return msg;
}

} // namespace facebook::velox::ucx_exchange
