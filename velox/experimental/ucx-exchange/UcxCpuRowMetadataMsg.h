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

#include <cinttypes>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"

/// CPU RowVector metadata wire format for the UCX exchange.
///
/// Mirrors the cudf MetadataMsg in UcxExchangeProtocol but drops the
/// `cudfMetadata` field; `PrestoVectorSerde`-produced bytes are
/// self-describing so no out-of-band metadata is needed.

namespace facebook::velox::ucx_exchange {

/// Wire-format metadata sent ahead of a RowVector data chunk.
///
/// Format on the wire:
///   [magic       (4 bytes)]
///   [totalSize   (4 bytes)]
///   [dataSize    (8 bytes)]   -- sum of all frameSizes (kept for
///                                end-marker sends and stats)
///   [numFrames   (8 bytes)]   -- number of frames being sent
///   [frameSizes  (numFrames * 8)]  -- size of each frame in bytes
///   [numRem      (8 bytes)]   -- number of `remainingBytes` entries
///   [remBytes    (numRem * 8)]
///   [atEnd       (1 byte)]
///   [transport   (1 byte)]    -- optional; defaults to UCX for older records
///   [shmNameSize (4 bytes)]   -- optional; only present with transport
///   [shmName     (N bytes)]
///   [numShmNames (8 bytes)]   -- optional; 1 SHM object per frame
///   [shmNames    (numShmNames * (4-byte size + bytes))]
///   [shmPoolNameSize (4 bytes)] -- optional; SHM slot-pool name
///   [shmPoolName     (N bytes)]
///   [shmPoolSize     (8 bytes)]
///   [shmSlotSize     (8 bytes)]
///   [numShmSlotIds   (8 bytes)]
///   [shmSlotIds      (numShmSlotIds * 8)]
///
/// UCX data bundles become N tagSends, one per frame, while SHM bundles
/// either carry the same frame layout inside one shared-memory object
/// (`shmName`), one already-serialized SHM object per frame (`shmNames`),
/// or one pre-mapped shared-memory slot per frame (`shmPoolName` +
/// `shmSlotIds`).
/// The receiver stitches frames into an IOBuf chain, which PrestoSerializer's
/// stream reader walks natively.
struct UcxCpuRowMetadataMsg {
  enum class Transport : uint8_t {
    kUcx = 0,
    kShm = 1,
    kShmSlot = 2,
  };

  WireDataSizeType dataSizeBytes{0};
  std::vector<WireDataSizeType> frameSizes;
  std::vector<WireRemainingElementType> remainingBytes;
  bool atEnd{false};
  Transport transport{Transport::kUcx};
  std::string shmName;
  std::vector<std::string> shmNames;
  std::string shmPoolName;
  WireDataSizeType shmPoolSize{0};
  WireDataSizeType shmSlotSize{0};
  std::vector<WireLengthType> shmSlotIds;

  uint32_t getSerializedSize() const {
    uint64_t totalSize = sizeof(kMagicNumber) + sizeof(uint32_t);
    totalSize += sizeof(dataSizeBytes);
    totalSize += sizeof(WireLengthType); // numFrames count
    totalSize += frameSizes.size() * sizeof(frameSizes[0]);
    totalSize += sizeof(WireLengthType); // numRemaining count
    totalSize += remainingBytes.size() * sizeof(remainingBytes[0]);
    totalSize += sizeof(uint8_t); // atEnd
    totalSize += sizeof(uint8_t); // transport
    totalSize += sizeof(uint32_t); // shmName size
    totalSize += shmName.size();
    totalSize += sizeof(WireLengthType); // shmNames count
    for (const auto& name : shmNames) {
      totalSize += sizeof(uint32_t);
      totalSize += name.size();
    }
    if (!shmPoolName.empty() || !shmSlotIds.empty()) {
      totalSize += sizeof(uint32_t); // shmPoolName size
      totalSize += shmPoolName.size();
      totalSize += sizeof(shmPoolSize);
      totalSize += sizeof(shmSlotSize);
      totalSize += sizeof(WireLengthType); // shmSlotIds count
      totalSize += shmSlotIds.size() * sizeof(shmSlotIds[0]);
    }
    if (totalSize > std::numeric_limits<uint32_t>::max()) {
      return std::numeric_limits<uint32_t>::max();
    }
    return static_cast<uint32_t>(totalSize);
  }

  /// Serializes this metadata record into a newly allocated buffer. The
  /// returned shared_ptr owns a `new uint8_t[size]`.
  std::pair<std::shared_ptr<uint8_t>, size_t> serialize() const;

  /// Deserializes a UcxCpuRowMetadataMsg from a buffer produced by serialize().
  static UcxCpuRowMetadataMsg deserialize(const uint8_t* buffer);
};

} // namespace facebook::velox::ucx_exchange
