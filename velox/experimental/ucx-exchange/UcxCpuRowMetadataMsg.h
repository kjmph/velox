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
///
/// UCX data bundles become N tagSends, one per frame.
/// The receiver stitches frames into an IOBuf chain, which PrestoSerializer's
/// stream reader walks natively.
struct UcxCpuRowMetadataMsg {
  WireDataSizeType dataSizeBytes{0};
  std::vector<WireDataSizeType> frameSizes;
  std::vector<WireRemainingElementType> remainingBytes;
  bool atEnd{false};

  uint32_t getSerializedSize() const {
    uint64_t totalSize = sizeof(kMagicNumber) + sizeof(uint32_t);
    totalSize += sizeof(dataSizeBytes);
    totalSize += sizeof(WireLengthType); // numFrames count
    totalSize += frameSizes.size() * sizeof(frameSizes[0]);
    totalSize += sizeof(WireLengthType); // numRemaining count
    totalSize += remainingBytes.size() * sizeof(remainingBytes[0]);
    totalSize += sizeof(uint8_t); // atEnd
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
