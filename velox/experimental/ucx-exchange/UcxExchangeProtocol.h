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
#include <memory>
#include <string>
#include <string_view>
#include <vector>

/// Definitions needed for the Ucx exchange protocol.
///
/// Byte order: all multi-byte fields are serialized with std::memcpy, which
/// preserves host byte order. The protocol assumes matching endianness between
/// peers (little-endian on x86 and ARM). Cross-endian transfers are not
/// supported.

namespace facebook::velox::ucx_exchange {

// Data and metadata tags are a uint64_t split into 3 fields, most-significant
// first:
// - Bits 63..32 (4 bytes): FNV-1a hash of the producing taskId, which is
//   unique within a cluster.
// - Bits 31..24 (1 byte): Operation type (metadata, data, or handshake
//   response).
// - Bits 23..0  (3 bytes): Sequence number of the chunk exchanged between 2
//   tasks.

// Definition of the operations.
constexpr uint64_t METADATA_TAG = 0x02000000;
constexpr uint64_t DATA_TAG = 0x03000000;
constexpr uint64_t HANDSHAKE_RESPONSE_TAG = 0x04000000;
constexpr uint64_t HANDSHAKE_ACK_TAG = 0x05000000;

// Implementation of the fowler-noll-vo hash function for 32 bits.
uint32_t fnv1a_32(std::string_view s);

// Gets the tag used for metadata communication
// Note: taskHash and sequenceNumber are implicitly converted to 64 bits.
inline uint64_t getMetadataTag(uint64_t taskHash, uint64_t sequenceNumber) {
  return (taskHash << 32) | METADATA_TAG | sequenceNumber;
}

// Gets the tag used for data communication
// Note: taskHash and sequenceNumber are implicitly converted to 64 bits.
inline uint64_t getDataTag(uint64_t taskHash, uint64_t sequenceNumber) {
  return (taskHash << 32) | DATA_TAG | sequenceNumber;
}

// Gets the tag used for handshake response communication.
// Note: taskHash is implicitly converted to 64 bits.
inline uint64_t getHandshakeResponseTag(uint64_t taskHash) {
  return (taskHash << 32) | HANDSHAKE_RESPONSE_TAG;
}

// Gets the tag used for CPU-row data-endpoint ACK communication.
// Note: taskHash is implicitly converted to 64 bits.
inline uint64_t getHandshakeAckTag(uint64_t taskHash) {
  return (taskHash << 32) | HANDSHAKE_ACK_TAG;
}

/// @brief Request that is sent from the client (UcxExchangeSource) to the
/// server (UcxExchangeServer) after connection.
///
/// The handshake establishes the partition key for data exchange.
/// The workerId identifies the source's Communicator instance (process).
/// If the server's workerId matches, both are in the same process, enabling
/// intra-node transfer via IntraNodeTransferRegistry instead of UCXX.
struct HandshakeMsg {
  char taskId[256];
  uint32_t destination;
  /// Unique identifier for the source's Communicator instance.
  /// Generated randomly at Communicator startup. The server compares this
  /// against its own workerId to detect same-process (intra-node) transfers.
  uint64_t workerId{0};
};

constexpr uint32_t kCpuRowHandshakeMagic = 0x43505558; // "CPUX"
constexpr uint16_t kCpuRowHandshakeVersion = 1;
constexpr uint32_t kCpuRowHandshakeResponseMagic = 0x43505258; // "CPRX"
constexpr uint16_t kCpuRowHandshakeResponseVersion = 1;
constexpr uint32_t kCpuRowHandshakeAckMagic = 0x4350414b; // "CPAK"
constexpr uint16_t kCpuRowHandshakeAckVersion = 1;
constexpr uint32_t kMaxCpuRowWorkerAddressBytes = 4096;

enum class CpuRowDataEndpointMode : uint8_t {
  kBootstrap = 0,
  kSameHostWorkerAddress = 1,
};

/// CPU-row handshake envelope. The legacy HandshakeMsg remains unchanged for
/// the GPU exchange path; CPU adds the source UCX worker address after this
/// header so the producer can create a worker-address endpoint back to the
/// source. Unlike UCX listener endpoints, worker-address endpoints let UCX
/// select intra-node transports such as posix, sysv, or cma.
struct CpuRowHandshakeHeader {
  uint32_t magic{kCpuRowHandshakeMagic};
  uint16_t version{kCpuRowHandshakeVersion};
  uint16_t headerSize{sizeof(CpuRowHandshakeHeader)};
  HandshakeMsg handshake;
  uint32_t sourceWorkerAddressBytes{0};
  /// Stable hash of the source same-host transport identity. Zero means
  /// unknown. CPU-row exchange uses this to disable UCX endpoint error
  /// handling only for compatible same-host data endpoints, allowing UCX to
  /// select posix/sysv/cma locally while preserving error handling for
  /// cross-host endpoints.
  uint32_t sourceHostIdHash{0};
};

/// CPU-row handshake response envelope. The listener endpoint remains the
/// control channel. When both sides are on the same host, this response carries
/// the producer worker address so the source can create the reciprocal UCX
/// worker-address data endpoint before posting metadata receives.
struct CpuRowHandshakeResponseHeader {
  uint32_t magic{kCpuRowHandshakeResponseMagic};
  uint16_t version{kCpuRowHandshakeResponseVersion};
  uint16_t headerSize{sizeof(CpuRowHandshakeResponseHeader)};
  CpuRowDataEndpointMode dataEndpointMode{CpuRowDataEndpointMode::kBootstrap};
  uint8_t reserved[3]{};
  uint32_t serverWorkerAddressBytes{0};
  uint32_t serverHostIdHash{0};
};

/// CPU-row data-endpoint ACK. The source sends this on the selected data
/// endpoint after it has created the endpoint and posted its first metadata
/// receive. The producer registers the server before the response, but does
/// not start metadata/data transfer on a worker-address endpoint until this
/// ACK arrives.
struct CpuRowHandshakeAckHeader {
  uint32_t magic{kCpuRowHandshakeAckMagic};
  uint16_t version{kCpuRowHandshakeAckVersion};
  uint16_t headerSize{sizeof(CpuRowHandshakeAckHeader)};
};

/// @brief Response sent from server to source after handshake.
/// The GPU exchange path uses this to report same-process intra-node transfer
/// availability. CPU row exchange uses CpuRowHandshakeResponseHeader instead.
struct HandshakeResponse {
  /// True if the GPU exchange source should use IntraNodeTransferRegistry.
  bool isIntraNodeTransfer{false};
  /// Padding for alignment
  uint8_t padding[7]{};
};

constexpr uint32_t kMagicNumber = 0x12345678;
/// Maximum metadata buffer size for receiving. This should be large enough
/// to handle tables with many columns. 1MB allows for ~10,000+ columns.
/// The sender allocates exact size needed; receiver pre-allocates this max.
constexpr uint32_t kMaxMetaBufSize = 1024 * 1024; // 1MB

/// Minimum header size needed to read the totalSize field.
/// Format: [magic (4 bytes)][totalSize (4 bytes)]
constexpr uint32_t kMetaHeaderSize = sizeof(kMagicNumber) + sizeof(uint32_t);

/// Wire-format types for MetadataMsg serialization. Using shared type aliases
/// ensures serialize() and deserializeMetadataMsg() agree on field widths.
using WireLengthType = uint64_t;
using WireDataSizeType = int64_t;
using WireRemainingElementType = int64_t;

struct MetadataMsg {
  std::unique_ptr<std::vector<uint8_t>> cudfMetadata;
  WireDataSizeType dataSizeBytes;
  std::vector<WireRemainingElementType> remainingBytes;
  bool atEnd;

  uint32_t getSerializedSize() const {
    // The header: the magic number and the metadata length.
    uint32_t totalSize = sizeof(kMagicNumber) + sizeof(totalSize);
    // cudfMetadata: length info and then the data.
    WireLengthType cudfSize = cudfMetadata ? cudfMetadata->size() : 0;
    totalSize += sizeof(cudfSize);
    totalSize += cudfSize;
    // dataSizeBytes
    totalSize += sizeof(dataSizeBytes);
    // remainingBytes: length and then the data.
    totalSize += sizeof(WireLengthType); // for numRemaining count
    totalSize += remainingBytes.size() * sizeof(remainingBytes[0]);
    // atEnd, encoded in a byte.
    totalSize += sizeof(uint8_t);

    return totalSize;
  }

  /// Serializes this metadata record into a newly allocated buffer.
  std::pair<std::shared_ptr<uint8_t>, size_t> serialize();

  /// Deserializes a MetadataMsg from a buffer produced by serialize().
  static MetadataMsg deserializeMetadataMsg(const uint8_t* buffer);
};

} // namespace facebook::velox::ucx_exchange
