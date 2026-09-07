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
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
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
constexpr uint64_t CPU_ROW_ABORT_TAG = 0x06000000;
constexpr uint64_t GPU_HANDSHAKE_ACK_TAG = 0x07000000;
constexpr uint64_t GPU_ABORT_TAG = 0x08000000;

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

// Gets the tag used by a CPU-row source to stop one producing partition.
// The producer responds in-band: after draining its current send, it publishes
// the normal final metadata record at the next sequence number.
inline uint64_t getCpuRowAbortTag(uint64_t taskHash) {
  return (taskHash << 32) | CPU_ROW_ABORT_TAG;
}

// Gets the tag used for GPU data-endpoint ACK communication.
// Note: taskHash is implicitly converted to 64 bits.
inline uint64_t getGpuHandshakeAckTag(uint64_t taskHash) {
  return (taskHash << 32) | GPU_HANDSHAKE_ACK_TAG;
}

// Gets the tag used by a GPU source to stop one producing partition.
// The producer responds in-band: after draining its current send, it publishes
// the normal final metadata record at the next sequence number.
inline uint64_t getGpuAbortTag(uint64_t taskHash) {
  return (taskHash << 32) | GPU_ABORT_TAG;
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
constexpr uint32_t kCpuRowAbortMagic = 0x43504142; // "CPAB"
constexpr uint16_t kCpuRowAbortVersion = 1;
constexpr uint32_t kMaxCpuRowWorkerAddressBytes = 4096;

enum class CpuRowDataEndpointMode : uint8_t {
  kBootstrap = 0,
  kSameHostWorkerAddress = 1,
  // Rejected responses use a mode that old version-1 sources already reject
  // as unsupported, preventing them from waiting forever for metadata.
  kRejected = 0xff,
};

/// Admission result returned by the producer in the CPU-row handshake.
/// This occupies one byte that was reserved in version 1 of the response, so
/// accepted handshakes remain wire-compatible with existing workers.
enum class CpuRowHandshakeResponseStatus : uint8_t {
  kAccepted = 0,
  kTaskRemoved = 1,
  kDuplicateServer = 2,
  kServerUnavailable = 3,
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
  CpuRowHandshakeResponseStatus status{
      CpuRowHandshakeResponseStatus::kAccepted};
  uint8_t reserved[2]{};
  uint32_t serverWorkerAddressBytes{0};
  uint32_t serverHostIdHash{0};
};

static_assert(std::is_standard_layout_v<CpuRowHandshakeResponseHeader>);
static_assert(sizeof(CpuRowHandshakeResponseHeader) == 20);
static_assert(offsetof(CpuRowHandshakeResponseHeader, dataEndpointMode) == 8);
static_assert(offsetof(CpuRowHandshakeResponseHeader, status) == 9);
static_assert(
    offsetof(CpuRowHandshakeResponseHeader, serverWorkerAddressBytes) == 12);

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

/// Per-partition CPU-row abort request. The source sends this only after its
/// handshake was accepted. The server stops pulling new output, lets its
/// single in-flight bundle finish, then sends the ordinary final metadata
/// marker. Reusing the sequenced final marker prevents an abort ACK from
/// overtaking data that the source still needs to drain.
struct CpuRowAbortHeader {
  uint32_t magic{kCpuRowAbortMagic};
  uint16_t version{kCpuRowAbortVersion};
  uint16_t headerSize{sizeof(CpuRowAbortHeader)};
};

static_assert(std::is_standard_layout_v<CpuRowAbortHeader>);
static_assert(sizeof(CpuRowAbortHeader) == 8);

constexpr uint32_t kGpuHandshakeAckMagic = 0x4750414b; // "GPAK"
constexpr uint16_t kGpuHandshakeAckVersion = 1;
constexpr uint32_t kGpuAbortMagic = 0x47504142; // "GPAB"
constexpr uint16_t kGpuAbortVersion = 1;

/// GPU data-endpoint ACK. The source sends this after it has posted its first
/// receive (or started polling the intra-process registry). The producer does
/// not pull output until this ACK arrives.
struct GpuHandshakeAckHeader {
  uint32_t magic{kGpuHandshakeAckMagic};
  uint16_t version{kGpuHandshakeAckVersion};
  uint16_t headerSize{sizeof(GpuHandshakeAckHeader)};
};

/// Per-partition GPU abort request. The source drains the at-most-one committed
/// packed table and then consumes the producer's sequenced final metadata
/// marker before releasing receive requests and CUDA buffers.
struct GpuAbortHeader {
  uint32_t magic{kGpuAbortMagic};
  uint16_t version{kGpuAbortVersion};
  uint16_t headerSize{sizeof(GpuAbortHeader)};
};

static_assert(std::is_standard_layout_v<GpuHandshakeAckHeader>);
static_assert(sizeof(GpuHandshakeAckHeader) == 8);
static_assert(std::is_standard_layout_v<GpuAbortHeader>);
static_assert(sizeof(GpuAbortHeader) == 8);

enum class GpuHandshakeResponseStatus : uint8_t {
  kAccepted = 0,
  kTaskRemoved = 1,
  kDuplicateServer = 2,
  kServerUnavailable = 3,
};

/// @brief Response sent from server to source after handshake.
/// The GPU exchange path uses this to report same-process intra-node transfer
/// availability. CPU row exchange uses CpuRowHandshakeResponseHeader instead.
struct HandshakeResponse {
  /// True if the GPU exchange source should use IntraNodeTransferRegistry.
  bool isIntraNodeTransfer{false};
  /// Admission result. This occupies a byte that was padding in the original
  /// response, preserving the eight-byte wire layout for accepted handshakes.
  GpuHandshakeResponseStatus status{GpuHandshakeResponseStatus::kAccepted};
  /// Padding for alignment and future protocol extensions.
  uint8_t padding[6]{};
};

static_assert(std::is_standard_layout_v<HandshakeResponse>);
static_assert(sizeof(HandshakeResponse) == 8);
static_assert(offsetof(HandshakeResponse, status) == 1);

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
using WireRowCountType = int32_t;
using WireRemainingElementType = int64_t;

struct MetadataMsg {
  std::unique_ptr<std::vector<uint8_t>> cudfMetadata;
  WireDataSizeType dataSizeBytes{0};
  /// Logical row count of the source vector. Unlike a cuDF table's row count,
  /// this remains meaningful for zero-column tables. -1 denotes metadata from
  /// an older sender that did not include this trailing field. Legacy records
  /// remain structurally parseable, but zero-column correctness requires a
  /// homogeneous rollout with this field present at both ends.
  WireRowCountType numRows{-1};
  std::vector<WireRemainingElementType> remainingBytes;
  bool atEnd{false};

  /// Whether the payload is device memory or host bytes. Written only when
  /// false, as a trailing byte, so device payloads stay byte-identical.
  bool isDeviceData{true};

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
    // Logical row count is a trailing extension, preserving structural wire
    // compatibility: older receivers ignore it and newer receivers can parse
    // legacy records. Zero-column logical rows require upgraded peers.
    totalSize += sizeof(numRows);
    if (!isDeviceData) {
      totalSize += sizeof(uint8_t);
    }

    return totalSize;
  }

  /// Serializes this metadata record into a newly allocated buffer.
  std::pair<std::shared_ptr<uint8_t>, size_t> serialize();

  /// Deserializes a MetadataMsg from a buffer produced by serialize().
  static MetadataMsg deserializeMetadataMsg(const uint8_t* buffer);
};

} // namespace facebook::velox::ucx_exchange
