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

#include <ucxx/api.h>
#include <map>
#include <set>
#include <unordered_map>
#include "velox/common/Enums.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/exec/Exchange.h"
#include "velox/experimental/ucx-exchange/CommElement.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/EndpointRef.h"
#include "velox/experimental/ucx-exchange/PartitionKey.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowExchangeQueue.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowMetadataMsg.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"

/// CPU RowVector mirror of UcxExchangeSource. Keeps the standard
/// handshake-then-recv-loop, backpressure, and at-most-once end-marker
/// delivery over UCX.

namespace facebook::velox::ucx_exchange {

struct UcxCpuRowShmSegment;
class UcxCpuRowShmSlotPool;

struct UcxCpuRowExchangeMetrics {
  UcxCpuRowExchangeMetrics()
      : numPayloads_(RuntimeCounter::Unit::kNone),
        totalBytes_(RuntimeCounter::Unit::kBytes),
        rttPerRequest_(RuntimeCounter::Unit::kNanos) {}
  RuntimeMetric numPayloads_;
  RuntimeMetric totalBytes_;
  RuntimeMetric rttPerRequest_;
};

class UcxCpuRowExchangeSource
    : public CommElement,
      public std::enable_shared_from_this<UcxCpuRowExchangeSource> {
 public:
  enum class ReceiverState : uint32_t {
    Created,
    WaitingForHandshakeComplete,
    WaitingForHandshakeResponse,
    ReadyToReceive,
    WaitingForMetadata,
    WaitingForData,
    Done,
  };

  VELOX_DECLARE_EMBEDDED_ENUM_NAME(ReceiverState);

  virtual ~UcxCpuRowExchangeSource() = default;

  static std::shared_ptr<UcxCpuRowExchangeSource> create(
      std::string_view taskId,
      std::string_view url,
      const std::shared_ptr<UcxCpuRowExchangeQueue>& queue);

  /// Starts communicator processing for this source. Call after the
  /// owning exchange queue has registered the source so any early
  /// failure can wake consumers correctly.
  void start();

  bool supportsMetrics() const {
    return true;
  }

  void process() override;

  void close();

  /// Marks this source as registered with the queue; only after this is
  /// set will deliverEndMarker() actually enqueue nullptr (otherwise we
  /// would spuriously increment numCompleted_ for unregistered sources).
  void setRegistered() {
    registered_.store(true, std::memory_order_release);
  }

  /// Called by UcxCpuRowExchangeClient::next() on the consumer thread to
  /// wake this source after it went dormant due to backpressure. CAS
  /// ensures exactly one wake-up per dormant period.
  void resumeFromBackpressure();

  // With 64 MB bundled sends we want a tall receive queue before we stall the
  // sender. The defaults cap at ~16 GB per destination while giving the
  // Communicator progress thread enough slack to keep landing tagSends.
  static constexpr int32_t kDefaultBackpressureHighWaterMark = 256;
  static constexpr int32_t kDefaultBackpressureLowWaterMark = 128;

  static int32_t backpressureHighWaterMark();

  static int32_t backpressureLowWaterMark();

  folly::F14FastMap<std::string, int64_t> stats() const;

  folly::F14FastMap<std::string, RuntimeMetric> metrics() const;

  std::string toString() const {
    std::stringstream out;
    out << "@" << taskId_ << " - @" << partitionKey_.toString();
    return out.str();
  }

  folly::dynamic toJson() const {
    folly::dynamic obj = folly::dynamic::object;
    obj["remoteTaskId"] = partitionKey_.taskId;
    obj["destination"] = partitionKey_.destination;
    obj["closed"] = std::to_string(closed_);
    return obj;
  }

 private:
  // Bundles a metadata record with N receiver buffers (one per frame).
  // Each frame is one chunk on the wire; we allocate per-frame so we
  // can stitch them into an IOBuf chain without a coalesce on receive.
  // pendingFrames counts down across the per-frame tagRecv callbacks;
  // when it hits zero we build the chain and enqueue.
  struct DataAndMetadata {
    uint32_t sequenceNumber{0};
    UcxCpuRowMetadataMsg metadata;
    std::vector<std::shared_ptr<uint8_t>> frameBufs;
    std::atomic<int32_t> pendingFrames{0};
    std::atomic<ucs_status_t> finalStatus{UCS_OK};
  };

  explicit UcxCpuRowExchangeSource(
      const std::shared_ptr<Communicator> communicator,
      std::string_view taskId,
      std::string_view host,
      uint16_t port,
      const PartitionKey& partitionKey,
      const std::shared_ptr<UcxCpuRowExchangeQueue> queue);

  static PartitionKey extractTaskAndDestinationId(std::string_view path);

  std::shared_ptr<UcxCpuRowExchangeSource> getSelfPtr();

  void enqueue(UcxCpuRowReceivedPtr data);

  void setEndpoint(std::shared_ptr<EndpointRef> endpointRef);

  void sendHandshake();

  void onHandshake(ucs_status_t status, std::shared_ptr<void> arg);

  void receiveHandshakeResponse();

  void onHandshakeResponse(ucs_status_t status, std::shared_ptr<void> arg);

  void postReceiveWindow();

  void getMetadata(uint32_t sequenceNumber);

  void onMetadata(
      uint32_t sequenceNumber,
      ucs_status_t status,
      std::shared_ptr<void> arg);

  void onData(
      uint32_t sequenceNumber,
      ucs_status_t status,
      std::shared_ptr<void> arg);

  void drainCompletedData();

  void setState(ReceiverState newState);

  ReceiverState getState() {
    return state_.load(std::memory_order_seq_cst);
  }

  void cleanUp();

  void deliverEndMarker();

  bool setStateIf(ReceiverState expected, ReceiverState desired);

  const std::string host_;
  uint16_t port_;
  const std::string taskId_;

  const PartitionKey partitionKey_;
  const uint32_t partitionKeyHash_;

  std::atomic<ReceiverState> state_;

  uint32_t nextMetadataSequence_{0};
  uint32_t nextDeliverySequence_{0};

  std::set<uint32_t> inFlightSequences_;
  std::map<uint32_t, UcxCpuRowReceivedPtr> completedData_;

  const std::shared_ptr<UcxCpuRowExchangeQueue> queue_{nullptr};
  std::atomic<bool> closed_{false};
  bool atEnd_{false};
  uint32_t endSequence_{0};

  std::atomic<bool> endMarkerDelivered_{false};
  std::atomic<bool> registered_{false};
  std::atomic<bool> backpressureActive_{false};

  UcxCpuRowExchangeMetrics metrics_;

  std::shared_ptr<ucxx::Request> handshakeRequest_{nullptr};
  std::shared_ptr<ucxx::Request> handshakeResponseRequest_{nullptr};
  std::vector<std::shared_ptr<ucxx::Request>> metadataRequests_;
  std::vector<std::shared_ptr<ucxx::Request>> dataRequests_;
  std::shared_ptr<UcxCpuRowShmSegment> handshakeShmProbe_;
  std::unordered_map<std::string, std::shared_ptr<UcxCpuRowShmSlotPool>>
      shmSlotPools_;
};

} // namespace facebook::velox::ucx_exchange
