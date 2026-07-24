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

#include <folly/Synchronized.h>
#include <folly/io/IOBuf.h>
#include <ucxx/api.h>
#include <ucxx/utils/ucx.h>
#include <velox/exec/Task.h>
#include <memory>
#include "velox/common/EnumDeclare.h"
#include "velox/experimental/ucx-exchange/CommElement.h"
#include "velox/experimental/ucx-exchange/EndpointRef.h"
#include "velox/experimental/ucx-exchange/PartitionKey.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowOutputQueueManager.h"

/// CPU RowVector mirror of UcxExchangeServer. Pulls serialized
/// PrestoVectorSerde payloads (UcxCpuRowPayload) from the producer-side
/// queue and ships them to a UcxCpuRowExchangeSource via UCX.

namespace facebook::velox::ucx_exchange {

class UcxCpuRowExchangeServer
    : public CommElement,
      public std::enable_shared_from_this<UcxCpuRowExchangeServer> {
 public:
  // Public for logging and the VELOX_DEFINE_EMBEDDED_ENUM_NAME names map.
  enum class ServerState : uint32_t {
    Created,
    WaitingForDataEndpointAck,
    ReadyToTransfer,
    WaitingForDataFromQueue,
    DataReady,
    WaitingForSendComplete,
    WaitingForFinalMetadataComplete,
    Done,
  };

  VELOX_DECLARE_EMBEDDED_ENUM_NAME(ServerState);

  /// Factory for UcxCpuRowExchangeServer.
  ///   communicator: the shared Communicator instance.
  ///   endpointRef: UCX endpoint for the connected source.
  ///   key: identifies the (taskId, destination) tuple this server feeds.
  static std::shared_ptr<UcxCpuRowExchangeServer> create(
      const std::shared_ptr<Communicator> communicator,
      std::shared_ptr<EndpointRef> endpointRef,
      const PartitionKey& key,
      bool waitForDataEndpointAck = false);

  void process() override;

  void close() override;

  std::string toString();

  const PartitionKey& getPartitionKey() const {
    return partitionKey_;
  }

  /// Posts the data-endpoint ACK receive before the handshake response is sent.
  void postDataEndpointAckReceive();

 private:
  explicit UcxCpuRowExchangeServer(
      const std::shared_ptr<Communicator> communicator,
      std::shared_ptr<EndpointRef> endpointRef,
      const PartitionKey& key,
      bool waitForDataEndpointAck);

  std::shared_ptr<UcxCpuRowExchangeServer> getSelfPtr();

  /// Sends metadata then payload bytes for one chunk to the source.
  void sendData(std::shared_ptr<UcxCpuRowPayload> firstChunk);

  /// Pulls producer queue data while the bounded UCX send window has room.
  void fillSendWindow();

  /// Installs one async queue waiter when no payload is immediately available.
  void requestData();

  /// Handles the producer queue callback.
  void onDataAvailable(std::shared_ptr<UcxCpuRowPayload> data);

  /// Sends the end-of-stream metadata record.
  void sendFinalMetadata();

  /// UCX completion handler for the data send.
  void sendComplete(ucs_status_t status, std::shared_ptr<void> arg);

  /// UCX completion handler for the final metadata send.
  void finalMetadataComplete(ucs_status_t status);

  /// UCX completion handler for the data-endpoint ACK.
  void onDataEndpointAck(ucs_status_t status, std::shared_ptr<void> arg);

  void maybeFinish();

  /// Sequentially-consistent state set.
  void setState(ServerState newState);

  ServerState getState() {
    return state_.load(std::memory_order_seq_cst);
  }

  const PartitionKey partitionKey_;
  // Hash of the partition-key string, used to derive UCX message tags.
  // See UcxExchangeProtocol.h for the tag layout.
  const uint32_t partitionKeyHash_;

  std::atomic<ServerState> state_;
  std::atomic<bool> closed_{false};

  uint32_t sequenceNumber_{0};
  uint32_t inFlightSends_{0};
  bool waitingForData_{false};
  bool waitForDataEndpointAck_{false};
  bool dataEndpointAckReceived_{false};
  bool hasPendingData_{false};
  bool finalMetadataSent_{false};
  bool finalMetadataCompleted_{false};

  std::shared_ptr<UcxCpuRowPayload> pendingData_;

  // The Request owns its callback closure; completed UCXX requests are
  // kept for the server lifetime because UCP wireup-replay can invoke
  // callbacks after ucxx marks a request complete.
  std::vector<std::shared_ptr<ucxx::Request>> metaRequests_;
  std::vector<std::shared_ptr<ucxx::Request>> dataRequests_;
  std::shared_ptr<ucxx::Request> dataEndpointAckRequest_{nullptr};

  std::shared_ptr<UcxCpuRowOutputQueueManager> queueMgr_;
};

} // namespace facebook::velox::ucx_exchange
