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

#include <deque>

#include <cudf/contiguous_split.hpp>
#include <folly/Synchronized.h>
#include <ucxx/api.h>
#include <ucxx/utils/ucx.h>
#include <velox/exec/Task.h>
#include <velox/experimental/ucx-exchange/DynamicUcxOutputBufferReader.h>
#include <velox/experimental/ucx-exchange/UcxOutputQueueManager.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "velox/common/EnumDeclare.h"
#include "velox/experimental/ucx-exchange/CommElement.h"
#include "velox/experimental/ucx-exchange/EndpointRef.h"
#include "velox/experimental/ucx-exchange/PartitionKey.h"

namespace facebook::velox::ucx_exchange {

class UcxExchangeServer
    : public CommElement,
      public UcxExchangeServerLifecycle,
      public std::enable_shared_from_this<UcxExchangeServer> {
 public:
  // Public for logging and the VELOX_DEFINE_EMBEDDED_ENUM_NAME names map.
  enum class ServerState : uint32_t {
    Created,
    WaitingForDataEndpointAck,
    ReadyToTransfer,
    WaitingForDataFromQueue,
    DataReady,
    WaitingForSendComplete,
    WaitingForIntraNodeRetrieve,
    WaitingForFinalMetadataComplete,
    Done,
  };

  VELOX_DECLARE_EMBEDDED_ENUM_NAME(ServerState);

  /// @brief Factory method to create a UcxExchangeServer.
  /// @param communicator The Communicator instance.
  /// @param endpointRef The endpoint reference for UCXX communication.
  /// @param key The partition key identifying the data to serve.
  /// @param isIntraNodeTransfer True if the source is on the same node,
  ///        determined by checking if the peer's IP is in the local IP set.
  static std::shared_ptr<UcxExchangeServer> create(
      const std::shared_ptr<Communicator> communicator,
      std::shared_ptr<EndpointRef> endpointRef,
      const PartitionKey& key,
      bool isIntraNodeTransfer);

  void process() override;

  void close() override;

  bool supportsCommunicatorShutdownDrain() const override {
    return true;
  }

  void beginCommunicatorShutdownDrain() override {
    requestAbort();
  }

  void forceCloseForShutdown() override;

  void requestAbort() override;

  bool activate() override;

  bool isClosed() const override {
    return closed_.load(std::memory_order_acquire);
  }

  std::string toString();

  const PartitionKey& getPartitionKey() const override {
    return partitionKey_;
  }

  /// @brief Returns true if this server detected same-node with the source.
  bool isIntraNodeTransfer() const {
    return isIntraNodeTransfer_;
  }

 private:
  enum class AdmissionState : uint8_t {
    Pending,
    Accepted,
    Rejected,
  };

  explicit UcxExchangeServer(
      const std::shared_ptr<Communicator> communicator,
      std::shared_ptr<EndpointRef> endpointRef,
      const PartitionKey& key,
      bool isIntraNodeTransfer);

  /// @return A shared pointer to itself.
  std::shared_ptr<UcxExchangeServer> getSelfPtr();

  /// Posts the data-endpoint ACK receive after admission and before the
  /// accepted handshake response is sent.
  void postDataEndpointAckReceive();

  /// Posts the per-partition abort receive after admission and before the
  /// accepted handshake response is sent.
  void postAbortReceive();

  /// @brief Sends metadata and data to the connected receiver.
  void sendData();

  /// Handles data becoming available from the output queue.
  void onDataAvailable(std::shared_ptr<UcxGpuPayload> data);

  /// Handles completion of one non-terminal metadata send.
  void metadataSendComplete(ucs_status_t status);

  /// @brief Completion handler after data has been sent.
  void sendComplete(ucs_status_t status);

  /// Sends the ordinary sequenced end-of-stream marker.
  void sendFinalMetadata();

  /// Handles completion of the terminal metadata send.
  void finalMetadataComplete(ucs_status_t status);

  /// Handles the data-endpoint ACK.
  void onDataEndpointAck(ucs_status_t status, std::shared_ptr<void> arg);

  /// Handles a per-partition abort request.
  void onAbortRequest(ucs_status_t status, std::shared_ptr<void> arg);

  /// Stops queue pulls, drains the committed transfer, and publishes the
  /// ordinary sequenced end-of-stream marker.
  void processAbort();

  /// Starts one asynchronous output-queue request.
  void requestData();

  /// Installs a callback on the stable producer queue. A callback that races
  /// server destruction releases any dequeued in-flight accounting itself.
  void installDataCallback();

  // Arranges the next payload when the pages live in the ordinary output
  // buffer.
  void requestDynamicUcxData();

  // Turns one drained batch into payloads and delivers the first.
  void onDynamicUcxData(DynamicUcxOutputBufferReader::Data data);

  // Delivers the next buffered payload, or a null one to signal end of
  // stream, which is what onDataAvailable already expects.
  void deliverDynamicUcxPage();

  // Undoes what the dynamic UCX path set up. Idempotent. 'releaseBuffer' is
  // right only when the stream finished.
  void releaseDynamicUcxResources(bool releaseBuffer);

  /// Completes a remote data bundle once both tag sends have completed.
  void maybeCompleteRemoteSend();

  /// Polls the current intra-process retrieval future.
  bool pollIntraNodeRetrieve();

  /// @brief Completion handler for intra-node transfer after source retrieves
  /// data.
  void onIntraNodeRetrieveComplete();

  /// Releases a dequeued packed table that was never committed to transport.
  void releasePendingData();

  void releaseIntraNodeInFlightBytes();

  /// Deletes this destination from the stable producer queue.
  void deleteOutputResults();

  /// Advances to Done after the terminal marker has completed.
  void maybeFinish();

  /// @brief Sets the new state of this exchange server using
  /// sequential consistency. Logs transitions at VLOG(2).
  /// @param newState the new state of the UcxExchangeServer.
  void setState(ServerState newState);

  /// @brief Returns the state.
  ServerState getState() {
    return state_.load(std::memory_order_seq_cst);
  }

  const PartitionKey partitionKey_;
  const uint32_t
      partitionKeyHash_; // A hash of above, used to create unique tags.

  /// True if server and source are on the same node (determined by checking
  /// if peer's actual IP is in the local IP set). When true, data is passed
  /// via IntraNodeTransferRegistry instead of UCXX transfer.
  bool isIntraNodeTransfer_{false};

  std::atomic<ServerState> state_;
  std::shared_ptr<UcxGpuPayload> dataPtr_{nullptr};
  /// Protects dataPtr_ across queue callbacks, state-machine dispatch, and
  /// close/error cleanup paths.
  std::recursive_mutex dataMutex_;
  std::atomic<bool> closed_{false};
  std::atomic<bool> abortRequested_{false};
  std::atomic<bool> activated_{false};
  std::atomic<AdmissionState> admissionState_{AdmissionState::Pending};

  bool dataEndpointAckReceived_{false};
  bool normalMetadataInFlight_{false};
  bool dataSendInFlight_{false};
  bool finalMetadataSent_{false};
  bool finalMetadataCompleted_{false};
  bool outputResultsDeleted_{false};

  /// Future for intra-node transfer - signaled when source retrieves data.
  std::future<void> intraNodeRetrieveFuture_;

  /// For intra-node transfer: true if the last published entry was atEnd.
  bool intraNodeAtEndPublished_{false};

  uint32_t sequenceNumber_{0};
  uint32_t intraNodePollCount_{0};

  // The outstanding requests - there can only be one outstanding request
  // of each type at any point in time.
  // NOTE: The request owns/holds references to the upcall function
  // and must therefore exist until the upcall is done.
  std::shared_ptr<ucxx::Request> metaRequest_{nullptr};
  std::shared_ptr<ucxx::Request> dataRequest_{nullptr};
  std::shared_ptr<ucxx::Request> dataEndpointAckRequest_{nullptr};
  std::shared_ptr<ucxx::Request> abortRequest_{nullptr};

  // Completed UCXX requests are kept alive here to prevent use-after-free.
  // UCP's ucp_wireup_replay_pending_requests can fire callbacks on already-
  // completed requests; if the ucxx::Request has been freed, the callback
  // lambda is in freed memory and crashes. Retaining them here ensures the
  // Request (and its callback lambda) stays valid for the lifetime of this
  // server.
  std::vector<std::shared_ptr<ucxx::Request>> completedRequests_;

  std::chrono::time_point<std::chrono::high_resolution_clock> sendStart_;
  std::size_t bytes_{0};
  bool intraNodeBytesInFlight_{false};

  std::shared_ptr<UcxOutputQueueManager> queueMgr_;
  // Early placeholder queues are initialized in place and task IDs cannot be
  // reused after removal. Keep this exact queue alive so completion callbacks
  // can release in-flight accounting after manager removal.
  // Holds no pages with useDynamicUcx(); kept for the accounting.
  std::shared_ptr<UcxOutputQueue> outputQueue_;

  // Read once: the switch is global and settable at runtime.
  const bool useDynamicUcx_;

  std::shared_ptr<DynamicUcxOutputBufferReader> dynamicUcxReader_;
  std::deque<std::shared_ptr<UcxGpuPayload>> dynamicUcxPages_;
  bool dynamicUcxAtEnd_{false};
  // One payload per request, matching the queue's notify on the other path.
  // A drain can produce several pages at once.
  bool dynamicUcxRequestPending_{false};
};

} // namespace facebook::velox::ucx_exchange
