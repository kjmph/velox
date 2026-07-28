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
#include "velox/experimental/ucx-exchange/UcxExchangeServer.h"
#include <folly/ScopeGuard.h>
#include <glog/logging.h>
#include <atomic>
#include <limits>
#include "velox/common/EnumDefine.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"

namespace facebook::velox::ucx_exchange {

namespace {
const folly::F14FastMap<UcxExchangeServer::ServerState, std::string_view>&
serverStateNames() {
  static const folly::
      F14FastMap<UcxExchangeServer::ServerState, std::string_view>
          kNames = {
              {UcxExchangeServer::ServerState::Created, "Created"},
              {UcxExchangeServer::ServerState::WaitingForDataEndpointAck,
               "WaitingForDataEndpointAck"},
              {UcxExchangeServer::ServerState::ReadyToTransfer,
               "ReadyToTransfer"},
              {UcxExchangeServer::ServerState::WaitingForDataFromQueue,
               "WaitingForDataFromQueue"},
              {UcxExchangeServer::ServerState::DataReady, "DataReady"},
              {UcxExchangeServer::ServerState::WaitingForSendComplete,
               "WaitingForSendComplete"},
              {UcxExchangeServer::ServerState::WaitingForIntraNodeRetrieve,
               "WaitingForIntraNodeRetrieve"},
              {UcxExchangeServer::ServerState::WaitingForFinalMetadataComplete,
               "WaitingForFinalMetadataComplete"},
              {UcxExchangeServer::ServerState::Done, "Done"},
          };
  return kNames;
}
} // namespace

VELOX_DEFINE_EMBEDDED_ENUM_NAME(
    UcxExchangeServer,
    ServerState,
    serverStateNames)

// Context wrappers for UCXX tagSend callbackData. These decouple the
// ucxx::Request lifetime (which must survive for UCP wireup replay) from
// the buffer lifetime.
//
// The Request holds a shared_ptr to the context via callbackData. The context
// holds a shared_ptr to the actual buffer until the UCXX completion callback
// releases it.
struct MetaSendContext {
  std::shared_ptr<uint8_t> metadata;
  std::atomic<bool> callbackClaimed{false};

  bool tryClaimCallback() {
    bool expected = false;
    return callbackClaimed.compare_exchange_strong(
        expected, true, std::memory_order_acq_rel);
  }
};

struct DataSendContext {
  std::shared_ptr<cudf::packed_columns> data;
  std::shared_ptr<UcxOutputQueue> outputQueue;
  int destination{0};
  int64_t bytes{0};
  std::atomic<bool> callbackClaimed{false};

  bool tryClaimCallback() {
    bool expected = false;
    return callbackClaimed.compare_exchange_strong(
        expected, true, std::memory_order_acq_rel);
  }
};

struct ControlReceiveCallbackOnce {
  std::atomic<bool> callbackClaimed{false};

  bool tryClaim() {
    bool expected = false;
    return callbackClaimed.compare_exchange_strong(
        expected, true, std::memory_order_acq_rel);
  }
};

void UcxExchangeServer::setState(ServerState newState) {
  auto oldState = state_.exchange(newState, std::memory_order_seq_cst);
  VLOG(2) << (isIntraNodeTransfer_ ? "[INTRA]" : "[REMOTE]") << " [ExSrv "
          << partitionKey_.toString() << " seq=" << sequenceNumber_ << "] "
          << toName(oldState) << " -> " << toName(newState);
}

// This constructor is private
UcxExchangeServer::UcxExchangeServer(
    const std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> endpointRef,
    const PartitionKey& key,
    bool isIntraNodeTransfer)
    : CommElement(communicator, endpointRef, true),
      partitionKey_(key),
      partitionKeyHash_(fnv1a_32(partitionKey_.toString())),
      isIntraNodeTransfer_(isIntraNodeTransfer),
      queueMgr_(UcxOutputQueueManager::getInstanceRef()) {
  setState(ServerState::Created);

  if (isIntraNodeTransfer_) {
    VLOG(3) << "@" << partitionKey_.taskId
            << " Detected same-node source (intra-node transfer) for "
            << partitionKey_.toString();
  }
}

// static
std::shared_ptr<UcxExchangeServer> UcxExchangeServer::create(
    const std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> endpointRef,
    const PartitionKey& key,
    bool isIntraNodeTransfer) {
  auto ptr = std::shared_ptr<UcxExchangeServer>(new UcxExchangeServer(
      communicator, endpointRef, key, isIntraNodeTransfer));
  return ptr;
}

void UcxExchangeServer::process() {
  std::lock_guard<std::recursive_mutex> lock(processMutex_);
  try {
    if (callbackDispatchFailed_.exchange(false, std::memory_order_acq_rel)) {
      handleProcessFailure();
      return;
    }
    processStateMachine();
  } catch (...) {
    handleProcessFailure();
  }
}

void UcxExchangeServer::handleCallbackDispatchFailure() noexcept {
  callbackDispatchFailed_.store(true, std::memory_order_release);
  try {
    communicator_->addToWorkQueue(getSelfPtr());
  } catch (...) {
  }
}

void UcxExchangeServer::handleProcessFailure() noexcept {
  // A partially submitted bundle cannot be repaired by sending a final marker
  // alone. Endpoint cleanup force-closes every sibling and all exact receives.
  try {
    if (endpointRef_) {
      communicator_->deferEndpointCleanup(endpointRef_);
    }
  } catch (...) {
  }
  state_.store(ServerState::Done, std::memory_order_release);
  close();
}

void UcxExchangeServer::retireRequest(std::shared_ptr<ucxx::Request>& request) {
  if (!request) {
    return;
  }
  if (request->isCompleted()) {
    request.reset();
    return;
  }

  // Preserve our reference until insertion succeeds. Incomplete UCXX callback
  // state remains communicator-owned until isCompleted() becomes true.
  communicator_->deferRequestCleanup(request);
  request.reset();
}

void UcxExchangeServer::processStateMachine() {
  while (true) {
    drainStateEvents();
    if (closed_.load(std::memory_order_acquire)) {
      if (!cleanupComplete_) {
        close();
      }
      return;
    }
    if (abortRequested_.load(std::memory_order_acquire)) {
      processAbort();
      return;
    }
    if (!activated_.load(std::memory_order_acquire)) {
      return;
    }

    switch (state_) {
      case ServerState::Created:
        if (!dataEndpointAckReceived_) {
          setState(ServerState::WaitingForDataEndpointAck);
          return;
        }
        setState(ServerState::ReadyToTransfer);
        continue;
      case ServerState::WaitingForDataEndpointAck:
        return;
      case ServerState::ReadyToTransfer:
        requestData();
        return;
      case ServerState::WaitingForDataFromQueue:
        return;
      case ServerState::DataReady:
        sendData();
        return;
      case ServerState::WaitingForSendComplete:
        return;
      case ServerState::WaitingForIntraNodeRetrieve:
        if (pollIntraNodeRetrieve()) {
          continue;
        }
        return;
      case ServerState::WaitingForFinalMetadataComplete:
        if (isIntraNodeTransfer_) {
          if (pollIntraNodeRetrieve()) {
            maybeFinish();
            continue;
          }
          return;
        }
        maybeFinish();
        if (getState() == ServerState::Done) {
          continue;
        }
        return;
      case ServerState::Done:
        close();
        return;
    };
  }
}

bool UcxExchangeServer::activate() {
  std::lock_guard<std::recursive_mutex> processLock(processMutex_);
  AdmissionState expected = AdmissionState::Pending;
  if (!admissionState_.compare_exchange_strong(
          expected, AdmissionState::Accepted, std::memory_order_acq_rel)) {
    return false;
  }

  // Only the admitted server may own these task-scoped tags. Post both
  // receives before returning to the acceptor, which sends the accepted
  // handshake response immediately after registration.
  postDataEndpointAckReceive();
  postAbortReceive();

  bool expectedActivation = false;
  if (activated_.compare_exchange_strong(
          expectedActivation, true, std::memory_order_acq_rel)) {
    communicator_->addToWorkQueue(getSelfPtr());
  }
  return true;
}

void UcxExchangeServer::requestAbort() {
  AdmissionState pending = AdmissionState::Pending;
  admissionState_.compare_exchange_strong(
      pending, AdmissionState::Rejected, std::memory_order_acq_rel);

  bool expected = false;
  if (!abortRequested_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    return;
  }
  communicator_->addToWorkQueue(getSelfPtr());
}

void UcxExchangeServer::forceCloseForShutdown() {
  close();
  // close() can synchronously clear an output-queue callback while another
  // thread has already dequeued its payload. Drain events that won that race;
  // later callbacks observe closed_ and release through their queue context.
  drainStateEvents();
}

void UcxExchangeServer::close() noexcept {
  try {
    std::lock_guard<std::recursive_mutex> lock(processMutex_);
    AdmissionState pending = AdmissionState::Pending;
    admissionState_.compare_exchange_strong(
        pending, AdmissionState::Rejected, std::memory_order_acq_rel);
    closed_.store(true, std::memory_order_release);
    if (cleanupComplete_) {
      return;
    }
    drainStateEvents();
    if (isIntraNodeTransfer_ && intraNodeRetrieveFuture_.valid()) {
      IntraNodeTransferRegistry::getInstance()->cancelTransfer(
          {partitionKey_.taskId, partitionKey_.destination, sequenceNumber_});
      try {
        intraNodeRetrieveFuture_.get();
      } catch (const IntraNodeTransferCancelled&) {
      }
    }
    // Retain member ownership until each potentially allocating handoff
    // commits.
    retireRequest(metaRequest_);
    retireRequest(dataRequest_);
    std::vector<std::shared_ptr<ucxx::Request>> receives;
    if (dataEndpointAckRequest_) {
      receives.push_back(dataEndpointAckRequest_);
    }
    if (abortRequest_) {
      receives.push_back(abortRequest_);
    }
    communicator_->deferTagReceiveCancellation(std::move(receives));
    dataEndpointAckRequest_.reset();
    abortRequest_.reset();
    releasePendingData();
    releaseIntraNodeInFlightBytes();
    deleteOutputResults();
    auto self = getSelfPtr();
    queueMgr_->unregisterExchangeServer(self);
    if (endpointRef_) {
      endpointRef_->removeCommElem(self);
      endpointRef_.reset();
    }
    communicator_->unregister(std::move(self));
    cleanupComplete_ = true;
  } catch (...) {
    try {
      communicator_->addToWorkQueue(getSelfPtr());
    } catch (...) {
    }
  }
}

std::string UcxExchangeServer::toString() {
  std::stringstream out;
  out << "[ExSrv " << partitionKey_.toString() << " - " << sequenceNumber_
      << "]";
  return out.str();
}

// ------ private methods ---------

std::shared_ptr<UcxExchangeServer> UcxExchangeServer::getSelfPtr() {
  return shared_from_this();
}

void UcxExchangeServer::postDataEndpointAckReceive() {
  if (dataEndpointAckRequest_) {
    return;
  }

  auto ack = std::make_shared<GpuHandshakeAckHeader>();
  const uint64_t ackTag = getGpuHandshakeAckTag(partitionKeyHash_);
  std::weak_ptr<UcxExchangeServer> weak = weak_from_this();
  auto callbackOnce = std::make_shared<ControlReceiveCallbackOnce>();
  dataEndpointAckRequest_ =
      endpointRef_->endpoint_
          ->tagRecvBuilder(
              ack.get(), sizeof(*ack), ucxx::Tag{ackTag}, ucxx::TagMaskFull)
          .callbackFunction([weak, callbackOnce](
                                ucs_status_t status,
                                std::shared_ptr<void> arg) noexcept {
            if (!callbackOnce->tryClaim()) {
              return;
            }
            if (auto self = weak.lock()) {
              self->dispatchCompletion(
                  self,
                  [raw = self.get(), status, arg = std::move(arg)]() mutable {
                    raw->onDataEndpointAck(status, std::move(arg));
                  });
            }
          })
          .callbackData(ack)
          .build();
}

void UcxExchangeServer::postAbortReceive() {
  if (abortRequest_) {
    return;
  }

  auto abort = std::make_shared<GpuAbortHeader>();
  const uint64_t abortTag = getGpuAbortTag(partitionKeyHash_);
  std::weak_ptr<UcxExchangeServer> weak = weak_from_this();
  auto callbackOnce = std::make_shared<ControlReceiveCallbackOnce>();
  abortRequest_ =
      endpointRef_->endpoint_
          ->tagRecvBuilder(
              abort.get(),
              sizeof(*abort),
              ucxx::Tag{abortTag},
              ucxx::TagMaskFull)
          .callbackFunction([weak, callbackOnce](
                                ucs_status_t status,
                                std::shared_ptr<void> arg) noexcept {
            if (!callbackOnce->tryClaim()) {
              return;
            }
            if (auto self = weak.lock()) {
              self->dispatchCompletion(
                  self,
                  [raw = self.get(), status, arg = std::move(arg)]() mutable {
                    raw->onAbortRequest(status, std::move(arg));
                  });
            }
          })
          .callbackData(abort)
          .build();
}

void UcxExchangeServer::onDataEndpointAck(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  if (closed_.load(std::memory_order_acquire) || status == UCS_ERR_CANCELED) {
    return;
  }
  if (status != UCS_OK) {
    LOG(ERROR) << "[ExSrv " << partitionKey_.toString()
               << "] failed to receive data endpoint ACK: "
               << ucs_status_string(status);
    requestAbort();
    return;
  }

  auto ack = std::static_pointer_cast<GpuHandshakeAckHeader>(arg);
  if (ack->magic != kGpuHandshakeAckMagic ||
      ack->version != kGpuHandshakeAckVersion ||
      ack->headerSize != sizeof(GpuHandshakeAckHeader)) {
    LOG(ERROR) << "[ExSrv " << partitionKey_.toString()
               << "] invalid GPU data endpoint ACK";
    requestAbort();
    return;
  }

  dataEndpointAckReceived_ = true;
  if (getState() == ServerState::WaitingForDataEndpointAck) {
    setState(ServerState::ReadyToTransfer);
  }
}

void UcxExchangeServer::onAbortRequest(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  if (closed_.load(std::memory_order_acquire) || status == UCS_ERR_CANCELED) {
    return;
  }
  if (status != UCS_OK) {
    LOG(ERROR) << "[ExSrv " << partitionKey_.toString()
               << "] failed to receive abort control: "
               << ucs_status_string(status);
    requestAbort();
    return;
  }

  auto abort = std::static_pointer_cast<GpuAbortHeader>(arg);
  if (abort->magic != kGpuAbortMagic || abort->version != kGpuAbortVersion ||
      abort->headerSize != sizeof(GpuAbortHeader)) {
    LOG(ERROR) << "[ExSrv " << partitionKey_.toString()
               << "] invalid GPU abort control";
    // UCXX can replay a completion while wireup is settling. Retain the
    // completed request and its callback state before posting the replacement.
    retireRequest(abortRequest_);
    postAbortReceive();
    return;
  }
  requestAbort();
}

void UcxExchangeServer::requestData() {
  setState(ServerState::WaitingForDataFromQueue);
  installDataCallback();
}

void UcxExchangeServer::installDataCallback() {
  std::weak_ptr<UcxExchangeServer> weak = weak_from_this();
  const auto destination = partitionKey_.destination;
  auto notify = [weak, destination](
                    std::shared_ptr<UcxOutputQueue> stableQueue,
                    std::shared_ptr<cudf::packed_columns> data,
                    vector_size_t numRows,
                    std::vector<int64_t>) noexcept {
    auto self = weak.lock();
    auto releaseUncommitted = folly::makeGuard([&]() noexcept {
      if (data) {
        try {
          stableQueue->releaseInFlightBytes(
              destination, data->gpu_data->size(), 1);
        } catch (...) {
          if (self) {
            self->handleCallbackDispatchFailure();
          }
        }
      }
    });
    try {
      if (!self) {
        return;
      }
      std::lock_guard<std::recursive_mutex> lock(self->processMutex_);
      if (self->closed_.load(std::memory_order_acquire)) {
        return;
      }
      // Exact queue ownership is known before synchronous callbacks or manager
      // removal. No allocation failure may strand the dequeued accounting.
      if (!self->enqueueStateEvent(
              self, [raw = self.get(), stableQueue, data, numRows]() mutable {
                raw->outputQueue_ = std::move(stableQueue);
                raw->onDataAvailable(std::move(data), numRows);
              })) {
        self->handleCallbackDispatchFailure();
        return;
      }
      releaseUncommitted.dismiss();
    } catch (...) {
      if (self) {
        self->handleCallbackDispatchFailure();
      }
    }
  };
  if (outputQueue_) {
    std::weak_ptr<UcxOutputQueue> weakQueue = outputQueue_;
    outputQueue_->getData(
        destination,
        [weakQueue, notify = std::move(notify)](
            std::shared_ptr<cudf::packed_columns> data,
            vector_size_t numRows,
            std::vector<int64_t> remaining) mutable {
          auto stableQueue = weakQueue.lock();
          VELOX_CHECK_NOT_NULL(stableQueue);
          notify(
              std::move(stableQueue),
              std::move(data),
              numRows,
              std::move(remaining));
        });
  } else {
    outputQueue_ = queueMgr_->getDataWithQueue(
        partitionKey_.taskId, destination, std::move(notify));
  }
}

void UcxExchangeServer::onDataAvailable(
    std::shared_ptr<cudf::packed_columns> data,
    vector_size_t numRows) {
  auto releaseUncommitted = folly::makeGuard([&]() noexcept {
    if (data) {
      try {
        outputQueue_->releaseInFlightBytes(
            partitionKey_.destination, data->gpu_data->size(), 1);
      } catch (...) {
        handleCallbackDispatchFailure();
      }
    }
  });
  if (closed_.load(std::memory_order_acquire) ||
      abortRequested_.load(std::memory_order_acquire) || finalMetadataSent_) {
    return;
  }

  VLOG(3) << "@" << partitionKey_.taskId
          << " Found data for client: " << partitionKey_.toString();
  std::lock_guard<std::recursive_mutex> lock(dataMutex_);
  VELOX_CHECK_NULL(dataPtr_, "Data pointer exists: Illegal state!");
  dataPtr_ = std::move(data);
  dataNumRows_ = numRows;
  setState(ServerState::DataReady);
}

void UcxExchangeServer::processAbort() {
  releasePendingData();

  // A rejected server has no accepted source waiting for an end marker.
  if (admissionState_.load(std::memory_order_acquire) !=
      AdmissionState::Accepted) {
    close();
    return;
  }

  if (getState() == ServerState::Done) {
    close();
    return;
  }

  // An intra-process table has already been published. The source's abort
  // drain retrieves (and discards) it before looking for the final marker.
  if (intraNodeRetrieveFuture_.valid() && !pollIntraNodeRetrieve()) {
    return;
  }

  // Remote metadata and data sends are both committed once submitted. Wait
  // for both callbacks before publishing the next sequenced metadata record.
  if (normalMetadataInFlight_ || dataSendInFlight_) {
    return;
  }

  if (!finalMetadataSent_) {
    sendFinalMetadata();
    return;
  }

  if (isIntraNodeTransfer_ && intraNodeRetrieveFuture_.valid()) {
    if (!pollIntraNodeRetrieve()) {
      return;
    }
  }
  if (!finalMetadataCompleted_) {
    return;
  }

  close();
}

void UcxExchangeServer::sendData() {
  std::lock_guard<std::recursive_mutex> lock(dataMutex_);

  VLOG(2) << (isIntraNodeTransfer_ ? "[INTRA]" : "[REMOTE]") << " [ExSrv "
          << partitionKey_.toString() << " seq=" << sequenceNumber_
          << "] sendData hasData=" << (dataPtr_ != nullptr)
          << (dataPtr_ && dataPtr_->gpu_data
                  ? " size=" + std::to_string(dataPtr_->gpu_data->size())
                  : "");

  if (!dataPtr_) {
    sendFinalMetadata();
    return;
  }

  if (isIntraNodeTransfer_) {
    sendStart_ = std::chrono::high_resolution_clock::now();
    bytes_ = dataPtr_->gpu_data->size();

    VLOG(3) << "@" << partitionKey_.taskId
            << " Intra-node transfer: publishing data for sequence "
            << sequenceNumber_ << " of size " << bytes_;

    IntraNodeTransferKey key{
        partitionKey_.taskId, partitionKey_.destination, sequenceNumber_};
    intraNodeRetrieveFuture_ =
        IntraNodeTransferRegistry::getInstance()->publish(
            key, dataPtr_, dataNumRows_, /*atEnd=*/false);
    dataPtr_.reset();
    intraNodeAtEndPublished_ = false;
    intraNodeBytesInFlight_ = true;

    setState(ServerState::WaitingForIntraNodeRetrieve);
    communicator_->addToWorkQueue(getSelfPtr());
    return;
  }

  auto data = dataPtr_;
  VELOX_CHECK_NOT_NULL(outputQueue_);
  bytes_ = data->gpu_data->size();

  MetadataMsg metadataMsg;
  // Copy metadata because broadcast destinations can share packed_columns.
  metadataMsg.cudfMetadata =
      std::make_unique<std::vector<uint8_t>>(*data->metadata);
  metadataMsg.dataSizeBytes = bytes_;
  metadataMsg.numRows = dataNumRows_;
  metadataMsg.remainingBytes = {};
  metadataMsg.atEnd = false;
  auto [serializedMetadata, serMetaSize] = metadataMsg.serialize();

  const uint64_t metadataTag =
      getMetadataTag(partitionKeyHash_, sequenceNumber_);
  const uint64_t dataTag = getDataTag(partitionKeyHash_, sequenceNumber_);
  retireRequest(metaRequest_);
  retireRequest(dataRequest_);

  auto metaCtx = std::make_shared<MetaSendContext>();
  metaCtx->metadata = serializedMetadata;
  std::weak_ptr<UcxExchangeServer> weakMeta = weak_from_this();

  normalMetadataInFlight_ = true;
  dataSendInFlight_ = true;
  sendStart_ = std::chrono::high_resolution_clock::now();
  setState(ServerState::WaitingForSendComplete);

  metaRequest_ =
      endpointRef_->endpoint_
          ->tagSendBuilder(
              metaCtx->metadata.get(), serMetaSize, ucxx::Tag{metadataTag})
          .callbackFunction(
              [weakMeta, metadataTag, task = partitionKey_.toString()](
                  ucs_status_t status, std::shared_ptr<void> arg) noexcept {
                auto ctx = std::static_pointer_cast<MetaSendContext>(arg);
                if (!ctx->tryClaimCallback()) {
                  return;
                }
                ctx->metadata.reset();
                if (auto self = weakMeta.lock()) {
                  self->dispatchCompletion(self, [raw = self.get(), status]() {
                    raw->metadataSendComplete(status);
                  });
                }
              })
          .callbackData(metaCtx)
          .build();

  VLOG(3) << "@" << partitionKey_.taskId << " Sending rmm::buffer: " << std::hex
          << data->gpu_data.get()
          << " pointing to device memory: " << data->gpu_data->data()
          << std::dec << " to task " << partitionKey_.toString() << ":"
          << sequenceNumber_ << " of size " << bytes_;

  auto dataCtx = std::make_shared<DataSendContext>();
  dataCtx->data = std::move(data);
  dataCtx->outputQueue = outputQueue_;
  dataCtx->destination = partitionKey_.destination;
  dataCtx->bytes = bytes_;
  std::weak_ptr<UcxExchangeServer> weakData = weak_from_this();
  auto releaseRejectedData = folly::makeGuard([&]() noexcept {
    if (dataCtx->tryClaimCallback()) {
      dataCtx->data.reset();
      dataPtr_.reset();
      try {
        dataCtx->outputQueue->releaseInFlightBytes(
            dataCtx->destination, dataCtx->bytes, 1);
      } catch (...) {
      }
    }
  });

  dataRequest_ =
      endpointRef_->endpoint_
          ->tagSendBuilder(
              dataCtx->data->gpu_data->data(),
              dataCtx->data->gpu_data->size(),
              ucxx::Tag{dataTag})
          .callbackFunction(
              [weakData](
                  ucs_status_t status, std::shared_ptr<void> arg) noexcept {
                auto ctx = std::static_pointer_cast<DataSendContext>(arg);
                if (!ctx->tryClaimCallback()) {
                  return;
                }
                ctx->data.reset();
                try {
                  ctx->outputQueue->releaseInFlightBytes(
                      ctx->destination, ctx->bytes, 1L);
                } catch (...) {
                  if (auto self = weakData.lock()) {
                    self->handleCallbackDispatchFailure();
                  }
                  return;
                }

                if (auto self = weakData.lock()) {
                  self->dispatchCompletion(self, [raw = self.get(), status]() {
                    raw->sendComplete(status);
                  });
                }
              })
          .callbackData(dataCtx)
          .build();
  releaseRejectedData.dismiss();
  dataPtr_.reset();
  dataNumRows_ = 0;
}

void UcxExchangeServer::sendFinalMetadata() {
  if (finalMetadataSent_) {
    return;
  }
  finalMetadataSent_ = true;
  VLOG(3) << "@" << partitionKey_.taskId << " Final exchange for "
          << partitionKey_.toString() << " sequence " << sequenceNumber_;

  if (isIntraNodeTransfer_) {
    sendStart_ = std::chrono::high_resolution_clock::now();
    IntraNodeTransferKey key{
        partitionKey_.taskId, partitionKey_.destination, sequenceNumber_};
    intraNodeRetrieveFuture_ =
        IntraNodeTransferRegistry::getInstance()->publish(
            key, nullptr, /*numRows=*/0, /*atEnd=*/true);
    intraNodeAtEndPublished_ = true;
    setState(ServerState::WaitingForFinalMetadataComplete);
    deleteOutputResults();
    communicator_->addToWorkQueue(getSelfPtr());
    return;
  }

  MetadataMsg metadataMsg;
  metadataMsg.cudfMetadata = nullptr;
  metadataMsg.dataSizeBytes = 0;
  metadataMsg.numRows = 0;
  metadataMsg.remainingBytes = {};
  metadataMsg.atEnd = true;
  auto [serializedMetadata, serMetaSize] = metadataMsg.serialize();
  const uint64_t metadataTag =
      getMetadataTag(partitionKeyHash_, sequenceNumber_);

  retireRequest(metaRequest_);
  retireRequest(dataRequest_);

  auto metaCtx = std::make_shared<MetaSendContext>();
  metaCtx->metadata = serializedMetadata;
  std::weak_ptr<UcxExchangeServer> weakMeta = weak_from_this();
  setState(ServerState::WaitingForFinalMetadataComplete);
  metaRequest_ =
      endpointRef_->endpoint_
          ->tagSendBuilder(
              metaCtx->metadata.get(), serMetaSize, ucxx::Tag{metadataTag})
          .callbackFunction(
              [weakMeta](
                  ucs_status_t status, std::shared_ptr<void> arg) noexcept {
                auto ctx = std::static_pointer_cast<MetaSendContext>(arg);
                if (!ctx->tryClaimCallback()) {
                  return;
                }
                ctx->metadata.reset();
                if (auto self = weakMeta.lock()) {
                  self->dispatchCompletion(self, [raw = self.get(), status]() {
                    raw->finalMetadataComplete(status);
                  });
                }
              })
          .callbackData(metaCtx)
          .build();
  deleteOutputResults();
}

void UcxExchangeServer::metadataSendComplete(ucs_status_t status) {
  if (closed_.load(std::memory_order_acquire)) {
    return;
  }
  normalMetadataInFlight_ = false;
  if (status != UCS_OK) {
    LOG(ERROR) << "@" << partitionKey_.taskId
               << " Error in metadata send: " << ucs_status_string(status)
               << " task: " << partitionKey_.toString();
    setState(ServerState::Done);
    return;
  }
  maybeCompleteRemoteSend();
}

void UcxExchangeServer::sendComplete(ucs_status_t status) {
  if (closed_.load(std::memory_order_acquire)) {
    return;
  }
  dataSendInFlight_ = false;

  if (status == UCS_OK) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - sendStart_;
    auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    auto throughput = micros > 0 ? bytes_ / micros : 0;

    VLOG(3) << "@" << partitionKey_.taskId << " duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                   .count()
            << " ms ";
    VLOG(3) << "@" << partitionKey_.taskId << " throughput: " << throughput
            << " MByte/s";
    maybeCompleteRemoteSend();
  } else {
    LOG(ERROR) << "@" << partitionKey_.taskId
               << " Error in data send: " << ucs_status_string(status);
    setState(ServerState::Done);
  }
}

void UcxExchangeServer::maybeCompleteRemoteSend() {
  if (normalMetadataInFlight_ || dataSendInFlight_ ||
      getState() == ServerState::Done) {
    return;
  }
  ++sequenceNumber_;
  VLOG(3) << "@" << partitionKey_.taskId
          << " Send complete; advancing to next sequence.";
  if (!abortRequested_.load(std::memory_order_acquire)) {
    setState(ServerState::ReadyToTransfer);
  }
}

void UcxExchangeServer::finalMetadataComplete(ucs_status_t status) {
  if (closed_.load(std::memory_order_acquire)) {
    return;
  }
  if (status != UCS_OK) {
    LOG(ERROR) << "@" << partitionKey_.taskId
               << " Error in final metadata send: " << ucs_status_string(status)
               << " task: " << partitionKey_.toString();
    setState(ServerState::Done);
    return;
  }
  finalMetadataCompleted_ = true;
  maybeFinish();
}

bool UcxExchangeServer::pollIntraNodeRetrieve() {
  if (!intraNodeRetrieveFuture_.valid()) {
    return true;
  }
  auto status = intraNodeRetrieveFuture_.wait_for(std::chrono::milliseconds(0));
  if (status != std::future_status::ready) {
    ++intraNodePollCount_;
    if (intraNodePollCount_ % 100 == 0) {
      VLOG(2) << "[INTRA] [ExSrv " << partitionKey_.toString()
              << " seq=" << sequenceNumber_
              << "] still waiting for source retrieval, polls="
              << intraNodePollCount_;
    }
    communicator_->addToWorkQueue(getSelfPtr());
    return false;
  }

  intraNodeRetrieveFuture_.get();
  intraNodePollCount_ = 0;
  onIntraNodeRetrieveComplete();
  return true;
}

void UcxExchangeServer::onIntraNodeRetrieveComplete() {
  if (closed_.load(std::memory_order_acquire)) {
    return;
  }

  if (!intraNodeAtEndPublished_) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - sendStart_;
    auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    auto throughput = (micros > 0) ? (bytes_ / micros) : 0;

    VLOG(3) << "@" << partitionKey_.taskId << " Intra-node transfer duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                   .count()
            << " ms ";
    VLOG(3) << "@" << partitionKey_.taskId
            << " Intra-node transfer throughput: " << throughput << " MByte/s";
  }

  releaseIntraNodeInFlightBytes();

  if (intraNodeAtEndPublished_) {
    VLOG(3) << "@" << partitionKey_.taskId
            << " Intra-node transfer: atEnd retrieved, finishing";
    finalMetadataCompleted_ = true;
    maybeFinish();
  } else {
    ++sequenceNumber_;
    if (!abortRequested_.load(std::memory_order_acquire)) {
      setState(ServerState::ReadyToTransfer);
    }
  }
}

void UcxExchangeServer::releasePendingData() {
  std::shared_ptr<cudf::packed_columns> pending;
  {
    std::lock_guard<std::recursive_mutex> lock(dataMutex_);
    pending = std::move(dataPtr_);
  }
  if (!pending) {
    return;
  }
  VELOX_CHECK_NOT_NULL(
      outputQueue_, "Dequeued GPU data has no stable output queue");
  outputQueue_->releaseInFlightBytes(
      partitionKey_.destination, pending->gpu_data->size(), 1L);
}

void UcxExchangeServer::releaseIntraNodeInFlightBytes() {
  if (!intraNodeBytesInFlight_) {
    return;
  }
  VELOX_CHECK_NOT_NULL(
      outputQueue_, "Intra-node GPU data has no stable output queue");
  outputQueue_->releaseInFlightBytes(partitionKey_.destination, bytes_, 1L);
  intraNodeBytesInFlight_ = false;
}

void UcxExchangeServer::deleteOutputResults() {
  if (outputResultsDeleted_) {
    return;
  }
  if (!outputQueue_) {
    // The accepted source can abort before READY reaches requestData(). Use
    // the ordinary server callback here: getData() may synchronously dequeue
    // one table, and that table still needs its in-flight accounting released
    // after the stable queue pointer has been returned.
    installDataCallback();
  }
  if (outputQueue_) {
    outputQueue_->deleteResults(partitionKey_.destination);
  }
  outputResultsDeleted_ = true;
}

void UcxExchangeServer::maybeFinish() {
  if (finalMetadataCompleted_ && !normalMetadataInFlight_ &&
      !dataSendInFlight_ && !intraNodeRetrieveFuture_.valid() &&
      getState() != ServerState::Done) {
    ++sequenceNumber_;
    setState(ServerState::Done);
  }
}

} // namespace facebook::velox::ucx_exchange
