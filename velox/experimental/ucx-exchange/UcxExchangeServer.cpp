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

#include <cstring>

#include <glog/logging.h>
#include <atomic>
#include "velox/common/EnumDefine.h"
#include "velox/exec/DefaultOutputBufferManager.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/CudfPackedPage.h"
#include "velox/experimental/ucx-exchange/DynamicUcxTransport.h"
#include "velox/experimental/ucx-exchange/ExchangeFormatRegistry.h"
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"

namespace facebook::velox::ucx_exchange {

static constexpr uint64_t kDynamicUcxFetchBytes{1ULL << 20};

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
  std::shared_ptr<UcxGpuPayload> data;
  std::shared_ptr<UcxOutputQueue> outputQueue;
  int destination{0};
  int64_t bytes{0};
  bool charged{false};
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

struct OutputQueueCallbackContext {
  void setQueue(const std::shared_ptr<UcxOutputQueue>& queue) {
    std::lock_guard<std::mutex> lock(mutex);
    outputQueue = queue;
  }

  void releaseIfDequeued(
      const std::shared_ptr<UcxGpuPayload>& data,
      int destination) {
    if (!data) {
      return;
    }
    std::shared_ptr<UcxOutputQueue> queue;
    {
      std::lock_guard<std::mutex> lock(mutex);
      queue = outputQueue.lock();
    }
    if (queue) {
      queue->releaseInFlightBytes(destination, data->payloadBytes(), 1L);
    }
  }

  std::mutex mutex;
  std::weak_ptr<UcxOutputQueue> outputQueue;
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
      queueMgr_(UcxOutputQueueManager::getInstanceRef()),
      useDynamicUcx_(useDynamicUcx()) {
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
  while (true) {
    drainStateEvents();
    if (closed_.load(std::memory_order_acquire)) {
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

void UcxExchangeServer::close() {
  std::lock_guard<std::recursive_mutex> processLock(processMutex_);
  AdmissionState pending = AdmissionState::Pending;
  admissionState_.compare_exchange_strong(
      pending, AdmissionState::Rejected, std::memory_order_acq_rel);

  bool expected = false;
  if (!closed_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    return;
  }
  VLOG(3) << "@" << partitionKey_.taskId
          << " Close UcxExchangeServer to remote " << partitionKey_.toString();

  releasePendingData();
  releaseIntraNodeInFlightBytes();

  releaseDynamicUcxResources(/*releaseBuffer=*/false);
  if (outputQueue_ && !outputResultsDeleted_ && !useDynamicUcx_) {
    // Hard transport cleanup can arrive while getData() is waiting. Clear the
    // installed callback and discard queued output before unregistering this
    // server. Do not create a placeholder here for a rejected server.
    outputResultsDeleted_ = true;
    outputQueue_->deleteResults(partitionKey_.destination);
  }
  // Skipped under discovery: the pages are not in this queue, and nulling it
  // makes UcxOutputQueue::deleteResults tell the task its output was consumed.

  if (communicator_) {
    // UCP cancellation is supported for tag receives, not tag sends. Retain
    // every send until UCX has completed any delayed submission and callback.
    if (metaRequest_) {
      communicator_->deferRequestCleanup(std::move(metaRequest_));
    }
    if (dataRequest_) {
      communicator_->deferRequestCleanup(std::move(dataRequest_));
    }
    for (auto& req : completedRequests_) {
      communicator_->deferRequestCleanup(std::move(req));
    }
    completedRequests_.clear();

    std::vector<std::shared_ptr<ucxx::Request>> receiveRequests;
    receiveRequests.reserve(
        (dataEndpointAckRequest_ ? 1 : 0) + (abortRequest_ ? 1 : 0));
    if (dataEndpointAckRequest_) {
      receiveRequests.push_back(std::move(dataEndpointAckRequest_));
    }
    if (abortRequest_) {
      receiveRequests.push_back(std::move(abortRequest_));
    }
    communicator_->deferTagReceiveCancellation(std::move(receiveRequests));
  }

  auto self = getSelfPtr();
  queueMgr_->unregisterExchangeServer(self);
  if (endpointRef_) {
    auto endpointRef = std::move(endpointRef_);
    endpointRef->removeCommElem(self);
  }
  if (communicator_) {
    communicator_->unregister(std::move(self));
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
  dataEndpointAckRequest_ = endpointRef_->endpoint_->tagRecv(
      ack.get(),
      sizeof(*ack),
      ucxx::Tag{ackTag},
      ucxx::TagMaskFull,
      false,
      [weak, callbackOnce](ucs_status_t status, std::shared_ptr<void> arg) {
        if (!callbackOnce->tryClaim()) {
          return;
        }
        if (auto self = weak.lock()) {
          self->enqueueStateEvent(
              self, [raw = self.get(), status, arg = std::move(arg)]() mutable {
                raw->onDataEndpointAck(status, std::move(arg));
              });
        }
      },
      ack);
}

void UcxExchangeServer::postAbortReceive() {
  if (abortRequest_) {
    return;
  }

  auto abort = std::make_shared<GpuAbortHeader>();
  const uint64_t abortTag = getGpuAbortTag(partitionKeyHash_);
  std::weak_ptr<UcxExchangeServer> weak = weak_from_this();
  auto callbackOnce = std::make_shared<ControlReceiveCallbackOnce>();
  abortRequest_ = endpointRef_->endpoint_->tagRecv(
      abort.get(),
      sizeof(*abort),
      ucxx::Tag{abortTag},
      ucxx::TagMaskFull,
      false,
      [weak, callbackOnce](ucs_status_t status, std::shared_ptr<void> arg) {
        if (!callbackOnce->tryClaim()) {
          return;
        }
        if (auto self = weak.lock()) {
          self->enqueueStateEvent(
              self, [raw = self.get(), status, arg = std::move(arg)]() mutable {
                raw->onAbortRequest(status, std::move(arg));
              });
        }
      },
      abort);
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
    completedRequests_.push_back(std::move(abortRequest_));
    postAbortReceive();
    return;
  }
  requestAbort();
}

void UcxExchangeServer::requestData() {
  setState(ServerState::WaitingForDataFromQueue);
  installDataCallback();
}

void UcxExchangeServer::requestDynamicUcxData() {
  if (outputQueue_ == nullptr) {
    outputQueue_ = queueMgr_->getQueueIfExists(partitionKey_.taskId);
  }
  if (dynamicUcxReader_ == nullptr) {
    // Must precede the first read; the acceptor is too late.
    ExchangeFormatRegistry::instance().setExchangeFormat(
        partitionKey_.taskId,
        partitionKey_.destination,
        ExchangeFormatRegistry::Format::kCudf);
    dynamicUcxReader_ = std::make_shared<DynamicUcxOutputBufferReader>(
        exec::DefaultOutputBufferManager::getInstanceRef(),
        partitionKey_.taskId,
        partitionKey_.destination);
  }

  dynamicUcxRequestPending_ = true;
  if (!dynamicUcxPages_.empty() || dynamicUcxAtEnd_) {
    deliverDynamicUcxPage();
    return;
  }

  std::weak_ptr<UcxExchangeServer> weakSelf = weak_from_this();
  dynamicUcxReader_->request(
      kDynamicUcxFetchBytes,
      [weakSelf](DynamicUcxOutputBufferReader::Data data) {
        auto self = weakSelf.lock();
        if (!self || self->closed_.load(std::memory_order_acquire)) {
          return;
        }
        // Data holds unique_ptrs; the event queue copies its callable.
        auto held = std::make_shared<DynamicUcxOutputBufferReader::Data>(
            std::move(data));
        self->enqueueStateEvent(self, [raw = self.get(), held]() mutable {
          raw->onDynamicUcxData(std::move(*held));
        });
        self->communicator_->addToWorkQueue(self);
      });
}

void UcxExchangeServer::onDynamicUcxData(
    DynamicUcxOutputBufferReader::Data data) {
  // A batch can arrive after close(): both drain paths run events first.
  if (closed_.load(std::memory_order_acquire) || dynamicUcxReader_ == nullptr) {
    return;
  }

  if (outputQueue_ == nullptr) {
    // May not have existed when this server was built.
    outputQueue_ = queueMgr_->getQueueIfExists(partitionKey_.taskId);
  }
  for (auto& page : data.pages) {
    bool deviceResident = false;
    if (page->length() >= kCudfPackedPageHeaderBytes) {
      uint32_t magic = 0;
      std::memcpy(&magic, page->data(), sizeof(magic));
      deviceResident = magic == kCudfPackedPageMagic;
    }

    auto payload = std::make_shared<UcxGpuPayload>();
    payload->deviceResident = deviceResident;
    if (deviceResident) {
      int32_t rows = 0;
      std::memcpy(
          &rows, page->data() + sizeof(kCudfPackedPageMagic), sizeof(rows));
      payload->numRows = rows;
    } else {
      // tagSend needs one region; a serialized page is a chain.
      page->coalesce();
      payload->numRows = 0;
    }
    payload->buffer = std::shared_ptr<folly::IOBuf>(std::move(page));
    // Not dequeued, so nothing else charged it.
    payload->inFlightCharged = outputQueue_ != nullptr &&
        outputQueue_->acquireInFlightBytes(
            partitionKey_.destination,
            static_cast<int64_t>(payload->payloadBytes()),
            1L);
    dynamicUcxPages_.push_back(std::move(payload));
  }
  dynamicUcxAtEnd_ |= data.atEnd;
  if (dynamicUcxRequestPending_) {
    deliverDynamicUcxPage();
  }
}

void UcxExchangeServer::releaseDynamicUcxResources(bool releaseBuffer) {
  if (dynamicUcxReader_ != nullptr) {
    // Pages taken but never sent were charged when they were taken.
    if (outputQueue_ != nullptr) {
      for (const auto& page : dynamicUcxPages_) {
        if (page != nullptr && page->inFlightCharged) {
          outputQueue_->releaseInFlightBytes(
              partitionKey_.destination,
              static_cast<int64_t>(page->payloadBytes()),
              1L);
        }
      }
    }
    dynamicUcxPages_.clear();
    // Undone here; a stock sink has no operator hook to do it.
    ExchangeFormatRegistry::instance().setExchangeFormat(
        partitionKey_.taskId,
        partitionKey_.destination,
        ExchangeFormatRegistry::Format::kUnknown);
    dynamicUcxReader_->close();
    if (releaseBuffer) {
      dynamicUcxReader_->deleteResults();
    }
    dynamicUcxReader_.reset();
  }

  // The driver adapter's task registration is released by the producing
  // operator at task completion, not here: other destinations may be unread.
}

void UcxExchangeServer::deliverDynamicUcxPage() {
  if (!dynamicUcxRequestPending_) {
    return;
  }
  dynamicUcxRequestPending_ = false;

  std::shared_ptr<UcxGpuPayload> next;
  if (!dynamicUcxPages_.empty()) {
    next = std::move(dynamicUcxPages_.front());
    dynamicUcxPages_.pop_front();
  }
  if (next == nullptr && !dynamicUcxAtEnd_) {
    requestDynamicUcxData();
    return;
  }

  onDataAvailable(std::move(next));

  communicator_->addToWorkQueue(getSelfPtr());
}

void UcxExchangeServer::installDataCallback() {
  if (useDynamicUcx_) {
    requestDynamicUcxData();
    return;
  }
  std::weak_ptr<UcxExchangeServer> weakQueue = weak_from_this();
  auto callbackContext = std::make_shared<OutputQueueCallbackContext>();
  auto notify =
      [weakQueue, callbackContext, destination = partitionKey_.destination](
          std::shared_ptr<UcxGpuPayload> data,
          std::vector<int64_t> /*remainingBytes*/) {
        auto self = weakQueue.lock();
        if (!self || self->closed_.load(std::memory_order_acquire)) {
          callbackContext->releaseIfDequeued(data, destination);
          return;
        }
        self->enqueueStateEvent(
            self, [raw = self.get(), data = std::move(data)]() mutable {
              raw->onDataAvailable(std::move(data));
            });
      };

  if (outputQueue_) {
    callbackContext->setQueue(outputQueue_);
    outputQueue_->getData(partitionKey_.destination, std::move(notify));
  } else {
    outputQueue_ = queueMgr_->getData(
        partitionKey_.taskId, partitionKey_.destination, std::move(notify));
    callbackContext->setQueue(outputQueue_);
  }
}

void UcxExchangeServer::onDataAvailable(std::shared_ptr<UcxGpuPayload> data) {
  if (closed_.load(std::memory_order_acquire) ||
      abortRequested_.load(std::memory_order_acquire) || finalMetadataSent_) {
    if (data) {
      if (outputQueue_ != nullptr && data->inFlightCharged) {
        outputQueue_->releaseInFlightBytes(
            partitionKey_.destination, data->payloadBytes(), 1L);
      }
    }
    return;
  }

  VLOG(3) << "@" << partitionKey_.taskId
          << " Found data for client: " << partitionKey_.toString();
  std::lock_guard<std::recursive_mutex> lock(dataMutex_);
  VELOX_CHECK_NULL(dataPtr_, "Data pointer exists: Illegal state!");
  dataPtr_ = std::move(data);
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
          << (dataPtr_ ? " size=" + std::to_string(dataPtr_->payloadBytes())
                       : "");

  if (!dataPtr_) {
    sendFinalMetadata();
    return;
  }

  if (isIntraNodeTransfer_) {
    sendStart_ = std::chrono::high_resolution_clock::now();
    bytes_ = dataPtr_->payloadBytes();

    VLOG(3) << "@" << partitionKey_.taskId
            << " Intra-node transfer: publishing data for sequence "
            << sequenceNumber_ << " of size " << bytes_;

    IntraNodeTransferKey key{
        partitionKey_.taskId, partitionKey_.destination, sequenceNumber_};
    // Read before the payload is handed over below.
    const bool charged = dataPtr_->inFlightCharged;
    intraNodeRetrieveFuture_ =
        IntraNodeTransferRegistry::getInstance()->publish(
            key, dataPtr_, /*atEnd=*/false);
    dataPtr_.reset();
    intraNodeAtEndPublished_ = false;
    intraNodeBytesInFlight_ = charged;

    setState(ServerState::WaitingForIntraNodeRetrieve);
    communicator_->addToWorkQueue(getSelfPtr());
    return;
  }

  auto data = std::move(dataPtr_);
  bytes_ = data->payloadBytes();

  MetadataMsg metadataMsg;
  // Copy metadata because broadcast destinations can share packed_columns.
  // Null iterators are UB even for an empty range.
  metadataMsg.cudfMetadata = data->metadata() != nullptr
      ? std::make_unique<std::vector<uint8_t>>(
            data->metadata(), data->metadata() + data->metadataBytes())
      : std::make_unique<std::vector<uint8_t>>();
  metadataMsg.dataSizeBytes = bytes_;
  metadataMsg.numRows = data->numRows;
  metadataMsg.isDeviceData = data->deviceResident;
  metadataMsg.remainingBytes = {};
  metadataMsg.atEnd = false;
  auto [serializedMetadata, serMetaSize] = metadataMsg.serialize();

  const uint64_t metadataTag =
      getMetadataTag(partitionKeyHash_, sequenceNumber_);
  const uint64_t dataTag = getDataTag(partitionKeyHash_, sequenceNumber_);
  if (metaRequest_) {
    completedRequests_.push_back(std::move(metaRequest_));
  }
  if (dataRequest_) {
    completedRequests_.push_back(std::move(dataRequest_));
  }

  auto metaCtx = std::make_shared<MetaSendContext>();
  metaCtx->metadata = serializedMetadata;
  std::weak_ptr<UcxExchangeServer> weakMeta = weak_from_this();

  normalMetadataInFlight_ = true;
  dataSendInFlight_ = true;
  sendStart_ = std::chrono::high_resolution_clock::now();
  setState(ServerState::WaitingForSendComplete);

  metaRequest_ = endpointRef_->endpoint_->tagSend(
      metaCtx->metadata.get(),
      serMetaSize,
      ucxx::Tag{metadataTag},
      false,
      [weakMeta, metadataTag, task = partitionKey_.toString()](
          ucs_status_t status, std::shared_ptr<void> arg) {
        auto ctx = std::static_pointer_cast<MetaSendContext>(arg);
        if (!ctx->tryClaimCallback()) {
          return;
        }
        ctx->metadata.reset();
        if (status == UCS_OK) {
          VLOG(3) << "metadata successfully sent to " << task
                  << " with tag: " << std::hex << metadataTag;
        }
        if (auto self = weakMeta.lock()) {
          self->enqueueStateEvent(self, [raw = self.get(), status]() {
            raw->metadataSendComplete(status);
          });
        }
      },
      metaCtx);

  VLOG(3) << "@" << partitionKey_.taskId << " Sending rmm::buffer: " << std::hex
          << data->payload()
          << " pointing to device memory: " << data->payload() << std::dec
          << " to task " << partitionKey_.toString() << ":" << sequenceNumber_
          << " of size " << bytes_;

  auto dataCtx = std::make_shared<DataSendContext>();
  dataCtx->data = std::move(data);
  dataCtx->outputQueue = outputQueue_;
  dataCtx->destination = partitionKey_.destination;
  dataCtx->bytes = bytes_;
  dataCtx->charged = dataCtx->data->inFlightCharged;
  std::weak_ptr<UcxExchangeServer> weakData = weak_from_this();

  dataRequest_ = endpointRef_->endpoint_->tagSend(
      const_cast<void*>(dataCtx->data->payload()),
      dataCtx->data->payloadBytes(),
      ucxx::Tag{dataTag},
      false,
      [weakData](ucs_status_t status, std::shared_ptr<void> arg) {
        auto ctx = std::static_pointer_cast<DataSendContext>(arg);
        if (!ctx->tryClaimCallback()) {
          return;
        }
        ctx->data.reset();
        if (ctx->outputQueue != nullptr && ctx->charged) {
          ctx->outputQueue->releaseInFlightBytes(
              ctx->destination, ctx->bytes, 1L);
        }

        if (auto self = weakData.lock()) {
          self->enqueueStateEvent(self, [raw = self.get(), status]() {
            raw->sendComplete(status);
          });
        }
      },
      dataCtx);
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
            key, nullptr, /*atEnd=*/true);
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

  if (metaRequest_) {
    completedRequests_.push_back(std::move(metaRequest_));
  }
  if (dataRequest_) {
    completedRequests_.push_back(std::move(dataRequest_));
  }

  auto metaCtx = std::make_shared<MetaSendContext>();
  metaCtx->metadata = serializedMetadata;
  std::weak_ptr<UcxExchangeServer> weakMeta = weak_from_this();
  setState(ServerState::WaitingForFinalMetadataComplete);
  metaRequest_ = endpointRef_->endpoint_->tagSend(
      metaCtx->metadata.get(),
      serMetaSize,
      ucxx::Tag{metadataTag},
      false,
      [weakMeta](ucs_status_t status, std::shared_ptr<void> arg) {
        auto ctx = std::static_pointer_cast<MetaSendContext>(arg);
        if (!ctx->tryClaimCallback()) {
          return;
        }
        ctx->metadata.reset();
        if (auto self = weakMeta.lock()) {
          self->enqueueStateEvent(self, [raw = self.get(), status]() {
            raw->finalMetadataComplete(status);
          });
        }
      },
      metaCtx);
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
  std::shared_ptr<UcxGpuPayload> pending;
  {
    std::lock_guard<std::recursive_mutex> lock(dataMutex_);
    pending = std::move(dataPtr_);
  }
  if (!pending) {
    return;
  }
  // Release only what was charged; the queue can be absent.
  if (outputQueue_ != nullptr && pending->inFlightCharged) {
    outputQueue_->releaseInFlightBytes(
        partitionKey_.destination, pending->payloadBytes(), 1L);
  }
}

void UcxExchangeServer::releaseIntraNodeInFlightBytes() {
  if (!intraNodeBytesInFlight_) {
    return;
  }
  if (outputQueue_ != nullptr) {
    outputQueue_->releaseInFlightBytes(partitionKey_.destination, bytes_, 1L);
  }
  intraNodeBytesInFlight_ = false;
}

void UcxExchangeServer::deleteOutputResults() {
  if (outputResultsDeleted_) {
    return;
  }
  outputResultsDeleted_ = true;

  // Only reader of the ordinary output buffer, so it releases the results.
  releaseDynamicUcxResources(/*releaseBuffer=*/true);

  if (!outputQueue_ && !useDynamicUcx_) {
    // The accepted source can abort before READY reaches requestData(). Use
    // the ordinary server callback here: getData() may synchronously dequeue
    // one table, and that table still needs its in-flight accounting released
    // after the stable queue pointer has been returned.
    //
    // Queue path only: under discovery this re-arms the reader being torn down.
    installDataCallback();
  }
  if (outputQueue_ && !useDynamicUcx_) {
    outputQueue_->deleteResults(partitionKey_.destination);
  }
  // Skipped under discovery, as in close(). The deleteResults above is
  // what finishes the destination.
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
