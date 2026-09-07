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

#include "velox/experimental/ucx-exchange/UcxExchangeSource.h"
#include <cudf/contiguous_split.hpp>
#include <folly/String.h>
#include <folly/Uri.h>
#include <limits>
#include <new>
#include "velox/common/EnumDefine.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"
#include "velox/experimental/ucx-exchange/UcxQueues.h"

using namespace facebook::velox::exec;
namespace facebook::velox::ucx_exchange {

namespace {
struct GpuCallbackOnce {
  bool tryClaim() {
    bool expected = false;
    return claimed.compare_exchange_strong(
        expected, true, std::memory_order_acq_rel);
  }

  std::atomic<bool> claimed{false};
};

const folly::F14FastMap<UcxExchangeSource::ReceiverState, std::string_view>&
receiverStateNames() {
  static const folly::F14FastMap<
      UcxExchangeSource::ReceiverState,
      std::string_view>
      kNames = {
          {UcxExchangeSource::ReceiverState::Created, "Created"},
          {UcxExchangeSource::ReceiverState::WaitingForHandshakeComplete,
           "WaitingForHandshakeComplete"},
          {UcxExchangeSource::ReceiverState::WaitingForHandshakeResponse,
           "WaitingForHandshakeResponse"},
          {UcxExchangeSource::ReceiverState::ReadyToReceive, "ReadyToReceive"},
          {UcxExchangeSource::ReceiverState::WaitingForMetadata,
           "WaitingForMetadata"},
          {UcxExchangeSource::ReceiverState::WaitingForReceiveCredit,
           "WaitingForReceiveCredit"},
          {UcxExchangeSource::ReceiverState::WaitingForData, "WaitingForData"},
          {UcxExchangeSource::ReceiverState::WaitingForIntraNodeData,
           "WaitingForIntraNodeData"},
          {UcxExchangeSource::ReceiverState::DrainingAfterAbort,
           "DrainingAfterAbort"},
          {UcxExchangeSource::ReceiverState::Done, "Done"},
      };
  return kNames;
}
} // namespace

VELOX_DEFINE_EMBEDDED_ENUM_NAME(
    UcxExchangeSource,
    ReceiverState,
    receiverStateNames)

int32_t UcxExchangeSource::backpressureHighWaterMark() {
  return kDefaultBackpressureHighWaterMark;
}

int32_t UcxExchangeSource::backpressureLowWaterMark() {
  return kDefaultBackpressureLowWaterMark;
}

void UcxExchangeSource::setState(ReceiverState newState) {
  auto oldState = state_.exchange(newState, std::memory_order_seq_cst);
  VLOG(2) << (isIntraNodeTransfer_ ? "[INTRA]" : "[REMOTE]") << " [ExSrc "
          << toString() << " seq=" << sequenceNumber_ << "] "
          << toName(oldState) << " -> " << toName(newState);
}

// This constructor is private.
UcxExchangeSource::UcxExchangeSource(
    const std::shared_ptr<Communicator> communicator,
    std::string_view taskId,
    std::string_view host,
    uint16_t port,
    const PartitionKey& partitionKey,
    const std::shared_ptr<UcxExchangeQueue> queue)
    : CommElement(communicator, true),
      host_(host),
      port_(port),
      taskId_(taskId),
      partitionKey_(partitionKey),
      partitionKeyHash_(fnv1a_32(partitionKey_.toString())),
      queue_(std::move(queue)) {
  setState(ReceiverState::Created);
}

/*static*/
std::shared_ptr<UcxExchangeSource> UcxExchangeSource::create(
    std::string_view taskId,
    std::string_view url,
    const std::shared_ptr<UcxExchangeQueue>& queue) {
  folly::Uri uri(url);
  // Note that there is no distinct schema for the UCXX exchange.
  // The approach is to ignore the schema and not check for HTTP or HTTPS.
  // FIXME: Can't use the HTTP port as this conflicts with Prestissimo!
  // For the time being, there's an ugly hack that just increases the port by 3.
  const std::string host = uri.host();
  int port = uri.port() + 3;
  std::shared_ptr<Communicator> communicator = Communicator::getInstance();
  auto key = extractTaskAndDestinationId(uri.path());
  auto source = std::shared_ptr<UcxExchangeSource>(
      new UcxExchangeSource(communicator, taskId, host, port, key, queue));
  VLOG(3) << source->toString()
          << " creating UcxExchangeSource for url: " << url;
  return source;
}

void UcxExchangeSource::start() {
  std::lock_guard<std::recursive_mutex> processLock(processMutex_);
  bool expected = false;
  VELOX_CHECK(
      started_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel),
      "UcxExchangeSource::start called more than once");
  auto self = getSelfPtr();
  VELOX_CHECK_NOT_NULL(self, "UcxExchangeSource must be owned by shared_ptr");
  communicator_->registerCommElement(std::move(self));
}

void UcxExchangeSource::process() {
  while (true) {
    drainStateEvents();
    if (closed_.load(std::memory_order_acquire)) {
      return;
    }

    if (closeRequested_.load(std::memory_order_acquire)) {
      if (getState() == ReceiverState::Created) {
        setState(ReceiverState::Done);
      } else if (
          handshakeAccepted_ &&
          getState() != ReceiverState::DrainingAfterAbort &&
          getState() != ReceiverState::Done) {
        startAbortDrain();
      }
    }

    switch (state_) {
      case ReceiverState::Created: {
        HostPort hp{host_, port_};
        auto self = getSelfPtr();
        auto epRef = communicator_->assocEndpointRef(self, hp);
        if (epRef) {
          setEndpoint(epRef);
          setStateIf(
              ReceiverState::Created,
              ReceiverState::WaitingForHandshakeComplete);
          sendHandshake();
          return;
        }

        auto errorMsg = fmt::format(
            "Failed to connect GPU UCX exchange source to {}:{}, task {}",
            host_,
            port_,
            partitionKey_.toString());
        LOG(ERROR) << toString() << " " << errorMsg;
        queue_->setError(errorMsg);
        deliverEndMarker();
        setState(ReceiverState::Done);
        continue;
      }
      case ReceiverState::WaitingForHandshakeComplete:
      case ReceiverState::WaitingForHandshakeResponse:
      case ReceiverState::WaitingForMetadata:
      case ReceiverState::WaitingForData:
        return;
      case ReceiverState::ReadyToReceive: {
        if (pauseForBackpressureIfNeeded()) {
          return;
        }

        if (isIntraNodeTransfer_) {
          setStateIf(
              ReceiverState::ReadyToReceive,
              ReceiverState::WaitingForIntraNodeData);
          waitForIntraNodeData();
        } else {
          setStateIf(
              ReceiverState::ReadyToReceive, ReceiverState::WaitingForMetadata);
          getMetadata();
        }

        // READY-gate the producer: the first receive/poll is established
        // before the ACK can let it publish sequence zero.
        if (handshakeAckNeeded_) {
          handshakeAckNeeded_ = false;
          sendHandshakeAck();
        }
        return;
      }
      case ReceiverState::WaitingForReceiveCredit:
        if (pendingDataReceive_) {
          startDataReceive(std::move(pendingDataReceive_));
        }
        return;
      case ReceiverState::WaitingForIntraNodeData:
        waitForIntraNodeData();
        return;
      case ReceiverState::DrainingAfterAbort:
        processAbortDrain();
        if (getState() == ReceiverState::Done) {
          continue;
        }
        return;
      case ReceiverState::Done:
        cleanUp();
        return;
    }
  }
}

void UcxExchangeSource::cleanUp() {
  if (cleanupStarted_) {
    return;
  }
  cleanupStarted_ = true;
  closed_.store(true, std::memory_order_release);

  uint32_t value = static_cast<uint32_t>(getState());
  if (value != static_cast<uint32_t>(ReceiverState::Done)) {
    // Unexpected cleanup
    VLOG(3) << toString()
            << " In UcxExchangeSource::cleanUp state == " << value;
  }

  releaseReceiveBytes(pendingDataReceive_);
  releaseReceiveBytes(activeDataReceive_);
  pendingDataReceive_.reset();
  activeDataReceive_.reset();

  if (communicator_) {
    // UCP cancellation is defined for tag receives, not tag or active-message
    // sends. Retain control sends and exact-cancel only receives that cannot
    // have a useful completion after terminal cleanup.
    if (handshakeRequest_) {
      communicator_->deferRequestCleanup(std::move(handshakeRequest_));
    }
    if (handshakeAckRequest_) {
      communicator_->deferRequestCleanup(std::move(handshakeAckRequest_));
    }
    if (abortRequest_) {
      communicator_->deferRequestCleanup(std::move(abortRequest_));
    }

    std::vector<std::shared_ptr<ucxx::Request>> receiveRequests;
    receiveRequests.reserve(3);
    if (handshakeResponseRequest_) {
      receiveRequests.push_back(std::move(handshakeResponseRequest_));
    }
    if (metadataRequest_) {
      receiveRequests.push_back(std::move(metadataRequest_));
    }
    if (dataRequest_) {
      receiveRequests.push_back(std::move(dataRequest_));
    }
    communicator_->deferTagReceiveCancellation(std::move(receiveRequests));

    for (auto& req : completedRequests_) {
      communicator_->deferRequestCleanup(std::move(req));
    }
    completedRequests_.clear();
  }

  if (endpointRef_) {
    endpointRef_->removeCommElem(getSelfPtr());
    endpointRef_ = nullptr;
  }
  if (communicator_) {
    communicator_->unregister(getSelfPtr());
  }
}

void UcxExchangeSource::close() {
  std::lock_guard<std::recursive_mutex> processLock(processMutex_);
  bool expected = false;
  if (!closeRequested_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    return;
  }

  VLOG(1) << toString() << " UcxExchangeSource::close called.";
  deliverEndMarker();
  if (started_.load(std::memory_order_acquire)) {
    communicator_->addToWorkQueue(getSelfPtr());
  }
}

void UcxExchangeSource::forceCloseForShutdown() {
  closeRequested_.store(true, std::memory_order_release);
  deliverEndMarker();
  setState(ReceiverState::Done);
  cleanUp();
  // Release receive reservations carried by callbacks that completed just
  // before forced cleanup. Later callbacks observe closed_ and become no-ops.
  drainStateEvents();
}

void UcxExchangeSource::resumeFromBackpressure() {
  std::lock_guard<std::recursive_mutex> processLock(processMutex_);
  if (closeRequested_.load(std::memory_order_acquire) ||
      closed_.load(std::memory_order_acquire)) {
    backpressureActive_.store(false, std::memory_order_release);
    return;
  }
  bool expected = true;
  if (backpressureActive_.compare_exchange_strong(
          expected, false, std::memory_order_acq_rel)) {
    communicator_->addToWorkQueue(getSelfPtr());
  }
}

folly::F14FastMap<std::string, int64_t> UcxExchangeSource::stats() const {
  VELOX_UNREACHABLE();
}

folly::F14FastMap<std::string, RuntimeMetric> UcxExchangeSource::metrics()
    const {
  folly::F14FastMap<std::string, RuntimeMetric> map;

  // these metrics will be aggregated over all exchange sources of the same
  // exchange client.
  map["ucxExchangeSource.numPackedColumns"] = metrics_.numPackedColumns_;
  map["ucxExchangeSource.totalBytes"] = metrics_.totalBytes_;
  map["ucxExchangeSource.rttPerRequest"] = metrics_.rttPerRequest_;
  return map;
}

// private methods ---
PartitionKey UcxExchangeSource::extractTaskAndDestinationId(
    std::string_view path) {
  // The URL path has the form: /v1/task/<taskId>/results/<destinationId>"
  std::vector<folly::StringPiece> components;
  folly::split('/', path, components, true);

  VELOX_CHECK_EQ(components[0], "v1");
  VELOX_CHECK_EQ(components[1], "task");
  VELOX_CHECK_EQ(components[3], "results");

  uint32_t destinationId;
  try {
    destinationId = static_cast<uint32_t>(std::stoul(components[4].str()));
  } catch (const std::exception& e) {
    VELOX_UNSUPPORTED("Illegal destination in task URL: {}", path);
  }

  return PartitionKey{components[2].str(), destinationId};
}

std::shared_ptr<UcxExchangeSource> UcxExchangeSource::getSelfPtr() {
  std::shared_ptr<UcxExchangeSource> ptr;
  try {
    ptr = shared_from_this();
  } catch (std::bad_weak_ptr& exp) {
    ptr = nullptr;
  }
  return ptr;
}

void UcxExchangeSource::enqueue(
    PackedTableWithStreamPtr data,
    uint64_t reservedReceiveBytes) {
  std::vector<velox::ContinuePromise> queuePromises;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());

    if (reservedReceiveBytes > 0) {
      queue_->releaseReceiveBytesLocked(reservedReceiveBytes);
    }
    queue_->enqueueLocked(std::move(data), queuePromises);
  }
  // wake up consumers of the UcxExchangeQueue
  for (auto& promise : queuePromises) {
    promise.setValue();
  }
}

void UcxExchangeSource::deliverEndMarker() {
  if (!registered_.load(std::memory_order_acquire)) {
    // Never registered with queue -- don't deliver end marker to avoid
    // spurious numCompleted_ increments.
    return;
  }
  bool expected = false;
  if (!endMarkerDelivered_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    // Already delivered by another thread/path.
    return;
  }
  VLOG(3) << toString() << " delivering end-of-stream marker to queue";
  enqueue(nullptr);
}

void UcxExchangeSource::setEndpoint(std::shared_ptr<EndpointRef> endpointRef) {
  endpointRef_ = std::move(endpointRef);
}

void UcxExchangeSource::sendHandshake() {
  std::shared_ptr<HandshakeMsg> handshakeReq = std::make_shared<HandshakeMsg>();
  handshakeReq->destination = partitionKey_.destination;
  // Use sizeof(...) - 1 and explicitly null-terminate to prevent buffer
  // overread if taskId is longer than the destination buffer.
  strncpy(
      handshakeReq->taskId,
      partitionKey_.taskId.c_str(),
      sizeof(handshakeReq->taskId) - 1);
  handshakeReq->taskId[sizeof(handshakeReq->taskId) - 1] = '\0';
  handshakeReq->workerId = communicator_->getWorkerId();

  VLOG(3) << toString() << " Sending handshake with initial value: "
          << partitionKey_.toString() << " to server";

  // Create the handshake which will register client's existence with the server
  ucxx::AmReceiverCallbackInfo info(
      communicator_->kAmCallbackOwner, communicator_->kAmCallbackId);
  // Use weak_ptr to prevent use-after-free if close() is called during callback
  std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
  auto callbackOnce = std::make_shared<GpuCallbackOnce>();
  // Pass handshakeReq as the callback arg to keep the send buffer alive until
  // the async amSend completes. UCXX stores it as shared_ptr<void> but the
  // type-erased deleter still calls ~HandshakeMsg correctly.
  handshakeRequest_ = endpointRef_->endpoint_->amSend(
      handshakeReq.get(),
      sizeof(*handshakeReq),
      UCS_MEMORY_TYPE_HOST,
      info,
      false,
      [weak, callbackOnce](ucs_status_t status, std::shared_ptr<void> arg) {
        if (!callbackOnce->tryClaim()) {
          return;
        }
        if (auto self = weak.lock()) {
          self->enqueueStateEvent(
              self, [raw = self.get(), status, arg = std::move(arg)]() mutable {
                raw->onHandshake(status, std::move(arg));
              });
        }
      },
      handshakeReq);
}

void UcxExchangeSource::onHandshake(
    ucs_status_t status,
    std::shared_ptr<void> /*arg*/) {
  // arg holds the HandshakeMsg that was sent. It is unused here because this
  // is a send completion callback (the outgoing data has already been
  // transmitted). The parameter exists only because UCXX uses it as a lifetime
  // handle; letting it go out of scope releases the send buffer.

  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString() << " onHandshake called after close, ignoring";
    deliverEndMarker();
    return;
  }
  // Guard against replayed callbacks from UCP wireup replay.
  if (getState() != ReceiverState::WaitingForHandshakeComplete) {
    VLOG(2) << toString() << " onHandshake called in state "
            << toName(getState()) << ", ignoring (possible UCXX replay)";
    return;
  }
  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to send handshake to host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    failAndCloseEndpoint(std::move(errorMsg));
  } else {
    VLOG(3) << toString() << "+ onHandshake " << ucs_status_string(status);
    // Now wait for the HandshakeResponse from the server
    setStateIf(
        ReceiverState::WaitingForHandshakeComplete,
        ReceiverState::WaitingForHandshakeResponse);
    receiveHandshakeResponse();
  }
}

void UcxExchangeSource::getMetadata() {
  // Use kMaxMetaBufSize to support tables with many columns.
  // The sender allocates exact size needed; receiver pre-allocates max.
  auto metadataReq = std::make_shared<std::vector<uint8_t>>(kMaxMetaBufSize);
  uint64_t metadataTag = getMetadataTag(partitionKeyHash_, sequenceNumber_);

  VLOG(3) << toString()
          << " waiting for metadata for chunk: " << sequenceNumber_
          << " using tag: " << std::hex << metadataTag << std::dec;

  // Use weak_ptr to prevent use-after-free if close() is called during callback
  std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
  auto callbackOnce = std::make_shared<GpuCallbackOnce>();
  if (metadataRequest_) {
    completedRequests_.push_back(std::move(metadataRequest_));
  }
  metadataReceivePosted_ = true;
  metadataRequest_ = endpointRef_->endpoint_->tagRecv(
      reinterpret_cast<void*>(metadataReq->data()),
      kMaxMetaBufSize,
      ucxx::Tag{metadataTag},
      ucxx::TagMaskFull,
      false,
      [weak, callbackOnce](ucs_status_t status, std::shared_ptr<void> arg) {
        if (!callbackOnce->tryClaim()) {
          return;
        }
        if (auto self = weak.lock()) {
          self->enqueueStateEvent(
              self, [raw = self.get(), status, arg = std::move(arg)]() mutable {
                raw->onMetadata(status, std::move(arg));
              });
        }
      },
      metadataReq);
}

void UcxExchangeSource::onMetadata(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  metadataReceivePosted_ = false;
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString() << " onMetadata called after close, ignoring";
    deliverEndMarker();
    return;
  }
  // Guard against replayed callbacks from UCP wireup replay.
  const auto state = getState();
  const bool draining = state == ReceiverState::DrainingAfterAbort;
  if (state != ReceiverState::WaitingForMetadata && !draining) {
    VLOG(2) << toString() << " onMetadata called in state "
            << toName(getState()) << ", ignoring (possible UCXX replay)";
    return;
  }
  VLOG(3) << toString() << " + onMetadata " << ucs_status_string(status);

  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to receive metadata from host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    failAndCloseEndpoint(std::move(errorMsg));
  } else {
    if (!arg) {
      failAndCloseEndpoint(
          fmt::format(
              "GPU exchange metadata callback returned no buffer for {}",
              partitionKey_.toString()));
      return;
    }

    // arg contains the actual serialized metadata, deserialize the metadata
    std::shared_ptr<std::vector<uint8_t>> metadataMsg =
        std::static_pointer_cast<std::vector<uint8_t>>(arg);

    auto ptr = std::make_shared<DataAndMetadata>();

    try {
      ptr->metadata =
          std::move(MetadataMsg::deserializeMetadataMsg(metadataMsg->data()));
      VELOX_CHECK_GE(
          ptr->metadata.dataSizeBytes, 0, "UCX metadata data size is negative");
    } catch (const std::exception& e) {
      failAndCloseEndpoint(
          fmt::format(
              "Failed to decode GPU exchange metadata for {}: {}",
              partitionKey_.toString(),
              e.what()));
      return;
    } catch (...) {
      failAndCloseEndpoint(
          fmt::format(
              "Failed to decode GPU exchange metadata for {}",
              partitionKey_.toString()));
      return;
    }

    VLOG(3) << toString()
            << " Datasize bytes == " << ptr->metadata.dataSizeBytes;

    if (ptr->metadata.atEnd) {
      // It seems that all data has been transferred
      atEnd_ = true;
      // enqueue a nullpointer to mark the end for this source.
      VLOG(3) << "There is no more data to transfer for " << toString();
      deliverEndMarker();
      setState(ReceiverState::Done);
      communicator_->addToWorkQueue(getSelfPtr());
      // jump out of this function.
      return;
    }

    startDataReceive(std::move(ptr));
  }
}

bool UcxExchangeSource::tryReserveReceiveBytes(
    std::shared_ptr<DataAndMetadata> ptr) {
  VELOX_CHECK_NOT_NULL(ptr);
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    const auto dataSizeBytes =
        static_cast<uint64_t>(ptr->metadata.dataSizeBytes);
    if (queue_->tryReserveReceiveBytesLocked(dataSizeBytes)) {
      ptr->receiveBytesReserved = true;
      return true;
    }

    pendingDataReceive_ = ptr;
    backpressureActive_.store(true, std::memory_order_release);
  }

  const auto state = getState();
  if (state == ReceiverState::WaitingForMetadata) {
    setStateIf(
        ReceiverState::WaitingForMetadata,
        ReceiverState::WaitingForReceiveCredit);
  } else {
    VELOX_CHECK(
        state == ReceiverState::WaitingForReceiveCredit,
        "Unexpected state {} while waiting for receive credit",
        toName(state));
  }
  return false;
}

void UcxExchangeSource::releaseReceiveBytes(
    std::shared_ptr<DataAndMetadata> ptr) {
  if (!ptr || !ptr->receiveBytesReserved) {
    return;
  }

  std::lock_guard<std::mutex> l(queue_->mutex());
  queue_->releaseReceiveBytesLocked(
      static_cast<uint64_t>(ptr->metadata.dataSizeBytes));
  ptr->receiveBytesReserved = false;
}

void UcxExchangeSource::startDataReceive(std::shared_ptr<DataAndMetadata> ptr) {
  VELOX_CHECK_NOT_NULL(ptr);
  if (closed_.load(std::memory_order_acquire)) {
    deliverEndMarker();
    return;
  }

  const auto expectedState = getState();
  const bool draining = expectedState == ReceiverState::DrainingAfterAbort;
  VELOX_CHECK(
      expectedState == ReceiverState::WaitingForMetadata ||
          expectedState == ReceiverState::WaitingForReceiveCredit || draining,
      "Unexpected state {} before starting UCX data receive",
      toName(expectedState));

  // Once abort has been sent, consumer flow control must not prevent us from
  // landing and discarding a payload the producer already committed.
  if (!draining && !tryReserveReceiveBytes(ptr)) {
    return;
  }

  void* receiveBuffer{nullptr};
  if (draining) {
    // A query can be canceled because device memory is exhausted. Land the
    // already-committed GPU payload in ordinary host memory so cancellation
    // never depends on another RMM allocation, then discard it in onData().
    const auto discardSize = static_cast<uint64_t>(ptr->metadata.dataSizeBytes);
    if (discardSize > std::numeric_limits<size_t>::max()) {
      failAndCloseEndpoint(
          fmt::format(
              "GPU exchange drain payload {} is too large for host address space "
              "for {}",
              discardSize,
              partitionKey_.toString()));
      return;
    }
    try {
      // Default-initialized scalar array storage avoids zero-filling a payload
      // that is immediately overwritten by UCX and discarded.
      ptr->discardBuf = std::unique_ptr<uint8_t[]>(
          new uint8_t[static_cast<size_t>(discardSize)]);
    } catch (const std::exception& e) {
      auto errorMsg = fmt::format(
          "Failed to allocate {} host bytes while draining GPU exchange {}: {}",
          discardSize,
          partitionKey_.toString(),
          e.what());
      failAndCloseEndpoint(std::move(errorMsg));
      return;
    }
    receiveBuffer = ptr->discardBuf.get();
  } else if (!ptr->metadata.isDeviceData) {
    // The producer did not pack this one. It is a serialized page, and it is
    // deserialized on the host, so receive it there.
    try {
      ptr->hostBuf = std::unique_ptr<uint8_t[]>(
          new uint8_t[static_cast<size_t>(ptr->metadata.dataSizeBytes)]);
    } catch (const std::exception& e) {
      // Neither the pending nor the active request, so close() misses it.
      releaseReceiveBytes(ptr);
      auto errorMsg = fmt::format(
          "Failed to allocate {} host bytes for GPU exchange {}: {}",
          ptr->metadata.dataSizeBytes,
          partitionKey_.toString(),
          e.what());
      failAndCloseEndpoint(std::move(errorMsg));
      return;
    }
    receiveBuffer = ptr->hostBuf.get();
  } else {
    // Normal remote exchange receives directly into device memory.
    auto stream =
        facebook::velox::cudf_velox::cudfGlobalStreamPool().get_stream();
    ptr->stream = stream;
    try {
      ptr->dataBuf = std::make_unique<rmm::device_buffer>(
          ptr->metadata.dataSizeBytes,
          stream,
          cudf::get_current_device_resource_ref());
    } catch (const rmm::bad_alloc& e) {
      releaseReceiveBytes(ptr);
      VLOG(0) << toString() << " *** RMM  failed to allocate: " << e.what();
      bool retryLater = false;
      {
        std::lock_guard<std::mutex> l(queue_->mutex());
        retryLater = queue_->recordReceiveAllocationPressureLocked(
            static_cast<uint64_t>(ptr->metadata.dataSizeBytes));
        if (retryLater) {
          // Publish the dormant state while holding the same mutex consumers
          // use to decide whether to wake sources. This prevents a dequeue
          // from observing the old value and losing the only wake-up.
          backpressureActive_.store(true, std::memory_order_release);
        }
      }
      if (retryLater) {
        pendingDataReceive_ = std::move(ptr);
        if (expectedState == ReceiverState::WaitingForMetadata) {
          setStateIf(
              ReceiverState::WaitingForMetadata,
              ReceiverState::WaitingForReceiveCredit);
        }
        return;
      }

      // The producer committed this data tag when it sent the metadata. Keep
      // the metadata, enter the abort protocol, and land the payload in the
      // host discard buffer. Going directly to Done would leave the producer's
      // GPU send unmatched—the exact stale state that cancellation must clear.
      pendingDataReceive_ = std::move(ptr);
      failAndStartAbortDrain(
          fmt::format(
              "Failed to allocate {} GPU bytes for exchange {}",
              pendingDataReceive_->metadata.dataSizeBytes,
              partitionKey_.toString()));
      return;
    }

    // UCX receives into this raw pointer on a UCX-owned CUDA stream. The
    // allocation comes from an async RMM pool on `stream`, so make the
    // allocation complete before handing the pointer to UCX.
    stream.synchronize();
    receiveBuffer = ptr->dataBuf->data();
  }

  VLOG(3) << toString() << " Allocated " << ptr->metadata.dataSizeBytes
          << (draining ? " host drain bytes" : " bytes of device memory");

  uint64_t dataTag = getDataTag(partitionKeyHash_, sequenceNumber_);
  VLOG(3) << toString() << " waiting for data for chunk: " << sequenceNumber_
          << " using tag: " << std::hex << dataTag << std::dec;

  if (!draining && !setStateIf(expectedState, ReceiverState::WaitingForData)) {
    releaseReceiveBytes(ptr);
    VLOG(1) << toString() << " startDataReceive invalid previous state "
            << toName(getState());
    return;
  }

  std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
  auto callbackOnce = std::make_shared<GpuCallbackOnce>();
  if (dataRequest_) {
    completedRequests_.push_back(std::move(dataRequest_));
  }
  activeDataReceive_ = ptr;
  dataReceivePosted_ = true;
  dataRequest_ = endpointRef_->endpoint_->tagRecv(
      receiveBuffer,
      ptr->metadata.dataSizeBytes,
      ucxx::Tag{dataTag},
      ucxx::TagMaskFull,
      false,
      [weak, callbackOnce](ucs_status_t status, std::shared_ptr<void> arg) {
        if (!callbackOnce->tryClaim()) {
          return;
        }
        if (auto self = weak.lock()) {
          self->enqueueStateEvent(
              self, [raw = self.get(), status, arg = std::move(arg)]() mutable {
                raw->onData(status, std::move(arg));
              });
        }
      },
      ptr);
}

void UcxExchangeSource::onData(ucs_status_t status, std::shared_ptr<void> arg) {
  dataReceivePosted_ = false;
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString() << " onData called after close, ignoring";
    releaseReceiveBytes(std::static_pointer_cast<DataAndMetadata>(arg));
    deliverEndMarker();
    return;
  }
  // Guard against replayed callbacks from UCP wireup replay.
  const auto state = getState();
  const bool draining = state == ReceiverState::DrainingAfterAbort;
  if (state != ReceiverState::WaitingForData && !draining) {
    VLOG(2) << toString() << " onData called in state " << toName(getState())
            << ", ignoring (possible UCXX replay)";
    releaseReceiveBytes(std::static_pointer_cast<DataAndMetadata>(arg));
    return;
  }
  VLOG(3) << toString() << " + onData " << ucs_status_string(status);

  if (status != UCS_OK) {
    auto ptr = std::static_pointer_cast<DataAndMetadata>(arg);
    releaseReceiveBytes(ptr);
    activeDataReceive_.reset();
    std::string errorMsg = fmt::format(
        "Failed to receive data from host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    failAndCloseEndpoint(std::move(errorMsg));
  } else {
    if (!arg) {
      releaseReceiveBytes(activeDataReceive_);
      activeDataReceive_.reset();
      failAndCloseEndpoint(
          fmt::format(
              "GPU exchange data callback returned no buffer for {}",
              partitionKey_.toString()));
      return;
    }
    VLOG(3) << toString() << "+ onData " << ucs_status_string(status)
            << " got chunk: " << sequenceNumber_;

    this->sequenceNumber_++;

    std::shared_ptr<DataAndMetadata> ptr =
        std::static_pointer_cast<DataAndMetadata>(arg);

    if (draining) {
      releaseReceiveBytes(ptr);
      activeDataReceive_.reset();
      communicator_->addToWorkQueue(getSelfPtr());
      return;
    }

    const uint64_t reservedReceiveBytes = ptr->receiveBytesReserved
        ? static_cast<uint64_t>(ptr->metadata.dataSizeBytes)
        : 0;

    metrics_.numPackedColumns_.addValue(1);
    metrics_.totalBytes_.addValue(ptr->metadata.dataSizeBytes);

    PackedTableWithStreamPtr data;
    if (!ptr->metadata.isDeviceData) {
      // Stays host bytes; the exchange operator decides whether to upload.
      auto* raw = ptr->hostBuf.release();
      auto page = folly::IOBuf::takeOwnership(
          raw,
          static_cast<size_t>(ptr->metadata.dataSizeBytes),
          [](void* buf, void* /*userData*/) {
            delete[] static_cast<uint8_t*>(buf);
          });
      data = std::make_unique<PackedTableWithStream>(
          std::move(page),
          facebook::velox::cudf_velox::cudfGlobalStreamPool().get_stream(),
          ptr->metadata.numRows);
    } else {
      try {
        // Create packed_columns from the received metadata and data buffer.
        cudf::packed_columns packedCols(
            std::move(ptr->metadata.cudfMetadata), std::move(ptr->dataBuf));

        // Unpack to get the table_view and create a packed_table.
        cudf::table_view tableView = cudf::unpack(packedCols);
        auto packedTable = std::make_unique<cudf::packed_table>(
            cudf::packed_table{tableView, std::move(packedCols)});

        // Bundle the packed_table with the stream used for allocation.
        auto numRows = ptr->metadata.numRows;
        if (numRows < 0) {
          VELOX_CHECK_GT(
              tableView.num_columns(),
              0,
              "Legacy UCX metadata cannot represent logical rows for a "
              "zero-column payload; upgrade producer and consumer together");
          numRows = tableView.num_rows();
        }
        data = std::make_unique<PackedTableWithStream>(
            std::move(packedTable), ptr->stream, numRows);
      } catch (const std::exception& e) {
        releaseReceiveBytes(ptr);
        activeDataReceive_.reset();
        failAndStartAbortDrain(
            fmt::format(
                "Failed to unpack GPU exchange payload for {}: {}",
                partitionKey_.toString(),
                e.what()));
        return;
      } catch (...) {
        releaseReceiveBytes(ptr);
        activeDataReceive_.reset();
        failAndStartAbortDrain(
            fmt::format(
                "Failed to unpack GPU exchange payload for {}",
                partitionKey_.toString()));
        return;
      }
    }

    ptr->receiveBytesReserved = false;
    enqueue(std::move(data), reservedReceiveBytes);
    activeDataReceive_.reset();
    setStateIf(ReceiverState::WaitingForData, ReceiverState::ReadyToReceive);
  }
  communicator_->addToWorkQueue(getSelfPtr());
}

void UcxExchangeSource::receiveHandshakeResponse() {
  auto responseBuffer = std::make_shared<HandshakeResponse>();
  uint64_t responseTag = getHandshakeResponseTag(partitionKeyHash_);

  VLOG(3) << toString()
          << " waiting for HandshakeResponse with tag: " << std::hex
          << responseTag << std::dec;

  // Use weak_ptr to prevent use-after-free if close() is called during callback
  std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
  auto callbackOnce = std::make_shared<GpuCallbackOnce>();
  handshakeResponseRequest_ = endpointRef_->endpoint_->tagRecv(
      responseBuffer.get(),
      sizeof(*responseBuffer),
      ucxx::Tag{responseTag},
      ucxx::TagMaskFull,
      false,
      [weak, callbackOnce](ucs_status_t status, std::shared_ptr<void> arg) {
        if (!callbackOnce->tryClaim()) {
          return;
        }
        if (auto self = weak.lock()) {
          self->enqueueStateEvent(
              self, [raw = self.get(), status, arg = std::move(arg)]() mutable {
                raw->onHandshakeResponse(status, std::move(arg));
              });
        }
      },
      responseBuffer);
}

void UcxExchangeSource::onHandshakeResponse(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString()
            << " onHandshakeResponse called after close, ignoring";
    deliverEndMarker();
    return;
  }
  // Guard against replayed callbacks from UCP wireup replay.
  if (getState() != ReceiverState::WaitingForHandshakeResponse) {
    VLOG(2) << toString() << " onHandshakeResponse called in state "
            << toName(getState()) << ", ignoring (possible UCXX replay)";
    return;
  }

  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to receive HandshakeResponse from host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    failAndCloseEndpoint(std::move(errorMsg));
    return;
  }

  std::shared_ptr<HandshakeResponse> response =
      std::static_pointer_cast<HandshakeResponse>(arg);
  if (!response) {
    failAndCloseEndpoint(
        fmt::format(
            "GPU exchange handshake response returned no buffer for {}",
            partitionKey_.toString()));
    return;
  }

  if (response->status != GpuHandshakeResponseStatus::kAccepted) {
    std::string errorMsg = fmt::format(
        "GPU exchange producer rejected the handshake from host {}:{}, task "
        "{} (status={})",
        host_,
        port_,
        partitionKey_.toString(),
        static_cast<int>(response->status));
    LOG(ERROR) << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
    communicator_->addToWorkQueue(getSelfPtr());
    return;
  }

  isIntraNodeTransfer_ = response->isIntraNodeTransfer;
  handshakeAccepted_ = true;
  handshakeAckNeeded_ = true;

  VLOG(3) << toString() << " + onHandshakeResponse isIntraNodeTransfer="
          << isIntraNodeTransfer_;

  if (closeRequested_.load(std::memory_order_acquire)) {
    startAbortDrain();
  } else {
    setStateIf(
        ReceiverState::WaitingForHandshakeResponse,
        ReceiverState::ReadyToReceive);
  }
  communicator_->addToWorkQueue(getSelfPtr());
}

void UcxExchangeSource::sendHandshakeAck() {
  auto ack = std::make_shared<GpuHandshakeAckHeader>();
  const uint64_t ackTag = getGpuHandshakeAckTag(partitionKeyHash_);

  std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
  auto callbackOnce = std::make_shared<GpuCallbackOnce>();
  handshakeAckRequest_ = endpointRef_->endpoint_->tagSend(
      ack.get(),
      sizeof(*ack),
      ucxx::Tag{ackTag},
      false,
      [weak, callbackOnce](ucs_status_t status, std::shared_ptr<void> arg) {
        if (!callbackOnce->tryClaim()) {
          return;
        }
        if (auto self = weak.lock()) {
          self->enqueueStateEvent(
              self, [raw = self.get(), status, arg = std::move(arg)]() mutable {
                raw->onHandshakeAck(status, std::move(arg));
              });
        }
      },
      ack);
}

void UcxExchangeSource::onHandshakeAck(
    ucs_status_t status,
    std::shared_ptr<void> /*arg*/) {
  if (closed_.load(std::memory_order_acquire) || status == UCS_OK) {
    return;
  }

  auto errorMsg = fmt::format(
      "Failed to send GPU handshake ACK to host {}:{}, task {}: {}",
      host_,
      port_,
      partitionKey_.toString(),
      ucs_status_string(status));
  failAndCloseEndpoint(std::move(errorMsg));
}

void UcxExchangeSource::sendAbort() {
  if (abortSent_ || !handshakeAccepted_) {
    return;
  }
  abortSent_ = true;

  auto abort = std::make_shared<GpuAbortHeader>();
  const uint64_t abortTag = getGpuAbortTag(partitionKeyHash_);
  std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
  auto callbackOnce = std::make_shared<GpuCallbackOnce>();
  abortRequest_ = endpointRef_->endpoint_->tagSend(
      abort.get(),
      sizeof(*abort),
      ucxx::Tag{abortTag},
      false,
      [weak, callbackOnce](ucs_status_t status, std::shared_ptr<void> arg) {
        if (!callbackOnce->tryClaim()) {
          return;
        }
        if (auto self = weak.lock()) {
          self->enqueueStateEvent(
              self, [raw = self.get(), status, arg = std::move(arg)]() mutable {
                raw->onAbortSent(status, std::move(arg));
              });
        }
      },
      abort);
}

void UcxExchangeSource::onAbortSent(
    ucs_status_t status,
    std::shared_ptr<void> /*arg*/) {
  if (closed_.load(std::memory_order_acquire) || status == UCS_OK) {
    return;
  }

  auto errorMsg = fmt::format(
      "Failed to send GPU abort control to host {}:{}, task {}: {}",
      host_,
      port_,
      partitionKey_.toString(),
      ucs_status_string(status));
  failAndCloseEndpoint(std::move(errorMsg));
}

void UcxExchangeSource::startAbortDrain() {
  if (!handshakeAccepted_ || getState() == ReceiverState::Done) {
    return;
  }
  if (atEnd_) {
    setState(ReceiverState::Done);
    return;
  }

  setState(ReceiverState::DrainingAfterAbort);
  backpressureActive_.store(false, std::memory_order_release);
  processAbortDrain();
}

void UcxExchangeSource::processAbortDrain() {
  if (atEnd_) {
    deliverEndMarker();
    setState(ReceiverState::Done);
    return;
  }

  if (isIntraNodeTransfer_) {
    waitForIntraNodeData();
  } else if (pendingDataReceive_) {
    auto pending = std::move(pendingDataReceive_);
    startDataReceive(std::move(pending));
  } else if (!metadataReceivePosted_ && !dataReceivePosted_) {
    getMetadata();
  }

  // The first receive/poll above precedes the READY ACK. The abort follows the
  // ACK so the producer can stop before pulling another queue entry.
  if (handshakeAckNeeded_) {
    handshakeAckNeeded_ = false;
    sendHandshakeAck();
  }
  sendAbort();
}

void UcxExchangeSource::failAndStartAbortDrain(std::string errorMessage) {
  LOG(ERROR) << toString() << " " << errorMessage;
  queue_->setError(errorMessage);
  deliverEndMarker();
  closeRequested_.store(true, std::memory_order_release);

  if (handshakeAccepted_ && getState() != ReceiverState::Done) {
    startAbortDrain();
  } else {
    setState(ReceiverState::Done);
  }
  communicator_->addToWorkQueue(getSelfPtr());
}

void UcxExchangeSource::failAndCloseEndpoint(std::string errorMessage) {
  LOG(ERROR) << toString() << " " << errorMessage;
  queue_->setError(errorMessage);
  deliverEndMarker();
  closeRequested_.store(true, std::memory_order_release);
  setState(ReceiverState::Done);

  if (!endpointCleanupRequested_ && endpointRef_) {
    endpointCleanupRequested_ = true;
    communicator_->deferEndpointCleanup(endpointRef_);
  }
  communicator_->addToWorkQueue(getSelfPtr());
}

void UcxExchangeSource::waitForIntraNodeData() {
  // Check if close() was called
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString()
            << " waitForIntraNodeData called after close, ignoring";
    deliverEndMarker();
    return;
  }

  IntraNodeTransferKey key{
      partitionKey_.taskId, partitionKey_.destination, sequenceNumber_};

  auto result = IntraNodeTransferRegistry::getInstance()->poll(key);

  if (!result.has_value()) {
    // Data not ready yet, re-queue to try again
    ++intraNodePollCount_;
    if (intraNodePollCount_ % 100 == 0) {
      VLOG(2) << "[INTRA] [ExSrc " << toString() << " seq=" << sequenceNumber_
              << "] still polling for data, polls=" << intraNodePollCount_;
    }
    communicator_->addToWorkQueue(getSelfPtr());
    return;
  }

  intraNodePollCount_ = 0;
  onIntraNodeData(std::move(result->data), result->atEnd);
}

void UcxExchangeSource::onIntraNodeData(
    std::shared_ptr<UcxGpuPayload> data,
    bool atEnd) {
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString() << " onIntraNodeData called after close, ignoring";
    deliverEndMarker();
    return;
  }

  if (atEnd) {
    // End of stream
    atEnd_ = true;
    VLOG(3) << toString() << " Intra-node transfer: end of stream";
    deliverEndMarker();
    setState(ReceiverState::Done);

    communicator_->addToWorkQueue(getSelfPtr());
    return;
  }

  if (!data) {
    // Error - should not happen if atEnd is false
    std::string errorMsg = fmt::format(
        "Intra-node transfer data is null for task {}, dest {}, seq {}",
        partitionKey_.taskId,
        partitionKey_.destination,
        sequenceNumber_);
    ++sequenceNumber_;
    failAndStartAbortDrain(std::move(errorMsg));
    return;
  }

  const bool draining = getState() == ReceiverState::DrainingAfterAbort;
  if (draining) {
    ++sequenceNumber_;
    communicator_->addToWorkQueue(getSelfPtr());
    return;
  }

  if (getState() != ReceiverState::WaitingForIntraNodeData) {
    VLOG(2) << toString() << " onIntraNodeData called in state "
            << toName(getState()) << ", ignoring";
    return;
  }

  VLOG(3) << toString()
          << " Intra-node transfer: received data for seq=" << sequenceNumber_
          << " size=" << data->payloadBytes();

  metrics_.numPackedColumns_.addValue(1);
  metrics_.totalBytes_.addValue(data->payloadBytes());

  PackedTableWithStreamPtr tableWithStream;
  try {
    // The producer synchronized before publishing, so the GPU data is ready.
    auto stream =
        facebook::velox::cudf_velox::cudfGlobalStreamPool().get_stream();

    if (data->data != nullptr) {
      // The producer's own queues held the table, so take ownership of it.
      cudf::packed_columns packedCols(
          std::move(data->data->metadata), std::move(data->data->gpu_data));

      cudf::table_view tableView = cudf::unpack(packedCols);
      auto packedTable = std::make_unique<cudf::packed_table>(
          cudf::packed_table{tableView, std::move(packedCols)});
      tableWithStream = std::make_unique<PackedTableWithStream>(
          std::move(packedTable), stream, data->numRows);
    } else if (!data->deviceResident) {
      auto page = data->buffer->clone();
      tableWithStream = std::make_unique<PackedTableWithStream>(
          std::move(page), stream, data->numRows);
    } else {
      // The output buffer still owns the bytes; view them and keep it alive.
      cudf::table_view tableView = cudf::unpack(
          static_cast<uint8_t const*>(data->metadata()),
          static_cast<uint8_t const*>(data->payload()));
      const auto deviceBytes = data->payloadBytes();
      tableWithStream = std::make_unique<PackedTableWithStream>(
          tableView,
          std::shared_ptr<void>(data),
          deviceBytes,
          stream,
          data->numRows);
    }
  } catch (const std::exception& e) {
    ++sequenceNumber_;
    failAndStartAbortDrain(
        fmt::format(
            "Failed to unpack intra-node GPU exchange payload for {}: {}",
            partitionKey_.toString(),
            e.what()));
    return;
  } catch (...) {
    ++sequenceNumber_;
    failAndStartAbortDrain(
        fmt::format(
            "Failed to unpack intra-node GPU exchange payload for {}",
            partitionKey_.toString()));
    return;
  }

  enqueue(std::move(tableWithStream));

  this->sequenceNumber_++;
  setStateIf(
      ReceiverState::WaitingForIntraNodeData, ReceiverState::ReadyToReceive);
  communicator_->addToWorkQueue(getSelfPtr());
}

bool UcxExchangeSource::setStateIf(
    UcxExchangeSource::ReceiverState expected,
    UcxExchangeSource::ReceiverState desired) {
  ReceiverState exp = expected;
  // since spurious failures can happen even if state_ == expected, we need
  // to do this in a loop.
  while (!state_.compare_exchange_strong(
      exp, desired, std::memory_order_acq_rel, std::memory_order_relaxed)) {
    if (exp != expected) {
      // no spurious failure, state isn't what we've expected.
      return false;
    }
    // spurious failure.
    exp = expected; // reset for the next try
  }
  VLOG(2) << (isIntraNodeTransfer_ ? "[INTRA]" : "[REMOTE]") << " [ExSrc "
          << toString() << " seq=" << sequenceNumber_ << "] "
          << toName(expected) << " -> " << toName(desired);
  return true;
}

bool UcxExchangeSource::pauseForBackpressureIfNeeded() {
  std::lock_guard<std::mutex> l(queue_->mutex());
  const bool shouldPause = !queue_->receiveBytesBelowPrefetchLimitLocked() ||
      queue_->size() >= backpressureHighWaterMark();
  if (shouldPause) {
    // Consumers inspect the same queue state under this mutex before calling
    // resumeFromBackpressure(). Publish the sleep decision before releasing
    // it so they cannot drain the queue and miss the transition.
    backpressureActive_.store(true, std::memory_order_release);
  }
  return shouldPause;
}

} // namespace facebook::velox::ucx_exchange
