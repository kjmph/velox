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
#include "velox/experimental/ucx-exchange/UcxCpuRowExchangeSource.h"
#include <fmt/format.h>
#include <folly/String.h>
#include <folly/Uri.h>
#include <folly/io/IOBuf.h>
#include <glog/logging.h>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include "velox/common/EnumDefine.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowMetadataMsg.h"

using namespace facebook::velox::exec;
namespace facebook::velox::ucx_exchange {

namespace {
const folly::
    F14FastMap<UcxCpuRowExchangeSource::ReceiverState, std::string_view>&
    receiverStateNames() {
  static const folly::F14FastMap<
      UcxCpuRowExchangeSource::ReceiverState,
      std::string_view>
      kNames = {
          {UcxCpuRowExchangeSource::ReceiverState::Created, "Created"},
          {UcxCpuRowExchangeSource::ReceiverState::WaitingForHandshakeComplete,
           "WaitingForHandshakeComplete"},
          {UcxCpuRowExchangeSource::ReceiverState::WaitingForHandshakeResponse,
           "WaitingForHandshakeResponse"},
          {UcxCpuRowExchangeSource::ReceiverState::ReadyToReceive,
           "ReadyToReceive"},
          {UcxCpuRowExchangeSource::ReceiverState::WaitingForMetadata,
           "WaitingForMetadata"},
          {UcxCpuRowExchangeSource::ReceiverState::WaitingForData,
           "WaitingForData"},
          {UcxCpuRowExchangeSource::ReceiverState::Done, "Done"},
      };
  return kNames;
}

uint32_t maxInFlightBundles() {
  static const uint32_t kDepth = []() -> uint32_t {
    const char* env = std::getenv("VELOX_UCX_CPU_PIPELINE_DEPTH");
    if (env == nullptr || *env == '\0') {
      return 1U;
    }

    char* end = nullptr;
    errno = 0;
    long value = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || errno != 0 || value < 1) {
      VLOG(1) << "Ignoring invalid VELOX_UCX_CPU_PIPELINE_DEPTH=" << env;
      return 1U;
    }
    return static_cast<uint32_t>(std::min<long>(value, 64));
  }();
  return kDepth;
}

int32_t readIntEnv(
    const char* name,
    int32_t defaultValue,
    int32_t minValue,
    int32_t maxValue) {
  const char* env = std::getenv(name);
  if (env == nullptr || *env == '\0') {
    return defaultValue;
  }

  char* end = nullptr;
  errno = 0;
  long value = std::strtol(env, &end, 10);
  if (end == env || *end != '\0' || errno != 0 || value < minValue ||
      value > maxValue) {
    VLOG(1) << "Ignoring invalid " << name << "=" << env << " (expected "
            << minValue << ".." << maxValue << ")";
    return defaultValue;
  }
  return static_cast<int32_t>(value);
}
} // namespace

VELOX_DEFINE_EMBEDDED_ENUM_NAME(
    UcxCpuRowExchangeSource,
    ReceiverState,
    receiverStateNames)

int32_t UcxCpuRowExchangeSource::backpressureHighWaterMark() {
  static const int32_t kHighWater = readIntEnv(
      "VELOX_UCX_CPU_RECV_QUEUE_HIGH_WATER",
      kDefaultBackpressureHighWaterMark,
      1,
      16384);
  return kHighWater;
}

int32_t UcxCpuRowExchangeSource::backpressureLowWaterMark() {
  static const int32_t kLowWater = [] {
    const int32_t highWater =
        UcxCpuRowExchangeSource::backpressureHighWaterMark();
    const int32_t defaultLowWater =
        std::min(kDefaultBackpressureLowWaterMark, highWater);
    return readIntEnv(
        "VELOX_UCX_CPU_RECV_QUEUE_LOW_WATER", defaultLowWater, 0, highWater);
  }();
  return kLowWater;
}

void UcxCpuRowExchangeSource::setState(ReceiverState newState) {
  auto oldState = state_.exchange(newState, std::memory_order_seq_cst);
  VLOG(4) << "[ExSrc-CPU " << toString()
          << " nextMeta=" << nextMetadataSequence_
          << " nextDeliver=" << nextDeliverySequence_ << "] "
          << toName(oldState) << " -> " << toName(newState);
}

UcxCpuRowExchangeSource::UcxCpuRowExchangeSource(
    const std::shared_ptr<Communicator> communicator,
    std::string_view taskId,
    std::string_view host,
    uint16_t port,
    const PartitionKey& partitionKey,
    const std::shared_ptr<UcxCpuRowExchangeQueue> queue)
    : CommElement(communicator),
      host_(host),
      port_(port),
      taskId_(taskId),
      partitionKey_(partitionKey),
      partitionKeyHash_(fnv1a_32(partitionKey_.toString())),
      queue_(std::move(queue)) {
  setState(ReceiverState::Created);
}

/* static */
std::shared_ptr<UcxCpuRowExchangeSource> UcxCpuRowExchangeSource::create(
    std::string_view taskId,
    std::string_view url,
    const std::shared_ptr<UcxCpuRowExchangeQueue>& queue) {
  folly::Uri uri(url);
  // The CPU UCX listener follows the cuDF exchange convention:
  // derive the UCX port from the Prestissimo HTTP task URL as HTTP+3.
  const std::string host = uri.host();
  const int httpPort = uri.port();
  VELOX_CHECK(
      !host.empty(),
      "CPU UCX remote split URL is missing host: {}",
      std::string(url));
  VELOX_CHECK_GT(
      httpPort,
      0,
      "CPU UCX remote split URL is missing HTTP port: {}",
      std::string(url));
  VELOX_CHECK_LE(
      httpPort,
      65532,
      "CPU UCX remote split HTTP port is too high to derive UCX port: {}",
      std::string(url));
  uint16_t port = static_cast<uint16_t>(httpPort + 3);
  std::shared_ptr<Communicator> communicator = Communicator::getInstance();
  auto key = extractTaskAndDestinationId(uri.path());
  auto source =
      std::shared_ptr<UcxCpuRowExchangeSource>(new UcxCpuRowExchangeSource(
          communicator, taskId, host, port, key, queue));
  VLOG(3) << source->toString()
          << " creating UcxCpuRowExchangeSource for url: " << url;
  return source;
}

void UcxCpuRowExchangeSource::start() {
  auto self = getSelfPtr();
  VELOX_CHECK_NOT_NULL(
      self, "UcxCpuRowExchangeSource must be owned by shared_ptr");
  communicator_->registerCommElement(std::move(self));
}

void UcxCpuRowExchangeSource::process() {
  drainStateEvents();
  if (closed_) {
    cleanUp();
    return;
  }

  switch (state_) {
    case ReceiverState::Created: {
      HostPort hp{host_, port_};
      std::shared_ptr<UcxCpuRowExchangeSource> selfPtr = getSelfPtr();
      auto epRef = communicator_->assocEndpointRef(selfPtr, hp);
      if (epRef) {
        setEndpoint(epRef);
        setStateIf(
            ReceiverState::Created, ReceiverState::WaitingForHandshakeComplete);
        sendHandshake();
        // The amSend completion callback (onHandshake) re-queues us when
        // the handshake send completes; no explicit addToWorkQueue
        // needed here. Saves an extra Communicator iteration that
        // would just no-op in WaitingForHandshakeComplete.
      } else {
        auto errorMsg = fmt::format(
            "Failed to connect CPU UCX exchange source to {}:{}, task {}. "
            "The remote split must point at a native worker exposing the CPU "
            "UCX listener; coordinator-routed or HTTP-only exchange edges "
            "cannot be consumed by UcxCpuRowExchange.",
            host_,
            port_,
            partitionKey_.toString());
        VLOG(0) << toString() << " " << errorMsg;
        queue_->setError(errorMsg);
        deliverEndMarker();
        setState(ReceiverState::Done);
        // Failure path has no callback; re-queue so process()
        // picks up the Done state and runs cleanUp().
        communicator_->addToWorkQueue(getSelfPtr());
      }
    } break;
    case ReceiverState::WaitingForHandshakeComplete:
      // Driven by the amSend completion callback (onHandshake).
      break;
    case ReceiverState::WaitingForHandshakeResponse:
      // Driven by the tagRecv completion callback (onHandshakeResponse).
      break;
    case ReceiverState::ReadyToReceive: {
      // Backpressure: skip posting the next metadata recv if the queue
      // is over the high-water mark. Source goes dormant; the consumer
      // will call resumeFromBackpressure() once the queue drains.
      int32_t queueSize = queue_->size();
      const int32_t highWater = backpressureHighWaterMark();
      if (queueSize > highWater) {
        backpressureActive_.store(true, std::memory_order_release);
        break;
      }

      postReceiveWindow();
    } break;
    case ReceiverState::WaitingForMetadata:
      // Legacy state from the single-flight receiver. The pipelined
      // receiver stays ReadyToReceive while sequence-specific callbacks
      // advance individual bundles.
      break;
    case ReceiverState::WaitingForData:
      // Legacy state from the single-flight receiver.
      break;
    case ReceiverState::Done:
      cleanUp();
      break;
  }
}

void UcxCpuRowExchangeSource::cleanUp() {
  if (getState() != ReceiverState::Done) {
    VLOG(3) << toString() << " UcxCpuRowExchangeSource::cleanUp in state "
            << toName(getState());
  }

  if (handshakeRequest_ && !handshakeRequest_->isCompleted()) {
    handshakeRequest_->cancel();
  }
  if (handshakeResponseRequest_ && !handshakeResponseRequest_->isCompleted()) {
    handshakeResponseRequest_->cancel();
  }
  for (auto& req : metadataRequests_) {
    if (req && !req->isCompleted()) {
      req->cancel();
    }
  }
  for (auto& req : dataRequests_) {
    if (req && !req->isCompleted()) {
      req->cancel();
    }
  }

  if (communicator_) {
    if (handshakeRequest_) {
      communicator_->deferRequestCleanup(std::move(handshakeRequest_));
    }
    if (handshakeResponseRequest_) {
      communicator_->deferRequestCleanup(std::move(handshakeResponseRequest_));
    }
    for (auto& req : metadataRequests_) {
      if (req) {
        communicator_->deferRequestCleanup(std::move(req));
      }
    }
    metadataRequests_.clear();
    for (auto& req : dataRequests_) {
      if (req) {
        communicator_->deferRequestCleanup(std::move(req));
      }
    }
    dataRequests_.clear();
  }

  if (endpointRef_) {
    endpointRef_->removeCommElem(getSelfPtr());
    endpointRef_ = nullptr;
  }
  if (communicator_) {
    communicator_->unregister(getSelfPtr());
  }
}

void UcxCpuRowExchangeSource::close() {
  std::lock_guard<std::recursive_mutex> processLock(processMutex_);
  bool expected = false;
  if (!closed_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    return;
  }
  VLOG(1) << toString() << " UcxCpuRowExchangeSource::close called.";
  deliverEndMarker();
  setState(ReceiverState::Done);
  communicator_->addToWorkQueue(getSelfPtr());
}

void UcxCpuRowExchangeSource::resumeFromBackpressure() {
  bool expected = true;
  if (backpressureActive_.compare_exchange_strong(
      expected, false, std::memory_order_acq_rel)) {
    communicator_->addToWorkQueue(getSelfPtr());
  }
}

folly::F14FastMap<std::string, int64_t> UcxCpuRowExchangeSource::stats() const {
  VELOX_UNREACHABLE();
}

folly::F14FastMap<std::string, RuntimeMetric> UcxCpuRowExchangeSource::metrics()
    const {
  folly::F14FastMap<std::string, RuntimeMetric> map;
  map["ucxCpuRowExchangeSource.numPayloads"] = metrics_.numPayloads_;
  map["ucxCpuRowExchangeSource.totalBytes"] = metrics_.totalBytes_;
  map["ucxCpuRowExchangeSource.rttPerRequest"] = metrics_.rttPerRequest_;
  return map;
}

// ---- private methods ----

PartitionKey UcxCpuRowExchangeSource::extractTaskAndDestinationId(
    std::string_view path) {
  // Path: /v1/task/<taskId>/results/<destinationId>
  std::vector<folly::StringPiece> components;
  folly::split('/', path, components, true);

  VELOX_CHECK_EQ(
      components.size(),
      5,
      "Malformed CPU UCX remote split path: {}",
      std::string(path));
  VELOX_CHECK_EQ(
      components[0], "v1", "Malformed task URL path: {}", std::string(path));
  VELOX_CHECK_EQ(
      components[1], "task", "Malformed task URL path: {}", std::string(path));
  VELOX_CHECK(
      !components[2].empty(),
      "Task URL has empty task ID: {}",
      std::string(path));
  VELOX_CHECK_EQ(
      components[3],
      "results",
      "Malformed task URL path: {}",
      std::string(path));

  const auto destinationText = components[4].str();
  VELOX_CHECK(
      !destinationText.empty() &&
          std::all_of(
              destinationText.begin(),
              destinationText.end(),
              [](unsigned char c) { return std::isdigit(c); }),
      "Illegal destination in task URL: {}",
      std::string(path));

  uint64_t destinationId;
  try {
    destinationId = std::stoull(destinationText);
  } catch (const std::exception& e) {
    VELOX_UNSUPPORTED("Illegal destination in task URL: {}", std::string(path));
  }
  VELOX_CHECK_LE(
      destinationId,
      std::numeric_limits<uint32_t>::max(),
      "Destination in task URL exceeds uint32: {}",
      std::string(path));

  return PartitionKey{
      components[2].str(), static_cast<uint32_t>(destinationId)};
}

std::shared_ptr<UcxCpuRowExchangeSource> UcxCpuRowExchangeSource::getSelfPtr() {
  std::shared_ptr<UcxCpuRowExchangeSource> ptr;
  try {
    ptr = shared_from_this();
  } catch (std::bad_weak_ptr&) {
    ptr = nullptr;
  }
  return ptr;
}

void UcxCpuRowExchangeSource::enqueue(UcxCpuRowReceivedPtr data) {
  std::vector<velox::ContinuePromise> queuePromises;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    queue_->enqueueLocked(std::move(data), queuePromises);
  }
  for (auto& promise : queuePromises) {
    promise.setValue();
  }
}

void UcxCpuRowExchangeSource::deliverEndMarker() {
  if (!registered_.load(std::memory_order_acquire)) {
    return;
  }
  bool expected = false;
  if (!endMarkerDelivered_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    return;
  }
  VLOG(3) << toString() << " delivering end-of-stream marker to queue";
  enqueue(nullptr);
}

void UcxCpuRowExchangeSource::setEndpoint(
    std::shared_ptr<EndpointRef> endpointRef) {
  endpointRef_ = std::move(endpointRef);
}

void UcxCpuRowExchangeSource::sendHandshake() {
  std::string workerAddress = communicator_->getWorkerAddress();
  VELOX_CHECK_LE(
      workerAddress.size(),
      std::numeric_limits<uint32_t>::max(),
      "UCX worker address is too large: {} bytes",
      workerAddress.size());
  auto handshakeReq = std::make_shared<std::vector<uint8_t>>(
      sizeof(CpuRowHandshakeHeader) + workerAddress.size());
  CpuRowHandshakeHeader header{};
  header.sourceWorkerAddressBytes = static_cast<uint32_t>(workerAddress.size());
  header.sourceHostIdHash = communicator_->getHostIdHash();

  auto* handshake = &header.handshake;
  handshake->destination = partitionKey_.destination;
  strncpy(
      handshake->taskId,
      partitionKey_.taskId.c_str(),
      sizeof(handshake->taskId) - 1);
  handshake->taskId[sizeof(handshake->taskId) - 1] = '\0';
  handshake->workerId = communicator_->getWorkerId();
  std::memcpy(handshakeReq->data(), &header, sizeof(header));
  if (!workerAddress.empty()) {
    std::memcpy(
        handshakeReq->data() + sizeof(CpuRowHandshakeHeader),
        workerAddress.data(),
        workerAddress.size());
  }

  VLOG(3) << toString() << " Sending handshake for " << partitionKey_.toString()
          << " (CPU AM id)";

  // Dispatch to the CPU-specific AM callback. This is the only piece
  // that diverges from the cudf source: same HandshakeMsg struct, but
  // a different callback id, so the cudf Acceptor never sees it.
  ucxx::AmReceiverCallbackInfo info(
      communicator_->kAmCallbackOwner, communicator_->kAmCpuCallbackId);
  std::weak_ptr<UcxCpuRowExchangeSource> weak = weak_from_this();
  handshakeRequest_ = endpointRef_->endpoint_->amSend(
      handshakeReq->data(),
      handshakeReq->size(),
      UCS_MEMORY_TYPE_HOST,
      info,
      false,
      [weak](ucs_status_t status, std::shared_ptr<void> arg) {
        if (auto self = weak.lock()) {
          self->enqueueStateEvent(
              self, [raw = self.get(), status, arg = std::move(arg)]() mutable {
                raw->onHandshake(status, std::move(arg));
              });
        }
      },
      handshakeReq);
}

void UcxCpuRowExchangeSource::onHandshake(
    ucs_status_t status,
    std::shared_ptr<void> /*arg*/) {
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString() << " onHandshake after close, ignoring";
    deliverEndMarker();
    return;
  }
  if (getState() != ReceiverState::WaitingForHandshakeComplete) {
    VLOG(2) << toString() << " onHandshake in state " << toName(getState())
            << ", ignoring (possible UCXX replay)";
    return;
  }
  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to send CPU handshake to host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
  } else {
    VLOG(3) << toString() << " + onHandshake " << ucs_status_string(status);
    setStateIf(
        ReceiverState::WaitingForHandshakeComplete,
        ReceiverState::WaitingForHandshakeResponse);
    receiveHandshakeResponse();
  }
}

void UcxCpuRowExchangeSource::receiveHandshakeResponse() {
  auto responseBuffer = std::make_shared<HandshakeResponse>();
  uint64_t responseTag = getHandshakeResponseTag(partitionKeyHash_);

  VLOG(3) << toString() << " waiting for CPU HandshakeResponse tag=" << std::hex
          << responseTag << std::dec;

  std::weak_ptr<UcxCpuRowExchangeSource> weak = weak_from_this();
  handshakeResponseRequest_ = endpointRef_->endpoint_->tagRecv(
      responseBuffer.get(),
      sizeof(*responseBuffer),
      ucxx::Tag{responseTag},
      ucxx::TagMaskFull,
      false,
      [weak](ucs_status_t status, std::shared_ptr<void> arg) {
        if (auto self = weak.lock()) {
          self->enqueueStateEvent(
              self, [raw = self.get(), status, arg = std::move(arg)]() mutable {
                raw->onHandshakeResponse(status, std::move(arg));
              });
        }
      },
      responseBuffer);
}

void UcxCpuRowExchangeSource::onHandshakeResponse(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString() << " onHandshakeResponse after close, ignoring";
    deliverEndMarker();
    return;
  }
  if (getState() != ReceiverState::WaitingForHandshakeResponse) {
    VLOG(2) << toString() << " onHandshakeResponse in state "
            << toName(getState()) << ", ignoring (possible UCXX replay)";
    return;
  }
  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to receive CPU HandshakeResponse from host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
    return;
  }

  auto response = std::static_pointer_cast<HandshakeResponse>(arg);
  VLOG(3) << toString() << " + CPU HandshakeResponse isIntraNodeTransfer="
          << response->isIntraNodeTransfer;

  setStateIf(
      ReceiverState::WaitingForHandshakeResponse,
      ReceiverState::ReadyToReceive);
}

void UcxCpuRowExchangeSource::postReceiveWindow() {
  while (!atEnd_ && !closed_.load(std::memory_order_acquire) &&
         inFlightSequences_.size() < maxInFlightBundles()) {
    int32_t queueSize = queue_->size();
    const int32_t highWater = backpressureHighWaterMark();
    if (queueSize > highWater) {
      backpressureActive_.store(true, std::memory_order_release);
      break;
    }

    auto sequenceNumber = nextMetadataSequence_++;
    inFlightSequences_.insert(sequenceNumber);
    getMetadata(sequenceNumber);
  }
}

void UcxCpuRowExchangeSource::getMetadata(uint32_t sequenceNumber) {
  auto metadataReq = std::make_shared<std::vector<uint8_t>>(kMaxMetaBufSize);
  uint64_t metadataTag = getMetadataTag(partitionKeyHash_, sequenceNumber);

  VLOG(3) << toString() << " waiting for metadata seq=" << sequenceNumber
          << " tag=" << std::hex << metadataTag << std::dec;

  std::weak_ptr<UcxCpuRowExchangeSource> weak = weak_from_this();
  auto request = endpointRef_->endpoint_->tagRecv(
      reinterpret_cast<void*>(metadataReq->data()),
      kMaxMetaBufSize,
      ucxx::Tag{metadataTag},
      ucxx::TagMaskFull,
      false,
      [weak, sequenceNumber](ucs_status_t status, std::shared_ptr<void> arg) {
        if (auto self = weak.lock()) {
          self->enqueueStateEvent(
              self,
              [raw = self.get(),
               sequenceNumber,
               status,
               arg = std::move(arg)]() mutable {
                raw->onMetadata(sequenceNumber, status, std::move(arg));
              });
        }
      },
      metadataReq);
  metadataRequests_.push_back(std::move(request));
}

void UcxCpuRowExchangeSource::onMetadata(
    uint32_t sequenceNumber,
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString() << " onMetadata after close, ignoring";
    deliverEndMarker();
    return;
  }
  if (getState() == ReceiverState::Done) {
    VLOG(2) << toString() << " onMetadata seq=" << sequenceNumber
            << " after Done, ignoring (possible UCXX replay)";
    return;
  }
  if (!inFlightSequences_.count(sequenceNumber)) {
    VLOG(2) << toString() << " onMetadata seq=" << sequenceNumber
            << " is not in flight, ignoring (possible UCXX replay)";
    return;
  }
  VLOG(3) << toString() << " + onMetadata seq=" << sequenceNumber << " "
          << ucs_status_string(status);

  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to receive CPU metadata from host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
    return;
  }

  VELOX_CHECK_NOT_NULL(arg, "Didn't get metadata");
  auto metadataMsg = std::static_pointer_cast<std::vector<uint8_t>>(arg);

  auto ptr = std::make_shared<DataAndMetadata>();
  ptr->sequenceNumber = sequenceNumber;
  ptr->metadata = UcxCpuRowMetadataMsg::deserialize(metadataMsg->data());

  VLOG(3) << toString() << " seq=" << sequenceNumber
          << " Datasize bytes == " << ptr->metadata.dataSizeBytes
          << " numFrames=" << ptr->metadata.frameSizes.size();

  if (ptr->metadata.atEnd) {
    atEnd_ = true;
    endSequence_ = sequenceNumber;
    inFlightSequences_.erase(sequenceNumber);
    VLOG(3) << "There is no more data to transfer for " << toString()
            << " endSeq=" << endSequence_;
    drainCompletedData();
    return;
  }

  const size_t numFrames = ptr->metadata.frameSizes.size();
  VELOX_CHECK_GT(
      numFrames, 0, "Non-end metadata must carry at least one frame");

  int64_t totalFrameBytes = 0;
  for (auto frameSize : ptr->metadata.frameSizes) {
    VELOX_CHECK_GT(frameSize, 0, "Frame size must be positive");
    VELOX_CHECK_LE(
        frameSize,
        ptr->metadata.dataSizeBytes - totalFrameBytes,
        "CPU row metadata frame bytes overflow total data size");
    totalFrameBytes += frameSize;
  }
  VELOX_CHECK_EQ(
      totalFrameBytes,
      ptr->metadata.dataSizeBytes,
      "CPU row metadata frame bytes do not match total data size");

  // Allocate one buffer per frame. The receiver stitches them into an
  // IOBuf chain on the last completion; no coalesce, no extra
  // memcpy. PrestoSerializer's stream walks the chain natively.
  ptr->frameBufs.reserve(numFrames);
  for (size_t i = 0; i < numFrames; ++i) {
    ptr->frameBufs.emplace_back(
        new uint8_t[ptr->metadata.frameSizes[i]],
        [](uint8_t* p) { delete[] p; });
  }
  ptr->pendingFrames.store(
      static_cast<int32_t>(numFrames), std::memory_order_release);
  ptr->finalStatus.store(UCS_OK, std::memory_order_release);

  uint64_t dataTag = getDataTag(partitionKeyHash_, sequenceNumber);
  VLOG(3) << toString() << " waiting for data seq=" << sequenceNumber
          << " tag=" << std::hex << dataTag << std::dec
          << " numFrames=" << numFrames
          << " totalBytes=" << ptr->metadata.dataSizeBytes;

  std::weak_ptr<UcxCpuRowExchangeSource> weak = weak_from_this();
  // Post N tagRecvs all sharing the same dataTag. UCX guarantees FIFO
  // order on the (sender, receiver, tag) triple, so frames match the
  // sender's tagSend order and our frameBufs[i] receives the right
  // bytes. Per-frame callback decrements pendingFrames; the one that
  // hits zero invokes onData with UCS_OK (or the first error seen).
  for (size_t i = 0; i < numFrames; ++i) {
    auto req = endpointRef_->endpoint_->tagRecv(
        ptr->frameBufs[i].get(),
        ptr->metadata.frameSizes[i],
        ucxx::Tag{dataTag},
        ucxx::TagMaskFull,
        false,
        [weak](ucs_status_t status, std::shared_ptr<void> arg) {
          auto state = std::static_pointer_cast<DataAndMetadata>(arg);
          if (status != UCS_OK) {
            ucs_status_t expected = UCS_OK;
            state->finalStatus.compare_exchange_strong(
                expected, status, std::memory_order_acq_rel);
          }
          const int32_t before =
              state->pendingFrames.fetch_sub(1, std::memory_order_acq_rel);
          if (before != 1) {
            return;
          }
          // Last frame: fire onData on whatever final status we ended
          // up with.
          const ucs_status_t finalStatus =
              state->finalStatus.load(std::memory_order_acquire);
          if (auto self = weak.lock()) {
            self->enqueueStateEvent(
                self,
                [raw = self.get(),
                 sequenceNumber = state->sequenceNumber,
                 finalStatus,
                 arg = std::move(arg)]() mutable {
                  raw->onData(sequenceNumber, finalStatus, std::move(arg));
                });
          }
        },
        ptr);
    dataRequests_.push_back(std::move(req));
  }
}

void UcxCpuRowExchangeSource::onData(
    uint32_t sequenceNumber,
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString() << " onData after close, ignoring";
    deliverEndMarker();
    return;
  }
  if (getState() == ReceiverState::Done) {
    VLOG(2) << toString() << " onData seq=" << sequenceNumber
            << " after Done, ignoring (possible UCXX replay)";
    return;
  }
  if (!inFlightSequences_.count(sequenceNumber)) {
    VLOG(2) << toString() << " onData seq=" << sequenceNumber
            << " is not in flight, ignoring (possible UCXX replay)";
    return;
  }
  VLOG(3) << toString() << " + onData seq=" << sequenceNumber << " "
          << ucs_status_string(status);

  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to receive CPU data from host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << toString() << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
  } else {
    VLOG(3) << toString() << "+ onData " << ucs_status_string(status)
            << " got chunk: " << sequenceNumber;
    auto ptr = std::static_pointer_cast<DataAndMetadata>(arg);
    VELOX_CHECK_EQ(ptr->sequenceNumber, sequenceNumber);

    metrics_.numPayloads_.addValue(1);
    metrics_.totalBytes_.addValue(ptr->metadata.dataSizeBytes);

    // Stitch all per-frame buffers into one IOBuf chain. Each segment
    // wraps its frame buffer with a takeOwnership freeFn that releases
    // the shared_ptr<uint8_t> when the IOBuf is destroyed. No coalesce,
    // no inter-frame memcpy. PrestoSerializer's ByteInputStream walks
    // the chain natively across self-describing vector boundaries.
    std::unique_ptr<folly::IOBuf> chain;
    for (size_t i = 0; i < ptr->frameBufs.size(); ++i) {
      auto bufHolder = std::make_unique<std::shared_ptr<uint8_t>>(
          std::move(ptr->frameBufs[i]));
      auto* rawData = bufHolder->get();
      auto seg = folly::IOBuf::takeOwnership(
          rawData,
          ptr->metadata.frameSizes[i],
          [](void* /*buf*/, void* userData) {
            delete static_cast<std::shared_ptr<uint8_t>*>(userData);
          },
          bufHolder.release());
      if (!chain) {
        chain = std::move(seg);
      } else {
        chain->appendToChain(std::move(seg));
      }
    }
    ptr->frameBufs.clear();

    auto received = std::make_unique<UcxCpuRowPayload>();
    received->data = std::move(chain);
    received->numBytes = ptr->metadata.dataSizeBytes;
    received->numRows = 0; // unknown until deserialization

    inFlightSequences_.erase(sequenceNumber);
    completedData_.emplace(sequenceNumber, std::move(received));
    drainCompletedData();
    postReceiveWindow();
  }
}

void UcxCpuRowExchangeSource::drainCompletedData() {
  while (true) {
    if (atEnd_ && nextDeliverySequence_ == endSequence_) {
      deliverEndMarker();
      setState(ReceiverState::Done);
      return;
    }

    auto it = completedData_.find(nextDeliverySequence_);
    if (it == completedData_.end()) {
      return;
    }

    enqueue(std::move(it->second));
    completedData_.erase(it);
    ++nextDeliverySequence_;
  }
}

bool UcxCpuRowExchangeSource::setStateIf(
    UcxCpuRowExchangeSource::ReceiverState expected,
    UcxCpuRowExchangeSource::ReceiverState desired) {
  ReceiverState exp = expected;
  while (!state_.compare_exchange_strong(
      exp, desired, std::memory_order_acq_rel, std::memory_order_relaxed)) {
    if (exp != expected) {
      return false;
    }
    exp = expected;
  }
  VLOG(4) << "[ExSrc-CPU " << toString()
          << " nextMeta=" << nextMetadataSequence_
          << " nextDeliver=" << nextDeliverySequence_ << "] "
          << toName(expected) << " -> " << toName(desired);
  return true;
}

} // namespace facebook::velox::ucx_exchange
