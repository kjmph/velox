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
#include "velox/experimental/ucx-exchange/UcxCpuRowExchangeServer.h"
#include <glog/logging.h>
#include <algorithm>
#include <cstring>
#include "velox/common/EnumDefine.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowMetadataMsg.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"

namespace facebook::velox::ucx_exchange {

namespace {
const folly::F14FastMap<UcxCpuRowExchangeServer::ServerState, std::string_view>&
serverStateNames() {
  static const folly::
      F14FastMap<UcxCpuRowExchangeServer::ServerState, std::string_view>
          kNames = {
              {UcxCpuRowExchangeServer::ServerState::Created, "Created"},
              {UcxCpuRowExchangeServer::ServerState::WaitingForDataEndpointAck,
               "WaitingForDataEndpointAck"},
              {UcxCpuRowExchangeServer::ServerState::ReadyToTransfer,
               "ReadyToTransfer"},
              {UcxCpuRowExchangeServer::ServerState::WaitingForDataFromQueue,
               "WaitingForDataFromQueue"},
              {UcxCpuRowExchangeServer::ServerState::DataReady, "DataReady"},
              {UcxCpuRowExchangeServer::ServerState::WaitingForSendComplete,
               "WaitingForSendComplete"},
              {UcxCpuRowExchangeServer::ServerState::
                   WaitingForFinalMetadataComplete,
               "WaitingForFinalMetadataComplete"},
              {UcxCpuRowExchangeServer::ServerState::Done, "Done"},
          };
  return kNames;
}

// Keep CPU row exchange single-flight to match the receive-side state machine.
constexpr uint32_t kMaxInFlightBundles = 1;

} // namespace

VELOX_DEFINE_EMBEDDED_ENUM_NAME(
    UcxCpuRowExchangeServer,
    ServerState,
    serverStateNames)

// Decouples ucxx::Request lifetime from buffer lifetime. The Request must
// survive UCP wireup-replay; the buffer should be released as soon as DMA
// completes. The Request's callbackData is a shared_ptr<context>; the
// context holds the buffer; the completion callback moves the buffer out
// of the context, freeing it while the (now-empty) context shell stays
// pinned to the Request.
struct CpuRowMetaSendContext {
  std::shared_ptr<uint8_t> metadata;
};

// One on-the-wire frame. Large chunks ship standalone when already
// contiguous enough for UCX to read directly; smaller chunks are packed
// into heap buffers to keep frame count and per-frame UCX overhead bounded.
struct CpuRowSendFrame {
  void* ptr{nullptr};
  size_t len{0};
  std::shared_ptr<UcxCpuRowPayload> standaloneChunk;
  std::shared_ptr<uint8_t> packedBuf;
};

// Shared state across the N per-frame tagSend callbacks of one bundle.
// `pendingFrames` counts down as each frame completes; the callback
// that brings it to zero invokes the unified sendComplete. `frames`
// keeps every byte UCX is reading from alive across the whole bundle's
// DMA.
struct CpuRowMultiSendState {
  std::vector<CpuRowSendFrame> frames;
  std::atomic<int32_t> pendingFrames{0};
  std::atomic<ucs_status_t> finalStatus{UCS_OK};
};

// Target packed-frame size on the wire. Small chunks are accumulated into a
// coalesced heap buffer of roughly this size. Larger chunks ship standalone
// to avoid copying bytes the producer already serialized contiguously.
constexpr int64_t kFrameTargetBytes = 1L << 20; // 1 MiB
constexpr int64_t kStandaloneFrameMinBytes = 512L << 10; // 512 KiB

// Bundle target. Multiple small queue chunks are coalesced up to this
// many bytes per UCX send. We drain hard; modern boxes with hundreds
// of GB of RAM can comfortably absorb tens of MB in flight per
// destination, and the per-send overhead (tag-matching, callback
// dispatch, single-threaded Communicator progress) dominates over the
// memory cost. Tail sends (queue contents below this target) ship as
// whatever is queued.
constexpr int64_t kBundleTargetBytes = 64L << 20; // 64 MiB

void UcxCpuRowExchangeServer::setState(ServerState newState) {
  state_.store(newState, std::memory_order_seq_cst);
}

UcxCpuRowExchangeServer::UcxCpuRowExchangeServer(
    const std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> endpointRef,
    const PartitionKey& key,
    bool waitForDataEndpointAck)
    : CommElement(communicator, endpointRef, true),
      partitionKey_(key),
      partitionKeyHash_(fnv1a_32(partitionKey_.toString())),
      waitForDataEndpointAck_(waitForDataEndpointAck),
      queueMgr_(UcxCpuRowOutputQueueManager::getInstanceRef()) {
  setState(ServerState::Created);
}

// static
std::shared_ptr<UcxCpuRowExchangeServer> UcxCpuRowExchangeServer::create(
    const std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> endpointRef,
    const PartitionKey& key,
    bool waitForDataEndpointAck) {
  return std::shared_ptr<UcxCpuRowExchangeServer>(
      new UcxCpuRowExchangeServer(
          communicator, endpointRef, key, waitForDataEndpointAck));
}

void UcxCpuRowExchangeServer::process() {
  while (true) {
    drainStateEvents();
    if (closed_.load(std::memory_order_acquire)) {
      return;
    }

    switch (state_) {
      case ServerState::Created:
        if (waitForDataEndpointAck_ && !dataEndpointAckReceived_) {
          setState(ServerState::WaitingForDataEndpointAck);
          return;
        }
        setState(ServerState::ReadyToTransfer);
        continue;
      case ServerState::WaitingForDataEndpointAck:
        return;
      case ServerState::ReadyToTransfer:
      case ServerState::WaitingForDataFromQueue:
      case ServerState::DataReady:
      case ServerState::WaitingForSendComplete:
        fillSendWindow();
        if (getState() == ServerState::Done) {
          continue;
        }
        return;
      case ServerState::WaitingForFinalMetadataComplete:
        maybeFinish();
        if (getState() == ServerState::Done) {
          continue;
        }
        return;
      case ServerState::Done:
        close();
        if (endpointRef_) {
          endpointRef_->removeCommElem(getSelfPtr());
          endpointRef_ = nullptr;
        }
        return;
    };
  }
}

void UcxCpuRowExchangeServer::close() {
  std::lock_guard<std::recursive_mutex> processLock(processMutex_);
  bool expected = false;
  if (!closed_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    return;
  }
  for (auto& req : metaRequests_) {
    if (req && !req->isCompleted()) {
      req->cancel();
    }
  }
  for (auto& req : dataRequests_) {
    if (req && !req->isCompleted()) {
      req->cancel();
    }
  }
  if (dataEndpointAckRequest_ && !dataEndpointAckRequest_->isCompleted()) {
    dataEndpointAckRequest_->cancel();
  }
  pendingData_.reset();
  hasPendingData_ = false;

  if (communicator_) {
    for (auto& req : metaRequests_) {
      if (req) {
        communicator_->deferRequestCleanup(std::move(req));
      }
    }
    metaRequests_.clear();
    for (auto& req : dataRequests_) {
      if (req) {
        communicator_->deferRequestCleanup(std::move(req));
      }
    }
    dataRequests_.clear();
    if (dataEndpointAckRequest_) {
      communicator_->deferRequestCleanup(std::move(dataEndpointAckRequest_));
    }
  }

  communicator_->unregister(getSelfPtr());
}

std::string UcxCpuRowExchangeServer::toString() {
  std::stringstream out;
  out << "[ExSrv-CPU " << partitionKey_.toString() << " - " << sequenceNumber_
      << "]";
  return out.str();
}

// ------ private methods ---------

std::shared_ptr<UcxCpuRowExchangeServer> UcxCpuRowExchangeServer::getSelfPtr() {
  return shared_from_this();
}

void UcxCpuRowExchangeServer::postDataEndpointAckReceive() {
  if (!waitForDataEndpointAck_ || dataEndpointAckRequest_) {
    return;
  }

  auto ack = std::make_shared<CpuRowHandshakeAckHeader>();
  uint64_t ackTag = getHandshakeAckTag(partitionKeyHash_);

  std::weak_ptr<UcxCpuRowExchangeServer> weak = weak_from_this();
  dataEndpointAckRequest_ = endpointRef_->endpoint_->tagRecv(
      ack.get(),
      sizeof(*ack),
      ucxx::Tag{ackTag},
      ucxx::TagMaskFull,
      false,
      [weak](ucs_status_t status, std::shared_ptr<void> arg) {
        if (auto self = weak.lock()) {
          self->enqueueStateEvent(
              self, [raw = self.get(), status, arg = std::move(arg)]() mutable {
                raw->onDataEndpointAck(status, std::move(arg));
              });
        }
      },
      ack);
}

void UcxCpuRowExchangeServer::onDataEndpointAck(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  if (closed_.load(std::memory_order_acquire)) {
    return;
  }

  if (status != UCS_OK) {
    LOG(ERROR) << "[ExSrv-CPU " << partitionKey_.toString()
               << "] failed to receive data endpoint ACK: "
               << ucs_status_string(status);
    setState(ServerState::Done);
    return;
  }

  auto ack = std::static_pointer_cast<CpuRowHandshakeAckHeader>(arg);
  if (ack->magic != kCpuRowHandshakeAckMagic ||
      ack->version != kCpuRowHandshakeAckVersion ||
      ack->headerSize != sizeof(CpuRowHandshakeAckHeader)) {
    LOG(ERROR) << "[ExSrv-CPU " << partitionKey_.toString()
               << "] invalid CPU data endpoint ACK";
    setState(ServerState::Done);
    return;
  }

  dataEndpointAckReceived_ = true;
  setState(ServerState::ReadyToTransfer);
}

void UcxCpuRowExchangeServer::fillSendWindow() {
  if (closed_.load(std::memory_order_acquire) || finalMetadataSent_) {
    return;
  }

  while (inFlightSends_ < kMaxInFlightBundles) {
    if (hasPendingData_) {
      auto data = std::move(pendingData_);
      hasPendingData_ = false;
      if (!data) {
        sendFinalMetadata();
        return;
      }
      sendData(std::move(data));
      continue;
    }

    auto data =
        queueMgr_->tryGetData(partitionKey_.taskId, partitionKey_.destination);
    if (data) {
      sendData(std::move(data));
      continue;
    }

    if (!waitingForData_) {
      requestData();
    }
    return;
  }
}

void UcxCpuRowExchangeServer::requestData() {
  waitingForData_ = true;
  setState(ServerState::WaitingForDataFromQueue);

  std::weak_ptr<UcxCpuRowExchangeServer> weakSelf = weak_from_this();
  queueMgr_->getData(
      partitionKey_.taskId,
      partitionKey_.destination,
      [weakSelf](
          std::shared_ptr<UcxCpuRowPayload> data,
          std::vector<int64_t> /*remainingBytes*/) {
        auto self = weakSelf.lock();
        if (!self) {
          return;
        }
        self->enqueueStateEvent(
            self, [raw = self.get(), data = std::move(data)]() mutable {
              raw->onDataAvailable(std::move(data));
            });
      });
}

void UcxCpuRowExchangeServer::onDataAvailable(
    std::shared_ptr<UcxCpuRowPayload> data) {
  waitingForData_ = false;
  if (closed_.load(std::memory_order_acquire)) {
    return;
  }
  if (finalMetadataSent_) {
    return;
  }

  VELOX_CHECK(
      !hasPendingData_, "CPU row exchange queue callback overlap detected");
  pendingData_ = std::move(data);
  hasPendingData_ = true;
  setState(ServerState::DataReady);
}

void UcxCpuRowExchangeServer::sendData(
    std::shared_ptr<UcxCpuRowPayload> firstChunk) {
  // Drain the queue up to kBundleTargetBytes. The collected chunks are
  // then packed into ~kFrameTargetBytes frames before the metadata is
  // built; the receiver allocates one buffer per frame and we must
  // advertise the post-pack sizes, not the per-chunk sizes.
  std::vector<std::shared_ptr<UcxCpuRowPayload>> chunks;
  int64_t combinedBytes = 0;
  if (firstChunk) {
    combinedBytes = firstChunk->numBytes;
    chunks.push_back(std::move(firstChunk));
    while (combinedBytes < kBundleTargetBytes) {
      auto extra = queueMgr_->tryGetData(
          partitionKey_.taskId, partitionKey_.destination);
      if (!extra) {
        break;
      }
      combinedBytes += extra->numBytes;
      chunks.push_back(std::move(extra));
    }
  }
  VELOX_CHECK_GT(chunks.size(), 0, "sendData requires a non-empty data bundle");
  const uint32_t sequenceNumber = sequenceNumber_++;

  // Pack chunks into ~kFrameTargetBytes frames:
  //   - A chunk >= kStandaloneFrameMinBytes ships standalone
  //     (single-segment IOBuf coalesce is a no-op on the send side).
  //   - Smaller chunks are accumulated into a packed heap buffer and
  //     flushed when the running total hits the target.
  // Goal: keep UCX request overhead bounded without copying pages that are
  // already large enough to stand alone.
  auto multiState = std::make_shared<CpuRowMultiSendState>();

  if (!chunks.empty()) {
    std::vector<std::shared_ptr<UcxCpuRowPayload>> pendingChunks;
    int64_t pendingBytes = 0;

    auto flushPending = [&pendingChunks, &pendingBytes, &multiState]() {
      if (pendingChunks.empty()) {
        return;
      }
      std::shared_ptr<uint8_t> buf(
          new uint8_t[pendingBytes], [](uint8_t* p) { delete[] p; });
      size_t off = 0;
      for (auto& c : pendingChunks) {
        for (auto range : *c->data) {
          std::memcpy(buf.get() + off, range.data(), range.size());
          off += range.size();
        }
      }
      CpuRowSendFrame f;
      f.ptr = buf.get();
      f.len = static_cast<size_t>(pendingBytes);
      f.packedBuf = std::move(buf);
      multiState->frames.push_back(std::move(f));
      pendingChunks.clear();
      pendingBytes = 0;
    };

    for (auto& chunk : chunks) {
      const int64_t chunkBytes = chunk->numBytes;
      if (chunkBytes >= kStandaloneFrameMinBytes) {
        flushPending();
        auto byteRange = chunk->data->coalesce();
        VELOX_CHECK_EQ(
            static_cast<int64_t>(byteRange.size()),
            chunkBytes,
            "Coalesced chunk size mismatch");
        CpuRowSendFrame f;
        f.ptr = const_cast<uint8_t*>(byteRange.data());
        f.len = byteRange.size();
        f.standaloneChunk = std::move(chunk);
        multiState->frames.push_back(std::move(f));
      } else {
        pendingChunks.push_back(std::move(chunk));
        pendingBytes += chunkBytes;
        if (pendingBytes >= kFrameTargetBytes) {
          flushPending();
        }
      }
    }
    flushPending();
    chunks.clear();
  }
  const size_t numFrames = multiState->frames.size();
  VELOX_CHECK_GT(numFrames, 0, "sendData requires a non-empty data bundle");

  // Build metadata reflecting the packed frame layout.
  UcxCpuRowMetadataMsg metadataMsg;
  metadataMsg.dataSizeBytes = combinedBytes;
  metadataMsg.frameSizes.reserve(numFrames);
  for (const auto& f : multiState->frames) {
    metadataMsg.frameSizes.push_back(static_cast<WireDataSizeType>(f.len));
  }
  metadataMsg.remainingBytes = {};
  metadataMsg.atEnd = false;

  auto [serializedMetadata, serMetaSize] = metadataMsg.serialize();

  // Send metadata.
  uint64_t metadataTag = getMetadataTag(partitionKeyHash_, sequenceNumber);
  std::weak_ptr<UcxCpuRowExchangeServer> weakMeta = weak_from_this();

  auto metaCtx = std::make_shared<CpuRowMetaSendContext>();
  metaCtx->metadata = serializedMetadata;

  auto metaRequest = endpointRef_->endpoint_->tagSend(
      metaCtx->metadata.get(),
      serMetaSize,
      ucxx::Tag{metadataTag},
      false,
      [tid = partitionKey_.toString(), weakMeta](
          ucs_status_t status, std::shared_ptr<void> arg) {
        auto ctx = std::static_pointer_cast<CpuRowMetaSendContext>(arg);
        auto metaHolder = std::move(ctx->metadata);

        auto self = weakMeta.lock();
        if (!self) {
          return;
        }
        self->enqueueStateEvent(
            self, [raw = self.get(), status, tid]() {
              if (raw->closed_.load(std::memory_order_acquire)) {
                return;
              }
              if (status != UCS_OK) {
                LOG(ERROR) << "@" << raw->partitionKey_.taskId
                           << " Error in sendData metadata: "
                           << ucs_status_string(status) << " task: " << tid;
                raw->setState(ServerState::Done);
              }
            });
      },
      metaCtx);
  metaRequests_.push_back(std::move(metaRequest));

  setState(ServerState::WaitingForSendComplete);
  uint64_t dataTagBase = getDataTag(partitionKeyHash_, sequenceNumber);

  multiState->pendingFrames.store(
      static_cast<int32_t>(numFrames), std::memory_order_release);
  ++inFlightSends_;

  std::weak_ptr<UcxCpuRowExchangeServer> weakData = weak_from_this();
  for (size_t frameIdx = 0; frameIdx < numFrames; ++frameIdx) {
    auto& frame = multiState->frames[frameIdx];
    // Frames within one bundle share the same tag. UCX guarantees
    // FIFO order on a (sender, receiver, tag) triple, so the
    // receiver's N tagRecvs match in send order.
    auto req = endpointRef_->endpoint_->tagSend(
        frame.ptr,
        frame.len,
        ucxx::Tag{dataTagBase},
        false,
        [weakData](ucs_status_t status, std::shared_ptr<void> arg) {
          auto state = std::static_pointer_cast<CpuRowMultiSendState>(arg);
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
          const ucs_status_t finalStatus =
              state->finalStatus.load(std::memory_order_acquire);
          state->frames.clear();
          if (auto self = weakData.lock()) {
            self->enqueueStateEvent(
                self,
                [raw = self.get(),
                 finalStatus,
                 arg = std::move(arg)]() mutable {
                  raw->sendComplete(finalStatus, std::move(arg));
                });
          }
        },
        multiState);
    dataRequests_.push_back(std::move(req));
  }
}

void UcxCpuRowExchangeServer::sendFinalMetadata() {
  if (finalMetadataSent_) {
    return;
  }

  finalMetadataSent_ = true;
  const uint32_t sequenceNumber = sequenceNumber_++;

  UcxCpuRowMetadataMsg metadataMsg;
  metadataMsg.dataSizeBytes = 0;
  metadataMsg.remainingBytes = {};
  metadataMsg.atEnd = true;

  auto [serializedMetadata, serMetaSize] = metadataMsg.serialize();
  uint64_t metadataTag = getMetadataTag(partitionKeyHash_, sequenceNumber);

  auto metaCtx = std::make_shared<CpuRowMetaSendContext>();
  metaCtx->metadata = serializedMetadata;
  std::weak_ptr<UcxCpuRowExchangeServer> weakMeta = weak_from_this();

  setState(ServerState::WaitingForFinalMetadataComplete);
  auto metaRequest = endpointRef_->endpoint_->tagSend(
      metaCtx->metadata.get(),
      serMetaSize,
      ucxx::Tag{metadataTag},
      false,
      [weakMeta](ucs_status_t status, std::shared_ptr<void> arg) {
        auto ctx = std::static_pointer_cast<CpuRowMetaSendContext>(arg);
        auto metaHolder = std::move(ctx->metadata);
        if (auto self = weakMeta.lock()) {
          self->enqueueStateEvent(self, [raw = self.get(), status]() {
            raw->finalMetadataComplete(status);
          });
        }
      },
      metaCtx);
  metaRequests_.push_back(std::move(metaRequest));

  // The end marker has been handed to UCX and the request/context above keep
  // the metadata bytes alive until completion. Match the GPU UCX exchange
  // lifetime rule: producer output is consumed once the final marker is
  // published, not when the send completion callback eventually runs.
  queueMgr_->deleteResults(partitionKey_.taskId, partitionKey_.destination);
}

void UcxCpuRowExchangeServer::finalMetadataComplete(ucs_status_t status) {
  if (closed_.load(std::memory_order_acquire)) {
    return;
  }

  if (status == UCS_OK) {
    finalMetadataCompleted_ = true;
    maybeFinish();
  } else {
    LOG(ERROR) << "@" << partitionKey_.taskId
               << " Error in final metadata send: " << ucs_status_string(status)
               << " task: " << partitionKey_.toString();
    setState(ServerState::Done);
  }
}

void UcxCpuRowExchangeServer::maybeFinish() {
  if (finalMetadataCompleted_ && inFlightSends_ == 0 &&
      getState() != ServerState::Done) {
    setState(ServerState::Done);
  }
}

void UcxCpuRowExchangeServer::sendComplete(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  if (closed_.load(std::memory_order_acquire)) {
    return;
  }
  auto state = std::static_pointer_cast<CpuRowMultiSendState>(arg);
  VELOX_CHECK_GT(inFlightSends_, 0, "sendComplete without in-flight send");
  --inFlightSends_;

  if (status == UCS_OK) {
    // Completed UCXX requests are retained for the server lifetime to protect
    // callback closures from UCP wireup replay, but their captured send state
    // must not keep payload buffers alive.
    state->frames.clear();

    if (!finalMetadataSent_) {
      setState(ServerState::ReadyToTransfer);
    } else {
      maybeFinish();
    }
  } else {
    LOG(ERROR) << "@" << partitionKey_.taskId
               << " Error in sendComplete: " << ucs_status_string(status);
    setState(ServerState::Done);
  }
}

} // namespace facebook::velox::ucx_exchange
