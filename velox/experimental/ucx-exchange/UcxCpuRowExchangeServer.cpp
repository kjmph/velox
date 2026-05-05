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
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowMetadataMsg.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowShm.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"

namespace facebook::velox::ucx_exchange {

namespace {
const folly::F14FastMap<UcxCpuRowExchangeServer::ServerState, std::string_view>&
serverStateNames() {
  static const folly::
      F14FastMap<UcxCpuRowExchangeServer::ServerState, std::string_view>
          kNames = {
              {UcxCpuRowExchangeServer::ServerState::Created, "Created"},
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

int64_t elapsedMillis(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now()) {
  if (start == std::chrono::steady_clock::time_point{}) {
    return -1;
  }
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}
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

// One on-the-wire frame. Either we own a chunk's IOBuf directly
// (large chunks ship standalone; coalesce() on a single-segment
// IOBuf is a no-op so this is true zero-copy on the send side), or we
// own a packed heap buffer that aggregated several small chunks (one
// memcpy per packed byte but bounded; keeps frame count and
// per-frame UCX overhead in check).
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
  ~CpuRowMultiSendState() {
    if (slotPool &&
        !receiverSlotRefsCommitted.load(std::memory_order_acquire)) {
      for (auto slotId : slotIds) {
        slotPool->release(slotId);
      }
    }
  }

  std::vector<CpuRowSendFrame> frames;
  std::vector<std::shared_ptr<UcxCpuRowPayload>> shmChunks;
  std::shared_ptr<UcxCpuRowShmSegment> shmSegment;
  std::shared_ptr<UcxCpuRowShmSlotPool> slotPool;
  std::vector<uint32_t> slotIds;
  std::atomic<bool> receiverSlotRefsCommitted{false};
  std::atomic<int32_t> pendingFrames{0};
  std::atomic<ucs_status_t> finalStatus{UCS_OK};
  uint32_t sequenceNumber{0};
  int64_t bytes{0};
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
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
  auto oldState = state_.exchange(newState, std::memory_order_seq_cst);
  VLOG(4) << "[ExSrv-CPU " << partitionKey_.toString()
          << " seq=" << sequenceNumber_ << "] " << toName(oldState) << " -> "
          << toName(newState);
}

UcxCpuRowExchangeServer::UcxCpuRowExchangeServer(
    const std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> endpointRef,
    const PartitionKey& key,
    bool canUseCpuShm)
    : CommElement(communicator, endpointRef),
      partitionKey_(key),
      partitionKeyHash_(fnv1a_32(partitionKey_.toString())),
      canUseCpuShm_(canUseCpuShm),
      queueMgr_(UcxCpuRowOutputQueueManager::getInstanceRef()) {
  setState(ServerState::Created);
}

// static
std::shared_ptr<UcxCpuRowExchangeServer> UcxCpuRowExchangeServer::create(
    const std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> endpointRef,
    const PartitionKey& key,
    bool canUseCpuShm) {
  return std::shared_ptr<UcxCpuRowExchangeServer>(new UcxCpuRowExchangeServer(
      communicator, endpointRef, key, canUseCpuShm));
}

void UcxCpuRowExchangeServer::process() {
  drainStateEvents();
  if (closed_.load(std::memory_order_acquire)) {
    return;
  }
  switch (state_) {
    case ServerState::Created:
      setState(ServerState::ReadyToTransfer);
      communicator_->addToWorkQueue(getSelfPtr());
      break;
    case ServerState::ReadyToTransfer: {
      fillSendWindow();
    } break;
    case ServerState::WaitingForDataFromQueue:
      fillSendWindow();
      break;
    case ServerState::DataReady:
      fillSendWindow();
      break;
    case ServerState::WaitingForSendComplete:
      fillSendWindow();
      break;
    case ServerState::WaitingForFinalMetadataComplete:
      maybeFinish();
      break;
    case ServerState::Done:
      close();
      if (endpointRef_) {
        endpointRef_->removeCommElem(getSelfPtr());
        endpointRef_ = nullptr;
      }
      break;
  };
}

void UcxCpuRowExchangeServer::close() {
  std::lock_guard<std::recursive_mutex> processLock(processMutex_);
  bool expected = false;
  if (!closed_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    return;
  }
  VLOG(3) << "@" << partitionKey_.taskId
          << " Close UcxCpuRowExchangeServer to remote "
          << partitionKey_.toString();

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

void UcxCpuRowExchangeServer::fillSendWindow() {
  if (closed_.load(std::memory_order_acquire) || finalMetadataSent_) {
    return;
  }

  const uint32_t sendWindow = maxInFlightBundles();
  while (inFlightSends_ < sendWindow) {
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
    VLOG(3) << "@" << partitionKey_.taskId
            << " getData callback after close, ignoring";
    return;
  }
  if (finalMetadataSent_) {
    VLOG(3) << "@" << partitionKey_.taskId
            << " getData callback after final metadata, ignoring";
    return;
  }

  if (!data) {
    sendFinalMetadata();
    return;
  }

  VLOG(3) << "@" << partitionKey_.taskId
          << " Found data for client: " << partitionKey_.toString();
  sendData(std::move(data));
  fillSendWindow();
}

void UcxCpuRowExchangeServer::sendData(
    std::shared_ptr<UcxCpuRowPayload> firstChunk) {
  // Drain the queue up to kBundleTargetBytes. The collected chunks are
  // then packed into ~kFrameTargetBytes frames before the metadata is
  // built; the receiver allocates one buffer per frame and we must
  // advertise the post-pack sizes, not the per-chunk sizes.
  enum class BundleTransport {
    kProducerSlot,
    kDirectShm,
    kHeap,
  };

  std::vector<std::shared_ptr<UcxCpuRowPayload>> chunks;
  int64_t combinedBytes = 0;
  int64_t combinedRows = 0;
  BundleTransport firstTransport = BundleTransport::kHeap;
  if (firstChunk) {
    auto bundleTransport = [](const std::shared_ptr<UcxCpuRowPayload>& chunk) {
      if (chunk->shmSlotPool &&
          chunk->shmSlotId != std::numeric_limits<uint32_t>::max()) {
        return BundleTransport::kProducerSlot;
      }
      if (chunk->shmSegment) {
        return BundleTransport::kDirectShm;
      }
      return BundleTransport::kHeap;
    };

    firstTransport = bundleTransport(firstChunk);
    const auto* firstSlotPool =
        firstChunk->shmSlotPool ? firstChunk->shmSlotPool.get() : nullptr;
    auto canBundleWithFirst =
        [&](const std::shared_ptr<UcxCpuRowPayload>& extra) {
          if (bundleTransport(extra) != firstTransport) {
            return false;
          }
          if (firstTransport == BundleTransport::kProducerSlot) {
            return extra->shmSlotPool.get() == firstSlotPool;
          }
          return true;
        };

    combinedBytes = firstChunk->numBytes;
    combinedRows = firstChunk->numRows;
    chunks.push_back(std::move(firstChunk));
    while (combinedBytes < kBundleTargetBytes) {
      auto extra = queueMgr_->tryGetData(
          partitionKey_.taskId, partitionKey_.destination);
      if (!extra) {
        break;
      }
      if (!canBundleWithFirst(extra)) {
        queueMgr_->requeueFront(
            partitionKey_.taskId, partitionKey_.destination, std::move(extra));
        break;
      }
      combinedBytes += extra->numBytes;
      combinedRows += extra->numRows;
      chunks.push_back(std::move(extra));
    }
  }
  const int64_t bundledChunks = static_cast<int64_t>(chunks.size());
  VELOX_CHECK_GT(bundledChunks, 0, "sendData requires a non-empty data bundle");
  const uint32_t sequenceNumber = sequenceNumber_++;
  const auto now = std::chrono::steady_clock::now();
  if (firstPayloadTime_ == std::chrono::steady_clock::time_point{}) {
    firstPayloadTime_ = now;
  }
  lastPayloadTime_ = now;
  ++bundlesStarted_;
  payloadBytesStarted_ += combinedBytes;
  chunksStarted_ += bundledChunks;
  switch (firstTransport) {
    case BundleTransport::kProducerSlot:
      ++producerSlotBundlesStarted_;
      break;
    case BundleTransport::kDirectShm:
      ++directShmBundlesStarted_;
      break;
    case BundleTransport::kHeap:
      ++heapBundlesStarted_;
      break;
  }

  // Pack chunks into ~kFrameTargetBytes frames:
  //   - A chunk >= kStandaloneFrameMinBytes ships standalone
  //     (single-segment IOBuf coalesce is a no-op on the send side).
  //   - Smaller chunks are accumulated into a packed heap buffer and
  //     flushed when the running total hits the target.
  // Goal: keep UCX request overhead bounded without copying pages that are
  // already large enough to stand alone.
  auto multiState = std::make_shared<CpuRowMultiSendState>();

  auto producerSlotPool = [&]() -> std::shared_ptr<UcxCpuRowShmSlotPool> {
    if (!canUseCpuShm_ || chunks.empty()) {
      return nullptr;
    }
    auto pool = chunks.front() ? chunks.front()->shmSlotPool : nullptr;
    if (!pool) {
      return nullptr;
    }
    for (const auto& chunk : chunks) {
      if (!chunk || chunk->numBytes <= 0 || !chunk->shmSlotPool ||
          chunk->shmSlotPool.get() != pool.get() ||
          chunk->shmSlotId == std::numeric_limits<uint32_t>::max()) {
        return nullptr;
      }
    }
    return pool;
  }();

  // Producer-side slot-pool path. The producing driver serialized directly
  // into reusable SHM slots owned by the task output queue; this server only
  // publishes metadata. This is the slot path that avoids both per-message
  // shm_open/fstat/mmap and the extra heap-to-slot copy.
  if (producerSlotPool) {
    multiState->shmChunks = std::move(chunks);
    const size_t numFrames = multiState->shmChunks.size();
    VELOX_CHECK_GT(numFrames, 0, "sendData requires a non-empty data bundle");

    multiState->slotPool = producerSlotPool;
    multiState->slotIds.reserve(numFrames);

    UcxCpuRowMetadataMsg metadataMsg;
    metadataMsg.dataSizeBytes = combinedBytes;
    metadataMsg.frameSizes.reserve(numFrames);
    metadataMsg.remainingBytes = {};
    metadataMsg.atEnd = false;
    metadataMsg.transport = UcxCpuRowMetadataMsg::Transport::kShmSlot;
    metadataMsg.shmPoolName = producerSlotPool->name();
    metadataMsg.shmPoolSize =
        static_cast<WireDataSizeType>(producerSlotPool->totalSize());
    metadataMsg.shmSlotSize =
        static_cast<WireDataSizeType>(producerSlotPool->slotSize());
    metadataMsg.shmSlotIds.reserve(numFrames);

    for (const auto& chunk : multiState->shmChunks) {
      const auto frameBytes = static_cast<size_t>(chunk->numBytes);
      VELOX_CHECK_LE(
          frameBytes,
          producerSlotPool->slotSize(),
          "CPU SHM producer slot frame exceeds slot size");
      VELOX_CHECK(
          producerSlotPool->isReady(chunk->shmSlotId),
          "CPU SHM producer slot is not ready");
      VELOX_CHECK(
          producerSlotPool->addRef(chunk->shmSlotId),
          "CPU SHM producer slot was freed before metadata publish");
      multiState->slotIds.push_back(chunk->shmSlotId);
      metadataMsg.frameSizes.push_back(
          static_cast<WireDataSizeType>(frameBytes));
      metadataMsg.shmSlotIds.push_back(
          static_cast<WireLengthType>(chunk->shmSlotId));
    }

    multiState->sequenceNumber = sequenceNumber;
    multiState->bytes = combinedBytes;
    multiState->startTime = std::chrono::high_resolution_clock::now();

    auto [serializedMetadata, serMetaSize] = metadataMsg.serialize();
    uint64_t metadataTag = getMetadataTag(partitionKeyHash_, sequenceNumber);

    VLOG(3) << "@" << partitionKey_.taskId << " seq=" << sequenceNumber
            << " publishing CPU SHM producer slot-pool bundle pool="
            << producerSlotPool->name() << " frames=" << numFrames
            << " bytes=" << combinedBytes << " bundledChunks=" << bundledChunks;

    auto metaCtx = std::make_shared<CpuRowMetaSendContext>();
    metaCtx->metadata = serializedMetadata;
    std::weak_ptr<UcxCpuRowExchangeServer> weakMeta = weak_from_this();

    setState(ServerState::WaitingForSendComplete);
    ++inFlightSends_;

    auto metaRequest = endpointRef_->endpoint_->tagSend(
        metaCtx->metadata.get(),
        serMetaSize,
        ucxx::Tag{metadataTag},
        false,
        [tid = partitionKey_.toString(), metadataTag, weakMeta, multiState](
            ucs_status_t status, std::shared_ptr<void> arg) {
          auto ctx = std::static_pointer_cast<CpuRowMetaSendContext>(arg);
          auto metaHolder = std::move(ctx->metadata);

          if (status == UCS_OK) {
            // Metadata completion hands the advertised slot references to the
            // receiver. Producer-side payload references remain live until the
            // local queue/server releases its shared_ptrs.
            //
            // After metadata is successfully submitted to UCX, receivers own
            // unlinking the POSIX SHM name. The shared pool header contains
            // the expected opener count, and the last receiver to open the
            // pool unlinks the name. The local mapping stays valid through the
            // shared_ptrs above.
            if (multiState->slotPool) {
              multiState->slotPool->disableUnlinkOnDestroy();
            }
            multiState->receiverSlotRefsCommitted.store(
                true, std::memory_order_release);
          }

          auto self = weakMeta.lock();
          if (!self) {
            return;
          }
          self->enqueueStateEvent(
              self,
              [raw = self.get(),
               status,
               tid,
               metadataTag,
               multiState]() mutable {
                if (raw->closed_.load(std::memory_order_acquire)) {
                  VLOG(3) << "@" << raw->partitionKey_.taskId
                          << " producer slot-pool metadata send callback "
                             "after close, ignoring";
                  return;
                }
                if (status == UCS_OK) {
                  VLOG(3) << "@" << raw->partitionKey_.taskId
                          << " producer slot-pool metadata sent to " << tid
                          << " tag=" << std::hex << metadataTag;
                } else {
                  VLOG(0) << "@" << raw->partitionKey_.taskId
                          << " Error in producer slot-pool metadata send: "
                          << ucs_status_string(status) << " task: " << tid;
                }
                raw->sendComplete(status, multiState);
              });
        },
        metaCtx);
    metaRequests_.push_back(std::move(metaRequest));
    return;
  }

  // Producer-side SHM chunks are already materialized in named shared memory.
  // Publish those names directly and skip the communicator-thread copy into a
  // new bundle.
  const bool publishProducerShm =
      canUseCpuShm_ && !chunks.empty() &&
      std::all_of(chunks.begin(), chunks.end(), [](const auto& chunk) {
        return chunk && chunk->shmSegment && chunk->numBytes > 0;
      });
  if (publishProducerShm) {
    multiState->shmChunks = std::move(chunks);
    const size_t numFrames = multiState->shmChunks.size();
    VELOX_CHECK_GT(numFrames, 0, "sendData requires a non-empty data bundle");

    multiState->sequenceNumber = sequenceNumber;
    multiState->bytes = combinedBytes;
    multiState->startTime = std::chrono::high_resolution_clock::now();

    UcxCpuRowMetadataMsg metadataMsg;
    metadataMsg.dataSizeBytes = combinedBytes;
    metadataMsg.frameSizes.reserve(numFrames);
    metadataMsg.shmNames.reserve(numFrames);
    metadataMsg.remainingBytes = {};
    metadataMsg.atEnd = false;
    metadataMsg.transport = UcxCpuRowMetadataMsg::Transport::kShm;

    for (const auto& chunk : multiState->shmChunks) {
      const auto frameBytes = static_cast<size_t>(chunk->numBytes);
      VELOX_CHECK_LE(
          chunk->shmOffset,
          chunk->shmSegment->size,
          "CPU SHM direct-TX chunk offset exceeds backing segment");
      VELOX_CHECK_LE(
          frameBytes,
          chunk->shmSegment->size - chunk->shmOffset,
          "CPU SHM direct-TX chunk exceeds backing segment");
      metadataMsg.frameSizes.push_back(
          static_cast<WireDataSizeType>(frameBytes));
      metadataMsg.shmNames.push_back(chunk->shmSegment->name);
      VELOX_CHECK_EQ(
          chunk->shmOffset,
          0,
          "CPU SHM direct-TX standalone chunk must start at offset 0");
    }
    metadataMsg.shmName = metadataMsg.shmNames.front();

    auto [serializedMetadata, serMetaSize] = metadataMsg.serialize();
    uint64_t metadataTag = getMetadataTag(partitionKeyHash_, sequenceNumber);

    VLOG(3) << "@" << partitionKey_.taskId << " seq=" << sequenceNumber
            << " publishing CPU SHM direct-TX bundle frames=" << numFrames
            << " bytes=" << combinedBytes << " bundledChunks=" << bundledChunks;

    auto metaCtx = std::make_shared<CpuRowMetaSendContext>();
    metaCtx->metadata = serializedMetadata;
    std::weak_ptr<UcxCpuRowExchangeServer> weakMeta = weak_from_this();

    setState(ServerState::WaitingForSendComplete);
    ++inFlightSends_;

    auto metaRequest = endpointRef_->endpoint_->tagSend(
        metaCtx->metadata.get(),
        serMetaSize,
        ucxx::Tag{metadataTag},
        false,
        [tid = partitionKey_.toString(), metadataTag, weakMeta, multiState](
            ucs_status_t status, std::shared_ptr<void> arg) {
          auto ctx = std::static_pointer_cast<CpuRowMetaSendContext>(arg);
          auto metaHolder = std::move(ctx->metadata);

          if (status == UCS_OK) {
            for (const auto& chunk : multiState->shmChunks) {
              if (chunk->shmSegment) {
                chunk->shmSegment->unlinkOnDestroy = false;
              }
            }
          }

          auto self = weakMeta.lock();
          if (!self) {
            return;
          }
          self->enqueueStateEvent(
              self,
              [raw = self.get(),
               status,
               tid,
               metadataTag,
               multiState]() mutable {
                if (raw->closed_.load(std::memory_order_acquire)) {
                  VLOG(3) << "@" << raw->partitionKey_.taskId
                          << " direct-TX SHM metadata send callback after "
                             "close, ignoring";
                  return;
                }
                if (status == UCS_OK) {
                  VLOG(3) << "@" << raw->partitionKey_.taskId
                          << " direct-TX SHM metadata sent to " << tid
                          << " tag=" << std::hex << metadataTag;
                } else {
                  VLOG(0) << "@" << raw->partitionKey_.taskId
                          << " Error in direct-TX SHM metadata send: "
                          << ucs_status_string(status) << " task: " << tid;
                }
                raw->sendComplete(status, multiState);
              });
        },
        metaCtx);
    metaRequests_.push_back(std::move(metaRequest));
    return;
  }

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
  multiState->sequenceNumber = sequenceNumber;
  multiState->bytes = combinedBytes;
  multiState->startTime = std::chrono::high_resolution_clock::now();

  VLOG(4) << "[ExSrv-CPU " << partitionKey_.toString()
          << " seq=" << sequenceNumber
          << "] sendData hasData=" << (numFrames > 0)
          << " bundledChunks=" << bundledChunks << " frames=" << numFrames
          << " size=" << combinedBytes << " rows=" << combinedRows;

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
      [tid = partitionKey_.toString(), metadataTag, weakMeta](
          ucs_status_t status, std::shared_ptr<void> arg) {
        auto ctx = std::static_pointer_cast<CpuRowMetaSendContext>(arg);
        auto metaHolder = std::move(ctx->metadata);

        auto self = weakMeta.lock();
        if (!self) {
          return;
        }
        self->enqueueStateEvent(
            self, [raw = self.get(), status, tid, metadataTag]() {
              if (raw->closed_.load(std::memory_order_acquire)) {
                VLOG(3) << "@" << raw->partitionKey_.taskId
                        << " metadata send callback after close, ignoring";
                return;
              }
              if (status == UCS_OK) {
                VLOG(3) << "@" << raw->partitionKey_.taskId
                        << " metadata sent to " << tid << " tag=" << std::hex
                        << metadataTag;
              } else {
                VLOG(0) << "@" << raw->partitionKey_.taskId
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
  finalMetadataSendTime_ = std::chrono::steady_clock::now();
  const uint32_t sequenceNumber = sequenceNumber_++;
  VLOG(1) << "[TAIL-CPU] task=" << partitionKey_.taskId
          << " destination=" << partitionKey_.destination
          << " event=final-marker-send seq=" << sequenceNumber
          << " inFlight=" << inFlightSends_
          << " bundlesStarted=" << bundlesStarted_
          << " bundlesCompleted=" << bundlesCompleted_
          << " bytesStarted=" << payloadBytesStarted_
          << " bytesCompleted=" << payloadBytesCompleted_
          << " chunksStarted=" << chunksStarted_
          << " producerSlotBundles=" << producerSlotBundlesStarted_
          << " directShmBundles=" << directShmBundlesStarted_
          << " heapBundles=" << heapBundlesStarted_ << " firstPayloadToFinalMs="
          << elapsedMillis(firstPayloadTime_, finalMetadataSendTime_)
          << " lastPayloadToFinalMs="
          << elapsedMillis(lastPayloadTime_, finalMetadataSendTime_);

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
  const auto deleteStart = std::chrono::steady_clock::now();
  queueMgr_->deleteResults(partitionKey_.taskId, partitionKey_.destination);
  const auto deleteMicros =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - deleteStart)
          .count();
  VLOG(1) << "[TAIL-CPU] task=" << partitionKey_.taskId
          << " destination=" << partitionKey_.destination
          << " event=delete-results durationUs=" << deleteMicros;
}

void UcxCpuRowExchangeServer::finalMetadataComplete(ucs_status_t status) {
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << "@" << partitionKey_.taskId
            << " final metadata callback after close, ignoring";
    return;
  }

  if (status == UCS_OK) {
    VLOG(1) << "[TAIL-CPU] task=" << partitionKey_.taskId
            << " destination=" << partitionKey_.destination
            << " event=final-marker-complete status=OK"
            << " finalMetadataMs=" << elapsedMillis(finalMetadataSendTime_)
            << " inFlight=" << inFlightSends_;
    finalMetadataCompleted_ = true;
    maybeFinish();
  } else {
    VLOG(0) << "@" << partitionKey_.taskId
            << " Error in final metadata send: " << ucs_status_string(status)
            << " task: " << partitionKey_.toString();
    setState(ServerState::Done);
  }
}

void UcxCpuRowExchangeServer::maybeFinish() {
  if (finalMetadataCompleted_ && inFlightSends_ == 0 &&
      getState() != ServerState::Done) {
    const auto now = std::chrono::steady_clock::now();
    VLOG(1) << "[TAIL-CPU] task=" << partitionKey_.taskId
            << " destination=" << partitionKey_.destination
            << " event=server-done"
            << " firstPayloadToDoneMs=" << elapsedMillis(firstPayloadTime_, now)
            << " finalMarkerToDoneMs="
            << elapsedMillis(finalMetadataSendTime_, now)
            << " lastSendCompleteToDoneMs="
            << elapsedMillis(lastSendCompleteTime_, now)
            << " bundlesStarted=" << bundlesStarted_
            << " bundlesCompleted=" << bundlesCompleted_
            << " bytesStarted=" << payloadBytesStarted_
            << " bytesCompleted=" << payloadBytesCompleted_;
    setState(ServerState::Done);
    communicator_->addToWorkQueue(getSelfPtr());
  }
}

void UcxCpuRowExchangeServer::sendComplete(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << "@" << partitionKey_.taskId
            << " sendComplete after close, ignoring";
    return;
  }
  auto state = std::static_pointer_cast<CpuRowMultiSendState>(arg);
  VELOX_CHECK_GT(inFlightSends_, 0, "sendComplete without in-flight send");
  --inFlightSends_;

  if (status == UCS_OK) {
    auto end = std::chrono::high_resolution_clock::now();
    lastSendCompleteTime_ = std::chrono::steady_clock::now();
    ++bundlesCompleted_;
    payloadBytesCompleted_ += state->bytes;
    auto duration = end - state->startTime;
    auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    auto throughput = (micros > 0) ? (state->bytes / micros) : 0;

    VLOG(3) << "@" << partitionKey_.taskId << " seq=" << state->sequenceNumber
            << " duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                   .count()
            << " ms";
    VLOG(3) << "@" << partitionKey_.taskId << " seq=" << state->sequenceNumber
            << " throughput: " << throughput
            << " MByte/s inFlight=" << inFlightSends_;

    // Completed UCXX requests are retained for the server lifetime to protect
    // callback closures from UCP wireup replay, but their captured send state
    // must not keep payload buffers alive. Slot-pool payload destruction is
    // what releases producer references and makes slots reusable; retaining
    // shmChunks here turns a slot pool into a single-use pool for the entire
    // task.
    state->shmChunks.clear();
    state->frames.clear();
    state->shmSegment.reset();

    if (!finalMetadataSent_) {
      setState(ServerState::ReadyToTransfer);
      fillSendWindow();
    } else {
      maybeFinish();
    }
  } else {
    VLOG(3) << "@" << partitionKey_.taskId
            << " Error in sendComplete: " << ucs_status_string(status);
    setState(ServerState::Done);
  }
}

} // namespace facebook::velox::ucx_exchange
