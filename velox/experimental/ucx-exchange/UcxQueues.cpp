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
#include "velox/experimental/ucx-exchange/UcxQueues.h"

#include <algorithm>
#include <limits>
#include <sstream>

#include <glog/logging.h>

namespace facebook::velox::ucx_exchange {

namespace {
int64_t addSaturated(int64_t left, int64_t right) {
  VELOX_CHECK_GE(left, 0);
  VELOX_CHECK_GE(right, 0);
  if (right > std::numeric_limits<int64_t>::max() - left) {
    return std::numeric_limits<int64_t>::max();
  }
  return left + right;
}

int64_t multiplySaturated(int64_t value, int64_t multiplier) {
  VELOX_CHECK_GE(value, 0);
  VELOX_CHECK_GE(multiplier, 0);
  if (value != 0 && multiplier > std::numeric_limits<int64_t>::max() / value) {
    return std::numeric_limits<int64_t>::max();
  }
  return value * multiplier;
}
} // namespace

void UcxDestinationQueue::Stats::recordEnqueue(const UcxGpuPayload* data) {
  if (data != nullptr) {
    bytesQueued += data->payloadBytes();
    packedColumnsQueued++;
  }
}

void UcxDestinationQueue::Stats::recordDequeue(const UcxGpuPayload* data) {
  if (data != nullptr) {
    const int64_t size = data->payloadBytes();

    bytesQueued -= size;
    VELOX_DCHECK_GE(bytesQueued, 0, "bytesQueued must be non-negative");
    --packedColumnsQueued;
    VELOX_DCHECK_GE(
        packedColumnsQueued, 0, "packedColumnsQueued must be non-negative");

    bytesSent += size;
    packedColumnsSent++;
    bytesInFlight += size;
    packedColumnsInFlight++;
  }
}

void UcxDestinationQueue::enqueueBack(std::shared_ptr<UcxGpuPayload> data) {
  // drop duplicate end markers.
  if (data == nullptr && !queue_.empty() && queue_.back() == nullptr) {
    return;
  }

  if (data != nullptr) {
    stats_.recordEnqueue(data.get());
  }
  queue_.push_back(std::move(data));
}

void UcxDestinationQueue::enqueueFront(std::shared_ptr<UcxGpuPayload> data) {
  // ignore nullptr.
  if (data == nullptr) {
    return;
  }

  // insert at the front.
  queue_.push_front(std::move(data));
}

UcxDestinationQueue::Data UcxDestinationQueue::getData(
    UcxDataAvailableCallback notify) {
  if (queue_.empty()) {
    // delay notification.
    notify_ = std::move(notify);
    return {};
  }

  // queue is not empty.
  auto data = std::move(queue_.front());
  queue_.pop_front();
  stats_.recordDequeue(data.get());

  std::vector<int64_t> remainingBytes;
  remainingBytes.reserve(queue_.size());
  // fill in the remainingbytes vector.
  for (std::size_t i = 0; i < queue_.size(); ++i) {
    if (queue_[i] == nullptr) {
      VELOX_CHECK_EQ(i, queue_.size() - 1, "null marker found in the middle");
      break;
    }
    remainingBytes.push_back(queue_[i]->data->gpu_data->size());
  }
  return {std::move(data), std::move(remainingBytes), true};
}

void UcxDestinationQueue::deleteResults() {
  for (auto i = 0; i < queue_.size(); ++i) {
    if (queue_[i] == nullptr) {
      VELOX_CHECK_EQ(i, queue_.size() - 1, "null marker found in the middle");
      break;
    }
  }
  queue_.clear();
}

UcxDataAvailable UcxDestinationQueue::getAndClearNotify() {
  if (notify_ == nullptr) {
    return UcxDataAvailable();
  }
  UcxDataAvailable result;
  result.callback = notify_;
  auto data = getData(nullptr);
  result.data = std::move(data.data);
  result.remainingBytes = std::move(data.remainingBytes);
  clearNotify();
  return result;
}

void UcxDestinationQueue::clearNotify() {
  notify_ = nullptr;
}

void UcxDestinationQueue::finish() {
  VELOX_CHECK_NULL(notify_, "notify must be cleared before finish");
  VELOX_CHECK(queue_.empty(), "data must be fetched before finish");
}

UcxDestinationQueue::Stats UcxDestinationQueue::stats() const {
  return stats_;
}

int64_t UcxDestinationQueue::transferBytes() const {
  return stats_.bytesQueued + stats_.bytesInFlight;
}

bool UcxDestinationQueue::waitingForData() const {
  return notify_ != nullptr;
}

void UcxDestinationQueue::acquireInFlight(
    int64_t bytes,
    int64_t numPackedCols) {
  stats_.bytesInFlight += bytes;
  stats_.packedColumnsInFlight += numPackedCols;
}

void UcxDestinationQueue::releaseInFlight(
    int64_t bytes,
    int64_t numPackedCols) {
  stats_.bytesInFlight -= bytes;
  stats_.packedColumnsInFlight -= numPackedCols;

  VELOX_CHECK_GE(stats_.bytesInFlight, 0);
  VELOX_CHECK_GE(stats_.packedColumnsInFlight, 0);
}

std::string UcxDestinationQueue::toString() {
  std::stringstream out;
  out << "[available: " << queue_.size() << ", "
      << (notify_ ? "notify registered, " : "") << this << "]";
  return out.str();
}

// ---------- UcxOutputQueue ----------

UcxOutputQueue::UcxOutputQueue(
    std::shared_ptr<exec::Task> task,
    uint32_t numDestinations,
    uint32_t numDrivers,
    core::PartitionedOutputNode::Kind kind)
    : task_(task), kind_(kind), numDrivers_(numDrivers) {
  if (task_) {
    maxSize_ = task_->queryCtx()->queryConfig().maxOutputBufferSize();
    continueSize_ = (maxSize_ * kContinuePct) / 100;
  } // else: maxSize_ and continueSize_ will be set once the task is created and
    // initialize called.
  if (numDestinations > 0) {
    addOutputBuffersLocked(numDestinations);
  }
  if (task_) {
    initialized_.store(true, std::memory_order_release);
  }
}

bool UcxOutputQueue::initialize(
    std::shared_ptr<exec::Task> task,
    uint32_t numDestinations,
    uint32_t numDrivers,
    core::PartitionedOutputNode::Kind kind) {
  std::lock_guard<std::mutex> l(mutex_);
  if (task_) {
    // already initialized!
    return false;
  }
  kind_ = kind;
  numDrivers_ = numDrivers;
  task_ = task;
  maxSize_ = task_->queryCtx()->queryConfig().maxOutputBufferSize();
  continueSize_ = (maxSize_ * kContinuePct) / 100;
  // Publish task metadata before destination queue expansion. Acceptor only
  // needs task/kind to choose the intra-node path; getData() takes mutex_ and
  // waits for any queue expansion in this function to finish.
  initialized_.store(true, std::memory_order_release);
  if (numDestinations > queues_.size()) {
    addOutputBuffersLocked(numDestinations);
  }
  VELOX_CHECK_EQ(transferReservedBytes_.size(), queues_.size());
  VELOX_CHECK_EQ(transferPromises_.size(), queues_.size());
  VELOX_CHECK_EQ(transferWindowBytes_.size(), queues_.size());
  return true;
}

void UcxOutputQueue::addOutputBuffersLocked(int numBuffers) {
  using Kind = core::PartitionedOutputNode::Kind;

  VELOX_CHECK_GE(numBuffers, 0);
  VELOX_CHECK_GT(numBuffers, queues_.size());
  VELOX_CHECK(!noMoreQueues_, "Cannot add destinations after noMoreBuffers");
  VELOX_CHECK_EQ(transferReservedBytes_.size(), queues_.size());
  VELOX_CHECK_EQ(transferPromises_.size(), queues_.size());
  VELOX_CHECK_EQ(transferWindowBytes_.size(), queues_.size());

  queues_.reserve(numBuffers);
  transferReservedBytes_.reserve(numBuffers);
  transferPromises_.reserve(numBuffers);
  transferWindowBytes_.reserve(numBuffers);
  if (kind_ == Kind::kBroadcast && !dataToBroadcast_.empty()) {
    updateTotalQueuedBytesMsLocked();
  }
  for (int32_t i = queues_.size(); i < numBuffers; ++i) {
    auto buffer = std::make_unique<UcxDestinationQueue>();
    if (kind_ == Kind::kBroadcast) {
      for (const auto& data : dataToBroadcast_) {
        buffer->enqueueBack(data);
        // Each backfilled payload will be dequeued and accounted independently
        // for this destination.
        queuedBytes_ += data->payloadBytes();
        queuedPackedColumns_++;
      }
    }
    // Preserve FIFO ordering: retained broadcast payloads precede EOS.
    if (atEnd_) {
      buffer->enqueueBack(nullptr);
    }
    queues_.emplace_back(std::move(buffer));
    transferReservedBytes_.push_back(0);
    transferPromises_.emplace_back();
    transferWindowBytes_.push_back(0);
  }
}

void UcxOutputQueue::updateNumDrivers(uint32_t newNumDrivers) {
  bool isNoMoreDrivers{false};
  {
    std::lock_guard<std::mutex> l(mutex_);
    numDrivers_ = newNumDrivers;
    // If we finished all drivers, ensure we register that we are 'done'.
    if (numDrivers_ == numFinished_) {
      isNoMoreDrivers = true;
    }
  }
  if (isNoMoreDrivers) {
    noMoreDrivers();
  }
}

void UcxOutputQueue::enqueue(
    int destination,
    std::unique_ptr<cudf::packed_columns> data,
    int32_t numRows,
    int64_t transferReservationBytes) {
  VELOX_CHECK_NOT_NULL(data);
  VELOX_CHECK_NOT_NULL(task_);
  VELOX_CHECK_GE(numRows, 0);
  VELOX_CHECK_GE(transferReservationBytes, 0);
  if (!task_->isRunning()) {
    std::vector<ContinuePromise> transferPromises;
    if (transferReservationBytes > 0) {
      std::lock_guard<std::mutex> l(mutex_);
      if (destination >= 0 &&
          static_cast<size_t>(destination) < transferReservedBytes_.size()) {
        releaseTransferReservationLocked(destination, transferReservationBytes);
        collectTransferPromisesLocked(destination, transferPromises);
      }
    }
    for (auto& promise : transferPromises) {
      promise.setValue();
    }
    return;
  }
  std::vector<UcxDataAvailable> dataAvailableCallbacks;
  {
    std::lock_guard<std::mutex> l(mutex_);
    auto numBytes = data->gpu_data->size();
    auto sharedData = std::make_shared<UcxGpuPayload>(UcxGpuPayload{
        .data = std::shared_ptr<cudf::packed_columns>(std::move(data)),
        .numRows = numRows});

    bool success = false;
    if (kind_ == core::PartitionedOutputNode::Kind::kBroadcast) {
      VELOX_CHECK_EQ(
          transferReservationBytes,
          0,
          "Broadcast output does not use transfer reservations");
      VELOX_CHECK_EQ(destination, 0, "Broadcast uses destination 0");
      enqueueBroadcastOutputLocked(
          std::move(sharedData), dataAvailableCallbacks);
      // For broadcast, count queuedBytes_ once per active destination so
      // that each destination's dequeue symmetrically decrements it. The
      // total sent stats count the logical data once.
      int numActive = 0;
      for (auto& q : queues_) {
        if (q != nullptr) {
          numActive++;
        }
      }
      updateTotalQueuedBytesMsLocked();
      queuedBytes_ += numBytes * numActive;
      queuedPackedColumns_ += numActive;
      totalBytesSent_ += numBytes;
      totalRowsSent_ += numRows;
      totalPackedColumnsSent_++;
      success = true;
    } else if (kind_ == core::PartitionedOutputNode::Kind::kArbitrary) {
      VELOX_CHECK_EQ(
          transferReservationBytes,
          0,
          "Arbitrary output does not use transfer reservations");
      VELOX_CHECK_EQ(destination, 0, "Arbitrary uses destination 0");
      enqueueArbitraryOutputLocked(
          std::move(sharedData), dataAvailableCallbacks);
      updateStatsWithEnqueuedLocked(numBytes, numRows);
      success = true;
    } else {
      VELOX_CHECK_LT(destination, queues_.size());
      success = enqueuePartitionedOutputLocked(
          destination,
          std::move(sharedData),
          dataAvailableCallbacks,
          transferReservationBytes);
      if (success) {
        updateStatsWithEnqueuedLocked(numBytes, numRows);
      }
    }
  }
  // Now that data is enqueued, notify blocked readers (outside of mutex.)
  for (auto& callback : dataAvailableCallbacks) {
    callback.notify();
  }
}

bool UcxOutputQueue::checkBlocked(ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  const auto producerBlockedBytes = producerBlockedBytesLocked();
  if (producerBlockedBytes >= maxSize_ && future) {
    promises_.emplace_back("UcxOutputQueue::checkBlocked");
    *future = promises_.back().getSemiFuture();
    return true;
  }
  return false;
}

bool UcxOutputQueue::checkTransferCapacity(
    int destination,
    int64_t maxBytes,
    ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK_GT(maxBytes, 0);

  VELOX_CHECK_GE(destination, 0);
  VELOX_CHECK_LT(destination, queues_.size());
  auto* queue = queues_[destination].get();
  if (queue == nullptr) {
    return false;
  }

  const auto destinationStats = queue->stats();
  const auto reservedBytes = transferReservedBytes_[destination];
  if (queue->waitingForData() && destinationStats.bytesQueued == 0 &&
      reservedBytes == 0) {
    return false;
  }

  const auto transferBytes = queue->transferBytes() + reservedBytes;
  if (transferBytes >= maxBytes) {
    if (future) {
      transferPromises_[destination].emplace_back(
          "UcxOutputQueue::checkTransferCapacity");
      *future = transferPromises_[destination].back().getSemiFuture();
    }
    return true;
  }
  return false;
}

bool UcxOutputQueue::reserveTransferBytes(
    int destination,
    int64_t bytes,
    int64_t maxBytes,
    ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK_GT(bytes, 0);
  VELOX_CHECK_GT(maxBytes, 0);
  VELOX_CHECK_GE(destination, 0);
  VELOX_CHECK_LT(destination, queues_.size());
  auto* queue = queues_[destination].get();
  if (queue == nullptr) {
    return false;
  }

  if (fullTransferCongested_) {
    const auto retainedBytes = retainedBytesWithTransferReservationsLocked();
    if (retainedBytes > 0) {
      maybeGrowFullTransferRetainedLimitLocked(retainedBytes);
      const auto retainedLimit = fullTransferRetainedLimitLocked();
      if (retainedLimit > 0 &&
          addSaturated(retainedBytes, bytes) > retainedLimit) {
        if (future) {
          promises_.emplace_back("UcxOutputQueue::reserveTransferBytes");
          *future = promises_.back().getSemiFuture();
        }
        return true;
      }
    }
  }

  const auto destinationStats = queue->stats();
  if (queue->waitingForData() && destinationStats.bytesQueued == 0 &&
      transferReservedBytes_[destination] == 0) {
    VELOX_CHECK_LE(
        transferReservedBytes_[destination],
        std::numeric_limits<int64_t>::max() - bytes);
    transferReservedBytes_[destination] += bytes;
    return false;
  }

  const auto transferBytes =
      queue->transferBytes() + transferReservedBytes_[destination];
  if (bytes > maxBytes || transferBytes > maxBytes - bytes) {
    if (future) {
      transferPromises_[destination].emplace_back(
          "UcxOutputQueue::reserveTransferBytes");
      *future = transferPromises_[destination].back().getSemiFuture();
    }
    return true;
  }

  VELOX_CHECK_LE(
      transferReservedBytes_[destination],
      std::numeric_limits<int64_t>::max() - bytes);
  transferReservedBytes_[destination] += bytes;
  return false;
}

bool UcxOutputQueue::waitForFullTransferCapacity(
    int64_t bytes,
    ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK_GT(bytes, 0);

  if (!fullTransferCongested_) {
    return false;
  }

  const auto retainedBytes = retainedBytesWithTransferReservationsLocked();
  if (retainedBytes <= 0) {
    // A full-transfer wait can only be satisfied by transfer bytes draining
    // from this queue. If nothing is retained or reserved, no later queue event
    // is guaranteed to wake a promise created here.
    return false;
  }
  maybeGrowFullTransferRetainedLimitLocked(retainedBytes);
  const auto retainedLimit = fullTransferRetainedLimitLocked();
  if (retainedLimit > 0 && addSaturated(retainedBytes, bytes) > retainedLimit) {
    if (future) {
      promises_.emplace_back("UcxOutputQueue::waitForFullTransferCapacity");
      *future = promises_.back().getSemiFuture();
    }
    return true;
  }
  return false;
}

bool UcxOutputQueue::reserveFullTransferBytes(
    int destination,
    int64_t bytes,
    ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK_GT(bytes, 0);
  VELOX_CHECK_GE(destination, 0);
  VELOX_CHECK_LT(destination, queues_.size());
  auto* queue = queues_[destination].get();
  if (queue == nullptr) {
    return false;
  }

  const auto retainedBytes = retainedBytesWithTransferReservationsLocked();
  if (retainedBytes > 0) {
    maybeGrowFullTransferRetainedLimitLocked(retainedBytes);

    auto retainedLimit = fullTransferRetainedLimitLocked();
    const auto requestedRetainedBytes = addSaturated(retainedBytes, bytes);
    if (retainedLimit > 0 && requestedRetainedBytes > retainedLimit) {
      if (!fullTransferCongested_) {
        fullTransferRetainedLimit_ = requestedRetainedBytes;
        retainedLimit = fullTransferRetainedLimitLocked();
      }
    }
    if (retainedLimit > 0 && requestedRetainedBytes > retainedLimit) {
      if (future) {
        promises_.emplace_back("UcxOutputQueue::reserveFullTransferBytes");
        *future = promises_.back().getSemiFuture();
      }
      return true;
    }
  }

  if (maxSize_ > 0 && fullTransferCongested_) {
    const auto activeDestinations = activeDestinationCountLocked();
    const auto fairDestinationBudget = std::max<int64_t>(
        1, fullTransferRetainedLimitLocked() / activeDestinations);
    const auto destinationBytes = addSaturated(
        queue->transferBytes(), transferReservedBytes_[destination]);
    const auto minimumDestinationBudget =
        addSaturated(transferReservedBytes_[destination], bytes);
    const auto destinationBudget =
        std::max(fairDestinationBudget, minimumDestinationBudget);
    if (destinationBytes > 0 &&
        (bytes > destinationBudget ||
         destinationBytes > destinationBudget - bytes)) {
      if (future) {
        transferPromises_[destination].emplace_back(
            "UcxOutputQueue::reserveFullTransferBytes");
        *future = transferPromises_[destination].back().getSemiFuture();
      }
      return true;
    }
  }

  VELOX_CHECK_LE(
      transferReservedBytes_[destination],
      std::numeric_limits<int64_t>::max() - bytes);
  transferReservedBytes_[destination] += bytes;
  return false;
}

void UcxOutputQueue::releaseTransferReservation(
    int destination,
    int64_t bytes) {
  std::vector<ContinuePromise> transferPromises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    releaseTransferReservationLocked(destination, bytes);
    collectTransferPromisesLocked(destination, transferPromises);
  }
  for (auto& promise : transferPromises) {
    promise.setValue();
  }
}

int64_t UcxOutputQueue::transferWindowBytes(
    int destination,
    int64_t baseBytes,
    int64_t normalBytes,
    int64_t maxBytes) {
  std::lock_guard<std::mutex> l(mutex_);
  return transferWindowBytesLocked(
      destination, baseBytes, normalBytes, maxBytes);
}

void UcxOutputQueue::recordTransferCongestion(
    int destination,
    int64_t baseBytes) {
  std::lock_guard<std::mutex> l(mutex_);
  if (destination < 0 || destination >= transferWindowBytes_.size()) {
    return;
  }

  auto& window = transferWindowBytes_[destination];
  if (window <= 0) {
    window = baseBytes;
    return;
  }
  window = std::max<int64_t>(baseBytes, window / 2);
}

void UcxOutputQueue::recordTransferDemand(
    int destination,
    int64_t targetBytes,
    int64_t baseBytes,
    int64_t maxBytes) {
  std::lock_guard<std::mutex> l(mutex_);
  if (destination < 0 || destination >= transferWindowBytes_.size()) {
    return;
  }

  auto& window = transferWindowBytes_[destination];
  if (window <= 0) {
    window = baseBytes;
  }

  const auto requestedWindow =
      std::clamp<int64_t>(targetBytes, baseBytes, maxBytes);
  const auto growthWindow =
      window > maxBytes - baseBytes ? maxBytes : window + baseBytes;
  const auto nextProbeWindow =
      std::min<int64_t>(maxBytes, std::max(growthWindow, requestedWindow));
  window = std::max<int64_t>(window, nextProbeWindow);
}

void UcxOutputQueue::recordFullTransferCongestion() {
  std::lock_guard<std::mutex> l(mutex_);
  if (maxSize_ == 0) {
    return;
  }
  fullTransferCongested_ = true;

  const auto defaultLimit = defaultFullTransferRetainedLimitLocked();
  auto retainedBytes = retainedBytesWithTransferReservationsLocked();
  if (retainedBytes <= 0) {
    retainedBytes = fullTransferRetainedLimit_ > 0 ? fullTransferRetainedLimit_
                                                   : defaultLimit;
  }

  // Back off slightly from the observed pressure point. This preserves most of
  // the successfully discovered backlog budget, but avoids immediately probing
  // the same allocation cliff again.
  const auto reducedLimit = retainedBytes - (retainedBytes / 8);
  const auto boundedMaxSize = static_cast<int64_t>(std::min<uint64_t>(
      maxSize_, static_cast<uint64_t>(std::numeric_limits<int64_t>::max())));
  const auto minimumLimit = std::max<int64_t>(1, boundedMaxSize);
  fullTransferRetainedLimit_ = std::max<int64_t>(minimumLimit, reducedLimit);
}

UcxDestinationTransferStats UcxOutputQueue::transferStats(int destination) {
  std::lock_guard<std::mutex> l(mutex_);
  UcxDestinationTransferStats stats;
  stats.retainedBytes = retainedBytesLocked();
  stats.maxBytes = maxSize_;

  VELOX_CHECK_GE(destination, 0);
  VELOX_CHECK_LT(destination, queues_.size());
  auto* queue = queues_[destination].get();
  if (queue == nullptr) {
    return stats;
  }

  const auto destinationStats = queue->stats();
  stats.bytesQueued = destinationStats.bytesQueued;
  stats.bytesInFlight = destinationStats.bytesInFlight;
  stats.bytesReserved = transferReservedBytes_[destination];
  stats.waitingForData = queue->waitingForData();
  return stats;
}

bool UcxOutputQueue::reserveOutputBytes(int64_t bytes, ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK_GT(bytes, 0);
  const auto reservationBytes = static_cast<uint64_t>(bytes);
  if (maxSize_ > 0) {
    const auto retainedBytes = static_cast<uint64_t>(retainedBytesLocked());
    if (retainedBytes > 0 &&
        (reservationBytes > maxSize_ ||
         retainedBytes > maxSize_ - reservationBytes)) {
      VELOX_CHECK_NOT_NULL(future);
      promises_.emplace_back("UcxOutputQueue::reserveOutputBytes");
      *future = promises_.back().getSemiFuture();
      return true;
    }
  }

  reservedBytes_ += bytes;
  return false;
}

void UcxOutputQueue::releaseOutputReservation(int64_t bytes) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (reservedBytes_ < bytes) {
      reservedBytes_ = 0;
    } else {
      reservedBytes_ -= bytes;
    }
    maybeContinueProducersLocked(promises);
  }
  for (auto& promise : promises) {
    promise.setValue();
  }
}

bool UcxOutputQueue::acquireInFlightBytes(
    int destination,
    int64_t bytes,
    int64_t numPackedCols) {
  std::lock_guard<std::mutex> l(mutex_);
  if (destination < 0 || destination >= queues_.size() ||
      queues_[destination] == nullptr) {
    return false;
  }
  queues_[destination]->acquireInFlight(bytes, numPackedCols);

  // These were never queued here, so only the in-flight side moves.
  inFlightBytes_ += bytes;
  inFlightPackedColumns_ += numPackedCols;
  return true;
}

void UcxOutputQueue::releaseInFlightBytes(
    int destination,
    int64_t bytes,
    int64_t numPackedCols) {
  std::vector<ContinuePromise> promises;
  std::vector<ContinuePromise> transferPromises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    // The destination's counters only while it has a queue; the task-level
    // total always, or bytes for a page still being sent are stranded.
    if (destination >= 0 && destination < queues_.size() &&
        queues_[destination] != nullptr) {
      queues_[destination]->releaseInFlight(bytes, numPackedCols);
    }
    updateStatsWithSendCompleteLocked(bytes, numPackedCols, promises);
    collectTransferPromisesLocked(destination, transferPromises);
  }
  for (auto& promise : promises) {
    promise.setValue();
  }
  for (auto& promise : transferPromises) {
    promise.setValue();
  }
}

void UcxOutputQueue::getData(int destination, UcxDataAvailableCallback notify) {
  VELOX_CHECK_GE(destination, 0);
  UcxDestinationQueue::Data data;
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (destination >= queues_.size()) {
      VELOX_CHECK_LT(destination, std::numeric_limits<int>::max());
      // An uninitialized placeholder has the default partitioned kind, but
      // must still grow to hold early getData() callbacks. Once task metadata
      // is published, partitioned output has a fixed destination count.
      VELOX_CHECK(
          !isInitialized() ||
              kind_ != core::PartitionedOutputNode::Kind::kPartitioned,
          "Cannot add destinations to initialized partitioned output");
      addOutputBuffersLocked(destination + 1);
    }
    auto* queue = queues_[destination].get();
    // queue can be nullptr here if the task has terminated and results
    // have been removed. In this case, no data is returned.
    if (queue) {
      // For arbitrary mode, pull from the shared buffer if the destination
      // queue is empty. This ensures demand-driven distribution.
      if (kind_ == core::PartitionedOutputNode::Kind::kArbitrary &&
          !arbitraryBuffer_.empty()) {
        queue->enqueueBack(std::move(arbitraryBuffer_.front()));
        arbitraryBuffer_.pop_front();
      }
      // Capture weak_ptr instead of raw `this` to prevent use-after-free.
      // The callback fires outside the lock (from enqueue() or terminate()),
      // and concurrent removeTask() can destroy the UcxOutputQueue while
      // the callback is still executing.
      std::weak_ptr<UcxOutputQueue> weakSelf = shared_from_this();
      data = queue->getData([destination, notify, weakSelf](
                                std::shared_ptr<UcxGpuPayload> data,
                                std::vector<int64_t> remainingBytes) {
        int64_t bytes = data ? data->payloadBytes() : -1L;
        if (bytes >= 0L) {
          auto self = weakSelf.lock();
          if (self) {
            std::vector<ContinuePromise> promises;
            {
              std::lock_guard<std::mutex> l(self->mutex_);
              self->updateStatsWithDequeuedLocked(bytes, 1L, promises);
            }
            for (auto& promise : promises) {
              promise.setValue();
            }
          }
        }
        notify(std::move(data), std::move(remainingBytes));
      });
      if (data.data) {
        // This implies data.immediate and no notify upcall will be done.
        // Need to update the stats here. The data is retained by the server
        // until transfer completion.
        updateStatsWithDequeuedLocked(data.data->payloadBytes(), 1L, promises);
      }
    } else {
      data = UcxDestinationQueue::Data{nullptr, {}, true};
    }
  }
  for (auto& promise : promises) {
    promise.setValue();
  }
  // outside lock: If we have data, then return it immediately.
  if (data.immediate) {
    notify(std::move(data.data), std::move(data.remainingBytes));
  } else {
    VLOG(2) << "[QUEUE] task=" << (task_ ? task_->taskId() : "n/a")
            << " dest=" << destination
            << " server waiting for data (callback installed)";
  }
}

void UcxOutputQueue::noMoreData() {
  // Increment number of finished drivers.
  checkIfDone(true);
}

void UcxOutputQueue::noMoreDrivers() {
  // Do not increment number of finished drivers.
  checkIfDone(false);
}

void UcxOutputQueue::checkIfDone(bool oneDriverFinished) {
  std::vector<UcxDataAvailable> finished;
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (oneDriverFinished) {
      ++numFinished_;
    }
    VELOX_CHECK_LE(
        numFinished_,
        numDrivers_,
        "Each driver should call noMoreData exactly once");
    atEnd_ = numFinished_ == numDrivers_;
    if (!atEnd_) {
      maybeContinueProducersLocked(promises);
    } else {
      // For arbitrary, drain remaining shared pool to destination queues
      // round-robin before sending end markers.
      if (kind_ == core::PartitionedOutputNode::Kind::kArbitrary) {
        int32_t bufferId =
            queues_.empty() ? 0 : nextArbitraryLoadIndex_ % queues_.size();
        int32_t nullCount = 0;
        while (!arbitraryBuffer_.empty() && !queues_.empty()) {
          auto* queue = queues_[bufferId].get();
          if (queue != nullptr) {
            queue->enqueueBack(std::move(arbitraryBuffer_.front()));
            arbitraryBuffer_.pop_front();
            nullCount = 0;
          } else if (++nullCount >= queues_.size()) {
            arbitraryBuffer_.clear();
            break;
          }
          bufferId = (bufferId + 1) % queues_.size();
        }
      }
      for (auto& queue : queues_) {
        if (queue != nullptr) {
          queue->enqueueBack(nullptr);
          finished.push_back(queue->getAndClearNotify());
        }
      }
    }
  }
  // Notify outside of mutex.
  for (auto& notification : finished) {
    notification.notify();
  }
  for (auto& promise : promises) {
    promise.setValue();
  }
}

bool UcxOutputQueue::enqueuePartitionedOutputLocked(
    int destination,
    std::shared_ptr<UcxGpuPayload> data,
    std::vector<UcxDataAvailable>& dataAvailableCbs,
    int64_t transferReservationBytes) {
  VELOX_DCHECK(dataAvailableCbs.empty());
  VELOX_CHECK_LT(destination, queues_.size());
  bool success = false;
  auto* queue = queues_[destination].get();
  releaseTransferReservationLocked(destination, transferReservationBytes);
  if (queue != nullptr) {
    queue->enqueueBack(std::move(data));
    dataAvailableCbs.emplace_back(queue->getAndClearNotify());
    success = true;
  }
  return success;
}

void UcxOutputQueue::releaseTransferReservationLocked(
    int destination,
    int64_t bytes) {
  if (bytes <= 0) {
    return;
  }

  VELOX_CHECK_GE(destination, 0);
  VELOX_CHECK_LT(destination, transferReservedBytes_.size());
  if (transferReservedBytes_[destination] < bytes) {
    transferReservedBytes_[destination] = 0;
  } else {
    transferReservedBytes_[destination] -= bytes;
  }
}

int64_t UcxOutputQueue::transferWindowBytesLocked(
    int destination,
    int64_t baseBytes,
    int64_t normalBytes,
    int64_t maxBytes) {
  VELOX_CHECK_GT(baseBytes, 0);
  VELOX_CHECK_GE(normalBytes, baseBytes);
  VELOX_CHECK_GE(maxBytes, normalBytes);
  VELOX_CHECK_GE(destination, 0);
  VELOX_CHECK_LT(destination, queues_.size());

  if (destination >= transferWindowBytes_.size()) {
    transferWindowBytes_.resize(queues_.size(), 0);
  }

  auto* queue = queues_[destination].get();
  if (queue == nullptr) {
    return baseBytes;
  }

  auto& window = transferWindowBytes_[destination];
  if (window <= 0) {
    window = normalBytes;
  }
  window = std::clamp<int64_t>(window, baseBytes, maxBytes);

  const auto destinationStats = queue->stats();
  const auto transferBytes =
      queue->transferBytes() + transferReservedBytes_[destination];
  const auto retainedBytes =
      static_cast<uint64_t>(std::max<int64_t>(retainedBytesLocked(), 0));
  const bool retainedPressure =
      maxSize_ > 0 && retainedBytes >= maxSize_ - (maxSize_ / 10);

  if (retainedPressure) {
    window = baseBytes;
  } else if (
      queue->waitingForData() && destinationStats.bytesQueued == 0 &&
      transferBytes == 0) {
    window = std::min<int64_t>(
        maxBytes, std::max<int64_t>(normalBytes, window + baseBytes));
  } else if (transferBytes < window / 2 && window < normalBytes) {
    window = std::min<int64_t>(normalBytes, window + baseBytes);
  }

  return window;
}

void UcxOutputQueue::collectTransferPromisesLocked(
    int destination,
    std::vector<ContinuePromise>& promises) {
  if (destination < 0 || destination >= transferPromises_.size()) {
    return;
  }

  for (auto& promise : transferPromises_[destination]) {
    promises.push_back(std::move(promise));
  }
  transferPromises_[destination].clear();
}

void UcxOutputQueue::collectAllTransferPromisesLocked(
    std::vector<ContinuePromise>& promises) {
  for (auto& destinationPromises : transferPromises_) {
    for (auto& promise : destinationPromises) {
      promises.push_back(std::move(promise));
    }
    destinationPromises.clear();
  }
}

void UcxOutputQueue::enqueueBroadcastOutputLocked(
    std::shared_ptr<UcxGpuPayload> data,
    std::vector<UcxDataAvailable>& dataAvailableCbs) {
  VELOX_DCHECK(dataAvailableCbs.empty());

  for (auto& queue : queues_) {
    if (queue != nullptr) {
      queue->enqueueBack(data);
      dataAvailableCbs.emplace_back(queue->getAndClearNotify());
    }
  }

  // Store for late-arriving destinations (backfill).
  if (!noMoreQueues_) {
    dataToBroadcast_.emplace_back(std::move(data));
  }
}

void UcxOutputQueue::enqueueArbitraryOutputLocked(
    std::shared_ptr<UcxGpuPayload> data,
    std::vector<UcxDataAvailable>& dataAvailableCbs) {
  VELOX_DCHECK(dataAvailableCbs.empty());

  arbitraryBuffer_.push_back(std::move(data));

  // Distribute from the shared pool to destinations with waiting consumers,
  // probing round-robin so no single destination starves.
  int32_t bufferId =
      queues_.empty() ? 0 : nextArbitraryLoadIndex_ % queues_.size();
  for (int32_t i = 0; i < queues_.size(); ++i) {
    if (arbitraryBuffer_.empty()) {
      break;
    }
    auto* queue = queues_[bufferId].get();
    if (queue != nullptr) {
      auto pending = queue->getAndClearNotify();
      if (pending.callback) {
        pending.data = std::move(arbitraryBuffer_.front());
        arbitraryBuffer_.pop_front();
        pending.remainingBytes.clear();
        for (const auto& item : arbitraryBuffer_) {
          if (item != nullptr) {
            pending.remainingBytes.push_back(item->data->gpu_data->size());
          }
        }
        dataAvailableCbs.push_back(std::move(pending));
      }
    }
    bufferId = (bufferId + 1) % queues_.size();
  }
  nextArbitraryLoadIndex_ = bufferId;
}

bool UcxOutputQueue::isFinished() {
  std::lock_guard<std::mutex> l(mutex_);
  return isFinishedLocked();
}

bool UcxOutputQueue::isFinishedLocked() {
  // For broadcast, we can only be finished after receiving the no more
  // (destination) buffers signal, matching OutputBuffer::isFinishedLocked().
  // For arbitrary, the coordinator lazily manages consumers and may never send
  // noMoreBufferIds, so we only check that all queues have been consumed.
  if (kind_ == core::PartitionedOutputNode::Kind::kBroadcast &&
      !noMoreQueues_) {
    return false;
  }
  for (auto& queue : queues_) {
    if (queue != nullptr) {
      return false;
    }
  }
  return true;
}

std::optional<exec::TaskState> UcxOutputQueue::taskState() {
  std::shared_ptr<exec::Task> task;
  {
    std::lock_guard<std::mutex> l(mutex_);
    task = task_;
  }
  if (task == nullptr) {
    return std::nullopt;
  }
  return task->state();
}

void UcxOutputQueue::updateOutputBuffers(int numBuffers, bool noMoreBuffers) {
  using Kind = core::PartitionedOutputNode::Kind;
  bool isFinished{false};
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_GE(numBuffers, 0);
    if (kind_ == Kind::kPartitioned) {
      VELOX_CHECK_EQ(queues_.size(), numBuffers);
      VELOX_CHECK(noMoreBuffers);
      noMoreQueues_ = true;
      return;
    }

    VELOX_CHECK(kind_ == Kind::kBroadcast || kind_ == Kind::kArbitrary);

    if (numBuffers > queues_.size()) {
      addOutputBuffersLocked(numBuffers);
    }

    if (!noMoreBuffers) {
      return;
    }

    noMoreQueues_ = true;
    dataToBroadcast_.clear();
    isFinished = isFinishedLocked();
  }

  if (isFinished && task_) {
    task_->setAllOutputConsumed();
  }
}

void UcxOutputQueue::deleteResults(int destination) {
  bool isFinished;
  UcxDataAvailable dataAvailable;
  std::vector<ContinuePromise> promises;
  std::vector<ContinuePromise> transferPromises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (destination >= queues_.size()) {
      VLOG(1) << "deleteResults: destination " << destination
              << " out of range (size=" << queues_.size() << "), ignoring";
      return;
    }
    auto* queue = queues_[destination].get();
    if (queue == nullptr) {
      VLOG(1) << "Extra delete received for destination " << destination;
      return;
    }
    // remember destination queue fill stats
    int64_t bytes = queue->stats().bytesQueued;
    int64_t packedCols = queue->stats().packedColumnsQueued;
    queue->deleteResults();
    dataAvailable = queue->getAndClearNotify();
    queue->finish();
    queues_[destination] = nullptr;
    isFinished = isFinishedLocked();
    // update UcxOutputQueue stats
    updateStatsWithFreedLocked(bytes, packedCols, promises);
    if (destination < transferReservedBytes_.size()) {
      transferReservedBytes_[destination] = 0;
    }
    if (destination < transferWindowBytes_.size()) {
      transferWindowBytes_[destination] = 0;
    }
    collectTransferPromisesLocked(destination, transferPromises);
  }

  // Outside of mutex.
  dataAvailable.notify();
  // wake up any producers that are waiting for queue to become less full.
  for (auto& promise : promises) {
    promise.setValue();
  }
  for (auto& promise : transferPromises) {
    promise.setValue();
  }

  if (isFinished && task_) {
    task_->setAllOutputConsumed();
  }
}

void UcxOutputQueue::terminate() {
  std::vector<UcxDataAvailable> pendingCallbacks;
  std::vector<ContinuePromise> promises;
  std::vector<ContinuePromise> transferPromises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (task_ && task_->isRunning()) {
      LOG(WARNING) << "UcxOutputQueue::terminate() called while task "
                   << task_->taskId() << " is still running";
    }
    arbitraryBuffer_.clear();
    // Fire all pending getData callbacks with nullptr to signal end-of-stream.
    // This handles the case where a producer task fails or is cancelled before
    // noMoreData() is called, preventing consumers from being orphaned.
    for (auto& queue : queues_) {
      if (queue != nullptr) {
        queue->enqueueBack(nullptr);
        pendingCallbacks.push_back(queue->getAndClearNotify());
      }
    }
    // Release any outstanding producer-side promises (blocked on queue-full).
    reservedBytes_ = 0;
    std::fill(transferReservedBytes_.begin(), transferReservedBytes_.end(), 0);
    std::fill(transferWindowBytes_.begin(), transferWindowBytes_.end(), 0);
    promises = std::move(promises_);
    collectAllTransferPromisesLocked(transferPromises);
  }

  // Fire callbacks outside of mutex to avoid potential deadlocks.
  for (auto& callback : pendingCallbacks) {
    callback.notify();
  }
  // Unblock any blocked producers.
  for (auto& promise : promises) {
    promise.setValue();
  }
  for (auto& promise : transferPromises) {
    promise.setValue();
  }
}

std::string UcxOutputQueue::toString() {
  std::stringstream out;
  std::lock_guard<std::mutex> l(mutex_);
  out << "[UcxOutputQueue task="
      << (task_ ? task_->taskId() : "<uninitialized>") << " queues=";
  for (const auto& queue : queues_) {
    out << (queue ? queue->toString() : "<deleted>") << ", ";
  }
  out << " reservedBytes=" << reservedBytes_
      << ", transferReservedBytes=" << transferReservedBytesLocked()
      << ", queuedBytes=" << queuedBytes_
      << ", queuedPackedColumns=" << queuedPackedColumns_
      << ", inFlightBytes=" << inFlightBytes_
      << ", inFlightPackedColumns=" << inFlightPackedColumns_
      << ", numFinished=" << numFinished_ << "/" << numDrivers_
      << ", atEnd=" << atEnd_ << ", noMoreQueues=" << noMoreQueues_ << "]";
  return out.str();
}

exec::OutputBuffer::Stats UcxOutputQueue::stats() {
  std::lock_guard<std::mutex> l(mutex_);
  std::vector<UcxDestinationQueue::Stats> queueStats;

  updateTotalQueuedBytesMsLocked();

  auto stats = exec::OutputBuffer::Stats(
      kind(),
      noMoreQueues_,
      atEnd_,
      isFinishedLocked(),
      retainedBytesLocked(),
      retainedPackedColumnsLocked(),
      totalBytesSent_,
      totalRowsSent_,
      totalPackedColumnsSent_,
      getAverageQueueTimeMsLocked(),
      0 /* FIXME: compute num top buffers. */,
      {/* FIXME: transition queueStats to exec::DestinationBuffer::Stats */});
  return stats;
}

double UcxOutputQueue::getUtilization() {
  std::lock_guard<std::mutex> l(mutex_);
  return maxSize_ == 0 ? 0.0
                       : retainedBytesLocked() / static_cast<double>(maxSize_);
}

bool UcxOutputQueue::isOverutilized() {
  std::lock_guard<std::mutex> l(mutex_);
  return atEnd_ || (maxSize_ > 0 && retainedBytesLocked() > (0.5 * maxSize_));
}

void UcxOutputQueue::updateStatsWithEnqueuedLocked(
    int64_t bytes,
    int64_t rows) {
  updateTotalQueuedBytesMsLocked();

  queuedBytes_ += bytes;
  queuedPackedColumns_++;

  totalBytesSent_ += bytes;
  totalRowsSent_ += rows;
  totalPackedColumnsSent_++;
}

void UcxOutputQueue::updateStatsWithDequeuedLocked(
    int64_t bytes,
    int64_t numPackedCols,
    std::vector<ContinuePromise>& promises) {
  updateTotalQueuedBytesMsLocked();

  queuedBytes_ -= bytes;
  queuedPackedColumns_ -= numPackedCols;
  inFlightBytes_ += bytes;
  inFlightPackedColumns_ += numPackedCols;

  VELOX_CHECK_GE(queuedBytes_, 0);
  VELOX_CHECK_GE(queuedPackedColumns_, 0);
  VELOX_CHECK_GE(inFlightBytes_, 0);
  VELOX_CHECK_GE(inFlightPackedColumns_, 0);

  maybeContinueProducersLocked(promises);
}

void UcxOutputQueue::updateStatsWithFreedLocked(
    int64_t bytes,
    int64_t numPackedCols,
    std::vector<ContinuePromise>& promises) {
  updateTotalQueuedBytesMsLocked();

  queuedBytes_ -= bytes;
  queuedPackedColumns_ -= numPackedCols;

  VELOX_CHECK_GE(queuedBytes_, 0);
  VELOX_CHECK_GE(queuedPackedColumns_, 0);

  // Check whether queue is below low-water mark and return outstanding
  // promises
  maybeContinueProducersLocked(promises);
}

void UcxOutputQueue::updateStatsWithSendCompleteLocked(
    int64_t bytes,
    int64_t numPackedCols,
    std::vector<ContinuePromise>& promises) {
  inFlightBytes_ -= bytes;
  inFlightPackedColumns_ -= numPackedCols;

  VELOX_CHECK_GE(inFlightBytes_, 0);
  VELOX_CHECK_GE(inFlightPackedColumns_, 0);

  maybeContinueProducersLocked(promises);
}

void UcxOutputQueue::updateTotalQueuedBytesMsLocked() {
  const auto nowMs = getCurrentTimeMs();
  if (queuedBytes_ > 0) {
    const auto deltaMs = nowMs - queueStartMs_;
    totalQueuedBytesMs_ += queuedBytes_ * deltaMs;
  }

  queueStartMs_ = nowMs;
}

int64_t UcxOutputQueue::getAverageQueueTimeMsLocked() const {
  if (totalBytesSent_ > 0) {
    return totalQueuedBytesMs_ / totalBytesSent_;
  }

  return 0;
}

void UcxOutputQueue::maybeContinueProducersLocked(
    std::vector<ContinuePromise>& promises) {
  auto producerBlockedBytes = producerBlockedBytesLocked();
  auto continueBytes = static_cast<int64_t>(continueSize_);
  if (fullTransferCongested_ && fullTransferRetainedLimit_ > 0) {
    const auto retainedBytes = retainedBytesWithTransferReservationsLocked();
    maybeGrowFullTransferRetainedLimitLocked(retainedBytes);
    producerBlockedBytes = retainedBytes;
    continueBytes = fullTransferRetainedLimitLocked();
  }
  if (producerBlockedBytes > continueBytes || promises_.empty()) {
    return;
  }
  const auto numProducersToUnblock =
      producerBlockedBytes == 0 ? promises_.size() : 1;
  for (size_t i = 0; i < numProducersToUnblock; ++i) {
    promises.push_back(std::move(promises_[i]));
  }
  promises_.erase(promises_.begin(), promises_.begin() + numProducersToUnblock);
}

int64_t UcxOutputQueue::retainedBytesLocked() const {
  return reservedBytes_ + queuedBytes_ + inFlightBytes_;
}

int64_t UcxOutputQueue::producerBlockedBytesLocked() const {
  return reservedBytes_ + queuedBytes_;
}

int64_t UcxOutputQueue::retainedPackedColumnsLocked() const {
  return queuedPackedColumns_ + inFlightPackedColumns_;
}

int64_t UcxOutputQueue::transferBytesLocked(int destination) const {
  VELOX_CHECK_GE(destination, 0);
  VELOX_CHECK_LT(destination, queues_.size());
  auto* queue = queues_[destination].get();
  return (queue == nullptr ? 0 : queue->transferBytes()) +
      transferReservedBytes_[destination];
}

int64_t UcxOutputQueue::transferReservedBytesLocked() const {
  int64_t total = 0;
  for (const auto bytes : transferReservedBytes_) {
    total = addSaturated(total, std::max<int64_t>(bytes, 0));
  }
  return total;
}

int64_t UcxOutputQueue::retainedBytesWithTransferReservationsLocked() const {
  return addSaturated(retainedBytesLocked(), transferReservedBytesLocked());
}

int64_t UcxOutputQueue::activeDestinationCountLocked() const {
  int64_t count = 0;
  for (const auto& queue : queues_) {
    if (queue != nullptr) {
      ++count;
    }
  }
  return std::max<int64_t>(count, 1);
}

int64_t UcxOutputQueue::defaultFullTransferRetainedLimitLocked() const {
  if (maxSize_ == 0) {
    return 0;
  }
  const auto boundedMaxSize = static_cast<int64_t>(std::min<uint64_t>(
      maxSize_, static_cast<uint64_t>(std::numeric_limits<int64_t>::max())));
  return multiplySaturated(boundedMaxSize, activeDestinationCountLocked());
}

int64_t UcxOutputQueue::fullTransferRetainedLimitLocked() const {
  const auto defaultLimit = defaultFullTransferRetainedLimitLocked();
  if (defaultLimit == 0) {
    return 0;
  }
  if (fullTransferRetainedLimit_ <= 0) {
    return defaultLimit;
  }
  return std::max<int64_t>(1, fullTransferRetainedLimit_);
}

void UcxOutputQueue::maybeGrowFullTransferRetainedLimitLocked(
    int64_t retainedBytes) {
  if (maxSize_ == 0 || !fullTransferCongested_ ||
      fullTransferRetainedLimit_ <= 0) {
    return;
  }

  if (retainedBytes <= fullTransferRetainedLimit_ / 2) {
    const auto boundedMaxSize = static_cast<int64_t>(std::min<uint64_t>(
        maxSize_, static_cast<uint64_t>(std::numeric_limits<int64_t>::max())));
    const auto increment = std::max<int64_t>(1, boundedMaxSize);
    fullTransferRetainedLimit_ =
        addSaturated(fullTransferRetainedLimit_, increment);
  }
}

} // namespace facebook::velox::ucx_exchange
