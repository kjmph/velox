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
#include "velox/experimental/ucx-exchange/UcxCpuRowQueues.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <string>

#include "velox/common/time/Timer.h"

namespace facebook::velox::ucx_exchange {
namespace {

int64_t saturatingMultiplyToInt64(uint64_t lhs, uint64_t rhs) {
  constexpr uint64_t kMax =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  if (lhs == 0 || rhs == 0) {
    return 0;
  }
  if (lhs > kMax / rhs) {
    return std::numeric_limits<int64_t>::max();
  }
  return static_cast<int64_t>(lhs * rhs);
}

} // namespace

void UcxCpuRowDestinationQueue::Stats::recordEnqueue(
    const UcxCpuRowPayload* payload) {
  if (payload != nullptr) {
    bytesQueued += payload->numBytes;
    rowsQueued += payload->numRows;
    payloadsQueued++;
  }
}

void UcxCpuRowDestinationQueue::Stats::recordDequeue(
    const UcxCpuRowPayload* payload) {
  if (payload != nullptr) {
    const int64_t size = payload->numBytes;

    bytesQueued -= size;
    VELOX_DCHECK_GE(bytesQueued, 0, "bytesQueued must be non-negative");
    rowsQueued -= payload->numRows;
    VELOX_DCHECK_GE(rowsQueued, 0, "rowsQueued must be non-negative");
    --payloadsQueued;
    VELOX_DCHECK_GE(payloadsQueued, 0, "payloadsQueued must be non-negative");

    bytesSent += size;
    rowsSent += payload->numRows;
    payloadsSent++;
  }
}

exec::DestinationBuffer::Stats
UcxCpuRowDestinationQueue::Stats::toOutputBufferStats() const {
  exec::DestinationBuffer::Stats out;
  out.finished = finished;
  out.bytesBuffered = bytesQueued;
  out.rowsBuffered = rowsQueued;
  out.pagesBuffered = payloadsQueued;
  out.bytesSent = bytesSent;
  out.rowsSent = rowsSent;
  out.pagesSent = payloadsSent;
  return out;
}

void UcxCpuRowDestinationQueue::enqueueBack(
    std::shared_ptr<UcxCpuRowPayload> data) {
  // Drop duplicate end-of-stream markers.
  if (data == nullptr && !queue_.empty() && queue_.back() == nullptr) {
    return;
  }

  if (data != nullptr) {
    stats_.recordEnqueue(data.get());
  }
  queue_.push_back(std::move(data));
}

UcxCpuRowDestinationQueue::Data UcxCpuRowDestinationQueue::getData(
    UcxCpuRowDataAvailableCallback notify) {
  if (queue_.empty()) {
    notify_ = std::move(notify);
    return {};
  }

  auto data = std::move(queue_.front());
  queue_.pop_front();
  stats_.recordDequeue(data.get());
  return {std::move(data), {}, true};
}

std::shared_ptr<UcxCpuRowPayload>
UcxCpuRowDestinationQueue::tryDequeueLocked() {
  if (queue_.empty()) {
    return nullptr;
  }
  // The end-of-stream null marker must go through the regular getData
  // path; bundling shouldn't swallow it. Leave it at the head.
  if (queue_.front() == nullptr) {
    return nullptr;
  }
  auto data = std::move(queue_.front());
  queue_.pop_front();
  stats_.recordDequeue(data.get());
  return data;
}

void UcxCpuRowDestinationQueue::deleteResults() {
  for (size_t i = 0; i < queue_.size(); ++i) {
    if (queue_[i] == nullptr) {
      VELOX_CHECK_EQ(i, queue_.size() - 1, "null marker found in the middle");
      break;
    }
    stats_.recordDequeue(queue_[i].get());
  }
  queue_.clear();
}

UcxCpuRowDataAvailable UcxCpuRowDestinationQueue::getAndClearNotify() {
  if (notify_ == nullptr) {
    return UcxCpuRowDataAvailable();
  }
  UcxCpuRowDataAvailable result;
  result.callback = notify_;
  auto data = getData(nullptr);
  result.data = std::move(data.data);
  result.remainingBytes = std::move(data.remainingBytes);
  clearNotify();
  return result;
}

void UcxCpuRowDestinationQueue::clearNotify() {
  notify_ = nullptr;
}

void UcxCpuRowDestinationQueue::finish() {
  VELOX_CHECK_NULL(notify_, "notify must be cleared before finish");
  VELOX_CHECK(queue_.empty(), "data must be fetched before finish");
  stats_.finished = true;
}

UcxCpuRowDestinationQueue::Stats UcxCpuRowDestinationQueue::stats() const {
  return stats_;
}

std::string UcxCpuRowDestinationQueue::toString() {
  std::stringstream out;
  out << "[available: " << queue_.size() << ", "
      << (notify_ ? "notify registered, " : "") << this << "]";
  return out.str();
}

// ---------- UcxCpuRowOutputQueue ----------

UcxCpuRowOutputQueue::UcxCpuRowOutputQueue(
    std::shared_ptr<exec::Task> task,
    uint32_t numDestinations,
    uint32_t numDrivers,
    core::PartitionedOutputNode::Kind kind)
    : task_(task), kind_(kind), numDrivers_(numDrivers) {
  if (task_) {
    maxSize_ = task_->queryCtx()->queryConfig().maxOutputBufferSize();
    continueSize_ = (maxSize_ * kContinuePct) / 100;
  }
  queues_.reserve(numDestinations);
  for (int i = 0; i < numDestinations; ++i) {
    queues_.emplace_back(std::make_unique<UcxCpuRowDestinationQueue>());
  }
  finishedBufferStats_.resize(numDestinations);
  if (task_) {
    initialized_.store(true, std::memory_order_release);
  }
}

bool UcxCpuRowOutputQueue::initialize(
    std::shared_ptr<exec::Task> task,
    uint32_t numDestinations,
    uint32_t numDrivers,
    core::PartitionedOutputNode::Kind kind) {
  std::lock_guard<std::mutex> l(mutex_);
  if (task_) {
    return false;
  }
  kind_ = kind;
  numDrivers_ = numDrivers;
  task_ = task;
  maxSize_ = task_->queryCtx()->queryConfig().maxOutputBufferSize();
  continueSize_ = (maxSize_ * kContinuePct) / 100;
  if (numDestinations > queues_.size()) {
    addOutputBuffersLocked(numDestinations);
  }
  // Release-store: paired with isInitialized()'s acquire-load so
  // lock-free readers see a fully-populated queue.
  initialized_.store(true, std::memory_order_release);
  return true;
}

void UcxCpuRowOutputQueue::addOutputBuffersLocked(int numBuffers) {
  using Kind = core::PartitionedOutputNode::Kind;

  VELOX_CHECK_GE(numBuffers, 0);
  VELOX_CHECK_LT(queues_.size(), numBuffers);
  VELOX_CHECK(!noMoreQueues_, "Cannot add destinations after noMoreBuffers");
  // An uninitialized placeholder has the default partitioned kind, but must
  // still grow to hold early getData() callbacks. Once task metadata is
  // published, partitioned output has a fixed destination count.
  VELOX_CHECK(
      !isInitialized() || kind_ != Kind::kPartitioned,
      "Cannot add destinations to initialized partitioned output");

  queues_.reserve(numBuffers);
  if (kind_ == Kind::kBroadcast && !dataToBroadcast_.empty()) {
    updateTotalQueuedBytesMsLocked();
  }
  for (int32_t i = queues_.size(); i < numBuffers; ++i) {
    auto buffer = std::make_unique<UcxCpuRowDestinationQueue>();
    if (kind_ == Kind::kBroadcast) {
      for (const auto& data : dataToBroadcast_) {
        buffer->enqueueBack(data);
        // Each backfilled payload will be decremented independently when this
        // destination drains.
        queuedBytes_ += data->numBytes;
        queuedPayloads_++;
      }
    }
    // Preserve FIFO ordering: retained broadcast payloads precede EOS.
    if (atEnd_) {
      buffer->enqueueBack(nullptr);
    }
    queues_.emplace_back(std::move(buffer));
  }
  finishedBufferStats_.resize(queues_.size());
}

void UcxCpuRowOutputQueue::updateNumDrivers(uint32_t newNumDrivers) {
  bool isNoMoreDrivers{false};
  {
    std::lock_guard<std::mutex> l(mutex_);
    numDrivers_ = newNumDrivers;
    if (numDrivers_ == numFinished_) {
      isNoMoreDrivers = true;
    }
  }
  if (isNoMoreDrivers) {
    noMoreDrivers();
  }
}

void UcxCpuRowOutputQueue::enqueue(
    int destination,
    std::unique_ptr<UcxCpuRowPayload> data,
    int32_t numRows) {
  VELOX_CHECK_NOT_NULL(data);
  VELOX_CHECK_NOT_NULL(task_);
  VELOX_CHECK(
      task_->isRunning(), "Task is terminated, cannot add data to output.");
  std::vector<UcxCpuRowDataAvailable> dataAvailableCallbacks;
  {
    std::lock_guard<std::mutex> l(mutex_);
    auto numBytes = data->numBytes;
    auto sharedData = std::shared_ptr<UcxCpuRowPayload>(std::move(data));

    bool success = false;
    if (kind_ == core::PartitionedOutputNode::Kind::kBroadcast) {
      VELOX_CHECK_EQ(destination, 0, "Broadcast uses destination 0");
      enqueueBroadcastOutputLocked(
          std::move(sharedData), dataAvailableCallbacks);
      // For broadcast, count queuedBytes_ once per active destination so
      // each destination's dequeue symmetrically decrements it. Total
      // sent stats count the logical payload once.
      int numActive = 0;
      for (auto& q : queues_) {
        if (q != nullptr) {
          numActive++;
        }
      }
      updateTotalQueuedBytesMsLocked();
      queuedBytes_ += numBytes * numActive;
      queuedPayloads_ += numActive;
      totalBytesSent_ += numBytes;
      totalRowsSent_ += numRows;
      totalPayloadsSent_++;
      success = true;
    } else if (kind_ == core::PartitionedOutputNode::Kind::kArbitrary) {
      VELOX_CHECK_EQ(destination, 0, "Arbitrary uses destination 0");
      enqueueArbitraryOutputLocked(
          std::move(sharedData), dataAvailableCallbacks);
      updateStatsWithEnqueuedLocked(numBytes, numRows);
      success = true;
    } else {
      VELOX_CHECK_LT(destination, queues_.size());
      success = enqueuePartitionedOutputLocked(
          destination, std::move(sharedData), dataAvailableCallbacks);
      if (success) {
        updateStatsWithEnqueuedLocked(numBytes, numRows);
      }
    }
  }
  // Notify outside the mutex.
  for (auto& callback : dataAvailableCallbacks) {
    callback.notify();
  }
}

bool UcxCpuRowOutputQueue::checkBlocked(ContinueFuture* future) {
  std::vector<ContinuePromise> promises;
  bool blocked = false;
  {
    std::lock_guard<std::mutex> l(mutex_);
    maybeUnblockProducersLocked(promises);
    const auto highWaterMark = highWaterMarkLocked();
    if (queuedBytes_ >= highWaterMark) {
      if (future == nullptr) {
        blocked = true;
      } else {
        promises_.emplace_back("UcxCpuRowOutputQueue::checkBlocked");
        *future = promises_.back().getSemiFuture();
        blocked = true;
      }
    }
  }
  for (auto& promise : promises) {
    promise.setValue();
  }
  return blocked;
}

void UcxCpuRowOutputQueue::getData(
    int destination,
    UcxCpuRowDataAvailableCallback notify) {
  VELOX_CHECK_GE(destination, 0);
  UcxCpuRowDestinationQueue::Data data;
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (destination >= queues_.size()) {
      VELOX_CHECK_LT(destination, std::numeric_limits<int>::max());
      // Before initialization this creates placeholders for early callbacks.
      // After initialization it also backfills a late broadcast destination.
      addOutputBuffersLocked(destination + 1);
    }
    auto* queue = queues_[destination].get();
    // queue can be nullptr if the destination's results have already
    // been deleted (post-cancellation). Return empty in that case.
    if (queue) {
      // For arbitrary mode, pull from the shared buffer if the destination
      // queue is empty. This ensures demand-driven distribution.
      if (kind_ == core::PartitionedOutputNode::Kind::kArbitrary &&
          !arbitraryBuffer_.empty()) {
        queue->enqueueBack(std::move(arbitraryBuffer_.front()));
        arbitraryBuffer_.pop_front();
      }
      // weak_ptr capture: removeTask() can destroy this UcxCpuRowOutputQueue
      // while the callback is still scheduled.
      std::weak_ptr<UcxCpuRowOutputQueue> weakSelf = shared_from_this();
      data = queue->getData([notify, weakSelf](
                                std::shared_ptr<UcxCpuRowPayload> data,
                                std::vector<int64_t> remainingBytes) {
        std::vector<ContinuePromise> promises;
        int64_t bytes = data ? data->numBytes : -1L;
        notify(std::move(data), std::move(remainingBytes));
        if (bytes >= 0L) {
          auto self = weakSelf.lock();
          if (!self) {
            return;
          }
          std::lock_guard<std::mutex> l(self->mutex_);
          self->acknowledgeDirectHandoffLocked(bytes, 1L);
          self->updateStatsWithFreedLocked(bytes, 1L, promises);
        }
        for (auto& promise : promises) {
          promise.setValue();
        }
      });
      if (data.data) {
        // Synchronous return path; update stats here since the notify
        // callback path that does so was bypassed.
        updateStatsWithFreedLocked(data.data->numBytes, 1L, promises);
      } else if (!data.immediate) {
        // A parked consumer also gives producers a chance to observe that the
        // exact incremental queue accounting is below the low-water mark.
        maybeUnblockProducersLocked(promises);
      }
    } else {
      data = UcxCpuRowDestinationQueue::Data{nullptr, {}, true};
    }
  }
  if (data.immediate) {
    notify(std::move(data.data), std::move(data.remainingBytes));
  }
  for (auto& promise : promises) {
    promise.setValue();
  }
}

std::shared_ptr<UcxCpuRowPayload> UcxCpuRowOutputQueue::tryGetData(
    int destination) {
  std::shared_ptr<UcxCpuRowPayload> data;
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (destination < 0 || static_cast<size_t>(destination) >= queues_.size()) {
      return nullptr;
    }
    auto* queue = queues_[destination].get();
    if (!queue) {
      return nullptr;
    }
    data = queue->tryDequeueLocked();
    if (data) {
      updateStatsWithFreedLocked(data->numBytes, 1L, promises);
    }
  }
  for (auto& promise : promises) {
    promise.setValue();
  }
  return data;
}

void UcxCpuRowOutputQueue::noMoreData() {
  checkIfDone(true);
}

void UcxCpuRowOutputQueue::noMoreDrivers() {
  checkIfDone(false);
}

void UcxCpuRowOutputQueue::checkIfDone(bool oneDriverFinished) {
  std::vector<UcxCpuRowDataAvailable> finished;
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
      return;
    }
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
          clearArbitraryBufferLocked(promises);
          break;
        }
        bufferId = (bufferId + 1) % queues_.size();
      }
      if (queues_.empty()) {
        clearArbitraryBufferLocked(promises);
      }
    }
    for (auto& queue : queues_) {
      if (queue != nullptr) {
        queue->enqueueBack(nullptr);
        finished.push_back(queue->getAndClearNotify());
      }
    }
  }
  for (auto& notification : finished) {
    notification.notify();
  }
  for (auto& promise : promises) {
    promise.setValue();
  }
}

bool UcxCpuRowOutputQueue::enqueuePartitionedOutputLocked(
    int destination,
    std::shared_ptr<UcxCpuRowPayload> data,
    std::vector<UcxCpuRowDataAvailable>& dataAvailableCbs) {
  VELOX_DCHECK(dataAvailableCbs.empty());
  VELOX_CHECK_LT(destination, queues_.size());
  bool success = false;
  auto* queue = queues_[destination].get();
  if (queue != nullptr) {
    queue->enqueueBack(std::move(data));
    auto dataAvailable = queue->getAndClearNotify();
    recordDirectHandoffLocked(dataAvailable);
    dataAvailableCbs.emplace_back(std::move(dataAvailable));
    success = true;
  }
  return success;
}

void UcxCpuRowOutputQueue::enqueueBroadcastOutputLocked(
    std::shared_ptr<UcxCpuRowPayload> data,
    std::vector<UcxCpuRowDataAvailable>& dataAvailableCbs) {
  VELOX_DCHECK(dataAvailableCbs.empty());

  for (auto& queue : queues_) {
    if (queue != nullptr) {
      queue->enqueueBack(data);
      auto dataAvailable = queue->getAndClearNotify();
      recordDirectHandoffLocked(dataAvailable);
      dataAvailableCbs.emplace_back(std::move(dataAvailable));
    }
  }

  if (!noMoreQueues_) {
    dataToBroadcast_.emplace_back(std::move(data));
  }
}

void UcxCpuRowOutputQueue::enqueueArbitraryOutputLocked(
    std::shared_ptr<UcxCpuRowPayload> data,
    std::vector<UcxCpuRowDataAvailable>& dataAvailableCbs) {
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
        // The CPU row transport has no remaining-bytes prefetch contract.
        pending.remainingBytes.clear();
        recordDirectHandoffLocked(pending);
        dataAvailableCbs.push_back(std::move(pending));
      }
    }
    bufferId = (bufferId + 1) % queues_.size();
  }
  nextArbitraryLoadIndex_ = bufferId;
}

bool UcxCpuRowOutputQueue::isFinished() {
  std::lock_guard<std::mutex> l(mutex_);
  return isFinishedLocked();
}

bool UcxCpuRowOutputQueue::isFinishedLocked() {
  // For arbitrary, the coordinator lazily manages consumers and may never send
  // noMoreBufferIds, so only broadcast gates on noMoreQueues_.
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

std::optional<exec::TaskState> UcxCpuRowOutputQueue::taskState() {
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

void UcxCpuRowOutputQueue::updateOutputBuffers(
    int numBuffers,
    bool noMoreBuffers) {
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

void UcxCpuRowOutputQueue::deleteResults(int destination) {
  bool isFinished;
  UcxCpuRowDataAvailable dataAvailable;
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (destination >= queues_.size()) {
      return;
    }
    auto* queue = queues_[destination].get();
    if (queue == nullptr) {
      return;
    }
    int64_t bytes = queue->stats().bytesQueued;
    int64_t payloads = queue->stats().payloadsQueued;
    queue->deleteResults();
    dataAvailable = queue->getAndClearNotify();
    queue->finish();
    if (destination >= finishedBufferStats_.size()) {
      finishedBufferStats_.resize(destination + 1);
    }
    finishedBufferStats_[destination] = queue->stats().toOutputBufferStats();
    queues_[destination] = nullptr;
    isFinished = isFinishedLocked();
    updateStatsWithFreedLocked(bytes, payloads, promises);
    if (isFinished && kind_ == core::PartitionedOutputNode::Kind::kArbitrary) {
      clearArbitraryBufferLocked(promises);
    }
  }

  dataAvailable.notify();
  for (auto& promise : promises) {
    promise.setValue();
  }

  if (isFinished && task_) {
    task_->setAllOutputConsumed();
  }
}

void UcxCpuRowOutputQueue::terminate() {
  std::vector<UcxCpuRowDataAvailable> pendingCallbacks;
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (task_ && task_->isRunning()) {
      LOG(WARNING) << "UcxCpuRowOutputQueue::terminate() called while task "
                   << task_->taskId() << " is still running";
    }
    clearArbitraryBufferLocked(promises);
    // End-of-stream marker on every destination so consumers don't
    // wait forever after a producer-side abort.
    for (auto& queue : queues_) {
      if (queue != nullptr) {
        queue->enqueueBack(nullptr);
        pendingCallbacks.push_back(queue->getAndClearNotify());
      }
    }
    if (!promises_.empty()) {
      promises.reserve(promises.size() + promises_.size());
      for (auto& promise : promises_) {
        promises.push_back(std::move(promise));
      }
      promises_.clear();
    }
  }

  for (auto& callback : pendingCallbacks) {
    callback.notify();
  }
  for (auto& promise : promises) {
    promise.setValue();
  }
}

namespace {

int32_t countTopBuffers(
    const std::vector<exec::DestinationBuffer::Stats>& bufferStats,
    int64_t totalBytes) {
  if (totalBytes <= 0) {
    return 0;
  }

  std::vector<int64_t> bufferSizes;
  bufferSizes.reserve(bufferStats.size());
  for (const auto& stats : bufferStats) {
    bufferSizes.push_back(stats.bytesBuffered + stats.bytesSent);
  }

  std::sort(bufferSizes.begin(), bufferSizes.end(), std::greater<int64_t>());

  const auto limit = totalBytes * 0.8;
  int32_t numBuffers = 0;
  int64_t runningTotal = 0;
  for (auto size : bufferSizes) {
    if (size <= 0) {
      continue;
    }
    runningTotal += size;
    ++numBuffers;
    if (runningTotal >= limit) {
      break;
    }
  }
  return numBuffers;
}

} // namespace

exec::OutputBuffer::Stats UcxCpuRowOutputQueue::stats() {
  std::lock_guard<std::mutex> l(mutex_);
  std::vector<exec::DestinationBuffer::Stats> bufferStats;
  bufferStats.resize(queues_.size());
  for (size_t i = 0; i < queues_.size(); ++i) {
    auto* queue = queues_[i].get();
    if (queue != nullptr) {
      bufferStats[i] = queue->stats().toOutputBufferStats();
    } else if (i < finishedBufferStats_.size()) {
      bufferStats[i] = finishedBufferStats_[i];
    }
  }

  updateTotalQueuedBytesMsLocked();

  auto stats = exec::OutputBuffer::Stats(
      kind(),
      noMoreQueues_,
      atEnd_,
      isFinishedLocked(),
      queuedBytes_,
      queuedPayloads_,
      totalBytesSent_,
      totalRowsSent_,
      totalPayloadsSent_,
      getAverageQueueTimeMsLocked(),
      countTopBuffers(bufferStats, totalBytesSent_),
      bufferStats);
  return stats;
}

double UcxCpuRowOutputQueue::getUtilization() {
  std::lock_guard<std::mutex> l(mutex_);
  const auto highWaterMark = highWaterMarkLocked();
  return highWaterMark == 0 ? 0.0
                            : queuedBytes_ / static_cast<double>(highWaterMark);
}

bool UcxCpuRowOutputQueue::isOverutilized() {
  std::lock_guard<std::mutex> l(mutex_);
  const auto highWaterMark = highWaterMarkLocked();
  return atEnd_ || (highWaterMark > 0 && queuedBytes_ > (0.5 * highWaterMark));
}

void UcxCpuRowOutputQueue::updateStatsWithEnqueuedLocked(
    int64_t bytes,
    int64_t rows) {
  updateTotalQueuedBytesMsLocked();

  queuedBytes_ += bytes;
  queuedPayloads_++;

  totalBytesSent_ += bytes;
  totalRowsSent_ += rows;
  totalPayloadsSent_++;
}

void UcxCpuRowOutputQueue::updateStatsWithFreedLocked(
    int64_t bytes,
    int64_t numPayloads,
    std::vector<ContinuePromise>& promises) {
  updateTotalQueuedBytesMsLocked();

  VELOX_CHECK_GE(queuedBytes_, bytes, "Queued byte accounting underflow");
  VELOX_CHECK_GE(
      queuedPayloads_, numPayloads, "Queued payload accounting underflow");
  queuedBytes_ -= bytes;
  queuedPayloads_ -= numPayloads;

  maybeUnblockProducersLocked(promises);
}

void UcxCpuRowOutputQueue::recordDirectHandoffLocked(
    const UcxCpuRowDataAvailable& notification) {
  if (!notification.callback || notification.data == nullptr) {
    return;
  }

  pendingDirectHandoffBytes_ += notification.data->numBytes;
  pendingDirectHandoffPayloads_++;
}

void UcxCpuRowOutputQueue::acknowledgeDirectHandoffLocked(
    int64_t bytes,
    int64_t numPayloads) {
  VELOX_CHECK_GE(
      pendingDirectHandoffBytes_,
      bytes,
      "Direct handoff byte accounting underflow");
  VELOX_CHECK_GE(
      pendingDirectHandoffPayloads_,
      numPayloads,
      "Direct handoff payload accounting underflow");

  pendingDirectHandoffBytes_ -= bytes;
  pendingDirectHandoffPayloads_ -= numPayloads;
}

void UcxCpuRowOutputQueue::clearArbitraryBufferLocked(
    std::vector<ContinuePromise>& promises) {
  int64_t bytes = 0;
  int64_t payloads = 0;
  for (const auto& payload : arbitraryBuffer_) {
    if (payload != nullptr) {
      bytes += payload->numBytes;
      ++payloads;
    }
  }
  arbitraryBuffer_.clear();
  if (payloads > 0) {
    updateStatsWithFreedLocked(bytes, payloads, promises);
  }
}

void UcxCpuRowOutputQueue::maybeUnblockProducersLocked(
    std::vector<ContinuePromise>& promises) {
  const auto lowWaterMark = lowWaterMarkLocked();
  if (queuedBytes_ <= lowWaterMark && !promises_.empty()) {
    promises = std::move(promises_);
  }
}

void UcxCpuRowOutputQueue::updateTotalQueuedBytesMsLocked() {
  const auto nowMs = getCurrentTimeMs();
  if (queuedBytes_ > 0) {
    const auto deltaMs = nowMs - queueStartMs_;
    totalQueuedBytesMs_ += queuedBytes_ * deltaMs;
  }
  queueStartMs_ = nowMs;
}

int64_t UcxCpuRowOutputQueue::getAverageQueueTimeMsLocked() const {
  if (totalBytesSent_ > 0) {
    return totalQueuedBytesMs_ / totalBytesSent_;
  }
  return 0;
}

int64_t UcxCpuRowOutputQueue::backpressureFanoutLocked() const {
  if (kind_ != core::PartitionedOutputNode::Kind::kBroadcast) {
    return 1;
  }

  // Broadcast enqueues the same shared payload into each destination queue.
  // queuedBytes_ is still tracked per destination so dequeue/delete accounting
  // stays symmetric, but backpressure should be applied to physical retained
  // payload bytes, not to the fanout-inflated logical byte count.
  uint64_t fanout = 0;
  for (const auto& queue : queues_) {
    if (queue != nullptr) {
      ++fanout;
    }
  }
  return static_cast<int64_t>(std::max<uint64_t>(fanout, 1));
}

int64_t UcxCpuRowOutputQueue::highWaterMarkLocked() const {
  return saturatingMultiplyToInt64(
      maxSize_, static_cast<uint64_t>(backpressureFanoutLocked()));
}

int64_t UcxCpuRowOutputQueue::lowWaterMarkLocked() const {
  return saturatingMultiplyToInt64(
      continueSize_, static_cast<uint64_t>(backpressureFanoutLocked()));
}

std::string UcxCpuRowOutputQueue::toString() {
  std::stringstream out;
  std::lock_guard<std::mutex> l(mutex_);
  out << "[UcxCpuRowOutputQueue task="
      << (task_ ? task_->taskId() : "<uninitialized>") << " queues=";
  for (auto& q : queues_) {
    out << (q ? q->toString() : "<deleted>") << ", ";
  }
  out << " queuedBytes=" << queuedBytes_
      << ", queuedPayloads=" << queuedPayloads_
      << ", pendingDirectHandoffBytes=" << pendingDirectHandoffBytes_
      << ", pendingDirectHandoffPayloads=" << pendingDirectHandoffPayloads_
      << ", numFinished=" << numFinished_ << "/" << numDrivers_
      << ", atEnd=" << atEnd_ << ", noMoreQueues=" << noMoreQueues_ << "]";
  return out.str();
}

} // namespace facebook::velox::ucx_exchange
