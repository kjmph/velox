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
#include "velox/experimental/ucx-exchange/UcxCpuRowPartitionedOutput.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/Task.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowShm.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>

namespace facebook::velox::ucx_exchange {
namespace {

constexpr uint64_t kDefaultMaxPageBytes = 1UL << 20; // 1 MiB
constexpr uint64_t kDirectShmDefaultMaxPageBytes = 8UL << 20; // 8 MiB
constexpr uint64_t kDefaultShmMinPayloadBytes = 64UL << 10; // 64 KiB
constexpr int32_t kDirectShmDefaultTargetNumRows = 256'000;

std::optional<uint64_t>
readUint64Env(const char* name, uint64_t minValue, uint64_t maxValue) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return std::nullopt;
  }

  errno = 0;
  char* end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || errno != 0) {
    VLOG(1) << "Ignoring invalid " << name << "=" << value;
    return std::nullopt;
  }

  return std::clamp<uint64_t>(parsed, minValue, maxValue);
}

uint64_t configuredMaxPageBytes(bool directShmEnabled) {
  static const auto kConfigured = readUint64Env(
      "VELOX_UCX_CPU_MAX_PAGE_BYTES",
      UcxCpuRowPartitionedOutput::kMinDestinationSize,
      64UL << 20);
  if (kConfigured.has_value()) {
    return *kConfigured;
  }
  return directShmEnabled ? kDirectShmDefaultMaxPageBytes
                          : kDefaultMaxPageBytes;
}

int32_t configuredTargetNumRows(bool directShmEnabled) {
  static const auto kConfigured =
      readUint64Env("VELOX_UCX_CPU_TARGET_ROWS", 1, 16'000'000);
  if (kConfigured.has_value()) {
    return static_cast<int32_t>(*kConfigured);
  }
  return directShmEnabled ? kDirectShmDefaultTargetNumRows
                          : UcxCpuRowPartitionedOutput::kTargetNumRows;
}

uint64_t configuredShmMinPayloadBytes() {
  static const auto kConfigured =
      readUint64Env("VELOX_UCX_CPU_SHM_MIN_PAYLOAD_BYTES", 0, 64UL << 20);
  return kConfigured.value_or(kDefaultShmMinPayloadBytes);
}

bool useDirectShmOutput(bool isBroadcast) {
  return !isBroadcast && ucxCpuRowShmDirectTxEnabled();
}

bool useBroadcastSlotPoolOutput() {
  static const bool kEnabled =
      readUint64Env("VELOX_UCX_CPU_SHM_BROADCAST_SLOT_POOL", 0, 1)
          .value_or(0) == 1;
  return kEnabled;
}

bool useSlotPoolOutput(bool isBroadcast) {
  return ucxCpuRowShmSlotPoolEnabled() &&
      (!isBroadcast || useBroadcastSlotPoolOutput());
}

bool useShmSizedOutputPages(bool isBroadcast) {
  return useDirectShmOutput(isBroadcast) || useSlotPoolOutput(isBroadcast);
}

class DirectShmOutputStream final : public OutputStream {
 public:
  explicit DirectShmOutputStream(size_t initialSize)
      : segment_(createUcxCpuRowShmSegment(initialSize)),
        capacity_(segment_ ? initialSize : 0) {
    if (segment_) {
      segment_->unlinkOnDestroy = true;
    }
  }

  bool valid() const {
    return segment_ != nullptr;
  }

  void write(const char* s, std::streamsize count) override {
    VELOX_CHECK_GE(count, 0);
    const auto offset = static_cast<size_t>(position_);
    VELOX_CHECK_LE(
        offset,
        capacity_,
        "CPU SHM direct-TX stream offset exceeded backing segment");
    const auto bytes = static_cast<size_t>(count);
    VELOX_CHECK_LE(
        bytes,
        std::numeric_limits<size_t>::max() - offset,
        "CPU SHM direct-TX stream size overflow");
    ensureCapacity(offset + bytes);
    std::memcpy(segment_->data() + offset, s, bytes);
    position_ += static_cast<std::streamoff>(bytes);
    highWater_ = std::max(highWater_, offset + bytes);
    if (listener_) {
      listener_->onWrite(s, count);
    }
  }

  std::streampos tellp() const override {
    return position_;
  }

  void seekp(std::streampos pos) override {
    const auto offset = static_cast<std::streamoff>(pos);
    VELOX_CHECK_GE(offset, 0);
    ensureCapacity(static_cast<size_t>(offset));
    position_ = pos;
  }

  std::shared_ptr<UcxCpuRowShmSegment> segment() const {
    return segment_;
  }

  size_t offset() const {
    return 0;
  }

  std::unique_ptr<folly::IOBuf> getIOBuf(size_t size) const {
    VELOX_CHECK_NOT_NULL(segment_);
    VELOX_CHECK_LE(
        size, capacity_, "CPU SHM direct-TX IOBuf exceeds backing segment");
    auto segmentHolder =
        std::make_unique<std::shared_ptr<UcxCpuRowShmSegment>>(segment_);
    return folly::IOBuf::takeOwnership(
        segment_->data(),
        size,
        [](void* /*buf*/, void* userData) {
          delete static_cast<std::shared_ptr<UcxCpuRowShmSegment>*>(userData);
        },
        segmentHolder.release());
  }

 private:
  void ensureCapacity(size_t required) {
    if (required <= capacity_) {
      return;
    }

    size_t newSize = std::max<size_t>(capacity_, 1);
    while (newSize < required) {
      VELOX_CHECK_LE(
          newSize,
          std::numeric_limits<size_t>::max() / 2,
          "CPU SHM direct-TX stream size overflow");
      newSize *= 2;
    }

    auto newSegment = createUcxCpuRowShmSegment(newSize);
    VELOX_CHECK_NOT_NULL(newSegment, "Failed to grow CPU SHM direct-TX stream");
    newSegment->unlinkOnDestroy = true;
    if (highWater_ > 0) {
      VELOX_CHECK_NOT_NULL(segment_);
      std::memcpy(newSegment->data(), segment_->data(), highWater_);
    }
    segment_ = std::move(newSegment);
    capacity_ = newSize;
  }

  std::shared_ptr<UcxCpuRowShmSegment> segment_;
  size_t capacity_;
  std::streampos position_{0};
  size_t highWater_{0};
};

class SlotOutputStream final : public OutputStream {
 public:
  explicit SlotOutputStream(UcxCpuRowShmSlotLease lease)
      : pool_(std::move(lease.pool)),
        slotId_(lease.slot.id),
        data_(lease.slot.data),
        capacity_(pool_->slotSize()) {}

  ~SlotOutputStream() override {
    if (releaseOnDestroy_ && pool_) {
      pool_->release(slotId_);
    }
  }

  void write(const char* s, std::streamsize count) override {
    VELOX_CHECK_GE(count, 0);
    const auto offset = static_cast<size_t>(position_);
    VELOX_CHECK_LE(
        offset,
        capacity_,
        "CPU SHM slot-pool stream offset exceeded backing slot");
    const auto bytes = static_cast<size_t>(count);
    VELOX_CHECK_LE(
        bytes,
        capacity_ - offset,
        "CPU SHM slot-pool stream exceeded backing slot");
    std::memcpy(data_ + offset, s, bytes);
    position_ += static_cast<std::streamoff>(bytes);
  }

  std::streampos tellp() const override {
    return position_;
  }

  void seekp(std::streampos pos) override {
    const auto offset = static_cast<std::streamoff>(pos);
    VELOX_CHECK_GE(offset, 0);
    VELOX_CHECK_LE(
        static_cast<size_t>(offset),
        capacity_,
        "CPU SHM slot-pool seek exceeded backing slot");
    position_ = pos;
  }

  std::shared_ptr<UcxCpuRowShmSlotPool> pool() const {
    return pool_;
  }

  uint32_t slotId() const {
    return slotId_;
  }

  void markReady() {
    pool_->markReady(slotId_);
    releaseOnDestroy_ = false;
  }

  std::unique_ptr<folly::IOBuf> getIOBuf(size_t size) const {
    VELOX_CHECK_LE(size, capacity_, "CPU SHM slot-pool IOBuf exceeds slot");
    auto poolHolder =
        std::make_unique<std::shared_ptr<UcxCpuRowShmSlotPool>>(pool_);
    auto* rawData = data_;
    auto* rawHolder = poolHolder.get();
    auto buf = folly::IOBuf::takeOwnership(
        rawData,
        size,
        [](void* /*buf*/, void* userData) {
          delete static_cast<std::shared_ptr<UcxCpuRowShmSlotPool>*>(userData);
        },
        rawHolder);
    poolHolder.release();
    return buf;
  }

 private:
  std::shared_ptr<UcxCpuRowShmSlotPool> pool_;
  uint32_t slotId_;
  uint8_t* data_;
  size_t capacity_;
  std::streampos position_{0};
  bool releaseOnDestroy_{true};
};

} // namespace

using exec::BlockingReason;

UcxCpuRowPartitionedOutput::Destination::~Destination() = default;

UcxCpuRowPartitionedOutput::Destination::Destination(
    std::string taskId,
    int destination,
    VectorSerde* serde,
    VectorSerde::Options* serdeOptions,
    memory::MemoryPool* pool,
    bool eagerFlush,
    bool directShmEnabled,
    bool slotPoolEnabled,
    int32_t targetNumRowsBase,
    std::shared_ptr<UcxCpuRowOutputQueueManager> queueMgr,
    std::function<void(uint64_t bytes, uint64_t rows)> recordEnqueued)
    : taskId_(std::move(taskId)),
      destination_(destination),
      serde_(serde),
      serdeOptions_(serdeOptions),
      pool_(pool),
      eagerFlush_(eagerFlush),
      directShmEnabled_(directShmEnabled),
      slotPoolEnabled_(slotPoolEnabled),
      targetNumRowsBase_(targetNumRowsBase),
      queueMgr_(std::move(queueMgr)),
      recordEnqueued_(std::move(recordEnqueued)),
      rows_(raw_vector<vector_size_t>(pool)) {
  setTargetSizePct();
}

void UcxCpuRowPartitionedOutput::Destination::createVectorStreamGroup(
    const RowVectorPtr& output) {
  if (current_ == nullptr || needsStreamTreeRecreation_) {
    if (current_ == nullptr) {
      current_ = std::make_unique<VectorStreamGroup>(pool_, serde_);
    }
    const auto rowType = asRowType(output->type());
    current_->createStreamTree(rowType, rowsInCurrent_, serdeOptions_);
    needsStreamTreeRecreation_ = false;
  }
}

void UcxCpuRowPartitionedOutput::Destination::clearVectorStreamGroup() {
  current_->clear();
  // The underlying StreamArena's memory is recycled by clear(); the
  // serializer holds raw pointers into it, so we must call
  // createStreamTree() again before the next append. Mirrors the
  // standard PartitionedOutput's same-name fix.
  needsStreamTreeRecreation_ = true;
}

BlockingReason UcxCpuRowPartitionedOutput::Destination::advance(
    uint64_t maxBytes,
    const std::vector<vector_size_t>& sizes,
    const RowVectorPtr& output,
    bool* atEnd,
    ContinueFuture* future,
    Scratch& scratch) {
  if (rowIdx_ >= rows_.size()) {
    *atEnd = true;
    return BlockingReason::kNotBlocked;
  }

  const auto firstRow = rowIdx_;
  const uint32_t adjustedMaxBytes = (maxBytes * targetSizePct_) / 100;
  if (bytesInCurrent_ >= adjustedMaxBytes) {
    return flush(future);
  }

  bool shouldFlush = false;
  while (rowIdx_ < rows_.size() && !shouldFlush) {
    bytesInCurrent_ += sizes[rows_[rowIdx_]];
    ++rowIdx_;
    ++rowsInCurrent_;
    shouldFlush =
        bytesInCurrent_ >= adjustedMaxBytes || rowsInCurrent_ >= targetNumRows_;
  }

  createVectorStreamGroup(output);

  const auto rowsRange = folly::Range(&rows_[firstRow], rowIdx_ - firstRow);
  // The CPU UCX path is Presto-serde only; we don't bother with the
  // CompactRow / UnsafeRow branches the standard Destination supports.
  // Prestissimo always uses Presto serde for shuffle, and the cudf UCX
  // exchange already constrains itself to a single serde format.
  VELOX_CHECK_EQ(serde_->kind(), "Presto");
  current_->append(output, rowsRange, scratch);

  if (rowIdx_ == rows_.size()) {
    *atEnd = true;
  }
  if (shouldFlush || (eagerFlush_ && rowsInCurrent_ > 0)) {
    return flush(future);
  }
  return BlockingReason::kNotBlocked;
}

BlockingReason UcxCpuRowPartitionedOutput::Destination::flush(
    ContinueFuture* future) {
  if (!current_ || rowsInCurrent_ == 0) {
    return BlockingReason::kNotBlocked;
  }

  constexpr int32_t kMinMessageSize = 128;
  const auto initialSize =
      static_cast<size_t>(std::max<int64_t>(kMinMessageSize, current_->size()));
  const int64_t flushedRows = rowsInCurrent_;

  auto enqueuePayload = [&](std::unique_ptr<UcxCpuRowPayload> payload,
                            int64_t flushedBytes) -> BlockingReason {
    bytesInCurrent_ = 0;
    rowsInCurrent_ = 0;
    setTargetSizePct();

    payload->numRows = static_cast<int32_t>(flushedRows);
    payload->numBytes = flushedBytes;

    queueMgr_->enqueue(taskId_, destination_, std::move(payload), flushedRows);
    recordEnqueued_(flushedBytes, flushedRows);

    if (future == nullptr) {
      return BlockingReason::kNotBlocked;
    }
    const bool blocked = queueMgr_->checkBlocked(taskId_, future);
    return blocked ? BlockingReason::kWaitForConsumer
                   : BlockingReason::kNotBlocked;
  };

  const bool useShmForPayload = initialSize >= configuredShmMinPayloadBytes();

  if (slotPoolEnabled_ && useShmForPayload) {
    auto lease = queueMgr_->tryAcquireSlot(taskId_, destination_, initialSize);
    if (lease.has_value()) {
      SlotOutputStream stream(std::move(*lease));
      current_->flush(&stream);
      clearVectorStreamGroup();

      const int64_t flushedBytes = stream.tellp();
      auto payload = std::make_unique<UcxCpuRowPayload>();
      payload->shmSlotPool = stream.pool();
      payload->shmSlotId = stream.slotId();
      payload->releaseShmSlotOnDestroy = true;
      payload->data = stream.getIOBuf(static_cast<size_t>(flushedBytes));
      stream.markReady();
      return enqueuePayload(std::move(payload), flushedBytes);
    }
  }

  if (directShmEnabled_ && useShmForPayload) {
    DirectShmOutputStream stream(initialSize);
    if (stream.valid()) {
      current_->flush(&stream);
      clearVectorStreamGroup();

      const int64_t flushedBytes = stream.tellp();
      auto payload = std::make_unique<UcxCpuRowPayload>();
      payload->shmSegment = stream.segment();
      payload->shmOffset = stream.offset();
      payload->data = stream.getIOBuf(static_cast<size_t>(flushedBytes));
      return enqueuePayload(std::move(payload), flushedBytes);
    }

    VLOG(1) << "Failed to create CPU SHM direct-TX stream; falling back to "
               "IOBuf serialization";
  }

  IOBufOutputStream stream(
      *current_->pool(),
      /*listener=*/nullptr,
      initialSize);

  current_->flush(&stream);
  clearVectorStreamGroup();

  const int64_t flushedBytes = stream.tellp();

  auto payload = std::make_unique<UcxCpuRowPayload>();
  payload->data = stream.getIOBuf();
  return enqueuePayload(std::move(payload), flushedBytes);
}

UcxCpuRowPartitionedOutput::UcxCpuRowPartitionedOutput(
    int32_t operatorId,
    exec::DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
    bool eagerFlush)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "UcxCpuRowPartitionedOutput"),
      keyChannels_(exec::toChannels(planNode->inputType(), planNode->keys())),
      numDestinations_(planNode->numPartitions()),
      replicateNullsAndAny_(planNode->isReplicateNullsAndAny()),
      partitionFunction_(
          numDestinations_ == 1 ? nullptr
                                : planNode->partitionFunctionSpec().create(
                                      numDestinations_,
                                      /*localExchange=*/false)),
      outputChannels_(exec::calculateOutputChannels(
          planNode->inputType(),
          planNode->outputType(),
          planNode->outputType())),
      queueMgr_(UcxCpuRowOutputQueueManager::getInstanceRef()),
      maxBufferedBytes_(ctx->task->queryCtx()
                            ->queryConfig()
                            .maxPartitionedOutputBufferSize()),
      // Force eager flush off for the UCX path. The query-config
      // partitionedOutputEagerFlush flag is tuned for the HTTP /
      // OutputBufferManager world where many tiny pages get bundled
      // downstream by the buffer manager. Our UCX server bundles, but
      // only sees what's already in the queue when sendData() runs;
      // upstream eager-flushing fragments per-driver output before it
      // can be bundled, so honoring the config flag here works against
      // throughput.
      eagerFlush_(false),
      directShmEnabled_(useDirectShmOutput(planNode->isBroadcast())),
      slotPoolEnabled_(useSlotPoolOutput(planNode->isBroadcast())),
      maxPageBytes_(configuredMaxPageBytes(
          useShmSizedOutputPages(planNode->isBroadcast()))),
      targetNumRowsBase_(configuredTargetNumRows(
          useShmSizedOutputPages(planNode->isBroadcast()))),
      serde_(getNamedVectorSerde(planNode->serdeKind())),
      serdeOptions_(exec::getVectorSerdeOptions(
          common::stringToCompressionKind(operatorCtx_->driverCtx()
                                              ->queryConfig()
                                              .shuffleCompressionKind()),
          planNode->serdeKind(),
          /*minCompressionRatio=*/0.8,
          operatorCtx_->driverCtx()
              ->queryConfig()
              .minShuffleCompressionPageSizeBytes())) {
  if (!planNode->isPartitioned()) {
    VELOX_USER_CHECK_EQ(numDestinations_, 1);
  }
  if (numDestinations_ == 1) {
    VELOX_USER_CHECK(keyChannels_.empty());
    VELOX_USER_CHECK_NULL(partitionFunction_);
  }
}

void UcxCpuRowPartitionedOutput::initializeInput(RowVectorPtr input) {
  input_ = std::move(input);
  if (outputType_->size() == 0) {
    output_ = std::make_shared<RowVector>(
        input_->pool(),
        outputType_,
        nullptr,
        input_->size(),
        std::vector<VectorPtr>{});
  } else if (outputChannels_.empty()) {
    output_ = input_;
  } else {
    std::vector<VectorPtr> outputColumns;
    outputColumns.reserve(outputChannels_.size());
    for (auto i : outputChannels_) {
      outputColumns.push_back(input_->childAt(i));
    }
    output_ = std::make_shared<RowVector>(
        input_->pool(), outputType_, nullptr, input_->size(), outputColumns);
  }

  for (auto i = 0; i < output_->childrenSize(); ++i) {
    output_->childAt(i)->loadedVector();
  }
}

void UcxCpuRowPartitionedOutput::initializeDestinations() {
  if (destinations_.empty()) {
    auto taskId = operatorCtx_->taskId();
    for (int i = 0; i < numDestinations_; ++i) {
      destinations_.push_back(std::make_unique<Destination>(
          taskId,
          i,
          serde_,
          serdeOptions_.get(),
          pool(),
          eagerFlush_,
          directShmEnabled_,
          slotPoolEnabled_,
          targetNumRowsBase_,
          queueMgr_,
          [this](uint64_t bytes, uint64_t rows) {
            auto lockedStats = stats_.wlock();
            lockedStats->addOutputVector(bytes, rows);
          }));
    }
  }
}

void UcxCpuRowPartitionedOutput::initializeSizeBuffers() {
  const auto numInput = input_->size();
  if (numInput > rowSize_.size()) {
    rowSize_.resize(numInput);
    sizePointers_.resize(numInput);
    for (vector_size_t i = 0; i < numInput; ++i) {
      sizePointers_[i] = &rowSize_[i];
    }
  }
}

void UcxCpuRowPartitionedOutput::estimateRowSizes() {
  const auto numInput = input_->size();
  std::fill(rowSize_.begin(), rowSize_.end(), 0);
  raw_vector<vector_size_t> storage(pool());
  const auto numbers = iota(numInput, storage);
  const auto rows = folly::Range(numbers, numInput);
  serde_->estimateSerializedSize(
      output_.get(), rows, sizePointers_.data(), scratch_);
}

void UcxCpuRowPartitionedOutput::collectNullRows() {
  auto size = input_->size();
  rows_.resize(size);
  rows_.setAll();

  nullRows_.resize(size);
  nullRows_.clearAll();

  decodedVectors_.resize(keyChannels_.size());

  for (size_t keyChannelIndex = 0; keyChannelIndex < keyChannels_.size();
       ++keyChannelIndex) {
    column_index_t keyChannel = keyChannels_[keyChannelIndex];
    if (keyChannel == kConstantChannel) {
      continue;
    }
    auto& keyVector = input_->childAt(keyChannel);
    if (keyVector->mayHaveNulls()) {
      DecodedVector& decodedVector = decodedVectors_[keyChannelIndex];
      decodedVector.decode(*keyVector, rows_);
      if (auto* rawNulls = decodedVector.nulls(&rows_)) {
        bits::orWithNegatedBits(
            nullRows_.asMutableRange().bits(), rawNulls, 0, size);
      }
    }
  }
  nullRows_.updateBounds();
}

void UcxCpuRowPartitionedOutput::addInput(RowVectorPtr input) {
  initializeInput(std::move(input));
  initializeDestinations();
  initializeSizeBuffers();
  estimateRowSizes();

  for (auto& destination : destinations_) {
    destination->beginBatch();
  }

  auto numInput = input_->size();
  if (numDestinations_ == 1) {
    destinations_[0]->addRows(IndexRange{0, numInput});
  } else {
    auto singlePartition = partitionFunction_->partition(*input_, partitions_);
    if (replicateNullsAndAny_) {
      collectNullRows();
      vector_size_t start = 0;
      if (!replicatedAny_) {
        for (auto& destination : destinations_) {
          destination->addRow(0);
        }
        replicatedAny_ = true;
        start = 1;
      }
      for (auto i = start; i < numInput; ++i) {
        if (nullRows_.isValid(i)) {
          for (auto& destination : destinations_) {
            destination->addRow(i);
          }
        } else {
          if (singlePartition.has_value()) {
            destinations_[singlePartition.value()]->addRow(i);
          } else {
            destinations_[partitions_[i]]->addRow(i);
          }
        }
      }
    } else {
      if (singlePartition.has_value()) {
        destinations_[singlePartition.value()]->addRows(
            IndexRange{0, numInput});
      } else {
        for (vector_size_t i = 0; i < numInput; ++i) {
          destinations_[partitions_[i]]->addRow(i);
        }
      }
    }
  }
}

RowVectorPtr UcxCpuRowPartitionedOutput::getOutput() {
  if (finished_) {
    return nullptr;
  }

  blockingReason_ = BlockingReason::kNotBlocked;
  Destination* blockedDestination = nullptr;

  const uint64_t maxPageSize = std::max<uint64_t>(
      kMinDestinationSize,
      std::min<uint64_t>(maxPageBytes_, maxBufferedBytes_ / numDestinations_));

  bool workLeft;
  do {
    workLeft = false;
    for (auto& destination : destinations_) {
      bool atEnd = false;
      blockingReason_ = destination->advance(
          maxPageSize, rowSize_, output_, &atEnd, &future_, scratch_);
      if (blockingReason_ != BlockingReason::kNotBlocked) {
        blockedDestination = destination.get();
        workLeft = false;
        break;
      }
      if (!atEnd) {
        workLeft = true;
      }
    }
  } while (workLeft);

  if (blockedDestination) {
    // Even though one destination is blocked, drain the other
    // destinations' partial buffers so they're not left stranded for
    // the duration of the block. Skip ones too small to be worth
    // sending. Mirrors PartitionedOutput.
    for (auto& destination : destinations_) {
      if (destination.get() == blockedDestination ||
          destination->serializedBytes() < kMinDestinationSize) {
        continue;
      }
      destination->flush(/*future=*/nullptr);
    }
    return nullptr;
  }

  if (noMoreInput_) {
    for (auto& destination : destinations_) {
      if (destination->isFinished()) {
        continue;
      }
      destination->flush(/*future=*/nullptr);
      destination->setFinished();
    }
    queueMgr_->noMoreData(operatorCtx_->task()->taskId());
    finished_ = true;
  }

  input_ = nullptr;
  return nullptr;
}

exec::BlockingReason UcxCpuRowPartitionedOutput::isBlocked(
    ContinueFuture* future) {
  if (blockingReason_ != BlockingReason::kNotBlocked) {
    *future = std::move(future_);
    blockingReason_ = BlockingReason::kNotBlocked;
    return BlockingReason::kWaitForConsumer;
  }
  return BlockingReason::kNotBlocked;
}

bool UcxCpuRowPartitionedOutput::isFinished() {
  return finished_;
}

void UcxCpuRowPartitionedOutput::close() {
  Operator::close();
  destinations_.clear();
}

} // namespace facebook::velox::ucx_exchange
