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
#include "velox/common/memory/ByteStream.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/Task.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace facebook::velox::ucx_exchange {

namespace {

class UcxRetainedStreamArena final : public StreamArena {
 public:
  UcxRetainedStreamArena(
      memory::MemoryPool* pool,
      bool retainMemory,
      uint64_t maxRetainedBytes)
      : StreamArena(pool),
        pool_(pool),
        retainMemory_(retainMemory),
        maxRetainedBytes_(maxRetainedBytes) {}

  ~UcxRetainedStreamArena() override {
    releaseRetained();
  }

  void newRange(int64_t bytes, ByteRange* lastRange, ByteRange* range)
      override {
    newRangeImpl(bytes, lastRange, range);
  }

  size_t size() const override {
    return size_;
  }

  void clear() override {
    releaseRetained();
  }

  memory::MemoryPool* pool() const {
    return pool_;
  }

  void resetForReuse() {
    if (!retainMemory_ || retainedBytes_ > maxRetainedBytes_) {
      releaseRetained();
    }

    size_ = 0;
    allocationIndex_ = 0;
    countedAllocationIndex_ = 0;
    currentRun_ = 0;
    currentOffset_ = 0;
    largeAllocationIndex_ = 0;
  }

 private:
  static constexpr memory::MachinePageCount kAllocationQuantum{2};

  void newRangeImpl(int64_t bytes, ByteRange* /*lastRange*/, ByteRange* range) {
    VELOX_CHECK_GT(bytes, 0, "StreamArena::newRange can't be zero length");
    const memory::MachinePageCount numPages =
        memory::AllocationTraits::numPages(bytes);
    if (numPages > pool_->largestSizeClass()) {
      newLargeRange(numPages, range);
      return;
    }

    for (;;) {
      if (allocationIndex_ >= allocations_.size()) {
        memory::Allocation allocation;
        pool_->allocateNonContiguous(
            std::max(kAllocationQuantum, numPages), allocation);
        retainedBytes_ += allocation.byteSize();
        allocations_.push_back(std::move(allocation));
      }

      auto& allocation = allocations_[allocationIndex_];
      if (allocationIndex_ >= countedAllocationIndex_) {
        size_ += allocation.byteSize();
        countedAllocationIndex_ = allocationIndex_ + 1;
      }

      if (currentRun_ >= allocation.numRuns()) {
        ++allocationIndex_;
        currentRun_ = 0;
        currentOffset_ = 0;
        continue;
      }

      auto run = allocation.runAt(currentRun_);
      range->buffer = run.data() + currentOffset_;
      const int64_t availableBytes = run.numBytes() - currentOffset_;
      range->size = std::min<int64_t>(bytes, availableBytes);
      range->position = 0;
      currentOffset_ += range->size;
      VELOX_DCHECK_LE(currentOffset_, run.numBytes());
      if (currentOffset_ == run.numBytes()) {
        ++currentRun_;
        currentOffset_ = 0;
      }
      return;
    }
  }

  void newLargeRange(memory::MachinePageCount numPages, ByteRange* range) {
    const auto requestedBytes = memory::AllocationTraits::pageBytes(numPages);
    std::optional<size_t> selected;
    for (size_t i = largeAllocationIndex_; i < largeAllocations_.size(); ++i) {
      if (largeAllocations_[i].size() >= requestedBytes) {
        selected = i;
        break;
      }
    }

    if (selected.has_value()) {
      if (*selected != largeAllocationIndex_) {
        std::swap(
            largeAllocations_[*selected],
            largeAllocations_[largeAllocationIndex_]);
      }
    } else {
      memory::ContiguousAllocation allocation;
      pool_->allocateContiguous(numPages, allocation);
      retainedBytes_ += allocation.size();
      largeAllocations_.insert(
          largeAllocations_.begin() + largeAllocationIndex_,
          std::move(allocation));
    }

    auto& allocation = largeAllocations_[largeAllocationIndex_++];
    range->buffer = allocation.data();
    range->size = allocation.size();
    range->position = 0;
    size_ += range->size;
  }

  void releaseRetained() {
    allocations_.clear();
    largeAllocations_.clear();
    retainedBytes_ = 0;
    size_ = 0;
    allocationIndex_ = 0;
    countedAllocationIndex_ = 0;
    currentRun_ = 0;
    currentOffset_ = 0;
    largeAllocationIndex_ = 0;
  }

  memory::MemoryPool* const pool_;
  const bool retainMemory_;
  const uint64_t maxRetainedBytes_;
  std::vector<memory::Allocation> allocations_;
  std::vector<memory::ContiguousAllocation> largeAllocations_;
  uint64_t retainedBytes_{0};
  uint64_t size_{0};
  size_t allocationIndex_{0};
  size_t countedAllocationIndex_{0};
  int32_t currentRun_{0};
  int32_t currentOffset_{0};
  size_t largeAllocationIndex_{0};
};

constexpr uint64_t kDefaultMaxPageBytes = 16UL << 20; // 16 MiB

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
    LOG(WARNING) << "Ignoring invalid " << name << "=" << value;
    return std::nullopt;
  }

  return std::clamp<uint64_t>(parsed, minValue, maxValue);
}

uint64_t configuredMaxPageBytes() {
  static const auto kConfigured = readUint64Env(
      "VELOX_UCX_CPU_MAX_PAGE_BYTES",
      UcxCpuRowPartitionedOutput::kMinDestinationSize,
      64UL << 20);
  if (kConfigured.has_value()) {
    return *kConfigured;
  }
  return kDefaultMaxPageBytes;
}

int32_t configuredTargetNumRows() {
  static const auto kConfigured =
      readUint64Env("VELOX_UCX_CPU_TARGET_ROWS", 1, 16'000'000);
  if (kConfigured.has_value()) {
    return static_cast<int32_t>(*kConfigured);
  }
  return UcxCpuRowPartitionedOutput::kTargetNumRows;
}

bool retainStreamArenaEnabled() {
  static const bool kEnabled = [] {
    const char* value = std::getenv("VELOX_UCX_CPU_RETAIN_STREAM_ARENA");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return kEnabled;
}

uint64_t retainedStreamArenaMaxBytes() {
  static const auto kConfigured = readUint64Env(
      "VELOX_UCX_CPU_RETAIN_STREAM_ARENA_MAX_BYTES",
      memory::AllocationTraits::kPageSize,
      1UL << 30);
  if (kConfigured.has_value()) {
    return *kConfigured;
  }
  return 64UL << 20;
}

} // namespace

class UcxVectorStreamGroup final {
 public:
  UcxVectorStreamGroup(
      memory::MemoryPool* pool,
      VectorSerde* serde,
      bool retainMemory,
      uint64_t maxRetainedBytes)
      : serde_(serde != nullptr ? serde : getVectorSerde()) {
    if (retainMemory) {
      retainedArena_ = std::make_unique<UcxRetainedStreamArena>(
          pool, retainMemory, maxRetainedBytes);
    } else {
      stock_ = std::make_unique<VectorStreamGroup>(pool, serde_);
    }
  }

  void createStreamTree(
      RowTypePtr type,
      int32_t numRows,
      const VectorSerde::Options* options = nullptr) {
    if (stock_ != nullptr) {
      stock_->createStreamTree(std::move(type), numRows, options);
      return;
    }
    serializer_ = serde_->createIterativeSerializer(
        type, numRows, retainedArena_.get(), options);
  }

  void append(
      const RowVectorPtr& vector,
      const folly::Range<const IndexRange*>& ranges,
      Scratch& scratch) {
    if (stock_ != nullptr) {
      stock_->append(vector, ranges, scratch);
      return;
    }
    serializer_->append(vector, ranges, scratch);
  }

  void append(
      const RowVectorPtr& vector,
      const folly::Range<const vector_size_t*>& rows,
      Scratch& scratch) {
    if (stock_ != nullptr) {
      stock_->append(vector, rows, scratch);
      return;
    }
    serializer_->append(vector, rows, scratch);
  }

  void flush(OutputStream* stream) {
    if (stock_ != nullptr) {
      stock_->flush(stream);
      return;
    }
    serializer_->flush(stream);
  }

  size_t size() const {
    return stock_ != nullptr ? stock_->size() : retainedArena_->size();
  }

  memory::MemoryPool* pool() const {
    return stock_ != nullptr ? stock_->pool() : retainedArena_->pool();
  }

  void resetForNextStreamTree() {
    if (stock_ != nullptr) {
      stock_->clear();
      return;
    }
    // Destroy the serializer before rewinding the arena. Serializer streams
    // hold raw ByteRanges into the arena; no old stream state is reused after
    // flush.
    serializer_.reset();
    retainedArena_->resetForReuse();
  }

 private:
  std::unique_ptr<VectorStreamGroup> stock_;
  std::unique_ptr<UcxRetainedStreamArena> retainedArena_;
  VectorSerde* serde_{nullptr};
  std::unique_ptr<IterativeVectorSerializer> serializer_;
};

using exec::BlockingReason;

UcxCpuRowPartitionedOutput::Destination::~Destination() = default;

UcxCpuRowPartitionedOutput::Destination::Destination(
    std::string taskId,
    int destination,
    VectorSerde* serde,
    VectorSerde::Options* serdeOptions,
    memory::MemoryPool* pool,
    bool eagerFlush,
    int32_t targetNumRowsBase,
    std::shared_ptr<UcxCpuRowOutputQueueManager> queueMgr,
    std::function<void(uint64_t bytes, uint64_t rows)> recordEnqueued)
    : taskId_(std::move(taskId)),
      destination_(destination),
      serde_(serde),
      serdeOptions_(serdeOptions),
      pool_(pool),
      eagerFlush_(eagerFlush),
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
      current_ = std::make_unique<UcxVectorStreamGroup>(
          pool_,
          serde_,
          retainStreamArenaEnabled(),
          retainedStreamArenaMaxBytes());
    }
    const auto rowType = asRowType(output->type());
    current_->createStreamTree(rowType, rowsInCurrent_, serdeOptions_);
    needsStreamTreeRecreation_ = false;
  }
}

void UcxCpuRowPartitionedOutput::Destination::clearVectorStreamGroup() {
  current_->resetForNextStreamTree();
  // Serializer streams hold raw ByteRanges into the arena. After flush, drop
  // the serializer and create a fresh stream tree before the next append.
  // resetForNextStreamTree() may retain the arena allocation behind an opt-in
  // flag, but no old stream state or payload length is reused.
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
      maxPageBytes_(configuredMaxPageBytes()),
      targetNumRowsBase_(configuredTargetNumRows()),
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
    const auto singlePartition =
        partitionFunction_->partition(*input_, partitions_);
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

void UcxCpuRowPartitionedOutput::noMoreInput() {
  Operator::noMoreInput();
  finishOutput();
}

void UcxCpuRowPartitionedOutput::finishOutput() {
  if (finished_) {
    return;
  }

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
    finishOutput();
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
