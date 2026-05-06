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
#pragma once

#include <folly/Random.h>
#include "velox/common/memory/ByteStream.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowOutputQueueManager.h"
#include "velox/vector/VectorStream.h"

/// CPU RowVector mirror of UcxPartitionedOutput. Forks the partitioning
/// logic from velox::exec::PartitionedOutput because the standard
/// `detail::Destination` flushes through `OutputBufferManager` and
/// there's no public hook for redirecting the enqueue target.
///
/// What we keep from the standard operator:
///   - keyChannels / partitionFunction parsed straight off the planNode
///   - per-destination buffering with an embedded VectorStreamGroup
///   - row-routing scratch via `partitions_` (one entry per input row)
/// What we replace:
///   - the OutputBufferManager flush target: we enqueue serialized
///     IOBuf chains into UcxCpuRowOutputQueueManager so that
///     UcxCpuRowExchangeServer can pick them up

namespace facebook::velox::ucx_exchange {

class UcxCpuRowPartitionedOutput : public exec::Operator {
 public:
  // Mirrors PartitionedOutput::kMinDestinationSize.
  static constexpr uint64_t kMinDestinationSize = 60 * 1024;
  // Keep row count from capping pages before the byte target has a chance to
  // fire on wide, high-throughput CPU exchange runs.
  static constexpr int32_t kTargetNumRows = 512'000;

  UcxCpuRowPartitionedOutput(
      int32_t operatorId,
      exec::DriverCtx* ctx,
      const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
      bool eagerFlush);

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  bool needsInput() const override {
    return true;
  }

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  void close() override;

 private:
  /// Per-destination buffer: accumulates row indices, runs the
  /// PrestoSerializer-backed VectorStreamGroup, and flushes an IOBuf
  /// chain into UcxCpuRowOutputQueueManager when full or finishing.
  struct Destination {
    Destination(
        std::string taskId,
        int destination,
        VectorSerde* serde,
        VectorSerde::Options* serdeOptions,
        memory::MemoryPool* pool,
        bool eagerFlush,
        int32_t targetNumRowsBase,
        std::shared_ptr<UcxCpuRowOutputQueueManager> queueMgr,
        std::function<void(uint64_t bytes, uint64_t rows)> recordEnqueued);
    ~Destination();

    void beginBatch() {
      rows_.clear();
      rowIdx_ = 0;
    }

    void addRow(vector_size_t row) {
      rows_.push_back(row);
    }

    void addRows(const IndexRange& rows) {
      for (auto i = 0; i < rows.size; ++i) {
        rows_.push_back(rows.begin + i);
      }
    }

    /// Serializes rows from `output` (selected via `rows_`) until
    /// either `maxBytes` has been buffered or all selected rows have
    /// been consumed. Sets `*atEnd` when the row set is fully drained.
    /// Calls flush() once the byte/row threshold is hit.
    exec::BlockingReason advance(
        uint64_t maxBytes,
        const std::vector<vector_size_t>& sizes,
        const RowVectorPtr& output,
        bool* atEnd,
        ContinueFuture* future,
        Scratch& scratch);

    /// Materializes the buffered VectorStreamGroup as an IOBuf chain,
    /// hands it to the queue manager, and returns whether the producer
    /// must block.
    exec::BlockingReason flush(ContinueFuture* future);

    bool isFinished() const {
      return finished_;
    }

    void setFinished() {
      finished_ = true;
    }

    uint64_t serializedBytes() const {
      return bytesInCurrent_;
    }

   private:
    void setTargetSizePct() {
      targetSizePct_ = 70 + (folly::Random::rand32(rng_) % 50);
      targetNumRows_ = (targetNumRowsBase_ * targetSizePct_) / 100;
    }

    void createVectorStreamGroup(const RowVectorPtr& output);
    void clearVectorStreamGroup();

    const std::string taskId_;
    const int destination_;
    VectorSerde* const serde_;
    VectorSerde::Options* const serdeOptions_;
    memory::MemoryPool* const pool_;
    const bool eagerFlush_;
    const int32_t targetNumRowsBase_;
    const std::shared_ptr<UcxCpuRowOutputQueueManager> queueMgr_;
    const std::function<void(uint64_t bytes, uint64_t rows)> recordEnqueued_;

    uint64_t bytesInCurrent_{0};
    vector_size_t rowsInCurrent_{0};
    raw_vector<vector_size_t> rows_;
    vector_size_t rowIdx_{0};

    std::unique_ptr<VectorStreamGroup> current_;
    bool needsStreamTreeRecreation_{false};
    bool finished_{false};

    int32_t targetSizePct_;
    int32_t targetNumRows_;
    folly::Random::DefaultGenerator rng_;
  };

  void initializeInput(RowVectorPtr input);
  void initializeDestinations();
  void initializeSizeBuffers();
  void estimateRowSizes();
  void collectNullRows();

  const std::vector<column_index_t> keyChannels_;
  const int numDestinations_;
  const bool replicateNullsAndAny_;
  std::unique_ptr<core::PartitionFunction> partitionFunction_;
  const std::vector<column_index_t> outputChannels_;
  const std::shared_ptr<UcxCpuRowOutputQueueManager> queueMgr_;
  const int64_t maxBufferedBytes_;
  const bool eagerFlush_;
  const uint64_t maxPageBytes_;
  const int32_t targetNumRowsBase_;
  VectorSerde* const serde_;
  const std::unique_ptr<VectorSerde::Options> serdeOptions_;

  exec::BlockingReason blockingReason_{exec::BlockingReason::kNotBlocked};
  ContinueFuture future_;
  bool finished_{false};

  std::vector<vector_size_t*> sizePointers_;
  std::vector<vector_size_t> rowSize_;
  std::vector<std::unique_ptr<Destination>> destinations_;
  bool replicatedAny_{false};
  RowVectorPtr output_;

  SelectivityVector rows_;
  SelectivityVector nullRows_;
  std::vector<uint32_t> partitions_;
  std::vector<DecodedVector> decodedVectors_;
  Scratch scratch_;
};

} // namespace facebook::velox::ucx_exchange
