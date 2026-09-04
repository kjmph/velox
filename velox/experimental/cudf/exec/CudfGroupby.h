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

#include "velox/experimental/cudf/exec/CudfAggregation.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"

#include <cudf/column/column_view.hpp>
#include <cudf/groupby.hpp>

#include <deque>
#include <memory>
#include <unordered_map>

namespace facebook::velox::cudf_velox {

struct GroupbyAggregationResultRef {
  uint32_t requestIndex;
  uint32_t resultIndex;
  std::shared_ptr<bool> shared;
};

struct GroupbyPartialMeanResultRefs {
  GroupbyAggregationResultRef sum;
  GroupbyAggregationResultRef count;
};

class GroupbyAggregationRequestBuilder {
 public:
  GroupbyAggregationRequestBuilder(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream);

  GroupbyAggregationResultRef addAggregationForColumn(
      uint32_t inputIndex,
      std::unique_ptr<cudf::groupby_aggregation> aggregation);

  GroupbyAggregationResultRef addAggregationForValues(
      cudf::column_view values,
      std::unique_ptr<cudf::groupby_aggregation> aggregation);

  GroupbyAggregationResultRef addSum(
      uint32_t inputIndex,
      uint32_t reusableInputKey,
      core::AggregationNode::Step step);

  GroupbyPartialMeanResultRefs addPartialMean(
      uint32_t inputIndex,
      uint32_t reusableInputKey);

  cudf::table_view const& tableView() const {
    return tableView_;
  }

  rmm::cuda_stream_view stream() const {
    return stream_;
  }

  cudf::groupby::aggregation_request& aggregationRequest(
      const GroupbyAggregationResultRef& ref);

 private:
  GroupbyAggregationResultRef addReusablePartialSum(
      uint32_t inputIndex,
      uint32_t reusableInputKey);

  cudf::table_view tableView_;
  std::vector<cudf::groupby::aggregation_request>& requests_;
  rmm::cuda_stream_view stream_;
  std::unordered_map<uint32_t, GroupbyAggregationResultRef>
      reusablePartialSums_;
};

struct GroupbyAggregator {
  core::AggregationNode::Step step;
  uint32_t inputIndex;
  uint32_t reusableInputKey;
  VectorPtr constant;
  TypePtr resultType;

  virtual void addGroupbyRequest(GroupbyAggregationRequestBuilder& builder) = 0;

  virtual std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) = 0;

  // Some aggregate requests reference temporary input columns owned by the
  // aggregator (for example decoded decimal state). The requests no longer
  // need these columns after cudf::groupby::aggregate returns. Releasing them
  // before output conversion lets the stream-ordered memory resource reuse
  // their storage for result serialization.
  virtual void releaseRequestInputs() {}

  virtual ~GroupbyAggregator() = default;

 protected:
  GroupbyAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      uint32_t reusableInputKey,
      VectorPtr constant,
      const TypePtr& resultType)
      : step(step),
        inputIndex(inputIndex),
        reusableInputKey(reusableInputKey),
        constant(constant),
        resultType(resultType) {}
};

// Factory functions for creating groupby aggregators from plan nodes.
std::vector<std::unique_ptr<GroupbyAggregator>> toGroupbyAggregators(
    core::AggregationNode const& aggregationNode,
    core::AggregationNode::Step step,
    TypePtr const& outputType,
    std::vector<VectorPtr> const& constants,
    const std::vector<column_index_t>* aggregationInputChannels = nullptr);

// Groupby-specific validation
bool canGroupbyBeEvaluatedByCudf(
    const core::AggregationNode& aggregationNode,
    core::QueryCtx* queryCtx);

bool canGroupbyAggregationBeEvaluatedByCudf(
    const core::CallTypedExpr& call,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& rawInputTypes,
    core::QueryCtx* queryCtx);

class CudfGroupby : public CudfOperatorBase {
 public:
  CudfGroupby(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::AggregationNode> const& aggregationNode);

  void initialize() override;

  bool needsInput() const override {
    return !noMoreInput_ && pendingInput_ == nullptr &&
        !(flushGroupIdPartialInput_ && bufferedResult_ != nullptr) &&
        !(isPartialOutput_ && streamingEnabled_ && bufferedResult_ != nullptr &&
          bufferedResult_->estimateFlatSize() >
              partialAggregationFlushThresholdBytes_) &&
        pendingPartialResult_ == nullptr;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /* unused */) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;

  RowVectorPtr doGetOutput() override;

  void doNoMoreInput() override;

  void doClose() override;

 private:
  struct ParkedFinalAggregationState {
    RowVectorPtr state;
    uint64_t deviceBytes{0};
  };

  struct FinalAggregationBucket {
    std::vector<CudfVectorPtr> states;
    std::vector<ParkedFinalAggregationState> parkedStates;
    uint64_t rows{0};
    uint64_t bytes{0};
    uint32_t hashDepth{0};

    bool empty() const {
      return states.empty() && parkedStates.empty();
    }

    size_t numStates() const {
      return states.size() + parkedStates.size();
    }
  };

  CudfVectorPtr doGroupByAggregation(
      cudf::table_view tableView,
      std::vector<column_index_t> const& groupByKeys,
      std::vector<std::unique_ptr<GroupbyAggregator>>& aggregators,
      TypePtr const& outputType,
      rmm::cuda_stream_view stream);

  CudfVectorPtr releaseAndResetBufferedResult();

  void computePartialGroupbyStreaming(CudfVectorPtr tbl);
  void computeFinalGroupbyStreaming(CudfVectorPtr tbl);
  void computeSingleGroupbyStreaming(CudfVectorPtr tbl);
  void installPendingInput(CudfVectorPtr input);
  void processNextPendingInputSlice();
  bool shouldFlushBeforeNextPendingInputSlice() const;
  uint64_t inputSliceTargetRows(const CudfVector& input) const;
  uint64_t projectedIntermediateBytesPerRow(
      const core::AggregationNode& aggregationNode) const;
  CudfVectorPtr getPartialAggregationOutput();
  void retainPendingGroupbyState(CudfVectorPtr state);
  void addPendingGroupbyState(
      CudfVectorPtr state,
      bool projectAggregationInputs);
  bool pendingGroupbyStatesExceedFinalBucketEnvelope() const;
  void initializeFinalAggregationBucketEnvelope(
      uint64_t inputRows,
      uint64_t inputBytes);
  void startFinalAggregationCollection(bool projectAggregationInputs);
  void collectFinalAggregationState(
      CudfVectorPtr state,
      bool projectAggregationInputs);
  void routeFinalAggregationState(CudfVectorPtr state);
  void routeFinalAggregationBucket(FinalAggregationBucket&& bucket);
  void collectFinalAggregationSlice(
      CudfVectorPtr state,
      bool projectAggregationInputs);
  void growFinalAggregationCollection();
  CudfVectorPtr finalizeCollectedGroupbyStates();
  void recordFinalAggregationCompactionInput(
      uint64_t inputRows,
      uint64_t inputBytes);
  void recordFinalAggregationRetainedState();
  ParkedFinalAggregationState parkFinalAggregationState(CudfVectorPtr state);
  CudfVectorPtr restoreFinalAggregationState(
      ParkedFinalAggregationState state,
      rmm::cuda_stream_view stream);
  void parkFinalAggregationBucket(FinalAggregationBucket& bucket);
  void restoreFinalAggregationBucket(FinalAggregationBucket& bucket);
  void compactPendingGroupbyStates(bool projectAggregationInputs);
  CudfVectorPtr compactGroupbyStates(
      std::vector<CudfVectorPtr>&& states,
      bool projectAggregationInputs,
      CudfVectorPtr existingState = nullptr);
  CudfVectorPtr aggregateGroupbyStates(
      std::vector<CudfVectorPtr>&& states,
      bool projectAggregationInputs,
      CudfVectorPtr existingState,
      std::vector<std::unique_ptr<GroupbyAggregator>>& aggregators,
      const TypePtr& outputType);
  CudfVectorPtr finalizePendingGroupbyStates();
  CudfVectorPtr getNextFinalAggregationBucket();
  void initializeFinalAggregationBuckets(
      std::vector<CudfVectorPtr>&& states,
      CudfVectorPtr existingState,
      bool projectAggregationInputs,
      uint64_t inputRows,
      uint64_t inputBytes);
  void hashPartitionFinalState(
      CudfVectorPtr state,
      bool projectAggregationInputs,
      uint32_t numBuckets,
      uint32_t hashSeed,
      uint32_t hashDepth,
      std::vector<FinalAggregationBucket>& buckets,
      bool materializeBuckets = false);
  void repartitionFinalAggregationBucket(FinalAggregationBucket&& bucket);
  FinalAggregationBucket compactSkewedFinalAggregationBucket(
      FinalAggregationBucket&& bucket);
  std::vector<CudfVectorPtr> splitFinalAggregationState(
      CudfVectorPtr state) const;
  bool finalAggregationBucketIsOversized(
      const FinalAggregationBucket& bucket) const;
  uint64_t finalAggregationWorkBytes(uint64_t inputRows, uint64_t inputBytes)
      const;
  uint32_t finalAggregationPartitionCount(
      uint64_t rows,
      uint64_t bytes,
      bool requireSplit) const;
  uint64_t finalAggregationBucketTargetBytes() const;
  uint64_t finalAggregationBucketTargetRows(
      uint64_t inputRows,
      uint64_t inputBytes,
      uint64_t targetBytes) const;
  int64_t partialAggregationFlushThresholdBytes() const;

  std::vector<column_index_t> groupingKeyInputChannels_;
  std::vector<column_index_t> groupingKeyOutputChannels_;
  std::vector<column_index_t> aggregationInputChannels_;

  std::shared_ptr<const core::AggregationNode> aggregationNode_;
  std::vector<std::unique_ptr<GroupbyAggregator>> aggregators_;
  std::vector<std::unique_ptr<GroupbyAggregator>> intermediateAggregators_;
  // Used for kSingle streaming: partial-step aggregators (raw -> intermediate)
  // and final-step aggregators (intermediate -> final).
  std::vector<std::unique_ptr<GroupbyAggregator>> partialAggregators_;
  std::vector<std::unique_ptr<GroupbyAggregator>> finalAggregators_;

  const bool isPartialOutput_;
  const bool isSingleStep_;
  // GroupId emits one grouping-set view at a time. Partial aggregation can
  // reduce each view independently and let the downstream final aggregation
  // merge duplicate states, avoiding materialization of the expanded input.
  const bool flushGroupIdPartialInput_;
  // Streaming aggregation is disabled if companion aggregates are present.
  bool streamingEnabled_{true};
  const int64_t maxPartialAggregationMemoryUsage_;
  int64_t partialAggregationFlushThresholdBytes_{0};
  uint64_t groupbyInputSliceTargetBytes_{0};
  uint64_t projectedIntermediateBytesPerRow_{1};
  int64_t numInputRows_ = 0;
  uint64_t finalAggregationReceivedInputBatches_{0};
  uint64_t finalAggregationReceivedInputRows_{0};
  uint64_t finalAggregationReceivedInputBytes_{0};

  bool finished_ = false;
  size_t numAggregates_;
  bool ignoreNullKeys_;

  std::vector<CudfVectorPtr> inputs_;
  // Streaming partial aggregation can expand one raw input row into a
  // substantially wider intermediate state. Retain one input owner and expose
  // only one budgeted zero-copy view to cuDF at a time.
  CudfVectorPtr pendingInput_;
  vector_size_t pendingInputOffset_{0};
  vector_size_t pendingInputSliceRows_{0};
  std::vector<CudfVectorPtr> pendingGroupbyStates_;
  // A partial result that could not be compacted with bufferedResult_ without
  // crossing the flush threshold. needsInput() stays false until the buffered
  // result has been returned and this becomes the next buffered result.
  CudfVectorPtr pendingPartialResult_;
  int64_t pendingPartialInputRows_{0};
  int64_t pendingGroupbyStateBytes_{0};
  // Set after an intermediate compaction observes no duplicate-key reduction.
  // Further compactions would only rewrite an ever-growing state; retain the
  // bounded input batches and merge them once with the final aggregators.
  bool intermediateCompactionAbandoned_{false};
  // A high-cardinality final aggregation cannot safely concatenate every
  // intermediate state into one device table. Hash buckets are disjoint on
  // the grouping keys, so each bucket can be finalized and returned before
  // the next one is materialized.
  bool finalAggregationInitialized_{false};
  bool hashBucketFinalization_{false};
  uint64_t finalAggregationBucketTargetRows_{0};
  uint64_t finalAggregationBucketTargetBytes_{0};
  uint32_t nextFinalAggregationHashSeed_{0};
  bool finalAggregationCollecting_{false};
  uint32_t finalAggregationCollectionFanout_{0};
  uint32_t finalAggregationCollectionHashSeed_{0};
  std::vector<FinalAggregationBucket> finalAggregationCollectionBuckets_;
  std::deque<FinalAggregationBucket> finalAggregationBuckets_;
  uint64_t finalAggregationHostParkedRows_{0};
  uint64_t finalAggregationHostParkedBytes_{0};
  TypePtr inputType_;
  RowTypePtr bufferedResultType_;
  CudfVectorPtr bufferedResult_;
};

} // namespace facebook::velox::cudf_velox
