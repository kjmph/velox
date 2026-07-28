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
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#include <cudf/column/column_factories.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <folly/Executor.h>
#include <folly/synchronization/EventCount.h>
#include <gtest/gtest.h>
#include <rmm/device_buffer.hpp>
#include <atomic>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/ExchangeTransportRegistry.h"
#include "velox/exec/OutputTransportRegistry.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"
#include "velox/experimental/ucx-exchange/UcxExchangeRegistration.h"
#include "velox/experimental/ucx-exchange/tests/UcxTestHelpers.h"

using namespace facebook::velox::ucx_exchange;
using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::core;

class UcxOutputQueueManagerTest : public testing::Test {
 protected:
  UcxOutputQueueManagerTest() {}

  void SetUp() override {
    pool_ = facebook::velox::memory::memoryManager()->addLeafPool();
    queueManager_ = std::make_shared<UcxOutputQueueManager>();
  }

  std::shared_ptr<Task> initializeTask(
      const std::string& taskId,
      int numDestinations,
      int numDrivers,
      core::PartitionedOutputNode::Kind kind =
          core::PartitionedOutputNode::Kind::kPartitioned,
      uint64_t maxOutputBufferSize = FOUR_GBYTES) {
    auto task = createSourceTask(
        taskId, pool_, UcxTestData::kTestRowType, maxOutputBufferSize);

    queueManager_->initializeTask(task, kind, numDestinations, numDrivers);
    return task;
  }

  std::unique_ptr<cudf::packed_columns> makePackedColumns(std::size_t numRows) {
    cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{cudaStreamDefault}};
    // Create table directly without going through pack/unpack
    auto table = facebook::velox::ucx_exchange::makeTable(
        numRows, UcxTestData::kTestRowType, stream);
    auto cols = std::make_unique<cudf::packed_columns>(
        cudf::pack(table->view(), stream));
    stream.sync();
    return cols;
  }

  std::unique_ptr<cudf::packed_columns> makeZeroColumnPackedColumns() {
    cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{cudaStreamDefault}};
    cudf::table table;
    auto cols = std::make_unique<cudf::packed_columns>(
        cudf::pack(table.view(), stream));
    stream.sync();
    return cols;
  }

  void enqueue(std::string_view taskId, vector_size_t size) {
    enqueue(taskId, 0, size);
  }

  // Returns the enqueued page byte size.
  void enqueue(std::string_view taskId, int destination, vector_size_t size) {
    auto data = makePackedColumns(size);
    queueManager_->enqueue(taskId, destination, std::move(data), size);
  }

  void noMoreData(std::string_view taskId) {
    queueManager_->noMoreData(taskId);
  }

  void fetch(
      std::string_view taskId,
      int destination,
      bool expectedEndMarker = false) {
    bool receivedData = false;
    queueManager_->getDataWithQueue(
        taskId,
        destination,
        [destination, expectedEndMarker, &receivedData](
            std::shared_ptr<UcxOutputQueue> outputQueue,
            std::shared_ptr<cudf::packed_columns> data,
            vector_size_t /*numRows*/,
            std::vector<int64_t> remainingBytes) {
          ASSERT_EQ(expectedEndMarker, data == nullptr)
              << "for destination " << destination;
          if (data) {
            outputQueue->releaseInFlightBytes(
                destination, static_cast<int64_t>(data->gpu_data->size()), 1);
          }
          receivedData = true;
        });
    ASSERT_TRUE(receivedData) << "for destination " << destination;
  }

  UcxDataAvailableCallback receiveEndMarker(
      int destination,
      bool& receivedEndMarker) {
    return [destination, &receivedEndMarker](
               std::shared_ptr<cudf::packed_columns> data,
               vector_size_t /*numRows*/,
               std::vector<int64_t> remainingBytes) {
      EXPECT_FALSE(receivedEndMarker) << "for destination " << destination;
      EXPECT_TRUE(data == nullptr) << "for destination " << destination;
      EXPECT_TRUE(remainingBytes.empty());
      receivedEndMarker = true;
    };
  }

  void fetchEndMarker(std::string_view taskId, int destination) {
    bool receivedData = false;
    queueManager_->getData(
        taskId, destination, receiveEndMarker(destination, receivedData));
    EXPECT_TRUE(receivedData) << "for destination " << destination;
    queueManager_->deleteResults(taskId, destination);
  }

  void deleteResults(std::string_view taskId, int destination) {
    queueManager_->deleteResults(taskId, destination);
  }

  void registerForEndMarker(
      std::string_view taskId,
      int destination,
      bool& receivedEndMarker) {
    receivedEndMarker = false;
    queueManager_->getData(
        taskId, destination, receiveEndMarker(destination, receivedEndMarker));
    EXPECT_FALSE(receivedEndMarker) << "for destination " << destination;
  }

  void registerForData(
      std::string_view taskId,
      int destination,
      bool& receivedData) {
    receivedData = false;
    queueManager_->getDataWithQueue(
        taskId,
        destination,
        [destination, &receivedData](
            std::shared_ptr<UcxOutputQueue> outputQueue,
            std::shared_ptr<cudf::packed_columns> data,
            vector_size_t /*numRows*/,
            std::vector<int64_t> /*remainingBytes*/) {
          EXPECT_FALSE(receivedData) << "for destination " << destination;
          EXPECT_NE(outputQueue, nullptr) << "for destination " << destination;
          EXPECT_NE(data, nullptr) << "for destination " << destination;
          if (outputQueue && data) {
            outputQueue->releaseInFlightBytes(
                destination, static_cast<int64_t>(data->gpu_data->size()), 1);
          }
          receivedData = true;
        });
    EXPECT_FALSE(receivedData) << "for destination " << destination;
  }

  void dataFetcher(
      std::string_view taskId,
      int destination,
      int64_t& fetchedPackedColumns,
      bool earlyTermination) {
    int64_t received{0};
    folly::Random::DefaultGenerator rng;
    rng.seed(destination);
    while (true) {
      if (earlyTermination && folly::Random().oneIn(200)) {
        queueManager_->deleteResults(taskId, destination);
        return;
      }
      bool atEnd{false};
      folly::EventCount dataWait;
      auto dataWaitKey = dataWait.prepareWait();
      queueManager_->getDataWithQueue(
          taskId,
          destination,
          [&](std::shared_ptr<UcxOutputQueue> outputQueue,
              std::shared_ptr<cudf::packed_columns> data,
              vector_size_t /*numRows*/,
              std::vector<int64_t> /*remainingBytes*/) {
            if (data == nullptr) {
              atEnd = true;
            } else {
              received++;
              outputQueue->releaseInFlightBytes(
                  destination, static_cast<int64_t>(data->gpu_data->size()), 1);
            }
            dataWait.notify();
          });
      dataWait.wait(dataWaitKey);
      if (atEnd) {
        break;
      }
    }
    queueManager_->deleteResults(taskId, destination);
    // out of order requests are allowed (fetch after delete)
    {
      struct Response {
        std::shared_ptr<cudf::packed_columns> data;
        std::vector<int64_t> remainingBytes;
      };
      folly::Promise<Response> promise;
      auto future = promise.getSemiFuture();
      queueManager_->getData(
          taskId,
          destination,
          [&promise](
              std::shared_ptr<cudf::packed_columns> data,
              vector_size_t /*numRows*/,
              std::vector<int64_t> remainingBytes) {
            promise.setValue(
                Response{std::move(data), std::move(remainingBytes)});
          });
      future.wait();
      ASSERT_TRUE(future.isReady());
      auto& response = future.value();
      ASSERT_EQ(response.remainingBytes.size(), 0);
      ASSERT_EQ(response.data, nullptr);
    }

    fetchedPackedColumns = received;
  }

  std::shared_ptr<facebook::velox::memory::MemoryPool> pool_;
  std::shared_ptr<UcxOutputQueueManager> queueManager_;
};

// Drives all eight exec::OutputBufferManager virtuals through a base-class
// pointer, which is the only surface exec::Task and the Presto stats layer see.
// The utilization pair is pinned to the queue's own byte accounting rather than
// to a loose bound, so a constant-returning implementation cannot satisfy it.
TEST_F(UcxOutputQueueManagerTest, implementsOutputBufferManager) {
  static_assert(
      std::is_base_of_v<exec::OutputBufferManager, UcxOutputQueueManager>);

  std::shared_ptr<exec::OutputBufferManager> mgr = queueManager_;
  ASSERT_NE(mgr, nullptr);

  // An unknown task is tolerated by every method, and the two observability
  // methods report "no answer" rather than inventing one.
  const std::string unknownTaskId = "outputBufferManagerIface.unknown";
  EXPECT_NO_THROW(mgr->toString(unknownTaskId));
  EXPECT_FALSE(mgr->updateNumDrivers(unknownTaskId, 1));
  EXPECT_FALSE(mgr->updateOutputBuffers(unknownTaskId, 1, true));
  EXPECT_EQ(mgr->stats(unknownTaskId), std::nullopt);
  EXPECT_EQ(mgr->getUtilization(unknownTaskId), std::nullopt);
  EXPECT_EQ(mgr->isOverutilized(unknownTaskId), std::nullopt);

  // Use an id unique to this case: the manager is a process-wide singleton
  // shared with every other case in the binary.
  const std::string taskId = "outputBufferManagerIface";
  const int numDestinations = 2;
  // A capacity small enough that a handful of pages crosses the
  // over-utilization threshold, so isOverutilized() is seen in both states.
  const uint64_t maxOutputBufferSize{4'096};
  auto task = initializeTask(
      taskId,
      numDestinations,
      1 /*numDrivers*/,
      core::PartitionedOutputNode::Kind::kPartitioned,
      maxOutputBufferSize);

  // Empty but bounded: a real ratio of exactly zero, which is a different
  // answer from the unbounded placeholder queue's nullopt (see
  // lateTaskCreation).
  auto stats = mgr->stats(taskId);
  ASSERT_TRUE(stats.has_value());
  ASSERT_EQ(stats->bufferedBytes, 0);
  EXPECT_EQ(mgr->getUtilization(taskId), 0.0);
  EXPECT_EQ(mgr->isOverutilized(taskId), false);

  // One page: utilization is queued bytes over the configured capacity.
  // Producers are only blocked *after* adding data (UcxOutputQueue::
  // checkBlocked), so queuedBytes_ may exceed maxSize_ and the ratio is not
  // bounded by 1.0.
  enqueue(taskId, 0, /*size=*/10);
  stats = mgr->stats(taskId);
  ASSERT_TRUE(stats.has_value());
  const int64_t bufferedBytes = stats->bufferedBytes;
  ASSERT_GT(bufferedBytes, 0);

  auto utilization = mgr->getUtilization(taskId);
  ASSERT_TRUE(utilization.has_value());
  EXPECT_GT(*utilization, 0.0);
  EXPECT_DOUBLE_EQ(
      *utilization, bufferedBytes / static_cast<double>(maxOutputBufferSize));

  const auto halfCapacity = static_cast<int64_t>(maxOutputBufferSize / 2);
  auto overutilized = mgr->isOverutilized(taskId);
  ASSERT_TRUE(overutilized.has_value());
  EXPECT_EQ(*overutilized, bufferedBytes > halfCapacity);

  // Fill past half the capacity: the flag must follow the queued bytes. The
  // iteration bound only stops a runaway loop if enqueue ever stops
  // accumulating; the assertion after it is what the case actually checks.
  for (int i = 0; i < 100 && mgr->stats(taskId)->bufferedBytes <= halfCapacity;
       ++i) {
    enqueue(taskId, 0, /*size=*/10);
  }
  ASSERT_GT(mgr->stats(taskId)->bufferedBytes, halfCapacity);
  EXPECT_EQ(mgr->isOverutilized(taskId), true);
  ASSERT_TRUE(mgr->getUtilization(taskId).has_value());
  EXPECT_GT(*mgr->getUtilization(taskId), 0.5);

  EXPECT_TRUE(mgr->updateNumDrivers(taskId, 2));
  EXPECT_TRUE(mgr->updateOutputBuffers(taskId, numDestinations, true));
  EXPECT_FALSE(mgr->toString(taskId).empty());

  mgr->removeTask(taskId);
  EXPECT_EQ(mgr->stats(taskId), std::nullopt);
}

// The output half of the kUcx registration: the entry the manager is published
// under, and that a second registration replaces rather than rejects.
TEST_F(UcxOutputQueueManagerTest, registersUcxOutputTransport) {
  // Start from a clean baseline so this case does not depend on registration
  // state left behind by other suites in the binary. unregisterAll() re-seeds
  // the built-in in-memory default, so only kUcx is actually cleared.
  exec::OutputTransportRegistry::unregisterAll();
  ASSERT_EQ(
      exec::OutputTransportRegistry::tryGet(
          std::string{core::TransportKind::kUcx}),
      nullptr);

  registerUcxTransports();
  auto entry = exec::OutputTransportRegistry::tryGet(
      std::string{core::TransportKind::kUcx});
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->manager, UcxOutputQueueManager::getInstanceRef());
  EXPECT_TRUE(static_cast<bool>(entry->makeOutputOperator));

  // Idempotent: a second call replaces the entry rather than throwing on the
  // duplicate key.
  registerUcxTransports();
  auto entryAfterSecondCall = exec::OutputTransportRegistry::tryGet(
      std::string{core::TransportKind::kUcx});
  ASSERT_NE(entryAfterSecondCall, nullptr);
  EXPECT_EQ(
      entryAfterSecondCall->manager, UcxOutputQueueManager::getInstanceRef());

  // unregisterUcxTransports() erases kUcx from both registries and leaves the
  // in-memory default alone.
  unregisterUcxTransports();
  EXPECT_EQ(
      exec::OutputTransportRegistry::tryGet(
          std::string{core::TransportKind::kUcx}),
      nullptr);
  EXPECT_EQ(
      exec::ExchangeTransportRegistry::tryGet(
          std::string{core::TransportKind::kUcx}),
      nullptr);
  EXPECT_NE(
      exec::OutputTransportRegistry::tryGet(
          std::string{core::TransportKind::kInMemory}),
      nullptr);
}

class TestExchangeServerLifecycle : public UcxExchangeServerLifecycle {
 public:
  TestExchangeServerLifecycle(std::string taskId, uint32_t destination)
      : key_{std::move(taskId), destination} {}

  const PartitionKey& getPartitionKey() const override {
    return key_;
  }

  void requestAbort() override {
    bool expected = false;
    if (abortRequested_.compare_exchange_strong(expected, true)) {
      ++abortRequests_;
    }
  }

  bool activate() override {
    if (closed_.load() || abortRequested_.load()) {
      return false;
    }
    activated_.store(true);
    return true;
  }

  bool isClosed() const override {
    return closed_.load();
  }

  int abortRequests() const {
    return abortRequests_.load();
  }

  bool isActivated() const {
    return activated_.load();
  }

 private:
  const PartitionKey key_;
  std::atomic<int> abortRequests_{0};
  std::atomic<bool> abortRequested_{false};
  std::atomic<bool> closed_{false};
  std::atomic<bool> activated_{false};
};

using ExchangeServerAdmission = UcxOutputQueueManager::ExchangeServerAdmission;

TEST(GpuHandshakeResponseTest, admissionStatusPreservesWireLayout) {
  HandshakeResponse accepted;
  EXPECT_FALSE(accepted.isIntraNodeTransfer);
  EXPECT_EQ(accepted.status, GpuHandshakeResponseStatus::kAccepted);

  HandshakeResponse rejected;
  rejected.status = GpuHandshakeResponseStatus::kTaskRemoved;

  EXPECT_EQ(sizeof(rejected), sizeof(accepted));
  EXPECT_EQ(rejected.status, GpuHandshakeResponseStatus::kTaskRemoved);
}

TEST(GpuAbortProtocolTest, usesDedicatedTaskScopedControlTags) {
  constexpr uint32_t kTaskHash = 0xa1b2c3d4;
  const auto ackTag = getGpuHandshakeAckTag(kTaskHash);
  const auto abortTag = getGpuAbortTag(kTaskHash);

  EXPECT_EQ(ackTag >> 32, kTaskHash);
  EXPECT_EQ(ackTag & 0xff000000ULL, GPU_HANDSHAKE_ACK_TAG);
  EXPECT_EQ(ackTag & 0x00ffffffULL, 0);
  EXPECT_EQ(abortTag >> 32, kTaskHash);
  EXPECT_EQ(abortTag & 0xff000000ULL, GPU_ABORT_TAG);
  EXPECT_EQ(abortTag & 0x00ffffffULL, 0);
  EXPECT_NE(abortTag, ackTag);
  EXPECT_NE(ackTag, getHandshakeAckTag(kTaskHash));
  EXPECT_NE(abortTag, getCpuRowAbortTag(kTaskHash));
  EXPECT_NE(abortTag, getMetadataTag(kTaskHash, 0));
  EXPECT_NE(abortTag, getDataTag(kTaskHash, 0));

  GpuHandshakeAckHeader ack;
  EXPECT_EQ(ack.magic, kGpuHandshakeAckMagic);
  EXPECT_EQ(ack.version, kGpuHandshakeAckVersion);
  EXPECT_EQ(ack.headerSize, sizeof(GpuHandshakeAckHeader));

  GpuAbortHeader abort;
  EXPECT_EQ(abort.magic, kGpuAbortMagic);
  EXPECT_EQ(abort.version, kGpuAbortVersion);
  EXPECT_EQ(abort.headerSize, sizeof(GpuAbortHeader));
}

TEST(GpuMetadataProtocolTest, preservesLogicalRowCount) {
  MetadataMsg metadata;
  metadata.cudfMetadata = std::make_unique<std::vector<uint8_t>>(
      std::initializer_list<uint8_t>{1, 2, 3, 4});
  metadata.dataSizeBytes = 37;
  metadata.numRows = 123;
  metadata.remainingBytes = {19, 7};
  metadata.atEnd = false;

  auto [serialized, size] = metadata.serialize();
  EXPECT_EQ(size, metadata.getSerializedSize());
  auto decoded = MetadataMsg::deserializeMetadataMsg(serialized.get(), size);
  EXPECT_EQ(*decoded.cudfMetadata, *metadata.cudfMetadata);
  EXPECT_EQ(decoded.dataSizeBytes, metadata.dataSizeBytes);
  EXPECT_EQ(decoded.numRows, metadata.numRows);
  EXPECT_EQ(decoded.remainingBytes, metadata.remainingBytes);
  EXPECT_EQ(decoded.atEnd, metadata.atEnd);

  // The combined wire format requires the complete, bounded record.
  EXPECT_ANY_THROW(
      MetadataMsg::deserializeMetadataMsg(serialized.get(), size - 1));
}

TEST_F(UcxOutputQueueManagerTest, preservesLogicalRowCount) {
  const std::string taskId = "logical-row-count";
  constexpr vector_size_t kNumRows = 17;
  initializeTask(taskId, 1, 1);
  queueManager_->enqueue(taskId, 0, makeZeroColumnPackedColumns(), kNumRows);

  std::shared_ptr<cudf::packed_columns> received;
  vector_size_t receivedRows = 0;
  auto stableQueue = queueManager_->getDataWithQueue(
      taskId,
      0,
      [&](std::shared_ptr<UcxOutputQueue>,
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t numRows,
          std::vector<int64_t>) {
        received = std::move(data);
        receivedRows = numRows;
      });
  ASSERT_NE(received, nullptr);
  ASSERT_NE(stableQueue, nullptr);
  EXPECT_EQ(receivedRows, kNumRows);
  EXPECT_EQ(cudf::unpack(*received).num_rows(), 0);
  const auto bytes = received->gpu_data->size();
  received.reset();
  stableQueue->releaseInFlightBytes(0, bytes, 1);
  noMoreData(taskId);
  fetchEndMarker(taskId, 0);
  queueManager_->removeTask(taskId);
}

TEST_F(
    UcxOutputQueueManagerTest,
    abnormalTaskRemovalAbortsAllAndOnlyMatchingServers) {
  auto taskA = initializeTask(
      "gpu-aborted-task-a", 2 /* numDestinations */, 1 /* numDrivers */);
  auto taskB = initializeTask(
      "gpu-unrelated-task-b", 1 /* numDestinations */, 1 /* numDrivers */);

  auto a0 = std::make_shared<TestExchangeServerLifecycle>(taskA->taskId(), 0);
  auto a0Duplicate =
      std::make_shared<TestExchangeServerLifecycle>(taskA->taskId(), 0);
  auto a1 = std::make_shared<TestExchangeServerLifecycle>(taskA->taskId(), 1);
  auto b0 = std::make_shared<TestExchangeServerLifecycle>(taskB->taskId(), 0);

  EXPECT_EQ(
      queueManager_->registerExchangeServer(a0),
      ExchangeServerAdmission::kAccepted);
  EXPECT_EQ(
      queueManager_->registerExchangeServer(a0Duplicate),
      ExchangeServerAdmission::kDuplicateServer);
  EXPECT_EQ(
      queueManager_->registerExchangeServer(a1),
      ExchangeServerAdmission::kAccepted);
  EXPECT_EQ(
      queueManager_->registerExchangeServer(b0),
      ExchangeServerAdmission::kAccepted);
  EXPECT_TRUE(a0->isActivated());
  EXPECT_FALSE(a0Duplicate->isActivated());
  EXPECT_TRUE(a1->isActivated());
  EXPECT_TRUE(b0->isActivated());
  EXPECT_EQ(a0Duplicate->abortRequests(), 1);

  taskA->requestAbort().wait();
  queueManager_->removeTask(taskA->taskId());

  EXPECT_EQ(a0->abortRequests(), 1);
  EXPECT_EQ(a0Duplicate->abortRequests(), 1);
  EXPECT_EQ(a1->abortRequests(), 1);
  EXPECT_EQ(b0->abortRequests(), 0);

  // Repeated removal preserves the tombstone and does not reissue aborts.
  queueManager_->removeTask(taskA->taskId());
  EXPECT_EQ(a0->abortRequests(), 1);
  EXPECT_EQ(a1->abortRequests(), 1);

  auto late = std::make_shared<TestExchangeServerLifecycle>(taskA->taskId(), 0);
  EXPECT_EQ(
      queueManager_->registerExchangeServer(late),
      ExchangeServerAdmission::kTaskRemoved);
  EXPECT_EQ(late->abortRequests(), 1);

  queueManager_->unregisterExchangeServer(b0);
  taskB->requestAbort().wait();
  queueManager_->removeTask(taskB->taskId());
  EXPECT_EQ(b0->abortRequests(), 0);
}

TEST_F(
    UcxOutputQueueManagerTest,
    finishedTaskServersDrainInsteadOfBeingAborted) {
  const std::string taskId = "gpu-normally-finished-task";
  auto task =
      initializeTask(taskId, 1 /* numDestinations */, 1 /* numDrivers */);
  auto active = std::make_shared<TestExchangeServerLifecycle>(taskId, 0);
  EXPECT_EQ(
      queueManager_->registerExchangeServer(active),
      ExchangeServerAdmission::kAccepted);

  task->testingFinish();
  queueManager_->removeTask(taskId);
  EXPECT_EQ(active->abortRequests(), 0);

  auto late = std::make_shared<TestExchangeServerLifecycle>(taskId, 0);
  EXPECT_EQ(
      queueManager_->registerExchangeServer(late),
      ExchangeServerAdmission::kTaskRemoved);
  EXPECT_FALSE(late->isActivated());
  EXPECT_EQ(late->abortRequests(), 1);

  auto reusedTask = createSourceTask(taskId, pool_, UcxTestData::kTestRowType);
  EXPECT_ANY_THROW(queueManager_->initializeTask(
      reusedTask,
      core::PartitionedOutputNode::Kind::kPartitioned,
      1 /* numDestinations */,
      1 /* numDrivers */));
  reusedTask->requestAbort().wait();
}

TEST_F(
    UcxOutputQueueManagerTest,
    removedTaskTombstoneRejectsLateServerAndReinitialization) {
  const std::string taskId = "gpu-removed-before-initialization";

  queueManager_->removeTask(taskId);
  queueManager_->removeTask(taskId);

  auto late = std::make_shared<TestExchangeServerLifecycle>(taskId, 0);
  EXPECT_EQ(
      queueManager_->registerExchangeServer(late),
      ExchangeServerAdmission::kTaskRemoved);
  EXPECT_EQ(late->abortRequests(), 1);

  auto task = createSourceTask(taskId, pool_, UcxTestData::kTestRowType);
  EXPECT_ANY_THROW(queueManager_->initializeTask(
      task,
      core::PartitionedOutputNode::Kind::kPartitioned,
      1 /* numDestinations */,
      1 /* numDrivers */));
  task->requestAbort().wait();
}

TEST_F(UcxOutputQueueManagerTest, serverMayArriveBeforeTaskInitialization) {
  const std::string taskId = "gpu-server-before-task-initialization";
  auto early = std::make_shared<TestExchangeServerLifecycle>(taskId, 0);
  EXPECT_EQ(
      queueManager_->registerExchangeServer(early),
      ExchangeServerAdmission::kAccepted);
  EXPECT_TRUE(early->isActivated());

  auto task =
      initializeTask(taskId, 1 /* numDestinations */, 1 /* numDrivers */);
  EXPECT_EQ(early->abortRequests(), 0);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
  EXPECT_EQ(early->abortRequests(), 1);
}

TEST_F(UcxOutputQueueManagerTest, diagnostics) {
  const std::string taskId = "diagnostics";
  initializeTask(taskId, 2 /* numDestinations */, 1 /* numDrivers */);

  const auto diagnostics = queueManager_->toString(taskId);
  EXPECT_NE(
      diagnostics.find("UcxOutputQueue task=" + taskId), std::string::npos);
  EXPECT_NE(diagnostics.find("queues="), std::string::npos);
  EXPECT_NE(diagnostics.find("numFinished=0/1"), std::string::npos);

  queueManager_->removeTask(taskId);
  EXPECT_EQ(
      queueManager_->toString(taskId),
      "UcxOutputQueue[" + taskId + " not found]");
}

TEST_F(UcxOutputQueueManagerTest, basicPartitioned) {
  vector_size_t size = 100;
  std::string taskId = "basic-partitioned";
  auto task = initializeTask(taskId, 5 /* numDestinations*/, 1 /*numDrivers*/);

  ASSERT_FALSE(queueManager_->isFinished(taskId));

  // - enqueue one group per destination
  // - fetch and ask one group per destination
  // - enqueue one more group for destinations 0-2
  // - fetch and ask one group for destinations 0-2
  // - register end-marker callback for destination 3 and data callback for 4
  // - enqueue data for 4 and assert callback was called
  // - enqueue end marker for all destinations
  // - assert callback was called for destination 3
  // - fetch end markers for destinations 0-2 and 4

  for (int destination = 0; destination < 5; ++destination) {
    enqueue(taskId, destination, size);
  }
  EXPECT_FALSE(queueManager_->isFinished(taskId));

  for (int destination = 0; destination < 5; destination++) {
    fetch(taskId, destination);
  }

  EXPECT_FALSE(queueManager_->isFinished(taskId));

  for (int destination = 0; destination < 3; destination++) {
    enqueue(taskId, destination, size);
  }

  for (int destination = 0; destination < 3; destination++) {
    fetch(taskId, destination);
  }

  // destinations 0-2 received 2 packed_columns each
  // destinations 3-4 received 1 packed_columns each

  bool receivedEndMarker3;
  registerForEndMarker(taskId, 3, receivedEndMarker3);

  bool receivedData4;
  registerForData(taskId, 4, receivedData4);

  enqueue(taskId, 4, size);
  EXPECT_TRUE(receivedData4);

  noMoreData(taskId);
  EXPECT_TRUE(receivedEndMarker3);
  EXPECT_FALSE(queueManager_->isFinished(taskId));

  for (int destination = 0; destination < 3; destination++) {
    fetchEndMarker(taskId, destination);
  }

  EXPECT_TRUE(task->isRunning());

  deleteResults(taskId, 3);
  fetchEndMarker(taskId, 4);
  EXPECT_TRUE(queueManager_->isFinished(taskId));

  queueManager_->removeTask(taskId);
  EXPECT_TRUE(task->isFinished());
}

TEST_F(UcxOutputQueueManagerTest, basicAsyncFetch) {
  const vector_size_t size = 10;
  const std::string taskId = "basic-async-fetch";
  int numPartitions = 1;
  bool earlyTermination = false;

  initializeTask(taskId, numPartitions, 1);

  // asynchronously fetch data.
  int64_t fetchedPackedColumns = 0;
  std::thread thread([&]() {
    dataFetcher(taskId, 0, fetchedPackedColumns, earlyTermination);
  });
  const int totalPackedColumns = 100;
  int64_t producedPackedColumns = 0;

  for (int i = 0; i < totalPackedColumns; ++i) {
    const int partition = 0;
    try {
      enqueue(taskId, partition, size);
    } catch (...) {
      // Early termination might cause the task to fail early.
      ASSERT_TRUE(earlyTermination);
      break;
    }
    ++producedPackedColumns;
    if (folly::Random().oneIn(4)) {
      std::this_thread::sleep_for(std::chrono::microseconds(5)); // NOLINT
    }
  }
  noMoreData(taskId);

  thread.join();

  if (!earlyTermination) {
    ASSERT_EQ(fetchedPackedColumns, producedPackedColumns);
  }
  queueManager_->removeTask(taskId);
}

TEST_F(UcxOutputQueueManagerTest, lateTaskCreation) {
  const vector_size_t size = 10;
  const std::string taskId = "lateTaskCreation.placeholder";
  int numPartitions = 1;
  bool earlyTermination = false;
  int destination = 0;

  // Fetch data from a non-existing task.
  struct Response {
    std::shared_ptr<UcxOutputQueue> outputQueue;
    std::shared_ptr<cudf::packed_columns> data;
    std::vector<int64_t> remainingBytes;
  };
  folly::Promise<Response> promise;
  auto future = promise.getSemiFuture();
  queueManager_->getDataWithQueue(
      taskId,
      destination,
      [&promise](
          std::shared_ptr<UcxOutputQueue> outputQueue,
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> remainingBytes) {
        promise.setValue(
            Response{
                std::move(outputQueue),
                std::move(data),
                std::move(remainingBytes)});
      });

  // getData() above created a placeholder queue with no known capacity yet:
  // maxSize_ is only set from the task's query config, in initializeTask().
  // Both observability methods must report that honestly as nullopt -- one so
  // it does not divide by zero, the other so it does not call an
  // unknown-capacity queue over-utilized the moment any byte is queued.
  std::shared_ptr<exec::OutputBufferManager> mgr = queueManager_;
  EXPECT_EQ(mgr->getUtilization(taskId), std::nullopt);
  EXPECT_EQ(mgr->isOverutilized(taskId), std::nullopt);

  // initialize task.
  auto task = initializeTask(taskId, numPartitions, 1);
  // enqueue some data.
  enqueue(taskId, destination, size);
  noMoreData(taskId);
  uint32_t i = 0;
  while (true) {
    ++i;
    future.wait();
    ASSERT_TRUE(future.isReady());
    Response& response = future.value();
    if (!response.data) {
      // nullptr -> end of transmission.
      break;
    }
    response.outputQueue->releaseInFlightBytes(
        destination, static_cast<int64_t>(response.data->gpu_data->size()), 1);
    folly::Promise<Response> promise;
    future = promise.getSemiFuture();
    queueManager_->getDataWithQueue(
        taskId,
        destination,
        [&promise](
            std::shared_ptr<UcxOutputQueue> outputQueue,
            std::shared_ptr<cudf::packed_columns> data,
            vector_size_t /*numRows*/,
            std::vector<int64_t> remainingBytes) {
          promise.setValue(
              Response{
                  std::move(outputQueue),
                  std::move(data),
                  std::move(remainingBytes)});
        });
  }
  ASSERT_EQ(i, 2);

  queueManager_->deleteResults(taskId, destination);
  queueManager_->removeTask(taskId);
  EXPECT_TRUE(task->isFinished());
}

TEST_F(UcxOutputQueueManagerTest, multiFetchers) {
  const std::vector<bool> earlyTerminations = {false, true};
  for (const auto earlyTermination : earlyTerminations) {
    SCOPED_TRACE(fmt::format("earlyTermination {}", earlyTermination));

    const vector_size_t size = 10;
    const std::string taskId =
        earlyTermination ? "multi-fetchers-early" : "multi-fetchers-complete";
    int numPartitions = 10;
    initializeTask(taskId, numPartitions, 1);

    // create one thread per partition to asynchronously fetch data.
    std::vector<std::thread> threads;
    std::vector<int64_t> fetchedPackedColumns(numPartitions, 0);

    for (size_t i = 0; i < numPartitions; ++i) {
      threads.emplace_back([&, i]() {
        dataFetcher(taskId, i, fetchedPackedColumns.at(i), earlyTermination);
      });
    }

    // enqueue packed columns.
    folly::Random::DefaultGenerator rng;
    rng.seed(1234);
    const int totalPackedColumns = 1000;
    std::vector<int64_t> producedPackedColumns(numPartitions, 0);
    for (int i = 0; i < totalPackedColumns; ++i) {
      // randomly chose a partition.
      const int partition = folly::Random().rand32(rng) % numPartitions;
      try {
        enqueue(taskId, partition, size);
      } catch (...) {
        // Early termination might cause the task to fail early.
        ASSERT_TRUE(earlyTermination);
        break;
      }
      ++producedPackedColumns[partition];
      if (folly::Random().oneIn(4)) {
        std::this_thread::sleep_for(std::chrono::microseconds(5)); // NOLINT
      }
    }
    noMoreData(taskId);

    for (int i = 0; i < threads.size(); ++i) {
      threads[i].join();
    }

    if (!earlyTermination) {
      for (int i = 0; i < numPartitions; ++i) {
        ASSERT_EQ(fetchedPackedColumns[i], producedPackedColumns[i]);
      }
    }
    queueManager_->removeTask(taskId);
  }
}

// Test BUG B scenario (a): Producer never starts / never calls
// initializeTask(). getData() creates a stub queue, then removeTask() is
// called without initializeTask() ever being called. The pending callback
// must fire with nullptr to unblock the consumer.
TEST_F(UcxOutputQueueManagerTest, callbackFiredOnTerminateBeforeInit) {
  // This must be a fresh ID: removeTask() deliberately tombstones absent IDs
  // so stale servers cannot recreate placeholder queues.
  const std::string taskId = "callbackBeforeInit.fresh";

  bool callbackFired = false;
  bool receivedNullptr = false;
  queueManager_->getData(
      taskId,
      0, // destination
      [&callbackFired, &receivedNullptr](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> remainingBytes) {
        callbackFired = true;
        receivedNullptr = (data == nullptr);
      });

  // Callback should NOT have fired yet (queue is empty, no data).
  EXPECT_FALSE(callbackFired);

  // Now simulate the producer failing: removeTask fires terminate().
  // The stub queue has task_ == nullptr so the isRunning() check is skipped.
  queueManager_->removeTask(taskId);

  // Callback must have been fired with nullptr.
  EXPECT_TRUE(callbackFired);
  EXPECT_TRUE(receivedNullptr);
}

TEST_F(UcxOutputQueueManagerTest, removedPlaceholderEndsNewDestinations) {
  const std::string taskId = "adaptive.transport.removedPlaceholder";
  bool originalEnded = false;
  auto stableQueue = queueManager_->getDataWithQueue(
      taskId,
      0,
      [&](std::shared_ptr<UcxOutputQueue> queue,
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t numRows,
          std::vector<int64_t> remainingBytes) {
        EXPECT_NE(queue, nullptr);
        EXPECT_EQ(data, nullptr);
        EXPECT_EQ(numRows, 0);
        EXPECT_TRUE(remainingBytes.empty());
        originalEnded = true;
      });
  ASSERT_NE(stableQueue, nullptr);
  EXPECT_FALSE(originalEnded);
  queueManager_->removeTask(taskId);
  EXPECT_TRUE(originalEnded);

  // Model getData retaining the exact queue immediately before removal, then
  // entering the queue after terminate() for a previously absent destination.
  bool lateDestinationEnded = false;
  stableQueue->getData(
      3,
      [&](std::shared_ptr<cudf::packed_columns> data,
          vector_size_t numRows,
          std::vector<int64_t> remainingBytes) {
        EXPECT_EQ(data, nullptr);
        EXPECT_EQ(numRows, 0);
        EXPECT_TRUE(remainingBytes.empty());
        lateDestinationEnded = true;
      });
  EXPECT_TRUE(lateDestinationEnded);
  // Also clear a pending callback if a regression made the assertion fail.
  stableQueue->terminate();
}

// Test BUG B scenario (b): Producer starts but crashes before noMoreData().
// initializeTask() was called but noMoreData() was never called.
// removeTask() must fire pending callbacks with nullptr.
TEST_F(UcxOutputQueueManagerTest, callbackFiredOnTerminateAfterInit) {
  const std::string taskId = "callbackAfterInit.fresh";

  auto task =
      initializeTask(taskId, 2 /* numDestinations */, 1 /* numDrivers */);

  // Register callbacks on both destinations.
  bool callback0Fired = false;
  bool callback0Nullptr = false;
  bool callback1Fired = false;
  bool callback1Nullptr = false;
  queueManager_->getData(
      taskId,
      0,
      [&callback0Fired, &callback0Nullptr](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> remainingBytes) {
        callback0Fired = true;
        callback0Nullptr = (data == nullptr);
      });
  queueManager_->getData(
      taskId,
      1,
      [&callback1Fired, &callback1Nullptr](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> remainingBytes) {
        callback1Fired = true;
        callback1Nullptr = (data == nullptr);
      });

  // Neither callback should fire yet.
  EXPECT_FALSE(callback0Fired);
  EXPECT_FALSE(callback1Fired);

  // Simulate task failure: abort the task so isRunning() returns false,
  // which is required by terminate()'s VELOX_CHECK.
  task->requestAbort().wait();

  queueManager_->removeTask(taskId);

  EXPECT_TRUE(callback0Fired);
  EXPECT_TRUE(callback0Nullptr);
  EXPECT_TRUE(callback1Fired);
  EXPECT_TRUE(callback1Nullptr);
}

TEST_F(UcxOutputQueueManagerTest, callbackFiredOnDeleteResults) {
  const std::string taskId = "callbackOnDeleteResults.fresh";
  auto task =
      initializeTask(taskId, 1 /* numDestinations */, 1 /* numDrivers */);

  bool callbackFired = false;
  queueManager_->getData(
      taskId,
      0,
      [&](std::shared_ptr<cudf::packed_columns> data,
          vector_size_t numRows,
          std::vector<int64_t> remainingBytes) {
        EXPECT_EQ(data, nullptr);
        EXPECT_EQ(numRows, 0);
        EXPECT_TRUE(remainingBytes.empty());
        callbackFired = true;
      });
  ASSERT_FALSE(callbackFired);

  queueManager_->deleteResults(taskId, 0);
  EXPECT_TRUE(callbackFired);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
}

// --- Broadcast tests ---

// Basic broadcast: enqueue data, all destinations receive the same data.
TEST_F(UcxOutputQueueManagerTest, broadcastBasic) {
  const vector_size_t size = 100;
  const std::string taskId = "broadcast0";
  const int numDestinations = 3;

  auto task = initializeTask(
      taskId,
      numDestinations,
      1 /* numDrivers */,
      core::PartitionedOutputNode::Kind::kBroadcast);

  // Broadcast must not be finished before noMoreQueues.
  EXPECT_FALSE(queueManager_->isFinished(taskId));

  // Tell the queue manager the final number of destinations.
  queueManager_->updateOutputBuffers(taskId, numDestinations, true);

  // Enqueue one batch (broadcast always uses destination 0).
  enqueue(taskId, 0, size);

  // All destinations should receive the data.
  for (int dest = 0; dest < numDestinations; ++dest) {
    fetch(taskId, dest);
  }

  // Signal no more data.
  noMoreData(taskId);

  // Fetch end markers from all destinations.
  for (int dest = 0; dest < numDestinations; ++dest) {
    fetchEndMarker(taskId, dest);
  }

  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}

// Broadcast: late destination receives backfilled data.
TEST_F(UcxOutputQueueManagerTest, broadcastLateDestination) {
  const vector_size_t size = 50;
  const std::string taskId = "broadcast1";

  // Start with 2 destinations.
  auto task = initializeTask(
      taskId,
      2 /* numDestinations */,
      1 /* numDrivers */,
      core::PartitionedOutputNode::Kind::kBroadcast);

  // Enqueue 2 batches before the 3rd destination arrives.
  enqueue(taskId, 0, size);
  enqueue(taskId, 0, size);

  // Fetch from existing destinations to verify data is there.
  fetch(taskId, 0);
  fetch(taskId, 1);

  // Now add a 3rd destination (late arrival). It should be backfilled.
  queueManager_->updateOutputBuffers(taskId, 3, false);

  // The new destination should have both batches.
  fetch(taskId, 2);
  fetch(taskId, 2);

  // Finalize destinations and data.
  queueManager_->updateOutputBuffers(taskId, 3, true);
  noMoreData(taskId);

  // Fetch remaining data + end markers.
  fetch(taskId, 0);
  fetchEndMarker(taskId, 0);
  fetch(taskId, 1);
  fetchEndMarker(taskId, 1);
  fetchEndMarker(taskId, 2);

  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}

// Broadcast: a destination can register before the coordinator publishes its
// buffer ID. Retained data must be available immediately, and the later
// buffer-count update must not duplicate it.
TEST_F(
    UcxOutputQueueManagerTest,
    broadcastDestinationRegistersBeforeBufferUpdate) {
  const vector_size_t size = 50;
  const std::string taskId = "broadcast-register-first";

  auto task = initializeTask(
      taskId,
      1 /* numDestinations */,
      1 /* numDrivers */,
      core::PartitionedOutputNode::Kind::kBroadcast);

  enqueue(taskId, 0, size);
  enqueue(taskId, 0, size);

  int receivedData = 0;
  for (int i = 0; i < 2; ++i) {
    queueManager_->getDataWithQueue(
        taskId,
        1,
        [&receivedData](
            std::shared_ptr<UcxOutputQueue> queue,
            std::shared_ptr<cudf::packed_columns> data,
            vector_size_t numRows,
            std::vector<int64_t> /*remainingBytes*/) {
          ASSERT_NE(data, nullptr);
          EXPECT_EQ(numRows, 50);
          queue->releaseInFlightBytes(
              1, static_cast<int64_t>(data->gpu_data->size()), 1);
          ++receivedData;
        });
    EXPECT_EQ(receivedData, i + 1);
  }

  queueManager_->updateOutputBuffers(taskId, 2, false);

  bool receivedEndMarker = false;
  queueManager_->getData(taskId, 1, receiveEndMarker(1, receivedEndMarker));
  EXPECT_FALSE(receivedEndMarker);

  // Publishing the already-created destination is idempotent and must not
  // manufacture another retained payload.
  queueManager_->updateOutputBuffers(taskId, 2, false);
  EXPECT_FALSE(receivedEndMarker);

  queueManager_->updateOutputBuffers(taskId, 2, true);
  EXPECT_ANY_THROW(queueManager_->getData(taskId, 2, [](auto, auto, auto) {}));

  noMoreData(taskId);
  EXPECT_TRUE(receivedEndMarker);
  deleteResults(taskId, 1);

  fetch(taskId, 0);
  fetch(taskId, 0);
  fetchEndMarker(taskId, 0);

  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}

// Broadcast: a destination registered after production finishes receives all
// retained data before exactly one end marker.
TEST_F(UcxOutputQueueManagerTest, broadcastLateRegistrationAfterEnd) {
  const std::string taskId = "broadcast-register-after-end";

  auto task = initializeTask(
      taskId,
      1 /* numDestinations */,
      1 /* numDrivers */,
      core::PartitionedOutputNode::Kind::kBroadcast);

  enqueue(taskId, 0, 10);
  noMoreData(taskId);

  fetch(taskId, 1);
  fetchEndMarker(taskId, 1);

  queueManager_->updateOutputBuffers(taskId, 2, true);
  fetch(taskId, 0);
  fetchEndMarker(taskId, 0);

  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}

// Broadcast: isFinished requires noMoreQueues.
TEST_F(UcxOutputQueueManagerTest, broadcastNotFinishedWithoutNoMoreQueues) {
  const std::string taskId = "broadcast2";

  auto task = initializeTask(
      taskId,
      2 /* numDestinations */,
      1 /* numDrivers */,
      core::PartitionedOutputNode::Kind::kBroadcast);

  // Signal no more data.
  noMoreData(taskId);

  // Fetch end markers and delete results from all destinations.
  fetchEndMarker(taskId, 0);
  fetchEndMarker(taskId, 1);

  // Still not finished because noMoreQueues hasn't been signaled.
  EXPECT_FALSE(queueManager_->isFinished(taskId));

  // Signal no more queues.
  queueManager_->updateOutputBuffers(taskId, 2, true);

  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}

// Broadcast: end marker propagated to late-arriving destinations.
TEST_F(UcxOutputQueueManagerTest, broadcastEndMarkerToLateDestination) {
  const std::string taskId = "broadcast3";

  auto task = initializeTask(
      taskId,
      1 /* numDestinations */,
      1 /* numDrivers */,
      core::PartitionedOutputNode::Kind::kBroadcast);

  enqueue(taskId, 0, 10);
  noMoreData(taskId);

  // Add a late destination after end marker was set.
  queueManager_->updateOutputBuffers(taskId, 2, true);

  // Destination 0: data + end marker.
  fetch(taskId, 0);
  fetchEndMarker(taskId, 0);

  // Destination 1 (late): backfilled data + end marker.
  fetch(taskId, 1);
  fetchEndMarker(taskId, 1);

  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}

TEST_F(UcxOutputQueueManagerTest, transferReservationBlocksAndWakes) {
  const std::string taskId = "adaptive.reservation";
  auto task = initializeTask(taskId, 2, 1);

  EXPECT_FALSE(
      queueManager_->reserveTransferBytes(taskId, 0, 100, 100, nullptr));
  ContinueFuture destinationFuture;
  EXPECT_TRUE(queueManager_->reserveTransferBytes(
      taskId, 0, 1, 100, &destinationFuture));
  EXPECT_FALSE(destinationFuture.isReady());

  // Destination windows are independent.
  EXPECT_FALSE(
      queueManager_->reserveTransferBytes(taskId, 1, 100, 100, nullptr));
  queueManager_->releaseTransferReservation(taskId, 0, 100);
  EXPECT_TRUE(destinationFuture.isReady());
  EXPECT_ANY_THROW(queueManager_->releaseTransferReservation(taskId, 0, 1));

  ContinueFuture cancelledFuture;
  EXPECT_TRUE(
      queueManager_->reserveTransferBytes(taskId, 1, 1, 100, &cancelledFuture));
  EXPECT_FALSE(cancelledFuture.isReady());
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
  EXPECT_TRUE(cancelledFuture.isReady());
}

TEST_F(UcxOutputQueueManagerTest, enqueueConsumesReservationExactlyOnce) {
  const std::string taskId = "adaptive.enqueueReservationCommit";
  auto task = initializeTask(taskId, 1, 1);
  auto page = makePackedColumns(10);
  const auto bytes = static_cast<int64_t>(page->gpu_data->size());
  ASSERT_GT(bytes, 0);

  std::shared_ptr<UcxOutputQueue> stableQueue;
  std::shared_ptr<cudf::packed_columns> received;
  queueManager_->getDataWithQueue(
      taskId,
      0,
      [&](std::shared_ptr<UcxOutputQueue> outputQueue,
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> /*remainingBytes*/) {
        stableQueue = std::move(outputQueue);
        received = std::move(data);
      });
  ASSERT_EQ(received, nullptr);

  EXPECT_FALSE(
      queueManager_->reserveTransferBytes(taskId, 0, bytes, bytes, nullptr));
  EXPECT_NO_THROW(queueManager_->enqueue(
      taskId, 0, std::move(page), /*numRows=*/10, bytes));
  ASSERT_NE(stableQueue, nullptr);
  ASSERT_NE(received, nullptr);

  // enqueue atomically converted the reservation into in-flight ownership.
  // A producer scope guard must therefore be dismissed after the call returns.
  EXPECT_ANY_THROW(queueManager_->releaseTransferReservation(taskId, 0, bytes));
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedBytes, bytes);
  stableQueue->releaseInFlightBytes(0, bytes, 1);
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedBytes, 0);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
}

TEST_F(UcxOutputQueueManagerTest, retainedUntilTransferCompletion) {
  const std::string taskId = "adaptive.inFlight";
  auto task = initializeTask(
      taskId,
      1,
      1,
      core::PartitionedOutputNode::Kind::kPartitioned,
      1 /* maxOutputBufferSize */);

  enqueue(taskId, 0, 10);
  const auto queuedStats = queueManager_->stats(taskId);
  ASSERT_TRUE(queuedStats.has_value());
  ASSERT_GT(queuedStats->bufferedBytes, 0);

  ContinueFuture blockedFuture;
  ASSERT_TRUE(queueManager_->checkBlocked(taskId, &blockedFuture));
  std::shared_ptr<UcxOutputQueue> stableQueue;
  std::shared_ptr<cudf::packed_columns> packet;
  queueManager_->getDataWithQueue(
      taskId,
      0,
      [&](std::shared_ptr<UcxOutputQueue> queue,
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> /*remainingBytes*/) {
        stableQueue = std::move(queue);
        packet = std::move(data);
      });
  ASSERT_NE(packet, nullptr);
  EXPECT_EQ(
      queueManager_->stats(taskId)->bufferedBytes, queuedStats->bufferedBytes);
  EXPECT_TRUE(queueManager_->checkBlocked(taskId, nullptr));
  EXPECT_FALSE(blockedFuture.isReady());

  stableQueue->releaseInFlightBytes(
      0, static_cast<int64_t>(packet->gpu_data->size()), 1);
  EXPECT_TRUE(blockedFuture.isReady());
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedBytes, 0);
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedPages, 0);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
}

TEST_F(UcxOutputQueueManagerTest, cancellationRetainsExactInFlightQueue) {
  const std::string taskId = "adaptive.transport.cancelInFlight";
  auto task = initializeTask(
      taskId, 1, 1, core::PartitionedOutputNode::Kind::kPartitioned, 1);
  auto server = std::make_shared<TestExchangeServerLifecycle>(taskId, 0);
  ASSERT_EQ(
      queueManager_->registerExchangeServer(server),
      ExchangeServerAdmission::kAccepted);
  enqueue(taskId, 0, 10);

  std::shared_ptr<UcxOutputQueue> stableQueue;
  std::shared_ptr<cudf::packed_columns> packet;
  auto returnedQueue = queueManager_->getDataWithQueue(
      taskId,
      0,
      [&](std::shared_ptr<UcxOutputQueue> queue,
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t numRows,
          std::vector<int64_t>) {
        EXPECT_EQ(numRows, 10);
        stableQueue = std::move(queue);
        packet = std::move(data);
      });
  ASSERT_NE(stableQueue, nullptr);
  ASSERT_NE(packet, nullptr);
  EXPECT_EQ(returnedQueue, stableQueue);
  const auto bytes = static_cast<int64_t>(packet->gpu_data->size());
  ASSERT_GT(bytes, 0);

  bool terminalCallback = false;
  queueManager_->getDataWithQueue(
      taskId,
      0,
      [&](std::shared_ptr<UcxOutputQueue> queue,
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t numRows,
          std::vector<int64_t> remainingBytes) {
        // Abort must be visible before terminate wakes a pending sender.
        EXPECT_EQ(server->abortRequests(), 1);
        EXPECT_EQ(queue, stableQueue);
        EXPECT_EQ(data, nullptr);
        EXPECT_EQ(numRows, 0);
        EXPECT_TRUE(remainingBytes.empty());
        terminalCallback = true;
      });
  EXPECT_FALSE(terminalCallback);
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
  EXPECT_TRUE(terminalCallback);
  EXPECT_EQ(queueManager_->stats(taskId), std::nullopt);
  EXPECT_EQ(stableQueue->stats().bufferedBytes, bytes);

  // Transfer completion owns the exact queue after its manager entry is gone.
  packet.reset();
  stableQueue->releaseInFlightBytes(0, bytes, 1);
  EXPECT_EQ(stableQueue->stats().bufferedBytes, 0);
  EXPECT_EQ(stableQueue->stats().bufferedPages, 0);
}

TEST_F(UcxOutputQueueManagerTest, adaptiveWindowDecreaseAndGrowth) {
  const std::string taskId = "adaptive.window";
  auto task = initializeTask(taskId, 1, 1);

  EXPECT_EQ(queueManager_->transferWindowBytes(taskId, 0, 100, 200, 800), 200);
  // Keep enough destination-local demand present that the getter doesn't
  // immediately perform idle recovery from the decreased window.
  EXPECT_FALSE(
      queueManager_->reserveTransferBytes(taskId, 0, 60, 800, nullptr));
  queueManager_->recordTransferCongestion(taskId, 0, 100);
  EXPECT_EQ(queueManager_->transferWindowBytes(taskId, 0, 100, 200, 800), 100);
  queueManager_->recordTransferDemand(taskId, 0, 350, 100, 800);
  EXPECT_EQ(queueManager_->transferWindowBytes(taskId, 0, 100, 200, 800), 350);
  queueManager_->recordTransferDemand(taskId, 0, 100, 100, 800);
  EXPECT_EQ(queueManager_->transferWindowBytes(taskId, 0, 100, 200, 800), 450);
  queueManager_->recordTransferCongestion(taskId, 0, 100);
  EXPECT_EQ(queueManager_->transferWindowBytes(taskId, 0, 100, 200, 800), 225);
  queueManager_->releaseTransferReservation(taskId, 0, 60);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
}

TEST_F(UcxOutputQueueManagerTest, zeroBytePageReleasesPackedColumn) {
  const std::string taskId = "adaptive.zeroByte";
  auto task = initializeTask(taskId, 1, 1);
  auto zeroColumnPage = makeZeroColumnPackedColumns();
  ASSERT_EQ(zeroColumnPage->gpu_data->size(), 0);
  queueManager_->enqueue(taskId, 0, std::move(zeroColumnPage), /*numRows=*/17);
  ASSERT_EQ(queueManager_->stats(taskId)->bufferedBytes, 0);
  ASSERT_EQ(queueManager_->stats(taskId)->bufferedPages, 1);

  std::shared_ptr<UcxOutputQueue> stableQueue;
  vector_size_t receivedRows = 0;
  queueManager_->getDataWithQueue(
      taskId,
      0,
      [&](std::shared_ptr<UcxOutputQueue> queue,
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t numRows,
          std::vector<int64_t> /*remainingBytes*/) {
        stableQueue = std::move(queue);
        ASSERT_NE(data, nullptr);
        EXPECT_EQ(data->gpu_data->size(), 0);
        receivedRows = numRows;
      });
  ASSERT_NE(stableQueue, nullptr);
  EXPECT_EQ(receivedRows, 17);
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedPages, 1);
  EXPECT_NO_THROW(stableQueue->releaseInFlightBytes(0, 0, 1));
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedPages, 0);
  EXPECT_ANY_THROW(stableQueue->releaseInFlightBytes(0, 0, 1));

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
}

TEST_F(UcxOutputQueueManagerTest, broadcastHistoryRetainsOneCopy) {
  const std::string taskId = "adaptive.broadcastHistory";
  auto task = initializeTask(
      taskId,
      1,
      1,
      core::PartitionedOutputNode::Kind::kBroadcast,
      1 /* maxOutputBufferSize */);

  enqueue(taskId, 0, 10);
  std::shared_ptr<UcxOutputQueue> stableQueue;
  std::shared_ptr<cudf::packed_columns> packet;
  queueManager_->getDataWithQueue(
      taskId,
      0,
      [&](std::shared_ptr<UcxOutputQueue> queue,
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> /*remainingBytes*/) {
        stableQueue = std::move(queue);
        packet = std::move(data);
      });
  ASSERT_NE(packet, nullptr);
  const auto bytes = static_cast<int64_t>(packet->gpu_data->size());
  ASSERT_GT(bytes, 0);
  stableQueue->releaseInFlightBytes(0, bytes, 1);

  // The current destination is complete, but the same physical GPU buffer is
  // still retained for a possible late broadcast destination.
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedBytes, bytes);
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedPages, 1);
  ContinueFuture historyFuture;
  EXPECT_TRUE(queueManager_->checkBlocked(taskId, &historyFuture));
  EXPECT_FALSE(historyFuture.isReady());

  queueManager_->updateOutputBuffers(taskId, 1, true);
  EXPECT_TRUE(historyFuture.isReady());
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedBytes, 0);
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedPages, 0);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
}

TEST_F(UcxOutputQueueManagerTest, removeTaskTombstoneIsIdempotent) {
  const std::string taskId = "lifecycle.idempotentTombstone";
  queueManager_->removeTask(taskId);
  queueManager_->removeTask(taskId);

  int staleCallbacks = 0;
  auto staleQueue = queueManager_->getDataWithQueue(
      taskId,
      0,
      [&](std::shared_ptr<UcxOutputQueue> outputQueue,
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> /*remainingBytes*/) {
        EXPECT_EQ(outputQueue, nullptr);
        EXPECT_EQ(data, nullptr);
        ++staleCallbacks;
      });
  EXPECT_EQ(staleQueue, nullptr);
  EXPECT_EQ(staleCallbacks, 1);

  // Tags contain no task generation; even explicit initialization cannot
  // safely reuse a removed ID while old messages may still be unmatched.
  auto task = createSourceTask(taskId, pool_, UcxTestData::kTestRowType);
  VELOX_ASSERT_THROW(
      queueManager_->initializeTask(
          task, core::PartitionedOutputNode::Kind::kPartitioned, 1, 1),
      "Cannot reuse removed GPU exchange task ID");
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
  EXPECT_EQ(queueManager_->stats(taskId), std::nullopt);
  auto removedQueue = queueManager_->getDataWithQueue(
      taskId,
      0,
      [&](std::shared_ptr<UcxOutputQueue> outputQueue,
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t numRows,
          std::vector<int64_t> remainingBytes) {
        EXPECT_EQ(outputQueue, nullptr);
        EXPECT_EQ(data, nullptr);
        EXPECT_EQ(numRows, 0);
        EXPECT_TRUE(remainingBytes.empty());
        ++staleCallbacks;
      });
  EXPECT_EQ(removedQueue, nullptr);
  EXPECT_EQ(staleCallbacks, 2);
}

TEST_F(UcxOutputQueueManagerTest, maximumConfiguredCapacityDoesNotOverflow) {
  constexpr uint64_t kMaximumCapacity =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  auto directTask = createSourceTask(
      "capacity.maximum.direct",
      pool_,
      UcxTestData::kTestRowType,
      kMaximumCapacity);
  std::shared_ptr<UcxOutputQueue> directQueue;
  EXPECT_NO_THROW(
      directQueue = std::make_shared<UcxOutputQueue>(directTask, 1, 1));
  ASSERT_NE(directQueue, nullptr);
  EXPECT_EQ(directQueue->getUtilization(), 0.0);

  auto initializedTask = createSourceTask(
      "capacity.maximum.initialize",
      pool_,
      UcxTestData::kTestRowType,
      kMaximumCapacity);
  auto placeholder = std::make_shared<UcxOutputQueue>(nullptr, 1, 0);
  bool initialized = false;
  EXPECT_NO_THROW(initialized = placeholder->initialize(initializedTask, 1, 1));
  EXPECT_TRUE(initialized);
  EXPECT_EQ(placeholder->getUtilization(), 0.0);
}

TEST_F(UcxOutputQueueManagerTest, fullTransferWindowRecoveryIsBounded) {
  const std::string taskId = "adaptive.fullWindowRecovery";
  auto task = initializeTask(
      taskId,
      2,
      1,
      core::PartitionedOutputNode::Kind::kPartitioned,
      100 /* maxOutputBufferSize */);

  queueManager_->recordFullTransferCongestion(taskId);
  auto diagnostics = queueManager_->toString(taskId);
  EXPECT_NE(
      diagnostics.find("fullTransferRetainedLimit=175"), std::string::npos);
  EXPECT_NE(diagnostics.find("fullTransferCongested=1"), std::string::npos);

  EXPECT_FALSE(queueManager_->reserveFullTransferBytes(taskId, 0, 50, nullptr));
  // With 50 retained bytes, the 175-byte learned window is below half full.
  // One additive recovery step reaches the 200-byte default (100 * 2
  // destinations) and exits custom congestion mode.
  EXPECT_FALSE(queueManager_->waitForFullTransferCapacity(taskId, 1, nullptr));
  const auto recovered = queueManager_->toString(taskId);
  EXPECT_NE(recovered.find("fullTransferRetainedLimit=200"), std::string::npos);
  EXPECT_NE(recovered.find("fullTransferCongested=0"), std::string::npos);
  EXPECT_FALSE(queueManager_->waitForFullTransferCapacity(taskId, 1, nullptr));
  EXPECT_EQ(queueManager_->toString(taskId), recovered);

  queueManager_->releaseTransferReservation(taskId, 0, 50);
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
}

TEST_F(UcxOutputQueueManagerTest, fullTransferCanAdmitMaterializedAggregate) {
  const std::string taskId = "adaptive.fullWindowAggregateAdmission";
  auto task = initializeTask(
      taskId,
      2,
      1,
      core::PartitionedOutputNode::Kind::kPartitioned,
      100 /* maxOutputBufferSize */);

  EXPECT_FALSE(
      queueManager_->reserveFullTransferBytes(taskId, 0, 150, nullptr));
  // The already-materialized full split is larger than the ordinary
  // 100-bytes-per-destination aggregate. Before any congestion signal, admit
  // it exactly rather than deadlocking on memory that is already resident.
  EXPECT_FALSE(
      queueManager_->reserveFullTransferBytes(taskId, 1, 100, nullptr));
  queueManager_->releaseTransferReservation(taskId, 0, 150);
  queueManager_->releaseTransferReservation(taskId, 1, 100);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
}

TEST_F(
    UcxOutputQueueManagerTest,
    fullTransferWindowClampsAfterDestinationDelete) {
  const std::string taskId = "adaptive.fullWindowDestinationShrink";
  auto task = initializeTask(
      taskId,
      4,
      1,
      core::PartitionedOutputNode::Kind::kPartitioned,
      100 /* maxOutputBufferSize */);

  EXPECT_FALSE(
      queueManager_->reserveFullTransferBytes(taskId, 0, 300, nullptr));
  queueManager_->recordFullTransferCongestion(taskId);
  EXPECT_NE(
      queueManager_->toString(taskId).find("fullTransferRetainedLimit=263"),
      std::string::npos);

  queueManager_->deleteResults(taskId, 1);
  queueManager_->deleteResults(taskId, 2);
  queueManager_->deleteResults(taskId, 3);
  const auto shrunk = queueManager_->toString(taskId);
  EXPECT_NE(shrunk.find("fullTransferRetainedLimit=100"), std::string::npos);
  EXPECT_NE(shrunk.find("fullTransferCongested=1"), std::string::npos);
  EXPECT_TRUE(queueManager_->waitForFullTransferCapacity(taskId, 1, nullptr));

  queueManager_->releaseTransferReservation(taskId, 0, 300);
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
}

TEST_F(UcxOutputQueueManagerTest, broadcastInFlightIsRetained) {
  const std::string taskId = "adaptive.broadcastInFlight";
  auto task = initializeTask(
      taskId,
      2,
      1,
      core::PartitionedOutputNode::Kind::kBroadcast,
      1 /* maxOutputBufferSize */);
  // Disable late-destination history so this case isolates the two committed
  // sends. Broadcast has no transfer reservations, so base backpressure must
  // account for them directly.
  queueManager_->updateOutputBuffers(taskId, 2, true);
  enqueue(taskId, 0, 10);

  std::vector<std::shared_ptr<cudf::packed_columns>> packets(2);
  std::shared_ptr<UcxOutputQueue> stableQueue;
  for (int destination = 0; destination < 2; ++destination) {
    queueManager_->getDataWithQueue(
        taskId,
        destination,
        [&, destination](
            std::shared_ptr<UcxOutputQueue> queue,
            std::shared_ptr<cudf::packed_columns> data,
            vector_size_t /*numRows*/,
            std::vector<int64_t> /*remainingBytes*/) {
          stableQueue = std::move(queue);
          packets[destination] = std::move(data);
        });
    ASSERT_NE(packets[destination], nullptr);
  }

  ContinueFuture retainedFuture;
  ASSERT_TRUE(queueManager_->checkBlocked(taskId, &retainedFuture));
  EXPECT_FALSE(retainedFuture.isReady());
  stableQueue->releaseInFlightBytes(
      0, static_cast<int64_t>(packets[0]->gpu_data->size()), 1);
  EXPECT_FALSE(retainedFuture.isReady());
  EXPECT_TRUE(queueManager_->checkBlocked(taskId, nullptr));
  stableQueue->releaseInFlightBytes(
      1, static_cast<int64_t>(packets[1]->gpu_data->size()), 1);
  EXPECT_TRUE(retainedFuture.isReady());
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedBytes, 0);
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedPages, 0);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
}

TEST_F(UcxOutputQueueManagerTest, arbitraryPreservesRowsAndTransferAccounting) {
  const std::string taskId = "adaptive.transport.arbitrary";
  auto task = initializeTask(
      taskId, 2, 1, core::PartitionedOutputNode::Kind::kArbitrary);
  std::vector<vector_size_t> received;
  auto receive = [&](int destination) {
    queueManager_->getDataWithQueue(
        taskId,
        destination,
        [&, destination](
            std::shared_ptr<UcxOutputQueue> queue,
            std::shared_ptr<cudf::packed_columns> data,
            vector_size_t rows,
            std::vector<int64_t>) {
          ASSERT_NE(data, nullptr);
          received.push_back(rows);
          queue->releaseInFlightBytes(
              destination, static_cast<int64_t>(data->gpu_data->size()), 1);
        });
  };
  // A waiting destination receives the first page. The next page is retained
  // once in the shared pool and can be claimed by either destination.
  receive(1);
  queueManager_->enqueue(taskId, 0, makeZeroColumnPackedColumns(), 17);
  EXPECT_EQ(received, std::vector<vector_size_t>{17});
  queueManager_->enqueue(taskId, 0, makeZeroColumnPackedColumns(), 23);
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedPages, 1);
  receive(0);
  EXPECT_EQ(received, (std::vector<vector_size_t>{17, 23}));
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedPages, 0);
  noMoreData(taskId);
  fetchEndMarker(taskId, 0);
  fetchEndMarker(taskId, 1);
  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}

TEST_F(UcxOutputQueueManagerTest, arbitraryDrainsSharedPagesBeforeEnd) {
  const std::string taskId = "adaptive.transport.arbitrary.end";
  auto task = initializeTask(
      taskId, 2, 1, core::PartitionedOutputNode::Kind::kArbitrary);
  queueManager_->enqueue(taskId, 0, makeZeroColumnPackedColumns(), 11);
  queueManager_->enqueue(taskId, 0, makeZeroColumnPackedColumns(), 19);
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedPages, 2);
  noMoreData(taskId);
  fetch(taskId, 0);
  fetch(taskId, 1);
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedPages, 0);
  fetchEndMarker(taskId, 0);
  fetchEndMarker(taskId, 1);
  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}
