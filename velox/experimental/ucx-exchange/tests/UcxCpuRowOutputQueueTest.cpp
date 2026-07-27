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

#include <atomic>

#include <folly/io/IOBuf.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/core/PlanFragment.h"
#include "velox/core/QueryCtx.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowOutputQueueManager.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowPartitionedOutput.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"
#include "velox/serializers/PrestoSerializer.h"

namespace facebook::velox::ucx_exchange {

class UcxCpuRowOutputQueueTest : public testing::Test {
 protected:
  using Destination = UcxCpuRowPartitionedOutput::Destination;

  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    if (!isRegisteredNamedVectorSerde("Presto")) {
      serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
    }
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
  }

  std::shared_ptr<exec::Task> makeTask(const std::string& taskId) {
    const auto rowType = ROW({"c0"}, {BIGINT()});
    auto input =
        BaseVector::create<RowVector>(rowType, /*size=*/1, pool_.get());
    core::PlanFragment plan{std::make_shared<core::ValuesNode>(
        "values", std::vector<RowVectorPtr>{std::move(input)})};
    auto queryCtx =
        core::QueryCtx::Builder()
            .queryConfig(
                core::QueryConfig{{
                    {core::QueryConfig::kMaxOutputBufferSize, "1073741824"},
                }})
            .build();
    return exec::Task::create(
        taskId,
        std::move(plan),
        /*destination=*/0,
        std::move(queryCtx),
        exec::Task::ExecutionMode::kSerial,
        exec::Consumer{});
  }

  static std::unique_ptr<UcxCpuRowPayload> makePayload(std::string_view bytes) {
    auto payload = std::make_unique<UcxCpuRowPayload>();
    payload->data = folly::IOBuf::copyBuffer(bytes);
    payload->numRows = 1;
    payload->numBytes = bytes.size();
    return payload;
  }

  std::shared_ptr<memory::MemoryPool> pool_;
};

class TestCpuRowExchangeServerLifecycle
    : public UcxCpuRowExchangeServerLifecycle {
 public:
  TestCpuRowExchangeServerLifecycle(std::string taskId, uint32_t destination)
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

  void completeClose() {
    closed_.store(true);
  }

 private:
  const PartitionKey key_;
  std::atomic<int> abortRequests_{0};
  std::atomic<bool> abortRequested_{false};
  std::atomic<bool> closed_{false};
  std::atomic<bool> activated_{false};
};

using ExchangeServerAdmission =
    UcxCpuRowOutputQueueManager::ExchangeServerAdmission;

TEST(CpuRowHandshakeResponseTest, rejectionIsCompatibleWithVersionOneReaders) {
  CpuRowHandshakeResponseHeader accepted;
  EXPECT_EQ(accepted.status, CpuRowHandshakeResponseStatus::kAccepted);
  EXPECT_EQ(accepted.dataEndpointMode, CpuRowDataEndpointMode::kBootstrap);

  CpuRowHandshakeResponseHeader rejected;
  rejected.status = CpuRowHandshakeResponseStatus::kTaskRemoved;
  rejected.dataEndpointMode = CpuRowDataEndpointMode::kRejected;
  rejected.serverWorkerAddressBytes = 0;
  rejected.serverHostIdHash = 0;

  EXPECT_NE(rejected.dataEndpointMode, CpuRowDataEndpointMode::kBootstrap);
  EXPECT_NE(
      rejected.dataEndpointMode,
      CpuRowDataEndpointMode::kSameHostWorkerAddress);
  EXPECT_EQ(sizeof(rejected), sizeof(accepted));
  EXPECT_EQ(rejected.version, kCpuRowHandshakeResponseVersion);
}

TEST_F(UcxCpuRowOutputQueueTest, exportedPageRetainsReleaseOwner) {
  const std::string taskId = "exported-page-retains-release-owner";
  auto task = makeTask(taskId);
  auto queueManager = std::make_shared<UcxCpuRowOutputQueueManager>();
  queueManager->initializeTask(
      task,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1);

  auto child =
      BaseVector::create<FlatVector<int64_t>>(BIGINT(), 1, pool_.get());
  child->set(0, 42);
  auto output = std::make_shared<RowVector>(
      pool_.get(),
      ROW({"c0"}, {BIGINT()}),
      nullptr,
      1,
      std::vector<VectorPtr>{child});

  Destination destination(
      taskId,
      /*destination=*/0,
      getNamedVectorSerde("Presto"),
      /*serdeOptions=*/nullptr,
      pool_.get(),
      /*eagerFlush=*/false,
      UcxCpuRowPartitionedOutput::kTargetNumRows,
      queueManager,
      [](uint64_t /*bytes*/, uint64_t /*rows*/) {});
  destination.beginBatch();
  destination.addRow(0);

  std::vector<vector_size_t> sizes{sizeof(int64_t)};
  bool atEnd = false;
  Scratch scratch;
  auto releaseOwner = std::make_shared<int>(42);
  std::weak_ptr<int> weakReleaseOwner = releaseOwner;
  std::function<void()> releaseFn = [releaseOwner]() {};

  EXPECT_EQ(
      destination.advance(
          1 << 20,
          sizes,
          output,
          releaseFn,
          &atEnd,
          /*future=*/nullptr,
          scratch),
      exec::BlockingReason::kNotBlocked);
  EXPECT_TRUE(atEnd);
  EXPECT_EQ(
      destination.flush(releaseFn, /*future=*/nullptr),
      exec::BlockingReason::kNotBlocked);

  auto payload = queueManager->tryGetData(taskId, /*destination=*/0);
  ASSERT_NE(payload, nullptr);
  ASSERT_NE(payload->data, nullptr);

  releaseOwner.reset();
  releaseFn = nullptr;
  EXPECT_FALSE(weakReleaseOwner.expired());

  // The release callback is part of the exported IOBuf ownership. It must
  // remain alive after the serializer and caller drop their references, then
  // release only after the transport drops the page.
  payload.reset();
  EXPECT_TRUE(weakReleaseOwner.expired());

  task->requestAbort().wait();
  queueManager->removeTask(taskId);
}

TEST_F(
    UcxCpuRowOutputQueueTest,
    abnormalTaskRemovalAbortsAllAndOnlyMatchingServers) {
  auto queueManager = std::make_shared<UcxCpuRowOutputQueueManager>();
  auto taskA = makeTask("aborted-task-a");
  auto taskB = makeTask("unrelated-task-b");
  queueManager->initializeTask(
      taskA,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/2,
      /*numDrivers=*/1);
  queueManager->initializeTask(
      taskB,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1);

  auto a0 =
      std::make_shared<TestCpuRowExchangeServerLifecycle>(taskA->taskId(), 0);
  auto a0Duplicate =
      std::make_shared<TestCpuRowExchangeServerLifecycle>(taskA->taskId(), 0);
  auto a1 =
      std::make_shared<TestCpuRowExchangeServerLifecycle>(taskA->taskId(), 1);
  auto b0 =
      std::make_shared<TestCpuRowExchangeServerLifecycle>(taskB->taskId(), 0);
  EXPECT_EQ(
      queueManager->registerExchangeServer(a0),
      ExchangeServerAdmission::kAccepted);
  EXPECT_EQ(
      queueManager->registerExchangeServer(a0Duplicate),
      ExchangeServerAdmission::kDuplicateServer);
  EXPECT_EQ(
      queueManager->registerExchangeServer(a1),
      ExchangeServerAdmission::kAccepted);
  EXPECT_EQ(
      queueManager->registerExchangeServer(b0),
      ExchangeServerAdmission::kAccepted);
  EXPECT_TRUE(a0->isActivated());
  EXPECT_FALSE(a0Duplicate->isActivated());
  EXPECT_TRUE(a1->isActivated());
  EXPECT_TRUE(b0->isActivated());
  EXPECT_EQ(a0Duplicate->abortRequests(), 1);

  taskA->requestAbort().wait();
  queueManager->removeTask(taskA->taskId());

  EXPECT_EQ(a0->abortRequests(), 1);
  EXPECT_EQ(a0Duplicate->abortRequests(), 1);
  EXPECT_EQ(a1->abortRequests(), 1);
  EXPECT_EQ(b0->abortRequests(), 0);

  // Repeated removal must preserve the tombstone. requestAbort() is
  // idempotent while the servers remain registered until close completes.
  queueManager->removeTask(taskA->taskId());
  EXPECT_EQ(a0->abortRequests(), 1);
  EXPECT_EQ(a0Duplicate->abortRequests(), 1);
  EXPECT_EQ(a1->abortRequests(), 1);

  auto lateA =
      std::make_shared<TestCpuRowExchangeServerLifecycle>(taskA->taskId(), 0);
  EXPECT_EQ(
      queueManager->registerExchangeServer(lateA),
      ExchangeServerAdmission::kTaskRemoved);
  EXPECT_EQ(lateA->abortRequests(), 1);

  // Unregistering the unrelated server prevents later task cleanup from
  // touching a server that has already completed normally.
  queueManager->unregisterExchangeServer(b0);
  queueManager->unregisterExchangeServer(b0);
  taskB->requestAbort().wait();
  queueManager->removeTask(taskB->taskId());
  EXPECT_EQ(b0->abortRequests(), 0);
}

TEST_F(UcxCpuRowOutputQueueTest, removedTaskTombstoneRejectsReinitialization) {
  const std::string taskId = "removed-before-initialization";
  auto queueManager = std::make_shared<UcxCpuRowOutputQueueManager>();

  queueManager->removeTask(taskId);
  queueManager->removeTask(taskId);

  auto rejected =
      std::make_shared<TestCpuRowExchangeServerLifecycle>(taskId, 0);
  EXPECT_EQ(
      queueManager->registerExchangeServer(rejected),
      ExchangeServerAdmission::kTaskRemoved);
  EXPECT_EQ(rejected->abortRequests(), 1);

  auto task = makeTask(taskId);
  EXPECT_ANY_THROW(queueManager->initializeTask(
      task,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1));
  task->requestAbort().wait();
}

TEST_F(
    UcxCpuRowOutputQueueTest,
    finishedTaskServersDrainInsteadOfBeingAborted) {
  const std::string taskId = "normally-finished-task";
  auto queueManager = std::make_shared<UcxCpuRowOutputQueueManager>();
  auto task = makeTask(taskId);
  queueManager->initializeTask(
      task,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1);

  auto active = std::make_shared<TestCpuRowExchangeServerLifecycle>(taskId, 0);
  EXPECT_EQ(
      queueManager->registerExchangeServer(active),
      ExchangeServerAdmission::kAccepted);

  task->testingFinish();
  queueManager->removeTask(taskId);
  EXPECT_EQ(active->abortRequests(), 0);

  auto reusedTask = makeTask(taskId);
  EXPECT_ANY_THROW(queueManager->initializeTask(
      reusedTask,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1));

  queueManager->unregisterExchangeServer(active);

  // A server admitted before normal removal drains its final marker. A server
  // arriving after the finished tombstone is stale and must never start an
  // empty stream.
  auto late = std::make_shared<TestCpuRowExchangeServerLifecycle>(taskId, 0);
  EXPECT_EQ(
      queueManager->registerExchangeServer(late),
      ExchangeServerAdmission::kTaskRemoved);
  EXPECT_FALSE(late->isActivated());
  EXPECT_EQ(late->abortRequests(), 1);

  // UCX tags contain no generation. Cleanup makes the old server reclaimable
  // but cannot prove that no unexpected old message remains in the worker.
  EXPECT_ANY_THROW(queueManager->initializeTask(
      reusedTask,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1));
  reusedTask->requestAbort().wait();
}

TEST_F(UcxCpuRowOutputQueueTest, taskIdTombstoneOutlivesAbortedServerCleanup) {
  const std::string taskId = "tombstone-after-asynchronous-abort";
  auto queueManager = std::make_shared<UcxCpuRowOutputQueueManager>();
  auto task = makeTask(taskId);
  queueManager->initializeTask(
      task,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1);

  auto draining =
      std::make_shared<TestCpuRowExchangeServerLifecycle>(taskId, 0);
  EXPECT_EQ(
      queueManager->registerExchangeServer(draining),
      ExchangeServerAdmission::kAccepted);

  task->requestAbort().wait();
  queueManager->removeTask(taskId);
  EXPECT_EQ(draining->abortRequests(), 1);
  EXPECT_FALSE(draining->isClosed());

  auto reusedTask = makeTask(taskId);
  EXPECT_ANY_THROW(queueManager->initializeTask(
      reusedTask,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1));

  draining->completeClose();
  queueManager->unregisterExchangeServer(draining);
  EXPECT_ANY_THROW(queueManager->initializeTask(
      reusedTask,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1));
  reusedTask->requestAbort().wait();
}

TEST_F(
    UcxCpuRowOutputQueueTest,
    activationFailureRetainsRegistrationUntilClose) {
  const std::string taskId = "abort-before-activation";
  auto queueManager = std::make_shared<UcxCpuRowOutputQueueManager>();

  auto aborting =
      std::make_shared<TestCpuRowExchangeServerLifecycle>(taskId, 0);
  aborting->requestAbort();
  EXPECT_EQ(
      queueManager->registerExchangeServer(aborting),
      ExchangeServerAdmission::kServerUnavailable);
  EXPECT_EQ(aborting->abortRequests(), 1);

  auto duplicate =
      std::make_shared<TestCpuRowExchangeServerLifecycle>(taskId, 0);
  EXPECT_EQ(
      queueManager->registerExchangeServer(duplicate),
      ExchangeServerAdmission::kDuplicateServer);
  EXPECT_EQ(duplicate->abortRequests(), 1);

  aborting->completeClose();
  queueManager->unregisterExchangeServer(aborting);

  auto replacement =
      std::make_shared<TestCpuRowExchangeServerLifecycle>(taskId, 0);
  EXPECT_EQ(
      queueManager->registerExchangeServer(replacement),
      ExchangeServerAdmission::kDuplicateServer);
  EXPECT_EQ(replacement->abortRequests(), 1);

  queueManager->removeTask(taskId);
  auto afterRemoval =
      std::make_shared<TestCpuRowExchangeServerLifecycle>(taskId, 0);
  EXPECT_EQ(
      queueManager->registerExchangeServer(afterRemoval),
      ExchangeServerAdmission::kTaskRemoved);
}

TEST_F(UcxCpuRowOutputQueueTest, closedServerKeepsKeyClaimUntilTaskRemoval) {
  const std::string taskId = "closed-active-task-server";
  auto queueManager = std::make_shared<UcxCpuRowOutputQueueManager>();

  auto server = std::make_shared<TestCpuRowExchangeServerLifecycle>(taskId, 0);
  EXPECT_EQ(
      queueManager->registerExchangeServer(server),
      ExchangeServerAdmission::kAccepted);
  server->completeClose();
  queueManager->unregisterExchangeServer(server);

  auto retry = std::make_shared<TestCpuRowExchangeServerLifecycle>(taskId, 0);
  EXPECT_EQ(
      queueManager->registerExchangeServer(retry),
      ExchangeServerAdmission::kDuplicateServer);

  queueManager->removeTask(taskId);
  auto afterRemoval =
      std::make_shared<TestCpuRowExchangeServerLifecycle>(taskId, 0);
  EXPECT_EQ(
      queueManager->registerExchangeServer(afterRemoval),
      ExchangeServerAdmission::kTaskRemoved);
}

TEST_F(
    UcxCpuRowOutputQueueTest,
    closedBeforeRegistrationClaimsKeyUntilTaskRemoval) {
  const std::string taskId = "closed-before-registration";
  auto queueManager = std::make_shared<UcxCpuRowOutputQueueManager>();

  auto closed = std::make_shared<TestCpuRowExchangeServerLifecycle>(taskId, 0);
  closed->requestAbort();
  closed->completeClose();
  EXPECT_EQ(
      queueManager->registerExchangeServer(closed),
      ExchangeServerAdmission::kServerUnavailable);

  auto retry = std::make_shared<TestCpuRowExchangeServerLifecycle>(taskId, 0);
  EXPECT_EQ(
      queueManager->registerExchangeServer(retry),
      ExchangeServerAdmission::kDuplicateServer);

  queueManager->removeTask(taskId);
  auto afterRemoval =
      std::make_shared<TestCpuRowExchangeServerLifecycle>(taskId, 0);
  EXPECT_EQ(
      queueManager->registerExchangeServer(afterRemoval),
      ExchangeServerAdmission::kTaskRemoved);
}

TEST_F(UcxCpuRowOutputQueueTest, serverMayArriveBeforeFirstTaskInitialization) {
  const std::string taskId = "server-before-task-initialization";
  auto queueManager = std::make_shared<UcxCpuRowOutputQueueManager>();
  auto early = std::make_shared<TestCpuRowExchangeServerLifecycle>(taskId, 0);
  EXPECT_EQ(
      queueManager->registerExchangeServer(early),
      ExchangeServerAdmission::kAccepted);

  auto task = makeTask(taskId);
  queueManager->initializeTask(
      task,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1);
  EXPECT_EQ(early->abortRequests(), 0);

  task->requestAbort().wait();
  queueManager->removeTask(taskId);
  EXPECT_EQ(early->abortRequests(), 1);
}

TEST_F(
    UcxCpuRowOutputQueueTest,
    lateBroadcastDestinationReceivesBackfillAfterRegisteringFirst) {
  auto task = makeTask("late-broadcast-destination");
  auto queue = std::make_shared<UcxCpuRowOutputQueue>(
      task,
      /*numDestinations=*/1,
      /*numDrivers=*/1,
      core::PartitionedOutputNode::Kind::kBroadcast);

  constexpr std::string_view kData = "broadcast-before-registration";
  auto payload = makePayload(kData);
  const auto* expectedPayload = payload.get();
  queue->enqueue(/*destination=*/0, std::move(payload), /*numRows=*/1);

  int callbackCount = 0;
  std::shared_ptr<UcxCpuRowPayload> received;
  queue->getData(
      /*destination=*/1,
      [&](std::shared_ptr<UcxCpuRowPayload> data,
          std::vector<int64_t> /* remainingBytes */) {
        ++callbackCount;
        received = std::move(data);
      });

  EXPECT_EQ(callbackCount, 1);
  EXPECT_EQ(received.get(), expectedPayload);

  // The exchange server can register a destination before the coordinator
  // announces the new buffer count. The later update must not leave that
  // already-created destination without the preceding broadcast payload.
  queue->updateOutputBuffers(/*numBuffers=*/2, /*noMoreBuffers=*/false);

  EXPECT_EQ(callbackCount, 1);
  EXPECT_EQ(received.get(), expectedPayload);
  if (received != nullptr) {
    EXPECT_EQ(received->numRows, 1);
    EXPECT_EQ(received->numBytes, kData.size());
  }

  task->requestAbort().wait();
  queue->terminate();
}

TEST_F(
    UcxCpuRowOutputQueueTest,
    lateBroadcastDestinationAfterNoMoreDataReceivesPayloadsThenOneEnd) {
  auto task = makeTask("late-broadcast-after-end");
  auto queue = std::make_shared<UcxCpuRowOutputQueue>(
      task,
      /*numDestinations=*/1,
      /*numDrivers=*/1,
      core::PartitionedOutputNode::Kind::kBroadcast);

  auto first = makePayload("first");
  const auto* expectedFirst = first.get();
  queue->enqueue(/*destination=*/0, std::move(first), /*numRows=*/1);

  auto second = makePayload("second");
  const auto* expectedSecond = second.get();
  queue->enqueue(/*destination=*/0, std::move(second), /*numRows=*/1);
  queue->noMoreData();

  int callbackCount = 0;
  int endCount = 0;
  std::vector<std::shared_ptr<UcxCpuRowPayload>> received;
  auto getData = [&]() {
    queue->getData(
        /*destination=*/1,
        [&](std::shared_ptr<UcxCpuRowPayload> data,
            std::vector<int64_t> /* remainingBytes */) {
          ++callbackCount;
          if (data == nullptr) {
            ++endCount;
          } else {
            received.push_back(std::move(data));
          }
        });
  };

  // The first request registers the late destination before the coordinator
  // announces it. The retained payload must be immediately available.
  getData();
  queue->updateOutputBuffers(/*numBuffers=*/2, /*noMoreBuffers=*/false);
  getData();
  getData();

  ASSERT_EQ(received.size(), 2);
  EXPECT_EQ(received[0].get(), expectedFirst);
  EXPECT_EQ(received[1].get(), expectedSecond);
  EXPECT_EQ(callbackCount, 3);
  EXPECT_EQ(endCount, 1);

  // A fourth request parks instead of consuming a duplicate end marker.
  getData();
  EXPECT_EQ(callbackCount, 3);
  EXPECT_EQ(endCount, 1);

  task->requestAbort().wait();
  queue->terminate();
}

TEST_F(
    UcxCpuRowOutputQueueTest,
    rejectsGrowthAfterNoMoreBuffersWithoutBreakingExistingDestination) {
  auto task = makeTask("broadcast-growth-after-no-more-buffers");
  auto queue = std::make_shared<UcxCpuRowOutputQueue>(
      task,
      /*numDestinations=*/1,
      /*numDrivers=*/1,
      core::PartitionedOutputNode::Kind::kBroadcast);

  auto payload = makePayload("existing-destination");
  const auto* expectedPayload = payload.get();
  queue->enqueue(/*destination=*/0, std::move(payload), /*numRows=*/1);
  queue->updateOutputBuffers(/*numBuffers=*/1, /*noMoreBuffers=*/true);

  EXPECT_THROW(
      queue->updateOutputBuffers(/*numBuffers=*/2, /*noMoreBuffers=*/true),
      VeloxRuntimeError);
  EXPECT_THROW(
      queue->getData(
          /*destination=*/1,
          [](std::shared_ptr<UcxCpuRowPayload>,
             std::vector<int64_t> /* remainingBytes */) {}),
      VeloxRuntimeError);

  int callbackCount = 0;
  std::shared_ptr<UcxCpuRowPayload> received;
  queue->getData(
      /*destination=*/0,
      [&](std::shared_ptr<UcxCpuRowPayload> data,
          std::vector<int64_t> /* remainingBytes */) {
        ++callbackCount;
        received = std::move(data);
      });
  EXPECT_EQ(callbackCount, 1);
  EXPECT_EQ(received.get(), expectedPayload);

  task->requestAbort().wait();
  queue->terminate();
}

TEST_F(
    UcxCpuRowOutputQueueTest,
    updateFirstBroadcastBackfillAccountsPerDestinationOnlyOnce) {
  auto task = makeTask("broadcast-backfill-stats");
  auto queue = std::make_shared<UcxCpuRowOutputQueue>(
      task,
      /*numDestinations=*/1,
      /*numDrivers=*/1,
      core::PartitionedOutputNode::Kind::kBroadcast);

  constexpr std::string_view kData = "broadcast-stats";
  const auto numBytes = static_cast<int64_t>(kData.size());
  queue->enqueue(
      /*destination=*/0, makePayload(kData), /*numRows=*/1);
  queue->updateOutputBuffers(/*numBuffers=*/3, /*noMoreBuffers=*/false);

  auto stats = queue->stats();
  EXPECT_EQ(stats.bufferedBytes, 3 * numBytes);
  EXPECT_EQ(stats.bufferedPages, 3);
  EXPECT_EQ(stats.totalBytesSent, numBytes);
  EXPECT_EQ(stats.totalRowsSent, 1);
  EXPECT_EQ(stats.totalPagesSent, 1);
  ASSERT_EQ(stats.buffersStats.size(), 3);
  for (const auto& bufferStats : stats.buffersStats) {
    EXPECT_EQ(bufferStats.bytesBuffered, numBytes);
    EXPECT_EQ(bufferStats.rowsBuffered, 1);
    EXPECT_EQ(bufferStats.pagesBuffered, 1);
  }

  // Repeating the same buffer-count announcement is not additional growth.
  queue->updateOutputBuffers(/*numBuffers=*/3, /*noMoreBuffers=*/false);
  stats = queue->stats();
  EXPECT_EQ(stats.bufferedBytes, 3 * numBytes);
  EXPECT_EQ(stats.bufferedPages, 3);
  EXPECT_EQ(stats.totalBytesSent, numBytes);
  EXPECT_EQ(stats.totalRowsSent, 1);
  EXPECT_EQ(stats.totalPagesSent, 1);

  task->requestAbort().wait();
  queue->terminate();
}

} // namespace facebook::velox::ucx_exchange
