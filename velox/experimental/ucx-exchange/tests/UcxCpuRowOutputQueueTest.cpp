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

#include <folly/io/IOBuf.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/core/PlanFragment.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::ucx_exchange {
namespace {

class UcxCpuRowOutputQueueTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
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

} // namespace
} // namespace facebook::velox::ucx_exchange
