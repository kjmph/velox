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

#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"

#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

using namespace std::chrono_literals;
using facebook::velox::dwio::common::BufferedInput;

class TestCudaStream {
 public:
  TestCudaStream() {
    CUDF_CUDA_TRY(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }

  ~TestCudaStream() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  rmm::cuda_stream_view view() const {
    return rmm::cuda_stream_view{stream_};
  }

 private:
  cudaStream_t stream_{nullptr};
};

struct StreamGate {
  std::mutex mutex;
  std::condition_variable cv;
  bool open{false};
  std::promise<void> entered;
};

void CUDART_CB waitForGate(void* opaque) {
  auto* gate = static_cast<StreamGate*>(opaque);
  gate->entered.set_value();
  std::unique_lock<std::mutex> lock(gate->mutex);
  gate->cv.wait(lock, [gate] { return gate->open; });
}

void releaseGate(StreamGate& gate) {
  {
    std::lock_guard<std::mutex> lock(gate.mutex);
    gate.open = true;
  }
  gate.cv.notify_all();
}

class ExecutorBufferedInput final : public BufferedInput {
 public:
  ExecutorBufferedInput(
      std::shared_ptr<ReadFile> readFile,
      memory::MemoryPool& pool,
      folly::Executor* executor)
      : BufferedInput(std::move(readFile), pool), executor_(executor) {}

  folly::Executor* executor() const override {
    return executor_;
  }

 private:
  folly::Executor* const executor_;
};

struct PendingReadState {
  PendingReadState() : releaseFuture(release.get_future().share()) {}

  std::atomic<bool> destroyed{false};
  std::promise<void> entered;
  std::promise<void> release;
  std::shared_future<void> releaseFuture;
};

class PendingDeviceDataSource final : public cudf::io::datasource {
 public:
  explicit PendingDeviceDataSource(std::shared_ptr<PendingReadState> state)
      : state_(std::move(state)) {}

  ~PendingDeviceDataSource() override {
    state_->destroyed = true;
  }

  std::unique_ptr<datasource::buffer> host_read(
      size_t /*offset*/,
      size_t /*size*/) override {
    return datasource::buffer::create(std::vector<uint8_t>{});
  }

  size_t host_read(size_t /*offset*/, size_t /*size*/, uint8_t* /*dst*/)
      override {
    return 0;
  }

  bool supports_device_read() const override {
    return true;
  }

  std::future<size_t> device_read_async(
      size_t /*offset*/,
      size_t size,
      uint8_t* /*dst*/,
      rmm::cuda_stream_view /*stream*/) override {
    return std::async(std::launch::async, [state = state_, size] {
      state->entered.set_value();
      state->releaseFuture.wait();
      return size;
    });
  }

  size_t size() const override {
    return 16;
  }

 private:
  std::shared_ptr<PendingReadState> state_;
};

class CudfSplitReaderHelpersTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  std::shared_ptr<ExecutorBufferedInput> makeInput(
      const std::string& data,
      folly::Executor* executor) {
    return std::make_shared<ExecutorBufferedInput>(
        std::make_shared<InMemoryReadFile>(data), *pool_, executor);
  }

  const std::shared_ptr<memory::MemoryPool> pool_ =
      memory::memoryManager()->addLeafPool();
};

TEST_F(CudfSplitReaderHelpersTest, normalizeKvikioS3Uri) {
  EXPECT_EQ(normalizeKvikioUri("s3://bucket/key"), "s3://bucket/key");
  EXPECT_EQ(normalizeKvikioUri("s3a://bucket/key"), "s3://bucket/key");
  EXPECT_EQ(normalizeKvikioUri("s3n://bucket/key"), "s3://bucket/key");
  EXPECT_EQ(normalizeKvikioUri("file:/tmp/input"), "file:/tmp/input");
  EXPECT_EQ(
      normalizeKvikioUri("https://example.test/a"), "https://example.test/a");
}

TEST_F(CudfSplitReaderHelpersTest, deviceReadFutureWaitsForDeviceCopy) {
  const std::string inputData = "buffered-input-device-read";
  folly::CPUThreadPoolExecutor executor(1);
  auto input = makeInput(inputData, &executor);
  BufferedInputDataSource dataSource(input);

  TestCudaStream stream;
  rmm::device_buffer destination(
      inputData.size(), stream.view(), cudf::get_current_device_resource_ref());

  StreamGate gate;
  auto entered = gate.entered.get_future();
  CUDF_CUDA_TRY(cudaLaunchHostFunc(stream.view().value(), waitForGate, &gate));

  auto completion = dataSource.device_read_async(
      0,
      inputData.size(),
      static_cast<uint8_t*>(destination.data()),
      stream.view());

  if (entered.wait_for(10s) != std::future_status::ready) {
    releaseGate(gate);
    std::ignore = completion.get();
    FAIL() << "CUDA stream callback did not start";
  }
  EXPECT_EQ(completion.wait_for(0s), std::future_status::timeout);
  releaseGate(gate);

  EXPECT_EQ(completion.get(), inputData.size());
  std::string actual(inputData.size(), '\0');
  CUDF_CUDA_TRY(cudaMemcpy(
      actual.data(),
      destination.data(),
      actual.size(),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, inputData);
}

TEST_F(CudfSplitReaderHelpersTest, bufferedLoadWaitsForDeviceCopies) {
  const std::string inputData = "first-range-second-range";
  folly::CPUThreadPoolExecutor executor(1);
  auto input = makeInput(inputData, &executor);
  BufferedInputDataSource dataSource(input);

  TestCudaStream stream;
  rmm::device_buffer destination(
      inputData.size(), stream.view(), cudf::get_current_device_resource_ref());
  dataSource.enqueueForDevice(
      0, inputData.size(), static_cast<uint8_t*>(destination.data()));

  StreamGate gate;
  auto entered = gate.entered.get_future();
  CUDF_CUDA_TRY(cudaLaunchHostFunc(stream.view().value(), waitForGate, &gate));

  auto completion =
      std::async(std::launch::async, [&] { dataSource.load(stream.view()); });

  if (entered.wait_for(10s) != std::future_status::ready) {
    releaseGate(gate);
    completion.get();
    FAIL() << "CUDA stream callback did not start";
  }
  EXPECT_EQ(completion.wait_for(0s), std::future_status::timeout);
  releaseGate(gate);
  EXPECT_NO_THROW(completion.get());

  std::string actual(inputData.size(), '\0');
  CUDF_CUDA_TRY(cudaMemcpy(
      actual.data(),
      destination.data(),
      actual.size(),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, inputData);
}

TEST_F(CudfSplitReaderHelpersTest, discardedBufferedFutureRollsBackBatch) {
  const std::string inputData = "discarded-buffered-read";
  folly::CPUThreadPoolExecutor executor(1);
  auto input = makeInput(inputData, &executor);
  auto dataSource = std::make_shared<BufferedInputDataSource>(input);
  const std::vector<cudf::io::text::byte_range_info> ranges{
      {0, static_cast<int64_t>(inputData.size())}};
  TestCudaStream stream;

  auto [buffers, spans, completion] = fetchByteRangesAsync(
      dataSource,
      cudf::host_span<const cudf::io::text::byte_range_info>{
          ranges.data(), ranges.size()},
      stream.view(),
      cudf::get_current_device_resource_ref());
  EXPECT_EQ(dataSource->pendingDeviceLoadCount(), 1);

  completion = std::future<void>{};
  EXPECT_EQ(dataSource->pendingDeviceLoadCount(), 0);
}

TEST_F(CudfSplitReaderHelpersTest, discardedFutureDrainsBeforeDatasourceDies) {
  auto state = std::make_shared<PendingReadState>();
  auto entered = state->entered.get_future();
  auto dataSource = std::make_shared<PendingDeviceDataSource>(state);
  const std::vector<cudf::io::text::byte_range_info> ranges{{0, 16}};
  TestCudaStream stream;

  auto [buffers, spans, completion] = fetchByteRangesAsync(
      dataSource,
      cudf::host_span<const cudf::io::text::byte_range_info>{
          ranges.data(), ranges.size()},
      stream.view(),
      cudf::get_current_device_resource_ref());
  dataSource.reset();
  EXPECT_FALSE(state->destroyed);

  auto discard = std::async(
      std::launch::async, [completion = std::move(completion)]() mutable {
        completion = std::future<void>{};
      });

  if (entered.wait_for(10s) != std::future_status::ready) {
    state->release.set_value();
    discard.get();
    FAIL() << "device read did not start";
  }
  EXPECT_EQ(discard.wait_for(0s), std::future_status::timeout);
  EXPECT_FALSE(state->destroyed);

  state->release.set_value();
  EXPECT_NO_THROW(discard.get());
  EXPECT_TRUE(state->destroyed);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
