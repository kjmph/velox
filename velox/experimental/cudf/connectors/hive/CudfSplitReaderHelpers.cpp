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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"

#include "velox/dwio/common/BufferedInput.h"

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/types.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>

#include <exception>
#include <future>
#include <mutex>
#include <utility>
#include <vector>

namespace {

/**
 * @brief Static mutex to serialize batches of IO operations across drivers
 *
 * Mutex to ensure no interleaving of IO operations across drivers to ensure
 * drivers can move ahead without waiting for other drivers to finish their IO.
 */
std::mutex& ioBatchMutex() {
  static std::mutex mutex;
  return mutex;
}

int getStreamDevice(rmm::cuda_stream_view stream) {
  int device{};
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
  CUDF_CUDA_TRY(cudaStreamGetDevice(stream.value(), &device));
#else
  // CUDA versions before 12.8 cannot query a stream's device. Retain the
  // historical requirement that the stream belongs to the current device.
  CUDF_CUDA_TRY(cudaGetDevice(&device));
#endif
  return device;
}

void synchronizeStream(rmm::cuda_stream_view stream, int device) {
  auto const deviceScope =
      rmm::cuda_set_device_raii{rmm::cuda_device_id{device}};
  try {
    stream.synchronize();
  } catch (...) {
    const auto primaryError = std::current_exception();
    // Returning while a host buffer may still be in use would be unsafe. A
    // device-wide fence is the last recoverable fallback.
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::terminate();
    }
    std::rethrow_exception(primaryError);
  }
}

// Keeps the datasource alive while cuDF drains its completion future. The
// explicit destructor is important when the outer deferred future is discarded:
// lambda-capture destruction order alone cannot provide this lifetime rule.
class RetainedReadCompletion {
 public:
  RetainedReadCompletion(
      std::shared_ptr<cudf::io::datasource> dataSource,
      std::future<void> completion)
      : dataSource_(std::move(dataSource)),
        completion_(std::move(completion)) {}

  RetainedReadCompletion(RetainedReadCompletion&& other) noexcept
      : dataSource_(std::move(other.dataSource_)),
        completion_(std::move(other.completion_)) {}

  RetainedReadCompletion(const RetainedReadCompletion&) = delete;
  RetainedReadCompletion& operator=(const RetainedReadCompletion&) = delete;
  RetainedReadCompletion& operator=(RetainedReadCompletion&&) = delete;

  ~RetainedReadCompletion() noexcept {
    if (completion_.valid()) {
      try {
        completion_.get();
      } catch (...) {
      }
    }
  }

  void get() {
    completion_.get();
  }

 private:
  std::shared_ptr<cudf::io::datasource> dataSource_;
  std::future<void> completion_;
};

class BufferedReadCompletion {
 public:
  BufferedReadCompletion(
      std::shared_ptr<cudf::io::datasource> dataSource,
      size_t pendingCount,
      rmm::cuda_stream_view stream)
      : dataSource_(std::move(dataSource)),
        pendingCount_(pendingCount),
        stream_(stream) {}

  BufferedReadCompletion(BufferedReadCompletion&& other) noexcept
      : dataSource_(std::move(other.dataSource_)),
        pendingCount_(other.pendingCount_),
        stream_(other.stream_),
        consumed_(other.consumed_) {
    other.consumed_ = true;
  }

  BufferedReadCompletion(const BufferedReadCompletion&) = delete;
  BufferedReadCompletion& operator=(const BufferedReadCompletion&) = delete;
  BufferedReadCompletion& operator=(BufferedReadCompletion&&) = delete;

  ~BufferedReadCompletion() noexcept {
    if (!consumed_ && dataSource_ != nullptr) {
      static_cast<facebook::velox::cudf_velox::connector::hive::
                      BufferedInputDataSource*>(dataSource_.get())
          ->rollbackPendingDeviceLoads(pendingCount_);
    }
  }

  void get() {
    consumed_ = true;
    static_cast<
        facebook::velox::cudf_velox::connector::hive::BufferedInputDataSource*>(
        dataSource_.get())
        ->load(stream_);
  }

 private:
  std::shared_ptr<cudf::io::datasource> dataSource_;
  size_t pendingCount_;
  rmm::cuda_stream_view stream_;
  bool consumed_{false};
};
} // namespace

namespace facebook::velox::cudf_velox::connector::hive {

std::string normalizeKvikioUri(std::string_view path) {
  constexpr std::string_view kS3aPrefix = "s3a://";
  constexpr std::string_view kS3nPrefix = "s3n://";
  if (path.starts_with(kS3aPrefix) || path.starts_with(kS3nPrefix)) {
    return "s3://" + std::string(path.substr(kS3aPrefix.size()));
  }
  return std::string(path);
}

BufferedInputDataSource::BufferedInputDataSource(
    std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input)
    : input_(std::move(input)), fileSize_(input_->getReadFile()->size()) {}

size_t BufferedInputDataSource::size() const {
  return fileSize_;
}

void BufferedInputDataSource::enqueueForDevice(
    uint64_t offset,
    uint64_t size,
    uint8_t* dst) {
  auto inputStream = input_->enqueue({offset, size});
  std::shared_ptr sharedStream(std::move(inputStream));
  pendingDeviceLoads_.push_back([dst, size, sharedStream]() {
    std::vector<uint8_t> buffer(size);
    sharedStream->readFully(reinterpret_cast<char*>(buffer.data()), size);
    return std::pair{dst, std::move(buffer)};
  });
}

size_t BufferedInputDataSource::pendingDeviceLoadCount() const noexcept {
  return pendingDeviceLoads_.size();
}

void BufferedInputDataSource::rollbackPendingDeviceLoads(
    size_t count) noexcept {
  if (count < pendingDeviceLoads_.size()) {
    pendingDeviceLoads_.resize(count);
  }
}

void BufferedInputDataSource::load(rmm::cuda_stream_view stream) {
  auto deviceLoads = std::exchange(pendingDeviceLoads_, {});
  if (deviceLoads.empty()) {
    return;
  }
  // Clear queued destination pointers before loading. If loading fails, the
  // discarded callbacks cannot be retried after their device buffers die.
  input_->load(velox::dwio::common::LogType::FILE);

  const auto device = getStreamDevice(stream);
  auto const deviceScope =
      rmm::cuda_set_device_raii{rmm::cuda_device_id{device}};

  std::vector<std::vector<uint8_t>> hostBuffers;
  hostBuffers.reserve(deviceLoads.size());
  std::exception_ptr scheduleError;
  {
    std::lock_guard<std::mutex> lock(ioBatchMutex());
    try {
      for (auto& deviceLoad : deviceLoads) {
        auto [dst, buffer] = deviceLoad();
        hostBuffers.push_back(std::move(buffer));
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            dst,
            hostBuffers.back().data(),
            hostBuffers.back().size(),
            cudaMemcpyHostToDevice,
            stream.value()));
      }
    } catch (...) {
      scheduleError = std::current_exception();
    }
  }

  // The pageable host buffers must remain alive until every copy is complete.
  // synchronizeStream establishes completion even when stream synchronization
  // itself fails, or terminates if a safe completion fence cannot be obtained.
  try {
    synchronizeStream(stream, device);
  } catch (...) {
    if (scheduleError == nullptr) {
      throw;
    }
  }
  if (scheduleError != nullptr) {
    std::rethrow_exception(scheduleError);
  }
}

std::unique_ptr<cudf::io::datasource::buffer>
BufferedInputDataSource::host_read(size_t offset, size_t size) {
  if (offset >= fileSize_) {
    return cudf::io::datasource::buffer::create(std::vector<uint8_t>{});
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  std::vector<uint8_t> data(readSize);
  readContiguous(offset, readSize, data.data());
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t
BufferedInputDataSource::host_read(size_t offset, size_t size, uint8_t* dst) {
  if (offset >= fileSize_) {
    return 0;
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  readContiguous(offset, readSize, dst);
  return readSize;
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
BufferedInputDataSource::host_read_async(size_t offset, size_t size) {
  return std::async(std::launch::deferred, [this, offset, size]() {
    return this->host_read(offset, size);
  });
}

std::future<size_t> BufferedInputDataSource::host_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  return std::async(std::launch::deferred, [this, offset, size, dst]() {
    return this->host_read(offset, size, dst);
  });
}

std::future<size_t> BufferedInputDataSource::device_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(input_->executor() != nullptr, "IO executor is not initialized");
  const auto device = getStreamDevice(stream);
  auto promise = std::make_shared<std::promise<size_t>>();
  auto future = promise->get_future();
  try {
    input_->executor()->add(
        [this, offset, size, dst, stream, device, promise]() {
          try {
            auto const deviceScope =
                rmm::cuda_set_device_raii{rmm::cuda_device_id{device}};
            auto hostBuffer = this->host_read(offset, size);
            if (hostBuffer->size() != 0) {
              try {
                CUDF_CUDA_TRY(cudaMemcpyAsync(
                    dst,
                    hostBuffer->data(),
                    hostBuffer->size(),
                    cudaMemcpyHostToDevice,
                    stream.value()));
              } catch (...) {
                auto const copyError = std::current_exception();
                try {
                  synchronizeStream(stream, device);
                } catch (...) {
                  // synchronizeStream either establishes completion before it
                  // throws or terminates when completion cannot be proven.
                }
                std::rethrow_exception(copyError);
              }
              synchronizeStream(stream, device);
            }
            promise->set_value(hostBuffer->size());
          } catch (...) {
            promise->set_exception(std::current_exception());
          }
        });
  } catch (...) {
    // Executor rejection is synchronous and no task owns the destination.
    promise->set_exception(std::current_exception());
  }
  return future;
}

bool BufferedInputDataSource::supports_device_read() const {
  return true;
}

void BufferedInputDataSource::readContiguous(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  using namespace facebook::velox::dwio::common;
  // BufferedInput::read gives us a stream over the exact region.
  auto stream = input_->read(offset, size, LogType::FILE);
  VELOX_CHECK(stream != nullptr, "read() returned null stream");
  stream->readFully(reinterpret_cast<char*>(dst), size);
}

std::tuple<
    std::vector<rmm::device_buffer>,
    std::vector<cudf::device_span<const uint8_t>>,
    std::future<void>>
fetchByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto bufferedInput = dynamic_cast<BufferedInputDataSource*>(dataSource.get());
  if (bufferedInput == nullptr) {
    // cuDF owns the generic KvikIO/host implementation. Delegating avoids a
    // second copy of its coalescing, short-read validation, failure draining,
    // and host-buffer lifetime rules at the Velox boundary.
    auto [buffers, spans, completion] =
        cudf::io::parquet::fetch_byte_ranges_to_device_async(
            *dataSource, byteRanges, stream, mr);
    auto completionState =
        RetainedReadCompletion{std::move(dataSource), std::move(completion)};
    auto retainedCompletion = std::async(
        std::launch::deferred,
        [completionState = std::move(completionState)]() mutable {
          completionState.get();
        });
    return {
        std::move(buffers), std::move(spans), std::move(retainedCompletion)};
  }

  // Pad buffer sizes to be a multiple of 8 bytes. Required by
  // `decode_page_data_kernel` in cuDF Parquet reader.
  constexpr auto kBufferPaddingMultiple = 8;

  // Allocate device spans for each column chunk
  std::vector<cudf::device_span<const uint8_t>> columnChunkData{};
  columnChunkData.reserve(byteRanges.size());

  // Total IO size across all byte ranges
  auto totalSize = std::accumulate(
      byteRanges.begin(),
      byteRanges.end(),
      std::size_t{0},
      [&](auto acc, const auto& byteRange) { return acc + byteRange.size(); });

  // Allocate single device buffer for all column chunks
  std::vector<rmm::device_buffer> columnChunkBuffers{};
  columnChunkBuffers.emplace_back(
      cudf::util::round_up_safe<size_t>(totalSize, kBufferPaddingMultiple),
      stream,
      mr);

  // Compute device spans for each column chunk
  auto bufferData = static_cast<uint8_t*>(columnChunkBuffers.back().data());
  std::ignore = std::accumulate(
      byteRanges.begin(),
      byteRanges.end(),
      std::size_t{0},
      [&](auto acc, const auto& byteRange) {
        columnChunkData.emplace_back(
            bufferData + acc, static_cast<size_t>(byteRange.size()));
        return acc + byteRange.size();
      });

  // For BufferedInputDataSource, enqueue reads into the buffer and launch the
  // actual load asynchronously.
  auto const pendingCount = bufferedInput->pendingDeviceLoadCount();
  try {
    auto iter =
        cuda::make_zip_iterator(byteRanges.begin(), columnChunkData.begin());
    std::for_each(
        iter, iter + byteRanges.size(), [bufferedInput](const auto& tuple) {
          const auto& byteRange = cuda::std::get<0>(tuple);
          const auto& destination = cuda::std::get<1>(tuple);
          bufferedInput->enqueueForDevice(
              static_cast<uint64_t>(byteRange.offset()),
              static_cast<uint64_t>(byteRange.size()),
              const_cast<uint8_t*>(destination.data()));
        });

    auto completionState =
        BufferedReadCompletion{dataSource, pendingCount, stream};
    auto completion = std::async(
        std::launch::deferred,
        [completionState = std::move(completionState)]() mutable {
          completionState.get();
        });
    return {
        std::move(columnChunkBuffers),
        std::move(columnChunkData),
        std::move(completion)};
  } catch (...) {
    bufferedInput->rollbackPendingDeviceLoads(pendingCount);
    throw;
  }
}

} // namespace facebook::velox::cudf_velox::connector::hive
