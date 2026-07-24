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
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/buffer/Buffer.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/common/process/StackTrace.h"
#include "velox/vector/TypeAliases.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_stream.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>

#include <exception>

namespace facebook::velox::cudf_velox {
namespace {

uint64_t estimateColumnSize(cudf::column_view column) {
  uint64_t bytes = 0;
  if (cudf::is_fixed_width(column.type())) {
    bytes +=
        static_cast<uint64_t>(column.size()) * cudf::size_of(column.type());
  }
  if (column.nullable()) {
    bytes += cudf::bitmask_allocation_size_bytes(column.size());
  }
  for (auto child = column.child_begin(); child != column.child_end();
       ++child) {
    bytes += estimateColumnSize(*child);
  }
  return bytes;
}

uint64_t estimateTableSize(cudf::table_view table) {
  uint64_t totalBytes = 0;
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    totalBytes += estimateColumnSize(table.column(i));
  }
  return totalBytes;
}

void logDefaultStreamIfNeeded(
    cuda::stream_ref stream,
    const char* constructorName) {
  if (stream != cuda::stream_ref{cudaStream_t{cudaStreamDefault}}) {
    return;
  }
  LOG(WARNING) << constructorName
               << " constructed with default CUDA stream. Backtrace:\n"
               << process::StackTrace().toString();
}

} // namespace

CudfVector::CudfVector(
    velox::memory::MemoryPool* pool,
    TypePtr type,
    vector_size_t size,
    std::unique_ptr<cudf::table>&& table,
    cuda::stream_ref stream)
    : RowVector(
          pool,
          std::move(type),
          BufferPtr(nullptr),
          size,
          std::vector<VectorPtr>(),
          std::nullopt),
      tableStorage_{std::move(table)},
      stream_{stream} {
  logDefaultStreamIfNeeded(stream_, "CudfVector(table)");
  auto& tablePtr = std::get<std::unique_ptr<cudf::table>>(tableStorage_);
  tabView_ = tablePtr->view();
  flatSize_ = estimateTableSize(tabView_);
}

CudfVector::CudfVector(
    velox::memory::MemoryPool* pool,
    TypePtr type,
    vector_size_t size,
    std::unique_ptr<cudf::packed_table>&& packedTable,
    cuda::stream_ref stream,
    ReleaseCallback releaseCallback)
    : RowVector(
          pool,
          std::move(type),
          BufferPtr(nullptr),
          size,
          std::vector<VectorPtr>(),
          std::nullopt),
      tableStorage_{std::move(packedTable)},
      stream_{stream},
      releaseCallback_{std::move(releaseCallback)} {
  logDefaultStreamIfNeeded(stream_, "CudfVector(packed_table)");
  auto& packedPtr =
      std::get<std::unique_ptr<cudf::packed_table>>(tableStorage_);
  packedPtr->data.gpu_data->set_stream(stream_);
  tabView_ = packedPtr->table;
  // For packed table, flatSize is the size of the GPU data buffer
  flatSize_ = packedPtr->data.gpu_data->size();
}

CudfVector::CudfVector(
    velox::memory::MemoryPool* pool,
    TypePtr type,
    vector_size_t size,
    cudf::table_view tableView,
    std::shared_ptr<cudf::table> owner,
    cuda::stream_ref stream,
    uint64_t flatSize)
    : RowVector(
          pool,
          std::move(type),
          BufferPtr(nullptr),
          size,
          std::vector<VectorPtr>(),
          std::nullopt),
      tableStorage_{std::move(owner)},
      tabView_{tableView},
      stream_{stream},
      flatSize_{flatSize} {
  VELOX_CHECK_NOT_NULL(std::get<std::shared_ptr<cudf::table>>(tableStorage_));
  VELOX_CHECK_EQ(tabView_.num_rows(), size);
}

CudfVector::~CudfVector() {
  // Release the GPU allocation first. device_buffer deallocation is ordered on
  // its associated stream; runReleaseCallback() synchronizes that stream before
  // reporting the bytes as available to exchange flow control.
  if (auto* packedPtr =
          std::get_if<std::unique_ptr<cudf::packed_table>>(&tableStorage_);
      packedPtr && *packedPtr) {
    // The logical stream can differ from the original packing stream even if
    // the vector is dropped before its first rebindStream().
    (*packedPtr)->data.gpu_data->set_stream(stream_);
    packedPtr->reset();
  }
  runReleaseCallback();
}

void CudfVector::runReleaseCallback(
    bool synchronizeStream,
    std::exception_ptr synchronizeError) noexcept {
  ReleaseCallback callback;
  callback.swap(releaseCallback_);
  if (!callback) {
    return;
  }

  // rebindStream() may move ownership to a downstream operator's stream. Use
  // stream_ at release time rather than capturing the receive stream when the
  // callback is installed.
  if (synchronizeStream) {
    try {
      stream_.sync();
    } catch (const std::exception& e) {
      synchronizeError = std::current_exception();
      LOG(ERROR) << "Failed to synchronize CudfVector release stream: "
                 << e.what();
    } catch (...) {
      synchronizeError = std::current_exception();
      LOG(ERROR) << "Failed to synchronize CudfVector release stream";
    }
  }

  try {
    callback(std::move(synchronizeError));
  } catch (const std::exception& e) {
    LOG(ERROR) << "CudfVector release callback failed: " << e.what();
  } catch (...) {
    LOG(ERROR) << "CudfVector release callback failed";
  }
}

std::unique_ptr<cudf::table> CudfVector::release() {
  if (auto* tablePtr =
          std::get_if<std::unique_ptr<cudf::table>>(&tableStorage_)) {
    return std::move(*tablePtr);
  }
  if (auto* packedPtr =
          std::get_if<std::unique_ptr<cudf::packed_table>>(&tableStorage_)) {
    // Materialize on the same stream that releases the packed allocation.
    auto mr = (*packedPtr)->data.gpu_data->memory_resource();
    (*packedPtr)->data.gpu_data->set_stream(stream_);
    auto materializedTable =
        std::make_unique<cudf::table>(tabView_, stream_, mr);
    // Wait for both the copy and stream-ordered free before returning credit.
    packedPtr->reset();
    try {
      stream_.sync();
    } catch (...) {
      auto error = std::current_exception();
      runReleaseCallback(/*synchronizeStream=*/false, error);
      std::rethrow_exception(error);
    }
    runReleaseCallback(/*synchronizeStream=*/false);
    return materializedTable;
  }
  if (auto* ownerPtr =
          std::get_if<std::shared_ptr<cudf::table>>(&tableStorage_)) {
    auto materializedTable =
        std::make_unique<cudf::table>(tabView_, stream_, get_output_mr());
    stream_.sync();
    ownerPtr->reset();
    runReleaseCallback();
    return materializedTable;
  }
  VELOX_UNREACHABLE();
}

bool CudfVector::rebindStream(cuda::stream_ref stream) {
  if (auto* tablePtr =
          std::get_if<std::unique_ptr<cudf::table>>(&tableStorage_)) {
    if (!*tablePtr) {
      return false;
    }

    if (stream_.get() == stream.get()) {
      return true;
    }

    auto columns = (*tablePtr)->release();
    for (auto& column : columns) {
      column = cudf::rebind_stream(std::move(*column), stream);
    }

    *tablePtr = std::make_unique<cudf::table>(std::move(columns));
    tabView_ = (*tablePtr)->view();
    stream_ = stream;
    return true;
  }

  if (auto* packedPtr =
          std::get_if<std::unique_ptr<cudf::packed_table>>(&tableStorage_)) {
    if (!*packedPtr) {
      return false;
    }

    (*packedPtr)->data.gpu_data->set_stream(stream);
    stream_ = stream;
    return true;
  }

  if (auto* ownerPtr =
          std::get_if<std::shared_ptr<cudf::table>>(&tableStorage_)) {
    return *ownerPtr != nullptr && stream_.get() == stream.get();
  }

  return false;
}

uint64_t CudfVector::estimateFlatSize() const {
  return flatSize_;
}

uint64_t CudfVector::retainedSizeImpl(
    uint64_t& /*totalStringBufferSize*/) const {
  return flatSize_;
}

} // namespace facebook::velox::cudf_velox
