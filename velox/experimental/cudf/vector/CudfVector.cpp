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

} // namespace

CudfVector::CudfVector(
    velox::memory::MemoryPool* pool,
    TypePtr type,
    vector_size_t size,
    std::unique_ptr<cudf::table>&& table,
    rmm::cuda_stream_view stream)
    : RowVector(
          pool,
          std::move(type),
          BufferPtr(nullptr),
          size,
          std::vector<VectorPtr>(),
          std::nullopt),
      tableStorage_{std::move(table)},
      stream_{stream} {
  auto& tablePtr = std::get<std::unique_ptr<cudf::table>>(tableStorage_);
  tabView_ = tablePtr->view();
  flatSize_ = estimateTableSize(tabView_);
}

CudfVector::CudfVector(
    velox::memory::MemoryPool* pool,
    TypePtr type,
    vector_size_t size,
    std::unique_ptr<cudf::packed_table>&& packedTable,
    rmm::cuda_stream_view stream,
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
  auto& packedPtr =
      std::get<std::unique_ptr<cudf::packed_table>>(tableStorage_);
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
    rmm::cuda_stream_view stream,
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
  if (auto* tablePtr =
          std::get_if<std::unique_ptr<cudf::table>>(&tableStorage_)) {
    tablePtr->reset();
  } else if (
      auto* packedPtr =
          std::get_if<std::unique_ptr<cudf::packed_table>>(&tableStorage_)) {
    packedPtr->reset();
  } else if (
      auto* ownerPtr =
          std::get_if<std::shared_ptr<cudf::table>>(&tableStorage_)) {
    ownerPtr->reset();
  }
  runReleaseCallback();
}

void CudfVector::runReleaseCallback() {
  ReleaseCallback callback;
  callback.swap(releaseCallback_);
  if (!callback) {
    return;
  }

  try {
    callback();
  } catch (const std::exception& e) {
    LOG(ERROR) << "CudfVector release callback failed: " << e.what();
  }
}

std::unique_ptr<cudf::table> CudfVector::release() {
  flatSize_ = 0;
  if (auto* tablePtr =
          std::get_if<std::unique_ptr<cudf::table>>(&tableStorage_)) {
    // Constructed from owned table - just move it out
    return std::move(*tablePtr);
  }

  if (auto* packedPtr =
          std::get_if<std::unique_ptr<cudf::packed_table>>(&tableStorage_)) {
    // Constructed from packed_table - materialize a table from the view.
    // This copies the data since the view references the packed buffer.
    auto mr = (*packedPtr)->data.gpu_data->memory_resource();
    (*packedPtr)->data.gpu_data->set_stream(stream_);
    auto materializedTable =
        std::make_unique<cudf::table>(tabView_, stream_, mr);
    packedPtr->reset();
    runReleaseCallback();
    return materializedTable;
  }

  if (auto* ownerPtr =
          std::get_if<std::shared_ptr<cudf::table>>(&tableStorage_)) {
    auto materializedTable =
        std::make_unique<cudf::table>(tabView_, stream_, get_output_mr());
    ownerPtr->reset();
    runReleaseCallback();
    return materializedTable;
  }

  VELOX_UNREACHABLE();
}

bool CudfVector::rebindStream(rmm::cuda_stream_view stream) {
  if (auto* tablePtr =
          std::get_if<std::unique_ptr<cudf::table>>(&tableStorage_)) {
    if (!*tablePtr) {
      return false;
    }

    if (stream_.value() == stream.value()) {
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
    return *ownerPtr != nullptr && stream_.value() == stream.value();
  }

  return false;
}

uint64_t CudfVector::estimateFlatSize() const {
  return flatSize_;
}

} // namespace facebook::velox::cudf_velox
