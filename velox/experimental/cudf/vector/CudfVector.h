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

#include "velox/common/memory/MemoryPool.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/TypeAliases.h"

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>

#include <cuda/stream>

#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

namespace facebook::velox::cudf_velox {

// Vector class which holds GPU data from cuDF.
// Can be constructed from an owned cudf::table, a packed_table, or a
// table_view with opaque shared ownership. Packed and shared-view
// constructions avoid copying the underlying GPU data.
//
// CudfVector assumes that the input table data is already ready to consume.
// Constructors and rebindStream() do not insert CUDA stream waits for work that
// populated the input buffers; callers must establish that ordering before
// constructing or rebinding a CudfVector. All buffers retained by a ViewOwner
// must also be safe to deallocate on the supplied stream; opaque view ownership
// cannot be rebound to a different stream.
class CudfVector : public RowVector {
 public:
  /// Invoked after the vector releases its packed GPU storage. A non-null
  /// exception indicates that synchronizing the storage's final stream failed,
  /// so callers must not treat the allocation as safely reusable.
  using ReleaseCallback = std::function<void(std::exception_ptr)>;
  using ViewOwner = std::shared_ptr<void>;

  /// Constructs a CudfVector from an owned cudf::table.
  CudfVector(
      velox::memory::MemoryPool* pool,
      TypePtr type,
      vector_size_t size,
      std::unique_ptr<cudf::table>&& table,
      cuda::stream_ref stream);

  /// Constructs a CudfVector from packed_table.
  /// The packed data is retained and tabView_ references the table view inside
  /// packed_table. This avoids copying the underlying GPU data.
  CudfVector(
      velox::memory::MemoryPool* pool,
      TypePtr type,
      vector_size_t size,
      std::unique_ptr<cudf::packed_table>&& packedTable,
      cuda::stream_ref stream,
      ReleaseCallback releaseCallback = nullptr);

  /// Constructs a CudfVector from a table view backed by shared storage. The
  /// owner is retained until this vector is destroyed or materialized via
  /// release(). The view may reference one or more objects retained by owner.
  /// The logical flat size is inferred from the view unless supplied, e.g. for
  /// a slice whose size should be apportioned from its owner.
  CudfVector(
      velox::memory::MemoryPool* pool,
      TypePtr type,
      vector_size_t size,
      cudf::table_view tableView,
      ViewOwner owner,
      cuda::stream_ref stream,
      std::optional<uint64_t> flatSize = std::nullopt);

  /// Compatibility overload for a view backed by a single shared table with
  /// an explicitly supplied logical flat size.
  CudfVector(
      velox::memory::MemoryPool* pool,
      TypePtr type,
      vector_size_t size,
      cudf::table_view tableView,
      std::shared_ptr<cudf::table> owner,
      cuda::stream_ref stream,
      uint64_t flatSize);

  ~CudfVector() override;

  cuda::stream_ref stream() const {
    return stream_;
  }

  cudf::table_view getTableView() const {
    return tabView_;
  }

  /// Releases ownership of the underlying table. Packed and shared-view
  /// storage is materialized from the view first, which copies the data.
  std::unique_ptr<cudf::table> release();

  /// Rebinds owned table buffers to use 'stream' for future deallocation.
  /// This only changes future stream/deallocation association. It does not make
  /// 'stream' wait on the previous stream or any producer stream.
  /// Returns false when the storage cannot be rebound without materializing.
  bool rebindStream(cuda::stream_ref stream);

  uint64_t estimateFlatSize() const override;

 private:
  /// Synchronizes the vector's current stream, then invokes and clears the
  /// callback. noexcept because this is also called from the destructor.
  void runReleaseCallback(
      bool synchronizeStream = true,
      std::exception_ptr synchronizationError = nullptr) noexcept;

  uint64_t retainedSizeImpl(uint64_t& totalStringBufferSize) const override;

  // Storage for an owned table, packed table, or opaque shared view owner.
  // Only one is active at a time - using variant enforces this at compile time.
  using TableStorage = std::variant<
      std::unique_ptr<cudf::table>,
      std::unique_ptr<cudf::packed_table>,
      ViewOwner>;
  TableStorage tableStorage_;

  // Table view - always valid, backed by the active TableStorage alternative.
  cudf::table_view tabView_;

  cuda::stream_ref stream_;
  uint64_t flatSize_;
  ReleaseCallback releaseCallback_;
};

using CudfVectorPtr = std::shared_ptr<CudfVector>;

} // namespace facebook::velox::cudf_velox
