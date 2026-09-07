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

#include <memory>
#include <mutex>
#include <string>

#include <cudf/contiguous_split.hpp>

#include "velox/exec/SerializedPage.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::ucx_exchange {

/// Marks a packed cuDF table. A zero-column table has no device segment, so
/// shape alone cannot tell one from a serialized page.
constexpr uint32_t kCudfPackedPageMagic{0x43'55'44'46}; // "CUDF"

/// Magic plus logical row count, ahead of the cuDF metadata.
constexpr size_t kCudfPackedPageHeaderBytes{
    sizeof(kCudfPackedPageMagic) + sizeof(int32_t)};

/// An output page holding a packed cuDF table. getIOBuf() picks the form from
/// ExchangeFormatRegistry: the packed chain, or Presto host bytes.
class CudfPackedPage : public exec::SerializedPageBase {
 public:
  CudfPackedPage(
      std::unique_ptr<cudf::packed_columns> packed,
      std::string taskId,
      int destination,
      int64_t numRows,
      RowTypePtr rowType,
      std::string serdeKind,
      bool sharedAcrossDestinations,
      memory::MemoryPool* pool);

  /// Fixed on the first call: the output buffer subtracts the same value when
  /// the page is freed.
  uint64_t size() const override;

  std::optional<int64_t> numRows() const override;

  std::unique_ptr<folly::IOBuf> getIOBuf() const override;

  std::unique_ptr<ByteInputStream> prepareStreamForDeserialize() override;

  /// Renders the host form now, so a later getIOBuf() only clones it.
  /// getIOBuf() runs under the mutex every producer driver takes to enqueue.
  void renderHostBytes() const;

 private:
  // Cached: the buffer may hand the same page out more than once.
  const std::unique_ptr<folly::IOBuf>& hostBytes() const;

  // Released once a shared page has rendered; kept for a page with a
  // destination of its own, which may still be asked for the device form.
  mutable std::shared_ptr<cudf::packed_columns> packed_;
  const std::string taskId_;
  const int destination_;
  const int64_t numRows_;
  const RowTypePtr rowType_;
  // The serde the plan asked for. Rendering with a different one produces a
  // page the reader cannot parse.
  const std::string serdeKind_;
  // True for broadcast and arbitrary output, where the buffer hands this page
  // to more than one destination and there is no single reader to render for.
  const bool sharedAcrossDestinations_;
  memory::MemoryPool* const pool_;
  const uint64_t packedSize_;

  mutable std::mutex mutex_;
  mutable std::unique_ptr<folly::IOBuf> hostBytes_;
  // What size() answered the first time, and whether it has answered yet.
  mutable uint64_t reportedSize_{0};
  mutable bool sizeFixed_{false};
};

} // namespace facebook::velox::ucx_exchange
