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

#include "velox/experimental/ucx-exchange/CudfPackedPage.h"

#include <cstring>

#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/memory/ByteStream.h"
#include "velox/common/memory/Memory.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/ucx-exchange/ExchangeFormatRegistry.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::ucx_exchange {

namespace {

// Host bytes outlive the task, so not an operator pool. Not charged to the
// query and not reclaimable by the arbitrator.
memory::MemoryPool* hostRenderPool() {
  static const std::shared_ptr<memory::MemoryPool> pool =
      memory::memoryManager()->addLeafPool("ucx_host_render");
  return pool.get();
}

using PackedHolder = std::shared_ptr<cudf::packed_columns>;

void releasePackedHolder(void* /*buf*/, void* userData) {
  delete static_cast<PackedHolder*>(userData);
}

// Describes the pointer without dereferencing it, so device memory can
// travel through host code.
std::unique_ptr<folly::IOBuf>
wrapRegion(void* data, size_t size, const PackedHolder& holder) {
  return folly::IOBuf::takeOwnership(
      data, size, releasePackedHolder, new PackedHolder{holder});
}

} // namespace

CudfPackedPage::CudfPackedPage(
    std::unique_ptr<cudf::packed_columns> packed,
    std::string taskId,
    int destination,
    int64_t numRows,
    RowTypePtr rowType,
    std::string serdeKind,
    bool sharedAcrossDestinations,
    memory::MemoryPool* pool)
    : packed_(std::move(packed)),
      taskId_(std::move(taskId)),
      destination_(destination),
      numRows_(numRows),
      rowType_(std::move(rowType)),
      serdeKind_(std::move(serdeKind)),
      sharedAcrossDestinations_(sharedAcrossDestinations),
      pool_(pool),
      packedSize_(
          (packed_->metadata ? packed_->metadata->size() : 0) +
          (packed_->gpu_data ? packed_->gpu_data->size() : 0)) {}

uint64_t CudfPackedPage::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!sizeFixed_) {
    // Fixed on the first call; a later lazy render is not counted.
    reportedSize_ = (packed_ != nullptr ? packedSize_ : 0) +
        (hostBytes_ != nullptr ? hostBytes_->computeChainDataLength() : 0);
    sizeFixed_ = true;
  }
  return reportedSize_;
}

std::optional<int64_t> CudfPackedPage::numRows() const {
  return numRows_;
}

std::unique_ptr<folly::IOBuf> CudfPackedPage::getIOBuf() const {
  // A shared page can't be baked as gpu memory because there's no
  // guarantee every shuffle destination can use that
  if (!sharedAcrossDestinations_ &&
      ExchangeFormatRegistry::instance().iscuDFExchange(
          taskId_, destination_)) {
    const size_t metadataBytes =
        packed_->metadata ? packed_->metadata->size() : 0;
    const size_t deviceBytes =
        packed_->gpu_data ? packed_->gpu_data->size() : 0;

    // [magic][int32 numRows][cudf metadata] then the device data.
    const auto rows = static_cast<int32_t>(numRows_);
    const uint32_t magic = kCudfPackedPageMagic;
    auto chain =
        folly::IOBuf::create(kCudfPackedPageHeaderBytes + metadataBytes);
    std::memcpy(chain->writableData(), &magic, sizeof(magic));
    std::memcpy(chain->writableData() + sizeof(magic), &rows, sizeof(rows));
    if (metadataBytes > 0) {
      std::memcpy(
          chain->writableData() + kCudfPackedPageHeaderBytes,
          packed_->metadata->data(),
          metadataBytes);
    }
    chain->append(kCudfPackedPageHeaderBytes + metadataBytes);

    if (deviceBytes > 0) {
      chain->appendToChain(
          wrapRegion(packed_->gpu_data->data(), deviceBytes, packed_));
    }
    return chain;
  }
  // No UCX handshake
  if (!sharedAcrossDestinations_) {
    ExchangeFormatRegistry::instance().setExchangeFormat(
        taskId_, destination_, ExchangeFormatRegistry::Format::kVelox);
  }
  return hostBytes()->clone();
}

void CudfPackedPage::renderHostBytes() const {
  hostBytes();
}

const std::unique_ptr<folly::IOBuf>& CudfPackedPage::hostBytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (hostBytes_ != nullptr) {
    return hostBytes_;
  }

  RowVectorPtr rowVector;
  if (rowType_->size() == 0) {
    // No columns
    rowVector = std::make_shared<RowVector>(
        pool_,
        rowType_,
        /*nulls=*/nullptr,
        static_cast<vector_size_t>(numRows_),
        std::vector<VectorPtr>{});
  } else {
    auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
    auto table = cudf::unpack(
        static_cast<uint8_t const*>(packed_->metadata->data()),
        static_cast<uint8_t const*>(packed_->gpu_data->data()));
    rowVector = cudf_velox::with_arrow::toVeloxColumn(
        table,
        pool_,
        rowType_,
        /*namePrefix=*/"",
        stream,
        cudf_velox::get_temp_mr());
    stream.synchronize();
    rowVector->setType(rowType_);
  }

  auto* serde = getNamedVectorSerde(serdeKind_);
  VectorStreamGroup group(pool_, serde);
  group.createStreamTree(rowType_, static_cast<int32_t>(numRows_));
  const IndexRange range{0, static_cast<vector_size_t>(rowVector->size())};
  group.append(rowVector, folly::Range<const IndexRange*>(&range, 1));

  // Clones share this allocation and can outlive the task. Everything above
  // stays charged to the query.
  IOBufOutputStream out(
      *hostRenderPool(), nullptr, static_cast<int32_t>(group.size()));
  group.flush(&out);
  hostBytes_ = out.getIOBuf();

  if (sharedAcrossDestinations_) {
    packed_.reset();
  }
  return hostBytes_;
}

std::unique_ptr<ByteInputStream> CudfPackedPage::prepareStreamForDeserialize() {
  VELOX_NYI(
      "CudfPackedPage is read through getIOBuf(), not deserialized in place");
}

} // namespace facebook::velox::ucx_exchange
