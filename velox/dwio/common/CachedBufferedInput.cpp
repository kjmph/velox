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

#include "velox/dwio/common/CachedBufferedInput.h"
#include "folly/io/Cursor.h"
#include "velox/common/memory/Allocation.h"
#include "velox/common/time/Timer.h"
#include "velox/dwio/common/CacheInputStream.h"

DECLARE_int32(cache_prefetch_min_pct);

using ::facebook::velox::common::Region;

namespace facebook::velox::dwio::common {

using cache::CachePin;
using cache::CoalescedLoad;
using cache::RawFileCacheKey;
using cache::ScanTracker;
using cache::SsdFile;
using cache::SsdPin;
using cache::TrackingId;
using memory::MemoryAllocator;

std::unique_ptr<SeekableInputStream> CachedBufferedInput::enqueue(
    Region region,
    const StreamIdentifier* sid = nullptr) {
  if (region.length == 0) {
    return std::make_unique<SeekableArrayInputStream>(
        static_cast<const char*>(nullptr), 0);
  }

  TrackingId id;
  if (sid != nullptr) {
    id = TrackingId(sid->getId());
  }
  VELOX_CHECK_LE(region.offset + region.length, fileSize_);
  if (tracker_ != nullptr) {
    tracker_->recordReference(id, region.length, fileNum_.id(), groupId_.id());
  }
  auto stream = std::make_unique<CacheInputStream>(
      this,
      ioStatistics_.get(),
      region,
      input_,
      fileNum_.id(),
      options_.cacheable(),
      tracker_,
      id,
      groupId_.id(),
      options_.loadQuantum());
  if (preloaded()) {
    // Data is already in cache. Give the stream its own pin copy so it can
    // outlive this CachedBufferedInput and skip all loading/prefetch logic.
    stream->setPreloadedPin(preloadPin_);
  } else {
    requests_.emplace_back(
        RawFileCacheKey{fileNum_.id(), region.offset}, region.length, id);
    requests_.back().stream = stream.get();
  }
  return stream;
}

bool CachedBufferedInput::isBuffered(uint64_t /*offset*/, uint64_t /*length*/)
    const {
  // When preloaded, the entire file content is already in cache, so any
  // region within the file is considered buffered and can be served without
  // additional I/O.
  return preloaded();
}

bool CachedBufferedInput::shouldPreload(int32_t numPages) {
  // True if after scheduling this for preload, half the capacity would be in a
  // loading but not yet accessed state.
  if (requests_.empty() && (numPages == 0)) {
    return false;
  }
  for (const auto& request : requests_) {
    numPages += memory::AllocationTraits::numPages(
        std::min<int32_t>(request.size, options_.loadQuantum()));
  }
  const auto cachePages = cache_->cachedPages();
  auto* allocator = cache_->allocator();
  const auto maxPages =
      memory::AllocationTraits::numPages(allocator->capacity());
  const auto allocatedPages = allocator->numAllocated();
  if (numPages < maxPages - allocatedPages) {
    // There is free space for the read-ahead.
    return true;
  }
  const auto prefetchPages = cache_->incrementPrefetchPages(0);
  if (numPages + prefetchPages < cachePages / 2) {
    // The planned prefetch plus other prefetches are under half the cache.
    return true;
  }
  return false;
}

namespace {

bool isPrefetchPct(int32_t pct) {
  return pct >= FLAGS_cache_prefetch_min_pct;
}

std::vector<CacheRequest*> makeRequestParts(
    CacheRequest& request,
    const cache::TrackingData& trackingData,
    int32_t loadQuantum,
    std::vector<std::unique_ptr<CacheRequest>>& extraRequests) {
  if (request.size <= loadQuantum) {
    return {&request};
  }

  // Large columns will be part of coalesced reads if the access frequency
  // qualifies for read ahead and if over 80% of the column gets accessed. Large
  // metadata columns (empty no trackingData) always coalesce.
  const bool prefetchOne =
      request.trackingId.id() == StreamIdentifier::sequentialFile().id_;
  const auto readDensity =
      trackingData.readBytes / (1 + trackingData.referencedBytes);
  const auto readPct = 100 * readDensity;
  const bool prefetch = trackingData.referencedBytes > 0 &&
      isPrefetchPct(readPct) && readDensity >= 0.8;
  std::vector<CacheRequest*> parts;
  for (uint64_t offset = 0; offset < request.size; offset += loadQuantum) {
    const int32_t size = std::min<int32_t>(loadQuantum, request.size - offset);
    extraRequests.push_back(
        std::make_unique<CacheRequest>(
            RawFileCacheKey{request.key.fileNum, request.key.offset + offset},
            size,
            request.trackingId));
    parts.push_back(extraRequests.back().get());
    parts.back()->stream = request.stream;
    parts.back()->coalesces = prefetch;
    if (prefetchOne) {
      break;
    }
  }
  return parts;
}

template <bool kSsd>
uint64_t getOffset(const CacheRequest& request) {
  if constexpr (kSsd) {
    VELOX_DCHECK(!request.ssdPin.empty());
    return request.ssdPin.run().offset();
  } else {
    return request.key.offset;
  }
}

template <bool kSsd>
bool lessThan(const CacheRequest* left, const CacheRequest* right) {
  auto leftOffset = getOffset<kSsd>(*left);
  auto rightOffset = getOffset<kSsd>(*right);
  return leftOffset < rightOffset ||
      (leftOffset == rightOffset && left->size > right->size);
}

} // namespace

void CachedBufferedInput::preload() {
  VELOX_CHECK(preloadPin_.empty(), "preload() called more than once");
  VELOX_CHECK(requests_.empty(), "preload() must be called before enqueue()");
  cache::RawFileCacheKey key{fileNum_.id(), 0};
  folly::SemiFuture<bool> waitFuture(false);
  do {
    preloadPin_ =
        cache_->findOrCreate(key, fileSize_, /*contiguous=*/false, &waitFuture);
    if (preloadPin_.empty()) {
      uint64_t waitUs{0};
      {
        MicrosecondWallTimer timer(&waitUs);
        std::move(waitFuture).wait();
      }
      ioStatistics_->queryThreadIoLatencyUs().increment(waitUs);
      ioStatistics_->cacheWaitLatencyUs().increment(waitUs);
    }
  } while (preloadPin_.empty());

  auto* entry = preloadPin_.checkedEntry();
  if (!entry->getAndClearFirstUseFlag()) {
    // Already loaded by another concurrent query.
    ioStatistics_->ramHit().increment(fileSize_);
  }
  if (!entry->isExclusive()) {
    // Cache hit — already loaded.
    return;
  }

  entry->setGroupId(groupId_.id());
  entry->setTrackingId(
      cache::TrackingId(StreamIdentifier::sequentialFile().id_));
  auto ranges = entry->dataRanges(fileSize_);
  uint64_t storageReadUs{0};
  {
    MicrosecondWallTimer timer(&storageReadUs);
    input_->read(ranges, 0, LogType::FILE);
  }
  ioStatistics_->read().increment(fileSize_);
  ioStatistics_->incRawBytesRead(fileSize_);
  ioStatistics_->queryThreadIoLatencyUs().increment(storageReadUs);
  ioStatistics_->storageReadLatencyUs().increment(storageReadUs);
  ioStatistics_->incTotalScanTimeNs(storageReadUs * 1'000);
  entry->setExclusiveToShared(options_.cacheable());
}

void CachedBufferedInput::load(const LogType /*unused*/) {
  // 'requests_ is cleared on exit.
  auto requests = std::move(requests_);
  cache::SsdFile* ssdFile{nullptr};
  auto* ssdCache = cache_->ssdCache();
  if (ssdCache != nullptr) {
    ssdFile = &ssdCache->file(fileNum_.id());
  }

  // Extra requests made for pre-loadable regions that are larger than
  // 'loadQuantum'.
  std::vector<std::unique_ptr<CacheRequest>> extraRequests;
  std::vector<CacheRequest*> storageLoad[2];
  std::vector<CacheRequest*> ssdLoad[2];
  for (auto& request : requests) {
    cache::TrackingData trackingData;
    const bool prefetchAnyway = request.trackingId.empty() ||
        request.trackingId.id() == StreamIdentifier::sequentialFile().id_;
    if (!prefetchAnyway && (tracker_ != nullptr)) {
      trackingData = tracker_->trackingData(request.trackingId);
    }
    const int loadIndex =
        (prefetchAnyway || isPrefetchPct(adjustedReadPct(trackingData))) ? 1
                                                                         : 0;
    auto parts = makeRequestParts(
        request, trackingData, options_.loadQuantum(), extraRequests);
    for (auto* initialPart : parts) {
      auto* part = initialPart;
      VELOX_CHECK_NOT_NULL(part->stream);
      // A request can be composed from any number of adjacent memory-cache
      // entries, an SSD prefix, and a storage suffix. Cache keys identify the
      // start offset but not the length, so a shorter entry at the same key is
      // a valid prefix rather than stale data.
      while (part != nullptr) {
        part->stream->recordCacheEntryForEviction(part->key.offset);
        const auto cachedSize = cache_->entrySizeForPlanning(part->key);
        if (cachedSize.has_value()) {
          VELOX_CHECK_GT(
              cachedSize.value(), 0, "Cache entries must make progress");
          if (cachedSize.value() >= part->size) {
            break;
          }
          part->key.offset += cachedSize.value();
          part->size -= cachedSize.value();
          continue;
        }

        // An entry may have been published after entrySizeForPlanning(). If
        // so, leave it for the stream to consume rather than planning
        // duplicate I/O.
        if (cache_->exists(part->key)) {
          break;
        }

        if (ssdFile != nullptr) {
          part->ssdPin = ssdFile->find(part->key);
          if (!part->ssdPin.empty()) {
            const uint64_t ssdSize = part->ssdPin.run().size();
            VELOX_CHECK_GT(ssdSize, 0, "SSD entries must make progress");
            if (ssdSize < part->size) {
              const auto suffixOffset = part->key.offset + ssdSize;
              const auto suffixSize = part->size - ssdSize;
              extraRequests.push_back(
                  std::make_unique<CacheRequest>(
                      RawFileCacheKey{part->key.fileNum, suffixOffset},
                      suffixSize,
                      part->trackingId));
              auto* suffix = extraRequests.back().get();
              suffix->stream = part->stream;
              suffix->coalesces = part->coalesces;
              part->size = ssdSize;
              ssdLoad[loadIndex].push_back(part);
              part = suffix;
              continue;
            }
            ssdLoad[loadIndex].push_back(part);
            break;
          }
        }
        storageLoad[loadIndex].push_back(part);
        break;
      }
    }
  }

  std::sort(storageLoad[0].begin(), storageLoad[0].end(), lessThan<false>);
  std::sort(storageLoad[1].begin(), storageLoad[1].end(), lessThan<false>);
  std::sort(ssdLoad[0].begin(), ssdLoad[0].end(), lessThan<true>);
  std::sort(ssdLoad[1].begin(), ssdLoad[1].end(), lessThan<true>);
  makeLoads<false>(storageLoad);
  makeLoads<true>(ssdLoad);
}

template <bool kSsd>
void CachedBufferedInput::makeLoads(std::vector<CacheRequest*> requests[2]) {
  std::vector<int32_t> groupEnds[2];
  groupEnds[1] = groupRequests<kSsd>(requests[1], true);
  moveCoalesced(
      requests[1],
      groupEnds[1],
      requests[0],
      [](auto* request) { return getOffset<kSsd>(*request); },
      [](auto* request) { return getOffset<kSsd>(*request) + request->size; });
  groupEnds[0] = groupRequests<kSsd>(requests[0], false);
  readRegions(requests[1], true, groupEnds[1]);
  readRegions(requests[0], false, groupEnds[0]);
}

template <bool kSsd>
std::vector<int32_t> CachedBufferedInput::groupRequests(
    const std::vector<CacheRequest*>& requests,
    bool prefetch) const {
  if (requests.empty() || (requests.size() < 2 && !prefetch)) {
    return {};
  }
  const int32_t maxDistance = kSsd ? 20'000 : options_.maxCoalesceDistance();

  // Combine adjacent short reads.
  int64_t coalescedBytes = 0;
  std::vector<int32_t> ends;
  ends.reserve(requests.size());
  std::vector<char> ranges;
  const auto stats = coalesceIo<CacheRequest*, char>(
      requests,
      maxDistance,
      std::numeric_limits<int32_t>::max(),
      [&](int32_t index) { return getOffset<kSsd>(*requests[index]); },
      [&](int32_t index) {
        const auto size = requests[index]->size;
        coalescedBytes += size;
        return size;
      },
      [&](int32_t index) {
        if (coalescedBytes > options_.maxCoalesceBytes()) {
          coalescedBytes = 0;
          return kNoCoalesce;
        }
        return requests[index]->coalesces ? 1 : kNoCoalesce;
      },
      [&](CacheRequest* /*request*/, std::vector<char>& ranges) {
        ranges.push_back(0);
      },
      [&](int32_t /*gap*/, std::vector<char> /*ranges*/) { /*no op*/ },
      [&](const std::vector<CacheRequest*>& /*requests*/,
          int32_t /*begin*/,
          int32_t end,
          uint64_t /*offset*/,
          const std::vector<char>& /*ranges*/) { ends.push_back(end); });
  ioStatistics_->readGap().merge(stats.gaps);
  ioStatistics_->incDuplicateRead(stats.duplicateRegions, stats.duplicateBytes);
  return ends;
}

namespace {
// Base class for CoalescedLoads for different storage types.
class DwioCoalescedLoadBase : public cache::CoalescedLoad {
 public:
  DwioCoalescedLoadBase(
      cache::AsyncDataCache& cache,
      std::shared_ptr<IoStatistics> ioStatistics,
      std::shared_ptr<velox::IoStats> ioStats,
      uint64_t groupId,
      std::vector<CacheRequest*> requests)
      : CoalescedLoad(makeKeys(requests), makeSizes(requests)),
        cache_(cache),
        ioStatistics_(std::move(ioStatistics)),
        ioStats_(std::move(ioStats)),
        groupId_(groupId) {
    requests_.reserve(requests.size());
    for (const auto& request : requests) {
      size_ += request->size;
      requests_.push_back(std::move(*request));
    }
  }

  const std::vector<CacheRequest>& requests() {
    return requests_;
  }

  int64_t size() const override {
    return size_;
  }

  std::string toString() const override {
    int32_t payload = 0;
    VELOX_CHECK(!requests_.empty());

    int32_t total = requests_.back().key.offset + requests_.back().size -
        requests_[0].key.offset;
    for (const auto& request : requests_) {
      payload += request.size;
    }
    return fmt::format(
        "<CoalescedLoad: {} entries, {} total {} extra>",
        requests_.size(),
        succinctBytes(total),
        succinctBytes(total - payload));
  }

 protected:
  void updateStats(const CoalesceIoStats& stats, bool prefetch, bool ssd) {
    if (ioStatistics_ == nullptr) {
      return;
    }
    ioStatistics_->incRawOverreadBytes(stats.extraBytes);
    if (ssd) {
      ioStatistics_->ssdRead().increment(stats.payloadBytes);
    } else {
      ioStatistics_->read().increment(stats.payloadBytes);
    }
    if (prefetch) {
      ioStatistics_->prefetch().increment(stats.payloadBytes);
    }
  }

  static std::vector<RawFileCacheKey> makeKeys(
      std::vector<CacheRequest*>& requests) {
    std::vector<RawFileCacheKey> keys;
    keys.reserve(requests.size());
    for (auto& request : requests) {
      keys.push_back(request->key);
    }
    return keys;
  }

  std::vector<int32_t> makeSizes(std::vector<CacheRequest*> requests) {
    std::vector<int32_t> sizes;
    sizes.reserve(requests.size());
    for (auto& request : requests) {
      sizes.push_back(request->size);
    }
    return sizes;
  }

  cache::AsyncDataCache& cache_;
  std::vector<CacheRequest> requests_;
  std::shared_ptr<IoStatistics> ioStatistics_;
  std::shared_ptr<velox::IoStats> ioStats_;
  const uint64_t groupId_;
  int64_t size_{0};
};

// Represents a CoalescedLoad from ReadFile, e.g. disagg disk.
class DwioCoalescedLoad : public DwioCoalescedLoadBase {
 public:
  DwioCoalescedLoad(
      cache::AsyncDataCache& cache,
      std::shared_ptr<ReadFileInputStream> input,
      std::shared_ptr<IoStatistics> ioStatistics,
      std::shared_ptr<velox::IoStats> ioStats,
      uint64_t groupId,
      std::vector<CacheRequest*> requests,
      int32_t maxCoalesceDistance)
      : DwioCoalescedLoadBase(
            cache,
            std::move(ioStatistics),
            std::move(ioStats),
            groupId,
            std::move(requests)),
        input_(std::move(input)),
        maxCoalesceDistance_(maxCoalesceDistance) {}

  bool isSsdLoad() const override {
    return false;
  }

  std::vector<CachePin> loadData(bool prefetch) override {
    std::vector<CachePin> pins;
    pins.reserve(keys_.size());
    cache_.makePins(
        keys_,
        [&](int32_t index) { return sizes_[index]; },
        [&](int32_t /*index*/, CachePin pin) {
          if (prefetch) {
            pin.checkedEntry()->setPrefetch(true);
          }
          pins.push_back(std::move(pin));
        },
        /*contiguous=*/false,
        cache::CacheEntrySizePolicy::kAllowSmaller);
    if (pins.empty()) {
      return pins;
    }
    auto stats = cache::readPins(
        pins,
        maxCoalesceDistance_,
        1000,
        [&](int32_t i) { return pins[i].entry()->offset(); },
        [&](const std::vector<CachePin>& /*pins*/,
            int32_t /*begin*/,
            int32_t /*end*/,
            uint64_t offset,
            const std::vector<folly::Range<char*>>& buffers) {
          input_->read(buffers, offset, LogType::FILE);
        });
    updateStats(stats, prefetch, false);
    return pins;
  }

  std::shared_ptr<ReadFileInputStream> input_;
  const int32_t maxCoalesceDistance_;
};

// Represents a CoalescedLoad from local SSD cache.
class SsdLoad : public DwioCoalescedLoadBase {
 public:
  SsdLoad(
      cache::AsyncDataCache& cache,
      std::shared_ptr<IoStatistics> ioStatistics,
      std::shared_ptr<velox::IoStats> ioStats,
      uint64_t groupId,
      std::vector<CacheRequest*> requests)
      : DwioCoalescedLoadBase(
            cache,
            std::move(ioStatistics),
            std::move(ioStats),
            groupId,
            std::move(requests)) {}

  bool isSsdLoad() const override {
    return true;
  }

  std::vector<CachePin> loadData(bool prefetch) override {
    std::vector<SsdPin> ssdPins;
    std::vector<CachePin> pins;
    cache_.makePins(
        keys_,
        [&](int32_t index) { return sizes_[index]; },
        [&](int32_t index, CachePin pin) {
          if (prefetch) {
            pin.checkedEntry()->setPrefetch(true);
          }
          pins.push_back(std::move(pin));
          ssdPins.push_back(std::move(requests_[index].ssdPin));
        },
        /*contiguous=*/false,
        cache::CacheEntrySizePolicy::kAllowSmaller);
    if (pins.empty()) {
      return pins;
    }
    assert(!ssdPins.empty()); // for lint.
    const auto stats = ssdPins[0].file()->load(ssdPins, pins);
    updateStats(stats, prefetch, true);
    return pins;
  }
};

} // namespace

uint64_t CachedBufferedInput::StreamCoalescedLoad::firstOffset() const {
  VELOX_CHECK(!regions.empty());
  return regions.front().offset;
}

uint64_t CachedBufferedInput::StreamCoalescedLoad::endOffset() const {
  VELOX_CHECK(!regions.empty());
  uint64_t end = 0;
  for (const auto& region : regions) {
    end = std::max(end, region.offset + region.length);
  }
  return end;
}

bool CachedBufferedInput::StreamCoalescedLoad::covers(uint64_t position) const {
  return std::any_of(regions.begin(), regions.end(), [&](const Region& region) {
    return position >= region.offset &&
        position - region.offset < region.length;
  });
}

void CachedBufferedInput::readRegion(
    const std::vector<CacheRequest*>& requests,
    bool prefetch) {
  if (requests.empty() || (requests.size() == 1 && !prefetch)) {
    return;
  }

  std::shared_ptr<cache::CoalescedLoad> load;
  if (!requests[0]->ssdPin.empty()) {
    load = std::make_shared<SsdLoad>(
        *cache_, ioStatistics_, ioStats_, groupId_.id(), requests);
  } else {
    load = std::make_shared<DwioCoalescedLoad>(
        *cache_,
        input_,
        ioStatistics_,
        ioStats_,
        groupId_.id(),
        requests,
        options_.maxCoalesceDistance());
  }
  coalescedLoads_.push_back(load);
  folly::F14FastMap<const SeekableInputStream*, std::vector<Region>>
      streamRegions;
  for (const auto* request : requests) {
    if (request->stream == nullptr) {
      continue;
    }
    streamRegions[request->stream].emplace_back(
        request->key.offset, request->size);
  }
  streamToCoalescedLoad_.withWLock([&](auto& loads) {
    for (auto& [stream, regions] : streamRegions) {
      std::sort(regions.begin(), regions.end());
      auto& streamLoads = loads[stream];
      const auto insertion = std::lower_bound(
          streamLoads.begin(),
          streamLoads.end(),
          regions.front().offset,
          [](const StreamCoalescedLoad& queued, uint64_t newOffset) {
            return queued.firstOffset() < newOffset;
          });
      streamLoads.insert(
          insertion, StreamCoalescedLoad{std::move(regions), load});
    }
  });
}

void CachedBufferedInput::readRegions(
    const std::vector<CacheRequest*>& requests,
    bool prefetch,
    const std::vector<int32_t>& groupEnds) {
  if (requests.empty()) {
    VELOX_CHECK(groupEnds.empty());
    return;
  }
  // Record the starting position so that we only submit the loads created by
  // this call. Without this, non-prefetch loads or stale loads from previous
  // cycles could be incorrectly submitted for async prefetching.
  const int32_t startIndex = static_cast<int32_t>(coalescedLoads_.size());
  int32_t requestIdx{0};
  std::vector<CacheRequest*> requestGroup;
  for (auto groupEndIdx : groupEnds) {
    while (requestIdx < groupEndIdx) {
      requestGroup.push_back(requests[requestIdx++]);
    }
    readRegion(requestGroup, prefetch);
    requestGroup.clear();
  }

  if (prefetch && executor_) {
    // Only submit the loads created by this call to the executor.
    for (auto i = startIndex; i < coalescedLoads_.size(); ++i) {
      auto& load = coalescedLoads_[i];
      if (load->state() == CoalescedLoad::State::kPlanned) {
        executor_->add(
            [pendingLoad = load, ssdSavable = options_.cacheable()]() {
              pendingLoad->loadOrFuture(nullptr, ssdSavable);
            });
      }
    }
    // Remove the loads that were complete. There can be done loads if the same
    // CachedBufferedInput has multiple cycles of enqueues and loads.
    std::vector<int32_t> doneIndices;
    for (int32_t i = 0; i < startIndex; ++i) {
      if (coalescedLoads_[i]->state() != CoalescedLoad::State::kPlanned) {
        doneIndices.push_back(i);
      }
    }
    for (int i = 0, j = 0, k = 0; i < coalescedLoads_.size(); ++i) {
      if (j < doneIndices.size() && doneIndices[j] == i) {
        ++j;
      } else {
        coalescedLoads_[k++] = std::move(coalescedLoads_[i]);
      }
    }
    coalescedLoads_.resize(coalescedLoads_.size() - doneIndices.size());
  }
}

std::shared_ptr<cache::CoalescedLoad> CachedBufferedInput::coalescedLoad(
    const SeekableInputStream* stream,
    uint64_t position) {
  return streamToCoalescedLoad_.withWLock(
      [&](auto& loads) -> std::shared_ptr<cache::CoalescedLoad> {
        auto it = loads.find(stream);
        if (it == loads.end()) {
          return nullptr;
        }
        VELOX_CHECK(!it->second.empty());
        auto& streamLoads = it->second;
        streamLoads.erase(
            std::remove_if(
                streamLoads.begin(),
                streamLoads.end(),
                [&](const auto& queued) {
                  return queued.endOffset() <= position;
                }),
            streamLoads.end());
        if (streamLoads.empty()) {
          loads.erase(it);
          return nullptr;
        }

        const auto covering = std::find_if(
            streamLoads.begin(), streamLoads.end(), [&](const auto& queued) {
              return queued.covers(position);
            });
        if (covering == streamLoads.end()) {
          return nullptr;
        }
        auto load = covering->load;

        // The selected load may cover requests from multiple streams. Once
        // one stream triggers it, none of the correlated streams should try
        // to trigger the same load again.
        for (auto streamIt = loads.begin(); streamIt != loads.end();) {
          auto& queuedLoads = streamIt->second;
          queuedLoads.erase(
              std::remove_if(
                  queuedLoads.begin(),
                  queuedLoads.end(),
                  [&](const auto& queued) { return queued.load == load; }),
              queuedLoads.end());
          if (queuedLoads.empty()) {
            auto eraseIt = streamIt++;
            loads.erase(eraseIt);
          } else {
            ++streamIt;
          }
        }
        return load;
      });
}

void CachedBufferedInput::discardCoalescedLoads(
    const SeekableInputStream* stream) {
  streamToCoalescedLoad_.withWLock([&](auto& loads) {
    const auto it = loads.find(stream);
    if (it == loads.end()) {
      return;
    }
    folly::F14FastSet<const cache::CoalescedLoad*> discarded;
    for (const auto& queued : it->second) {
      discarded.insert(queued.load.get());
    }
    for (auto streamIt = loads.begin(); streamIt != loads.end();) {
      auto& queuedLoads = streamIt->second;
      queuedLoads.erase(
          std::remove_if(
              queuedLoads.begin(),
              queuedLoads.end(),
              [&](const auto& queued) {
                return discarded.contains(queued.load.get());
              }),
          queuedLoads.end());
      if (queuedLoads.empty()) {
        auto eraseIt = streamIt++;
        loads.erase(eraseIt);
      } else {
        ++streamIt;
      }
    }
  });
}

void CachedBufferedInput::reset() {
  BufferedInput::reset();
  for (auto& load : coalescedLoads_) {
    load->cancel();
  }
  coalescedLoads_.clear();
  streamToCoalescedLoad_.wlock()->clear();
  requests_.clear();
}

std::unique_ptr<SeekableInputStream> CachedBufferedInput::read(
    uint64_t offset,
    uint64_t length,
    LogType /*logType*/) const {
  VELOX_CHECK_LE(offset + length, fileSize_);
  auto stream = std::make_unique<CacheInputStream>(
      const_cast<CachedBufferedInput*>(this),
      ioStatistics_.get(),
      Region{offset, length},
      input_,
      fileNum_.id(),
      options_.cacheable(),
      nullptr,
      TrackingId(),
      0,
      options_.loadQuantum());
  if (preloaded()) {
    stream->setPreloadedPin(preloadPin_);
  }
  return stream;
}

bool CachedBufferedInput::prefetch(Region region) {
  const int32_t numPages = memory::AllocationTraits::numPages(region.length);
  if (!shouldPreload(numPages)) {
    return false;
  }
  auto stream = enqueue(region, nullptr);
  load(LogType::FILE);
  // Remove all coalesced loads made for the temporary stream. They are
  // submitted independently when an executor is present and will not be
  // triggered through this stream.
  discardCoalescedLoads(stream.get());
  return true;
}

void CachedBufferedInput::cacheRegion(
    uint64_t offset,
    uint64_t length,
    std::string_view data) {
  VELOX_CHECK_EQ(data.size(), length);
  auto iobuf = folly::IOBuf::wrapBufferAsValue(data.data(), data.size());
  cacheRegion(offset, length, iobuf, 0);
}

void CachedBufferedInput::cacheRegion(
    uint64_t offset,
    uint64_t length,
    const folly::IOBuf& buffer,
    uint64_t bufferOffset) {
  auto pin =
      cache_->findOrCreate(RawFileCacheKey{fileNum_.id(), offset}, length);
  // Empty pin means the cache is at capacity and cannot accept new entries.
  // Non-exclusive means another thread already cached this region; skip the
  // duplicate write.
  if (pin.empty() || !pin.checkedEntry()->isExclusive()) {
    return;
  }

  folly::io::Cursor cursor(&buffer);
  cursor.skip(bufferOffset);
  VELOX_CHECK_GE(
      cursor.totalLength(),
      length,
      "IOBuf has {} bytes after offset {}, need {}",
      cursor.totalLength(),
      bufferOffset,
      length);

  auto* entry = pin.checkedEntry();
  if (entry->hasContiguousData()) {
    cursor.pull(entry->contiguousData(), length);
  } else {
    auto& allocation = entry->nonContiguousData();
    uint64_t copyBytes = 0;
    for (int i = 0; i < allocation.numRuns() && copyBytes < length; ++i) {
      const auto run = allocation.runAt(i);
      const uint64_t copySize =
          std::min<uint64_t>(run.numBytes(), length - copyBytes);
      cursor.pull(run.data(), copySize);
      copyBytes += copySize;
    }
    VELOX_CHECK_EQ(copyBytes, length);
  }

  // Clear the first-use flag since this entry is being populated externally
  // (not loaded on-demand). The first findCachedRegion access should count
  // as a cache hit.
  entry->getAndClearFirstUseFlag();
  entry->setExclusiveToShared();
}

std::optional<CachedRegion> CachedBufferedInput::findCachedRegion(
    uint64_t offset) const {
  const cache::RawFileCacheKey key{fileNum_.id(), offset};
  for (;;) {
    folly::SemiFuture<bool> waitFuture(false);
    auto result = cache_->find(key, &waitFuture);
    if (!result.has_value()) {
      return std::nullopt;
    }
    if (!result->empty()) {
      auto* entry = result->checkedEntry();
      if (!entry->getAndClearFirstUseFlag()) {
        ioStatistics_->ramHit().increment(entry->size());
      }
      return CachedRegion{std::move(*result)};
    }
    // Entry is exclusive — wait for it to become shared, then retry.
    uint64_t waitUs{0};
    {
      MicrosecondWallTimer timer(&waitUs);
      std::move(waitFuture)
          .via(&folly::QueuedImmediateExecutor::instance())
          .wait();
    }
    ioStatistics_->queryThreadIoLatencyUs().increment(waitUs);
    ioStatistics_->cacheWaitLatencyUs().increment(waitUs);
  }
}

} // namespace facebook::velox::dwio::common
