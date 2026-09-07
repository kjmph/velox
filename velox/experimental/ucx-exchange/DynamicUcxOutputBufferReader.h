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

#include <folly/io/IOBuf.h>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace facebook::velox::exec {
class DefaultOutputBufferManager;
}

namespace facebook::velox::ucx_exchange {

class DynamicUcxOutputBufferReader
    : public std::enable_shared_from_this<DynamicUcxOutputBufferReader> {
 public:
  /// One batch of pages, as handed over by the buffer manager.
  struct Data {
    /// Pages in sequence order. Empty when the producer had nothing ready.
    std::vector<std::unique_ptr<folly::IOBuf>> pages;

    /// Sequence number of the first page in 'pages'.
    int64_t sequence{0};

    /// Sizes of pages still buffered at the producer, for flow control.
    std::vector<int64_t> remainingBytes;

    /// True once the producer has signalled that no more data will follow.
    bool atEnd{false};
  };

  using DataCallback = std::function<void(Data)>;

  /// Reads 'destination' of 'taskId' from 'manager', from the start.
  DynamicUcxOutputBufferReader(
      std::shared_ptr<exec::DefaultOutputBufferManager> manager,
      std::string taskId,
      int destination);

  ~DynamicUcxOutputBufferReader();

  /// Requests the next batch, capped at 'maxBytes'. Also acknowledges every
  /// page the previous batch delivered.
  void request(uint64_t maxBytes, DataCallback callback);

  /// Stops reading. Idempotent. Does not release the producer's results --
  /// sibling destinations may still be reading them; deleteResults() does.
  void close();

  /// Releases the producer's buffered results for this destination. Call once,
  /// after end-of-stream has been consumed.
  void deleteResults();

 private:
  // The ordinary output buffer manager. OutputBufferManager is abstract on this
  // branch, with the data plane on the concrete managers.
  const std::shared_ptr<exec::DefaultOutputBufferManager> manager_;
  const std::string taskId_;
  const int destination_;

  std::atomic<int64_t> sequence_;
  std::atomic<bool> atEnd_{false};
  std::atomic<bool> closed_{false};
};

} // namespace facebook::velox::ucx_exchange
