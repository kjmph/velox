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

#include <ucs/type/status.h>
#include <atomic>
#include <cstddef>
#include <memory>

namespace facebook::velox::ucx_exchange {

/// Ensures a UCX callback body runs at most once, even if the transport invokes
/// the callback more than once for the same request.
class CpuRowCallbackOnce {
 public:
  bool tryClaim() noexcept;

 private:
  std::atomic<bool> claimed_{false};
};

/// Aggregates the completion status of a fixed set of CPU-row UCX frames.
///
/// Each frame index contributes at most once. Duplicate callbacks neither
/// decrement the remaining-frame count nor change the aggregate status.
class CpuRowFrameCompletionTracker {
 public:
  enum class State {
    kDuplicate,
    kPending,
    kComplete,
  };

  struct Result {
    State state;
    ucs_status_t finalStatus;
  };

  explicit CpuRowFrameCompletionTracker(size_t frameCount);

  Result complete(size_t frameIndex, ucs_status_t status);

 private:
  const size_t frameCount_;
  std::unique_ptr<std::atomic<bool>[]> completed_;
  std::atomic<size_t> remaining_;
  std::atomic<ucs_status_t> finalStatus_{UCS_OK};
};

} // namespace facebook::velox::ucx_exchange
