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

#include "velox/experimental/ucx-exchange/CpuRowFrameCompletionTracker.h"

#include <stdexcept>

namespace facebook::velox::ucx_exchange {

bool CpuRowCallbackOnce::tryClaim() noexcept {
  bool expected = false;
  return claimed_.compare_exchange_strong(
      expected, true, std::memory_order_acq_rel, std::memory_order_acquire);
}

CpuRowFrameCompletionTracker::CpuRowFrameCompletionTracker(size_t frameCount)
    : frameCount_(frameCount),
      completed_(std::make_unique<std::atomic<bool>[]>(frameCount)),
      remaining_(frameCount) {
  if (frameCount == 0) {
    throw std::invalid_argument(
        "CpuRowFrameCompletionTracker requires at least one frame");
  }
  for (size_t i = 0; i < frameCount_; ++i) {
    completed_[i].store(false, std::memory_order_relaxed);
  }
}

CpuRowFrameCompletionTracker::Result CpuRowFrameCompletionTracker::complete(
    size_t frameIndex,
    ucs_status_t status) {
  if (frameIndex >= frameCount_) {
    throw std::out_of_range(
        "CpuRowFrameCompletionTracker frame index is out of range");
  }

  bool expected = false;
  if (!completed_[frameIndex].compare_exchange_strong(
          expected,
          true,
          std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    return {State::kDuplicate, finalStatus_.load(std::memory_order_acquire)};
  }

  if (status != UCS_OK) {
    ucs_status_t expectedStatus = UCS_OK;
    finalStatus_.compare_exchange_strong(
        expectedStatus,
        status,
        std::memory_order_acq_rel,
        std::memory_order_acquire);
  }

  const size_t before = remaining_.fetch_sub(1, std::memory_order_acq_rel);
  return {
      before == 1 ? State::kComplete : State::kPending,
      finalStatus_.load(std::memory_order_acquire)};
}

} // namespace facebook::velox::ucx_exchange
