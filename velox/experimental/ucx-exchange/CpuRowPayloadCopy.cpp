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

#include "velox/experimental/ucx-exchange/CpuRowPayloadCopy.h"

#include <cstring>
#include <stdexcept>

namespace facebook::velox::ucx_exchange {

void copyCpuRowPayloads(
    const std::vector<const folly::IOBuf*>& payloads,
    std::span<uint8_t> destination) {
  size_t offset = 0;
  for (const auto* payload : payloads) {
    if (payload == nullptr) {
      throw std::invalid_argument("CPU row payload must not be null");
    }
    for (const auto range : *payload) {
      if (range.size() > destination.size() - offset) {
        throw std::length_error(
            "CPU row payload bytes exceed advertised packed frame size");
      }
      std::memcpy(destination.data() + offset, range.data(), range.size());
      offset += range.size();
    }
  }

  if (offset != destination.size()) {
    throw std::length_error(
        "CPU row payload bytes do not match advertised packed frame size");
  }
}

} // namespace facebook::velox::ucx_exchange
