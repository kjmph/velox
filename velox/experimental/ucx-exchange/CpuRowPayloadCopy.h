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
#include <span>
#include <vector>

namespace facebook::velox::ucx_exchange {

/// Copies IOBuf chains into one advertised frame.
///
/// Throws std::length_error before an individual range would overrun the
/// destination, or after copying if the payloads contain fewer bytes than the
/// advertised frame.
void copyCpuRowPayloads(
    const std::vector<const folly::IOBuf*>& payloads,
    std::span<uint8_t> destination);

} // namespace facebook::velox::ucx_exchange
