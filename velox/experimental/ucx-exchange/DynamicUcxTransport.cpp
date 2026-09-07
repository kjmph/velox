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

#include "velox/experimental/ucx-exchange/DynamicUcxTransport.h"

#include <cstdlib>
#include <string_view>

namespace facebook::velox::ucx_exchange {
namespace {

bool& dynamicFlag() {
  static bool enabled{[] {
    const char* value = std::getenv("VELOX_UCX_DYNAMIC");
    return value != nullptr && std::string_view{value} == "1";
  }()};
  return enabled;
}

} // namespace

bool useDynamicUcx() {
  return dynamicFlag();
}

void enableDynamicUcx(bool enabled) {
  dynamicFlag() = enabled;
}

} // namespace facebook::velox::ucx_exchange
