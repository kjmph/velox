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

#include "velox/experimental/ucx-exchange/ExchangeFormatRegistry.h"

namespace facebook::velox::ucx_exchange {

// static
ExchangeFormatRegistry& ExchangeFormatRegistry::instance() {
  static ExchangeFormatRegistry registry;
  return registry;
}

void ExchangeFormatRegistry::setExchangeFormat(
    const std::string& taskId,
    int destination,
    Format format) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (format == Format::kCudf) {
    formats_[taskId][destination] = format;
    return;
  }
  auto it = formats_.find(taskId);
  if (it == formats_.end()) {
    return;
  }
  it->second[destination] = format;
}

ExchangeFormatRegistry::Format ExchangeFormatRegistry::getExchangeFormat(
    const std::string& taskId,
    int destination) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto taskIt = formats_.find(taskId);
  if (taskIt == formats_.end()) {
    return Format::kUnknown;
  }
  auto it = taskIt->second.find(destination);
  return it == taskIt->second.end() ? Format::kUnknown : it->second;
}

void ExchangeFormatRegistry::trackTask(const std::string& taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  formats_.try_emplace(taskId);
}

void ExchangeFormatRegistry::forgetTask(const std::string& taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  formats_.erase(taskId);
}

} // namespace facebook::velox::ucx_exchange
