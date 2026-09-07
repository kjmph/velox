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

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace facebook::velox::ucx_exchange {

/// What format each shuffle destination's pages take, used by dynamic UCX.
/// Thread-safe.

class ExchangeFormatRegistry {
 public:
  static ExchangeFormatRegistry& instance();

  enum class Format {
    kUnknown,
    kCudf,
    kVelox,
  };

  void
  setExchangeFormat(const std::string& taskId, int destination, Format format);
  Format getExchangeFormat(const std::string& taskId, int destination) const;

  bool iscuDFExchange(const std::string& taskId, int destination) const {
    return getExchangeFormat(taskId, destination) == Format::kCudf;
  }

  bool isVeloxExchange(const std::string& taskId, int destination) const {
    return getExchangeFormat(taskId, destination) == Format::kVelox;
  }

  void trackTask(const std::string& taskId);
  void forgetTask(const std::string& taskId);

 private:
  mutable std::mutex mutex_;
  // Keyed by task so forgetTask() is one erase.
  std::unordered_map<std::string, std::unordered_map<int, Format>> formats_;
};

} // namespace facebook::velox::ucx_exchange
