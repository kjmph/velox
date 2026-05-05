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

/// CPU UCX wiring for Velox.
///
/// The process-wide flag only enables the adapter. The actual replacement
/// decision is made from each ExchangeNode and PartitionedOutputNode transport
/// type, so HTTP edges stay on the standard Velox path.
///
/// Configuration is via environment variables, not QueryConfig: the
/// listener port has to be known at process startup, and Prestissimo's
/// `etc/config.properties` doesn't auto-flow into Velox's QueryConfig
/// (only session properties do, which are per-query).
///
///   VELOX_UCX_CPU_EXCHANGE=1       # any of "1", "true", "TRUE", "yes"
///   VELOX_UCX_CPU_PORT=10083       # default kDefaultCpuExchangePort
///
/// The adapter assumes that exchange splits on UCX edges point at native
/// workers running the same CPU UCX listener convention. Coordinator-bound
/// output and other HTTP edges remain HTTP.

namespace facebook::velox::ucx_exchange {

constexpr const char* kCpuExchangeEnabledEnv = "VELOX_UCX_CPU_EXCHANGE";
constexpr const char* kCpuExchangePortEnv = "VELOX_UCX_CPU_PORT";

/// Default UCX listener port when VELOX_UCX_CPU_PORT isn't set. Picked
/// to not collide with Prestissimo's HTTP ports (10000-10080) or with
/// the cudf path's `cudf.exchange.server.port` default (30050).
constexpr uint16_t kDefaultCpuExchangePort = 30060;

/// Registers the CPU UCX DriverAdapter with `exec::DriverFactory`. Idempotent.
///
/// Called automatically at library load via a static initializer. Exposed
/// as a function too so callers (tests, an explicit Prestissimo hook)
/// can force-init in a deterministic order.
void registerCpuUcxDriverAdapter();

} // namespace facebook::velox::ucx_exchange
