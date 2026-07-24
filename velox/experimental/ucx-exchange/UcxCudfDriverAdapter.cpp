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
#include "velox/experimental/ucx-exchange/UcxCudfDriverAdapter.h"

#include <atomic>
#include <mutex>
#include <thread>
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/UcxExchangeRegistration.h"

namespace facebook::velox::ucx_exchange {
namespace {
std::once_flag communicatorStartedFlag;
std::atomic<bool> communicatorStarted{false};
} // namespace

bool startCudfUcxExchange() {
  const auto& config = cudf_velox::CudfConfig::getInstance();
  if (!config.enabled || !config.exchange) {
    return false;
  }

  if (config.exchangeServerPort <= 0 || config.exchangeServerPort > 65535) {
    LOG(ERROR) << "[CUDF-UCX] Invalid cudf.exchange.server.port="
               << config.exchangeServerPort
               << "; keeping standard exchange operators";
    return false;
  }

  // Register the receive adapter before advertising the output transport.
  registerCudfUcxDriverAdapter();
  registerUcxTransports();

  std::call_once(communicatorStartedFlag, [&config]() {
    const auto port = static_cast<uint16_t>(config.exchangeServerPort);
    LOG(INFO) << "[CUDF-UCX] Starting Communicator on port "
              << config.exchangeServerPort;
    ContinueFuture readyFuture;
    auto communicator = Communicator::initAndGet(port, "", &readyFuture);
    if (!communicator) {
      LOG(ERROR) << "[CUDF-UCX] Communicator::initAndGet failed";
      return;
    }
    std::thread([communicator = std::move(communicator)]() {
      communicator->run();
    }).detach();
    if (readyFuture.valid()) {
      std::move(readyFuture).wait();
    }
    communicatorStarted.store(true, std::memory_order_release);
  });

  return communicatorStarted.load(std::memory_order_acquire);
}

bool cudfUcxExchangeStarted() {
  return communicatorStarted.load(std::memory_order_acquire);
}

void registerCudfUcxDriverAdapter() {
  // The receive and output registries construct UCX operators directly. The
  // cuDF adapter registered by registerCudf() handles their GPU boundaries.
}

} // namespace facebook::velox::ucx_exchange
