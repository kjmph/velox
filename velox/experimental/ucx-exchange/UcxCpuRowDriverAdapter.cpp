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
#include "velox/experimental/ucx-exchange/UcxCpuRowDriverAdapter.h"

#include <glog/logging.h>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/ByteStream.h"
#include "velox/exec/Driver.h"
#include "velox/exec/ExchangeTransportRegistry.h"
#include "velox/exec/Merge.h"
#include "velox/exec/MergeSource.h"
#include "velox/exec/OutputTransportRegistry.h"
#include "velox/exec/SerializedPage.h"
#include "velox/exec/Task.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowExchange.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowExchangeClient.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowOutputQueueManager.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowPartitionedOutput.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::ucx_exchange {

namespace {

// Cache the env var lookups: they don't change at runtime, and getenv()
// is the kind of thing you don't want in a hot path.
bool readEnabledFromEnv() {
  const char* value = std::getenv(kCpuExchangeEnabledEnv);
  if (value == nullptr || *value == '\0') {
    return false;
  }
  if (value[0] == '1' && value[1] == '\0') {
    return true;
  }
  if (value[0] == '0' && value[1] == '\0') {
    return false;
  }
  LOG(WARNING) << "[CPU-UCX] Ignoring invalid " << kCpuExchangeEnabledEnv << "="
               << value << "; expected 0 or 1";
  return false;
}

uint16_t readPortFromEnv() {
  const char* value = std::getenv(kCpuExchangePortEnv);
  if (value == nullptr || *value == '\0') {
    return kDefaultCpuExchangePort;
  }
  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || errno != 0 || parsed <= 0 ||
      parsed > 65535) {
    LOG(ERROR) << "[CPU-UCX] " << kCpuExchangePortEnv
               << " invalid value: " << value << ", using default "
               << kDefaultCpuExchangePort;
    return kDefaultCpuExchangePort;
  }
  return static_cast<uint16_t>(parsed);
}

// Read configuration lazily at explicit service startup, avoiding
// cross-translation-unit static initialization order and repeated getenv()
// calls. C++11 guarantees thread-safe function-local static initialization.
struct CpuUcxConfig {
  bool enabled;
  uint16_t port;
};

const CpuUcxConfig& cpuUcxConfig() {
  static const CpuUcxConfig instance{readEnabledFromEnv(), readPortFromEnv()};
  return instance;
}

void registerCpuUcxOutputTransport() {
  auto manager = UcxCpuRowOutputQueueManager::getInstanceRef();
  if (auto existing = exec::OutputTransportRegistry::tryGet(
          std::string{core::TransportKind::kUcx})) {
    VELOX_CHECK(
        std::dynamic_pointer_cast<UcxCpuRowOutputQueueManager>(
            existing->manager) == manager,
        "UCX output transport is already registered with an incompatible "
        "manager");
    return;
  }

  auto entry = exec::OutputTransportEntry::make<UcxCpuRowOutputQueueManager>(
      manager,
      [](int32_t operatorId,
         exec::DriverCtx* ctx,
         const std::shared_ptr<const core::PartitionedOutputNode>& node,
         bool eagerFlush,
         const std::shared_ptr<UcxCpuRowOutputQueueManager>& manager)
          -> std::unique_ptr<exec::Operator> {
        return std::make_unique<UcxCpuRowPartitionedOutput>(
            operatorId, ctx, node, eagerFlush, manager);
      });
  exec::OutputTransportRegistry::global().insert(
      std::string{core::TransportKind::kUcx}, std::move(entry));
}

// Communicator is a singleton. Prestissimo starts it eagerly before announcing
// the worker, while the client retains lazy startup as a fallback for other
// embedders. Once running, it stays up for the process lifetime. Detach only
// releases std::thread ownership; it does not stop the Communicator. A graceful
// service shutdown still needs to call Communicator::stop(). If that does not
// happen, process exit tears down the detached thread with the rest of the
// worker.
std::once_flag communicatorStartedFlag;
std::atomic<bool> communicatorReady{false};

class UcxCpuRowMergeExchangeSource : public exec::MergeSource {
 public:
  UcxCpuRowMergeExchangeSource(
      exec::MergeExchange* mergeExchange,
      const std::string& taskId,
      int destination)
      : mergeExchange_(mergeExchange),
        client_(std::make_shared<UcxCpuRowExchangeClient>(
            mergeExchange->taskId(),
            destination,
            1)) {
    client_->addRemoteTaskId(taskId);
    client_->noMoreRemoteTasks();
  }

  ~UcxCpuRowMergeExchangeSource() override {
    close();
  }

  void start() override {}

  exec::BlockingReason started(ContinueFuture* /*unused*/) override {
    VELOX_NYI();
  }

  exec::BlockingReason
  next(RowVectorPtr& data, ContinueFuture* future, bool& drained) override {
    drained = false;
    data.reset();

    if (atEnd_ && !currentPage_) {
      return exec::BlockingReason::kNotBlocked;
    }

    if (!currentPage_) {
      currentReceived_ = client_->next(0, &atEnd_, future);
      if (!currentReceived_) {
        if (atEnd_) {
          return exec::BlockingReason::kNotBlocked;
        }
        return exec::BlockingReason::kWaitForProducer;
      }

      currentPage_ = std::make_unique<exec::PrestoSerializedPage>(
          std::move(currentReceived_->data),
          /*onDestructionCb=*/nullptr,
          /*numRows=*/std::nullopt);
      inputStream_ = currentPage_->prepareStreamForDeserialize();
      mergeExchange_->stats().wlock()->rawInputBytes += currentPage_->size();
    }

    if (!inputStream_->atEnd()) {
      VectorStreamGroup::read(
          inputStream_.get(),
          mergeExchange_->pool(),
          mergeExchange_->outputType(),
          mergeExchange_->serde(),
          &data,
          mergeExchange_->serdeOptions());

      auto lockedStats = mergeExchange_->stats().wlock();
      lockedStats->addInputVector(data->estimateFlatSize(), data->size());
      lockedStats->rawInputPositions += data->size();
    }

    if (inputStream_->atEnd()) {
      currentPage_.reset();
      currentReceived_.reset();
      inputStream_.reset();
    }

    return exec::BlockingReason::kNotBlocked;
  }

  void close() override {
    if (client_) {
      client_->close();
      client_ = nullptr;
    }
    currentReceived_.reset();
    currentPage_.reset();
    inputStream_.reset();
  }

  exec::BlockingReason
  enqueue(RowVectorPtr input, ContinueFuture* future, bool drained) override {
    VELOX_FAIL();
  }

 private:
  exec::MergeExchange* const mergeExchange_;
  std::shared_ptr<UcxCpuRowExchangeClient> client_;
  UcxCpuRowReceivedPtr currentReceived_;
  std::unique_ptr<ByteInputStream> inputStream_;
  std::unique_ptr<exec::SerializedPageBase> currentPage_;
  bool atEnd_ = false;
};

void ensureCommunicatorStarted() {
  std::call_once(communicatorStartedFlag, []() {
    const auto& cfg = cpuUcxConfig();
    LOG(INFO) << "[CPU-UCX] Starting Communicator on port " << cfg.port;
    // coordinatorURL is currently dead state in Communicator (unused).
    ContinueFuture readyFuture;
    auto comm = Communicator::initAndGet(cfg.port, "", &readyFuture);
    if (!comm) {
      LOG(ERROR) << "[CPU-UCX] Communicator::initAndGet failed";
      return;
    }
    std::thread([c = comm]() { c->run(); }).detach();
    if (readyFuture.valid()) {
      std::move(readyFuture).wait();
    }
    communicatorReady.store(true, std::memory_order_release);
    LOG(INFO) << "[CPU-UCX] Communicator ready on port " << cfg.port;
  });
}

void registerCpuUcxExchangeTransport() {
  auto entry = exec::ExchangeTransportEntry::make<UcxCpuRowExchangeClient>(
      [](const exec::ExchangeClientContext& context) {
        return std::make_shared<UcxCpuRowExchangeClient>(
            context.taskId, context.destination, context.numberOfConsumers);
      },
      [](int32_t operatorId,
         exec::DriverCtx* ctx,
         const std::shared_ptr<const core::ExchangeNode>& node,
         const std::shared_ptr<UcxCpuRowExchangeClient>& client)
          -> std::unique_ptr<exec::Operator> {
        return std::make_unique<UcxCpuRowExchange>(
            operatorId, ctx, node, client);
      },
      [](int32_t operatorId,
         exec::DriverCtx* ctx,
         const std::shared_ptr<const core::ExchangeNode>& node,
         const std::shared_ptr<UcxCpuRowExchangeClient>& /*client*/)
          -> std::unique_ptr<exec::Operator> {
        auto mergeNode =
            std::dynamic_pointer_cast<const core::MergeExchangeNode>(node);
        VELOX_CHECK_NOT_NULL(mergeNode);
        // MergeExchange preserves ordering with one independent UCX client
        // per producer via the registered MergeSource factory.
        return std::make_unique<exec::MergeExchange>(operatorId, ctx, mergeNode);
      });
  exec::ExchangeTransportRegistry::global().insert(
      std::string{core::TransportKind::kUcx}, std::move(entry));
}

std::once_flag registeredFlag;

} // namespace

void registerCpuUcxDriverAdapter() {
  if (!cpuUcxConfig().enabled) {
    return;
  }
  std::call_once(registeredFlag, []() {
    registerCpuUcxOutputTransport();
    registerCpuUcxExchangeTransport();
    exec::MergeSource::registerMergeExchangeSourceFactory(
        std::string{core::TransportKind::kUcx},
        [](exec::MergeExchange* mergeExchange,
           const std::string& taskId,
           int destination) -> std::shared_ptr<exec::MergeSource> {
          ensureCommunicatorStarted();
          return std::make_shared<UcxCpuRowMergeExchangeSource>(
              mergeExchange, taskId, destination);
        });
    LOG(INFO) << "[CPU-UCX] Exchange and output transports registered";
  });
}

bool cpuUcxExchangeEnabled() {
  return cpuUcxConfig().enabled;
}

bool startCpuUcxExchange() {
  if (!cpuUcxConfig().enabled) {
    return false;
  }

  // Prestissimo calls this before service announcement. Keep the startup API
  // independently safe and idempotent for other embedders.
  registerCpuUcxDriverAdapter();
  registerCpuUcxOutputTransport();
  ensureCommunicatorStarted();
  return communicatorReady.load(std::memory_order_acquire);
}

} // namespace facebook::velox::ucx_exchange
