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
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/ByteStream.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/Merge.h"
#include "velox/exec/MergeSource.h"
#include "velox/exec/PartitionedOutput.h"
#include "velox/exec/SerializedPage.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowExchange.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowExchangeClient.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowPartitionedOutput.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::ucx_exchange {

namespace {

constexpr const char* kAdapterLabel = "CpuUcx";

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

// Lazy config read. We can't use file-scope `const bool kEnabled = ...`
// because that competes for ordering with the __attribute__((constructor))
// auto-register below; the env-reading global initializer can run AFTER
// the constructor function fires, leaving kEnabled=0/kPort=0 even though
// the env vars are set. Function-local statics defer the read until first
// call, well past static-init time. C++11 guarantees thread-safe init.
struct CpuUcxConfig {
  bool enabled;
  uint16_t port;
};

const CpuUcxConfig& cpuUcxConfig() {
  static const CpuUcxConfig instance{readEnabledFromEnv(), readPortFromEnv()};
  return instance;
}

// Communicator is a singleton; the adapter lazy-starts it on first driver that
// opts in. Once running, it stays up for the process lifetime. Detach only
// releases std::thread ownership; it does not stop the Communicator. A graceful
// service shutdown still needs to call Communicator::stop(). If that does not
// happen, process exit tears down the detached thread with the rest of the
// worker.
std::once_flag communicatorStartedFlag;

// Per-(taskId, pipelineId) shared UcxCpuRowExchangeClient. All drivers
// in the same pipeline of the same task share one client so that splits
// added by driver 0 (the one that processes them) populate the queue
// the other drivers wait on. Without this every driver had its own
// queue with numSources=0 and would block forever.
struct TaskPipelineKey {
  std::string taskId;
  int pipelineId;
  bool operator==(const TaskPipelineKey& o) const {
    return taskId == o.taskId && pipelineId == o.pipelineId;
  }
};
struct TaskPipelineKeyHash {
  size_t operator()(const TaskPipelineKey& k) const {
    return std::hash<std::string>{}(k.taskId) ^
        (static_cast<size_t>(k.pipelineId) << 1);
  }
};
std::mutex& exchangeClientMapMutex() {
  static std::mutex m;
  return m;
}
auto& exchangeClientMap() {
  static std::unordered_map<
      TaskPipelineKey,
      std::weak_ptr<UcxCpuRowExchangeClient>,
      TaskPipelineKeyHash>
      m;
  return m;
}

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
    auto comm = Communicator::initAndGet(cfg.port, "");
    if (!comm) {
      LOG(ERROR) << "[CPU-UCX] Communicator::initAndGet failed";
      return;
    }
    std::thread([c = comm]() { c->run(); }).detach();
  });
}

bool adaptDriver(const exec::DriverFactory& factory, exec::Driver& driver) {
  if (!cpuUcxConfig().enabled) {
    return false;
  }
  // The env var only enables the adapter. Per-edge transport comes from the
  // plan, so coordinator-bound and other HTTP exchanges stay on the standard
  // path.
  ensureCommunicatorStarted();
  auto* ctx = driver.driverCtx();

  // planNodeId -> planNode lookup, mirroring the cudf adapter pattern.
  auto findPlanNode =
      [&factory](const core::PlanNodeId& id) -> core::PlanNodePtr {
    for (const auto& node : factory.planNodes) {
      if (node->id() == id) {
        return node;
      }
    }
    if (factory.consumerNode && factory.consumerNode->id() == id) {
      return factory.consumerNode;
    }
    return nullptr;
  };

  bool replacedAny = false;
  auto operators = driver.operators();
  // Iterate in reverse so replaceOperators() index math doesn't shift
  // the operators we haven't visited yet.
  for (int32_t i = static_cast<int32_t>(operators.size()) - 1; i >= 0; --i) {
    auto* op = operators[i];
    if (!op) {
      continue;
    }
    const auto& planNodeId = op->planNodeId();
    if (planNodeId.empty() || planNodeId == "N/A") {
      continue;
    }

    if (dynamic_cast<exec::PartitionedOutput*>(op) != nullptr) {
      auto planNode = findPlanNode(planNodeId);
      auto poNode =
          std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
              planNode);
      if (!poNode) {
        continue;
      }

      if (poNode->transportType() !=
          core::PartitionedOutputNode::TransportType::kUcx) {
        VLOG(1) << "[CPU-UCX] keeping standard PartitionedOutput at index " << i
                << " (planNodeId=" << planNodeId << ", taskId=" << op->taskId()
                << ", transport="
                << core::PartitionedOutputNode::toName(poNode->transportType())
                << ")";
        continue;
      }

      std::vector<std::unique_ptr<exec::Operator>> replacement;
      // The UCX operator ignores query-level eager flush because it
      // bundles on the exchange-server side.
      replacement.push_back(std::make_unique<UcxCpuRowPartitionedOutput>(
          op->operatorId(), ctx, poNode, /*eagerFlush=*/false));
      VLOG(1) << "[CPU-UCX] replacing PartitionedOutput at index " << i
              << " (planNodeId=" << planNodeId << ")";
      [[maybe_unused]] auto replaced =
          factory.replaceOperators(driver, i, i + 1, std::move(replacement));
      replacedAny = true;
      continue;
    }

    if (dynamic_cast<exec::Exchange*>(op) != nullptr) {
      auto planNode = findPlanNode(planNodeId);
      auto exchangeNode =
          std::dynamic_pointer_cast<const core::ExchangeNode>(planNode);
      if (!exchangeNode) {
        continue;
      }

      if (exchangeNode->transportType() !=
          core::ExchangeNode::TransportType::kUcx) {
        VLOG(1) << "[CPU-UCX] keeping standard Exchange at index " << i
                << " (planNodeId=" << planNodeId << ", taskId=" << op->taskId()
                << ", transport="
                << core::ExchangeNode::toName(exchangeNode->transportType())
                << ")";
        continue;
      }
      // Share one UcxCpuRowExchangeClient across all drivers in the
      // same (taskId, pipelineId). Driver 0 is the only one that
      // processes splits; without sharing, drivers 1..N would have
      // their own empty queues and block forever.
      const TaskPipelineKey key{op->taskId(), ctx->pipelineId};
      std::shared_ptr<UcxCpuRowExchangeClient> client;
      {
        std::lock_guard<std::mutex> lock(exchangeClientMapMutex());
        auto& clientMap = exchangeClientMap();
        for (auto it = clientMap.begin(); it != clientMap.end();) {
          if (it->second.expired()) {
            it = clientMap.erase(it);
          } else {
            ++it;
          }
        }
        auto it = clientMap.find(key);
        if (it != clientMap.end()) {
          client = it->second.lock();
        }
        if (!client) {
          client = std::make_shared<UcxCpuRowExchangeClient>(
              op->taskId(), ctx->task->destination(), factory.numDrivers);
          clientMap[key] = client;
        }
      }
      std::vector<std::unique_ptr<exec::Operator>> replacement;
      replacement.push_back(std::make_unique<UcxCpuRowExchange>(
          op->operatorId(), ctx, exchangeNode, client));
      VLOG(1) << "[CPU-UCX] replacing Exchange at index " << i
              << " (planNodeId=" << planNodeId << ")";
      [[maybe_unused]] auto replaced =
          factory.replaceOperators(driver, i, i + 1, std::move(replacement));
      replacedAny = true;
      continue;
    }
  }

  return replacedAny;
}

std::atomic<bool> registered{false};

} // namespace

void registerCpuUcxDriverAdapter() {
  bool expected = false;
  if (!registered.compare_exchange_strong(expected, true)) {
    return;
  }
  exec::DriverAdapter adapter{
      std::string(kAdapterLabel),
      /*inspect=*/{},
      &adaptDriver};
  exec::DriverFactory::registerAdapter(std::move(adapter));
  exec::MergeSource::registerMergeExchangeSourceFactory(
      [](exec::MergeExchange* mergeExchange,
         const std::string& taskId,
         int destination) -> std::shared_ptr<exec::MergeSource> {
        if (!cpuUcxConfig().enabled) {
          return nullptr;
        }
        ensureCommunicatorStarted();
        return std::make_shared<UcxCpuRowMergeExchangeSource>(
            mergeExchange, taskId, destination);
      });
  const auto& cfg = cpuUcxConfig();
  LOG(INFO) << "[CPU-UCX] DriverAdapter registered (enabled=" << cfg.enabled
            << ", port=" << cfg.port << ")";
}

// __attribute__((constructor)) puts this function pointer in
// .init_array, which the C runtime invokes before main(). Survives
// linker GC and LTO because the runtime startup code holds an explicit
// reference to .init_array. An anonymous-namespace `const AutoRegister`
// would be vulnerable to --gc-sections dropping the whole section.
__attribute__((constructor)) static void cpuUcxAutoRegister() {
  registerCpuUcxDriverAdapter();
}

} // namespace facebook::velox::ucx_exchange
