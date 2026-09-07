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

#include "velox/exec/PartitionedOutput.h"
#include "velox/experimental/ucx-exchange/DynamicUcxTransport.h"
#include "velox/experimental/ucx-exchange/ExchangeFormatRegistry.h"

#include <glog/logging.h>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "velox/core/PlanNode.h"
#include "velox/exec/DefaultOutputBufferManager.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/Merge.h"
#include "velox/exec/OutputTransportRegistry.h"
#include "velox/exec/Task.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/UcxExchange.h"
#include "velox/experimental/ucx-exchange/UcxExchangeClient.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#include "velox/experimental/ucx-exchange/UcxPartitionedOutput.h"

namespace facebook::velox::ucx_exchange {

namespace {

constexpr const char* kAdapterLabel = "CudfUcx";

std::once_flag communicatorStartedFlag;

struct TaskPipelineKey {
  std::string taskId;
  int pipelineId;

  bool operator==(const TaskPipelineKey& other) const {
    return taskId == other.taskId && pipelineId == other.pipelineId;
  }
};

struct TaskPipelineKeyHash {
  size_t operator()(const TaskPipelineKey& key) const {
    return std::hash<std::string>{}(key.taskId) ^
        (static_cast<size_t>(key.pipelineId) << 1);
  }
};

std::mutex& exchangeClientMapMutex() {
  static std::mutex mutex;
  return mutex;
}

auto& exchangeClientMap() {
  static std::unordered_map<
      TaskPipelineKey,
      std::weak_ptr<UcxExchangeClient>,
      TaskPipelineKeyHash>
      map;
  return map;
}

std::atomic<bool> communicatorStarted{false};
std::once_flag communicatorNotStartedLoggedFlag;

bool canUseCudfUcxExchange() {
  if (communicatorStarted.load()) {
    return true;
  }

  std::call_once(communicatorNotStartedLoggedFlag, []() {
    LOG(WARNING)
        << "[CUDF-UCX] cudf.exchange is enabled, but the startup communicator "
           "is not running; keeping standard exchange operators";
  });
  return false;
}

// Whether this pipeline's output node names the UCX transport, which decides
// how its output operator was built.
bool namesUcxOutput(const exec::DriverFactory& factory) {
  for (const auto& node : factory.planNodes) {
    if (const auto* outputNode =
            dynamic_cast<const core::PartitionedOutputNode*>(node.get())) {
      if (outputNode->transportKind() == core::TransportKind::kUcx) {
        return true;
      }
    }
  }
  if (const auto* outputNode = dynamic_cast<const core::PartitionedOutputNode*>(
          factory.consumerNode.get())) {
    return outputNode->transportKind() == core::TransportKind::kUcx;
  }
  return false;
}

bool usesUcxTransport(const core::PlanNode& planNode) {
  if (const auto* exchangeNode =
          dynamic_cast<const core::ExchangeNode*>(&planNode)) {
    return exchangeNode->transportKind() == core::TransportKind::kUcx;
  }
  if (const auto* outputNode =
          dynamic_cast<const core::PartitionedOutputNode*>(&planNode)) {
    return outputNode->transportKind() == core::TransportKind::kUcx;
  }
  return false;
}

void registerCudfUcxOutputTransport() {
  auto manager = UcxOutputQueueManager::getInstanceRef();
  if (auto existing = exec::OutputTransportRegistry::tryGet(
          std::string{core::TransportKind::kUcx})) {
    VELOX_CHECK(
        std::dynamic_pointer_cast<UcxOutputQueueManager>(existing->manager) ==
            manager,
        "UCX output transport is already registered with an incompatible "
        "manager");
    return;
  }

  auto entry = exec::OutputTransportEntry::make<UcxOutputQueueManager>(
      manager,
      [](int32_t operatorId,
         exec::DriverCtx* ctx,
         const std::shared_ptr<const core::PartitionedOutputNode>& node,
         bool eagerFlush,
         const std::shared_ptr<UcxOutputQueueManager>& manager)
          -> std::unique_ptr<exec::Operator> {
        return std::make_unique<UcxPartitionedOutput>(
            operatorId, ctx, node, eagerFlush, manager);
      });
  exec::OutputTransportRegistry::global().insert(
      std::string{core::TransportKind::kUcx}, std::move(entry));
}

core::PlanNodePtr findPlanNode(
    const exec::DriverFactory& factory,
    const core::PlanNodeId& id) {
  for (const auto& node : factory.planNodes) {
    if (node->id() == id) {
      return node;
    }
  }
  if (factory.consumerNode && factory.consumerNode->id() == id) {
    return factory.consumerNode;
  }
  return nullptr;
}

std::shared_ptr<UcxExchangeClient> getOrCreateExchangeClient(
    exec::Exchange* exchangeOp,
    const exec::Operator* op,
    exec::DriverCtx* ctx) {
  const TaskPipelineKey key{op->taskId(), ctx->pipelineId};
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
    if (auto existing = it->second.lock()) {
      exchangeOp->resetExchangeClient();
      return existing;
    }
  }

  auto veloxExchangeClient = exchangeOp->releaseExchangeClient();
  VELOX_CHECK_NOT_NULL(
      veloxExchangeClient, "Velox exchange client can't be null.");
  auto client = std::make_shared<UcxExchangeClient>(
      op->taskId(),
      veloxExchangeClient->getDestination(),
      veloxExchangeClient->getNumberOfConsumers(),
      ctx->task->queryCtx()->queryConfig().maxOutputBufferSize());
  clientMap[key] = client;
  return client;
}

// Mirrors the rule in LocalPlanner: a partial limit upstream means flush
// early.

bool eagerFlush(const core::PlanNode& node) {
  if (const auto* limit = dynamic_cast<const core::LimitNode*>(&node)) {
    return limit->isPartial() && limit->offset() + limit->count() < 10'000;
  }
  if (node.sources().empty()) {
    return false;
  }
  return eagerFlush(*node.sources()[0]);
}

std::unique_ptr<exec::Operator> makeGpuOutputSink(
    int32_t operatorId,
    exec::DriverCtx* ctx,
    const core::PlanNodePtr& planNode,
    uint32_t numTotalDrivers) {
  if (!useDynamicUcx() || !canUseCudfUcxExchange()) {
    return nullptr;
  }
  auto outputNode =
      std::dynamic_pointer_cast<const core::PartitionedOutputNode>(planNode);
  if (outputNode == nullptr) {
    return nullptr;
  }

  const auto& taskId = ctx->task->taskId();
  ExchangeFormatRegistry::instance().trackTask(taskId);

  auto queueManager = UcxOutputQueueManager::getInstanceRef();
  if (ctx->driverId == 0 && queueManager->getQueueIfExists(taskId) == nullptr) {
    queueManager->initializeTask(
        ctx->task,
        outputNode->kind(),
        outputNode->numPartitions(),
        numTotalDrivers);
  }

  // always sink into UcxPartitionedOutput which can render Presto
  // to any downstream Exchange automatically.

  return std::make_unique<UcxPartitionedOutput>(
      operatorId,
      ctx,
      outputNode,
      eagerFlush(*outputNode),
      std::move(queueManager));
}

bool adaptDriver(const exec::DriverFactory& factory, exec::Driver& driver) {
  const auto& config = cudf_velox::CudfConfig::getInstance();
  if (!config.enabled || !config.exchange) {
    return false;
  }

  auto* ctx = driver.driverCtx();

  if (useDynamicUcx() &&
      !ctx->queryConfig().get<bool>(
          cudf_velox::CudfConfig::kCudfEnabled, config.enabled) &&
      !namesUcxOutput(factory)) {
    return false;
  }

  auto operators = driver.operators();

  for (int32_t i = static_cast<int32_t>(operators.size()) - 1; i >= 0; --i) {
    auto* op = operators[i];
    if (!op || op->planNodeId().empty() || op->planNodeId() == "N/A") {
      continue;
    }

    if (dynamic_cast<exec::MergeExchange*>(op) != nullptr) {
      auto planNode = findPlanNode(factory, op->planNodeId());
      auto mergeExchangeNode =
          std::dynamic_pointer_cast<const core::MergeExchangeNode>(planNode);
      if (!mergeExchangeNode) {
        continue;
      }
      if (!useDynamicUcx() && !usesUcxTransport(*mergeExchangeNode)) {
        continue;
      }

      if (!canUseCudfUcxExchange()) {
        continue;
      }
      std::vector<std::unique_ptr<exec::Operator>> replacement;
      replacement.push_back(
          std::make_unique<UcxExchange>(
              op->operatorId(), ctx, mergeExchangeNode, nullptr));
      replacement.push_back(
          std::make_unique<cudf_velox::CudfOrderBy>(
              op->operatorId(), ctx, mergeExchangeNode));
      [[maybe_unused]] auto replaced =
          factory.replaceOperators(driver, i, i + 1, std::move(replacement));
      VLOG(1) << "[CUDF-UCX] replacing MergeExchange at index " << i
              << " (planNodeId=" << op->planNodeId() << ")";
      continue;
    }

    if (useDynamicUcx() && op->operatorType() == "cudfPartitionedOutput") {
      // dynamicUcx shuffle format needs to track this task
      // this block is only here to support the case where the transportKind is
      // set to Ucx and could go away if we remove transportKind

      auto planNode = findPlanNode(factory, op->planNodeId());
      auto outputNode =
          std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
              planNode);

      ExchangeFormatRegistry::instance().trackTask(op->taskId());

      auto bufferManager = exec::DefaultOutputBufferManager::getInstanceRef();
      if (outputNode != nullptr && ctx->driverId == 0 &&
          bufferManager->getBufferIfExists(op->taskId()) == nullptr) {
        // updateOutputBuffers goes to the manager kUcx named, not to this
        // buffer, so a broadcast task would never finish.

        VELOX_USER_CHECK(
            outputNode->kind() != core::PartitionedOutputNode::Kind::kBroadcast,
            "A broadcast plan cannot name the '{}' transport while transport "
            "discovery is on; leave transportKind unset",
            core::TransportKind::kUcx);
        bufferManager->initializeTask(
            ctx->task,
            outputNode->kind(),
            outputNode->numPartitions(),
            factory.numTotalDrivers);
      }
      continue;
    }

    if (dynamic_cast<exec::Exchange*>(op) != nullptr) {
      auto* exchangeOp = dynamic_cast<exec::Exchange*>(op);
      auto planNode = findPlanNode(factory, op->planNodeId());
      auto exchangeNode =
          std::dynamic_pointer_cast<const core::ExchangeNode>(planNode);
      if (!exchangeNode) {
        continue;
      }
      // Chosen because this is a cuDF plan, not because the plan named a
      // transport. Whether the peer speaks UCX is settled when it connects.
      if (!useDynamicUcx() && !usesUcxTransport(*exchangeNode)) {
        continue;
      }

      if (!canUseCudfUcxExchange()) {
        continue;
      }
      std::vector<std::unique_ptr<exec::Operator>> replacement;
      replacement.push_back(
          std::make_unique<UcxExchange>(
              op->operatorId(),
              ctx,
              exchangeNode,
              getOrCreateExchangeClient(exchangeOp, op, ctx)));
      [[maybe_unused]] auto replaced =
          factory.replaceOperators(driver, i, i + 1, std::move(replacement));
      VLOG(1) << "[CUDF-UCX] replacing Exchange at index " << i
              << " (planNodeId=" << op->planNodeId() << ")";
      continue;
    }
  }

  return false;
}

std::atomic<bool> registered{false};

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
  registerCudfUcxOutputTransport();
  cudf_velox::registerGpuOutputSinkFactory(&makeGpuOutputSink);

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
  bool expected = false;
  if (!registered.compare_exchange_strong(expected, true)) {
    return;
  }

  // Front rather than appended: ToCudf places the host/device boundary from
  // the operators it sees, so it must see these already substituted.
  exec::DriverAdapter adapter{
      std::string(kAdapterLabel),
      /*inspect=*/{},
      &adaptDriver};
  exec::DriverFactory::adapters.insert(
      exec::DriverFactory::adapters.begin(), std::move(adapter));
  LOG(INFO) << "[CUDF-UCX] DriverAdapter registered";
}

} // namespace facebook::velox::ucx_exchange
