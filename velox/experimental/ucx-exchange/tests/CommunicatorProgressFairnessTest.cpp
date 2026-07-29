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

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <folly/ScopeGuard.h>

#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/EndpointRef.h"

namespace facebook::velox::ucx_exchange::test {
namespace {

using namespace std::chrono_literals;

template <typename Predicate>
bool waitUntil(std::chrono::milliseconds timeout, Predicate&& predicate) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  return predicate();
}

size_t readPositiveEnvironmentSize(const char* name, size_t defaultValue) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return defaultValue;
  }

  try {
    const auto parsed = std::stoull(value);
    return parsed == 0 ? defaultValue : static_cast<size_t>(parsed);
  } catch (const std::exception&) {
    return defaultValue;
  }
}

TEST(
    CommunicatorProgressFairnessTest,
    lateEndpointsCompleteDuringEstablishedBidirectionalTraffic) {
  const size_t batchSize =
      readPositiveEnvironmentSize("VELOX_UCX_TEST_FAIRNESS_BATCH_SIZE", 512);
  const size_t endpointCount =
      readPositiveEnvironmentSize("VELOX_UCX_TEST_FAIRNESS_ENDPOINT_COUNT", 32);
  const size_t maxEndpointLatencyMs = readPositiveEnvironmentSize(
      "VELOX_UCX_TEST_FAIRNESS_MAX_ENDPOINT_LATENCY_MS", 2000);

  ContinueFuture ready;
  auto communicator = Communicator::initAndGet(0, "", &ready);
  ASSERT_NE(communicator, nullptr);

  std::shared_ptr<ucxx::Worker> peerWorker;
  auto peerProgressGuard = folly::makeGuard([&]() {
    if (peerWorker && peerWorker->isProgressThreadRunning()) {
      peerWorker->stopProgressThread();
    }
  });

  std::thread communicatorThread([&]() { communicator->run(); });
  auto communicatorGuard = folly::makeGuard([&]() {
    communicator->stop();
    if (communicatorThread.joinable()) {
      communicatorThread.join();
    }
  });

  // This is the production readiness signal: it is fulfilled only after the
  // listener, worker address, and UCXX progress thread have been initialized.
  ready.wait();

  auto peerContext = ucxx::contextBuilder(UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP)
                         .configMap({})
                         .build();
  ASSERT_NE(peerContext, nullptr);
  peerWorker = peerContext->workerBuilder()
                   .delayedSubmission(true)
                   .pythonFuture(false)
                   .build();
  ASSERT_NE(peerWorker, nullptr);
  auto peerAddress = peerWorker->addressBuilder().build();
  ASSERT_NE(peerAddress, nullptr);
  const auto peerAddressView = peerAddress->getStringView();
  ASSERT_FALSE(peerAddressView.empty());
  const std::string peerAddressString(
      peerAddressView.data(), peerAddressView.size());
  peerWorker->startProgressThread(true, 1);

  // Establish the path first. Late endpoint creation below deliberately
  // bypasses Communicator's address cache so each call schedules a real
  // ucp_ep_create generic-pre callback on the production worker.
  auto establishedEndpointRef =
      communicator->createSameHostEndpointRefFromWorkerAddress(
          peerAddressString, "127.0.0.1", communicator->getHostIdHash());
  ASSERT_NE(establishedEndpointRef, nullptr);
  ASSERT_NE(establishedEndpointRef->endpoint_, nullptr);
  auto communicatorWorker = establishedEndpointRef->endpoint_->getWorker();
  ASSERT_NE(communicatorWorker, nullptr);

  auto communicatorAddress =
      ucxx::AddressBuilder(communicator->getWorkerAddress()).build();
  ASSERT_NE(communicatorAddress, nullptr);
  auto peerEndpoint = ucxx::endpointBuilder(peerWorker, communicatorAddress)
                          .endpointErrorHandling(false)
                          .build();
  ASSERT_NE(peerEndpoint, nullptr);

  std::atomic<bool> stopProducer{false};
  std::atomic<bool> trafficWindowPublished{false};
  std::atomic<size_t> transfersCompleted{0};
  std::mutex failureMutex;
  std::string producerFailure;

  auto setProducerFailure = [&](std::string message) {
    std::lock_guard<std::mutex> lock(failureMutex);
    if (producerFailure.empty()) {
      producerFailure = std::move(message);
    }
  };

  std::thread producer([&]() {
    struct TrafficSlot {
      uint64_t toPeerSend{0};
      uint64_t toPeerReceive{0};
      uint64_t toCommunicatorSend{0};
      uint64_t toCommunicatorReceive{0};
      std::shared_ptr<ucxx::Request> toPeerReceiveRequest;
      std::shared_ptr<ucxx::Request> toPeerSendRequest;
      std::shared_ptr<ucxx::Request> toCommunicatorReceiveRequest;
      std::shared_ptr<ucxx::Request> toCommunicatorSendRequest;
    };

    uint64_t nextSequence{1};
    try {
      auto submit = [&](TrafficSlot& slot) {
        const auto sequence = nextSequence++;
        slot.toPeerSend = sequence;
        slot.toPeerReceive = 0;
        slot.toCommunicatorSend = ~sequence;
        slot.toCommunicatorReceive = 0;
        const ucxx::Tag toPeerTag{0x1000'0000'0000'0000ULL + sequence};
        const ucxx::Tag toCommunicatorTag{0x2000'0000'0000'0000ULL + sequence};

        slot.toPeerReceiveRequest = peerWorker
                                        ->tagRecvBuilder(
                                            &slot.toPeerReceive,
                                            sizeof(slot.toPeerReceive),
                                            toPeerTag,
                                            ucxx::TagMaskFull)
                                        .pythonFuture(false)
                                        .callbackFunction(nullptr)
                                        .callbackData(nullptr)
                                        .build();
        slot.toPeerSendRequest =
            establishedEndpointRef->endpoint_
                ->tagSendBuilder(
                    &slot.toPeerSend, sizeof(slot.toPeerSend), toPeerTag)
                .pythonFuture(false)
                .callbackFunction(nullptr)
                .callbackData(nullptr)
                .build();
        slot.toCommunicatorReceiveRequest =
            communicatorWorker
                ->tagRecvBuilder(
                    &slot.toCommunicatorReceive,
                    sizeof(slot.toCommunicatorReceive),
                    toCommunicatorTag,
                    ucxx::TagMaskFull)
                .pythonFuture(false)
                .callbackFunction(nullptr)
                .callbackData(nullptr)
                .build();
        slot.toCommunicatorSendRequest =
            peerEndpoint
                ->tagSendBuilder(
                    &slot.toCommunicatorSend,
                    sizeof(slot.toCommunicatorSend),
                    toCommunicatorTag)
                .pythonFuture(false)
                .callbackFunction(nullptr)
                .callbackData(nullptr)
                .build();
      };

      auto slotCompleted = [](const TrafficSlot& slot) {
        return slot.toPeerReceiveRequest->isCompleted() &&
            slot.toPeerSendRequest->isCompleted() &&
            slot.toCommunicatorReceiveRequest->isCompleted() &&
            slot.toCommunicatorSendRequest->isCompleted();
      };

      auto checkSlot = [](const TrafficSlot& slot) {
        slot.toPeerReceiveRequest->checkError();
        slot.toPeerSendRequest->checkError();
        slot.toCommunicatorReceiveRequest->checkError();
        slot.toCommunicatorSendRequest->checkError();
        if (slot.toPeerReceive != slot.toPeerSend ||
            slot.toCommunicatorReceive != slot.toCommunicatorSend) {
          throw std::runtime_error("background tag data did not match");
        }
      };

      std::vector<TrafficSlot> slots(batchSize);
      for (auto& slot : slots) {
        submit(slot);
      }
      trafficWindowPublished.store(true, std::memory_order_release);

      while (!stopProducer.load(std::memory_order_acquire)) {
        bool progressedSlot{false};
        for (auto& slot : slots) {
          if (!slotCompleted(slot)) {
            continue;
          }
          checkSlot(slot);
          transfersCompleted.fetch_add(2, std::memory_order_release);
          submit(slot);
          progressedSlot = true;
        }
        if (!progressedSlot) {
          std::this_thread::yield();
        }
      }

      if (!waitUntil(20s, [&]() {
            return std::all_of(slots.begin(), slots.end(), slotCompleted);
          })) {
        throw std::runtime_error(
            "timed out draining the bidirectional traffic window");
      }
      for (const auto& slot : slots) {
        checkSlot(slot);
      }
      transfersCompleted.fetch_add(slots.size() * 2, std::memory_order_release);
    } catch (const std::exception& error) {
      setProducerFailure(error.what());
    } catch (...) {
      setProducerFailure("unknown exception in background traffic producer");
    }
  });

  const bool trafficStarted = waitUntil(20s, [&]() {
    return trafficWindowPublished.load(std::memory_order_acquire) &&
        transfersCompleted.load(std::memory_order_acquire) > 0;
  });

  std::vector<std::shared_ptr<ucxx::Endpoint>> lateEndpoints;
  lateEndpoints.reserve(endpointCount);
  std::chrono::steady_clock::duration maxEndpointLatency{};
  std::string endpointFailure;

  if (trafficStarted) {
    try {
      for (size_t i = 0; i < endpointCount; ++i) {
        const auto start = std::chrono::steady_clock::now();
        auto endpoint = ucxx::endpointBuilder(communicatorWorker, peerAddress)
                            .endpointErrorHandling(false)
                            .build();
        maxEndpointLatency = std::max(
            maxEndpointLatency, std::chrono::steady_clock::now() - start);
        if (endpoint == nullptr) {
          endpointFailure = "late endpoint creation returned null";
          break;
        }
        lateEndpoints.push_back(std::move(endpoint));
      }
    } catch (const std::exception& error) {
      endpointFailure = error.what();
    }
  }

  stopProducer.store(true, std::memory_order_release);
  producer.join();

  std::string validationFailure;
  const size_t createdEndpointCount = lateEndpoints.size();
  std::vector<std::shared_ptr<std::atomic<size_t>>> completionCounts;
  if (endpointFailure.empty() && createdEndpointCount == endpointCount) {
    try {
      std::vector<uint64_t> send(endpointCount);
      std::vector<uint64_t> receive(endpointCount, 0);
      std::vector<std::shared_ptr<ucxx::Request>> requests;
      requests.reserve(endpointCount * 2);
      completionCounts.reserve(endpointCount * 2);

      auto countCompletion = [](ucs_status_t, std::shared_ptr<void> data) {
        std::static_pointer_cast<std::atomic<size_t>>(data)->fetch_add(
            1, std::memory_order_relaxed);
      };

      for (size_t i = 0; i < endpointCount; ++i) {
        send[i] = 0xf000'0000'0000'0000ULL + i;
        const ucxx::Tag tag{send[i]};
        auto receiveCompletions = std::make_shared<std::atomic<size_t>>(0);
        auto sendCompletions = std::make_shared<std::atomic<size_t>>(0);
        completionCounts.push_back(receiveCompletions);
        completionCounts.push_back(sendCompletions);
        requests.push_back(
            peerWorker
                ->tagRecvBuilder(
                    &receive[i], sizeof(receive[i]), tag, ucxx::TagMaskFull)
                .pythonFuture(false)
                .callbackFunction(countCompletion)
                .callbackData(receiveCompletions)
                .build());
        requests.push_back(
            lateEndpoints[i]
                ->tagSendBuilder(&send[i], sizeof(send[i]), tag)
                .pythonFuture(false)
                .callbackFunction(countCompletion)
                .callbackData(sendCompletions)
                .build());
      }

      if (!waitUntil(10s, [&]() {
            return std::all_of(
                requests.begin(), requests.end(), [](const auto& request) {
                  return request->isCompleted();
                });
          })) {
        validationFailure = "timed out validating late endpoints";
      } else {
        for (const auto& request : requests) {
          request->checkError();
        }
        if (receive != send) {
          validationFailure = "late-endpoint tag data did not match";
        }
        if (std::any_of(
                completionCounts.begin(),
                completionCounts.end(),
                [](const auto& count) {
                  return count->load(std::memory_order_relaxed) != 1;
                })) {
          validationFailure =
              "a late-endpoint callback did not run exactly once";
        }
      }
    } catch (const std::exception& error) {
      validationFailure = error.what();
    }
  }

  lateEndpoints.clear();
  peerEndpoint.reset();

  communicator->stop();
  communicatorThread.join();
  communicatorGuard.dismiss();

  peerWorker->stopProgressThread();
  peerProgressGuard.dismiss();

  std::string producerFailureCopy;
  {
    std::lock_guard<std::mutex> lock(failureMutex);
    producerFailureCopy = producerFailure;
  }

  EXPECT_TRUE(trafficStarted);
  EXPECT_TRUE(endpointFailure.empty()) << endpointFailure;
  EXPECT_EQ(createdEndpointCount, endpointCount);
  EXPECT_TRUE(producerFailureCopy.empty()) << producerFailureCopy;
  EXPECT_TRUE(validationFailure.empty()) << validationFailure;
  EXPECT_GT(transfersCompleted.load(std::memory_order_acquire), 0);

  const auto maxEndpointLatencyMilliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(maxEndpointLatency)
          .count();
  RecordProperty(
      "background_transfers_completed",
      std::to_string(transfersCompleted.load(std::memory_order_acquire)));
  RecordProperty(
      "late_endpoints_created", std::to_string(createdEndpointCount));
  RecordProperty(
      "max_endpoint_latency_ms",
      std::to_string(maxEndpointLatencyMilliseconds));
  if (createdEndpointCount > 0) {
    EXPECT_LT(
        maxEndpointLatencyMilliseconds,
        static_cast<int64_t>(maxEndpointLatencyMs));
  }
}

} // namespace
} // namespace facebook::velox::ucx_exchange::test
