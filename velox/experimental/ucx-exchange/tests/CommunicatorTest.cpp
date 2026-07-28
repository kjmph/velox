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

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <folly/ScopeGuard.h>

#include "velox/experimental/ucx-exchange/Communicator.h"

namespace facebook::velox::ucx_exchange::test {
namespace {

class AsyncShutdownElement final
    : public CommElement,
      public std::enable_shared_from_this<AsyncShutdownElement> {
 public:
  explicit AsyncShutdownElement(
      const std::shared_ptr<Communicator>& communicator)
      : CommElement(communicator, true) {}

  ~AsyncShutdownElement() override {
    join();
  }

  bool supportsCommunicatorShutdownDrain() const override {
    return true;
  }

  void close() override {
    bool expected = false;
    if (!closeStarted_.compare_exchange_strong(expected, true)) {
      return;
    }
    std::weak_ptr<AsyncShutdownElement> weak = weak_from_this();
    completionThread_ = std::thread([weak]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      if (auto self = weak.lock()) {
        self->completionReady_.store(true, std::memory_order_release);
        self->communicator_->addToWorkQueue(self);
      }
    });
  }

  void forceCloseForShutdown() override {
    forced_.store(true, std::memory_order_release);
    completionReady_.store(true, std::memory_order_release);
  }

  void process() override {
    if (!closeStarted_.load(std::memory_order_acquire) ||
        !completionReady_.load(std::memory_order_acquire)) {
      return;
    }
    processedAfterClose_.store(true, std::memory_order_release);
    communicator_->unregister(shared_from_this());
  }

  void join() {
    if (completionThread_.joinable()) {
      completionThread_.join();
    }
  }

  bool processedAfterClose() const {
    return processedAfterClose_.load(std::memory_order_acquire);
  }

  bool forced() const {
    return forced_.load(std::memory_order_acquire);
  }

 private:
  std::atomic<bool> closeStarted_{false};
  std::atomic<bool> completionReady_{false};
  std::atomic<bool> processedAfterClose_{false};
  std::atomic<bool> forced_{false};
  std::thread completionThread_;
};

TEST(
    CommunicatorTest,
    delayedCancellationPreservesSharedEndpointAndDrainsOnShutdown) {
  ContinueFuture ready;
  auto communicator = Communicator::initAndGet(0, "", &ready);
  ASSERT_NE(communicator, nullptr);

  std::shared_ptr<ucxx::Context> peerContext;
  std::shared_ptr<ucxx::Worker> peerWorker;
  auto peerProgressGuard = folly::makeGuard([&]() {
    if (peerWorker && peerWorker->isProgressThreadRunning()) {
      peerWorker->stopProgressThread();
    }
  });

  std::thread communicatorThread([&]() { communicator->run(); });
  auto stopGuard = folly::makeGuard([&]() {
    communicator->stop();
    if (communicatorThread.joinable()) {
      communicatorThread.join();
    }
  });
  ready.wait();

  // A loopback endpoint to the communicator's own worker selects UCX's self
  // transport and does not exercise production endpoint sharing. Use a
  // separate context and worker so the follow-on transfer negotiates a real
  // same-host transport.
  peerContext = ucxx::createContext({}, UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP);
  ASSERT_NE(peerContext, nullptr);
  peerWorker = peerContext->createWorker(true);
  ASSERT_NE(peerWorker, nullptr);
  auto peerAddress = peerWorker->getAddress();
  ASSERT_NE(peerAddress, nullptr);
  const auto peerAddressView = peerAddress->getStringView();
  ASSERT_FALSE(peerAddressView.empty());
  const std::string peerAddressString(
      peerAddressView.data(), peerAddressView.size());
  peerWorker->startProgressThread(true, 1);

  auto endpointRef = communicator->createSameHostEndpointRefFromWorkerAddress(
      peerAddressString, "127.0.0.1", communicator->getHostIdHash());
  ASSERT_NE(endpointRef, nullptr);
  ASSERT_NE(endpointRef->endpoint_, nullptr);

  auto worker = endpointRef->endpoint_->getWorker();
  ASSERT_NE(worker, nullptr);

  auto communicatorAddress =
      ucxx::createAddressFromString(communicator->getWorkerAddress());
  ASSERT_NE(communicatorAddress, nullptr);
  auto peerEndpoint = ucxx::createEndpointFromWorkerAddress(
      peerWorker, communicatorAddress, false);
  ASSERT_NE(peerEndpoint, nullptr);

  auto waitUntil = [](auto predicate) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!predicate() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return predicate();
  };

  // Shutdown must keep dispatching callbacks for opted-in elements until they
  // unregister. A one-shot close/process pass would drop this delayed wake-up
  // after publishing shutdownDraining_ and require the forced fallback.
  auto asyncShutdownElement =
      std::make_shared<AsyncShutdownElement>(communicator);
  ASSERT_TRUE(communicator->registerCommElement(asyncShutdownElement));

  // Hold the progress thread inside the request-submission snapshot. Requests
  // created below miss that snapshot. The cancellation sentinel must remain
  // behind their publication callback in the same delayed-submission FIFO.
  std::mutex progressMutex;
  std::condition_variable progressCv;
  bool progressBlocked = false;
  bool releaseProgress = false;
  worker->registerDelayedSubmission(nullptr, [&]() {
    std::unique_lock<std::mutex> lock(progressMutex);
    progressBlocked = true;
    progressCv.notify_all();
    progressCv.wait(lock, [&]() { return releaseProgress; });
  });
  auto releaseProgressThread = [&]() {
    {
      std::lock_guard<std::mutex> lock(progressMutex);
      releaseProgress = true;
    }
    progressCv.notify_all();
  };
  auto progressBlockerGuard = folly::makeGuard(releaseProgressThread);
  bool reachedProgressBlock = false;
  {
    std::unique_lock<std::mutex> lock(progressMutex);
    reachedProgressBlock = progressCv.wait_for(
        lock, std::chrono::seconds(5), [&]() { return progressBlocked; });
  }
  if (!reachedProgressBlock) {
    releaseProgressThread();
  }
  ASSERT_TRUE(reachedProgressBlock);

  struct RequestContext {
    std::shared_ptr<std::vector<uint8_t>> payload;
    std::shared_ptr<std::atomic<int>> completionStatus;
  };

  auto completionStatus = std::make_shared<std::atomic<int>>(UCS_INPROGRESS);
  auto payload = std::make_shared<std::vector<uint8_t>>(16 << 20, 0x5a);
  std::weak_ptr<std::vector<uint8_t>> weakPayload = payload;
  auto requestContext = std::make_shared<RequestContext>(
      RequestContext{payload, completionStatus});
  constexpr uint64_t kCancelledTag = 0x7fff'0000'0000'0001ULL;
  auto cancelledRequest = endpointRef->endpoint_->tagRecv(
      payload->data(),
      payload->size(),
      ucxx::Tag{kCancelledTag},
      ucxx::TagMaskFull,
      false,
      [](ucs_status_t status, std::shared_ptr<void> arg) {
        auto context = std::static_pointer_cast<RequestContext>(arg);
        context->completionStatus->store(status, std::memory_order_release);
        context->payload.reset();
      },
      requestContext);
  payload.reset();
  requestContext.reset();

  // UCX only supports request cancellation for tag receives. Leave another
  // exact receive on the same shared endpoint while the target is canceled.
  // It must remain posted and complete normally afterward.
  constexpr uint64_t kFollowOnTag = 0x7fff'0000'0000'0002ULL;
  std::array<uint8_t, 4> sendBytes{1, 2, 3, 4};
  std::array<uint8_t, 4> receiveBytes{};
  auto receiveRequest = endpointRef->endpoint_->tagRecv(
      receiveBytes.data(),
      receiveBytes.size(),
      ucxx::Tag{kFollowOnTag},
      ucxx::TagMaskFull);

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(cancelledRequest);
  communicator->deferTagReceiveCancellation(std::move(requests));

  releaseProgressThread();

  ASSERT_TRUE(waitUntil([&]() { return cancelledRequest->isCompleted(); }));
  EXPECT_EQ(
      completionStatus->load(std::memory_order_acquire), UCS_ERR_CANCELED);
  EXPECT_TRUE(weakPayload.expired());
  EXPECT_FALSE(receiveRequest->isCompleted());

  // Cancellation is request-scoped. A later transfer on the same shared
  // endpoint must still complete and carry the expected bytes.
  auto sendRequest = peerEndpoint->tagSend(
      sendBytes.data(), sendBytes.size(), ucxx::Tag{kFollowOnTag});

  ASSERT_TRUE(waitUntil([&]() {
    return receiveRequest->isCompleted() && sendRequest->isCompleted();
  }));
  EXPECT_EQ(receiveRequest->getStatus(), UCS_OK);
  EXPECT_EQ(sendRequest->getStatus(), UCS_OK);
  EXPECT_EQ(receiveBytes, sendBytes);

  // Recreate the delayed-submission race while stop() moves run() into its
  // shutdown drain. The target request and cancellation sentinel are queued
  // while the progress thread is blocked inside the previous request
  // snapshot. Releasing it only after shutdown begins proves that shutdown
  // crosses the delayed-submission boundary before stopping UCX progress.
  std::mutex shutdownProgressMutex;
  std::condition_variable shutdownProgressCv;
  bool shutdownProgressBlocked = false;
  bool releaseShutdownProgress = false;
  worker->registerDelayedSubmission(nullptr, [&]() {
    std::unique_lock<std::mutex> lock(shutdownProgressMutex);
    shutdownProgressBlocked = true;
    shutdownProgressCv.notify_all();
    shutdownProgressCv.wait(lock, [&]() { return releaseShutdownProgress; });
  });
  auto releaseShutdownProgressThread = [&]() {
    {
      std::lock_guard<std::mutex> lock(shutdownProgressMutex);
      releaseShutdownProgress = true;
    }
    shutdownProgressCv.notify_all();
  };
  auto shutdownProgressBlockerGuard =
      folly::makeGuard(releaseShutdownProgressThread);
  bool reachedShutdownProgressBlock = false;
  {
    std::unique_lock<std::mutex> lock(shutdownProgressMutex);
    reachedShutdownProgressBlock =
        shutdownProgressCv.wait_for(lock, std::chrono::seconds(5), [&]() {
          return shutdownProgressBlocked;
        });
  }
  if (!reachedShutdownProgressBlock) {
    releaseShutdownProgressThread();
  }
  ASSERT_TRUE(reachedShutdownProgressBlock);

  auto shutdownCompletionStatus =
      std::make_shared<std::atomic<int>>(UCS_INPROGRESS);
  auto shutdownPayload = std::make_shared<std::vector<uint8_t>>(16 << 20, 0xa5);
  std::weak_ptr<std::vector<uint8_t>> weakShutdownPayload = shutdownPayload;
  auto shutdownRequestContext = std::make_shared<RequestContext>(
      RequestContext{shutdownPayload, shutdownCompletionStatus});
  constexpr uint64_t kShutdownCancelledTag = 0x7fff'0000'0000'0003ULL;
  auto shutdownCancelledRequest = endpointRef->endpoint_->tagRecv(
      shutdownPayload->data(),
      shutdownPayload->size(),
      ucxx::Tag{kShutdownCancelledTag},
      ucxx::TagMaskFull,
      false,
      [](ucs_status_t status, std::shared_ptr<void> arg) {
        auto context = std::static_pointer_cast<RequestContext>(arg);
        context->completionStatus->store(status, std::memory_order_release);
        context->payload.reset();
      },
      shutdownRequestContext);
  shutdownPayload.reset();
  shutdownRequestContext.reset();

  std::vector<std::shared_ptr<ucxx::Request>> shutdownRequests;
  shutdownRequests.push_back(shutdownCancelledRequest);
  communicator->deferTagReceiveCancellation(std::move(shutdownRequests));

  communicator->stop();
  ASSERT_TRUE(waitUntil([&]() { return communicator->isShutdownDraining(); }));
  releaseShutdownProgressThread();

  ASSERT_TRUE(
      waitUntil([&]() { return shutdownCancelledRequest->isCompleted(); }));
  EXPECT_EQ(
      shutdownCompletionStatus->load(std::memory_order_acquire),
      UCS_ERR_CANCELED);
  EXPECT_TRUE(weakShutdownPayload.expired());

  communicatorThread.join();
  stopGuard.dismiss();
  asyncShutdownElement->join();
  EXPECT_TRUE(asyncShutdownElement->processedAfterClose());
  EXPECT_FALSE(asyncShutdownElement->forced());

  // Handshakes only need the immutable address created with the worker.
  // The first read after progress stops proves startup cached it eagerly and
  // this accessor does not enqueue work on the progress thread.
  const auto workerAddress = communicator->getWorkerAddress();
  EXPECT_FALSE(workerAddress.empty());
  EXPECT_EQ(communicator->getWorkerAddress(), workerAddress);

  peerEndpoint.reset();
  peerWorker->stopProgressThread();
  peerProgressGuard.dismiss();
}

} // namespace
} // namespace facebook::velox::ucx_exchange::test
