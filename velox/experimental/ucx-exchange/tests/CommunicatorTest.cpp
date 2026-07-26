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

#include <thread>

#include "velox/experimental/ucx-exchange/Communicator.h"

namespace facebook::velox::ucx_exchange::test {
namespace {

TEST(CommunicatorTest, workerAddressDoesNotDependOnProgressAfterStartup) {
  ContinueFuture ready;
  auto communicator = Communicator::initAndGet(0, "", &ready);
  ASSERT_NE(communicator, nullptr);

  std::thread communicatorThread([&]() { communicator->run(); });
  ready.wait();

  communicator->stop();
  communicatorThread.join();

  // Handshakes only need the immutable address created with the worker.
  // The first read after progress stops proves startup cached it eagerly and
  // this accessor does not enqueue work on the progress thread.
  const auto workerAddress = communicator->getWorkerAddress();
  EXPECT_FALSE(workerAddress.empty());
  EXPECT_EQ(communicator->getWorkerAddress(), workerAddress);
}

} // namespace
} // namespace facebook::velox::ucx_exchange::test
