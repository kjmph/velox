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

#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

// The CommElement is the abstract base class of both the
// per-client context on the exchange server side as well as the
// exchange source side.
namespace facebook::velox::ucx_exchange {

class Communicator;
class EndpointRef;

class CommElement {
 public:
  CommElement(
      const std::shared_ptr<Communicator> communicator,
      std::shared_ptr<EndpointRef> endpointRef,
      bool coalesceWorkQueueEntries = false)
      : communicator_{communicator},
        endpointRef_{endpointRef},
        coalesceWorkQueueEntries_{coalesceWorkQueueEntries} {}

  CommElement(
      const std::shared_ptr<Communicator> communicator,
      bool coalesceWorkQueueEntries = false)
      : communicator_{communicator},
        endpointRef_{nullptr},
        coalesceWorkQueueEntries_{coalesceWorkQueueEntries} {}

  virtual ~CommElement() = default;

  /// @brief Advance the communication by executing the communication elements
  /// specific communication pattern.
  virtual void process() = 0;

  // Called when the underlying endpoint was closed
  // or the communicator is finished.
  virtual void close() = 0;

  /// Opts this element into Communicator-driven graceful shutdown. Legacy
  /// exchange elements retain their existing teardown behavior until their
  /// request types can be classified safely.
  virtual bool supportsCommunicatorShutdownDrain() const {
    return false;
  }

  /// Starts protocol-aware shutdown while UCXX progress and callback dispatch
  /// remain available.
  virtual void beginCommunicatorShutdownDrain() {
    close();
  }

  /// Bounded shutdown fallback after graceful close callbacks failed to drain.
  /// The Communicator still calls process() once after this hook. Most
  /// elements already have a synchronous close and need no override.
  virtual void forceCloseForShutdown() {
    close();
  }

  /// Per-element process exclusion. The Communicator's primary dispatch
  /// loop calls process() under this mutex; if close() or endpoint cleanup
  /// already owns it, dispatch pushes the element back to the queue and
  /// moves on. Source/server state machines assume this single-owner
  /// process path.
  ///
  /// Recursive because close() can be reached from process() and from
  /// endpoint cleanup paths; both use the same per-element exclusion.
  std::recursive_mutex processMutex_;

  bool markQueued() {
    if (!coalesceWorkQueueEntries_) {
      return true;
    }
    bool expected = false;
    return queued_.compare_exchange_strong(
        expected, true, std::memory_order_acq_rel);
  }

  void clearQueued() {
    if (coalesceWorkQueueEntries_) {
      queued_.store(false, std::memory_order_release);
    }
  }

 protected:
  using StateEvent = std::function<void()>;

  /// Hands a completion from an arbitrary UCXX callback thread to process().
  /// Returns false if the handoff could not be queued. This entry point is
  /// noexcept because UCXX invokes it from C callbacks, where allowing a C++
  /// exception to escape is undefined behavior.
  template <typename Event>
  bool enqueueStateEvent(
      std::shared_ptr<CommElement> self,
      Event&& event) noexcept {
    try {
      return enqueueStateEventImpl(
          std::move(self), StateEvent(std::forward<Event>(event)));
    } catch (...) {
      return false;
    }
  }

  /// Runs a snapshot of queued callbacks on the communicator state-machine
  /// thread. Events queued while draining are handled by a later dispatch.
  /// Returns true when at least one event was run.
  bool drainStateEvents();

  const std::shared_ptr<Communicator> communicator_;
  std::shared_ptr<EndpointRef> endpointRef_;

 private:
  bool enqueueStateEventImpl(
      std::shared_ptr<CommElement> self,
      StateEvent event) noexcept;

  const bool coalesceWorkQueueEntries_;
  std::atomic<bool> queued_{false};
  std::mutex stateEventMutex_;
  std::deque<StateEvent> stateEvents_;
};
} // namespace facebook::velox::ucx_exchange
