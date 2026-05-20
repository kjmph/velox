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

#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>

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
      std::shared_ptr<EndpointRef> endpointRef)
      : communicator_{communicator}, endpointRef_{endpointRef} {}

  CommElement(const std::shared_ptr<Communicator> communicator)
      : communicator_{communicator}, endpointRef_{nullptr} {}

  virtual ~CommElement() = default;

  /// @brief Advance the communication by executing the communication elements
  /// specific communication pattern.
  virtual void process() = 0;

  // Called when the underlying endpoint was closed
  // or the communicator is finished.
  virtual void close() = 0;

  /// Per-element process exclusion. The Communicator's primary dispatch
  /// loop calls process() under this mutex; if close() or endpoint cleanup
  /// already owns it, dispatch pushes the element back to the queue and
  /// moves on. Source/server state machines assume this single-owner
  /// process path.
  ///
  /// Recursive because close() can be reached from process() and from
  /// endpoint cleanup paths; both use the same per-element exclusion.
  std::recursive_mutex processMutex_;

 protected:
  using StateEvent = std::function<void()>;

  /// Queue a state-machine event from an arbitrary thread. UCX callbacks
  /// use this to hand completion work back to process(), preserving the
  /// single-owner state-machine invariant.
  void enqueueStateEvent(std::shared_ptr<CommElement> self, StateEvent event);

  /// Runs pending events. Must be called by process() while processMutex_
  /// is held by the Communicator dispatch loop.
  void drainStateEvents();

  const std::shared_ptr<Communicator> communicator_;
  std::shared_ptr<EndpointRef> endpointRef_;

 private:
  std::mutex stateEventMutex_;
  std::deque<StateEvent> stateEvents_;
};
} // namespace facebook::velox::ucx_exchange
