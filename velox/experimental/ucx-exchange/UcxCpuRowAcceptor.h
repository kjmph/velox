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

#include <ucxx/api.h>

/// CPU RowVector mirror of Acceptor. Registered on a separate
/// Communicator AM callback id (kAmCpuCallbackId) so it never sees
/// cudf-side handshakes; this keeps both transports independent and
/// avoids touching the existing cudf Acceptor / UcxExchangeServer
/// path.

namespace facebook::velox::ucx_exchange {

struct UcxCpuRowAcceptor {
  /// AM callback for the CPU-row handshake. Receives a HandshakeMsg
  /// (taskId + destination), creates a UcxCpuRowExchangeServer, and
  /// registers it with the Communicator. CPU-row exchange always uses
  /// UCX for handshaking; data may be published through shared memory
  /// when both processes share an IPC namespace.
  static void cStyleAMCallback(
      std::shared_ptr<ucxx::Request> request,
      ucp_ep_h ep);
};

} // namespace facebook::velox::ucx_exchange
