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
#include "velox/experimental/ucx-exchange/UcxCpuRowAcceptor.h"
#include <glog/logging.h>
#include <cstring>
#include "velox/common/base/Exceptions.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/EndpointRef.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowExchangeServer.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowShm.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"

namespace facebook::velox::ucx_exchange {

/* static */
void UcxCpuRowAcceptor::cStyleAMCallback(
    std::shared_ptr<ucxx::Request> request,
    ucp_ep_h ep) {
  VELOX_CHECK_NOT_NULL(request, "CPU AMCallback called with nullptr request");
  VELOX_CHECK(
      request->isCompleted(), "CPU AMCallback called with incomplete request");

  auto buffer =
      std::dynamic_pointer_cast<ucxx::Buffer>(request->getRecvBuffer());
  VELOX_CHECK_NOT_NULL(buffer, "CPU AMCallback: failed to get receive buffer");
  VELOX_CHECK_GE(
      buffer->getSize(),
      sizeof(HandshakeMsg),
      "CPU AMCallback: received buffer size ({}) is smaller than HandshakeMsg "
      "({}).",
      buffer->getSize(),
      sizeof(HandshakeMsg));

  const auto* handshakePtr =
      reinterpret_cast<const HandshakeMsg*>(buffer->data());
  const auto taskIdLength =
      ::strnlen(handshakePtr->taskId, sizeof(handshakePtr->taskId));
  VELOX_CHECK_LT(
      taskIdLength,
      sizeof(handshakePtr->taskId),
      "CPU AMCallback: task ID in HandshakeMsg is not NUL-terminated");
  VELOX_CHECK_GT(
      taskIdLength, 0, "CPU AMCallback: task ID in HandshakeMsg is empty");

  std::shared_ptr<Communicator> communicator = Communicator::getInstance();
  auto epRef = communicator->findEndpointRefByHandle(ep);
  VELOX_CHECK_NOT_NULL(epRef, "CPU AMCallback: could not find endpoint ref");

  const PartitionKey key{
      std::string(handshakePtr->taskId, taskIdLength),
      handshakePtr->destination};

  bool canUseCpuShm = false;
  if (ucxCpuRowShmEnabled() && handshakePtr->cpuShmProbeName[0] != '\0') {
    const auto probeNameLength = ::strnlen(
        handshakePtr->cpuShmProbeName, sizeof(handshakePtr->cpuShmProbeName));
    if (probeNameLength < sizeof(handshakePtr->cpuShmProbeName)) {
      auto probeSegment = openUcxCpuRowShmSegment(
          std::string_view(handshakePtr->cpuShmProbeName, probeNameLength),
          sizeof(uint64_t),
          false);
      if (probeSegment) {
        uint64_t probeToken = 0;
        std::memcpy(&probeToken, probeSegment->data(), sizeof(probeToken));
        canUseCpuShm = probeToken == handshakePtr->cpuShmProbeToken;
        probeSegment->unlinkOnDestroy = canUseCpuShm;
      }
    }
  }

  auto exchangeServer =
      UcxCpuRowExchangeServer::create(communicator, epRef, key, canUseCpuShm);

  epRef->addCommElem(exchangeServer);
  communicator->registerCommElement(exchangeServer);

  VLOG(2) << "[ACCEPTOR-CPU] new server: " << exchangeServer->toString()
          << " peerIp=" << epRef->getPeerIp()
          << " canUseCpuShm=" << canUseCpuShm;

  auto response = std::make_shared<HandshakeResponse>();
  response->isIntraNodeTransfer = false;
  response->canUseCpuShm = canUseCpuShm;

  uint64_t responseTag = getHandshakeResponseTag(fnv1a_32(key.toString()));
  epRef->endpoint_->tagSend(
      response.get(),
      sizeof(*response),
      ucxx::Tag{responseTag},
      false,
      [response, keyStr = key.toString()](
          ucs_status_t status, std::shared_ptr<void> /*arg*/) {
        if (status == UCS_OK) {
          VLOG(3) << "CPU HandshakeResponse sent successfully to " << keyStr;
        } else {
          VLOG(0) << "Failed to send CPU HandshakeResponse to " << keyStr
                  << ": " << ucs_status_string(status);
        }
      },
      response);
}

} // namespace facebook::velox::ucx_exchange
