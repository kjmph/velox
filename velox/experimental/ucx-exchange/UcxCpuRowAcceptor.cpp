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
#include <vector>
#include "velox/common/base/Exceptions.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/EndpointRef.h"
#include "velox/experimental/ucx-exchange/UcxCpuRowExchangeServer.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"

namespace facebook::velox::ucx_exchange {

namespace {

struct ParsedCpuHandshake {
  HandshakeMsg handshake;
  std::string_view sourceWorkerAddress;
  uint32_t sourceHostIdHash{0};
};

ParsedCpuHandshake parseCpuHandshake(ucxx::Buffer& buffer) {
  const auto bufferSize = buffer.getSize();
  VELOX_CHECK_GE(
      bufferSize,
      sizeof(HandshakeMsg),
      "CPU AMCallback: received buffer size ({}) is smaller than HandshakeMsg "
      "({}).",
      bufferSize,
      sizeof(HandshakeMsg));

  if (bufferSize < sizeof(CpuRowHandshakeHeader)) {
    ParsedCpuHandshake parsed;
    std::memcpy(&parsed.handshake, buffer.data(), sizeof(parsed.handshake));
    return parsed;
  }

  CpuRowHandshakeHeader header;
  std::memcpy(&header, buffer.data(), sizeof(header));
  if (header.magic != kCpuRowHandshakeMagic) {
    ParsedCpuHandshake parsed;
    std::memcpy(&parsed.handshake, buffer.data(), sizeof(parsed.handshake));
    return parsed;
  }

  VELOX_CHECK_EQ(
      header.version,
      kCpuRowHandshakeVersion,
      "CPU AMCallback: unsupported CpuRowHandshakeHeader version {}",
      header.version);
  VELOX_CHECK_GE(
      header.headerSize,
      sizeof(CpuRowHandshakeHeader),
      "CPU AMCallback: invalid CpuRowHandshakeHeader size {}",
      header.headerSize);
  VELOX_CHECK_LE(
      header.headerSize,
      bufferSize,
      "CPU AMCallback: CpuRowHandshakeHeader size {} exceeds buffer size {}",
      header.headerSize,
      bufferSize);
  VELOX_CHECK_LE(
      header.sourceWorkerAddressBytes,
      bufferSize - header.headerSize,
      "CPU AMCallback: source worker address length {} exceeds remaining "
      "handshake bytes {}",
      header.sourceWorkerAddressBytes,
      bufferSize - header.headerSize);

  const auto* address =
      reinterpret_cast<const char*>(buffer.data()) + header.headerSize;
  return {
      header.handshake,
      std::string_view(address, header.sourceWorkerAddressBytes),
      header.sourceHostIdHash};
}

std::shared_ptr<std::vector<uint8_t>> makeCpuHandshakeResponse(
    CpuRowDataEndpointMode dataEndpointMode,
    std::string_view serverWorkerAddress,
    uint32_t serverHostIdHash) {
  VELOX_CHECK_LE(
      serverWorkerAddress.size(),
      kMaxCpuRowWorkerAddressBytes,
      "CPU HandshakeResponse worker address length {} exceeds limit {}",
      serverWorkerAddress.size(),
      kMaxCpuRowWorkerAddressBytes);

  auto response = std::make_shared<std::vector<uint8_t>>(
      sizeof(CpuRowHandshakeResponseHeader) + serverWorkerAddress.size());
  CpuRowHandshakeResponseHeader header;
  header.dataEndpointMode = dataEndpointMode;
  header.serverWorkerAddressBytes =
      static_cast<uint32_t>(serverWorkerAddress.size());
  header.serverHostIdHash = serverHostIdHash;
  std::memcpy(response->data(), &header, sizeof(header));
  if (!serverWorkerAddress.empty()) {
    std::memcpy(
        response->data() + sizeof(CpuRowHandshakeResponseHeader),
        serverWorkerAddress.data(),
        serverWorkerAddress.size());
  }
  return response;
}

} // namespace

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

  const auto parsedHandshake = parseCpuHandshake(*buffer);
  const auto* handshakePtr = &parsedHandshake.handshake;
  const auto taskIdLength =
      ::strnlen(handshakePtr->taskId, sizeof(handshakePtr->taskId));
  VELOX_CHECK_LT(
      taskIdLength,
      sizeof(handshakePtr->taskId),
      "CPU AMCallback: task ID in HandshakeMsg is not NUL-terminated");
  VELOX_CHECK_GT(
      taskIdLength, 0, "CPU AMCallback: task ID in HandshakeMsg is empty");

  std::shared_ptr<Communicator> communicator = Communicator::getInstance();
  auto bootstrapEpRef = communicator->findEndpointRefByHandle(ep);
  VELOX_CHECK_NOT_NULL(
      bootstrapEpRef, "CPU AMCallback: could not find endpoint ref");

  const PartitionKey key{
      std::string(handshakePtr->taskId, taskIdLength),
      handshakePtr->destination};

  auto dataEpRef = bootstrapEpRef;
  CpuRowDataEndpointMode dataEndpointMode = CpuRowDataEndpointMode::kBootstrap;
  std::string serverWorkerAddress;
  const bool hasWorkerAddress = !parsedHandshake.sourceWorkerAddress.empty();
  const bool sameHost = communicator->hasSameHostTransportIdentity(
      parsedHandshake.sourceHostIdHash);
  if (hasWorkerAddress && sameHost) {
    serverWorkerAddress = communicator->getWorkerAddress();
    if (serverWorkerAddress.empty() ||
        serverWorkerAddress.size() > kMaxCpuRowWorkerAddressBytes) {
      serverWorkerAddress.clear();
    } else if (auto sameHostDataEpRef =
            communicator->createSameHostEndpointRefFromWorkerAddress(
                parsedHandshake.sourceWorkerAddress,
                bootstrapEpRef->getPeerIp(),
                parsedHandshake.sourceHostIdHash)) {
      dataEpRef = std::move(sameHostDataEpRef);
      dataEndpointMode = CpuRowDataEndpointMode::kSameHostWorkerAddress;
    } else {
      serverWorkerAddress.clear();
    }
  }

  const bool waitForDataEndpointAck =
      dataEndpointMode == CpuRowDataEndpointMode::kSameHostWorkerAddress;
  auto exchangeServer = UcxCpuRowExchangeServer::create(
      communicator, dataEpRef, key, waitForDataEndpointAck);
  exchangeServer->postDataEndpointAckReceive();

  dataEpRef->addCommElem(exchangeServer);
  communicator->registerCommElement(exchangeServer);

  auto response = makeCpuHandshakeResponse(
      dataEndpointMode, serverWorkerAddress, communicator->getHostIdHash());

  uint64_t responseTag = getHandshakeResponseTag(fnv1a_32(key.toString()));
  // Keep the control-plane response on the listener endpoint that received the
  // AM. If same-host data is selected, this response carries the server worker
  // address so the source can create the reciprocal data endpoint before
  // posting metadata receives.
  bootstrapEpRef->endpoint_->tagSend(
      response->data(),
      response->size(),
      ucxx::Tag{responseTag},
      false,
      [response, keyStr = key.toString()](
          ucs_status_t status, std::shared_ptr<void> /*arg*/) {
        if (status != UCS_OK) {
          LOG(WARNING) << "Failed to send CPU HandshakeResponse to " << keyStr
                       << ": " << ucs_status_string(status);
        }
      },
      response);
}

} // namespace facebook::velox::ucx_exchange
