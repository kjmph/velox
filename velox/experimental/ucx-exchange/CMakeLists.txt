# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Sources that don't need cudf or CUDA. Always built.
set(
  UCX_EXCHANGE_CORE_SOURCES
  Acceptor.cpp
  CommElement.cpp
  Communicator.cpp
  EndpointRef.cpp
  NetUtil.cpp
  UcxExchangeProtocol.cpp
  # CPU RowVector mirror of the cudf stack. PrestoSerializer-based, no
  # cudf dependency.
  UcxCpuRowMetadataMsg.cpp
  UcxCpuRowQueues.cpp
  UcxCpuRowOutputQueueManager.cpp
  UcxCpuRowExchangeServer.cpp
  UcxCpuRowExchangeQueue.cpp
  UcxCpuRowExchangeSource.cpp
  UcxCpuRowExchangeClient.cpp
  UcxCpuRowAcceptor.cpp
  UcxCpuRowPartitionedOutput.cpp
  UcxCpuRowExchange.cpp
  UcxCpuRowDriverAdapter.cpp
)

# Sources that depend on cudf::packed_columns / RMM / CUDA runtime. Only
# compiled when VELOX_ENABLE_CUDF is on.
set(
  UCX_EXCHANGE_CUDF_SOURCES
  UcxCudfOutputQueueManagerBridge.cpp
  UcxCudfDriverAdapter.cpp
  UcxPartitionedOutput.cpp
  UcxOutputQueueManager.cpp
  UcxQueues.cpp
  UcxExchange.cpp
  UcxExchangeSource.cpp
  UcxExchangeServer.cpp
  UcxExchangeQueue.cpp
  UcxExchangeClient.cpp
  IntraNodeTransferRegistry.cpp
)

if(VELOX_ENABLE_CUDF)
  set(UCX_EXCHANGE_SOURCES
    ${UCX_EXCHANGE_CORE_SOURCES} ${UCX_EXCHANGE_CUDF_SOURCES})
else()
  set(UCX_EXCHANGE_SOURCES ${UCX_EXCHANGE_CORE_SOURCES})
endif()

# Generate a header containing the module basenames (without .cpp extension)
# for use by Communicator to set per-file VLOG levels.
set(UCX_EXCHANGE_MODULE_LIST "")
foreach(src ${UCX_EXCHANGE_SOURCES})
  string(REGEX REPLACE "\\.cpp$" "" module "${src}")
  string(APPEND UCX_EXCHANGE_MODULE_LIST "\"${module}\",\n    ")
endforeach()
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/UcxExchangeModules.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/UcxExchangeModules.h
  @ONLY
)

add_library(velox_ucx_exchange OBJECT ${UCX_EXCHANGE_SOURCES})

target_include_directories(velox_ucx_exchange PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

find_package(ucx REQUIRED)

if(VELOX_ENABLE_CUDF)
  target_link_libraries(
    velox_ucx_exchange
    PUBLIC ucxx::ucxx ucx::ucp
    PRIVATE
      cudf::cudf
      CUDA::cudart
      CUDA::cuda_driver
      Folly::folly
      velox_core
      ucx::ucs
  )
else()
  # No cudf / CUDA in CPU-only mode. The Communicator's
  # CudfConfig usage and cuda_runtime calls are guarded behind
  # `#ifdef VELOX_ENABLE_CUDF`.
  target_link_libraries(
    velox_ucx_exchange
    PUBLIC ucxx::ucxx ucx::ucp
    PRIVATE Folly::folly velox_core ucx::ucs
  )
endif()

if(VELOX_BUILD_TESTING AND VELOX_ENABLE_CUDF)
  add_subdirectory(tests)
endif()
