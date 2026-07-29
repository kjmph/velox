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

include_guard(GLOBAL)

# Match the UCXX and rapids-cmake revisions used by cudf.cmake so CPU-only and
# combined cuDF builds cannot resolve two incompatible dependency copies.
set(VELOX_UCXX_rapids_cmake_VERSION 26.12)
set(VELOX_UCXX_rapids_cmake_COMMIT 037d4c5489cc13f5b378b54197238a6e2cdbff17)
set(
  VELOX_UCXX_rapids_cmake_BUILD_SHA256_CHECKSUM
  946ebc43d81dc7359e973df54bc0749f281a0b65b32a299b00ee6a66507b0a04
)
set(
  VELOX_UCXX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_UCXX_rapids_cmake_COMMIT}.tar.gz"
)

set(VELOX_UCXX_VERSION 0.53)
set(VELOX_UCXX_COMMIT f3e85e580fb4349240917f85f5a6260c7f3b94f6)
set(
  VELOX_UCXX_BUILD_SHA256_CHECKSUM
  63c5f8a9e7a84bab932b52d1dad82619c684a0383b3318914eba029d9b2fa9cd
)
set(
  VELOX_UCXX_SOURCE_URL
  "https://github.com/rapidsai/ucxx/archive/${VELOX_UCXX_COMMIT}.tar.gz"
)

block(SCOPE_FOR VARIABLES)
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)
  set(BUILD_EXAMPLES OFF)
  set(UCXX_ENABLE_RMM OFF)
  set(BUILD_SHARED_LIBS ON)

  FetchContent_Declare(
    rapids-cmake
    URL ${VELOX_UCXX_rapids_cmake_SOURCE_URL}
    URL_HASH SHA256=${VELOX_UCXX_rapids_cmake_BUILD_SHA256_CHECKSUM}
    UPDATE_DISCONNECTED 1
  )
  FetchContent_Declare(
    ucxx
    URL ${VELOX_UCXX_SOURCE_URL}
    URL_HASH SHA256=${VELOX_UCXX_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR cpp
    UPDATE_DISCONNECTED 1
    PATCH_COMMAND git init -q && git apply
      ${CMAKE_CURRENT_LIST_DIR}/ucxx/control-plane-fairness.patch
  )
  FetchContent_MakeAvailable(rapids-cmake ucxx)
endblock()
