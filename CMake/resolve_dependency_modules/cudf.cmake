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

# 4.0 is the minimum version required by cudf
cmake_minimum_required(VERSION 4.0)

# rapids_cmake commit 323d37b from 2026-06-23
set(VELOX_rapids_cmake_VERSION 26.08)
set(VELOX_rapids_cmake_COMMIT 323d37beeb2030cd5c9e7e981810915d59ecda09)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  bacf4aa0b253ddbc7b103793815909b5d61cee5604b2be14d715351b675e9de5
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit a4ab399 from 2026-06-17
set(VELOX_rmm_VERSION 26.08)
set(VELOX_rmm_COMMIT a4ab39907900d45f220ea7c2d3ecff1b56d39909)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  92a3280264ffa6225124452c1c10b38f047ae4a04b9c38052aa483e9b42f04cd
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# Caller-owned S3 receive requires the KvikIO research implementation for
# bounded pinned-host staging, event-fenced H2D copies, and strict path
# accounting. Keep the output-transport pin unchanged for ordinary cuDF
# builds.
if(VELOX_ENABLE_S3_DIRECT_RECEIVE)
  set(VELOX_kvikio_VERSION 26.10)
  set(VELOX_kvikio_COMMIT 1db0457edde588e5d048d11b392802966913edce)
  set(
    VELOX_kvikio_BUILD_SHA256_CHECKSUM
    22ee39ee354b171c315796e6f715dc97f12e57d90fdfd20af387c806fa90b061
  )
  set(
    VELOX_kvikio_SOURCE_URL
    "https://github.com/kjmph/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
  )
else()
  # kvikio commit bdb788f from 2026-06-16
  set(VELOX_kvikio_VERSION 26.08)
  set(VELOX_kvikio_COMMIT bdb788f45ef191384a294ecef3312ea2db35a2c7)
  set(
    VELOX_kvikio_BUILD_SHA256_CHECKSUM
    c8db1083756337a3b0dc1616f3960f53fea891763fd9e1645cd38d7e218c7a47
  )
  set(
    VELOX_kvikio_SOURCE_URL
    "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
  )
endif()
velox_resolve_dependency_url(kvikio)

set(VELOX_cudf_VERSION 26.08 CACHE STRING "cudf version")
# GPU input uses cuDF's lifetime-safe asynchronous reads and optional batched
# datasource interface regardless of the selected S3 reader mode. Keep one pin
# for direct-receive and ordinary builds so both expose the same datasource ABI.
set(VELOX_cudf_COMMIT 285b175a37736f4204395b4c293bf111f0ac34a0)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  c09034cad016449bc4f7f010e71417aee2519e7fb536bfc8aa5acd88906220bf
)
set(VELOX_cudf_SOURCE_URL "https://github.com/kjmph/cudf/archive/${VELOX_cudf_COMMIT}.tar.gz")
velox_resolve_dependency_url(cudf)

# Probe for a system UCX install. The variables are used only to gate ucxx
# fetching below; nothing in Velox links against UCX directly yet.
find_library(UCX_LIBRARY NAMES ucp)
find_path(UCX_INCLUDE_DIR NAMES ucp/api/ucp.h)
if(UCX_LIBRARY AND UCX_INCLUDE_DIR)
  set(UCX_FOUND TRUE)
else()
  set(UCX_FOUND FALSE)
endif()
if(UCX_FOUND)
  message(STATUS "Found UCX: ${UCX_LIBRARY} (headers: ${UCX_INCLUDE_DIR}) -- ucxx will be fetched")
  # kjmph/ucxx commit based on fe38756 (release/0.50 branch) with
  # progress-thread control-plane fairness.
  set(VELOX_ucxx_VERSION 0.51)
  set(VELOX_ucxx_COMMIT a7f9228bbc9fd45b1056758f3a7067f7d5d65947)
  set(
    VELOX_ucxx_BUILD_SHA256_CHECKSUM
    6ae638a3a86790d5e2b36a06f1ea6bf188e358e4ca3ecb4731cb9aa9145aa19c
  )
  set(VELOX_ucxx_SOURCE_URL "https://github.com/kjmph/ucxx/archive/${VELOX_ucxx_COMMIT}.tar.gz")
  velox_resolve_dependency_url(ucxx)
else()
  message(STATUS "UCX not found -- ucxx will not be fetched")
endif()

# Use block so we don't leak variables
block(SCOPE_FOR VARIABLES)
  # Setup libcudf build to not have testing components
  set(BUILD_TESTS OFF)
  set(CUDF_BUILD_TESTUTIL OFF)
  set(CUDF_BUILD_STREAMS_TEST_UTIL OFF)
  set(BUILD_SHARED_LIBS ON)
  if(VELOX_ENABLE_S3_DIRECT_RECEIVE)
    # KvikIO 26.10 enables its Nsight plugin by default. It is not part of the
    # runtime data path and would add an unnecessary build/runtime dependency.
    set(KvikIO_BUILD_NSYS_PLUGIN OFF)
  endif()

  # TODO(mh,bd): Remove this once we have a permanent solution for the spdlog/fmt
  # incompatibility.

  # Override cuDF's RAPIDS-CMake dependencies with the custom CCCL scan fix and
  # spdlog 1.15.3, which is compatible with the fmt 11.2.0 that Velox builds.
  # RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE is honored by every rapids_cpm_init,
  # so both overrides apply before cuDF resolves either dependency.
  set(RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE "${CMAKE_CURRENT_LIST_DIR}/cudf-cpm-overrides.json")

  FetchContent_Declare(
    rapids-cmake
    URL ${VELOX_rapids_cmake_SOURCE_URL}
    URL_HASH ${VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM}
    UPDATE_DISCONNECTED 1
  )

  FetchContent_Declare(
    rmm
    URL ${VELOX_rmm_SOURCE_URL}
    URL_HASH ${VELOX_rmm_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR
    cpp
    UPDATE_DISCONNECTED 1
  )

  FetchContent_Declare(
    kvikio
    URL ${VELOX_kvikio_SOURCE_URL}
    URL_HASH ${VELOX_kvikio_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR
    cpp
    UPDATE_DISCONNECTED 1
  )

  FetchContent_Declare(
    cudf
    URL ${VELOX_cudf_SOURCE_URL}
    URL_HASH ${VELOX_cudf_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR
    cpp
    UPDATE_DISCONNECTED 1
  )

  if(UCX_FOUND)
    FetchContent_Declare(
      ucxx
      URL ${VELOX_ucxx_SOURCE_URL}
      URL_HASH ${VELOX_ucxx_BUILD_SHA256_CHECKSUM}
      SOURCE_SUBDIR
      cpp
      UPDATE_DISCONNECTED 1
    )
  endif()

  FetchContent_MakeAvailable(cudf)

  if(UCX_FOUND)
    FetchContent_MakeAvailable(ucxx)
  endif()

  # cudf sets all warnings as errors, and therefore fails to compile with velox
  # expanded set of warnings. We selectively disable problematic warnings just for
  # cudf
  target_compile_options(
    cudf
    PRIVATE -Wno-non-virtual-dtor -Wno-missing-field-initializers -Wno-deprecated-copy -Wno-restrict
  )

  unset(BUILD_SHARED_LIBS)
  unset(BUILD_TESTING CACHE)
endblock()
