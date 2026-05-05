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

# Standalone fetch of ucxx for CPU-only builds.
#
# In cudf builds, ucxx is pulled in transitively by cudf.cmake's
# FetchContent_MakeAvailable(cudf). When cudf is off but
# VELOX_ENABLE_UCX_EXCHANGE is on, we still need
# the `ucxx::ucxx` target, so this module fetches just rapids-cmake +
# ucxx and skips the heavy rmm/cudf chain. Versions are matched against
# the cuDF 26.06 dependency set (ucxx 0.45.01, rapids-cmake 26.06).

include_guard(GLOBAL)

# rapids-cmake gives ucxx the rapids_* macros it uses internally. The
# version mirrors what cudf.cmake pulls so a future combined build won't
# end up with two different copies.
set(VELOX_UCXX_rapids_cmake_VERSION 26.06)
set(VELOX_UCXX_rapids_cmake_COMMIT cac9e0ee144fdab03e8bd19f2124ec0239c36924)
set(
  VELOX_UCXX_rapids_cmake_BUILD_SHA256_CHECKSUM
  a43b508ac36a1154a656b2996180ec1e73b08feaa344888856b67661f70ebac5
)
set(
  VELOX_UCXX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_UCXX_rapids_cmake_COMMIT}.tar.gz"
)

# ucxx 0.45.01 is what cudf 26.06 pulls; matching to avoid version skew.
set(VELOX_UCXX_VERSION 0.45.01)
set(
  VELOX_UCXX_SOURCE_URL
  "https://github.com/rapidsai/ucxx/archive/v${VELOX_UCXX_VERSION}.tar.gz"
)

block(SCOPE_FOR VARIABLES)
  # Ucxx options. RMM off means no cudf/CUDA dependency. Tests off means
  # no GoogleTest fetch.
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
    SOURCE_SUBDIR cpp
    UPDATE_DISCONNECTED 1
  )

  FetchContent_MakeAvailable(rapids-cmake ucxx)
endblock()
