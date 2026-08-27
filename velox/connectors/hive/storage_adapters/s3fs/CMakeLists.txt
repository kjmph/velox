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

# for generated headers

velox_add_library(
  velox_s3fs
  RegisterS3FileSystem.cpp
  HEADERS
  RegisterS3FileSystem.h
  S3Config.h
  S3Counters.h
  S3DirectReceive.h
  S3DirectReceiveCapabilityCache.h
  S3FileSystem.h
  S3ReadFile.h
  S3Util.h
  S3WriteFile.h
)
if(VELOX_ENABLE_S3)
  velox_sources(
    velox_s3fs
    PRIVATE
      S3FileSystem.cpp
      S3Util.cpp
      S3Config.cpp
      S3DirectReceive.cpp
      S3DirectReceiveCapabilityCache.cpp
      S3WriteFile.cpp
      S3ReadFile.cpp
  )

  # S3DirectReceive.h is a public header and directly includes AWS and Folly
  # headers. Propagate these usage requirements to all consumers.
  velox_include_directories(velox_s3fs PUBLIC ${AWSSDK_INCLUDE_DIRS})
  velox_link_libraries(velox_s3fs PUBLIC Folly::folly PRIVATE velox_dwio_common ${AWSSDK_LIBRARIES})
  if(VELOX_ENABLE_S3_DIRECT_RECEIVE)
    velox_compile_definitions(velox_s3fs PUBLIC VELOX_ENABLE_S3_DIRECT_RECEIVE)
    if(VELOX_S3_DIRECT_RECEIVE_KTLS_SUPPORTED)
      velox_compile_definitions(velox_s3fs PUBLIC VELOX_ENABLE_S3_DIRECT_RECEIVE_KTLS)
    endif()
    velox_link_libraries(velox_s3fs PRIVATE CURL::libcurl)
  endif()

  if(${VELOX_BUILD_TESTING})
    add_subdirectory(tests)
  endif()
endif()
