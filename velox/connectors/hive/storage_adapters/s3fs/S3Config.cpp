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

#include "velox/connectors/hive/storage_adapters/s3fs/S3Config.h"

#include "velox/common/config/Config.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3DirectReceive.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Util.h"

namespace facebook::velox::filesystems {

static constexpr size_t kMinimumMultipartMinPartSize = 5U << 20; // 5MB
static constexpr size_t kMaximumMultipartMinPartSize = 5U << 30; // 5GB

std::string S3Config::cacheKey(
    std::string_view bucket,
    std::shared_ptr<const config::ConfigBase> config) {
  std::string endpoint;
  auto bucketEndpoint = bucketConfigKey(Keys::kEndpoint, bucket);
  if (config->valueExists(bucketEndpoint)) {
    endpoint = config->get<std::string>(bucketEndpoint).value();
  } else {
    auto baseEndpoint = baseConfigKey(Keys::kEndpoint);
    if (config->valueExists(baseEndpoint)) {
      endpoint = config->get<std::string>(baseEndpoint).value();
    }
  }

  auto configuredValue = [&](Keys configKey) {
    const auto bucketKey = bucketConfigKey(configKey, bucket);
    if (const auto value = config->get<std::string>(bucketKey)) {
      return value.value();
    }
    const auto baseKey = baseConfigKey(configKey);
    if (const auto value = config->get<std::string>(baseKey)) {
      return value.value();
    }
    return std::string(configTraits().at(configKey).second.value());
  };
  auto directReceiveMode = configuredValue(Keys::kDirectReceiveMode);
  folly::toLowerAscii(directReceiveMode);
  auto sslEnabled = configuredValue(Keys::kSSLEnabled);
  folly::toLowerAscii(sslEnabled);

  // The filesystem registry accepts only a string cache key. Length-prefix
  // every field so endpoint and bucket text cannot collide with policy
  // delimiters or with each other.
  std::string key{"s3:v1"};
  auto appendField = [&](std::string_view value) {
    key.append(fmt::format(":{}:", value.size()));
    key.append(value.data(), value.size());
  };
  appendField(endpoint);
  appendField(bucket);
  appendField(directReceiveMode);
  appendField(sslEnabled);
  return key;
}

S3Config::S3Config(
    std::string_view bucket,
    const std::shared_ptr<const config::ConfigBase> properties)
    : bucket_(bucket) {
  for (int key = static_cast<int>(Keys::kBegin);
       key < static_cast<int>(Keys::kEnd);
       key++) {
    auto s3Key = static_cast<Keys>(key);
    auto value = S3Config::configTraits().find(s3Key)->second;
    auto configSuffix = value.first;
    auto configDefault = value.second;

    // Set bucket S3 config "hive.s3.bucket.*" if present.
    std::stringstream bucketConfig;
    bucketConfig << kS3BucketPrefix << bucket << "." << configSuffix;
    auto configVal = static_cast<std::optional<std::string>>(
        properties->get<std::string>(bucketConfig.str()));
    if (configVal.has_value()) {
      config_[s3Key] = configVal.value();
    } else {
      // Set base config "hive.s3.*" if present.
      std::stringstream baseConfig;
      baseConfig << kS3Prefix << configSuffix;
      configVal = static_cast<std::optional<std::string>>(
          properties->get<std::string>(baseConfig.str()));
      if (configVal.has_value()) {
        config_[s3Key] = configVal.value();
      } else {
        // Set the default value.
        config_[s3Key] = configDefault;
      }
    }
  }
  payloadSigningPolicy_ =
      properties->get<std::string>(kS3PayloadSigningPolicy, "Never");

  VELOX_CHECK_GE(
      minPartSize(),
      kMinimumMultipartMinPartSize,
      "The min-part-size S3 configuration must exceed 5MB.");
  VELOX_CHECK_LE(
      minPartSize(),
      kMaximumMultipartMinPartSize,
      "The min-part-size S3 configuration must not exceed 5GB.");
  // Validate both the configured value and its transport requirements while
  // constructing the filesystem configuration.
  (void)effectiveDirectReceiveMode();
}

std::optional<std::string> S3Config::endpointRegion() const {
  auto region = config_.find(Keys::kEndpointRegion)->second;
  if (!region.has_value()) {
    // If region is not set, try inferring from the endpoint value for AWS
    // endpoints.
    auto endpointValue = endpoint();
    if (endpointValue.has_value()) {
      region = parseAWSStandardRegionName(endpointValue.value());
    }
  }
  return region;
}

size_t S3Config::minPartSize() const {
  return config::toCapacity(
      config_.find(Keys::kMultipartMinPartSize)->second.value(),
      config::CapacityUnit::BYTE);
}

S3DirectReceiveMode S3Config::directReceiveMode() const {
  auto mode = config_.find(Keys::kDirectReceiveMode)->second.value();
  folly::toLowerAscii(mode);
  if (mode == "disabled") {
    return S3DirectReceiveMode::DISABLED;
  }
  if (mode == "caller-buffer") {
    return S3DirectReceiveMode::CALLER_BUFFER;
  }
  if (mode == "preferred") {
    return S3DirectReceiveMode::PREFERRED;
  }
  if (mode == "required") {
    return S3DirectReceiveMode::REQUIRED;
  }
  VELOX_USER_FAIL(
      "Invalid hive.s3.direct-receive-mode '{}'. Expected disabled, "
      "caller-buffer, preferred, or required.",
      mode);
}

S3DirectReceiveMode S3Config::effectiveDirectReceiveMode() const {
  const auto mode = directReceiveMode();
  if (useSSL()) {
    return mode;
  }
  VELOX_USER_CHECK(
      mode != S3DirectReceiveMode::REQUIRED,
      "hive.s3.direct-receive-mode=required requires hive.s3.ssl.enabled=true");
  return mode == S3DirectReceiveMode::PREFERRED
      ? S3DirectReceiveMode::CALLER_BUFFER
      : mode;
}

} // namespace facebook::velox::filesystems
