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
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/config/Config.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3DirectReceive.h"

#include <gtest/gtest.h>

namespace facebook::velox::filesystems {
namespace {
TEST(S3ConfigTest, defaultConfig) {
  auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>());
  auto s3Config = S3Config("", config);
  ASSERT_EQ(s3Config.useVirtualAddressing(), true);
  ASSERT_EQ(s3Config.useSSL(), true);
  ASSERT_EQ(s3Config.useInstanceCredentials(), false);
  ASSERT_EQ(s3Config.endpoint(), std::nullopt);
  ASSERT_EQ(s3Config.endpointRegion(), std::nullopt);
  ASSERT_EQ(s3Config.accessKey(), std::nullopt);
  ASSERT_EQ(s3Config.secretKey(), std::nullopt);
  ASSERT_EQ(s3Config.iamRole(), std::nullopt);
  ASSERT_EQ(s3Config.iamRoleSessionName(), "velox-session");
  ASSERT_EQ(s3Config.payloadSigningPolicy(), "Never");
  ASSERT_EQ(
      s3Config.cacheKey("foo", config),
      "s3:v2:0::3:foo:8:disabled:4:true:5:false");
  ASSERT_EQ(s3Config.bucket(), "");
  ASSERT_EQ(s3Config.useIMDS(), true);
  ASSERT_EQ(s3Config.minPartSize(), 10485760);
  ASSERT_EQ(s3Config.directReceiveMode(), S3DirectReceiveMode::DISABLED);
  ASSERT_FALSE(s3Config.adaptiveTcpMss());
}

TEST(S3ConfigTest, overrideConfig) {
  std::unordered_map<std::string, std::string> configFromFile = {
      {S3Config::baseConfigKey(S3Config::Keys::kPathStyleAccess), "true"},
      {S3Config::baseConfigKey(S3Config::Keys::kSSLEnabled), "false"},
      {S3Config::baseConfigKey(S3Config::Keys::kUseInstanceCredentials),
       "true"},
      {"hive.s3.payload-signing-policy", "RequestDependent"},
      {S3Config::baseConfigKey(S3Config::Keys::kEndpoint), "endpoint"},
      {S3Config::baseConfigKey(S3Config::Keys::kEndpointRegion), "region"},
      {S3Config::baseConfigKey(S3Config::Keys::kAccessKey), "access"},
      {S3Config::baseConfigKey(S3Config::Keys::kSecretKey), "secret"},
      {S3Config::baseConfigKey(S3Config::Keys::kIamRole), "iam"},
      {S3Config::baseConfigKey(S3Config::Keys::kIamRoleSessionName), "velox"},
      {S3Config::baseConfigKey(S3Config::Keys::kCredentialsProvider),
       "my-credentials-provider"},
      {S3Config::baseConfigKey(S3Config::Keys::kIMDSEnabled), "false"},
      {S3Config::baseConfigKey(S3Config::Keys::kMultipartMinPartSize), "20MB"},
      {S3Config::baseConfigKey(S3Config::Keys::kAdaptiveTcpMss), "true"},
      {S3Config::baseConfigKey(S3Config::Keys::kDirectReceiveMode),
       "preferred"}};
  auto configBase =
      std::make_shared<config::ConfigBase>(std::move(configFromFile));
  auto s3Config = S3Config("bucket", configBase);
  ASSERT_EQ(s3Config.useVirtualAddressing(), false);
  ASSERT_EQ(s3Config.useSSL(), false);
  ASSERT_EQ(s3Config.useInstanceCredentials(), true);
  ASSERT_EQ(s3Config.endpoint(), "endpoint");
  ASSERT_EQ(s3Config.endpointRegion(), "region");
  ASSERT_EQ(s3Config.accessKey(), std::optional("access"));
  ASSERT_EQ(s3Config.secretKey(), std::optional("secret"));
  ASSERT_EQ(s3Config.iamRole(), std::optional("iam"));
  ASSERT_EQ(s3Config.iamRoleSessionName(), "velox");
  ASSERT_EQ(s3Config.payloadSigningPolicy(), "RequestDependent");
  ASSERT_EQ(
      s3Config.cacheKey("foo", configBase),
      "s3:v2:8:endpoint:3:foo:9:preferred:5:false:4:true");
  ASSERT_EQ(
      s3Config.cacheKey("bar", configBase),
      "s3:v2:8:endpoint:3:bar:9:preferred:5:false:4:true");
  ASSERT_EQ(s3Config.bucket(), "bucket");
  ASSERT_EQ(s3Config.credentialsProvider(), "my-credentials-provider");
  ASSERT_EQ(s3Config.useIMDS(), false);
  ASSERT_EQ(s3Config.minPartSize(), 20971520);
  ASSERT_EQ(s3Config.directReceiveMode(), S3DirectReceiveMode::PREFERRED);
  ASSERT_TRUE(s3Config.adaptiveTcpMss());
  ASSERT_EQ(
      s3Config.effectiveDirectReceiveMode(),
      S3DirectReceiveMode::CALLER_BUFFER);
}

TEST(S3ConfigTest, directReceiveModeValidation) {
  for (const auto& [value, expected] :
       std::vector<std::pair<std::string, S3DirectReceiveMode>>{
           {"disabled", S3DirectReceiveMode::DISABLED},
           {"CALLER-BUFFER", S3DirectReceiveMode::CALLER_BUFFER},
           {"preferred", S3DirectReceiveMode::PREFERRED},
           {"required", S3DirectReceiveMode::REQUIRED},
       }) {
    auto properties = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>{
            {S3Config::baseConfigKey(S3Config::Keys::kDirectReceiveMode),
             value}});
    EXPECT_EQ(S3Config("bucket", properties).directReceiveMode(), expected);
  }

  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {S3Config::baseConfigKey(S3Config::Keys::kDirectReceiveMode),
           "sometimes"}});
  VELOX_ASSERT_USER_THROW(
      S3Config("bucket", properties),
      "Expected disabled, caller-buffer, preferred, or required");
}

TEST(S3ConfigTest, directReceiveTransportValidation) {
  auto requiredWithoutTls = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {S3Config::baseConfigKey(S3Config::Keys::kSSLEnabled), "false"},
          {S3Config::baseConfigKey(S3Config::Keys::kDirectReceiveMode),
           "required"}});
  VELOX_ASSERT_USER_THROW(
      S3Config("bucket", requiredWithoutTls),
      "direct-receive-mode=required requires hive.s3.ssl.enabled=true");

  auto preferredWithTls = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {S3Config::baseConfigKey(S3Config::Keys::kDirectReceiveMode),
           "preferred"}});
  EXPECT_EQ(
      S3Config("bucket", preferredWithTls).effectiveDirectReceiveMode(),
      S3DirectReceiveMode::PREFERRED);
}

TEST(S3ConfigTest, directReceivePolicyIsPartOfFileSystemCacheIdentity) {
  auto makeProperties = [](std::string mode, std::string ssl) {
    return std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>{
            {S3Config::baseConfigKey(S3Config::Keys::kEndpoint), "endpoint"},
            {S3Config::baseConfigKey(S3Config::Keys::kDirectReceiveMode),
             std::move(mode)},
            {S3Config::baseConfigKey(S3Config::Keys::kSSLEnabled),
             std::move(ssl)}});
  };

  EXPECT_EQ(
      "s3:v2:8:endpoint:6:bucket:8:disabled:4:true:5:false",
      S3Config::cacheKey("bucket", makeProperties("disabled", "true")));
  EXPECT_EQ(
      "s3:v2:8:endpoint:6:bucket:13:caller-buffer:4:true:5:false",
      S3Config::cacheKey("bucket", makeProperties("caller-buffer", "true")));
  EXPECT_EQ(
      "s3:v2:8:endpoint:6:bucket:9:preferred:4:true:5:false",
      S3Config::cacheKey("bucket", makeProperties("PREFERRED", "TRUE")));
  EXPECT_EQ(
      "s3:v2:8:endpoint:6:bucket:9:preferred:5:false:5:false",
      S3Config::cacheKey("bucket", makeProperties("preferred", "false")));

  auto disabledProperties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  EXPECT_NE(
      S3Config::cacheKey("bucket", makeProperties("preferred", "true")),
      S3Config::cacheKey(
          "endpoint-bucket-direct-receive-preferred-ssl-true",
          disabledProperties));

  auto adaptiveProperties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {S3Config::baseConfigKey(S3Config::Keys::kEndpoint), "endpoint"},
          {S3Config::baseConfigKey(S3Config::Keys::kDirectReceiveMode),
           "preferred"},
          {S3Config::baseConfigKey(S3Config::Keys::kSSLEnabled), "true"},
          {S3Config::baseConfigKey(S3Config::Keys::kAdaptiveTcpMss), "true"}});
  EXPECT_NE(
      S3Config::cacheKey("bucket", makeProperties("preferred", "true")),
      S3Config::cacheKey("bucket", adaptiveProperties));
}

TEST(S3ConfigTest, overrideBucketConfig) {
  std::string_view bucket = "bucket";
  std::unordered_map<std::string, std::string> bucketConfigFromFile = {
      {S3Config::baseConfigKey(S3Config::Keys::kPathStyleAccess), "true"},
      {S3Config::baseConfigKey(S3Config::Keys::kSSLEnabled), "false"},
      {S3Config::baseConfigKey(S3Config::Keys::kUseInstanceCredentials),
       "true"},
      {S3Config::baseConfigKey(S3Config::Keys::kEndpoint), "endpoint"},
      {S3Config::bucketConfigKey(S3Config::Keys::kEndpoint, bucket),
       "bucket.s3-region.amazonaws.com"},
      {S3Config::baseConfigKey(S3Config::Keys::kAccessKey), "access"},
      {S3Config::bucketConfigKey(S3Config::Keys::kAccessKey, bucket),
       "bucket-access"},
      {"hive.s3.payload-signing-policy", "Always"},
      {S3Config::baseConfigKey(S3Config::Keys::kSecretKey), "secret"},
      {S3Config::bucketConfigKey(S3Config::Keys::kSecretKey, bucket),
       "bucket-secret"},
      {S3Config::baseConfigKey(S3Config::Keys::kIamRole), "iam"},
      {S3Config::baseConfigKey(S3Config::Keys::kIamRoleSessionName), "velox"},
      {S3Config::baseConfigKey(S3Config::Keys::kCredentialsProvider),
       "my-credentials-provider"},
      {S3Config::bucketConfigKey(S3Config::Keys::kCredentialsProvider, bucket),
       "override-credentials-provider"},
      {S3Config::baseConfigKey(S3Config::Keys::kIMDSEnabled), "false"},
      {S3Config::baseConfigKey(S3Config::Keys::kMultipartMinPartSize), "20MB"}};
  auto configBase =
      std::make_shared<config::ConfigBase>(std::move(bucketConfigFromFile));
  auto s3Config = S3Config(bucket, configBase);
  ASSERT_EQ(s3Config.useVirtualAddressing(), false);
  ASSERT_EQ(s3Config.useSSL(), false);
  ASSERT_EQ(s3Config.useInstanceCredentials(), true);
  ASSERT_EQ(s3Config.endpoint(), "bucket.s3-region.amazonaws.com");
  // Inferred from the endpoint.
  ASSERT_EQ(s3Config.endpointRegion(), "region");
  ASSERT_EQ(s3Config.accessKey(), std::optional("bucket-access"));
  ASSERT_EQ(s3Config.secretKey(), std::optional("bucket-secret"));
  ASSERT_EQ(s3Config.iamRole(), std::optional("iam"));
  ASSERT_EQ(s3Config.iamRoleSessionName(), "velox");
  ASSERT_EQ(s3Config.payloadSigningPolicy(), "Always");
  ASSERT_EQ(
      s3Config.cacheKey(bucket, configBase),
      "s3:v2:30:bucket.s3-region.amazonaws.com:6:bucket:8:disabled:5:false:5:false");
  ASSERT_EQ(
      s3Config.cacheKey("foo", configBase),
      "s3:v2:8:endpoint:3:foo:8:disabled:5:false:5:false");
  ASSERT_EQ(s3Config.credentialsProvider(), "override-credentials-provider");
  ASSERT_EQ(s3Config.useIMDS(), false);
  ASSERT_EQ(s3Config.minPartSize(), 20971520);
}

TEST(S3ConfigTest, minPartSizeValidation) {
  // Test that setting min-part-size below 5MB throws an error.
  std::unordered_map<std::string, std::string> configFromFile = {
      {S3Config::baseConfigKey(S3Config::Keys::kMultipartMinPartSize), "4MB"}};
  auto configBase =
      std::make_shared<config::ConfigBase>(std::move(configFromFile));

  VELOX_ASSERT_THROW(
      S3Config("bucket", configBase),
      "The min-part-size S3 configuration must exceed 5MB");

  configFromFile = {
      {S3Config::baseConfigKey(S3Config::Keys::kMultipartMinPartSize), "10GB"}};
  configBase = std::make_shared<config::ConfigBase>(std::move(configFromFile));
  VELOX_ASSERT_THROW(
      S3Config("bucket", configBase),
      "The min-part-size S3 configuration must not exceed 5GB");
}

TEST(S3ConfigTest, minPartSizeValidationBucketConfig) {
  // Test that setting bucket-specific min-part-size below 5MB throws an error.
  std::string_view bucket = "testbucket";
  std::unordered_map<std::string, std::string> configFromFile = {
      {S3Config::bucketConfigKey(S3Config::Keys::kMultipartMinPartSize, bucket),
       "3MB"}};
  auto configBase =
      std::make_shared<config::ConfigBase>(std::move(configFromFile));

  VELOX_ASSERT_THROW(
      S3Config(bucket, configBase),
      "The min-part-size S3 configuration must exceed 5MB");

  configFromFile = {
      {S3Config::bucketConfigKey(S3Config::Keys::kMultipartMinPartSize, bucket),
       "10GB"}};
  configBase = std::make_shared<config::ConfigBase>(std::move(configFromFile));

  VELOX_ASSERT_THROW(
      S3Config(bucket, configBase),
      "The min-part-size S3 configuration must not exceed 5GB");
}

} // namespace
} // namespace facebook::velox::filesystems
