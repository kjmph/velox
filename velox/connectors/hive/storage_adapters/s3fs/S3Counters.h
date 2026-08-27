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

namespace facebook::velox::filesystems {

// The number of connections open for S3 read operations.
constexpr std::string_view kMetricS3ActiveConnections{
    "velox.s3_active_connections"};

// The number of S3 upload calls that started.
constexpr std::string_view kMetricS3StartedUploads{"velox.s3_started_uploads"};

// The number of S3 upload calls that were completed.
constexpr std::string_view kMetricS3SuccessfulUploads{
    "velox.s3_successful_uploads"};

// The number of S3 upload calls that failed.
constexpr std::string_view kMetricS3FailedUploads{"velox.s3_failed_uploads"};

// The number of S3 head (metadata) calls.
constexpr std::string_view kMetricS3MetadataCalls{"velox.s3_metadata_calls"};

// The number of S3 head (metadata) calls that failed.
constexpr std::string_view kMetricS3GetMetadataErrors{
    "velox.s3_get_metadata_errors"};

// The number of retries made during S3 head (metadata) calls.
constexpr std::string_view kMetricS3GetMetadataRetries{
    "velox.s3_get_metadata_retries"};

// The number of S3 getObject calls.
constexpr std::string_view kMetricS3GetObjectCalls{"velox.s3_get_object_calls"};

// The number of S3 getObject calls that failed.
constexpr std::string_view kMetricS3GetObjectErrors{
    "velox.s3_get_object_errors"};

// The number of retries made during S3 getObject calls.
constexpr std::string_view kMetricS3GetObjectRetries{
    "velox.s3_get_object_retries"};

constexpr std::string_view kMetricS3DirectReceiveKernelTlsAttempts{
    "velox.s3_direct_receive_kernel_tls_attempts"};
constexpr std::string_view kMetricS3DirectReceiveCallerBufferAttempts{
    "velox.s3_direct_receive_caller_buffer_attempts"};
constexpr std::string_view kMetricS3DirectReceiveKernelTlsUnavailable{
    "velox.s3_direct_receive_kernel_tls_unavailable"};
constexpr std::string_view kMetricS3DirectReceiveFallbacks{
    "velox.s3_direct_receive_fallbacks"};
constexpr std::string_view kMetricS3DirectReceiveResponseRejections{
    "velox.s3_direct_receive_response_rejections"};
constexpr std::string_view kMetricS3DirectReceiveMechanismFailures{
    "velox.s3_direct_receive_mechanism_failures"};

// Curl-observed response-body bytes across all direct-receive attempts,
// including rejected responses and failed or retried transfers. These are not
// wire bytes or de-duplicated logical-read bytes.
constexpr std::string_view kMetricS3DirectReceiveReceivedBodyBytes{
    "velox.s3_direct_receive_received_body_bytes"};
// Bytes successfully committed after curl placed them in caller-owned loaned
// buffers. Attempts that later fail are still included.
constexpr std::string_view kMetricS3DirectReceiveReceivedDirectBodyBytes{
    "velox.s3_direct_receive_received_direct_body_bytes"};
// Bytes from accepted responses copied by the stream into caller-owned
// destinations. Attempts that later fail are included; rejected response
// bodies on the error-stream path are not included.
constexpr std::string_view kMetricS3DirectReceiveReceivedCopiedBodyBytes{
    "velox.s3_direct_receive_received_copied_body_bytes"};
// Subset of received direct plus copied bytes consumed for null preadv ranges.
constexpr std::string_view kMetricS3DirectReceiveReceivedDiscardedBodyBytes{
    "velox.s3_direct_receive_received_discarded_body_bytes"};

// Placement bytes from the final transport-successful attempt, recorded only
// when the enclosing S3 GetObject also succeeds. Direct plus copied equals
// body; discarded is a subset of those two placement categories.
constexpr std::string_view kMetricS3DirectReceiveSuccessfulBodyBytes{
    "velox.s3_direct_receive_successful_body_bytes"};
constexpr std::string_view kMetricS3DirectReceiveSuccessfulDirectBodyBytes{
    "velox.s3_direct_receive_successful_direct_body_bytes"};
constexpr std::string_view kMetricS3DirectReceiveSuccessfulCopiedBodyBytes{
    "velox.s3_direct_receive_successful_copied_body_bytes"};
constexpr std::string_view kMetricS3DirectReceiveSuccessfulDiscardedBodyBytes{
    "velox.s3_direct_receive_successful_discarded_body_bytes"};

} // namespace facebook::velox::filesystems
