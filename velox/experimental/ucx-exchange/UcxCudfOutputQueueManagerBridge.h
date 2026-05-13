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

#include <memory>
#include <optional>
#include <string_view>

#include "velox/core/PlanNode.h"
#include "velox/exec/OutputBuffer.h"

namespace facebook::velox::exec {
class Task;
}

namespace facebook::velox::ucx_exchange {

void initializeCudfUcxOutputQueueManagerTask(
    std::shared_ptr<exec::Task> task,
    core::PartitionedOutputNode::Kind kind,
    int numDestinations,
    int numDrivers);

void updateCudfUcxOutputBuffers(
    std::string_view taskId,
    int numBuffers,
    bool noMoreBuffers);

std::optional<exec::OutputBuffer::Stats> cudfUcxOutputQueueStats(
    std::string_view taskId);

void removeCudfUcxOutputQueueManagerTask(std::string_view taskId);

} // namespace facebook::velox::ucx_exchange
