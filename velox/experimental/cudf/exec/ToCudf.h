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

#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"

#include <functional>

namespace facebook::velox::cudf_velox {

class CompileState {
 public:
  CompileState(const exec::DriverFactory& driverFactory, exec::Driver& driver)
      : driverFactory_(driverFactory), driver_(driver) {}

  exec::Driver& driver() {
    return driver_;
  }

  // Get plan node by id lookup.
  core::PlanNodePtr getPlanNode(const core::PlanNodeId& id) const;

  // Replaces sequences of Operators in the Driver given at construction with
  // cuDF equivalents. Returns true if the Driver was changed.
  bool compile(bool allow_cpu_fallback);

  // Swaps a trailing [CudfToVelox, PartitionedOutput] for a sink that takes
  // cuDF vectors, when a factory is registered. Returns true if it did.
  bool replaceGpuOutputSink();

  const exec::DriverFactory& driverFactory_;
  exec::Driver& driver_;
};

/// Builds an output operator that takes cuDF vectors directly, for the sink of
/// a driver whose chain ends on the GPU.
using GpuOutputSinkFactory = std::function<std::unique_ptr<exec::Operator>(
    int32_t operatorId,
    exec::DriverCtx* ctx,
    const core::PlanNodePtr& planNode,
    uint32_t numTotalDrivers)>;

/// Registers a factory for the sink above. Without one, a cuDF chain feeding a
/// PartitionedOutput ends in CudfToVelox.
void registerGpuOutputSinkFactory(GpuOutputSinkFactory factory);

/// Registers adapter to add cuDF operators to Drivers.
void registerCudf();
void unregisterCudf();

/// Returns true if cuDF is registered.
bool cudfIsRegistered();

} // namespace facebook::velox::cudf_velox
