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

namespace facebook::velox::ucx_exchange {

/// Whether the UCX transport is discovered at runtime instead of being named
/// in the plan. Defaults to VELOX_UCX_DYNAMIC; enableDynamicUcx() overrides.
bool useDynamicUcx();

/// Turns discovery on or off. Call before any task is started -- the driver
/// adapters read this while operators are being chosen.
void enableDynamicUcx(bool enabled = true);

} // namespace facebook::velox::ucx_exchange
