#!/usr/bin/env bash
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

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"

mode="${1:-fast}"
build_dir="${2:-${repo_root}/_build/release}"
test_binary="${build_dir}/velox/experimental/ucx-exchange/tests/ucx_communicator_progress_fairness_test"
timeout_command="${TIMEOUT_COMMAND:-timeout}"

if [[ ! -x "${test_binary}" ]]; then
  echo "Velox fairness test is missing or not executable: ${test_binary}" >&2
  echo "Build it first with:" >&2
  echo "  cmake --build ${build_dir} --target ucx_communicator_progress_fairness_test -j\$(nproc)" >&2
  exit 2
fi

case "${mode}" in
  fast)
    exec "${timeout_command}" 90s "${test_binary}" --gtest_break_on_failure
    ;;

  stress | stress-blocking)
    export VELOX_UCX_TEST_FAIRNESS_BATCH_SIZE="${VELOX_UCX_TEST_FAIRNESS_BATCH_SIZE:-32768}"
    export VELOX_UCX_TEST_FAIRNESS_ENDPOINT_COUNT="${VELOX_UCX_TEST_FAIRNESS_ENDPOINT_COUNT:-256}"
    export VELOX_UCX_TEST_FAIRNESS_MAX_ENDPOINT_LATENCY_MS="${VELOX_UCX_TEST_FAIRNESS_MAX_ENDPOINT_LATENCY_MS:-2000}"
    if [[ "${mode}" == "stress-blocking" ]]; then
      export VELOX_UCX_BLOCKING_POLLING=1
    fi
    gtest_repeat="${GTEST_REPEAT:-10}"

    exec "${timeout_command}" 300s "${test_binary}" \
      "--gtest_repeat=${gtest_repeat}" \
      --gtest_break_on_failure
    ;;

  *)
    echo "Usage: $0 [fast|stress|stress-blocking] [build-directory]" >&2
    exit 2
    ;;
esac
