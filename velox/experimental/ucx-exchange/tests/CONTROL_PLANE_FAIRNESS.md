# Velox UCX control-plane fairness

The CPU-only and combined cuDF dependency resolvers fetch published UCXX 0.53
revision `f3e85e580fb4349240917f85f5a6260c7f3b94f6` and verify archive SHA-256
`63c5f8a9e7a84bab932b52d1dad82619c684a0383b3318914eba029d9b2fa9cd`.
Both apply the checked-in
[`control-plane-fairness.patch`](../../../../CMake/resolve_dependency_modules/ucxx/control-plane-fairness.patch).

This patch ports the four published `kjmph/ucxx` fairness commits
`a7f9228bbc9fd45b1056758f3a7067f7d5d65947`,
`35d80e0be6209f38562ea5926437f61c03ce2f19`,
`f03ab50e3a6b5832b65e8963a30966895d1f6f7b`, and
`c52841f6554c6c645745068f7b983eb371c3df14` onto that published 0.53 base.
It bounds progress-thread drains and gives endpoint control callbacks their own
priority queue. It retains upstream's serialized request submission/cancellation
fix and adapts the fairness regressions to the 0.53 builder APIs.

A `SYSTEM` UCXX installation must use this same published revision plus the
checked-in patch. An unpatched 0.53 install lacks the fairness fixes; the older
0.51 fork lacks the submission/cancellation fixes and builder APIs required by
the adaptive transport. No unpublished dependency revision or separate repository
push is required to build this branch.

## Test contract

`ucx_communicator_progress_fairness_test`:

1. starts `Communicator::run()` and waits on its readiness future;
2. creates a separate delayed-submission peer worker;
3. establishes a rolling window of bidirectional tag traffic, replenishing
   each completed slot, and waits for confirmed transfer progress;
4. creates uncached endpoints on the communicator's production UCXX worker;
5. requires each late endpoint to complete below the configured deadline;
6. transfers data through every late endpoint; and
7. verifies every validation callback fires exactly once.

The late endpoints bypass only the communicator's endpoint cache. They still
use the exact worker and progress thread used by the exchange implementation,
so every creation executes UCXX's real `ucp_ep_create` generic-pre path.

## Build

Configure the normal CPU UCX build for this branch with
`-DVELOX_ENABLE_UCX_EXCHANGE=ON -DVELOX_BUILD_TESTING=ON`, then build:

```bash
cmake --build _build/release \
  --target ucx_communicator_progress_fairness_test \
  -j"$(nproc)"
```

Use a fresh CMake build directory for the first validation. A build directory
whose FetchContent UCXX source was populated before this change can retain the
old patch-step state.

If your build directory has a different name, substitute it in the commands
below. The executable is normally:

```text
_build/release/velox/experimental/ucx-exchange/tests/ucx_communicator_progress_fairness_test
```

The run commands below are wrapped by:

```bash
velox/experimental/ucx-exchange/tests/run_control_plane_fairness.sh fast
velox/experimental/ucx-exchange/tests/run_control_plane_fairness.sh stress
velox/experimental/ucx-exchange/tests/run_control_plane_fairness.sh stress-blocking
```

## Fast gate

```bash
timeout 90s \
  _build/release/velox/experimental/ucx-exchange/tests/ucx_communicator_progress_fairness_test \
  --gtest_break_on_failure
```

## Scale stress

```bash
timeout 300s env \
  VELOX_UCX_TEST_FAIRNESS_BATCH_SIZE=32768 \
  VELOX_UCX_TEST_FAIRNESS_ENDPOINT_COUNT=256 \
  VELOX_UCX_TEST_FAIRNESS_MAX_ENDPOINT_LATENCY_MS=2000 \
  _build/release/velox/experimental/ucx-exchange/tests/ucx_communicator_progress_fairness_test \
    --gtest_repeat=10 \
    --gtest_break_on_failure
```

Run the same test against blocking progress as a second mode:

```bash
timeout 300s env \
  VELOX_UCX_BLOCKING_POLLING=1 \
  VELOX_UCX_TEST_FAIRNESS_BATCH_SIZE=32768 \
  VELOX_UCX_TEST_FAIRNESS_ENDPOINT_COUNT=256 \
  VELOX_UCX_TEST_FAIRNESS_MAX_ENDPOINT_LATENCY_MS=2000 \
  _build/release/velox/experimental/ucx-exchange/tests/ucx_communicator_progress_fairness_test \
    --gtest_repeat=10 \
    --gtest_break_on_failure
```

`VELOX_UCX_TEST_FAIRNESS_BATCH_SIZE` controls the number of bidirectional
traffic slots kept outstanding. The test reports
`background_transfers_completed`, `late_endpoints_created`, and
`max_endpoint_latency_ms` as GoogleTest properties.

This is process-local traffic. It does not modify NIC, firmware, routing, IBM
Storage Scale, or Slurm state.

## End-to-end Presto soak

After building the CPU worker image from this Velox branch, use the existing
`velox-testing/presto/slurm/presto-nvl72` launcher. The original failure was Q1
at SF 30000 on 12 nodes, so the direct soak is:

```bash
./launch-run.sh \
  --cpu \
  --verify-cpu-ucx \
  --nodes 12 \
  --scale-factor 30000 \
  --queries 1 \
  --iterations 10
```

Then run the read-only result gate:

```bash
python3 \
  /path/to/velox/velox/experimental/ucx-exchange/tests/verify_presto_control_plane_fairness.py \
  result_dir_<jobid>
```

It fails on either `benchmark_result.json` query failures or these log
signatures:

- `Timeout waiting for ucp_ep_create`
- `Failed to connect CPU UCX exchange source`
- `remote split must point at a native worker exposing the CPU UCX listener`

The third message is included because it is the downstream error emitted after
endpoint creation fails; in this incident it did not establish that the remote
worker was actually HTTP-only.

A passing soak therefore requires all ten Q1 iterations to succeed and zero
matches for all three signatures. The validator only reads result files; it
does not contact or modify Presto, Slurm, UCX, or Storage Scale.
