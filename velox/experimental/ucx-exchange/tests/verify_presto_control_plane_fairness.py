#!/usr/bin/env python3
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

"""Validate a Presto result directory for UCXX control-plane starvation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Signature:
    name: str
    expression: re.Pattern[str]


SIGNATURES = (
    Signature(
        "ucp_ep_create timeout",
        re.compile(r"Timeout waiting for ucp_ep_create", re.IGNORECASE),
    ),
    Signature(
        "CPU UCX source connection failure",
        re.compile(r"Failed to connect CPU UCX exchange source", re.IGNORECASE),
    ),
    Signature(
        "native-worker/HTTP fallback diagnostic",
        re.compile(
            r"remote split must point at a native worker exposing the CPU UCX listener",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class LogMatch:
    signature: str
    path: Path
    line_number: int
    line: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail when a Presto benchmark result contains query failures or "
            "UCXX/CPU-UCX control-plane starvation signatures."
        )
    )
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--max-reported-matches",
        type=int,
        default=30,
        help="Maximum matching log lines to print (default: 30).",
    )
    return parser.parse_args()


def query_failures(result: object) -> list[str]:
    failures: list[str] = []
    if not isinstance(result, dict):
        return failures

    for benchmark_name, benchmark_result in result.items():
        if not isinstance(benchmark_result, dict):
            continue
        failed_queries = benchmark_result.get("failed_queries")
        if not failed_queries:
            continue
        if isinstance(failed_queries, dict):
            for query_name, reason in sorted(failed_queries.items()):
                failures.append(f"{benchmark_name}/{query_name}: {reason}")
        else:
            failures.append(f"{benchmark_name}: {failed_queries}")
    return failures


def candidate_logs(result_dir: Path) -> Iterable[Path]:
    names = ("coord.log", "cli.log", "benchmark_result.txt")
    paths = {result_dir / name for name in names}
    for pattern in ("worker_*.log", "*.out", "*.err"):
        paths.update(result_dir.glob(pattern))
    return sorted(path for path in paths if path.is_file())


def scan_logs(result_dir: Path) -> tuple[list[LogMatch], dict[str, int]]:
    matches: list[LogMatch] = []
    counts = {signature.name: 0 for signature in SIGNATURES}

    for path in candidate_logs(result_dir):
        try:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for line_number, line in enumerate(stream, start=1):
                    for signature in SIGNATURES:
                        if not signature.expression.search(line):
                            continue
                        counts[signature.name] += 1
                        matches.append(
                            LogMatch(
                                signature.name,
                                path,
                                line_number,
                                line.rstrip(),
                            )
                        )
        except OSError as error:
            raise RuntimeError(f"could not read {path}: {error}") from error

    return matches, counts


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    result_json = result_dir / "benchmark_result.json"

    if not result_dir.is_dir():
        print(f"ERROR: result directory does not exist: {result_dir}", file=sys.stderr)
        return 2
    if not result_json.is_file():
        print(f"ERROR: benchmark result is missing: {result_json}", file=sys.stderr)
        return 2
    if args.max_reported_matches < 0:
        print("ERROR: --max-reported-matches must be non-negative", file=sys.stderr)
        return 2

    try:
        with result_json.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
        failures = query_failures(result)
        matches, counts = scan_logs(result_dir)
    except (OSError, json.JSONDecodeError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if not failures and not matches:
        scanned = sum(1 for _ in candidate_logs(result_dir))
        print(
            f"PASS: {result_dir.name}: no query failures or control-plane "
            f"starvation signatures in {scanned} log files"
        )
        return 0

    print(f"FAIL: {result_dir.name}")
    if failures:
        print(f"  query failures: {len(failures)}")
        for failure in failures:
            print(f"    {failure}")

    if matches:
        print(f"  control-plane signatures: {len(matches)}")
        for name, count in counts.items():
            if count:
                print(f"    {name}: {count}")
        for match in matches[: args.max_reported_matches]:
            relative_path = match.path.relative_to(result_dir)
            print(
                f"    {relative_path}:{match.line_number} "
                f"[{match.signature}] {match.line}"
            )
        unreported = len(matches) - args.max_reported_matches
        if unreported > 0:
            print(f"    ... {unreported} additional matching lines omitted")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
