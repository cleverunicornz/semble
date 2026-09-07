"""Materialize every pinned Semble benchmark repository without full history."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from benchmarks.data import BENCH_ROOT, RepoSpec, load_repo_specs

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse repository synchronization arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BENCH_ROOT)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _run(*args: str, cwd: Path | None = None) -> str:
    """Run one Git command and return stripped stdout."""
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def sync_one(root: Path, spec: RepoSpec) -> dict[str, Any]:
    """Fetch and detach one checkout at its exact manifest revision."""
    target = root / spec.name
    target.mkdir(parents=True, exist_ok=True)
    if not (target / ".git").exists():
        _run("git", "init", "--quiet", cwd=target)
        _run("git", "remote", "add", "origin", spec.url, cwd=target)
    else:
        remote = _run("git", "remote", "get-url", "origin", cwd=target)
        if remote != spec.url:
            raise ValueError(f"{spec.name}: origin is {remote}, expected {spec.url}")
    _run("git", "fetch", "--quiet", "--depth", "1", "origin", spec.revision, cwd=target)
    _run("git", "checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=target)
    head = _run("git", "rev-parse", "HEAD", cwd=target)
    if head != spec.revision:
        raise ValueError(f"{spec.name}: materialized {head}, expected {spec.revision}")
    return {
        "name": spec.name,
        "language": spec.language,
        "url": spec.url,
        "revision": spec.revision,
        "path": str(target),
        "benchmark_root": spec.benchmark_root,
    }


def sync_all(root: Path, jobs: int) -> list[dict[str, Any]]:
    """Synchronize all benchmark repos with bounded concurrency."""
    if jobs < 1:
        raise ValueError("jobs must be positive")
    root.mkdir(parents=True, exist_ok=True)
    specs = list(load_repo_specs().values())
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(sync_one, root, spec): spec for spec in specs}
        for future in as_completed(futures):
            record = future.result()
            LOGGER.info("Synced %s at %s", record["name"], record["revision"][:12])
            records.append(record)
    return sorted(records, key=lambda record: record["name"])


def main() -> None:
    """Synchronize and record every exact repository checkout."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    records = sync_all(args.root, args.jobs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"schema_version": 1, "root": str(args.root), "repositories": records}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
