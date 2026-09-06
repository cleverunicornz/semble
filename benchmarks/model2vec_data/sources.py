"""Pinned Hugging Face dataset sources for Model2Vec sample preparation.

Revisions and file lists below are pins checked in parent static review;
dataset field schemas are premises from that review and are not re-verified
here. See the module README for the exact ``hf download`` commands that
reproduce the expected local layout (<input-root>/<slug>/<pinned file path>).
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

KIND_PAIRS = "pairs"
KIND_CORPUS = "corpus"

ROLE_TRAIN = "train"
ROLE_EVAL = "eval"


@dataclass(frozen=True)
class FilePin:
    """A single pinned input file and the split role it may feed."""

    name: str
    role: str


@dataclass(frozen=True)
class SourcePin:
    """A pinned dataset revision plus the adapter that normalizes its rows."""

    key: str
    slug: str
    dataset: str
    revision: str
    language: str
    kind: str
    adapter: str
    files: tuple[FilePin, ...]
    download_includes: tuple[str, ...]
    default_group_strength: str
    grouping_note: str

    @property
    def url(self) -> str:
        """Return the pinned revision URL on the Hugging Face Hub."""
        return f"https://huggingface.co/datasets/{self.dataset}/tree/{self.revision}"

    def download_command(self, local_dir: str) -> str:
        """Return the exact ``hf download`` command fetching this pin into local_dir."""
        parts = ["hf", "download", self.dataset, "--type", "dataset", "--revision", self.revision]
        for include in self.download_includes:
            parts += ["--include", include]
        parts += ["--local-dir", local_dir]
        return " ".join(shlex.quote(part) for part in parts)


def _kotlin_shards() -> tuple[FilePin, ...]:
    """Return all 33 pinned KStack train shard names (corpus-only source)."""
    return tuple(FilePin(f"data/train-{i:05d}-of-00033.parquet", ROLE_TRAIN) for i in range(33))


PINNED_SOURCES: dict[str, SourcePin] = {
    "rust": SourcePin(
        key="rust",
        slug="strandset-rust",
        dataset="Fortytwo-Network/Strandset-Rust-v1",
        revision="0a8d223302712a2b34a6ad4ce1fd679031894b3d",
        language="rust",
        kind=KIND_PAIRS,
        adapter="rust_pairs",
        files=(
            FilePin("data/train-00000-of-00001.parquet", ROLE_TRAIN),
            FilePin("data/test-00000-of-00001.parquet", ROLE_EVAL),
        ),
        download_includes=("data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"),
        default_group_strength="crate",
        grouping_note=(
            "Rows are grouped by crate_name when present and splits are crate-disjoint; "
            "rows without crate_name fall back to per-row grouping. The upstream test file "
            "is eval-only and its crates are reserved: their train-file rows are routed to "
            "eval, so no crate straddles splits over the scanned rows."
        ),
    ),
    "typescript": SourcePin(
        key="typescript",
        slug="typescript-treesitter",
        dataset="Shuu12121/typescript-treesitter-dedupe-filtered-datasetsV2",
        revision="1e2fcd3764fb9126a33eaea58961925e667769f0",
        language="typescript",
        kind=KIND_PAIRS,
        adapter="typescript_pairs",
        files=(
            FilePin("data/train-00000-of-00001.parquet", ROLE_TRAIN),
            FilePin("data/validation-00000-of-00001.parquet", ROLE_EVAL),
            FilePin("data/test-00000-of-00001.parquet", ROLE_EVAL),
        ),
        download_includes=(
            "data/train-00000-of-00001.parquet",
            "data/validation-00000-of-00001.parquet",
            "data/test-00000-of-00001.parquet",
        ),
        default_group_strength="repo",
        grouping_note=(
            "Rows are grouped by the repo column when present and splits are repository-disjoint; "
            "rows without repo fall back to per-row grouping. Upstream validation/test files are "
            "eval-only and their repos are reserved: their train-file rows are routed to eval, "
            "so no repo straddles splits over the scanned rows."
        ),
    ),
    "swift": SourcePin(
        key="swift",
        slug="rlvr-code-swift",
        dataset="saurabh5/rlvr-code-data-Swift",
        revision="e0c853088c2629d4879e2cd3f2cd4c77bd52d130",
        language="swift",
        kind=KIND_PAIRS,
        adapter="swift_pairs",
        files=(
            FilePin("data/train-00000-of-00002.parquet", ROLE_TRAIN),
            FilePin("data/train-00001-of-00002.parquet", ROLE_TRAIN),
        ),
        download_includes=("data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"),
        default_group_strength="row_id",
        grouping_note=(
            "No repository or crate grouping field exists in this dataset: groups are per source id, "
            "splits are per row, and repository-disjointness is NOT guaranteed. Exact-content "
            "deduplication is applied instead."
        ),
    ),
    "kotlin": SourcePin(
        key="kotlin",
        slug="kstack",
        dataset="JetBrains/KStack",
        revision="425774a783d87159d103d65b79ce37f58d0697bb",
        language="kotlin",
        kind=KIND_CORPUS,
        adapter="kotlin_corpus",
        files=_kotlin_shards(),
        # Deliberately pins only the first shard: the default preparation scans one
        # shard (see --kotlin-max-shards) so the 5.9 GB corpus is never pulled by accident.
        download_includes=("data/train-00000-of-00033.parquet",),
        default_group_strength="repo",
        grouping_note=(
            "Corpus-only source (raw Kotlin files, no query/answer pairs). Grouped by repo_id or "
            "owner/name when present; rows without either fall back to per-row grouping."
        ),
    ),
}


def load_pins() -> dict[str, SourcePin]:
    """Return a fresh mapping of all pinned sources keyed by source key."""
    return dict(PINNED_SOURCES)
