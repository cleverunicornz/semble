"""Tests for benchmarks.model2vec_data.prepare (tiny local fixtures; no network, no Parquet)."""

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.model2vec_data.prepare import (
    ConfigError,
    PreparationError,
    PrepareConfig,
    ResolvedFile,
    check_output_dir,
    main,
    prepare_sample,
    resolve_inputs,
    split_for_group,
)
from benchmarks.model2vec_data.sources import ROLE_EVAL, ROLE_TRAIN, load_pins

PINS = load_pins()
MAX_BYTES = 1 << 20
OUTPUT_NAMES = ("pairs.train.jsonl", "pairs.eval.jsonl", "corpus.train.jsonl", "corpus.eval.jsonl")


def make_config(**overrides: object) -> PrepareConfig:
    """Build a PrepareConfig with plumbing defaults, applying overrides."""
    values: dict[str, object] = {
        "seed": 0,
        "eval_fraction": 0.1,
        "caps": {"rust": 1000, "typescript": 100, "swift": 100, "kotlin": 100},
        "max_field_bytes": MAX_BYTES,
        "max_scan_rows": 100_000,
        "kotlin_max_shards": 1,
    }
    values.update(overrides)
    return PrepareConfig(**values)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows as a JSONL fixture file."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_file(directory: Path, name: str, rows: list[dict[str, object]], role: str = ROLE_TRAIN) -> ResolvedFile:
    """Write a JSONL fixture and wrap it as an explicit resolved input file."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    write_jsonl(path, rows)
    return ResolvedFile(path=path, pinned_name=name, role=role, resolution="explicit")


def rust_rows(
    count: int,
    crates: tuple[str, ...] = ("serde", "tokio", "clap", "anyhow"),
    categories: tuple[str, ...] = ("code_search", "code_generation", "code_summarization"),
) -> list[dict[str, object]]:
    """Build varied valid Rust rows across crates and supported categories."""
    rows: list[dict[str, object]] = []
    for i in range(count):
        crate = crates[i % len(crates)]
        category = categories[i % len(categories)]
        if category == "code_search":
            input_data: dict[str, object] = {"query": f"q{i} find handler"}
            output_data: dict[str, object] = {"code_snippet": f"fn handle_{i}() {{ {crate}(); }}"}
        elif category == "code_generation":
            input_data = {"description": f"q{i} write a parser"}
            output_data = {"code": f"fn gen_{i}_{crate}() {{}}"}
        else:
            input_data = {"code": f"fn sum_{i}_{crate}() {{}}"}
            output_data = {"summary": f"q{i} summarizes code"}
        rows.append(
            {
                "crate_name": crate,
                "task_category": category,
                "input_data": json.dumps(input_data),
                "output_data": json.dumps(output_data),
                "test": "null",
            }
        )
    return rows


def rust_row_with(crate: str, query: str, code: str) -> dict[str, object]:
    """Build a single code_search Rust row with explicit crate/query/code."""
    return {
        "crate_name": crate,
        "task_category": "code_search",
        "input_data": json.dumps({"query": query}),
        "output_data": json.dumps({"code_snippet": code}),
        "test": "null",
    }


def ts_rows(count: int) -> list[dict[str, object]]:
    """Build TypeScript pair rows across three repos."""
    return [
        {
            "docstring": f"Helper {i}.",
            "code": f"export function f{i}() {{ return {i}; }}",
            "repo": f"org/repo-{i % 3}",
            "func_name": f"f{i}",
            "path": f"src/f{i}.ts",
            "url": f"https://example.com/{i}",
            "license": "MIT",
        }
        for i in range(count)
    ]


def swift_rows(count: int) -> list[dict[str, object]]:
    """Build Swift pair rows with test-assertion ground_truth kept out of code."""
    return [
        {
            "id": f"swift-{i}",
            "translated_problem": f"Problem {i}.",
            "translated_solution": f"func solve{i}() -> Int {{ {i} }}",
            "ground_truth": json.dumps([f"assert solve{i}() == {i}"]),
            "translated_test_cases": json.dumps([{"input": "[]", "output": f"{i}"}]),
        }
        for i in range(count)
    ]


def kotlin_rows(count: int) -> list[dict[str, object]]:
    """Build Kotlin corpus rows with the verified schema: int64 repo_id or owner/name."""
    rows: list[dict[str, object]] = []
    for i in range(count):
        if i % 2 == 0:
            identity: dict[str, object] = {"repo_id": 42100 + (i % 3)}
        else:
            identity = {"owner": "kotlin", "name": f"module-{i % 2}"}
        rows.append(
            {
                "content": f"fun a{i}() {{\n    // doc {i}\n}}",
                "path": f"src/A{i}.kt",
                "commit_sha": f"sha{i}",
                "license": "Apache-2.0",
                **identity,
            }
        )
    return rows


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read a JSONL output file as parsed records."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def crates_routed_to(
    split: str, count: int, seed: int = 0, fraction: float = 0.1, exclude: set[str] | None = None
) -> list[str]:
    """Pick crate names deterministically routed to the requested split."""
    picked: list[str] = []
    index = 0
    skip = exclude or set()
    while len(picked) < count:
        crate = f"crate{index}"
        index += 1
        if crate in skip:
            continue
        if split_for_group(seed, f"crate:{crate}", fraction) == split:
            picked.append(crate)
    return picked


def numeric_repo_routed_to(split: str, seed: int = 0, fraction: float = 0.1) -> int:
    """Pick a numeric Kotlin repo_id deterministically routed to the requested split."""
    index = 0
    while True:
        if split_for_group(seed, f"repo:{index}", fraction) == split:
            return index
        index += 1


def train_side_contents(out: Path) -> set[str]:
    """Collect all training-side code/text across the pairs and corpus outputs."""
    contents = {record["code"] for record in read_jsonl(out / "pairs.train.jsonl")}
    contents |= {record["text"] for record in read_jsonl(out / "corpus.train.jsonl")}
    return contents


def eval_side_contents(out: Path) -> set[str]:
    """Collect all evaluation-side code/text across the pairs and corpus outputs."""
    contents = {record["code"] for record in read_jsonl(out / "pairs.eval.jsonl")}
    contents |= {record["text"] for record in read_jsonl(out / "corpus.eval.jsonl")}
    return contents


def test_rust_only_run_writes_outputs_and_verified_manifest(tmp_path: Path) -> None:
    """A Rust-only run produces all outputs plus a manifest with correct hashes."""
    inputs = tmp_path / "in"
    files = {
        "rust": [
            make_file(inputs, "rust-train.jsonl", rust_rows(24)),
            make_file(
                inputs,
                "rust-test.jsonl",
                rust_rows(6, crates=("upstream0", "upstream1", "upstream2")),
                role=ROLE_EVAL,
            ),
        ]
    }
    out = tmp_path / "out"
    manifest = prepare_sample(make_config(), files, [], out, PINS)

    assert sorted(path.name for path in out.iterdir()) == sorted([*OUTPUT_NAMES, "manifest.json"])
    assert manifest["schema_version"] == 1
    for name, info in manifest["outputs"].items():
        digest = hashlib.sha256((out / name).read_bytes()).hexdigest()
        assert digest == info["sha256"]
        assert info["rows"] == len(read_jsonl(out / name))

    rust = manifest["sources"]["rust"]
    assert rust["pin"]["dataset"] == PINS["rust"].dataset
    assert rust["pin"]["revision"] == PINS["rust"].revision
    assert manifest["disabled_sources"] == ["typescript", "swift", "kotlin"]
    for entry in rust["inputs"]:
        assert entry["sha256"] == hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
        assert entry["resolution"] == "explicit"

    pairs = read_jsonl(out / "pairs.train.jsonl") + read_jsonl(out / "pairs.eval.jsonl")
    assert len(pairs) == 30
    for pair in pairs:
        assert pair["language"] == "rust"
        assert pair["category"] in {"code_search", "code_generation", "code_summarization"}
        assert pair["query"] and pair["code"]
        assert pair["group"]["strength"] == "crate"
        assert pair["source"]["revision"] == PINS["rust"].revision
    upstream_eval = {record["id"] for record in read_jsonl(out / "pairs.eval.jsonl")}
    assert any(record_id.startswith("rust:rust-test.jsonl:") for record_id in upstream_eval)
    train_groups = {record["group"]["id"] for record in read_jsonl(out / "pairs.train.jsonl")}
    eval_groups = {record["group"]["id"] for record in read_jsonl(out / "pairs.eval.jsonl")}
    assert train_groups.isdisjoint(eval_groups)
    assert train_side_contents(out).isdisjoint(eval_side_contents(out))

    by_category = rust["counts"]["by_category"]
    assert sum(entry["scanned"] for entry in by_category.values()) == 30
    assert set(by_category) == {"code_search", "code_generation", "code_summarization"}


def test_corpus_is_derived_from_pairs_with_matching_split(tmp_path: Path) -> None:
    """Corpus rows are the selected pairs' code, split-locked to their pair."""
    inputs = tmp_path / "in"
    files = {"rust": [make_file(inputs, "rust-train.jsonl", rust_rows(15))]}
    out = tmp_path / "out"
    prepare_sample(make_config(), files, [], out, PINS)

    train_pairs = read_jsonl(out / "pairs.train.jsonl")
    eval_pairs = read_jsonl(out / "pairs.eval.jsonl")
    train_corpus = read_jsonl(out / "corpus.train.jsonl")
    eval_corpus = read_jsonl(out / "corpus.eval.jsonl")

    assert {record["text"] for record in train_corpus} == {record["code"] for record in train_pairs}
    assert {record["text"] for record in eval_corpus} == {record["code"] for record in eval_pairs}
    assert all(record["source"]["derived_from_pair"] is True for record in train_corpus + eval_corpus)
    train_codes = {record["code"] for record in train_pairs}
    eval_codes = {record["code"] for record in eval_pairs}
    assert train_codes.isdisjoint(eval_codes)


def test_selection_is_deterministic_hash_mixed_and_seed_sensitive(tmp_path: Path) -> None:
    """Same inputs and seed reproduce byte-identical outputs; sampling is not first-N."""
    inputs = tmp_path / "in"
    rows = rust_rows(40, crates=tuple(f"mix{i}" for i in range(8)))
    files = {"rust": [make_file(inputs, "rust-train.jsonl", rows)]}
    config = make_config(eval_fraction=0.0, caps={"rust": 10, "typescript": 100, "swift": 100, "kotlin": 100})

    out_a, out_b, out_c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    prepare_sample(config, files, [], out_a, PINS)
    prepare_sample(config, files, [], out_b, PINS)
    for name in (*OUTPUT_NAMES, "manifest.json"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()

    train_ids = [record["id"] for record in read_jsonl(out_a / "pairs.train.jsonl")]
    assert len(train_ids) == 10
    assert train_ids != [f"rust:rust-train.jsonl:{i}" for i in range(10)]

    prepare_sample(
        make_config(seed=1, eval_fraction=0.0, caps={"rust": 10, "typescript": 100, "swift": 100, "kotlin": 100}),
        files,
        [],
        out_c,
        PINS,
    )
    seeded_ids = [record["id"] for record in read_jsonl(out_c / "pairs.train.jsonl")]
    assert set(seeded_ids) != set(train_ids)


def test_cap_limits_selected_pairs(tmp_path: Path) -> None:
    """Eligible rows beyond the cap are cut down to exactly the cap."""
    inputs = tmp_path / "in"
    files = {"rust": [make_file(inputs, "rust-train.jsonl", rust_rows(30))]}
    config = make_config(eval_fraction=0.0, caps={"rust": 7, "typescript": 100, "swift": 100, "kotlin": 100})
    manifest = prepare_sample(config, files, [], tmp_path / "out", PINS)

    rust = manifest["sources"]["rust"]
    assert rust["counts"]["selected"] == {"train": 7, "eval": 0}
    assert rust["counts"]["eligible"] == 30
    assert rust["shortfalls"] == {}
    assert len(read_jsonl(tmp_path / "out" / "pairs.train.jsonl")) == 7


def test_shortfall_reported_when_eligible_below_budget(tmp_path: Path) -> None:
    """Fewer eligible rows than requested are selected without failure and reported."""
    inputs = tmp_path / "in"
    files = {"rust": [make_file(inputs, "rust-train.jsonl", rust_rows(5))]}
    config = make_config(eval_fraction=0.0, caps={"rust": 1000, "typescript": 100, "swift": 100, "kotlin": 100})
    manifest = prepare_sample(config, files, [], tmp_path / "out", PINS)

    rust = manifest["sources"]["rust"]
    assert rust["counts"]["selected"]["train"] == 5
    assert rust["shortfalls"] == {"train": 995}


def test_crate_groups_never_cross_splits(tmp_path: Path) -> None:
    """Splits are crate-disjoint and match the deterministic group split rule."""
    inputs = tmp_path / "in"
    crates = tuple(f"split{i}" for i in range(12))
    files = {"rust": [make_file(inputs, "rust-train.jsonl", rust_rows(60, crates=crates))]}
    config = make_config(eval_fraction=0.5, caps={"rust": 100, "typescript": 100, "swift": 100, "kotlin": 100})
    out = tmp_path / "out"
    prepare_sample(config, files, [], out, PINS)

    train_groups = {record["group"]["id"] for record in read_jsonl(out / "pairs.train.jsonl")}
    eval_groups = {record["group"]["id"] for record in read_jsonl(out / "pairs.eval.jsonl")}
    assert train_groups.isdisjoint(eval_groups)
    assert train_groups | eval_groups == {f"crate:{crate}" for crate in crates}
    for group in train_groups:
        assert split_for_group(0, group, 0.5) == ROLE_TRAIN
    for group in eval_groups:
        assert split_for_group(0, group, 0.5) == ROLE_EVAL


def test_upstream_eval_inputs_are_eval_only(tmp_path: Path) -> None:
    """Rows from eval-role files never land in the train output."""
    inputs = tmp_path / "in"
    # All train-file crates are chosen to route to train, so eval is exactly the upstream file.
    train_only_crates = tuple(crates_routed_to(ROLE_TRAIN, 4))
    files = {
        "rust": [
            make_file(inputs, "rust-train.jsonl", rust_rows(10, crates=train_only_crates)),
            make_file(inputs, "rust-test.jsonl", rust_rows(6), role=ROLE_EVAL),
        ]
    }
    out = tmp_path / "out"
    prepare_sample(make_config(), files, [], out, PINS)

    train_ids = [record["id"] for record in read_jsonl(out / "pairs.train.jsonl")]
    eval_ids = [record["id"] for record in read_jsonl(out / "pairs.eval.jsonl")]
    assert len(train_ids) == 10
    assert len(eval_ids) == 6
    assert all(record_id.startswith("rust:rust-test.jsonl:") for record_id in eval_ids)
    assert all(not record_id.startswith("rust:rust-test.jsonl:") for record_id in train_ids)


def test_exact_pair_and_cross_split_code_dedup(tmp_path: Path) -> None:
    """Held-out eval code wins; conflicting train pairs are excluded, not the reverse."""
    inputs = tmp_path / "in"
    train_crates = crates_routed_to(ROLE_TRAIN, 2)
    duplicate_code = "fn dup() {}"
    train_rows = [
        rust_row_with(train_crates[0], "same q", duplicate_code),
        rust_row_with(train_crates[1], "same q", duplicate_code),
        rust_row_with(train_crates[0], "other q", duplicate_code),
    ]
    eval_rows = [rust_row_with("anyhow", "eval q", duplicate_code)]
    files = {
        "rust": [
            make_file(inputs, "rust-train.jsonl", train_rows),
            make_file(inputs, "rust-test.jsonl", eval_rows, role=ROLE_EVAL),
        ]
    }
    out = tmp_path / "out"
    manifest = prepare_sample(make_config(), files, [], out, PINS)

    train_pairs = read_jsonl(out / "pairs.train.jsonl")
    eval_pairs = read_jsonl(out / "pairs.eval.jsonl")
    assert train_pairs == []
    assert [record["query"] for record in eval_pairs] == ["eval q"]

    rust = manifest["sources"]["rust"]
    assert rust["duplicates"] == {
        "duplicate_code": 3,
        "cross_split_duplicate": 3,
    }
    assert read_jsonl(out / "corpus.train.jsonl") == []
    assert {record["text"] for record in read_jsonl(out / "corpus.eval.jsonl")} == {duplicate_code}
    assert train_side_contents(out).isdisjoint(eval_side_contents(out))


def test_corpus_text_dedup_including_kotlin_cross_split(tmp_path: Path) -> None:
    """Kotlin dup texts drop; an eval-routed Kotlin doc claims its text over a train pair."""
    inputs = tmp_path / "in"
    train_crates = crates_routed_to(ROLE_TRAIN, 1)
    eval_repo = numeric_repo_routed_to(ROLE_EVAL)
    train_repo = numeric_repo_routed_to(ROLE_TRAIN)
    shared_code = "fun shared() {}"
    kotlin_docs = [
        {"content": "fun only_kotlin() {}", "repo_id": train_repo, "path": "A.kt"},
        {"content": "fun only_kotlin() {}", "repo_id": train_repo, "path": "A-copy.kt"},
        {"content": shared_code, "repo_id": eval_repo, "path": "B.kt"},
    ]
    files = {
        "rust": [make_file(inputs, "rust-train.jsonl", [rust_row_with(train_crates[0], "q", shared_code)])],
        "kotlin": [make_file(inputs, "kotlin-shard.jsonl", kotlin_docs)],
    }
    out = tmp_path / "out"
    manifest = prepare_sample(make_config(), files, [], out, PINS)

    assert {record["text"] for record in read_jsonl(out / "corpus.train.jsonl")} == {"fun only_kotlin() {}"}
    assert {record["text"] for record in read_jsonl(out / "corpus.eval.jsonl")} == {shared_code}
    assert read_jsonl(out / "pairs.train.jsonl") == []
    assert read_jsonl(out / "pairs.eval.jsonl") == []
    assert train_side_contents(out).isdisjoint(eval_side_contents(out))

    assert manifest["sources"]["rust"]["duplicates"] == {"cross_kind_eval_conflict": 1}
    kotlin = manifest["sources"]["kotlin"]
    assert kotlin["duplicates"] == {"duplicate_text": 1}
    assert kotlin["counts"]["selected"] == {"train": 1, "eval": 1}


def test_shared_group_between_train_and_upstream_eval_stays_disjoint(tmp_path: Path) -> None:
    """A crate in both train and upstream eval inputs is owned by eval entirely."""
    inputs = tmp_path / "in"
    shared = crates_routed_to(ROLE_TRAIN, 1)[0]  # would land in train without reservation
    other = crates_routed_to(ROLE_TRAIN, 1, exclude={shared})[0]
    train_rows = [
        rust_row_with(shared, "train q", "fn shared_train() {}"),
        rust_row_with(other, "other q", "fn other_train() {}"),
    ]
    eval_rows = [rust_row_with(shared, "eval q", "fn shared_eval() {}")]
    files = {
        "rust": [
            make_file(inputs, "rust-train.jsonl", train_rows),
            make_file(inputs, "rust-test.jsonl", eval_rows, role=ROLE_EVAL),
        ]
    }
    out = tmp_path / "out"
    manifest = prepare_sample(make_config(), files, [], out, PINS)

    train_groups = {record["group"]["id"] for record in read_jsonl(out / "pairs.train.jsonl")}
    eval_groups = {record["group"]["id"] for record in read_jsonl(out / "pairs.eval.jsonl")}
    assert train_groups == {f"crate:{other}"}
    assert eval_groups == {f"crate:{shared}"}
    assert train_groups.isdisjoint(eval_groups)
    eval_codes = [record["code"] for record in read_jsonl(out / "pairs.eval.jsonl")]
    assert sorted(eval_codes) == ["fn shared_eval() {}", "fn shared_train() {}"]

    grouping = manifest["sources"]["rust"]["grouping"]
    assert grouping["reserved_eval_groups"] == 1
    assert grouping["train_rows_routed_to_reserved_groups"] == 1
    assert train_side_contents(out).isdisjoint(eval_side_contents(out))


def test_eval_pair_conflicts_with_kotlin_train_doc(tmp_path: Path) -> None:
    """A Kotlin TRAIN doc matching eval pair code is excluded; eval keeps its copy."""
    inputs = tmp_path / "in"
    conflict = "fun conflict() {}"
    train_repo = numeric_repo_routed_to(ROLE_TRAIN)
    files = {
        "rust": [
            make_file(
                inputs,
                "rust-test.jsonl",
                [rust_row_with("anyhow", "eval q", conflict)],
                role=ROLE_EVAL,
            )
        ],
        "kotlin": [
            make_file(
                inputs,
                "kotlin-shard.jsonl",
                [
                    {"content": conflict, "repo_id": train_repo, "path": "A.kt"},
                    {"content": "fun distinct() {}", "repo_id": train_repo, "path": "B.kt"},
                ],
            )
        ],
    }
    out = tmp_path / "out"
    manifest = prepare_sample(make_config(), files, [], out, PINS)

    assert {record["code"] for record in read_jsonl(out / "pairs.eval.jsonl")} == {conflict}
    assert {record["text"] for record in read_jsonl(out / "corpus.train.jsonl")} == {"fun distinct() {}"}
    assert conflict not in train_side_contents(out)
    assert conflict in eval_side_contents(out)
    assert train_side_contents(out).isdisjoint(eval_side_contents(out))
    assert manifest["sources"]["kotlin"]["duplicates"] == {"cross_kind_eval_conflict": 1}


def test_train_pair_conflicts_with_kotlin_eval_doc(tmp_path: Path) -> None:
    """A train pair matching an eval-routed Kotlin doc is excluded; corpus.eval keeps it."""
    inputs = tmp_path / "in"
    conflict = "fun clash() {}"
    train_crate = crates_routed_to(ROLE_TRAIN, 1)[0]
    train_repo = numeric_repo_routed_to(ROLE_TRAIN)
    eval_repo = numeric_repo_routed_to(ROLE_EVAL)
    files = {
        "rust": [make_file(inputs, "rust-train.jsonl", [rust_row_with(train_crate, "q", conflict)])],
        "kotlin": [
            make_file(
                inputs,
                "kotlin-shard.jsonl",
                [
                    {"content": conflict, "repo_id": eval_repo, "path": "A.kt"},
                    {"content": "fun distinct() {}", "repo_id": train_repo, "path": "B.kt"},
                ],
            )
        ],
    }
    out = tmp_path / "out"
    manifest = prepare_sample(make_config(), files, [], out, PINS)

    assert read_jsonl(out / "pairs.train.jsonl") == []
    assert {record["text"] for record in read_jsonl(out / "corpus.eval.jsonl")} == {conflict}
    assert {record["text"] for record in read_jsonl(out / "corpus.train.jsonl")} == {"fun distinct() {}"}
    assert conflict not in train_side_contents(out)
    assert train_side_contents(out).isdisjoint(eval_side_contents(out))
    assert manifest["sources"]["rust"]["duplicates"] == {"cross_kind_eval_conflict": 1}


def test_optional_typescript_shortfall_does_not_block_rust(tmp_path: Path) -> None:
    """A short optional TypeScript sample reports shortfall while Rust completes."""
    inputs = tmp_path / "in"
    files = {
        "rust": [make_file(inputs, "rust-train.jsonl", rust_rows(12))],
        "typescript": [make_file(inputs, "ts-train.jsonl", ts_rows(5))],
    }
    manifest = prepare_sample(make_config(), files, [], tmp_path / "out", PINS)

    typescript = manifest["sources"]["typescript"]
    selected = typescript["counts"]["selected"]
    assert selected["train"] + selected["eval"] == 5
    assert typescript["shortfalls"]
    assert manifest["sources"]["rust"]["counts"]["eligible"] == 12
    ts_pairs = read_jsonl(tmp_path / "out" / "pairs.train.jsonl") + read_jsonl(tmp_path / "out" / "pairs.eval.jsonl")
    assert sum(1 for record in ts_pairs if record["language"] == "typescript") == 5
    assert sum(1 for record in ts_pairs if record["language"] == "rust") == 12


def test_swift_weak_grouping_visible_in_manifest(tmp_path: Path) -> None:
    """Swift's missing repository identity is explicit in the manifest."""
    inputs = tmp_path / "in"
    files = {
        "rust": [make_file(inputs, "rust-train.jsonl", rust_rows(4))],
        "swift": [make_file(inputs, "swift-train.jsonl", swift_rows(8))],
    }
    manifest = prepare_sample(make_config(), files, [], tmp_path / "out", PINS)

    grouping = manifest["sources"]["swift"]["grouping"]
    assert "NOT guaranteed" in grouping["note"]
    selected = manifest["sources"]["swift"]["counts"]["selected"]
    assert grouping["strengths_present"] == {"row_id": selected["train"] + selected["eval"]}
    assert manifest["sources"]["swift"]["pin"]["revision"] == PINS["swift"].revision
    swift_pairs = read_jsonl(tmp_path / "out" / "pairs.train.jsonl") + read_jsonl(tmp_path / "out" / "pairs.eval.jsonl")
    for record in swift_pairs:
        if record["language"] == "swift":
            assert "assert" not in record["code"]
            assert record["group"]["strength"] == "row_id"


def test_kotlin_partial_shard_coverage_reported(tmp_path: Path) -> None:
    """Using one KStack shard reports coverage against all pinned shards."""
    inputs = tmp_path / "in"
    files = {
        "rust": [make_file(inputs, "rust-train.jsonl", rust_rows(4))],
        "kotlin": [make_file(inputs, "kotlin-shard.jsonl", kotlin_rows(5))],
    }
    manifest = prepare_sample(make_config(), files, [], tmp_path / "out", PINS)

    coverage = manifest["sources"]["kotlin"]["coverage"]
    assert coverage["pinned_files"] == 33
    assert coverage["used"] == 1
    assert len(coverage["missing_pinned"]) == 33
    corpus = read_jsonl(tmp_path / "out" / "corpus.train.jsonl") + read_jsonl(tmp_path / "out" / "corpus.eval.jsonl")
    kotlin_rows_out = [record for record in corpus if record["language"] == "kotlin"]
    assert len(kotlin_rows_out) == 5
    assert all(record["source"]["derived_from_pair"] is False for record in kotlin_rows_out)


def test_unusable_rust_input_fails_clearly(tmp_path: Path) -> None:
    """A Rust input with zero eligible rows raises a clear preparation error."""
    inputs = tmp_path / "in"
    garbage = [
        {
            "crate_name": "x",
            "task_category": "code_edit",
            "input_data": "{}",
            "output_data": "{}",
        }
        for _ in range(3)
    ]
    files = {"rust": [make_file(inputs, "rust-train.jsonl", garbage)]}
    with pytest.raises(PreparationError, match="0 eligible"):
        prepare_sample(make_config(), files, [], tmp_path / "out", PINS)


def test_scan_bound_truncates_input_and_is_reported(tmp_path: Path) -> None:
    """The per-file scan bound stops reading and is visible in the manifest."""
    inputs = tmp_path / "in"
    files = {"rust": [make_file(inputs, "rust-train.jsonl", rust_rows(10))]}
    config = make_config(
        eval_fraction=0.0,
        caps={"rust": 100, "typescript": 100, "swift": 100, "kotlin": 100},
        max_scan_rows=4,
    )
    manifest = prepare_sample(config, files, [], tmp_path / "out", PINS)

    rust = manifest["sources"]["rust"]
    assert rust["counts"]["scanned"] == 4
    assert rust["counts"]["eligible"] == 4
    input_info = rust["inputs"][0]
    assert input_info["rows_scanned"] == 4
    assert input_info["hit_scan_bound"] is True
    assert len(read_jsonl(tmp_path / "out" / "pairs.train.jsonl")) == 4


def test_invalid_jsonl_lines_counted_as_invalid_rows(tmp_path: Path) -> None:
    """Unparsable JSONL lines become invalid-row skips, not crashes."""
    inputs = tmp_path / "in"
    path = inputs / "rust-train.jsonl"
    inputs.mkdir(parents=True, exist_ok=True)
    good_rows = rust_rows(2)
    path.write_text(
        json.dumps(good_rows[0]) + "\n\n" + json.dumps(good_rows[1]) + "\nnot-json\n",
        encoding="utf-8",
    )
    files = {"rust": [ResolvedFile(path=path, pinned_name=path.name, role=ROLE_TRAIN, resolution="explicit")]}
    manifest = prepare_sample(make_config(), files, [], tmp_path / "out", PINS)

    rust = manifest["sources"]["rust"]
    assert rust["counts"]["scanned"] == 3
    assert rust["counts"]["skipped"] == {"invalid_row": 1}
    assert rust["counts"]["eligible"] == 2


def test_split_for_group_boundaries_and_stability() -> None:
    """Group split is stable and honors 0.0/1.0 fractions."""
    assert split_for_group(7, "crate:serde", 0.1) == split_for_group(7, "crate:serde", 0.1)
    assert split_for_group(0, "any", 0.0) == ROLE_TRAIN
    assert split_for_group(0, "any", 1.0) == ROLE_EVAL


def test_resolve_inputs_missing_rust_is_config_error(tmp_path: Path) -> None:
    """An enabled source with no local inputs raises with the pinned download command."""
    with pytest.raises(ConfigError, match="hf download"):
        resolve_inputs(PINS, input_root=tmp_path, enabled={"rust": True})


def test_resolve_inputs_rejects_bad_roles_and_missing_files(tmp_path: Path) -> None:
    """Malformed ROLE:PATH specs and missing files are configuration errors."""
    inputs = tmp_path / "in"
    inputs.mkdir()
    train_file = make_file(inputs, "rust-train.jsonl", rust_rows(1))
    with pytest.raises(ConfigError, match="role"):
        resolve_inputs(
            PINS,
            input_root=inputs,
            enabled={"rust": True},
            overrides={"rust": [f"bogus:{train_file.path}"]},
        )
    with pytest.raises(ConfigError, match="not found"):
        resolve_inputs(
            PINS,
            input_root=inputs,
            enabled={"rust": True},
            overrides={"rust": [f"train:{inputs / 'missing.jsonl'}"]},
        )


def test_resolve_inputs_kotlin_only_train_role(tmp_path: Path) -> None:
    """The corpus-only Kotlin source rejects eval-role inputs."""
    inputs = tmp_path / "in"
    inputs.mkdir()
    shard = make_file(inputs, "kotlin-shard.jsonl", kotlin_rows(1))
    with pytest.raises(ConfigError, match="corpus-only"):
        resolve_inputs(
            PINS,
            input_root=inputs,
            enabled={"rust": False, "kotlin": True},
            overrides={"kotlin": [f"eval:{shard.path}"]},
        )


def test_check_output_dir_refuses_unsafe_destinations(tmp_path: Path) -> None:
    """Existing non-empty directories and file paths are refused."""
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(ConfigError, match="not empty"):
        check_output_dir(occupied)

    as_file = tmp_path / "a-file"
    as_file.write_text("x", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a directory"):
        check_output_dir(as_file)

    fresh = tmp_path / "fresh"
    check_output_dir(fresh)
    assert not fresh.exists()


def test_cli_end_to_end_with_local_jsonl(tmp_path: Path) -> None:
    """The CLI wires config, resolution, and preparation end to end."""
    inputs = tmp_path / "in"
    train = make_file(inputs, "rust-train.jsonl", rust_rows(20))
    upstream_eval = make_file(
        inputs, "rust-test.jsonl", rust_rows(4, crates=("cli-eval0", "cli-eval1")), role=ROLE_EVAL
    )
    out = tmp_path / "out"
    code = main(
        [
            "--output",
            str(out),
            "--rust-file",
            f"train:{train.path}",
            "--rust-file",
            f"eval:{upstream_eval.path}",
            "--max-rust-pairs",
            "20",
            "--eval-fraction",
            "0",
            "--seed",
            "3",
        ]
    )
    assert code == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["seed"] == 3
    assert manifest["config"]["eval_fraction"] == 0.0
    assert manifest["config"]["caps"]["rust"] == 20
    assert manifest["config"]["caps"]["typescript"] == 100
    train_ids = [record["id"] for record in read_jsonl(out / "pairs.train.jsonl")]
    eval_ids = [record["id"] for record in read_jsonl(out / "pairs.eval.jsonl")]
    assert len(train_ids) == 20
    assert all(record_id.startswith("rust:rust-train.jsonl:") for record_id in train_ids)
    assert eval_ids == []


def test_cli_rejects_invalid_config_and_unsafe_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Invalid numeric config and non-empty output dirs exit with code 2."""
    inputs = tmp_path / "in"
    train = make_file(inputs, "rust-train.jsonl", rust_rows(2))
    assert (
        main(
            [
                "--output",
                str(tmp_path / "a"),
                "--eval-fraction",
                "1.5",
                "--rust-file",
                f"train:{train.path}",
            ]
        )
        == 2
    )
    assert "eval-fraction" in capsys.readouterr().err

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "x.txt").write_text("x", encoding="utf-8")
    assert main(["--output", str(occupied), "--rust-file", f"train:{train.path}"]) == 2
    assert "not empty" in capsys.readouterr().err
    assert not (tmp_path / "a").exists()


def test_optional_source_without_local_inputs_is_skipped(tmp_path: Path) -> None:
    """An enabled optional source with no local inputs is skipped with a note, not an error."""
    inputs = tmp_path / "in"
    rust_train = make_file(inputs, "rust-train.jsonl", rust_rows(4))
    files, notes = resolve_inputs(
        PINS,
        input_root=tmp_path / "nowhere",
        enabled={"rust": True, "typescript": True},
        overrides={"rust": [f"train:{rust_train.path}"]},
    )
    assert "typescript" not in files
    assert any("typescript" in note and "skipped" in note for note in notes)

    manifest = prepare_sample(make_config(), files, notes, tmp_path / "out", PINS)
    assert "typescript" in manifest["disabled_sources"]
    assert any("typescript" in note and "skipped" in note for note in manifest["notes"])

    with pytest.raises(ConfigError, match="not found"):
        resolve_inputs(
            PINS,
            input_root=tmp_path / "nowhere",
            enabled={"rust": True, "typescript": True},
            overrides={"rust": [f"train:{rust_train.path}"], "typescript": ["train:/missing/file.jsonl"]},
        )


def test_prepare_sample_refuses_non_empty_output_dir(tmp_path: Path) -> None:
    """The programmatic entry enforces the new-or-empty output directory rule."""
    inputs = tmp_path / "in"
    files = {"rust": [make_file(inputs, "rust-train.jsonl", rust_rows(3))]}
    out = tmp_path / "out"
    out.mkdir()
    (out / "existing.jsonl").write_text("user data\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not empty"):
        prepare_sample(make_config(), files, [], out, PINS)
    assert (out / "existing.jsonl").read_text(encoding="utf-8") == "user data\n"
