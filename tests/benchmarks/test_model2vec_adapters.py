"""Tests for benchmarks.model2vec_data.adapters (no network, no Parquet)."""

import json

from benchmarks.model2vec_data.adapters import (
    SKIP_EMPTY_FIELD,
    SKIP_MISSING_FIELD,
    SKIP_NON_MAPPING_FIELD,
    SKIP_NON_STRING_FIELD,
    SKIP_OVERSIZED_FIELD,
    SKIP_UNPARSABLE_FIELD,
    SKIP_UNSUPPORTED_CATEGORY,
    CorpusRecord,
    PairRecord,
    Skip,
    adapt_kotlin_row,
    adapt_rust_row,
    adapt_swift_row,
    adapt_typescript_row,
    parse_serialized_mapping,
)

MAX_BYTES = 1 << 20


def _rust_row(**overrides: object) -> dict[str, object]:
    """Build a valid code_search Rust row, applying overrides last."""
    row: dict[str, object] = {
        "crate_name": "serde",
        "task_category": "code_search",
        "input_data": json.dumps({"query": "parse config file", "code_context": "// surrounding use"}),
        "output_data": json.dumps({"code_snippet": "fn parse() -> Config {\n    Config {}\n}"}),
        "test": json.dumps({"call": "parse()"}),
    }
    row.update(overrides)
    return row


def _adapt_rust(row: dict[str, object]) -> object:
    """Adapt a Rust row with a generous field limit."""
    return adapt_rust_row(row, row_id="rust:train:0", max_field_bytes=MAX_BYTES)


def test_rust_code_search_maps_query_and_snippet() -> None:
    """code_search pairs the input query with the output code_snippet."""
    outcome = _adapt_rust(_rust_row())
    assert isinstance(outcome, PairRecord)
    assert outcome.query == "parse config file"
    assert outcome.code == "fn parse() -> Config {\n    Config {}\n}"
    assert outcome.category == "code_search"
    assert outcome.group_id == "crate:serde"
    assert outcome.group_strength == "crate"
    assert outcome.provenance["crate_name"] == "serde"
    assert outcome.provenance["code_context"] == "// surrounding use"


def test_rust_code_generation_and_summarization_mappings() -> None:
    """code_generation uses description/code; code_summarization swaps query side to output."""
    generation = _adapt_rust(
        _rust_row(
            task_category="code_generation",
            input_data=json.dumps({"description": "write a parser"}),
            output_data=json.dumps({"code": "fn gen() {}"}),
        )
    )
    assert isinstance(generation, PairRecord)
    assert generation.query == "write a parser"
    assert generation.code == "fn gen() {}"

    summarization = _adapt_rust(
        _rust_row(
            task_category="code_summarization",
            input_data=json.dumps({"code": "fn sum() {}"}),
            output_data=json.dumps({"summary": "summarizes items"}),
        )
    )
    assert isinstance(summarization, PairRecord)
    assert summarization.query == "summarizes items"
    assert summarization.code == "fn sum() {}"


def test_rust_python_literal_input_fallback() -> None:
    """Input serialized as a Python dict literal parses when JSON fails."""
    outcome = _adapt_rust(_rust_row(input_data="{'query': 'literal query'}"))
    assert isinstance(outcome, PairRecord)
    assert outcome.query == "literal query"


def test_rust_unsupported_category_is_counted_not_condemned() -> None:
    """Categories outside the three verified mappings skip as unsupported, not bad."""
    outcome = _adapt_rust(_rust_row(task_category="code_edit"))
    assert isinstance(outcome, Skip)
    assert outcome.reason == SKIP_UNSUPPORTED_CATEGORY
    assert outcome.detail == "code_edit"

    missing = _adapt_rust({"crate_name": "serde", "input_data": "{}", "output_data": "{}"})
    assert isinstance(missing, Skip)
    assert missing.reason == SKIP_UNSUPPORTED_CATEGORY
    assert missing.detail == "<missing>"


def test_rust_missing_none_and_empty_fields_skip_clearly() -> None:
    """Missing, None, and blank required fields produce explicit skip reasons."""
    missing_input = _adapt_rust(_rust_row(input_data=None))
    assert isinstance(missing_input, Skip) and missing_input.reason == SKIP_MISSING_FIELD

    absent_input = _adapt_rust({"crate_name": "serde", "task_category": "code_search", "output_data": "{}"})
    assert isinstance(absent_input, Skip) and absent_input.reason == SKIP_MISSING_FIELD

    empty_query = _adapt_rust(_rust_row(input_data=json.dumps({"query": "   "})))
    assert isinstance(empty_query, Skip) and empty_query.reason == SKIP_EMPTY_FIELD

    empty_snippet = _adapt_rust(_rust_row(output_data=json.dumps({"code_snippet": ""})))
    assert isinstance(empty_snippet, Skip) and empty_snippet.reason == SKIP_EMPTY_FIELD


def test_rust_non_string_and_non_mapping_values_skip() -> None:
    """Non-string query values and non-mapping serialized fields skip explicitly."""
    non_string = _adapt_rust(_rust_row(input_data=json.dumps({"query": {"nested": True}})))
    assert isinstance(non_string, Skip) and non_string.reason == SKIP_NON_STRING_FIELD

    non_mapping = _adapt_rust(_rust_row(input_data=json.dumps(["a", "b"])))
    assert isinstance(non_mapping, Skip) and non_mapping.reason == SKIP_NON_MAPPING_FIELD

    raw_dict_input = _adapt_rust(_rust_row(input_data={"query": "already a dict"}))
    assert isinstance(raw_dict_input, Skip) and raw_dict_input.reason == SKIP_NON_STRING_FIELD


def test_rust_oversized_field_skipped_before_parsing() -> None:
    """Fields larger than the configured limit skip with an explicit reason."""
    big = "x" * 64
    outcome = adapt_rust_row(_rust_row(input_data=big), row_id="rust:train:0", max_field_bytes=32)
    assert isinstance(outcome, Skip)
    assert outcome.reason == SKIP_OVERSIZED_FIELD


def test_rust_never_executes_dataset_strings() -> None:
    """Malicious literals are skipped, never evaluated or executed."""
    malicious = '{"query": __import__("os").system("touch /tmp/pwned")}'
    outcome = _adapt_rust(_rust_row(input_data=malicious))
    assert isinstance(outcome, Skip)
    assert outcome.reason == SKIP_UNPARSABLE_FIELD

    truncated = "{'query': 'unterminated"
    outcome = _adapt_rust(_rust_row(input_data=truncated))
    assert isinstance(outcome, Skip)
    assert outcome.reason == SKIP_UNPARSABLE_FIELD


def test_parse_serialized_mapping_accepts_only_mappings() -> None:
    """JSON objects and dict literals pass; everything else skips with a reason."""
    ok = parse_serialized_mapping('{"a": 1}', "input_data", max_field_bytes=MAX_BYTES)
    assert ok == {"a": 1}
    literal = parse_serialized_mapping("{'a': 2}", "input_data", max_field_bytes=MAX_BYTES)
    assert literal == {"a": 2}

    scalar = parse_serialized_mapping("42", "input_data", max_field_bytes=MAX_BYTES)
    assert isinstance(scalar, Skip) and scalar.reason == SKIP_NON_MAPPING_FIELD
    broken = parse_serialized_mapping("not (valid", "input_data", max_field_bytes=MAX_BYTES)
    assert isinstance(broken, Skip) and broken.reason == SKIP_UNPARSABLE_FIELD
    oversized = parse_serialized_mapping('{"a": "' + "x" * 64 + '"}', "f", max_field_bytes=32)
    assert isinstance(oversized, Skip) and oversized.reason == SKIP_OVERSIZED_FIELD
    non_string = parse_serialized_mapping(12345, "f", max_field_bytes=MAX_BYTES)
    assert isinstance(non_string, Skip) and non_string.reason == SKIP_NON_STRING_FIELD


def test_rust_missing_crate_falls_back_to_row_group() -> None:
    """Without crate_name the row groups by stable row identity instead."""
    outcome = _adapt_rust(_rust_row(crate_name=""))
    assert isinstance(outcome, PairRecord)
    assert outcome.group_id == "row:rust:train:0"
    assert outcome.group_strength == "row_id"
    assert "crate_name" not in outcome.provenance


def test_typescript_adapter_mapping_and_provenance() -> None:
    """docstring/code pair grouped by repo with optional provenance preserved."""
    row: dict[str, object] = {
        "docstring": "Formats a name.",
        "code": "export function formatName(a: string) {\n  return a;\n}",
        "repo": "org/repo-1",
        "func_name": "formatName",
        "path": "src/format.ts",
        "url": "https://example.com/src/format.ts#L1",
        "license": "MIT",
    }
    outcome = adapt_typescript_row(row, row_id="typescript:train:0", max_field_bytes=MAX_BYTES)
    assert isinstance(outcome, PairRecord)
    assert outcome.query == "Formats a name."
    assert outcome.code.startswith("export function formatName")
    assert outcome.group_id == "repo:org/repo-1"
    assert outcome.group_strength == "repo"
    assert outcome.category is None
    assert outcome.provenance == {
        "func_name": "formatName",
        "path": "src/format.ts",
        "url": "https://example.com/src/format.ts#L1",
        "license": "MIT",
    }


def test_typescript_missing_repo_and_empty_docstring() -> None:
    """Missing repo falls back to per-row grouping; blank docstring skips."""
    no_repo = adapt_typescript_row(
        {"docstring": "Does things.", "code": "export const x = 1;"},
        row_id="typescript:train:3",
        max_field_bytes=MAX_BYTES,
    )
    assert isinstance(no_repo, PairRecord)
    assert no_repo.group_id == "row:typescript:train:3"
    assert no_repo.group_strength == "row_id"

    blank = adapt_typescript_row(
        {"docstring": "  ", "code": "export const y = 2;", "repo": "org/repo-2"},
        row_id="typescript:train:4",
        max_field_bytes=MAX_BYTES,
    )
    assert isinstance(blank, Skip) and blank.reason == SKIP_EMPTY_FIELD


def test_swift_uses_translated_solution_never_ground_truth() -> None:
    """Positive code is translated_solution; serialized test assertions stay out."""
    ground_truth = json.dumps(["assert solve([1, 2]) == [2, 3]"])
    row: dict[str, object] = {
        "id": "swift-17",
        "translated_problem": "Compute pairwise sums.",
        "translated_solution": "func solve(_ a: [Int]) -> [Int] {\n    a\n}",
        "ground_truth": ground_truth,
        "translated_test_cases": json.dumps([{"input": "[1, 2]", "output": "[2, 3]"}]),
    }
    outcome = adapt_swift_row(row, row_id="swift:train:0", max_field_bytes=MAX_BYTES)
    assert isinstance(outcome, PairRecord)
    assert outcome.query == "Compute pairwise sums."
    assert outcome.code == "func solve(_ a: [Int]) -> [Int] {\n    a\n}"
    assert ground_truth not in outcome.code
    assert "assert" not in outcome.code
    assert outcome.group_id == "row:swift-17"
    assert outcome.group_strength == "row_id"
    assert outcome.provenance == {"id": "swift-17"}


def test_swift_missing_solution_skips_even_when_ground_truth_exists() -> None:
    """A present ground_truth never becomes a fallback for missing solution code."""
    row: dict[str, object] = {
        "id": "swift-18",
        "translated_problem": "Solve it.",
        "translated_solution": "",
        "ground_truth": json.dumps(["assert solve() == 42"]),
    }
    outcome = adapt_swift_row(row, row_id="swift:train:1", max_field_bytes=MAX_BYTES)
    assert isinstance(outcome, Skip)
    assert outcome.reason == SKIP_EMPTY_FIELD


def test_kotlin_corpus_only_mapping_and_provenance() -> None:
    """KStack rows with the verified int64 repo_id group and preserve provenance."""
    row: dict[str, object] = {
        "content": 'fun main() {\n    println("hi")\n}',
        "repo_id": 918273,
        "path": "src/main/kotlin/Main.kt",
        "commit_sha": "abc123",
        "license": "Apache-2.0",
    }
    outcome = adapt_kotlin_row(row, row_id="kotlin:shard:0", max_field_bytes=MAX_BYTES)
    assert isinstance(outcome, CorpusRecord)
    assert outcome.text == 'fun main() {\n    println("hi")\n}'
    assert outcome.group_id == "repo:918273"
    assert outcome.group_strength == "repo"
    assert outcome.provenance == {
        "repo_id": "918273",
        "path": "src/main/kotlin/Main.kt",
        "commit_sha": "abc123",
        "license": "Apache-2.0",
    }

    string_id = adapt_kotlin_row(
        {"content": "val s = 1", "repo_id": "JetBrains/exposed"},
        row_id="kotlin:shard:4",
        max_field_bytes=MAX_BYTES,
    )
    assert isinstance(string_id, CorpusRecord)
    assert string_id.group_id == "repo:JetBrains/exposed"


def test_kotlin_owner_name_fallback_and_empty_content() -> None:
    """owner/name substitutes for repo_id; blank content skips."""
    fallback = adapt_kotlin_row(
        {"content": "val x = 1", "owner": "kotlin", "name": "ktor", "path": "A.kt"},
        row_id="kotlin:shard:1",
        max_field_bytes=MAX_BYTES,
    )
    assert isinstance(fallback, CorpusRecord)
    assert fallback.group_id == "repo:kotlin/ktor"
    assert fallback.provenance == {"repo": "kotlin/ktor", "path": "A.kt"}

    ungrouped = adapt_kotlin_row({"content": "val y = 2"}, row_id="kotlin:shard:2", max_field_bytes=MAX_BYTES)
    assert isinstance(ungrouped, CorpusRecord)
    assert ungrouped.group_id == "row:kotlin:shard:2"
    assert ungrouped.group_strength == "row_id"

    blank = adapt_kotlin_row({"content": ""}, row_id="kotlin:shard:3", max_field_bytes=MAX_BYTES)
    assert isinstance(blank, Skip) and blank.reason == SKIP_EMPTY_FIELD



def test_optional_adapters_apply_field_size_bound() -> None:
    """TS/Swift/Kotlin skip oversized query/code/text fields with an explicit reason."""
    long_field = "x" * 64
    ts_code = adapt_typescript_row(
        {"docstring": "Does things.", "code": long_field, "repo": "org/repo"},
        row_id="typescript:train:0",
        max_field_bytes=32,
    )
    assert isinstance(ts_code, Skip) and ts_code.reason == SKIP_OVERSIZED_FIELD

    ts_docstring = adapt_typescript_row(
        {"docstring": long_field, "code": "export const x = 1;", "repo": "org/repo"},
        row_id="typescript:train:1",
        max_field_bytes=32,
    )
    assert isinstance(ts_docstring, Skip) and ts_docstring.reason == SKIP_OVERSIZED_FIELD

    swift_code = adapt_swift_row(
        {"id": "swift-1", "translated_problem": "Solve.", "translated_solution": long_field},
        row_id="swift:train:0",
        max_field_bytes=32,
    )
    assert isinstance(swift_code, Skip) and swift_code.reason == SKIP_OVERSIZED_FIELD

    swift_problem = adapt_swift_row(
        {"id": "swift-2", "translated_problem": long_field, "translated_solution": "func f() {}"},
        row_id="swift:train:1",
        max_field_bytes=32,
    )
    assert isinstance(swift_problem, Skip) and swift_problem.reason == SKIP_OVERSIZED_FIELD

    kotlin_text = adapt_kotlin_row(
        {"content": long_field, "repo_id": 7},
        row_id="kotlin:shard:0",
        max_field_bytes=32,
    )
    assert isinstance(kotlin_text, Skip) and kotlin_text.reason == SKIP_OVERSIZED_FIELD