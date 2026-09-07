"""Row adapters normalizing raw dataset rows into pair or corpus records.

Dataset strings are parsed with ``json.loads`` and a safe ``ast.literal_eval``
fallback only; dataset content is never executed. Parsers accept mappings only,
and inputs larger than the configured field-size limit are skipped outright.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

SKIP_UNSUPPORTED_CATEGORY = "unsupported_category"
SKIP_OVERSIZED_FIELD = "oversized_field"
SKIP_UNPARSABLE_FIELD = "unparsable_field"
SKIP_NON_MAPPING_FIELD = "non_mapping_field"
SKIP_MISSING_FIELD = "missing_field"
SKIP_EMPTY_FIELD = "empty_field"
SKIP_NON_STRING_FIELD = "non_string_field"
SKIP_INVALID_ROW = "invalid_row"


@dataclass(frozen=True)
class PairRecord:
    """A normalized query/code pair."""

    query: str
    code: str
    category: str | None
    group_id: str
    group_strength: str
    provenance: Mapping[str, str]


@dataclass(frozen=True)
class CorpusRecord:
    """A corpus-only text document."""

    text: str
    group_id: str
    group_strength: str
    provenance: Mapping[str, str]


@dataclass(frozen=True)
class Skip:
    """A scanned row that was not adapted, with an explicit reason."""

    reason: str
    detail: str = ""


Adapted = PairRecord | CorpusRecord | Skip
AdapterFn = Callable[..., Adapted]


def parse_serialized_mapping(raw: object, field_name: str, *, max_field_bytes: int) -> Mapping[str, Any] | Skip:
    """Safely parse a serialized mapping from JSON or a Python literal.

    Never executes dataset strings: ``json.loads`` and ``ast.literal_eval`` are
    the only parsers, mappings are the only accepted results, and strings larger
    than ``max_field_bytes`` (UTF-8) are skipped before any parsing.
    """
    if raw is None:
        return Skip(SKIP_MISSING_FIELD, field_name)
    if not isinstance(raw, str):
        return Skip(SKIP_NON_STRING_FIELD, field_name)
    if len(raw.encode("utf-8")) > max_field_bytes:
        return Skip(SKIP_OVERSIZED_FIELD, field_name)
    parsed_any = False
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
        except Exception:  # parser boundary: any malformed input becomes a skip, never an execution
            continue
        parsed_any = True
        if isinstance(parsed, Mapping):
            return parsed
    return Skip(SKIP_NON_MAPPING_FIELD if parsed_any else SKIP_UNPARSABLE_FIELD, field_name)


def _required_text(
    container: Mapping[str, object], key: str, *, max_field_bytes: int | None = None
) -> str | Skip:
    """Extract a required non-empty string value, optionally bounded in size.

    When ``max_field_bytes`` is set (adapters that retain raw columns), values
    larger than the limit skip with an explicit reason before being retained.
    Parsed-container lookups (Rust) leave it unset: their parent string was
    already bounded before parsing.
    """
    if key not in container or container[key] is None:
        return Skip(SKIP_MISSING_FIELD, key)
    value = container[key]
    if not isinstance(value, str):
        return Skip(SKIP_NON_STRING_FIELD, key)
    if not value.strip():
        return Skip(SKIP_EMPTY_FIELD, key)
    if max_field_bytes is not None and len(value.encode("utf-8")) > max_field_bytes:
        return Skip(SKIP_OVERSIZED_FIELD, key)
    return value


def _optional_str(container: Mapping[str, object], key: str) -> str | None:
    """Return a non-empty string value from a mapping, or None."""
    value = container.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _optional_identifier(container: Mapping[str, object], key: str) -> str | None:
    """Return a non-empty identifier as a string; int64 identifiers are accepted."""
    value = container.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value
    return None


_RUST_CATEGORY_SOURCES: dict[str, tuple[str, str, str, str]] = {
    # category: (query container field, query key, code container field, code key)
    "code_search": ("input_data", "query", "output_data", "code_snippet"),
    "code_generation": ("input_data", "description", "output_data", "code"),
    "code_summarization": ("output_data", "summary", "input_data", "code"),
}


def _rust_group_and_provenance(
    row: Mapping[str, object], query_side: Mapping[str, object], code_side: Mapping[str, object], row_id: str
) -> tuple[str, str, dict[str, str]]:
    """Return (group_id, group_strength, provenance) for a Rust row.

    crate_name groups by crate; rows without one fall back to per-row grouping.
    Any supplied code_context is kept as provenance, separate from the code.
    """
    crate = _optional_str(row, "crate_name")
    provenance: dict[str, str] = {}
    if crate is not None:
        provenance["crate_name"] = crate
    for side in (query_side, code_side):
        context = _optional_str(side, "code_context")
        if context is not None:
            provenance["code_context"] = context
            break
    if crate is not None:
        return f"crate:{crate}", "crate", provenance
    return f"row:{row_id}", "row_id", provenance


def adapt_rust_row(row: Mapping[str, object], *, row_id: str, max_field_bytes: int) -> Adapted:
    """Adapt a Strandset-Rust-v1 row into a pair for its supported category.

    Categories outside ``_RUST_CATEGORY_SOURCES`` are unsupported by this first
    adapter and are reported as such (not as bad examples). The ``test`` column
    and any parsed test assertions are never used as positive code.
    """
    category = row.get("task_category")
    if not isinstance(category, str) or category not in _RUST_CATEGORY_SOURCES:
        detail = category if isinstance(category, str) else "<missing>"
        return Skip(SKIP_UNSUPPORTED_CATEGORY, detail)
    query_field, query_key, code_field, code_key = _RUST_CATEGORY_SOURCES[category]
    sides: dict[str, Mapping[str, Any] | Skip] = {}
    for field_name in (query_field, code_field):
        if field_name not in sides:
            sides[field_name] = parse_serialized_mapping(
                row.get(field_name), field_name, max_field_bytes=max_field_bytes
            )
    query_side = sides[query_field]
    if isinstance(query_side, Skip):
        return query_side
    code_side = sides[code_field]
    if isinstance(code_side, Skip):
        return code_side
    query = _required_text(query_side, query_key)
    if isinstance(query, Skip):
        return query
    code = _required_text(code_side, code_key)
    if isinstance(code, Skip):
        return code
    group_id, group_strength, provenance = _rust_group_and_provenance(row, query_side, code_side, row_id)
    return PairRecord(
        query=query,
        code=code,
        category=category,
        group_id=group_id,
        group_strength=group_strength,
        provenance=provenance,
    )


def adapt_typescript_row(row: Mapping[str, object], *, row_id: str, max_field_bytes: int) -> Adapted:
    """Adapt a typescript-treesitter row into a docstring/code pair grouped by repo."""
    query = _required_text(row, "docstring", max_field_bytes=max_field_bytes)
    if isinstance(query, Skip):
        return query
    code = _required_text(row, "code", max_field_bytes=max_field_bytes)
    if isinstance(code, Skip):
        return code
    repo = _optional_str(row, "repo")
    provenance = {
        key: value for key in ("func_name", "path", "url", "license") if (value := _optional_str(row, key)) is not None
    }
    if repo is not None:
        group_id, group_strength = f"repo:{repo}", "repo"
    else:
        group_id, group_strength = f"row:{row_id}", "row_id"
    return PairRecord(
        query=query,
        code=code,
        category=None,
        group_id=group_id,
        group_strength=group_strength,
        provenance=provenance,
    )


def adapt_swift_row(row: Mapping[str, object], *, row_id: str, max_field_bytes: int) -> Adapted:
    """Adapt an rlvr-code-data-Swift row into a problem/solution pair.

    ``ground_truth`` holds serialized TEST ASSERTIONS and ``translated_test_cases``
    is metadata: neither is positive code, neither is evidence a solution passed,
    neither is executed, and neither is ever used as a fallback for a missing
    ``translated_solution``. No repository/crate grouping field exists in this
    dataset, so grouping is per source id.
    """
    query = _required_text(row, "translated_problem", max_field_bytes=max_field_bytes)
    if isinstance(query, Skip):
        return query
    code = _required_text(row, "translated_solution", max_field_bytes=max_field_bytes)
    if isinstance(code, Skip):
        return code
    source_id = _optional_str(row, "id")
    if source_id is not None:
        group_id = f"row:{source_id}"
        provenance = {"id": source_id}
    else:
        group_id = f"row:{row_id}"
        provenance = {}
    return PairRecord(
        query=query,
        code=code,
        category=None,
        group_id=group_id,
        group_strength="row_id",
        provenance=provenance,
    )


def adapt_kotlin_row(row: Mapping[str, object], *, row_id: str, max_field_bytes: int) -> Adapted:
    """Adapt a KStack row into a corpus-only document (content is raw Kotlin source)."""
    text = _required_text(row, "content", max_field_bytes=max_field_bytes)
    if isinstance(text, Skip):
        return text
    repo_id = _optional_identifier(row, "repo_id")
    owner = _optional_str(row, "owner")
    name = _optional_str(row, "name")
    provenance: dict[str, str] = {}
    if repo_id is not None:
        group_id, group_strength = f"repo:{repo_id}", "repo"
        provenance["repo_id"] = repo_id
    elif owner is not None and name is not None:
        group_id, group_strength = f"repo:{owner}/{name}", "repo"
        provenance["repo"] = f"{owner}/{name}"
    else:
        group_id, group_strength = f"row:{row_id}", "row_id"
    for key in ("path", "commit_sha", "license"):
        value = _optional_str(row, key)
        if value is not None:
            provenance[key] = value
    return CorpusRecord(text=text, group_id=group_id, group_strength=group_strength, provenance=provenance)


ADAPTERS: dict[str, AdapterFn] = {
    "rust_pairs": adapt_rust_row,
    "typescript_pairs": adapt_typescript_row,
    "swift_pairs": adapt_swift_row,
    "kotlin_corpus": adapt_kotlin_row,
}
