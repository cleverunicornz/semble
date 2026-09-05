from collections.abc import Iterator

from benchmarks.profile_cold_index import (
    _embedding_counters,
    _PhaseRecorder,
    _summarize_records,
)


def _clock(values: list[int]) -> Iterator[int]:
    yield from values


def test_phase_recorder_tracks_exclusive_time_counters_and_call_sizes() -> None:
    """Nested time is removed once, while counters and call sizes remain raw."""
    ticks = _clock([0, 10, 40, 100])
    recorder = _PhaseRecorder(clock=lambda: next(ticks))

    with recorder.measure("instrumented_index"):
        with recorder.measure("static_model_encode", items=3, call_size=3):
            recorder.add_counter("encoded_texts", 3)

    total = recorder.phase("instrumented_index")
    encode = recorder.phase("static_model_encode")
    assert total.wall_ns == 100
    assert total.exclusive_wall_ns == 70
    assert encode.wall_ns == encode.exclusive_wall_ns == 30
    assert encode.calls == 1
    assert encode.items == recorder.counters["encoded_texts"] == 3
    assert encode.as_dict()["call_size_distribution"] == {
        "count": 1,
        "total": 3,
        "median": 3.0,
        "min": 3,
        "max": 3,
    }
    assert recorder.nested_wall_ns == {"instrumented_index>static_model_encode": 30}


def test_summary_reports_median_min_max_for_numeric_leaves_only() -> None:
    """Per-repetition metadata, booleans, and raw arrays do not pollute arithmetic summaries."""
    records = [
        {
            "repetition": 1,
            "total": {"instrumented_wall_ns": 30, "instrumented_process_cpu_ns": 9},
            "counts": {"chunks": 3, "proof": True},
            "sizes": [1, 2],
        },
        {
            "repetition": 2,
            "total": {"instrumented_wall_ns": 10, "instrumented_process_cpu_ns": 7},
            "counts": {"chunks": 1, "proof": True},
            "sizes": [2, 3],
        },
        {
            "repetition": 3,
            "total": {"instrumented_wall_ns": 20, "instrumented_process_cpu_ns": 8},
            "counts": {"chunks": 2, "proof": True},
            "sizes": [3, 4],
        },
    ]

    summary = _summarize_records(records)

    assert "repetition" not in summary
    assert "sizes" not in summary
    assert "proof" not in summary["counts"]
    assert summary["total"]["instrumented_wall_ns"] == {"median": 20.0, "min": 10, "max": 30}
    assert summary["total"]["instrumented_process_cpu_ns"] == {"median": 8.0, "min": 7, "max": 9}
    assert summary["counts"]["chunks"] == {"median": 2.0, "min": 1, "max": 3}


def test_embedding_counters_make_duplicate_pass_visible() -> None:
    """Equality evidence distinguishes one text per unique chunk from an extra pass."""
    exact = _embedding_counters(4, 4, 4, 4)
    duplicate = _embedding_counters(4, 4, 8, 8)
    summary = _summarize_records(
        [
            {"repetition": 1, "counts": exact},
            {"repetition": 2, "counts": duplicate},
        ]
    )

    assert exact["embedded_chunks_equal_produced_chunks"] is True
    assert exact["encoded_texts_equal_unique_chunks"] is True
    assert duplicate["embedded_chunks_equal_produced_chunks"] is False
    assert duplicate["encoded_texts_equal_produced_chunks"] is False
    assert duplicate["encoded_texts_equal_unique_chunks"] is False
    assert summary["embedding_invariants"]["embedded_chunks_equal_produced_chunks"] == {
        "all_repetitions": False,
        "matching_repetitions": 1,
        "repetitions": 2,
    }
    assert summary["embedding_invariants"]["encoded_texts_equal_produced_chunks"]["all_repetitions"] is False
    assert summary["embedding_invariants"]["encoded_texts_equal_unique_chunks"]["all_repetitions"] is False
