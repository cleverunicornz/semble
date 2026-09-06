## State

designed

## Judges

[P-000001](situation/promises/P-000001-evaluation-only-chunker-comparison.md)

## Inputs

- The implementation at `benchmarks/chunker_eval.py`.
- A valid invocation's selected mode, arguments, and retained labeled JSON output.
- For native modes, the loaded `code_chunker_native` boundary and its `supports_path`, `chunk_source`, and `chunk_files` responses.

## Pass

The promise passes only when all applicable legs hold within its Scope:

1. `current` uses the existing Semble chunk-source function at the evaluation seam and writes a labeled result identifying the current mode.
2. `legacy-cast` calls the native chunker only for paths its support predicate accepts, preserves the resulting path and line metadata in Semble chunks, and turns a native support or chunking failure into an evaluation error rather than a fallback measurement.
3. `chunk-timing` accepts native batch output only when it has the expected count and order, records its error count, and rejects a changed result shape across repetitions.

## Fail

The promise fails when any applicable leg is contradicted: the harness changes production source or persistent cache behavior; a native-boundary fault becomes a successful fallback measurement; or a timing result accepts missing, reordered, or unstable native output.

## References

- [D-000001](situation/decisions/D-000001-evaluation-only-native-chunker.md)
- `4a7b4bd442a0d6f3865568799b5cb1d92334ddb7:benchmarks/chunker_eval.py`
- `9d2d3e9ba4dc4f986b0aad3f2d51fff7b0dce1ec:benchmarks/chunker_eval.py`