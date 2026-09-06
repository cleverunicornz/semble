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

1. `current` runs the retrieval comparison through the evaluation seam using Semble's existing chunk-source function, and a successful run writes JSON whose `mode` is `current` and whose `label` is the invocation label.
2. `legacy-cast` runs the retrieval comparison at that seam; for a path accepted by `supports_path`, it calls native `chunk_source` and converts the result into a Semble chunk with the selected path and the native start and end lines, while for a rejected path it uses Semble's current chunker. A native support-check or chunking exception is an evaluation error rather than a successful fallback, and a successful run writes JSON whose `mode` is `legacy-cast` and whose `label` is the invocation label.
3. `chunk-timing` invokes native `chunk_files` independently from retrieval, accepts only a list containing one result for each expected path in the same order, records the count of returned per-file errors, rejects a changed `(files, chunks, errors)` shape across repetitions, and on success writes JSON whose `mode` is `chunk-timing` and whose `label` is the invocation label.

## Fail

The promise fails when any applicable leg is contradicted:

1. `current` bypasses the evaluation seam or Semble's existing chunk-source function, or its successful JSON result omits or mismatches the `current` mode or invocation label.
2. `legacy-cast` bypasses the evaluation seam; sends a path rejected by `supports_path` to native `chunk_source`; fails to use Semble's current chunker for such a path; loses the selected path or native line metadata while converting a supported-path result; turns a native support-check or chunking exception into a successful fallback measurement; or emits successful JSON that omits or mismatches the `legacy-cast` mode or invocation label.
3. `chunk-timing` uses a retrieval comparison instead of native `chunk_files`; accepts a non-list, missing, extra, or reordered batch result; fails to record the count of returned per-file errors; accepts a changed `(files, chunks, errors)` shape across repetitions; or emits successful JSON that omits or mismatches the `chunk-timing` mode or invocation label.

## References

- [D-000001](situation/decisions/D-000001-evaluation-only-native-chunker.md)
- `4a7b4bd442a0d6f3865568799b5cb1d92334ddb7:benchmarks/chunker_eval.py`
- `9d2d3e9ba4dc4f986b0aad3f2d51fff7b0dce1ec:benchmarks/chunker_eval.py`