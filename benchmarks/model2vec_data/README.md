# Model2Vec sample preparation

Reproducible extraction, normalization, and sampling of small query/code (and
corpus-only) samples for Semble's Model2Vec distillation experiment. This is
ordinary data plumbing for internal training/testing: no model loading, no
execution of dataset content, no semantic rewriting of the curated examples.

The default caps are **configurable smoke-sample defaults for pipeline
plumbing, not a claim of training sufficiency**. Whether the resulting sample
improves retrieval is the later experiment's question, not this tool's claim.

## Layout

- `sources.py` — pinned Hugging Face dataset revisions and file lists (the
  single source of truth for pins; pins were checked in parent static review,
  dataset schemas are premises from that review).
- `adapters.py` — thin row adapters (Rust/TypeScript/Swift pairs, Kotlin
  corpus) with safe `json.loads` + `ast.literal_eval` parsing (mappings only,
  nothing executed, per-field byte limit applied by every adapter).
- `readers.py` — streaming JSONL/Parquet readers and SHA-256 helpers.
- `prepare.py` — sampling/split/dedup pipeline and the CLI.

## Running without a project sync

JSONL inputs need only the standard library, so the converter runs without
resolving or syncing the production project:

```bash
uv run --no-project python -m benchmarks.model2vec_data.prepare \
  --rust-file train:benchmarks/model2vec_data/data/rust.jsonl \
  --output benchmarks/model2vec_data/out/model2vec-rust-sample
```

Parquet inputs additionally need `pyarrow`, declared only in this benchmark's
requirements file (production dependencies and `uv.lock` stay unchanged):

```bash
uv run --no-project --with pyarrow python -m benchmarks.model2vec_data.prepare \
  --rust-file train:data/rust/train-00000-of-00001.parquet \
  --output benchmarks/model2vec_data/out/model2vec-rust-sample
```

Or, from a prepared benchmark environment (repo root as cwd):

```bash
pip install -r benchmarks/model2vec_data/requirements.txt
python -m benchmarks.model2vec_data.prepare --output benchmarks/model2vec_data/out/model2vec-rust-sample
```

Example outputs point at `benchmarks/model2vec_data/out/`, which is gitignored;
never write samples into an untracked-at-root location by accident.

## Fetch pinned inputs (opt-in)

Default input root is `~/.cache/semble-model2vec-data`, layout
`<input-root>/<slug>/<pinned file path>`. Exact pinned commands (also printed
by `--list-sources`):

```bash
hf download Fortytwo-Network/Strandset-Rust-v1 --type dataset \
  --revision 0a8d223302712a2b34a6ad4ce1fd679031894b3d \
  --include data/train-00000-of-00001.parquet --include data/test-00000-of-00001.parquet \
  --local-dir ~/.cache/semble-model2vec-data/strandset-rust

hf download Shuu12121/typescript-treesitter-dedupe-filtered-datasetsV2 --type dataset \
  --revision 1e2fcd3764fb9126a33eaea58961925e667769f0 \
  --include data/train-00000-of-00001.parquet --include data/validation-00000-of-00001.parquet \
  --include data/test-00000-of-00001.parquet \
  --local-dir ~/.cache/semble-model2vec-data/typescript-treesitter

hf download saurabh5/rlvr-code-data-Swift --type dataset \
  --revision e0c853088c2629d4879e2cd3f2cd4c77bd52d130 \
  --include data/train-00000-of-00002.parquet --include data/train-00001-of-00002.parquet \
  --local-dir ~/.cache/semble-model2vec-data/rlvr-code-swift

# Kotlin corpus-only: the pinned command fetches just the first shard; the
# default scan also considers one shard (--kotlin-max-shards) so the 5.9 GB
# corpus is never pulled by accident.
hf download JetBrains/KStack --type dataset \
  --revision 425774a783d87159d103d65b79ce37f58d0697bb \
  --include data/train-00000-of-00033.parquet \
  --local-dir ~/.cache/semble-model2vec-data/kstack
```

Optional content verification against the Hub (recommended on the runner):

```bash
hf cache verify Fortytwo-Network/Strandset-Rust-v1 --type dataset \
  --revision 0a8d223302712a2b34a6ad4ce1fd679031894b3d \
  --local-dir ~/.cache/semble-model2vec-data/strandset-rust
```

## Prepare

One obvious command (Rust-only default from the default input root):

```bash
uv run --no-project --with pyarrow python -m benchmarks.model2vec_data.prepare \
  --output benchmarks/model2vec_data/out/model2vec-rust-sample
```

With explicit local files (JSONL or Parquet), no input root needed:

```bash
uv run --no-project --with pyarrow python -m benchmarks.model2vec_data.prepare \
  --rust-file train:data/rust/train-00000-of-00001.parquet \
  --rust-file eval:data/rust/test-00000-of-00001.parquet \
  --output benchmarks/model2vec_data/out/model2vec-rust-sample
```

Optional sources activate explicitly. Short or partially missing optional
inputs report shortfalls and notes; an enabled optional source with **no**
local inputs is skipped with a note (Rust stays required). Invalid explicit
`ROLE:PATH` configuration remains a hard error:

```bash
uv run --no-project --with pyarrow python -m benchmarks.model2vec_data.prepare \
  --with-typescript --with-swift --with-kotlin \
  --output benchmarks/model2vec_data/out/model2vec-full-sample
```

## Defaults

| Setting | Default | Notes |
| --- | --- | --- |
| `--max-rust-pairs` | 1000 | plumbing smoke sample |
| `--max-typescript-pairs` | 100 | when `--with-typescript` |
| `--max-swift-pairs` | 100 | when `--with-swift` |
| `--max-kotlin-docs` | 100 | corpus docs, when `--with-kotlin` |
| `--eval-fraction` | 0.1 | share of each cap reserved for eval |
| `--seed` | 0 | fixed; fully determines selection and splits |
| `--max-field-bytes` | 1 MiB | parser safety limit per serialized/retained field |
| `--max-scan-rows` | 100000 | per-file scan bound (truncation reported) |
| `--kotlin-max-shards` | 1 | KStack shards considered without explicit files |

The output directory must be new or empty (enforced in the CLI **and** in
`prepare_sample()`); existing user data is never overwritten. Invalid
configuration, unsafe output destinations, and an unusable required Rust input
fail with a clear error (exit code 2 / 1).

## Outputs

All UTF-8 JSONL, byte-reproducible for identical inputs + config:

- `pairs.train.jsonl` / `pairs.eval.jsonl` — `id`, `language`, `category`,
  `query`, `code`, `group` (`id` + `strength`), `source` (dataset, revision,
  input role, upstream split, file, row index, resolution), `provenance`
  (e.g. crate_name, code_context / repo fields / Swift id).
- `corpus.train.jsonl` / `corpus.eval.jsonl` — same envelope with `text`;
  derived from selected positive-code pairs plus explicitly enabled
  corpus-only (Kotlin) inputs.
- `manifest.json` — schema version, source pins and actual input
  filenames/SHA-256, seed/caps/budgets, scanned/eligible/selected counts by
  language/category/split, skip reasons, duplicate counts, grouping
  guarantees actually applied (including reserved eval groups), output
  hashes, and shortfalls.

The manifest records local files by path + SHA-256 only; it does not attest
that a local file matches its pinned Hub revision.

## Semantics

- **Sampling**: each eligible row gets `sha256(seed:select:<row identity>)`;
  the smallest hashes are selected per split budget (bounded memory, stable
  across runs, a genuine mix of crates/categories rather than the first N
  rows).
- **Splits**: upstream validation/test inputs are eval-only and their group
  identities (crate/repo) are **reserved**: train-file rows in a reserved
  group are routed to eval, so one group never straddles splits over the
  scanned rows. Remaining groups are assigned by
  `sha256(seed:split:<group_id>) < eval_fraction`. Swift has **no
  repository/crate field**: its groups are per source id, splits are per row,
  and repository-disjointness is **not** guaranteed — the manifest says so,
  and exact-content deduplication is applied instead.
- **Deduplication**: held-out evaluation content wins. Eval-side pairs and
  corpus texts are registered first; any training record — pair or corpus
  document — whose exact code/text is already owned by the eval side is
  excluded (counted per source, never silently). Identical content within one
  side collapses to the first occurrence. Result: `pairs.train` and
  `corpus.train` share no exact code/text with `pairs.eval` and
  `corpus.eval`, across all four output files.
- **Safety**: dataset strings are parsed with `json.loads` and a safe
  `ast.literal_eval` fallback only, accepting mappings, with an explicit size
  limit; nothing is ever executed. Swift `ground_truth` holds serialized test
  assertions — it is metadata, never positive code, and never a fallback.
  Categories beyond the three verified Rust mappings are counted as
  `unsupported_category`, not as bad examples.

## Tests

Focused tests use tiny local JSONL fixtures and run with only pytest. Disable
the production conftest and coverage configuration for this isolated run:

```bash
uv run --no-project --with pytest==9.0.3 python -m pytest -c /dev/null --noconftest \
  tests/benchmarks/test_model2vec_adapters.py tests/benchmarks/test_model2vec_prepare.py
```

## Status

Written in the `prepare-model2vec-samples` worktree for static review: the
tests and real sample generation above were **not executed on this machine**
and are handed to the automation runner. Model training, GPU jobs,
full-corpus processing, and deployment are later work.
