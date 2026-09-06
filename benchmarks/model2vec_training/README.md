# Rust-first Model2Vec training smoke

This experiment adapts the existing `potion-code-16M-v2` static student toward
Qwen3-Embedding-8B targets using the already-qualified Strandset Rust sample.
It is a bounded pipeline and quality signal, not a production model build.

## Fixed inputs

| Input | Revision |
| --- | --- |
| `Qwen/Qwen3-Embedding-8B` | `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` |
| `minishlab/potion-code-16M-v2` | `e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b` |
| `Fortytwo-Network/Strandset-Rust-v1` | `0a8d223302712a2b34a6ad4ce1fd679031894b3d` |

The prepared sample must be the output of
`benchmarks.model2vec_data.prepare`: 894 training pairs and 100 held-out
evaluation pairs for the qualified seed/configuration. The runner requires its
manifest and checks the exact qualified row counts and SHA-256 digests before
loading the teacher.

## What the smoke does

1. Splits the 894 prepared training pairs again by crate into student training
   and validation partitions. The 100 prepared evaluation pairs remain
   untouched until final scoring.
2. Encodes queries with an explicit code-retrieval instruction and encodes code
   without a prompt. Qwen's Sentence Transformers configuration supplies
   last-token pooling; targets are truncated and normalized to 256 dimensions.
3. Initializes a global affine projection from the current static student to
   the teacher space, then tunes the projection and observed static token
   vectors using cosine loss and bounded early stopping.
4. Folds the affine projection into every vocabulary row, zeros the PAD vector,
   and saves an ordinary float16 Model2Vec model.
5. Reports teacher, current-student, and candidate Recall@1/5/10 and MRR over
   the 100 held-out pairs. These metrics are a smoke signal only.
6. Loads and searches the exported model through Semble with CUDA hidden.

Stock Tokenlearn is not called. Its published featurizer mean-pools token
embeddings and its current training path assumes padding ID 0; Qwen3 uses
last-token pooling and tokenizer padding ID 151643. The runner instead uses
Qwen's supported sentence-embedding path and handles the static student's
padding explicitly.

## Gates

The run passes only when the candidate reloads, is exactly 256-dimensional,
has finite vectors, has an exactly zero PAD vector, is batch-invariant within a
minimum corresponding-row cosine of `0.99999`, produces finite teacher metrics,
and completes finite validation loss. Improvement over the baseline is
reported but is not a smoke gate.

## GPU command

`requirements-gpu.txt` pins the direct experiment dependencies;
`requirements-gpu.lock` freezes their Linux/Python 3.12 transitive resolution.
`run_gpu_smoke.sh` installs that lock, downloads both pinned public model
snapshots, runs the experiment, runs CPU verification, captures the frozen
environment, and archives the result. It expects a source bundle at
`$ROOT/source` and the prepared manifest plus two JSONL files under
`$ROOT/input`.

```bash
SOURCE_REVISION=<git-sha> RUN_ID=<run-id> \
  timeout 7200 bash benchmarks/model2vec_training/run_gpu_smoke.sh
```

The external controller retains the log and retrieves `$ROOT/out/<run-id>.tar.gz`
before terminating the disposable VM.

## First smoke result

`receipts/massed-20260906-ccd30f4.yml` records the first H100 run. The model
pipeline and CPU-loading gates passed, but the candidate regressed against the
current student on all held-out retrieval metrics (`Recall@1` 0.84 versus
0.89). It is retained as experiment evidence and is not a deployment
candidate. The run also exposed a missing Semble-runtime dependency in the
first wrapper revision; the corrected GPU lock contains those dependencies.

## Full Rust contrastive experiment

The Qwen smoke above is retained as rejected diagnostic evidence. The full
experiment follows the final retrieval-training stage published in
`potion-code-16M/train.py`: initialize from `potion-code-16M-v2` and optimize
real query/document pairs with `MultipleNegativesRankingLoss` in the existing
embedding space.

The complete pinned Strandset scan produces 30,893 training pairs and 3,750
crate-disjoint evaluation pairs after exact deduplication. Training uses every
one of those 30,893 Rust pairs plus exactly 30,893 deterministic replay pairs
from each original CornStack language: Go, Java, JavaScript, PHP, Python, and
Ruby. Total full-training size is 216,251 pairs; Rust is neither capped below
its usable corpus nor diluted below the other individual languages.

`prepare_contrastive.py` streams the six immutable CornStack revisions,
applies MinishLab's published minimum-length and exact query/document dedup
filters, excludes any exact overlap with the Rust holdout, and writes an
attested two-column training JSONL plus separate provenance.

`train_contrastive.py` has two explicit modes. `smoke` runs two steps only to
qualify wiring. `full` reloads a pristine v2 model and trains all 216,251 pairs
for three epochs at batch size 512 and learning rate `5e-3`, matching the
published v1 MNRL settings. It exports both the raw trained vectors and the
published v1-style PCA/SIF post-processed variant. Training success is never a
quality result.

`evaluate_full.py` compares potion v1, potion v2, the raw candidate, and the
post-SIF candidate on two separate surfaces:

- all 3,750 held-out Rust pairs, dense and normal Semble hybrid ranking, with
  paired bootstrap intervals and exact McNemar top-1 tests, both against the
  3,750 held-out documents alone and against all 34,643 extracted Rust
  documents as distractors;
- every annotated repository in Semble's pinned benchmark suite, including
  Tokio, Serde, and Axum, with dense and hybrid NDCG@10 reported overall and by
  language.

`run_cpu_full.sh` performs environment setup, Hub checksum verification, unit
tests, balanced-data preparation, the non-evidentiary smoke, exact shallow
materialization of all benchmark repositories, pristine full training, full
evaluation, environment capture, and result archiving. It is intended for a
disposable 28-vCPU Linux CPU instance and is externally time-bounded.

`receipts/massed-20260906-a31a6b90.yml` records the complete CPU run. The raw
candidate improved full-corpus Rust Recall@1 by 5.39 percentage points for
dense retrieval and 2.91 points for normal hybrid retrieval, with paired 95%
intervals excluding zero. Across the 63-repository suite, its all-language
hybrid NDCG@10 was effectively unchanged and its 60-query Rust result moved by
+0.0053, with an interval that includes zero. The evidence is promising but is
not a deployment qualification.
