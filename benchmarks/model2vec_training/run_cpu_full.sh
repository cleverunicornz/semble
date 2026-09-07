#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/semble-rust-contrastive}"
RUN_ID="${RUN_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION must identify the tested Semble commit}"
VENV="${VENV:-$ROOT/venv}"
UV_BOOTSTRAP="${UV_BOOTSTRAP:-$ROOT/uv-bootstrap}"
CACHE="${CACHE:-$ROOT/cache}"
RUST_INPUT="${RUST_INPUT:-$ROOT/input/rust}"
DATA_OUTPUT="$ROOT/data/$RUN_ID"
REPO_ROOT="$ROOT/repos"
RESULT="$ROOT/results/$RUN_ID"
SMOKE_OUTPUT="$RESULT/smoke"
TRAIN_OUTPUT="$RESULT/training"
EVAL_OUTPUT="$RESULT/evaluation"

SOURCE="$ROOT/source"
LOCK="$SOURCE/benchmarks/model2vec_training/requirements-cpu.lock"
mkdir -p "$ROOT/data" "$ROOT/repos" "$ROOT/results" "$CACHE" "$RESULT"
test -f "$LOCK"
test -f "$RUST_INPUT/pairs.train.jsonl"
test -f "$RUST_INPUT/pairs.eval.jsonl"
test -f "$RUST_INPUT/manifest.json"

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip git curl gzip
if [[ ! -x "$UV_BOOTSTRAP/bin/uv" ]]; then
  python3 -m venv "$UV_BOOTSTRAP"
  "$UV_BOOTSTRAP/bin/python" -m pip install 'uv==0.11.25'
fi
UV="$UV_BOOTSTRAP/bin/uv"
LOCK_SHA="$(sha256sum "$LOCK" | cut -d' ' -f1)"
if [[ ! -x "$VENV/bin/python" ]] || [[ ! -f "$VENV/.requirements-sha256" ]] || \
  [[ "$(<"$VENV/.requirements-sha256")" != "$LOCK_SHA" ]]; then
  "$UV" venv --clear --python python3 "$VENV"
  "$UV" pip install --python "$VENV/bin/python" --torch-backend cpu -r "$LOCK"
  printf '%s\n' "$LOCK_SHA" > "$VENV/.requirements-sha256"
fi

export HF_HOME="$CACHE/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28
export NUMEXPR_NUM_THREADS=28
export PYTHONPATH="$SOURCE/src:$SOURCE"

POTION_V1="$CACHE/models/potion-code-16m-v1"
POTION_V2="$CACHE/models/potion-code-16m-v2"
"$VENV/bin/hf" download minishlab/potion-code-16M \
  --revision 1b0ff71095656b23306542bbad34a09109673720 \
  --local-dir "$POTION_V1"
"$VENV/bin/hf" download minishlab/potion-code-16M-v2 \
  --revision e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b \
  --local-dir "$POTION_V2"
"$VENV/bin/hf" cache verify minishlab/potion-code-16M \
  --revision 1b0ff71095656b23306542bbad34a09109673720 \
  --local-dir "$POTION_V1"
"$VENV/bin/hf" cache verify minishlab/potion-code-16M-v2 \
  --revision e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b \
  --local-dir "$POTION_V2"

cd "$SOURCE"
"$VENV/bin/python" -m pytest -c /dev/null --rootdir=. --noconftest \
  tests/benchmarks/test_model2vec_contrastive.py \
  tests/benchmarks/test_model2vec_full_evaluation.py

"$VENV/bin/python" -m benchmarks.model2vec_training.prepare_contrastive \
  --rust-train "$RUST_INPUT/pairs.train.jsonl" \
  --rust-eval "$RUST_INPUT/pairs.eval.jsonl" \
  --rust-manifest "$RUST_INPUT/manifest.json" \
  --output "$DATA_OUTPUT"

"$VENV/bin/python" -m benchmarks.model2vec_training.train_contrastive \
  --training-data "$DATA_OUTPUT/train.jsonl" \
  --training-manifest "$DATA_OUTPUT/manifest.json" \
  --base-model "$POTION_V2" \
  --output "$SMOKE_OUTPUT" \
  --source-revision "$SOURCE_REVISION" \
  --run-id "$RUN_ID-smoke" \
  --mode smoke

"$VENV/bin/python" -m benchmarks.model2vec_training.sync_evaluation_repos \
  --root "$REPO_ROOT" \
  --jobs 6 \
  --output "$RESULT/repositories.json"

"$VENV/bin/python" -m benchmarks.model2vec_training.train_contrastive \
  --training-data "$DATA_OUTPUT/train.jsonl" \
  --training-manifest "$DATA_OUTPUT/manifest.json" \
  --base-model "$POTION_V2" \
  --output "$TRAIN_OUTPUT" \
  --source-revision "$SOURCE_REVISION" \
  --run-id "$RUN_ID" \
  --mode full

"$VENV/bin/python" -m benchmarks.model2vec_training.evaluate_full \
  --model "potion-v2=$POTION_V2" \
  --model "potion-v1=$POTION_V1" \
  --model "candidate-raw=$TRAIN_OUTPUT/candidate-raw" \
  --model "candidate-post-sif=$TRAIN_OUTPUT/candidate-post-sif" \
  --baseline-label potion-v2 \
  --rust-train "$RUST_INPUT/pairs.train.jsonl" \
  --rust-eval "$RUST_INPUT/pairs.eval.jsonl" \
  --repo-root "$REPO_ROOT" \
  --repo-manifest "$RESULT/repositories.json" \
  --output "$EVAL_OUTPUT" \
  --source-revision "$SOURCE_REVISION" \
  --run-id "$RUN_ID"

cp "$DATA_OUTPUT/manifest.json" "$RESULT/training-data-manifest.json"
gzip -c "$DATA_OUTPUT/train.jsonl" > "$RESULT/training-data.jsonl.gz"
gzip -c "$DATA_OUTPUT/provenance.jsonl" > "$RESULT/training-provenance.jsonl.gz"
"$UV" pip freeze --python "$VENV/bin/python" > "$RESULT/environment.freeze.txt"
sha256sum \
  "$SOURCE/benchmarks/model2vec_training/requirements-cpu.txt" \
  "$SOURCE/benchmarks/model2vec_training/requirements-cpu.lock" \
  "$RESULT/training-data.jsonl.gz" \
  "$RESULT/training-provenance.jsonl.gz" \
  > "$RESULT/artifacts.sha256"
lscpu > "$RESULT/lscpu.txt"
free -b > "$RESULT/memory.txt"
df -h / > "$RESULT/disk.txt"
tar -C "$ROOT/results" -czf "$ROOT/results/$RUN_ID.tar.gz" "$RUN_ID"
sha256sum "$ROOT/results/$RUN_ID.tar.gz"
