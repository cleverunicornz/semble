#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/semble-model2vec-smoke}"
RUN_ID="${RUN_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION must identify the tested Semble commit}"
TRAIN_PAIRS="${TRAIN_PAIRS:-$ROOT/input/pairs.train.jsonl}"
EVAL_PAIRS="${EVAL_PAIRS:-$ROOT/input/pairs.eval.jsonl}"
SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-$ROOT/input/manifest.json}"
OUTPUT="${OUTPUT:-$ROOT/out/$RUN_ID}"
VENV="${VENV:-$ROOT/venv}"
CACHE="${CACHE:-$ROOT/cache}"

mkdir -p "$ROOT/out" "$CACHE"
test -f "$TRAIN_PAIRS"
test -f "$EVAL_PAIRS"
test -f "$SAMPLE_MANIFEST"
test -f "$ROOT/source/benchmarks/model2vec_training/requirements-gpu.lock"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip git curl

LOCK="$ROOT/source/benchmarks/model2vec_training/requirements-gpu.lock"
LOCK_SHA="$(sha256sum "$LOCK" | cut -d' ' -f1)"
if [[ ! -x "$VENV/bin/python" ]] || [[ ! -f "$VENV/.requirements-sha256" ]] || \
  [[ "$(<"$VENV/.requirements-sha256")" != "$LOCK_SHA" ]]; then
  python3 -m venv --clear "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
  "$VENV/bin/python" -m pip install -r "$LOCK"
  printf '%s\n' "$LOCK_SHA" > "$VENV/.requirements-sha256"
fi

export HF_HOME="$CACHE/huggingface"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

TEACHER="$CACHE/models/qwen3-embedding-8b"
STUDENT="$CACHE/models/potion-code-16m-v2"
"$VENV/bin/hf" download Qwen/Qwen3-Embedding-8B \
  --revision 1d8ad4ca9b3dd8059ad90a75d4983776a23d44af \
  --local-dir "$TEACHER"
"$VENV/bin/hf" download minishlab/potion-code-16M-v2 \
  --revision e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b \
  --local-dir "$STUDENT"
"$VENV/bin/hf" cache verify Qwen/Qwen3-Embedding-8B \
  --revision 1d8ad4ca9b3dd8059ad90a75d4983776a23d44af \
  --local-dir "$TEACHER"
"$VENV/bin/hf" cache verify minishlab/potion-code-16M-v2 \
  --revision e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b \
  --local-dir "$STUDENT"

cd "$ROOT/source"
PYTHONPATH=src:. "$VENV/bin/python" -m benchmarks.model2vec_training.train_smoke \
  --train-pairs "$TRAIN_PAIRS" \
  --eval-pairs "$EVAL_PAIRS" \
  --sample-manifest "$SAMPLE_MANIFEST" \
  --teacher-model "$TEACHER" \
  --student-model "$STUDENT" \
  --output "$OUTPUT" \
  --source-revision "$SOURCE_REVISION" \
  --run-id "$RUN_ID"

CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:. "$VENV/bin/python" -m benchmarks.model2vec_training.verify_smoke \
  --model "$OUTPUT/model" \
  --eval-pairs "$EVAL_PAIRS" \
  --output "$OUTPUT/cpu-verification.json" \
  --source-revision "$SOURCE_REVISION" \
  --run-id "$RUN_ID"

"$VENV/bin/python" -m pip freeze > "$OUTPUT/environment.freeze.txt"
sha256sum \
  "$ROOT/source/benchmarks/model2vec_training/requirements-gpu.txt" \
  "$ROOT/source/benchmarks/model2vec_training/requirements-gpu.lock" \
  > "$OUTPUT/requirements.sha256"
tar -C "$ROOT/out" -czf "$ROOT/out/$RUN_ID.tar.gz" "$RUN_ID"
df -h /
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
