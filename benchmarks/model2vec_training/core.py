"""Core data, training, export, and retrieval helpers for the GPU smoke run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from model2vec import StaticModel
from tokenizers import Tokenizer
from torch import nn


@dataclass(frozen=True, slots=True)
class Pair:
    """One query/code pair and its leakage-control group."""

    id: str
    query: str
    code: str
    group_id: str


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Bounded static-student training settings."""

    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 5
    min_delta: float = 1e-4
    seed: int = 42


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Training history and the selected checkpoint metadata."""

    initial_validation_loss: float
    best_validation_loss: float
    best_epoch: int
    epochs_completed: int
    history: tuple[dict[str, float | int], ...]


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pairs(path: Path) -> list[Pair]:
    """Read and validate prepared query/code JSONL."""
    pairs: list[Pair] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            group = raw.get("group")
            values: dict[str, Any] = {
                "id": raw.get("id"),
                "query": raw.get("query"),
                "code": raw.get("code"),
                "group_id": group.get("id") if isinstance(group, dict) else None,
            }
            invalid = [key for key, value in values.items() if not isinstance(value, str) or not value.strip()]
            if invalid:
                raise ValueError(f"{path}:{line_number}: missing non-empty fields: {', '.join(invalid)}")
            pair = Pair(**values)
            if pair.id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate pair id {pair.id!r}")
            seen_ids.add(pair.id)
            pairs.append(pair)
    if not pairs:
        raise ValueError(f"{path}: no pairs found")
    return pairs


def stable_group_split(pairs: list[Pair], validation_fraction: float, seed: int) -> tuple[list[Pair], list[Pair]]:
    """Split pairs into group-disjoint training and validation partitions."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups = sorted({pair.group_id for pair in pairs})
    if len(groups) < 2:
        raise ValueError("at least two groups are needed for a group-disjoint validation split")

    def score(group_id: str) -> int:
        payload = f"{seed}:student-validation:{group_id}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    denominator = 1 << 64
    validation_groups = {group for group in groups if score(group) / denominator < validation_fraction}
    if not validation_groups:
        validation_groups.add(min(groups, key=score))
    if len(validation_groups) == len(groups):
        validation_groups.remove(max(groups, key=score))

    train = [pair for pair in pairs if pair.group_id not in validation_groups]
    validation = [pair for pair in pairs if pair.group_id in validation_groups]
    if not train or not validation:
        raise ValueError("group split produced an empty partition")
    return train, validation


def pair_texts(pairs: list[Pair]) -> list[str]:
    """Flatten pairs into query then code text for teacher regression."""
    return [text for pair in pairs for text in (pair.query, pair.code)]


def pair_targets(query_targets: np.ndarray, code_targets: np.ndarray) -> np.ndarray:
    """Interleave query and code targets to match :func:`pair_texts`."""
    if query_targets.shape != code_targets.shape:
        raise ValueError("query and code target arrays must have identical shapes")
    output = np.empty((len(query_targets) * 2, query_targets.shape[1]), dtype=np.float32)
    output[0::2] = query_targets
    output[1::2] = code_targets
    return output


def unpadded_tokenizer(model: StaticModel) -> Tokenizer:
    """Clone a Model2Vec tokenizer with batching padding and truncation disabled."""
    tokenizer = Tokenizer.from_str(model.tokenizer.to_str())
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


def probable_pad_id(model: StaticModel) -> int:
    """Resolve the tokenizer padding ID used by the exported static model."""
    padding = model.tokenizer.padding
    if isinstance(padding, dict) and isinstance(padding.get("pad_id"), int):
        return int(padding["pad_id"])
    for token in ("[PAD]", "<pad>", "<|endoftext|>"):
        token_id = model.tokenizer.token_to_id(token)
        if token_id is not None:
            return token_id
    raise ValueError("student tokenizer has no resolvable padding token")


def tokenize_texts(model: StaticModel, texts: list[str], max_length: int) -> list[list[int]]:
    """Tokenize texts without inherited batch padding and with explicit truncation."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    tokenizer = unpadded_tokenizer(model)
    encoded = tokenizer.encode_batch_fast(texts, add_special_tokens=False)
    sequences: list[list[int]] = []
    for index, encoding in enumerate(encoded):
        ids = encoding.ids
        if model.unk_token_id is not None:
            ids = [token_id for token_id in ids if token_id != model.unk_token_id]
        ids = ids[:max_length]
        if not ids:
            raise ValueError(f"text at index {index} produced no usable student tokens")
        sequences.append(ids)
    return sequences


def _padded_batch(sequences: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create padded IDs and an explicit non-padding mask."""
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.float32, device=device)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        ids[row, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[row, :length] = 1
    return ids, mask


class StaticStudent(nn.Module):
    """Trainable static token vectors followed by a foldable linear projection."""

    def __init__(self, vectors: np.ndarray, pad_id: int, output_dim: int) -> None:
        """Initialize token vectors and an identity-compatible projection."""
        super().__init__()
        tensor = torch.from_numpy(np.asarray(vectors, dtype=np.float32).copy())
        tensor[pad_id].zero_()
        self.pad_id = pad_id
        self.embeddings = nn.Embedding.from_pretrained(tensor, freeze=False, padding_idx=pad_id)
        self.projection = nn.Linear(tensor.shape[1], output_dim)
        if tensor.shape[1] == output_dim:
            nn.init.eye_(self.projection.weight)
        else:
            nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool real tokens, apply the projection, and L2-normalize."""
        embedded = self.embeddings(ids) * mask.unsqueeze(-1)
        pooled = embedded.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return nn.functional.normalize(self.projection(pooled), dim=1)

    @torch.no_grad()
    def encode_sequences(self, sequences: list[list[int]], batch_size: int, device: torch.device) -> np.ndarray:
        """Encode token ID sequences in bounded batches."""
        self.eval()
        outputs: list[np.ndarray] = []
        for start in range(0, len(sequences), batch_size):
            ids, mask = _padded_batch(sequences[start : start + batch_size], self.pad_id, device)
            outputs.append(self(ids, mask).cpu().numpy())
        return np.concatenate(outputs).astype(np.float32, copy=False)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """Return finite row-normalized float32 vectors."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not len(array):
        raise ValueError("expected a non-empty 2D vector array")
    if not np.isfinite(array).all():
        raise ValueError("vector array contains non-finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("vector array contains a zero row")
    return array / norms


def initialize_ridge_projection(
    student: StaticStudent,
    sequences: list[list[int]],
    targets: np.ndarray,
    ridge: float = 1e-3,
) -> None:
    """Fit a global affine map from initial student texts to teacher targets."""
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    vectors = student.embeddings.weight.detach().cpu().numpy()
    pooled = np.stack([vectors[sequence].mean(axis=0) for sequence in sequences])
    if not np.isfinite(pooled).all() or np.any(np.linalg.norm(pooled, axis=1) == 0):
        raise ValueError("initial student text vectors must be finite and nonzero")
    pooled = pooled.astype(np.float64)
    normalized_targets = normalize_rows(targets).astype(np.float64)
    design = np.column_stack([pooled, np.ones(len(pooled), dtype=np.float64)])
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ normalized_targets)
    with torch.no_grad():
        student.projection.weight.copy_(torch.from_numpy(coefficients[:-1].T.astype(np.float32)))
        student.projection.bias.copy_(torch.from_numpy(coefficients[-1].astype(np.float32)))


def cosine_loss(predicted: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Return mean cosine distance."""
    predicted = nn.functional.normalize(predicted, dim=1)
    expected = nn.functional.normalize(expected, dim=1)
    return (1 - (predicted * expected).sum(dim=1)).mean()


@torch.no_grad()
def validation_loss(
    student: StaticStudent,
    sequences: list[list[int]],
    targets: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    """Evaluate cosine loss without changing model state."""
    predicted = student.encode_sequences(sequences, batch_size, device)
    expected = normalize_rows(targets)
    return float(np.mean(1 - np.sum(predicted * expected, axis=1)))


def train_student(
    student: StaticStudent,
    train_sequences: list[list[int]],
    train_targets: np.ndarray,
    validation_sequences: list[list[int]],
    validation_targets: np.ndarray,
    config: TrainConfig,
    device_name: str,
) -> TrainResult:
    """Train the static vectors with bounded epochs and validation early stopping."""
    if len(train_sequences) != len(train_targets) or len(validation_sequences) != len(validation_targets):
        raise ValueError("student sequences and targets have different lengths")
    if config.max_epochs < 1 or config.batch_size < 1 or config.patience < 1:
        raise ValueError("epochs, batch size, and patience must be positive")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(device_name)
    student.to(device)
    targets_tensor = torch.from_numpy(normalize_rows(train_targets))
    optimizer = torch.optim.Adam(student.parameters(), lr=config.learning_rate)
    initial_loss = validation_loss(student, validation_sequences, validation_targets, config.batch_size, device)
    best_loss = initial_loss
    best_epoch = 0
    best_state = {name: value.detach().cpu().clone() for name, value in student.state_dict().items()}
    history: list[dict[str, float | int]] = []
    stale_epochs = 0

    for epoch in range(1, config.max_epochs + 1):
        student.train()
        generator = torch.Generator().manual_seed(config.seed + epoch)
        order = torch.randperm(len(train_sequences), generator=generator).tolist()
        weighted_loss = 0.0
        samples = 0
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            sequences = [train_sequences[index] for index in indices]
            ids, mask = _padded_batch(sequences, student.pad_id, device)
            expected = targets_tensor[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = cosine_loss(student(ids, mask), expected)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                student.embeddings.weight[student.pad_id].zero_()
            weighted_loss += float(loss.item()) * len(indices)
            samples += len(indices)

        current_validation = validation_loss(
            student, validation_sequences, validation_targets, config.batch_size, device
        )
        current_training = weighted_loss / samples
        history.append({"epoch": epoch, "train_loss": current_training, "validation_loss": current_validation})
        if current_validation < best_loss - config.min_delta:
            best_loss = current_validation
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in student.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    student.load_state_dict(best_state)
    student.to("cpu")
    student.eval()
    return TrainResult(
        initial_validation_loss=initial_loss,
        best_validation_loss=best_loss,
        best_epoch=best_epoch,
        epochs_completed=len(history),
        history=tuple(history),
    )


def export_student(
    student: StaticStudent,
    base_model: StaticModel,
    output_path: Path,
    metadata: dict[str, Any],
) -> StaticModel:
    """Fold the affine head into every token vector and save a normal Model2Vec model."""
    if base_model.weights is not None or base_model.token_mapping is not None:
        raise ValueError("the smoke runner expects an unquantized base Model2Vec vocabulary")
    with torch.no_grad():
        vectors = student.projection(student.embeddings.weight).cpu().numpy()
    vectors[student.pad_id] = 0
    config = dict(base_model.config)
    config.update({"normalize": True, "semble_training": metadata})
    model = StaticModel(
        vectors=np.asarray(vectors, dtype=np.float16),
        tokenizer=base_model.tokenizer,
        config=config,
        normalize=True,
        base_model_name=base_model.base_model_name,
        language=base_model.language,
    )
    model.save_pretrained(output_path)
    return model


def retrieval_metrics(query_vectors: np.ndarray, document_vectors: np.ndarray) -> dict[str, float]:
    """Measure paired retrieval with one positive document at the matching row."""
    queries = normalize_rows(query_vectors)
    documents = normalize_rows(document_vectors)
    if queries.shape != documents.shape:
        raise ValueError("query and document arrays must have identical shapes")
    scores = queries @ documents.T
    ranks: list[int] = []
    for row in range(len(scores)):
        ordering = np.argsort(-scores[row], kind="stable")
        ranks.append(int(np.flatnonzero(ordering == row)[0]) + 1)
    rank_array = np.asarray(ranks)
    return {
        "recall_at_1": float(np.mean(rank_array <= 1)),
        "recall_at_5": float(np.mean(rank_array <= 5)),
        "recall_at_10": float(np.mean(rank_array <= 10)),
        "mrr": float(np.mean(1 / rank_array)),
        "mean_positive_cosine": float(np.mean(np.diag(scores))),
    }


def minimum_row_cosine(first: np.ndarray, second: np.ndarray) -> float:
    """Return the minimum cosine between corresponding rows."""
    left = normalize_rows(first)
    right = normalize_rows(second)
    if left.shape != right.shape:
        raise ValueError("row-cosine inputs must have identical shapes")
    return float(np.min(np.sum(left * right, axis=1)))


def static_retrieval_vectors(model: StaticModel, pairs: list[Pair]) -> tuple[np.ndarray, np.ndarray]:
    """Encode documents globally and queries singly, matching Semble's cold/query topology."""
    documents = np.asarray(
        model.encode([pair.code for pair in pairs], batch_size=len(pairs), use_multiprocessing=False),
        dtype=np.float32,
    )
    queries = np.vstack(
        [model.encode([pair.query], batch_size=1, use_multiprocessing=False)[0] for pair in pairs]
    ).astype(np.float32)
    return queries, documents


def static_batch_invariance(model: StaticModel, texts: list[str]) -> float:
    """Compare one global encode call with single-text encode calls."""
    together = np.asarray(
        model.encode(texts, batch_size=len(texts), use_multiprocessing=False),
        dtype=np.float32,
    )
    singly = np.vstack([model.encode([text], batch_size=1, use_multiprocessing=False)[0] for text in texts])
    return minimum_row_cosine(together, singly)
