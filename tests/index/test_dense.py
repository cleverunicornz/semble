from pathlib import Path

import numpy as np
import pytest
from vicinity.backends.basic import BasicArgs

from semble.index.dense import SelectableBasicBackend


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Test save and load roundtrip."""
    vecs = np.random.default_rng(seed=42).normal(size=(10, 32))
    args = BasicArgs()
    selectable = SelectableBasicBackend(vecs, args)
    selectable.save(tmp_path)

    selectable_2 = SelectableBasicBackend.load(tmp_path)
    assert np.allclose(selectable.vectors, selectable_2.vectors)


def test_query_excludes_shadowed_indices_without_copying_visible_vectors() -> None:
    """Exclusion masking returns exact visible top-k and handles an all-shadowed corpus."""
    vectors = np.eye(4, dtype=np.float32)
    selectable = SelectableBasicBackend(vectors, BasicArgs())
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    indices, _scores = selectable.query(query, k=2, excluded=np.array([0], dtype=np.int_))[0]
    assert 0 not in indices
    assert len(indices) == 2

    indices, scores = selectable.query(query, k=2, excluded=np.arange(4, dtype=np.int_))[0]
    assert len(indices) == len(scores) == 0

    with pytest.raises(ValueError, match="cannot be combined"):
        selectable.query(
            query,
            k=1,
            selector=np.array([0], dtype=np.int_),
            excluded=np.array([1], dtype=np.int_),
        )
