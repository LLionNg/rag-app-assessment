from __future__ import annotations

import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row (or a lone vector) to unit length; all-zero rows pass through."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0.0, 1.0, norms)


def cosine_similarity(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity of `query` against every row of `matrix`.

    The same quantity as `dot(a, b) / (norm(a) * norm(b))`, and as pgvector's
    `1 - cosine_distance`, evaluated for the whole corpus in one matrix product.
    Normalisation lives here rather than at index time, so stored vectors stay
    exactly as the embedding model produced them.
    """
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    return l2_normalize(matrix) @ l2_normalize(query)
