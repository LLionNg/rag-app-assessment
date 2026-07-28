from __future__ import annotations

import numpy as np

from src.retrieval.similarity import cosine_similarity, l2_normalize


def _reference(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """The formula the reference service uses: dot / (norm * norm), row by row."""
    return np.array(
        [row @ query / (np.linalg.norm(row) * np.linalg.norm(query)) for row in matrix]
    )


def test_matches_the_reference_formula():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(20, 1024)).astype(np.float32)
    query = rng.normal(size=1024).astype(np.float32)

    assert np.allclose(
        cosine_similarity(matrix, query), _reference(matrix, query), atol=1e-5
    )


def test_identical_vectors_score_one_and_opposites_minus_one():
    vector = np.array([0.3, 0.4, 0.5], dtype=np.float32)
    matrix = np.stack([vector, -vector])

    scores = cosine_similarity(matrix, vector)

    assert scores[0] == np.float32(1.0)
    assert np.isclose(scores[1], -1.0)


def test_orthogonal_vectors_score_zero():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    assert np.allclose(
        cosine_similarity(matrix, np.array([1.0, 0.0], np.float32)), [1.0, 0.0]
    )


def test_scale_is_irrelevant():
    matrix = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    small = cosine_similarity(matrix, np.array([1.0, 2.0, 3.0], np.float32))
    large = cosine_similarity(matrix * 1000, np.array([50.0, 100.0, 150.0], np.float32))

    assert np.isclose(small[0], large[0])


def test_zero_vectors_do_not_divide_by_zero():
    matrix = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    scores = cosine_similarity(matrix, np.array([0.0, 0.0], np.float32))

    assert np.all(np.isfinite(scores))
    assert scores[0] == 0.0


def test_empty_matrix_returns_empty():
    assert (
        cosine_similarity(np.zeros((0, 0), np.float32), np.zeros(4, np.float32)).size
        == 0
    )


def test_l2_normalize_gives_unit_rows():
    rng = np.random.default_rng(3)
    normalized = l2_normalize(rng.normal(size=(5, 8)))

    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
