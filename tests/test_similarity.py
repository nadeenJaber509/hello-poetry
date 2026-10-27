
import numpy as np

from similarity import cosine_similarity


def test_identical_vectors():
    a = np.array([1, 2, 3])
    assert np.isclose(cosine_similarity(a, a), 1.0)


def test_orthogonal_vectors():
    a = np.array([1, 0])
    b = np.array([0, 1])
    assert np.isclose(cosine_similarity(a, b), 0.0)


def test_zero_vector():
    a = np.array([0, 0])
    b = np.array([1, 2])
    assert cosine_similarity(a, b) == 0.0


def test_similarity_is_symmetric():
    a = np.array([1, 2, 3])
    b = np.array([3, 2, 1])

    assert np.isclose(
        cosine_similarity(a, b),
        cosine_similarity(b, a),
    )
def test_opposite_vectors():
    a = np.array([1, 2])
    b = np.array([-1, -2])

    assert np.isclose(cosine_similarity(a, b), -1.0)