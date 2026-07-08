import math
from llm_language_limits.embeddings import cosine_drift, norm_trajectory

def test_cosine_drift_first_is_zero():
    d = cosine_drift([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert d[0] == 0.0
    assert math.isclose(d[1], 0.0, abs_tol=1e-9)
    assert math.isclose(d[2], 1.0, abs_tol=1e-9)  # orthogonal → distance 1

def test_norm_trajectory():
    n = norm_trajectory([[3.0, 4.0], [0.0, 0.0]])
    assert math.isclose(n[0], 5.0)
    assert n[1] == 0.0
