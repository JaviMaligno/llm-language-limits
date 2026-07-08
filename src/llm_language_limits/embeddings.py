from __future__ import annotations
import numpy as np


def cosine_drift(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    arr = np.asarray(vectors, dtype=float)
    ref = arr[0]
    ref_norm = np.linalg.norm(ref) or 1.0
    out = []
    for v in arr:
        vn = np.linalg.norm(v) or 1.0
        cos = float(np.dot(ref, v) / (ref_norm * vn))
        out.append(1.0 - cos)
    # Guarantee the first element is 0.0 (distance-to-self is 0 by definition)
    if out:
        out[0] = 0.0
    return out


def norm_trajectory(vectors: list[list[float]]) -> list[float]:
    return [float(np.linalg.norm(np.asarray(v, dtype=float))) for v in vectors]
