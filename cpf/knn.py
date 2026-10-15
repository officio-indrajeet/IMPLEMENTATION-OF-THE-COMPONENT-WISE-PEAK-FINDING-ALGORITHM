"""Exact k-nearest-neighbour backends for CPF.

Every backend answers the same question: for each row of ``X``, which ``k``
rows of ``X`` are closest (the row itself included, as in
``sklearn.neighbors.NearestNeighbors.kneighbors(X)``)?

All backends return **Euclidean** (not squared) distances, sorted ascending:

    dist : (n, k) float64
    idx  : (n, k) int64

Faiss' ``IndexFlatL2`` returns *squared* L2 distances.  The Faiss backend takes
the square root so that all backends are interchangeable.  (Forgetting this was
one of the bugs in the original dissertation code: CPF compares distances with
``knn_radius / rho**(1/d)`` and edge lengths, which only makes sense for real
distances.)
"""

from __future__ import annotations

import numpy as np

BACKENDS = ("sklearn", "faiss", "numpy")


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


def available_backends() -> list[str]:
    out = ["sklearn", "numpy"]
    if faiss_available():
        out.insert(1, "faiss")
    return out


def resolve_backend(backend: str) -> str:
    """Map ``"auto"`` to the fastest installed backend."""
    if backend == "auto":
        return "faiss" if faiss_available() else "sklearn"
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS} or 'auto'")
    return backend


def knn_search(X: np.ndarray, k: int, backend: str = "sklearn", n_jobs: int = 1):
    """Return ``(dist, idx)``: the k nearest rows of X for every row of X."""
    backend = resolve_backend(backend)
    n = X.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")

    if backend == "sklearn":
        return _knn_sklearn(X, k, n_jobs)
    if backend == "faiss":
        return _knn_faiss(X, k)
    return _knn_numpy(X, k)


def _knn_sklearn(X, k, n_jobs):
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(
        n_neighbors=k, metric="euclidean", algorithm="kd_tree", n_jobs=n_jobs
    ).fit(X)
    dist, idx = nn.kneighbors(X)
    return dist, idx.astype(np.int64, copy=False)


def _knn_faiss(X, k):
    import faiss

    # Faiss only speaks C-contiguous float32.
    Xf = np.ascontiguousarray(X, dtype=np.float32)
    index = faiss.IndexFlatL2(Xf.shape[1])
    index.add(Xf)
    sq_dist, idx = index.search(Xf, k)
    # IndexFlatL2 returns SQUARED distances -> sqrt to get Euclidean distances.
    dist = np.sqrt(np.maximum(sq_dist, 0.0)).astype(np.float64)
    idx = idx.astype(np.int64, copy=False)
    # The float32 dot-product trick leaves a tiny non-zero distance from a point to
    # itself, which the square root inflates (1e-5 -> 3e-3).  It is exactly 0.
    dist[idx == np.arange(idx.shape[0])[:, None]] = 0.0
    return dist, idx


def _knn_numpy(X, k, block: int = 512):
    """Dependency-free exact brute force (reference implementation / tests)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    sq = np.einsum("ij,ij->i", X, X)
    dist = np.empty((n, k))
    idx = np.empty((n, k), dtype=np.int64)
    for s in range(0, n, block):
        e = min(s + block, n)
        d2 = sq[s:e, None] + sq[None, :] - 2.0 * (X[s:e] @ X.T)
        np.maximum(d2, 0.0, out=d2)
        part = np.argpartition(d2, k - 1, axis=1)[:, :k] if k < n else np.tile(np.arange(n), (e - s, 1))
        pd2 = np.take_along_axis(d2, part, axis=1)
        order = np.argsort(pd2, axis=1, kind="stable")
        idx[s:e] = np.take_along_axis(part, order, axis=1)
        dist[s:e] = np.sqrt(np.take_along_axis(pd2, order, axis=1))
    return dist, idx
