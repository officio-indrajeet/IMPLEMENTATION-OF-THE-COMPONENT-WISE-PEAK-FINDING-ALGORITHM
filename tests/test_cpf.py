"""Correctness tests for the CPF package.

Run with either:  python -m unittest discover -s tests -v   |   pytest tests

The key property: every k-NN backend is *exact*, so swapping the backend must
not change the clustering.  These tests would have caught the two bugs in the
original dissertation code (graph built from the wrong data; squared distances
from Faiss used as if they were Euclidean).
"""

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cpf import CPFcluster, build_CCgraph, knn_search  # noqa: E402
from cpf.knn import faiss_available  # noqa: E402


# --------------------------------------------------------------------------- #
# A stand-in for the `faiss` module (used only when faiss is not installed).
# It mimics the two properties that matter: it insists on C-contiguous float32
# input, and IndexFlatL2.search returns SQUARED L2 distances.
# --------------------------------------------------------------------------- #
class _FakeIndexFlatL2:
    def __init__(self, d):
        self.d = d
        self.xb = None

    def add(self, x):
        assert x.dtype == np.float32 and x.flags["C_CONTIGUOUS"], "faiss needs C-contiguous float32"
        self.xb = x

    def search(self, x, k):
        assert x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
        d2 = (x ** 2).sum(1)[:, None] + (self.xb ** 2).sum(1)[None, :] - 2 * x @ self.xb.T
        d2 = np.maximum(d2, 0).astype(np.float32)
        idx = np.argsort(d2, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(d2, idx, axis=1), idx.astype(np.int64)


@contextlib.contextmanager
def faiss_or_fake():
    """Yield the name of the backend to test ('faiss'), faking the module if needed."""
    if faiss_available():
        yield "real faiss"
        return
    fake = type(sys)("faiss")
    fake.IndexFlatL2 = _FakeIndexFlatL2
    sys.modules["faiss"] = fake
    try:
        yield "fake faiss"
    finally:
        del sys.modules["faiss"]


def blobs(n=800, d=5, centers=5, std=1.4, seed=0):
    return make_blobs(n, d, centers=centers, cluster_std=std, random_state=seed)[0]


class KnnBackends(unittest.TestCase):
    def test_numpy_matches_sklearn(self):
        X = blobs()
        d1, i1 = knn_search(X, 10, "sklearn")
        d2, i2 = knn_search(X, 10, "numpy")
        np.testing.assert_allclose(d1, d2, atol=1e-6)
        np.testing.assert_array_equal(i1, i2)

    def test_faiss_returns_euclidean_not_squared(self):
        X = blobs()
        d_ref, _ = knn_search(X, 10, "sklearn")
        with faiss_or_fake():
            d_f, _ = knn_search(X, 10, "faiss")
        # float32 vs float64: loose tolerance, but squared distances would be way off
        np.testing.assert_allclose(d_f, d_ref, rtol=1e-3, atol=1e-3)

    def test_faiss_accepts_float64_noncontiguous_input(self):
        X = np.asfortranarray(blobs())  # float64, F-ordered
        with faiss_or_fake():
            d, idx = knn_search(X, 5, "faiss")
        self.assertEqual(d.shape, (X.shape[0], 5))
        self.assertEqual(d.dtype, np.float64)


class Graph(unittest.TestCase):
    def test_graph_is_mutual_and_has_euclidean_edge_lengths(self):
        X = blobs(400)
        k = 8
        _, CC, radius = build_CCgraph(X, k, cutoff=1, backend="sklearn")
        self.assertEqual((CC != CC.T).nnz, 0, "mutual k-NN graph must be symmetric")

        _, idx = knn_search(X, k, "sklearn")
        knn_sets = [set(r) for r in idx]
        coo = CC.tocoo()
        for i, j, w in zip(coo.row, coo.col, coo.data):
            self.assertIn(j, knn_sets[i])       # edge only if each is in the other's k-NN list
            self.assertIn(i, knn_sets[j])
            self.assertAlmostEqual(w, np.linalg.norm(X[i] - X[j]), places=6)

        # strictly fewer edges than the union graph
        n = X.shape[0]
        union = np.zeros((n, n), bool)
        for i, r in enumerate(idx):
            union[i, r] = True
        union |= union.T
        np.fill_diagonal(union, False)
        self.assertLess(CC.nnz, union.sum())

    def test_knn_radius_is_a_real_distance(self):
        X = blobs(300)
        k = 7
        _, _, radius = build_CCgraph(X, k, 1, "sklearn")
        i = 11
        kth = np.sort(np.linalg.norm(X - X[i], axis=1))[k - 1]
        self.assertAlmostEqual(radius[i], kth, places=6)


class BackendEquivalence(unittest.TestCase):
    """Exact search => identical clustering whichever backend is used."""

    def _compare(self, X, k=10, rho=0.4):
        ref = CPFcluster(k=k, rho=rho, backend="sklearn").fit(X)
        brute = CPFcluster(k=k, rho=rho, backend="numpy").fit(X)
        np.testing.assert_array_equal(ref.labels_, brute.labels_)
        with faiss_or_fake() as which:
            fa = CPFcluster(k=k, rho=rho, backend="faiss").fit(X)
        ari = adjusted_rand_score(ref.labels_, fa.labels_)
        self.assertGreaterEqual(ari, 0.99, f"{which}: ARI vs sklearn backend = {ari:.4f}")
        self.assertEqual((ref.labels_ == -1).sum(), (fa.labels_ == -1).sum())

    def test_low_dim_blobs(self):
        self._compare(blobs(900, 4, 5, 1.3, seed=1))

    def test_higher_dim_blobs(self):
        self._compare(blobs(900, 12, 6, 2.5, seed=2), k=12)

    def test_other_rho(self):
        self._compare(blobs(700, 3, 4, 1.0, seed=3), rho=0.6)

    def test_finds_the_obvious_clusters(self):
        X, y = make_blobs(900, centers=[[0, 0], [12, 0], [0, 12]], cluster_std=0.8, random_state=0)
        lab = CPFcluster(k=15, backend="numpy").fit(X).labels_
        ok = lab >= 0
        self.assertGreater(adjusted_rand_score(y[ok], lab[ok]), 0.95)


class Duplicates(unittest.TestCase):
    def test_zero_length_edges_between_duplicates_are_optional(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=(300, 3))
        X = np.vstack([base, base[:100]])               # 100 exact duplicates
        _, cc_legacy, _ = build_CCgraph(X, 5, 1, "sklearn")
        _, cc_keep, _ = build_CCgraph(X, 5, 1, "sklearn", keep_zero_length_edges=True)
        a, b = 0, 300                                   # X[0] == X[300]
        self.assertEqual(X[a].tolist(), X[b].tolist())
        self.assertEqual(cc_legacy[a, b], 0)            # dropped (original behaviour)
        self.assertGreater(cc_keep[a, b], 0)            # kept
        self.assertEqual(cc_keep.diagonal().sum(), 0)   # no self-loops either way

    def test_flag_is_a_noop_without_duplicates(self):
        X = blobs(500, 4, 4, 1.2, seed=7)
        a = CPFcluster(k=10, backend="sklearn").fit(X).labels_
        b = CPFcluster(k=10, backend="sklearn", keep_zero_length_edges=True).fit(X).labels_
        np.testing.assert_array_equal(a, b)


class LegacyEquivalence(unittest.TestCase):
    """The refactor must reproduce the original ``existing_CPF.py`` exactly."""

    def test_matches_original_implementation(self):
        legacy_dir = ROOT / "Turkey-Students-Datasets"
        legacy_file = legacy_dir / "existing_CPF.py"
        if not legacy_file.exists():
            self.skipTest("legacy script not present")
        try:
            import seaborn, matplotlib  # noqa: F401  (imported by the legacy module)
        except ImportError:
            self.skipTest("legacy script needs matplotlib and seaborn")

        from sklearn.metrics.pairwise import euclidean_distances

        had_in1d = hasattr(np, "in1d")
        if not had_in1d:  # the legacy code predates numpy 2.4
            np.in1d = np.isin
        sys.path.insert(0, str(legacy_dir))
        try:
            spec = importlib.util.spec_from_file_location("legacy_existing_cpf", legacy_file)
            legacy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(legacy)
            # numpy>=2 compat: legacy helper returned (m,1) arrays that can no longer be
            # assigned to scalar slots.  Same numbers, flattened.
            legacy.utils.density_broad_search_star = lambda ab: euclidean_distances(ab[1], ab[0]).ravel()

            X = blobs(600, 4, 5, 1.3, seed=5)
            with contextlib.redirect_stdout(io.StringIO()):
                legacy_model = legacy.CPFcluster(k=10, rho=0.4, n_jobs=1)
                legacy_model.fit(X)  # (legacy fit() does not return self)
            old = legacy_model.labels_
            new = CPFcluster(k=10, rho=0.4, backend="sklearn").fit(X).labels_
            np.testing.assert_array_equal(old, new)
        finally:
            sys.path.remove(str(legacy_dir))
            sys.modules.pop("utils", None)
            if not had_in1d:
                del np.in1d


if __name__ == "__main__":
    unittest.main()
