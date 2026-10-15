"""Component-wise Peak-Finding (CPF) clustering with a pluggable k-NN backend.

Reference
---------
J. Tobin and M. Zhang, "A Theoretical Analysis of Density Peaks Clustering and
the Component-wise Peak-Finding Algorithm", IEEE TPAMI 46(2):1109-1120, 2024.
https://www.tara.tcd.ie/items/bbde36f5-a6e4-4f26-b611-f16aff652732

The algorithm has three steps (names follow the paper / original code):

1. ``build_CCgraph``         mutual k-NN graph G(X, E) -> connected components
2. ``get_density_dists_bb``  density distance and "big brother" of every point
3. ``get_y``                 pick peaks per component, follow big brothers

Only the k-NN searches (steps 1 and 2) depend on the backend.  The searches are
*exact* for every backend, so ``backend="faiss"`` must give the same clustering
as ``backend="sklearn"`` (up to float32 rounding / distance ties); this is
checked in ``tests/test_cpf.py``.

Labels: ``labels_[i] == -1`` marks noise (points in components of size
<= ``cutoff``), as in DBSCAN.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from sklearn.metrics.pairwise import euclidean_distances

from .knn import knn_search, resolve_backend

_CHUNK = 100  # rows handled per block in the fallback big-brother search


# --------------------------------------------------------------------------- #
# Step 1
# --------------------------------------------------------------------------- #
def build_CCgraph(X, k, cutoff, backend="sklearn", n_jobs=1, keep_zero_length_edges=False):
    """Mutual k-NN graph, its connected components and every point's k-NN radius.

    ``keep_zero_length_edges``: exact duplicate points are at distance 0, and a
    zero-length edge stored in a sparse matrix is indistinguishable from "no edge",
    so the original implementation silently drops it (duplicates get disconnected;
    on duplicate-heavy data such as the Turkey survey this turns most points into
    noise).  ``False`` reproduces the original behaviour exactly; ``True`` keeps
    those edges (stored with a tiny positive weight).

    Returns
    -------
    components : (n,) float array, component id per point, NaN for outliers
    CCmat      : (n, n) CSR matrix of Euclidean edge lengths of the *mutual* graph
    knn_radius : (n,) distance to the k-th neighbour (the point itself counts as
                 the first neighbour, as in the original implementation)
    """
    n = X.shape[0]
    dist, idx = knn_search(X, k, backend, n_jobs)
    knn_radius = dist[:, k - 1]

    rows = np.repeat(np.arange(n), k)
    weights = dist
    if keep_zero_length_edges:
        zero_but_not_self = (dist == 0) & (idx != np.arange(n)[:, None])
        weights = np.where(zero_but_not_self, np.finfo(float).tiny, dist)
    knn_graph = scipy.sparse.csr_matrix((weights.ravel(), (rows, idx.ravel())), shape=(n, n))
    # Keep an edge only if BOTH endpoints list each other (mutual k-NN graph).
    # NB: a union graph (A | A.T) is a different, much better connected graph.
    CCmat = knn_graph.minimum(knn_graph.T).tocsr()

    _, components = scipy.sparse.csgraph.connected_components(
        CCmat, directed=False, return_labels=True
    )
    comp_labs, comp_count = np.unique(components, return_counts=True)
    outlier_components = comp_labs[comp_count <= cutoff]
    components = components.astype(float)
    components[np.isin(components, outlier_components)] = np.nan
    return components, CCmat, knn_radius


# --------------------------------------------------------------------------- #
# Step 2
# --------------------------------------------------------------------------- #
def get_density_dists_bb(X, k, components, knn_radius, backend="sklearn", n_jobs=1):
    """Density distance (``best_distance``) and big brother of every clustered point.

    The big brother of x is the nearest point (in the same component) whose
    k-NN radius is strictly smaller, i.e. the nearest point of higher density.
    The k-NN list usually contains one; if not, a chunked exact search over the
    whole component is used.  The point(s) of minimal radius have no big
    brother: they point to themselves with distance +inf.
    """
    n = X.shape[0]
    best_distance = np.full(n, np.nan)
    big_brother = np.full(n, -1, dtype=np.int64)

    comps = np.unique(components[~np.isnan(components)]).astype(int)
    for cc in comps:
        cc_idx = np.where(components == cc)[0]
        nc = len(cc_idx)
        Xc = X[cc_idx]
        kcc = min(k, nc - 1)

        distances, neighbors = knn_search(Xc, kcc, backend, n_jobs)
        cc_knn_radius = knn_radius[cc_idx]
        cc_best_distance = np.full(nc, np.nan)
        cc_big_brother = np.full(nc, -1, dtype=np.int64)

        # 1) first k-NN neighbour with a strictly smaller radius
        radius_diff = cc_knn_radius[:, None] - cc_knn_radius[neighbors]
        rows, cols = np.where(radius_diff > 0)
        rows, first = np.unique(rows, return_index=True)
        cols = cols[first]
        cc_big_brother[rows] = neighbors[rows, cols]
        cc_best_distance[rows] = distances[rows, cols]

        # 2) fallback: exact search for the points that found nobody in their k-NN list
        unresolved = np.setdiff1d(np.arange(nc), rows)
        for start in range(0, len(unresolved), _CHUNK):
            chunk = unresolved[start : start + _CHUNK]
            lower = cc_knn_radius[None, :] < cc_knn_radius[chunk][:, None]  # (m, nc)

            no_lower = np.flatnonzero(~lower.any(axis=1))
            keep = np.ones(len(chunk), dtype=bool)
            if no_lower.size:
                root = no_lower[0]
                for j in no_lower[1:]:  # ties at the minimal radius hang off the first one
                    lower[j, chunk[root]] = True
                cc_big_brother[chunk[root]] = chunk[root]
                cc_best_distance[chunk[root]] = np.inf
                keep[root] = False

            sel = np.flatnonzero(keep)
            if sel.size:
                D = euclidean_distances(Xc[chunk[sel]], Xc)
                D[~lower[sel]] = np.inf
                nearest = D.argmin(axis=1)
                cc_big_brother[chunk[sel]] = nearest
                cc_best_distance[chunk[sel]] = D[np.arange(sel.size), nearest]

        assert (cc_big_brother >= 0).all(), "every point must end up with a big brother"
        big_brother[cc_idx] = cc_idx[cc_big_brother]
        best_distance[cc_idx] = cc_best_distance
    return best_distance, big_brother


# --------------------------------------------------------------------------- #
# Step 3
# --------------------------------------------------------------------------- #
def get_y(CCmat, components, knn_radius, best_distance, big_brother, rho, alpha, d):
    """Choose the peaks (cluster centres) of every component and label the points."""
    n = components.shape[0]
    y_pred = np.repeat(-1, n)
    n_cent = 0
    comps = np.unique(components[~np.isnan(components)]).astype(int)

    for cc in comps:
        cc_idx = np.where(components == cc)[0]
        nc = len(cc_idx)
        if nc <= 2:
            y_pred[cc_idx] = n_cent
            n_cent += 1
            continue

        cc_knn_radius = knn_radius[cc_idx]
        cc_best_distance = best_distance[cc_idx]

        # big brother as a component-local index
        order = np.argsort(cc_idx)
        sorted_x = cc_idx[order]
        cc_big_brother = np.take(order, np.searchsorted(sorted_x, big_brother[cc_idx]), mode="clip")

        with np.errstate(divide="ignore", invalid="ignore"):
            peaked = cc_best_distance / cc_knn_radius
        peaked[(cc_best_distance == 0) & (cc_knn_radius == 0)] = np.inf

        not_tested = np.ones(nc, dtype=bool)
        cc_centers = [int(np.argmax(peaked))]
        not_tested[cc_centers[0]] = False

        CCmat_comp = CCmat[cc_idx, :][:, cc_idx].tocsr()  # sliced once, copied per iteration

        while not_tested.any():
            candidates = np.flatnonzero(not_tested)
            prop_cent = candidates[np.argmax(peaked[candidates])]
            CCmat_level = CCmat_comp.copy()

            if cc_knn_radius[prop_cent] > cc_knn_radius[~not_tested].max():
                level_set = np.where(cc_knn_radius <= cc_knn_radius[prop_cent])[0]
                n_cc, _ = scipy.sparse.csgraph.connected_components(
                    CCmat_level[level_set, :][:, level_set], directed=False, return_labels=True
                )
                if n_cc == 1:
                    break

            r = cc_knn_radius[prop_cent]
            v_cutoff = r / (rho ** (1 / d))
            e_cutoff = r / alpha
            if r > 0:
                CCmat_level.data[np.abs(CCmat_level.data) > e_cutoff] = 0
                cc_cut_idx = np.where(cc_knn_radius < v_cutoff)[0]
            else:
                CCmat_level.data[np.abs(CCmat_level.data) >= e_cutoff] = 0
                cc_cut_idx = np.where(cc_knn_radius <= v_cutoff)[0]
            CCmat_level.eliminate_zeros()
            CCmat_level = CCmat_level[cc_cut_idx, :][:, cc_cut_idx]

            _, cc_labels = scipy.sparse.csgraph.connected_components(
                CCmat_level, directed=False, return_labels=True
            )
            center_comp = cc_labels[np.isin(cc_cut_idx, cc_centers)]
            prop_cent_comp = cc_labels[np.where(cc_cut_idx == prop_cent)[0]]

            if np.isin(prop_cent_comp, center_comp):
                if peaked[prop_cent] == min(peaked[cc_centers]):
                    cc_centers.append(int(prop_cent))
                    not_tested[prop_cent] = False
                    continue
                break
            cc_centers.append(int(prop_cent))
            not_tested[prop_cent] = False

        cc_centers = np.array(cc_centers)
        BBTree = np.zeros((nc, 2), dtype=int)
        BBTree[:, 0] = np.arange(nc)
        BBTree[:, 1] = cc_big_brother
        BBTree[cc_centers, 1] = cc_centers
        Clustmat = scipy.sparse.csr_matrix(
            (np.ones(nc), (BBTree[:, 0], BBTree[:, 1])), shape=(nc, nc)
        )
        n_clusts, cc_y_pred = scipy.sparse.csgraph.connected_components(
            Clustmat, directed=True, return_labels=True
        )
        y_pred[cc_idx] = cc_y_pred + n_cent
        n_cent += n_clusts
    return y_pred


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #
class CPFcluster:
    """Component-wise Peak-Finding clustering.

    Parameters
    ----------
    k : int
        Neighbourhood size of the mutual k-NN graph (the point itself counts).
    rho : float in (0, 1]
        Fluctuation parameter (default 0.4).
    alpha : float
        Edge-length cut-off factor (default 1).
    cutoff : int
        Components with <= ``cutoff`` points are treated as noise (label -1).
    backend : {"sklearn", "faiss", "numpy", "auto"}
        Which exact k-NN implementation to use. ``"auto"`` picks Faiss if installed.
    n_jobs : int
        Threads for the scikit-learn backend (Faiss uses its own OpenMP threads).
    keep_zero_length_edges : bool
        Keep graph edges between exact duplicate points (see ``build_CCgraph``).
        Default ``False`` = identical to the original implementation.
    remove_duplicates : bool
        Run ``np.unique(X, axis=0)`` first.  Rows are then re-ordered/merged, so
        ``labels_`` refers to the unique rows, not to the input rows.

    Attributes
    ----------
    labels_ : (n,) int array, -1 for noise
    n_clusters_ : int, number of clusters (noise excluded)
    timings_ : dict with ``step1_s``, ``step2_s``, ``step3_s``, ``total_s``
    """

    def __init__(self, k=10, rho=0.4, alpha=1.0, cutoff=1, backend="sklearn",
                 n_jobs=1, remove_duplicates=False, keep_zero_length_edges=False):
        self.k = k
        self.rho = rho
        self.alpha = alpha
        self.cutoff = cutoff
        self.backend = backend
        self.n_jobs = n_jobs
        self.remove_duplicates = remove_duplicates
        self.keep_zero_length_edges = keep_zero_length_edges

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be an n x d array.")
        if self.remove_duplicates:
            X = np.unique(X, axis=0)
        n, d = X.shape
        if self.k > n:
            raise ValueError("k cannot be larger than n.")
        backend = resolve_backend(self.backend)

        t0 = time.perf_counter()
        self.components_, self.CCmat_, knn_radius = build_CCgraph(
            X, self.k, self.cutoff, backend, self.n_jobs, self.keep_zero_length_edges
        )
        t1 = time.perf_counter()
        best_distance, big_brother = get_density_dists_bb(
            X, self.k, self.components_, knn_radius, backend, self.n_jobs
        )
        t2 = time.perf_counter()
        self.labels_ = get_y(
            self.CCmat_, self.components_, knn_radius, best_distance, big_brother,
            self.rho, self.alpha, d,
        )
        t3 = time.perf_counter()

        self.backend_ = backend
        self.n_clusters_ = int(len(np.unique(self.labels_[self.labels_ >= 0])))
        self.timings_ = {
            "step1_s": t1 - t0,
            "step2_s": t2 - t1,
            "step3_s": t3 - t2,
            "total_s": t3 - t0,
        }
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_
