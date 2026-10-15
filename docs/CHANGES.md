# What was wrong in the dissertation scripts, and what changed

This repository originally contained (a) one `Existing-CPF.py` / `Optimized-CPF.py` pair for the Turkey
data and (b) per-dataset copies (`<Dataset folder>/Optimised_CPF.py`, `existing_CPF.py`, notebooks) that
the dissertation results were produced with. The code now lives in the `cpf/` package. The old scripts
are kept for reference and carry a warning header.

Findings below come from reading the code, and from running the new package against the old
`existing_CPF.py` (see [Verification](#verification)).

## Bugs in the per-dataset `Optimised_CPF.py` (the Faiss version)

1. **The graph was built from the wrong data.**
   `CCmat = kneighbors_graph(neighbors, k, mode='distance', include_self=False)` passes the
   `n × k` array of neighbour *indices* returned by Faiss as if it were the data points, so scikit-learn
   computes nearest neighbours between *rows of index numbers*. The mutual k-NN graph — and therefore the
   components, every edge length, and everything in Step 3 — no longer describes the data.
   Fix: the graph is built directly from Faiss' `(distance, index)` result.
2. **Faiss returns squared distances.** `IndexFlatL2.search` returns squared L2 distances, but they were used
   as Euclidean ones: for `knn_radius` and for `best_distance` of points whose big brother is in their k-NN
   list, while points handled by the fallback search got true Euclidean distances (so `best_distance`
   mixed two units). Thresholds such as `knn_radius / rho**(1/d)` and `knn_radius / alpha` assume real
   distances. Fix: the Faiss backend takes the square root (and sets each point's distance to itself
   to exactly 0, since the float32 dot-product trick otherwise leaves ~1e-5 there).
3. **Cluster-count and quality differences between "Existing" and "Optimized" were therefore not
   a property of Faiss.** Faiss' flat index is exact, so a correct swap must give (up to ties and
   float32 rounding) the same clustering. Large differences (e.g. 48 → 1,185 clusters on Shill Bidding)
   point to the changed graph, not to an optimisation trade-off. The Step 3 code is identical in both
   versions, so Step 3 timing changes were caused by different inputs as well.

## Problems in the top-level `Optimized-CPF.py` / `Existing-CPF.py` (now in `legacy/`)

* The optimised version builds a **union** k-NN graph (`CCmat[i,j] = CCmat[j,i] = 1` for every neighbour)
  instead of the **mutual** one; it also uses squared distances, has no fallback big-brother search, and
  ignores `n_jobs`. Its README timing table did not match the dissertation tables.

## Evaluation

* **ARI/AMI were computed against random labels** (`np.random.randint(...)`), so they are ≈ 0 for every
  algorithm and say nothing about quality. `cpf/datasets.py` now uses the real labels that the data sets
  provide (see README); Turkey has none and is evaluated with internal indices only.
* **Noise was counted as a cluster.** CPF labels noise `-1`, but the metric code filtered with
  `~np.isnan(labels)` on an integer array (which filters nothing), so the `-1` group was counted as a
  cluster and fed into ARI/AMI/silhouette. Noise is now excluded from the metrics and reported as
  `noise_frac`.
* **Timing** is now the median of repeated runs after a warm-up, with per-step breakdown.
* **Maths-Learning** is a 10 % sample in the original script (955 rows), while Table 5.1 of the dissertation
  lists 9,546 instances; the loader keeps the sample and says so.

## Other changes

* Step 2 fallback search (points with no lower-radius neighbour in their k-NN list) is vectorised;
  the per-chunk `multiprocessing.Pool` is gone. The logic — including how ties at the minimal radius are
  handled — is unchanged.
* Step 3 slices each component's sub-graph once per component (and copies it per iteration) instead of
  re-slicing the full graph in every iteration.
* The legacy code does not run on current NumPy (`np.in1d` was removed; assigning a size-1 array to a
  scalar slot is an error). The package uses `np.isin` and plain scalars.
* Added: `tests/`, `benchmark.py`, `requirements.txt`, `LICENSE` (the old README said MIT), `.gitignore`;
  untracked committed `.DS_Store`, `.idea/` and `__pycache__/` files.
* New option `keep_zero_length_edges` (default `False` = original behaviour) — see the README caveat on
  duplicate rows.

## Verification

* `python -m unittest discover -s tests -v` — 12 tests. They check that the graph is mutual with
  Euclidean edge lengths, that the sklearn, numpy and Faiss backends give the same clustering, that the
  Faiss wrapper returns Euclidean distances, and that the new code reproduces the labels of the original
  `existing_CPF.py` **exactly** on test data (the legacy module is imported and run with two numpy-2
  compatibility shims).
* The tests fail when the original bugs are re-introduced (graph from neighbour indices, squared
  distances, union graph).
* With `backend="sklearn"` and the dissertation's parameters (`k=10, rho=0.4`) the new code reproduces
  the cluster counts of the dissertation's **Existing CPF** columns when the `-1` noise group is counted as
  a cluster, as the old metric code did:

  | Dataset | Dissertation (Existing CPF) | `cpf`: clusters + noise group |
  |---------|----------------------------|-------------------------------|
  | Obesity | 107 | 106 + 1 = 107 |
  | Mice protein | 43 | 42 + 1 = 43 |
  | Shill bidding | 48 | 47 + 1 = 48 |
  | Anuran frogs | 193 | 192 + 1 = 193 |
  | Maths learning | 175 | 174 + 1 = 175 |
  | UR3 CobotOps | 158 | 157 + 1 = 158 |
  | Facebook live | 183 (Table 5.8) / 186 (Table 3.2) | 185 + 1 = 186 |
  | Online shoppers | 136 | 133 + 1 = 134 |
  | Turkey | 279 | 282 + 1 = 283 |

  The three non-exact rows are the datasets with many duplicate rows (ties in the k-NN lists).

## What has *not* been verified here

The real Faiss library could not be installed in the environment these changes were made in, so the
Faiss backend was tested against a stand-in that mimics `IndexFlatL2` (float32 input, squared distances),
not against `faiss-cpu` itself. After `pip install faiss-cpu`, run the tests (they then use the real
library) and `python benchmark.py` to regenerate the timing and quality tables with Faiss. The speed-up
figures in the dissertation were produced by the buggy graph construction described above and should not
be relied on.
