# Component-wise Peak-Finding (CPF) clustering with a Faiss backend

A Python implementation of the **Component-wise Peak-Finding (CPF)** density-based clustering
algorithm in which the expensive nearest-neighbour searches can be run with
[Faiss](https://github.com/facebookresearch/faiss) instead of scikit-learn's kd-tree.
It accompanies my MSc dissertation at Trinity College Dublin and ships with correctness tests and a
reproducible benchmark on nine UCI datasets.

## Background

CPF was introduced in:

> J. Tobin and M. Zhang, **[A Theoretical Analysis of Density Peaks Clustering and the Component-wise
> Peak-Finding Algorithm](https://www.tara.tcd.ie/items/bbde36f5-a6e4-4f26-b611-f16aff652732)**,
> *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 46(2):1109–1120, 2024.

CPF builds a mutual k-nearest-neighbour graph, splits it into connected components, and inside each
component finds density peaks (modal sets) and assigns every point to a peak by following its
"big brother" (nearest point of higher density). Its cost is dominated by k-NN searches, which is what
this project accelerates:

| Step | What it does | k-NN search? |
|------|--------------|--------------|
| 1. `build_CCgraph` | mutual k-NN graph → connected components | yes (whole data set) |
| 2. `get_density_dists_bb` | density distance + big brother of each point | yes (per component) |
| 3. `get_y` | choose peaks per component, follow big brothers | no |

All backends perform **exact** search, so swapping the backend must not change the clustering
(up to float32 rounding and ties — see [Notes](#notes-and-caveats)). The test-suite checks this.

## Install

```bash
git clone https://github.com/officio-indrajeet/IMPLEMENTATION-OF-THE-COMPONENT-WISE-PEAK-FINDING-ALGORITHM.git
cd IMPLEMENTATION-OF-THE-COMPONENT-WISE-PEAK-FINDING-ALGORITHM
pip install -r requirements.txt
pip install faiss-cpu          # optional, enables backend="faiss"
```

## Quick start

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from cpf import CPFcluster

X = StandardScaler().fit_transform(np.loadtxt("my_data.csv", delimiter=","))

model = CPFcluster(k=10, rho=0.4, backend="auto")   # "auto" = Faiss if installed, else scikit-learn
labels = model.fit_predict(X)                         # -1 marks noise, as in DBSCAN

print(model.n_clusters_, model.timings_)              # per-step timings in seconds
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `k` | 10 | neighbourhood size of the mutual k-NN graph (a point counts as its own first neighbour) |
| `rho` | 0.4 | fluctuation parameter in (0, 1] |
| `alpha` | 1 | edge-length cut-off factor |
| `cutoff` | 1 | components with ≤ `cutoff` points are noise (`-1`) |
| `backend` | `"sklearn"` | `"sklearn"` (kd-tree), `"faiss"` (IndexFlatL2), `"numpy"` (brute force, no dependencies), `"auto"` |
| `keep_zero_length_edges` | `False` | keep graph edges between exact duplicate points (see [Notes](#notes-and-caveats)) |
| `remove_duplicates` | `False` | run `np.unique(X, axis=0)` first (labels then refer to the unique rows) |

## Reproducing the benchmark

```bash
python benchmark.py                                  # all datasets, all installed backends
python benchmark.py --datasets obesity,mice_protein --repeats 5
python benchmark.py --backends sklearn,faiss --out results
```

For each dataset and backend it writes `results/benchmark.csv` and `results/benchmark.md` with the
median per-step timings, the number of clusters and noise fraction, **ARI / AMI against the real
labels** (non-noise points), silhouette and Davies–Bouldin indices, and `ARI_vs_sklearn`, which
tells you how closely each backend's clustering agrees with the scikit-learn one (1.0 = identical).

Whether Faiss helps depends on the data: `IndexFlatL2` is an optimised *brute-force* search, so it
tends to win as the dimensionality grows and can lose to a kd-tree in low dimensions. Run the benchmark
on your own machine and read the numbers per dataset rather than assuming a speed-up.

## Datasets

All nine datasets come from the UCI Machine Learning Repository and live in the dataset folders of
this repository; `cpf/datasets.py` reproduces the preprocessing used in the dissertation
(standardisation, one-hot encoding where needed) and returns a **real reference label** wherever one
exists, so that ARI/AMI are meaningful.

| Name (`--datasets`) | Folder | Reference label used for ARI/AMI |
|---------------------|--------|----------------------------------|
| `turkey` | `Turkey-Students-Datasets` | none (Likert survey answers) |
| `obesity` | `Obesity-Datasets` | `NObeyesdad` (7 levels) |
| `online_shoppers` | `Online-Shoppers-Datasets` | `Revenue` |
| `mice_protein` | `Mice-Protein-Datasets` | `class` (8 groups) |
| `shill_bidding` | `SHILL-BIDDING-DATASETS` | `Class` |
| `facebook_live` | `Facebook-Live-Datasets` | `status_type` (also a feature) |
| `anuran_frogs` | `Anuran-Frogs-Dataset` | `Species` (10) |
| `maths_learning` | `Maths-Learning-Datasets` | `Type of Answer` (10 % sample, as in the original script) |
| `ur3_cobotops` | `UR3-CORS-Datasets` | `Robot_ProtectiveStop` |

## Notes and caveats

* **Noise.** Points in graph components of at most `cutoff` points get label `-1`. They are excluded
  from ARI/AMI/silhouette and reported separately as `noise_frac`.
* **Duplicate rows.** Exact duplicates are at distance 0, and a zero-length edge stored in a sparse
  matrix is indistinguishable from "no edge", so the original CPF implementation drops it and
  duplicates become disconnected (on the Turkey survey data, with ~56 % duplicate rows, this makes most
  points noise). `keep_zero_length_edges=False` (default) reproduces that behaviour exactly;
  `True` keeps those edges. Which is appropriate is a modelling choice — compare both on your data.
  With many duplicates the exact k-NN *set* is also not unique (ties), so different backends can
  legitimately return slightly different neighbours and therefore slightly different clusterings;
  `ARI_vs_sklearn` in the benchmark makes this visible.
* **float32.** Faiss works in float32; the Faiss backend converts the data and returns float64 Euclidean
  distances (Faiss returns *squared* L2 distances, so the backend takes the square root, and sets each
  point's distance to itself to exactly 0). Near-ties can therefore resolve differently from the float64 kd-tree.
* Only exact (flat) search is implemented. For very large data, approximate indices (IVF, HNSW) would be
  the next step, at the price of exactness.

## Tests

```bash
python -m unittest discover -s tests -v        # or: pytest tests
```

The tests check that the graph is a true *mutual* k-NN graph with Euclidean edge lengths; that all
backends give the same clustering; that the Faiss wrapper returns Euclidean (not squared) distances
(using a faithful stand-in when `faiss` is not installed, and the real library when it is); and that the
refactored code reproduces the original `existing_CPF.py` labels exactly.

## Repository layout

```
cpf/                 the package: core.py (CPF), knn.py (k-NN backends), datasets.py (loaders)
benchmark.py         benchmark / evaluation script
tests/               correctness tests
docs/CHANGES.md      what was wrong in the dissertation scripts and what changed
legacy/              first simplified scripts (superseded)
<Dataset folders>/   data (CSV), notebooks and the original per-dataset dissertation scripts (superseded)
```

The per-dataset `Optimised_CPF.py` scripts and notebooks in the dataset folders are the code the
dissertation results were produced with. They are kept for reference but contain bugs, listed in
[`docs/CHANGES.md`](docs/CHANGES.md); please use the `cpf` package instead.

## Citation

If you use this code, please cite the original CPF paper and, if appropriate, the dissertation:

```bibtex
@article{tobin2024cpf,
  author  = {Tobin, J. and Zhang, M.},
  title   = {A Theoretical Analysis of Density Peaks Clustering and the Component-wise Peak-Finding Algorithm},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {46},
  number  = {2},
  pages   = {1109--1120},
  year    = {2024},
  url     = {https://www.tara.tcd.ie/items/bbde36f5-a6e4-4f26-b611-f16aff652732}
}

@mastersthesis{kumar2025cpf,
  author  = {Kumar, Indrajeet},
  title   = {Improving the Efficiency of Component-wise Peak-Finding Algorithms},
  school  = {School of Computer Science and Statistics, Trinity College Dublin},
  year    = {2025},
  note    = {MSc Computer Science dissertation; supervisor: Prof. Mimi Zhang}
}
```

## License

MIT — see [LICENSE](LICENSE).
