# Details

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `k` | 10 | neighbourhood size of the mutual k-NN graph (a point counts as its own first neighbour) |
| `rho` | 0.4 | fluctuation parameter in (0, 1] |
| `alpha` | 1 | edge-length cut-off factor |
| `cutoff` | 1 | components with ≤ `cutoff` points are noise (`-1`) |
| `backend` | `"sklearn"` | `"sklearn"` (kd-tree), `"faiss"` (IndexFlatL2), `"numpy"` (brute force, no dependencies), `"auto"` |
| `keep_zero_length_edges` | `False` | keep graph edges between exact duplicate points (see [Notes](#notes-and-caveats)) |
| `remove_duplicates` | `False` | run `np.unique(X, axis=0)` first (labels then refer to the unique rows) |

## Datasets

All nine datasets are from the UCI Machine Learning Repository. `cpf/datasets.py` applies the dissertation's preprocessing and returns a real reference label where one exists, so ARI/AMI are meaningful.

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
