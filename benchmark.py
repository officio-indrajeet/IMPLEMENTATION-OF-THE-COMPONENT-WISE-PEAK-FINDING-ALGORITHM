#!/usr/bin/env python3
"""Benchmark CPF k-NN backends (scikit-learn kd-tree vs Faiss) on the nine datasets.

    python benchmark.py                         # all datasets, all installed backends
    python benchmark.py --datasets obesity,mice_protein --repeats 5
    python benchmark.py --backends sklearn,faiss --out results

For every (dataset, backend) it reports median step timings over ``--repeats`` runs,
clustering quality against the *real* labels where a dataset has them (ARI / AMI on
the non-noise points), internal indices (silhouette on a sample, Davies-Bouldin), and
how closely each backend's clustering agrees with the first backend's (ARI == 1.0
means identical).  Results are written to ``<out>/benchmark.csv`` and
``<out>/benchmark.md``.
"""

from __future__ import annotations

import argparse
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             davies_bouldin_score, silhouette_score)

from cpf import CPFcluster, datasets
from cpf.knn import available_backends


def quality(X, y, labels):
    ok = labels >= 0
    out = {"n_clusters": int(len(np.unique(labels[ok]))), "noise_frac": float(1 - ok.mean())}
    n_ok = len(np.unique(labels[ok]))
    if y is not None and n_ok > 0:
        out["ARI"] = adjusted_rand_score(y[ok], labels[ok])
        out["AMI"] = adjusted_mutual_info_score(y[ok], labels[ok])
    else:
        out["ARI"] = out["AMI"] = np.nan
    if 2 <= n_ok <= ok.sum() - 1:
        out["silhouette"] = silhouette_score(X[ok], labels[ok], sample_size=min(5000, int(ok.sum())),
                                             random_state=0)
        out["davies_bouldin"] = davies_bouldin_score(X[ok], labels[ok])
    else:
        out["silhouette"] = out["davies_bouldin"] = np.nan
    return out


def run_one(X, backend, k, rho, repeats, keep_zero=False):
    runs, labels = [], None
    for _ in range(repeats):
        m = CPFcluster(k=k, rho=rho, backend=backend, keep_zero_length_edges=keep_zero).fit(X)
        runs.append(m.timings_)
        labels = m.labels_
    med = {key: float(np.median([r[key] for r in runs])) for key in runs[0]}
    return med, labels


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default="all", help="comma-separated names or 'all'")
    p.add_argument("--backends", default=",".join(available_backends()),
                   help="comma-separated subset of sklearn,faiss,numpy (default: installed ones)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--rho", type=float, default=0.4)
    p.add_argument("--keep-zero-edges", action="store_true",
                   help="keep graph edges between exact duplicates (default: original behaviour)")
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)

    names = list(datasets.LOADERS) if args.datasets == "all" else args.datasets.split(",")
    backends = args.backends.split(",")
    if "faiss" in backends and "faiss" not in available_backends():
        sys.exit("faiss is not installed:  pip install faiss-cpu   (or drop it from --backends)")

    warnings.filterwarnings("ignore")
    rows = []
    # warm-up (imports, thread pools) so the first timed run is not penalised
    warm = np.random.default_rng(0).normal(size=(300, 4))
    for b in backends:
        CPFcluster(k=10, backend=b).fit(warm)

    for name in names:
        X, y, info = datasets.load(name)
        print(f"\n== {name}: n={X.shape[0]} d={X.shape[1]}  labels: {info}", flush=True)
        ref_labels = None
        for b in backends:
            t, labels = run_one(X, b, args.k, args.rho, args.repeats, args.keep_zero_edges)
            row = {"dataset": name, "n": X.shape[0], "d": X.shape[1], "backend": b, **t,
                   **quality(X, y, labels)}
            if ref_labels is None:
                ref_labels = labels
                row["ARI_vs_" + backends[0]] = 1.0
            else:
                row["ARI_vs_" + backends[0]] = adjusted_rand_score(ref_labels, labels)
            rows.append(row)
            print(f"  {b:8s} total={t['total_s']:7.3f}s (step1={t['step1_s']:.3f} step2={t['step2_s']:.3f} "
                  f"step3={t['step3_s']:.3f})  clusters={row['n_clusters']} noise={row['noise_frac']:.3f} "
                  f"ARI={row['ARI']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "benchmark.csv", index=False)

    env = (f"python {platform.python_version()}, numpy {np.__version__}, {platform.machine()}, "
           f"backends={backends}, repeats={args.repeats} (median), k={args.k}, rho={args.rho}, "
           f"keep_zero_length_edges={args.keep_zero_edges}")
    if "faiss" in backends:
        import faiss
        env += f", faiss {faiss.__version__}"
    with open(out / "benchmark.md", "w") as f:
        f.write(f"<!-- {env} -->\n\n")
        f.write(df.round(4).to_markdown(index=False) if hasattr(df, "to_markdown") else df.round(4).to_string(index=False))
        f.write("\n")
    print(f"\nwrote {out / 'benchmark.csv'} and {out / 'benchmark.md'}")


if __name__ == "__main__":
    main()
