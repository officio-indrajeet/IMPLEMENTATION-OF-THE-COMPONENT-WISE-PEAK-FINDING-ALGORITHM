"""Loaders for the nine UCI datasets used in the dissertation.

Each loader returns ``(X, y, info)``:

* ``X``  standardised feature matrix (float64), preprocessing as in the
  dissertation scripts,
* ``y``  *real* reference labels (never random ones) or ``None`` when the
  dataset has no meaningful ground truth,
* ``info`` short description of how ``y`` was chosen.

The CSV files live in the dataset folders of this repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent


def _scale(df: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(df.astype(float))


def turkey():
    d = pd.read_csv(ROOT / "Turkey-Students-Datasets" / "turkiye-student-evaluation_generic.csv")
    X = _scale(d.drop(columns=["instr", "class", "nb.repeat", "attendance", "difficulty"]))
    return X, None, "no ground truth (Likert survey answers)"


def obesity():
    d = pd.read_csv(ROOT / "Obesity-Datasets" / "ObesityDataSet_raw_and_data_sinthetic.csv")
    y = d["NObeyesdad"].to_numpy()
    X = _scale(pd.get_dummies(d.drop(columns=["NObeyesdad"]), drop_first=True))
    return X, y, "NObeyesdad (7 obesity levels)"


def online_shoppers():
    d = pd.read_csv(ROOT / "Online-Shoppers-Datasets" / "online_shoppers_intention.csv")
    num = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
           "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
           "SpecialDay"]
    cat = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(drop="first", sparse_output=False), cat)])
    X = pre.fit_transform(d.drop(columns=["Revenue"]))
    return X, d["Revenue"].astype(int).to_numpy(), "Revenue (2 classes)"


def mice_protein():
    d = pd.read_csv(ROOT / "Mice-Protein-Datasets" / "Data_Cortex_Nuclear.csv")
    y = d["class"].to_numpy()
    f = d.drop(columns=["MouseID", "Genotype", "Treatment", "Behavior", "class"])
    X = _scale(f.fillna(f.mean()))
    return X, y, "class (8 experimental groups)"


def shill_bidding():
    d = pd.read_csv(ROOT / "SHILL-BIDDING-DATASETS" / "Shill_Bidding_Dataset.csv")
    d = d.drop(columns=d.select_dtypes(exclude=[np.number]).columns).dropna()
    y = d["Class"].to_numpy()
    X = _scale(d.drop(columns=["Class"]))
    return X, y, "Class (normal vs shill bidding)"


def facebook_live():
    d = pd.read_csv(ROOT / "Facebook-Live-Datasets" / "Live_20210128.csv")
    y = d["status_type"].to_numpy()
    d = pd.get_dummies(d, columns=["status_type"], drop_first=True)
    cols = ["num_reactions", "num_comments", "num_shares", "num_likes", "num_loves", "num_wows",
            "num_hahas", "num_sads", "num_angrys"] + [c for c in d.columns if c.startswith("status_type_")]
    return _scale(d[cols].fillna(0)), y, "status_type (video/photo/link/status; also used as a feature)"


def anuran_frogs():
    d = pd.read_csv(ROOT / "Anuran-Frogs-Dataset" / "Frogs_MFCCs.csv")
    y = d["Species"].to_numpy()
    X = _scale(d.drop(columns=["Family", "Genus", "Species", "RecordID"]))
    return X, y, "Species (10 species)"


def maths_learning(sample_frac: float = 0.10, seed: int = 42):
    """MathE data. One-hot encoding of the text columns makes this very high-dimensional,
    so (as in the original script) a 10 % sample is used by default."""
    d = pd.read_csv(ROOT / "Maths-Learning-Datasets" / "MathE_dataset.csv",
                    delimiter=";", encoding="ISO-8859-1")
    if sample_frac < 1.0:
        d = d.sample(frac=sample_frac, random_state=seed)
    y = d["Type of Answer"].to_numpy()
    X = _scale(pd.get_dummies(d.drop(columns=["Type of Answer"]), drop_first=True))
    return X, y, f"Type of Answer (correct/wrong), {int(sample_frac * 100)}% sample"


def ur3_cobotops():
    d = pd.read_csv(ROOT / "UR3-CORS-Datasets" / "dataset_02052023.csv")
    d = d.loc[:, ~d.columns.str.contains("^Unnamed")].drop(columns=["Timestamp"]).dropna()
    y = d["Robot_ProtectiveStop"].to_numpy()
    X = _scale(d.drop(columns=["Robot_ProtectiveStop", "grip_lost"]))
    return X, y, "Robot_ProtectiveStop (binary)"


LOADERS = {
    "turkey": turkey,
    "obesity": obesity,
    "online_shoppers": online_shoppers,
    "mice_protein": mice_protein,
    "shill_bidding": shill_bidding,
    "facebook_live": facebook_live,
    "anuran_frogs": anuran_frogs,
    "maths_learning": maths_learning,
    "ur3_cobotops": ur3_cobotops,
}


def load(name: str):
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(LOADERS)}")
    return LOADERS[name]()
