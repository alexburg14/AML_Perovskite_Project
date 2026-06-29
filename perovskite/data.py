"""
Dataset loading helpers shared by the model notebooks.

Replaces the copy-pasted ``pd.read_csv(...) + np.load(...)`` block that appeared
once per descriptor (SOAP / Coulomb / Ewald) in every notebook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_features_and_meta(descriptor: str = "soap"):
    """Load a feature matrix and its metadata table for one descriptor.

    Parameters
    ----------
    descriptor : {"soap", "coulomb", "ewald"}
        Which precomputed descriptor to load.

    Returns
    -------
    X : np.ndarray
        Feature matrix (rows aligned to ``df_meta``).
    df_meta : pandas.DataFrame
        Reduced metadata overview (entry_id, name, targets, ...).
    """
    if descriptor not in config.META_CSV:
        raise ValueError(
            f"unknown descriptor {descriptor!r}; expected one of {config.DESCRIPTORS}"
        )

    meta_path = config.META_CSV[descriptor]
    feat_path = config.FEATURES_NPY[descriptor]

    df_meta = pd.read_csv(meta_path)
    if not feat_path.exists():
        raise FileNotFoundError(
            f"{feat_path} not found. Regenerate it via the feature cells in "
            f"notebooks/eda_and_features.ipynb."
        )
    X = np.load(feat_path)

    if X.shape[0] != df_meta.shape[0]:
        raise ValueError(
            f"row mismatch: {feat_path.name} has {X.shape[0]} rows but "
            f"{meta_path.name} has {df_meta.shape[0]}."
        )
    return X, df_meta


def make_split(X, y, test_size: float = 0.2, random_state: int = 42):
    """Thin wrapper around train_test_split with the project's defaults."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
