"""
Dataset loading helpers shared by the model notebooks.

Replaces the copy-pasted ``pd.read_csv(...) + np.load(...)`` block that appeared
once per descriptor (SOAP / Coulomb / Ewald) in every notebook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold

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
        hint = (
            "scripts/build_soap_pca.py"
            if descriptor == "soap_pca"
            else "the feature cells in notebooks/eda_and_features.ipynb"
        )
        raise FileNotFoundError(f"{feat_path} not found. Regenerate it via {hint}.")
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


def make_group_cv(df_meta: pd.DataFrame, n_splits: int = 3):
    """Composition-grouped CV splitter.

    Groups rows by ``name`` (chemical formula) so every polymorph of a composition
    lands in the same fold. This removes polymorph leakage: a random split scatters
    same-composition, near-identical structures across train and test, which
    optimistically inflates scores.

    Returns
    -------
    (cv, groups) : a ``GroupKFold(n_splits)`` and the group labels, both passed
        straight to ``cross_validate(..., cv=cv, groups=groups)`` /
        ``cross_val_predict(..., cv=cv, groups=groups)``.
    """
    groups = df_meta["name"].to_numpy()
    return GroupKFold(n_splits=n_splits), groups
