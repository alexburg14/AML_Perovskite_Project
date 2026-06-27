"""
Threshold-based target labels + sensitivity analysis.

The raw OQMD targets are continuous: ``band_gap`` (eV) and ``energy_above_hull``
(eV/atom). The notebooks originally derived booleans by *exact equality*
(``band_gap == 0`` -> metal, ``energy_above_hull == 0`` -> stable). Exact
equality is numerically brittle, and for stability it is far stricter than the
physical "likely synthesizable" criterion (commonly < 0.025 eV/atom ~ k_B*T at
room temperature).

These helpers derive the labels from the continuous columns with explicit
thresholds (so the metadata CSVs never need regenerating) and let you sweep the
threshold to see how the class balance -- and, optionally, a model's score --
respond.

Note: thresholding does NOT fix the well-known DFT band-gap underestimation
(you cannot recover a true gap from a DFT zero). Its value is (a) a physically
meaningful, numerically robust stability cutoff and (b) quantifying how much a
result hinges on an arbitrary boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Continuous source column + sensible sweep for each target.
_TARGETS = {
    "metal": ("band_gap", (0.0, 0.05, 0.1, 0.2)),            # eV
    "stable": ("energy_above_hull", (0.0, 0.025, 0.05, 0.1)),  # eV/atom
}


def is_metal(df: pd.DataFrame, threshold: float = 0.0) -> np.ndarray:
    """Metallic if the (DFT) band gap is at/below ``threshold`` eV.

    ``threshold=0.0`` reproduces the original ``band_gap == 0`` label, since
    gaps are non-negative.
    """
    return df["band_gap"].to_numpy() <= threshold


def is_stable(df: pd.DataFrame, threshold: float = 0.0) -> np.ndarray:
    """Stable/synthesizable if energy above hull is at/below ``threshold`` eV/atom.

    ``threshold=0.0`` reproduces the original exact-on-hull label; ``0.025`` is
    the common synthesizability cutoff.
    """
    return df["energy_above_hull"].to_numpy() <= threshold


def label_sensitivity(
    df: pd.DataFrame,
    target: str = "metal",
    thresholds=None,
    X=None,
    model=None,
    cv: int = 3,
    scoring: str = "balanced_accuracy",
    random_state: int = 42,
) -> pd.DataFrame:
    """Sweep the labelling threshold and report how the task changes.

    Always reports class balance and the majority-class baseline (needs only the
    metadata DataFrame). If ``X`` is given, also cross-validates a classifier at
    each threshold so you can see whether the *learnability* shifts, not just the
    balance.

    Parameters
    ----------
    df : DataFrame with the continuous source column for ``target``.
    target : {"metal", "stable"}.
    thresholds : iterable of cutoffs; defaults to a sensible sweep per target.
    X : optional feature matrix (rows aligned to ``df``) to enable the
        model-based columns.
    model : optional sklearn classifier; defaults to a balanced RandomForest.
    cv, scoring, random_state : cross-validation settings.

    Returns
    -------
    DataFrame, one row per threshold.
    """
    if target not in _TARGETS:
        raise ValueError(f"target must be one of {tuple(_TARGETS)}, got {target!r}")
    column, default_thresholds = _TARGETS[target]
    thresholds = default_thresholds if thresholds is None else thresholds

    label_fn = is_metal if target == "metal" else is_stable

    if X is not None:
        from sklearn.base import clone
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        if model is None:
            model = RandomForestClassifier(
                n_estimators=100, max_depth=10, class_weight="balanced",
                random_state=random_state, n_jobs=-1,
            )

    rows = []
    n = len(df)
    for t in thresholds:
        y = label_fn(df, t)
        n_pos = int(y.sum())
        n_neg = n - n_pos
        pos_frac = n_pos / n
        row = {
            "threshold": t,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "positive_frac": round(pos_frac, 4),
            "majority_baseline": round(max(pos_frac, 1 - pos_frac), 4),
        }
        if X is not None:
            # A degenerate (single-class) split can't be cross-validated.
            if n_pos == 0 or n_neg == 0:
                row[f"cv_{scoring}_mean"] = np.nan
                row[f"cv_{scoring}_std"] = np.nan
            else:
                scores = cross_val_score(
                    clone(model), X, y, cv=cv, scoring=scoring, n_jobs=-1
                )
                row[f"cv_{scoring}_mean"] = round(float(scores.mean()), 4)
                row[f"cv_{scoring}_std"] = round(float(scores.std()), 4)
        rows.append(row)

    return pd.DataFrame(rows)
