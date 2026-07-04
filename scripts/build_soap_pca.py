"""
Compress the raw SOAP descriptor with PCA and save it as the `soap_pca` feature set.

The raw reduced SOAP matrix is (8000, 48870) float64 (~3.1 GB) -- too large and too
slow to train on directly (HistGradientBoosting needs ~6 min/fold). PCA to 500
components at float32 keeps ~95% of the variance in ~16 MB, making every downstream
model fit in seconds.

Caveat: the PCA is fit once on the full dataset (unsupervised transform), so CV
scores on `soap_pca` carry a mild optimism vs. re-fitting PCA per training fold.
Acceptable for exploration; noted in the notebooks.

Usage:
    python scripts/build_soap_pca.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perovskite import config
from perovskite.data import load_features_and_meta

N_COMPONENTS = 500
OUT = config.FEATURES_DIR / "soap_pca.npy"


def main() -> None:
    X, df_meta = load_features_and_meta("soap")
    print(f"raw SOAP: {X.shape} {X.dtype} ({X.nbytes / 1e9:.2f} GB)")

    pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=42)
    X_pca = pca.fit_transform(X).astype(np.float32)

    evr = float(pca.explained_variance_ratio_.sum())
    print(f"PCA -> {X_pca.shape} {X_pca.dtype} ({X_pca.nbytes / 1e6:.1f} MB)")
    print(f"cumulative explained variance: {evr:.4f}")

    np.save(OUT, X_pca)
    print(f"saved {OUT}")

    assert X_pca.shape[0] == df_meta.shape[0], "row count mismatch vs metadata"


if __name__ == "__main__":
    main()
