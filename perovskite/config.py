"""
Central path configuration for the AML Perovskite project.

Every path is derived from the repository root (detected from this file's
location), so the notebooks and scripts run unchanged on any machine -- no more
hardcoded absolute user paths.
"""
from __future__ import annotations

from pathlib import Path

# perovskite/config.py -> parents[1] is the repo root
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"          # gitignored .npy feature matrices live here
STRUCTURES_DIR = DATA_DIR / "structures"
FIGURES_DIR = ROOT / "analysis" / "figures"

# Ensure the (gitignored) features directory exists so feature writes never fail
# on a fresh clone.
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Base tabular datasets
OQMD_CSV = DATA_DIR / "oqmd_abx3_data.csv"
MP_CSV = DATA_DIR / "mp_abx3_data.csv"
STRUCTURES_JSONL = STRUCTURES_DIR / "oqmd_structures.jsonl"
SPECIES_LIST = STRUCTURES_DIR / "global_species_list.txt"

# Reduced metadata overviews (one per descriptor variant)
META_CSV = {
    "soap": DATA_DIR / "metadata_overview_reduced.csv",
    "coulomb": DATA_DIR / "metadata_overview_reduced_CoulombM.csv",
    "ewald": DATA_DIR / "metadata_overview_reduced_EwaldM.csv",
}

# Feature matrices, keyed by descriptor. Regenerated on demand by the feature
# cells in notebooks/eda_and_features.ipynb (none ship in the repo -- gitignored).
FEATURES_NPY = {
    "soap": FEATURES_DIR / "soap_reduced.npy",
    "coulomb": FEATURES_DIR / "coulomb_reduced.npy",
    "ewald": FEATURES_DIR / "ewald_reduced.npy",
}

DESCRIPTORS = tuple(META_CSV)  # ("soap", "coulomb", "ewald")
