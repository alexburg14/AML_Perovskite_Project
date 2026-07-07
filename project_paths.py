"""Central filesystem paths for the AML Perovskite project.

Import these instead of hard-coding absolute paths, so the notebooks and
scripts run on any machine, from any working directory.

Typical use inside a notebook (see the bootstrap cell each notebook now has)::

    from project_paths import DATA_DIR, MODELS_DIR
    df = pd.read_csv(DATA_DIR / "perovskite_metadata_overview_reduced.csv")
"""
from pathlib import Path

# Repo root = the folder containing this file.
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "analysis" / "figures"
