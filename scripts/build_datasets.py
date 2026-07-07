"""
Portable dataset generator for the ABX3 perovskite project.

Consolidates the dataset-generation cells of ``analysis/Test Skript.ipynb`` into
a single runnable, environment-agnostic script. Every path is resolved relative
to the repository root (derived from ``__file__``), so the same script runs
unchanged on any clone -- no hardcoded user directories.

Builds (in dependency order):
  1. perovskite_metadata_overview.csv            (+ perovskite_soap_features.npy)
  2. perovskite_metadata_overview_reduced.csv    (+ perovskite_soap_features_reduced.npy)
  3. perovskite_metadata_overview_reduced_CoulombM.csv (+ perovskite_coulomb_features_reduced.npy)
  4. perovskite_metadata_overview_reduced_EwaldM.csv   (+ perovskite_ewald_features_reduced.npy)

The reduced/Coulomb/Ewald builders read the full overview CSV, so step 1 must
run before steps 2-4.

Usage:
    python scripts/build_datasets.py                 # all steps -> _regen_output/
    python scripts/build_datasets.py --out DIR
    python scripts/build_datasets.py --steps overview,reduced
    python scripts/build_datasets.py --no-soap       # skip slow full SOAP npy (CSVs unaffected)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- portable paths: everything relative to the repo root -------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `from perovskite.structures import ...` works as a script

from perovskite.structures import iter_records, record_to_atoms  # noqa: E402

# One-time in-memory index of the structure cache. The notebook's
# load_by_entry_id() re-scans the whole JSONL on every lookup (O(N^2) over the
# full dataset); indexing once is O(N) and yields byte-identical Atoms objects
# because it goes through the same record_to_atoms().
_RECORDS: dict[int, dict] | None = None


def load_by_entry_id(entry_id: int):
    """Indexed replacement for perovskite.structures.load_by_entry_id (same output)."""
    global _RECORDS
    if _RECORDS is None:
        _RECORDS = {int(r["entry_id"]): r for r in iter_records()}
        print(f"[index] cached {len(_RECORDS)} structures from JSONL")
    rec = _RECORDS.get(int(entry_id))
    if rec is None:
        raise KeyError(entry_id)
    return record_to_atoms(rec)

OQMD_CSV = ROOT / "data" / "oqmd_abx3_data.csv"
SPECIES_FILE = ROOT / "data" / "structures" / "global_species_list.txt"

# SOAP hyper-parameters (unchanged from the notebook)
SOAP_KW = dict(periodic=True, r_cut=6.0, n_max=3, l_max=2, average="outer", sparse=False)

# Matrix-descriptor settings (unchanged from the notebook)
MAX_ATOMS = 30
MATRIX_KW = dict(n_atoms_max=MAX_ATOMS, permutation="sorted_l2", sparse=False)

# Cleaned element subset for the reduced datasets: no rare-earths / radioactives.
CLEAN_SPECIES = [
    "Ag", "Al", "As", "Au", "B", "Ba", "Be", "Bi", "Br", "C", "Ca", "Cd", "Cl",
    "Co", "Cr", "Cs", "Cu", "F", "Fe", "Ga", "Ge", "H", "Hf", "Hg", "I", "In",
    "Ir", "K", "Mg", "Mn", "Mo", "N", "Na", "Nb", "Ni", "O", "Os", "P", "Pb",
    "Pd", "Pt", "Rb", "Re", "Rh", "Ru", "S", "Sb", "Sc", "Se", "Si", "Sn", "Sr",
    "Ta", "Ti", "Tl", "V", "W", "Y", "Zn", "Zr",
]


def _entry_ids_from_oqmd() -> list[int]:
    """entry_ids live in the second column of the base OQMD CSV (matches notebook)."""
    df = pd.read_csv(OQMD_CSV)
    return df.iloc[:, 1].dropna().astype(int).tolist()


def _meta_row(idx: int, atoms) -> dict:
    bg = atoms.info.get("band_gap", np.nan)
    hull = atoms.info.get("stability", np.nan)
    return {
        "entry_id": idx,
        "name": atoms.info.get("name", np.nan),
        "formula": atoms.get_chemical_formula(),
        "spacegroup": atoms.info.get("spacegroup", np.nan),
        "crystal_structure": atoms.info.get("cs", np.nan),
        "band_gap": bg,
        "is_metal": bg == 0.0,
        "energy_above_hull": hull,
        "is_stable": hull == 0.0,
        "delta_e": atoms.info.get("delta_e", np.nan),
    }


# --- step 1: full overview (+ SOAP) ----------------------------------------
def build_overview(out: Path, include_soap: bool = True) -> Path:
    soap = None
    if include_soap:
        from dscribe.descriptors import SOAP
        species = json.loads(SPECIES_FILE.read_text())
        soap = SOAP(species=species, **SOAP_KW)

    ids = _entry_ids_from_oqmd()
    print(f"[overview] feature-engineering {len(ids)} structures (soap={include_soap})...")

    rows, vectors, missing = [], [], 0
    for i, idx in enumerate(ids):
        try:
            atoms = load_by_entry_id(idx)
        except (KeyError, FileNotFoundError):
            missing += 1
            continue
        if not atoms:
            continue
        rows.append(_meta_row(idx, atoms))
        if soap is not None:
            vectors.append(soap.create(atoms))
        if (i + 1) % 500 == 0:
            print(f"  -> {i + 1}/{len(ids)} processed...")

    print(f"[overview] done: {len(rows)} structures, {missing} missing in cache")
    csv_path = out / "perovskite_metadata_overview.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[overview] wrote {csv_path}")
    if soap is not None:
        npy = out / "perovskite_soap_features.npy"
        np.save(npy, np.array(vectors))
        print(f"[overview] wrote {npy}  shape={np.array(vectors).shape}")
    return csv_path


# --- shared filtered builder for reduced / coulomb / ewald ------------------
def _build_filtered(out: Path, overview_csv: Path, descriptor, out_csv: str,
                    out_npy: str, label: str, max_atoms: int | None,
                    dtype=np.float32) -> Path:
    """Filter the overview by element subset (+ optional atom cap), write the
    reduced CSV, and -- if a descriptor is given -- the matching feature matrix.

    Two-pass + preallocation keeps peak memory at one copy of the output matrix
    instead of doubling via a Python list (the full 8000 x ~49k SOAP matrix is
    ~3 GB at float64; float32 halves it so it fits an 8 GB machine). Structure
    loads are cheap (in-memory index) so the second pass is essentially free.
    """
    species_set = set(CLEAN_SPECIES)
    df = pd.read_csv(overview_csv)
    print(f"[{label}] filtering {len(df)} rows (max_atoms={max_atoms})...")

    # pass 1: determine which rows survive the filters
    kept_rows, kept_ids = [], []
    missing = too_large = wrong_elem = 0
    for i, row in df.iterrows():
        idx = int(row.iloc[0])
        try:
            atoms = load_by_entry_id(idx)
        except (KeyError, FileNotFoundError):
            missing += 1
            continue
        if not atoms:
            continue
        if max_atoms is not None and len(atoms) > max_atoms:
            too_large += 1
            continue
        if not set(atoms.get_chemical_symbols()).issubset(species_set):
            wrong_elem += 1
            continue
        kept_rows.append(row)
        kept_ids.append(idx)
        if (i + 1) % 1000 == 0:
            print(f"  -> {i + 1}/{len(df)} checked, {len(kept_rows)} kept...")

    print(f"[{label}] done: {len(kept_rows)} kept | missing={missing} "
          f"too_large={too_large} wrong_elements={wrong_elem}")
    csv_path = out / out_csv
    pd.DataFrame(kept_rows).to_csv(csv_path, index=False)
    print(f"[{label}] wrote {csv_path}")

    # pass 2: preallocate the feature matrix and fill it row by row
    if descriptor is not None:
        n_feat = descriptor.get_number_of_features()
        mat = np.empty((len(kept_ids), n_feat), dtype=dtype)
        print(f"[{label}] computing {label} features into {mat.shape} {mat.dtype} "
              f"(~{mat.nbytes/1e9:.2f} GB)...")
        for j, idx in enumerate(kept_ids):
            mat[j] = descriptor.create(load_by_entry_id(idx))
            if (j + 1) % 1000 == 0:
                print(f"  -> {j + 1}/{len(kept_ids)} features computed...")
        npy = out / out_npy
        np.save(npy, mat)
        print(f"[{label}] wrote {npy}  shape={mat.shape}")
    return csv_path


# --- step 2: reduced (SOAP) -------------------------------------------------
def build_reduced(out: Path, overview_csv: Path, compute: bool = True) -> Path:
    soap = None
    if compute:
        from dscribe.descriptors import SOAP
        soap = SOAP(species=CLEAN_SPECIES, **SOAP_KW)
    return _build_filtered(out, overview_csv, soap,
                           "perovskite_metadata_overview_reduced.csv",
                           "perovskite_soap_features_reduced.npy",
                           "reduced", max_atoms=None)


# --- step 3: reduced + Coulomb matrix --------------------------------------
def build_coulomb(out: Path, overview_csv: Path, compute: bool = True) -> Path:
    cm = None
    if compute:
        from dscribe.descriptors import CoulombMatrix
        cm = CoulombMatrix(**MATRIX_KW)
    return _build_filtered(out, overview_csv, cm,
                           "perovskite_metadata_overview_reduced_CoulombM.csv",
                           "perovskite_coulomb_features_reduced.npy",
                           "coulomb", max_atoms=MAX_ATOMS)


# --- step 4: reduced + Ewald sum matrix ------------------------------------
def build_ewald(out: Path, overview_csv: Path, compute: bool = True) -> Path:
    esm = None
    if compute:
        from dscribe.descriptors import EwaldSumMatrix
        esm = EwaldSumMatrix(**MATRIX_KW)
    return _build_filtered(out, overview_csv, esm,
                           "perovskite_metadata_overview_reduced_EwaldM.csv",
                           "perovskite_ewald_features_reduced.npy",
                           "ewald", max_atoms=MAX_ATOMS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "_regen_output"),
                    help="output directory (default: <repo>/_regen_output)")
    ap.add_argument("--steps", default="overview,reduced,coulomb,ewald",
                    help="comma-separated subset of: overview,reduced,coulomb,ewald")
    ap.add_argument("--no-soap", action="store_true",
                    help="skip SOAP .npy generation (CSV outputs are unaffected)")
    ap.add_argument("--csv-only", action="store_true",
                    help="produce only the CSV datasets; skip ALL descriptor (.npy) "
                         "computation. Requires no dscribe -- use when dscribe/numba "
                         "is unavailable in the environment.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    compute = not args.csv_only
    include_soap = compute and not args.no_soap
    print(f"repo root: {ROOT}")
    print(f"output dir: {out}")
    print(f"steps: {steps}")

    overview_csv = out / "perovskite_metadata_overview.csv"
    if "overview" in steps:
        overview_csv = build_overview(out, include_soap=include_soap)

    if any(s in steps for s in ("reduced", "coulomb", "ewald")):
        if not overview_csv.exists():
            sys.exit(f"error: {overview_csv} not found -- run the 'overview' step first")
    if "reduced" in steps:
        build_reduced(out, overview_csv, compute=include_soap)
    if "coulomb" in steps:
        build_coulomb(out, overview_csv, compute=compute)
    if "ewald" in steps:
        build_ewald(out, overview_csv, compute=compute)

    print("\nAll requested steps complete.")


if __name__ == "__main__":
    main()
