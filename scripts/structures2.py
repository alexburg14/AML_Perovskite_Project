"""
Loader + validator + visualizer for the cached OQMD atomic structures.

Reads data/structures/oqmd_structures.jsonl (one structure per line, joinable to
data/oqmd_abx3_data.csv on `entry_id`) and turns each record into an ASE Atoms
object using the lattice (`unit_cell`) and fractional `sites`.

Usage:
    python scripts/structures.py            # validate + render demo figures
    from scripts.structures import load_by_entry_id, record_to_atoms
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from ase import Atoms
import pandas as pd  # <--- NEU IMPORTIEREN für die CSV

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "oqmd_abx3_data.csv"  # <--- NEU: Pfad zur Basis-CSV
JSONL = ROOT / "data" / "structures" / "oqmd_structures.jsonl"
FIGDIR = ROOT / "analysis" / "figures"

# === NEU: Ein Mapping von entry_id -> cs aus der CSV erstellen ===
if SRC.exists():
    _df_base = pd.read_csv(SRC)
    # Erstellt ein schnelles Dictionary {647362: 'orthorhombic', ...}
    ID_TO_CS = dict(zip(_df_base["entry_id"], _df_base["cs"]))
else:
    ID_TO_CS = {}
    print(f"Warnung: {SRC} nicht gefunden. Kristallsysteme (cs) können nicht gemappt werden.")


def parse_site(s: str):
    """'Cs @ 0.5 0.5 0.5' -> ('Cs', [0.5, 0.5, 0.5])  (fractional coords)."""
    el, coords = s.split(" @ ")
    return el.strip(), [float(x) for x in coords.split()]


def record_to_atoms(rec: dict) -> Atoms:
    """OQMD JSONL record -> ASE Atoms (periodic, fractional positions)."""
    cell = np.array(rec["unit_cell"], dtype=float)          # 3x3 lattice vectors
    symbols, fracs = zip(*(parse_site(s) for s in rec["sites"]))
    atoms = Atoms(symbols=list(symbols),
                  scaled_positions=np.array(fracs),
                  cell=cell, pbc=True)
    
    # Standard-Metadaten aus der JSONL holen
    atoms.info.update({k: rec.get(k) for k in
                       ("name", "entry_id", "spacegroup", "band_gap",
                        "delta_e", "stability")})
    
    # === HIER DIE ÄNDERUNG: 'cs' dynamisch aus unserem CSV-Mapping anhängen ===
    eid = int(rec["entry_id"])
    atoms.info["cs"] = ID_TO_CS.get(eid, "Unbekannt")
    
    return atoms


def iter_records():
    with open(JSONL) as f:
        for line in f:
            yield json.loads(line)


def load_by_entry_id(entry_id: int) -> Atoms:
    for rec in iter_records():
        if int(rec["entry_id"]) == int(entry_id):
            return record_to_atoms(rec)
    raise KeyError(f"entry_id {entry_id} not in cache")


def records_for_formula(formula: str) -> list[dict]:
    return [r for r in iter_records() if r["name"] == formula]


# ---------------------------------------------------------------------------
def validate(n_check: int = 500):
    """Round-trip a sample: parse, and check formula stoichiometry is ABX3 (5/cell-unit)."""
    ok = bad = 0
    examples = []
    for i, rec in enumerate(iter_records()):
        if i >= n_check:
            break
        try:
            a = record_to_atoms(rec)
            assert len(a) >= 5 and len(a) % 5 == 0          # ABX3 -> multiples of 5 atoms
            assert np.linalg.det(a.cell) > 0                # non-degenerate cell
            ok += 1
            if len(examples) < 3:
                examples.append((rec["name"], rec["spacegroup"], len(a),
                                 round(a.get_volume(), 1)))
        except Exception as e:
            bad += 1
            print("  BAD:", rec.get("entry_id"), type(e).__name__, e)
    print(f"validated {ok}/{ok+bad} of first {n_check} records OK")
    for nm, sg, natoms, vol in examples:
        print(f"   e.g. {nm:8s} {sg:8s} {natoms:>3d} atoms  V={vol} A^3")


def visualize_polymorphs(formula: str = "CsSnI3", max_panels: int = 4):
    """Render distinct polymorphs of one formula side by side -> PNG.

    Same composition, different structure: the figure that motivates the whole
    'composition can't see polymorphs' argument. Each panel shows a 2x2x2
    supercell with the unit-cell box, a shared physical scale, and an element
    legend so the structural differences are legible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from ase.visualize.plot import plot_atoms
    from ase.data.colors import jmol_colors
    from ase.data import atomic_numbers

    # one representative per distinct space group, most stable first
    recs = sorted(records_for_formula(formula), key=lambda r: r.get("stability", 9))
    seen, picked = set(), []
    for r in recs:
        if r["spacegroup"] not in seen:
            seen.add(r["spacegroup"]); picked.append(r)
    picked = picked[:max_panels]
    if not picked:
        print(f"no cached structures for {formula}")
        return

    supercells = [record_to_atoms(r) * (2, 2, 2) for r in picked]
    span = max(np.ptp(a.get_positions(), axis=0).max() for a in supercells)
    rot = "12x,-8y,0z"

    n = len(picked)
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 4.4))
    if n == 1:
        axes = [axes]
    for ax, rec, atoms in zip(axes, picked, supercells):
        plot_atoms(atoms, ax, rotation=rot, radii=0.45, show_unit_cell=2)
        c = atoms.get_positions().mean(axis=0)              # centre, equal scale
        ax.set_xlim(c[0] - span / 2, c[0] + span / 2)
        ax.set_ylim(c[1] - span / 2, c[1] + span / 2)
        ax.set_aspect("equal"); ax.set_axis_off()
        ax.set_title(f"{rec['spacegroup']}\n$E_g$={rec.get('band_gap',0):.2f} eV   "
                     f"$E_{{hull}}$={rec.get('stability',0):.3f} eV/atom",
                     fontsize=10)

    elems = sorted({s for r in picked for s in
                    (sym for sym, _ in (parse_site(x) for x in r["sites"]))})
    handles = [Patch(facecolor=jmol_colors[atomic_numbers[e]], edgecolor="k", label=e)
               for e in elems]
    fig.legend(handles=handles, loc="lower center", ncol=len(elems),
               frameon=False, fontsize=11)
    fig.suptitle(f"{formula}: one composition → {n} distinct polymorphs "
                 f"(identical composition features, different structure & band gap)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    out = FIGDIR / f"fig10_{formula}_polymorphs.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}  ({n} polymorphs)")


if __name__ == "__main__":
    print("== validating cached structures ==")
    validate()
    print("\n== rendering polymorph visualizations ==")
    visualize_polymorphs("CsSnI3")
    visualize_polymorphs("BaTiO3")
