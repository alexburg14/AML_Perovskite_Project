"""
Fetch relaxed atomic structures (lattice + atomic positions) from the OQMD REST
API for every entry in data/oqmd_abx3_data.csv, and cache them losslessly.

Strategy: the API cannot filter by entry_id (501), but `composition=<formula>`
returns all OQMD polymorphs for a formula. We query each of the ~3,501 unique
formulas and keep the records whose entry_id is in our dataset.

Output (raw, lossless JSON Lines, one structure per line):
    data/structures/oqmd_structures.jsonl
Each line: {name, entry_id, spacegroup, delta_e, stability, band_gap,
            unit_cell (3x3 lattice vectors), sites ["El @ x y z", ...]}

Resumable: already-processed formulas are recorded in
    data/structures/_done_formulas.txt   and skipped on re-run.
Failures are logged to data/structures/_errors.txt (not marked done -> retried).

No feature engineering is performed here — this only downloads and caches.
Run:  python scripts/fetch_oqmd_structures.py
"""
from __future__ import annotations
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "oqmd_abx3_data.csv"
OUT_DIR = ROOT / "data" / "structures"
OUT_JSONL = OUT_DIR / "oqmd_structures.jsonl"
DONE_FILE = OUT_DIR / "_done_formulas.txt"
ERR_FILE = OUT_DIR / "_errors.txt"

API = "http://oqmd.org/oqmdapi/formationenergy"
FIELDS = "name,entry_id,spacegroup,delta_e,stability,band_gap,unit_cell,sites"
HEADERS = {"User-Agent": "Mozilla/5.0 (perovskite-coursework structure fetch)"}
SLEEP = 0.6          # base politeness between successful requests
RETRIES = 3          # attempts for non-429 transient errors
RL_MAX_WAIT = 600    # cap on cumulative wait for a single 429'd request (s)


def api_get(params: dict) -> dict:
    """GET with patient handling of 429 rate-limiting (respects Retry-After)."""
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    waited = 0.0
    backoff = 10.0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and waited < RL_MAX_WAIT:
                ra = e.headers.get("Retry-After")
                wait = float(ra) if (ra and ra.isdigit()) else backoff
                time.sleep(wait)
                waited += wait
                backoff = min(backoff * 1.5, 120)
                continue
            raise


def fetch_formula(formula: str) -> list[dict]:
    """All OQMD records (with structure) for one composition."""
    d = api_get({"composition": formula, "fields": FIELDS, "limit": "200"})
    return d.get("data", [])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    needed_ids = set(df["entry_id"].astype(int))
    formulas = sorted(df["name"].unique())

    done = set()
    if DONE_FILE.exists():
        done = set(DONE_FILE.read_text().splitlines())
    # entry_ids already cached (so a resumed run doesn't double-write)
    have_ids = set()
    if OUT_JSONL.exists():
        with open(OUT_JSONL) as f:
            for line in f:
                try:
                    have_ids.add(int(json.loads(line)["entry_id"]))
                except Exception:
                    pass

    todo = [f for f in formulas if f not in done]
    print(f"formulas total={len(formulas)} done={len(done)} todo={len(todo)} "
          f"| target entry_ids={len(needed_ids)} cached={len(have_ids)}",
          flush=True)

    out = open(OUT_JSONL, "a")
    done_fp = open(DONE_FILE, "a")
    written = 0
    for i, formula in enumerate(todo, 1):
        for attempt in range(1, RETRIES + 1):
            try:
                records = fetch_formula(formula)
                for rec in records:
                    eid = int(rec["entry_id"])
                    if eid in needed_ids and eid not in have_ids:
                        out.write(json.dumps(rec) + "\n")
                        have_ids.add(eid)
                        written += 1
                done_fp.write(formula + "\n")
                if i % 50 == 0:
                    out.flush(); done_fp.flush()
                    print(f"[{i}/{len(todo)}] {formula}  "
                          f"cached_ids={len(have_ids)}/{len(needed_ids)} "
                          f"written_this_run={written}", flush=True)
                break
            except Exception as e:
                if attempt == RETRIES:
                    with open(ERR_FILE, "a") as ef:
                        ef.write(f"{formula}\t{type(e).__name__}\t{e}\n")
                    print(f"  FAIL {formula}: {type(e).__name__} {e}", flush=True)
                else:
                    time.sleep(SLEEP * 2 ** attempt)
        time.sleep(SLEEP)

    out.flush(); out.close(); done_fp.close()
    print(f"\nDONE. cached entry_ids={len(have_ids)}/{len(needed_ids)} "
          f"({len(have_ids)/len(needed_ids):.1%}) -> {OUT_JSONL}", flush=True)
    missing = len(needed_ids) - len(have_ids)
    if missing:
        print(f"NOTE: {missing} entry_ids not found via composition query "
              f"(see coverage; some OQMD entries may have been re-keyed).",
              flush=True)


if __name__ == "__main__":
    main()
