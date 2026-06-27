# AML Perovskite Project

Deep learning for **synthesizability and property prediction of ABX₃ perovskites**.
The goal is to predict the thermodynamic stability and electronic properties of
ABX₃ inorganic perovskites directly from chemical composition — bypassing
expensive DFT — while penalizing toxic and rare-earth elements.

## Repository contents

| Path | Description |
|------|-------------|
| `data/oqmd_abx3_data.csv` | Main dataset — 16,323 ABX₃ perovskites (OQMD), composition features + targets `Ef`, `Eg`, `Es`, crystal system. |
| `data/mp_abx3_data.csv` | Supplementary Materials Project set (structural / elastic properties). |
| `data/structures/oqmd_structures.jsonl` | Relaxed **atomic structures** (lattice + atomic positions) fetched from the OQMD API for 16,316/16,323 entries (99.96%). One JSON record per line, joinable to the CSV on `entry_id`. |
| `data/metadata_overview*.csv` | Reduced metadata tables (`band_gap`, `is_metal`, `energy_above_hull`, `is_stable`, …); one variant per descriptor (`_reduced`, `_reduced_CoulombM`, `_reduced_EwaldM`). |
| `data/features/` | Regenerated descriptor matrices (`*_reduced.npy` for SOAP / Coulomb / Ewald). Gitignored — rebuilt by `notebooks/eda_and_features.ipynb`. |
| `perovskite/` | Lightweight shared package: `config` (paths), `data` (feature/metadata loaders), `evaluation` (confusion-matrix + regression metrics), `structures` (JSONL → ASE `Atoms` loader/validator/visualizer). |
| `notebooks/` | EDA + feature engineering (`eda_and_features`) and the model notebooks (`band_gap_classification`, `kernel_svm`, `hull_regression`). |
| `scripts/fetch_oqmd_structures.py` | Reproducible fetcher for the structures above (resumable; handles rate-limits/outages). |
| `analysis/figures/` | Exploratory-data-analysis figures, incl. `fig10_*` polymorph structure visualizations. |
| `analysis/stats_summary.json` | Machine-readable summary of the EDA (distributions, ceilings, baselines). |

## Data source & citation

The primary dataset is taken from the **ML_abx3_dataset** repository, extracted
from the [Open Quantum Materials Database (OQMD)](https://oqmd.org/):

- Repository: https://github.com/chenebuah/ML_abx3_dataset
- Paper: https://doi.org/10.48550/arXiv.2312.11335

If you use this data, please cite the original authors:

> Ericsson Tetteh Chenebuah and David Tetteh Chenebuah.
> *An inorganic ABX3 perovskite materials dataset for target property prediction
> and classification using machine learning.* arXiv:2312.11335 (2023).

```bibtex
@misc{chenebuah2023inorganic,
      title={An inorganic ABX3 perovskite materials dataset for target property prediction and classification using machine learning},
      author={Ericsson Tetteh Chenebuah and David Tetteh Chenebuah},
      year={2023},
      eprint={2312.11335},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
}
```

This repository redistributes the dataset solely for academic coursework; all
credit for the data belongs to the original authors.
