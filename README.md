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
| `analysis/figures/` | Exploratory-data-analysis figures. |
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
