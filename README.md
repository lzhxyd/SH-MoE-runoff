# SH-MoE-runoff
Stable Feature Pool and Hydrology-Informed Mixture-of-Experts (SH-MoE) for robust runoff prediction in human-impacted basins.
# SH-MoE: Hydrology-Informed Mixture-of-Experts for Runoff Prediction

This repository contains the source code for the paper:

> Hydrology-informed Mixture-of-Experts for Robust Runoff Prediction under Nonstationary and Human-Impacted Conditions (submitted to *Water Resources Research*).

## 🚀 Features

- **Stable feature pool** using resampling-based feature stability screening.
- **Hydrology-informed gating** driven by seasonal cycles, flow-memory signals, and human-activity indicators.
- **Diverse expert pool** combining:
  - conventional ML models,
  - lag-feature-based ML models,
  - deep sequence models (LSTM, TCN).
- **Perturbation experiments** for robustness analysis under ±10% prior uncertainty.

## 📁 Repository Structure

'''
SH-MoE-runoff/
SH-MOE-Code/ # core implementation of the SH-MoE model (features, gating, experts, training, evaluation)
Data/ # demo data used for running minimal examples (no sensitive raw data)
LICENSE # open-source license (MIT)
README.md # project documentation
.gitignore
'''

## 🛠 Environment Setup

```bash
conda create -n sh-moe python=3.10

```markdown
## ▶️ Run a minimal example
python SH-MOE-Code/Main_ML.py
Demo data used in the example is stored under the `Data/` folder.

