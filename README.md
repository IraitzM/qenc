# How Quantum Data Encoding Constrains Representation Learning: code

This repository reproduces every experiment, table and figure of the article and
its Supplementary Information (SI). It simulates angle encoding with data
re-uploading, amplitude embedding and the quantum spectral model (QSM) on five
classification tasks, and measures where accuracy is lost along the pipeline:
in pooling, in the function class of the encoding, or in training.

Every simulation is noiseless and uses exact expectation values.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/IraitzM/qenc
cd qenc
uv sync
```

`uv.lock` pins the library versions that produced the published results
(PyTorch 2.12.1, PennyLane 0.45.0, scikit-learn 1.9.0, NumPy 2.4.6).
Everything runs on a CPU. Only the regime map benefits from a GPU.

## Data

Every dataset is downloaded on first use into `data/` and checked against its
MD5 checksum. Set `QENC_DATA` to use another folder.

| Task | Source |
|---|---|
| ECG | MIT-BIH Arrhythmia Database 1.0.0, PhysioNet (records 106, 119, 200, 208 for training, 203 and 223 for testing), read with `wfdb` |
| EEG | Bonn recordings in the UCI Epileptic Seizure Recognition release (Hugging Face mirror) |
| MNIST1D | official release (Greydanus and Wang, 2024) |
| PneumoniaMNIST, NoduleMNIST3D | MedMNIST, Zenodo record 10519652 (28-pixel archives, plus the 64 and 224 MedMNIST+ archives for the regime map) |
| HLS-CMDS | heart and lung recordings, Zenodo record 15376628 |

PhysioNet can be slow. At the 5 kB/s we measured, the six MIT-BIH records take about a quarter of an hour to
download on first use.

## Layout

```
src/qenc/       the library
  data.py         downloads and the data draw of every experiment
  pooling.py      block averaging A, its projection A^+A and the energy it discards
  models.py       the three encodings as PennyLane classifiers, and the QSM uploads
  protocols.py    draws, candidates and inputs of the five-task study and the spatial pilot
  measures.py     final states, Pauli and readout features, kernels, and the classical fits
  rank.py         two-qubit entangler ladder for the controlled-rank regressions
  fastsim/        batched torch statevector simulator used by the regime map
experiments/    one script per result bundle, with the paper's settings fixed inside
analysis/       tables.py (every table and number) and figures.py (the main-text figures)
```

Results go to `results/<bundle>/<job>/` (set `QENC_RESULTS` to change the folder), the
summaries to `results/summary/` and the figures to `results/figures/`. Each
experiment takes `--list`, `--job NAME` (repeatable) or `--all`. A finished job is skipped
and its results are never overwritten, so an interrupted run resumes.

## Reproducing the paper

Run the experiments in this order (the probes and the ladder analyze the models
saved by the first two), then the analysis:

```bash
uv run experiments/five_task.py --all
uv run experiments/spatial_pilot.py --all
uv run experiments/volume_followup.py --all
uv run experiments/detail.py --all
uv run experiments/probes.py --all
uv run experiments/ladder.py --all
uv run experiments/planted.py --all
uv run experiments/regime_map.py --all
uv run experiments/regime_map.py --ntk-norm
uv run experiments/witnesses.py --all
uv run experiments/eeg_controls.py --all
uv run experiments/hls.py --all
uv run experiments/rank_restarts.py --all
uv run experiments/checkerboard.py --all
uv run analysis/tables.py
uv run analysis/figures.py
```

Jobs are independent, so `--job` lets you spread an experiment over several processes or
machines.

| Article | Experiment | Summary in `results/summary/` | Time per job on one core |
|---|---|---|---|
| Fig. 1 | `analysis/figures.py` (schematic) | `results/figures/fig_pipeline.pdf` | seconds |
| Fig. 2, SI S4 | `ladder.py` (25 jobs) | `ladder_gaps.csv`, `ladder_levels.csv`, `tex/ladder_table.tex` | about 10 minutes |
| Sec. 3.1, SI S5 | `witnesses.py` | `witness_phase.csv`, `witness_label_rule.csv` | seconds |
| Sec. 3.1, SI S6 | `five_task.py` (25 jobs) | `five_task_pooling_screen.csv` (G_A) | 5 to 30 minutes |
| Fig. 3a, SI S7 | `detail.py` (10 jobs) | `detail_contrasts.csv` | about 1 minute |
| Sec. 3.1, SI S8 | `eeg_controls.py` (5 jobs), `hls.py` (10 jobs) | `eeg_controls.csv`, `hls.csv` | about 12 and 5 minutes |
| Fig. 4, SI S9 | `planted.py` (70 jobs) | `planted_summary.csv`, `planted_contrasts.csv` | 5 to 10 minutes |
| Sec. 3.2, SI S10 | `rank_restarts.py`, `checkerboard.py` | `rank_restarts.csv`, `tex/rank_table.tex`, `checkerboard.csv` | 7680 fits of 20 seconds, run in parallel; 10 minutes |
| Fig. 3b, SI S11 | `probes.py` (25 jobs) | `probe_contrasts.csv`, `tex/probe_table.tex`, `tex/resource_table.tex` | a few minutes |
| Fig. 5, SI S12 | `regime_map.py` (320 jobs) | `regime_verdicts.json`, `regime_summary.csv`, `tex/regime_*.tex` | minutes on a GPU; the 10-qubit QSM and 16-qubit jobs take hours and about 24 GB |
| SI S3 | `five_task.py`, `spatial_pilot.py` (10 jobs) | `tex/outcomes_table.tex`, `five_task_outcomes.csv`, `spatial_outcomes.csv` | see above; 2 minutes |
| SI S13 | `five_task.py`, `volume_followup.py` (5 jobs) | `validation_selection.csv`, `volume_followup.csv` | see above; 3 minutes |

With the pinned library versions, reruns of sampled jobs of every experiment reproduced the
archived scores and predictions exactly, on CPU and, for the regime map, on the same GPU model.
Floating-point differences between machines or thread counts can still flip an occasional
prediction that sits on a decision boundary.

## Archived results

The per-seed results, predictions and model checkpoints of the article are deposited
separately [archive DOI]. Placed under `results/` with the names below, they let
`probes.py`, `ladder.py` and the analysis run without retraining.

| Archive folder | Here |
|---|---|
| `geometry-selection/<task>-<seed>` | `five-task/<task>-<seed>` |
| `geometry-selection/volume-encoding/<seed>` | `volume-followup/<seed>` |
| `geometry-selection/{label,phase}_witness.csv` | `witnesses/` |
| `spatial-qsm/<task>-<seed>` | `spatial-pilot/<task>-<seed>` |
| `representation-bottlenecks/resolution/<task>-<seed>` | `detail/<task>-<seed>` |
| `representation-bottlenecks/family-probes/<task>-<seed>` | `probes/<task>-<seed>` |
| `representation-bottlenecks/ladder-pilot/<task>-<seed>` | `ladder/<task>-<seed>` |
| `representation-bottlenecks/planted-v2/<job>` | `planted/<job>` |
| `programme/phase1/<job>`, `programme/phase1-diagnostics/ntk_norm.csv` | `regime-map/<job>`, `regime-map/ntk_norm.csv` |
| `qnn-rankdesign/table_init_sensitivity_final_crz.csv` | `rank/fits.csv` |

## Citation

If you use this code, please cite the article. `CITATION.cff` holds the reference, which will be completed once
the article is published.

## Licence

MIT, see `LICENSE`.

## Acknowledgements

The checkerboard generator in `qenc.data.checkerboard` is adapted from
[kernel-trainer](https://github.com/IraitzM/kernel-trainer) (MIT license).
