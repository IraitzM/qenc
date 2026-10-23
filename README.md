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
results/        the per-job results of the article, one folder per bundle
protocols/      the frozen protocols of its experiments, named by the identifiers the SI cites
```

Results go to `results/<bundle>/<job>/` (set `QENC_RESULTS` to change the folder), the
summaries to `results/summary/` and the figures to `results/figures/`. Each
experiment takes `--list`, `--job NAME` (repeatable) or `--all`. A finished job is skipped
and its results are never overwritten, so an interrupted run resumes.

## Reproducing the paper

`results/` already holds every job of the article, so the analysis rebuilds every table, quoted
number and figure in a few minutes, without retraining:

```bash
uv run analysis/tables.py
uv run analysis/figures.py
```

To rerun the experiments themselves, point `QENC_RESULTS` at another folder (in `results/` every
job is finished and would be skipped). Run them in this order, since the probes and the ladder
analyze the models saved by the first two, then the analysis on the same folder:

```bash
export QENC_RESULTS=$PWD/rerun
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
machines, and any job in `rerun/` can be compared file by file with the same job in `results/`.

| Article | Experiment | Summary in `results/summary/` | Time per job on one core |
|---|---|---|---|
| Fig. 1 | `analysis/figures.py` (schematic) | `results/figures/fig_pipeline.pdf` | seconds |
| Fig. 2, SI S4 | `ladder.py` (25 jobs) | `ladder_gaps.csv`, `ladder_levels.csv`, `tex/ladder_table.tex` | about 10 minutes |
| Sec. 3.1, SI S2, S5 | `witnesses.py` | `witness_phase.csv`, `witness_label_rule.csv` (S2: `results/witnesses/single_copy.json`) | seconds |
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
scores and predictions in `results/` exactly, on CPU and, for the regime map, on the same GPU model.
Floating-point differences between machines or thread counts can still flip an occasional
prediction that sits on a decision boundary. The planted-label jobs in `results/` ran on another
machine, and of three rerun on ours one matched exactly and two changed one validation and five
test predictions, by at most 0.7 points of test BA.

## Results and protocols

`results/` (14 MB) holds, for every job of the article, the files these scripts write: scores,
predictions, model checkpoints, splits and data-draw digests. Most bundles come from the original
runs, renamed to this layout and without the files these scripts do not use. `eeg-controls/`,
`hls/`, `checkerboard/` and `witnesses/` were run with this code, which reproduces the reported
numbers. The spatial pilot also kept the initial checkpoints of its QSM models, and `probes.py`
checks every saved model against its recorded data draw, splits, predictions and, where one was
kept, initialization before probing it.

`protocols/` holds the frozen protocols byte for byte, so the `protocol_sha256` recorded in the jobs
still verifies. Besides the design and the predictions, a protocol lists the hashes of the
development code that ran it, which is not part of this repository.

| Protocol | Script | Results |
|---|---|---|
| `geometry-validation-v2` | `five_task.py` | `five-task/` |
| `volume-encoding-followup-v1` | `volume_followup.py` | `volume-followup/` |
| `witnesses` | `witnesses.py` | `witnesses/` |
| `spatial-qsm-v1` | `spatial_pilot.py` | `spatial-pilot/` |
| `representation-bottlenecks-resolution-v1` | `detail.py` | `detail/` |
| `retrospective-three-family-probes-v1` | `probes.py` | `probes/` |
| `ladder-pilot-v1` | `ladder.py` | `ladder/` |
| `planted-labels-v2` | `planted.py` | `planted/` |
| `programme-phase1-v1` | `regime_map.py` | `regime-map/` |

## Citation

If you use this code, please cite the article. `CITATION.cff` holds the reference, which will be completed once
the article is published.

## Licence

MIT, see `LICENSE`.

## Acknowledgements

The checkerboard generator in `qenc.data.checkerboard` is adapted from
[kernel-trainer](https://github.com/IraitzM/kernel-trainer) (MIT license).
