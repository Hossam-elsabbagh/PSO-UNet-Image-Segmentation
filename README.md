# PSO-UNet-Image-Segmentation

Phase 1 implementation for **DSAI 403: Nature-Inspired Computation**, Project 08, at Zewail City of Science, Technology and Innovation. The project studies whether a small particle swarm can select U-Net training hyperparameters that improve binary pet segmentation under a practical compute budget. The [repository](https://github.com/Hossam-elsabbagh/PSO-UNet-Image-Segmentation) contains code, synthetic-data tests, a Colab notebook, and a technical report draft.

## Overview

RGB pet photographs are converted to binary foreground masks with a compact U-Net. A fixed-parameter baseline and a PSO-selected model use the same data, preprocessing, loss, metrics, final training budget, and checkpoint rule. PSO searches *hyperparameters*, not network weights. Results are reported only when the corresponding run has completed.

## Research Problem

**Can Particle Swarm Optimization select U-Net hyperparameters that improve pet image segmentation relative to a fixed baseline while maintaining a practical computational budget?** Complex backgrounds, fur boundaries, pose, scale, lighting, and small compute budgets make this a useful test of automated tuning. Improvement is a hypothesis, not an assumption.

## Objectives

1. Build a reproducible Oxford-IIIT Pet binary segmentation pipeline.
2. Train a fixed 8-epoch compact U-Net baseline.
3. Search learning rate, bottleneck dropout, and weight decay with a 3-particle, 2-round PSO pilot using 2 epochs per candidate.
4. Retrain a fresh final model for 8 epochs with the selected parameters.
5. Compare per-image foreground Dice, IoU, training time, and PSO search time. Use the held-out test split only after model selection.

## Technologies

Python 3.11, PyTorch, torchvision, NumPy, Pillow, pandas, Matplotlib, pytest, and GitHub Actions. The code selects CUDA when available and otherwise uses CPU.

## Dataset

The [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) has 37 cat and dog breeds and pixel-level trimap annotations. The official 3,680-image development split is divided deterministically (seed 42) into 2,944 training and 736 validation images. The official 3,669-image test split remains untouched until final evaluation. `torchvision.datasets.OxfordIIITPet` downloads and reuses the original files under `data/raw/`. Download size is approximately 800 MB; the data are excluded from Git.

If the Oxford host is unreachable, `python -m scripts.prepare_dataset --mirror huggingface` fetches the same two archives from a mirror and checks their MD5 values against Torchvision's official checksums before extraction.

## Data Preprocessing

Images are resized to 128 × 128 with bilinear interpolation, converted to RGB float tensors in `[0, 1]`, and paired with masks resized by nearest neighbor. Trimap label `1` becomes foreground, `2` becomes background, and `3` is excluded using an explicit valid-pixel mask. A training-only horizontal flip applies jointly to image, target, and valid mask. Validation and test samples are not augmented. Split memberships are saved under `splits/` after dataset preparation.

## U-Net Architecture

The compact, randomly initialized U-Net uses encoder widths `16 → 32 → 64 → 128`, a 256-channel bottleneck with configurable dropout, and symmetric skip-connected decoder widths `128 → 64 → 32 → 16`. The final `1 × 1` convolution returns one raw logit per pixel. Sigmoid and a 0.5 threshold are applied at inference. The common loss is `0.5 × masked BCEWithLogits + 0.5 × masked soft Dice`; images with no valid pixels do not enter loss or metric averages.

## PSO Algorithm

A particle position is `[log10(learning_rate), dropout, log10(weight_decay)]`. Search bounds are `0.0001–0.003`, `0–0.3`, and `0.000001–0.001` respectively. The pilot uses 3 particles, 2 evaluation rounds, inertia `0.7`, and cognitive/social coefficients `1.5`. Fitness is `1 − validation Dice`. Each candidate starts from a fresh U-Net and receives the same 2-epoch budget. The selected parameters are subsequently used to train a new 8-epoch model. The candidate's short-run fitness is not a fair direct comparison with the baseline's 8-epoch score.

## System Pipeline

The shared data loader feeds two branches: fixed baseline training and PSO candidate evaluation followed by final training. Both select the best checkpoint by validation Dice. Only after both configurations are fixed should the test set be evaluated. Editable architecture diagrams are in [`figures/architecture/`](figures/architecture/).

## Repository Structure

```text
config.json                selected experiment defaults
src/                       preprocessing, dataset, U-Net, loss, metrics, training, PSO
scripts/                   dataset, training, search, evaluation, result entry points
tests/                     CPU tests using synthetic data
notebooks/phase1.ipynb     Colab workflow, calling the shared modules
reports/                   technical report draft, PDF, run status
figures/                   architecture and measured visualizations
splits/                    generated deterministic membership files
results/                   generated measured CSV and metadata files
checkpoints/               local weights excluded from Git
```

## Installation

On Windows PowerShell:

```powershell
cd "D:\University\Year 4\NIC"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

If script activation is restricted, call `.\.venv\Scripts\python.exe` directly. For a GPU, choose a compatible [PyTorch installation](https://pytorch.org/get-started/locally/); the default installation must still work on CPU.

## Baseline Training

```powershell
python -m scripts.prepare_dataset
python -m scripts.train_baseline
```

The full command uses `config.json`: batch size 16, AdamW, learning rate 0.001, weight decay 0.0001, dropout 0.10, and 8 epochs. `--smoke` runs a synthetic-data integration check without downloading Oxford-IIIT Pet or producing production result files; it must not be reported as a full baseline. `--no-download` reuses existing dataset files for a full run. Best weights are local only (`checkpoints/baseline_best.pt`). Training and validation history is written only when full training occurs.

The repository includes the completed Phase 1 test metadata. To start a **new independent experiment**, use a separate project copy, preserve its `results/experiment_metadata.json` as an archive, then remove that copy's active metadata file before training. The guard prevents a rerun from silently replacing a finalized test protocol; regenerate all final result artifacts after the new run.

For an optional exploratory Oxford data check on a CPU, `python -m scripts.preliminary_subset --no-download` runs a small seeded subset experiment. Its measured artifacts are saved separately as `preliminary_*` and are excluded from the eight-epoch baseline-versus-PSO comparison. The latest local exploratory rerun used 128 training and 64 validation images for two epochs.

## PSO Experiments

```powershell
python -m scripts.run_pso
```

The configured pilot performs six candidate evaluations before retraining its final model. Candidate runs are computationally expensive even though each has only two epochs. `--smoke` is a synthetic-data integration check that writes no production results. The completed full run stored search history, selected parameters, final model history, and a local best checkpoint.

## Evaluation

```powershell
python -m scripts.evaluate_models
python -m scripts.generate_results
```

Dice and foreground IoU are computed per image over valid pixels, then averaged over images with at least one valid pixel. If both prediction and target contain no foreground, both metrics are 1. The decision threshold is fixed at 0.5. The evaluation command requires both final checkpoints and completed validation selection before it opens the held-out test set.

## Google Colab Instructions

Open [`notebooks/phase1.ipynb`](notebooks/phase1.ipynb) in Colab after the repository is available on GitHub. Select a GPU runtime if desired and run cells in order. The notebook clones this repository, installs dependencies, detects CUDA or CPU, and calls the same project modules. Its training toggle defaults to off so the committed results can be inspected without replacing them. For a fresh Colab experiment, first follow the separate-copy metadata step above, then enable baseline, PSO, and final evaluation in order. Google Colab storage is temporary; download or copy new result artifacts before ending the session.

## Experimental Results

Both final models completed eight epochs on the fixed Oxford-IIIT Pet splits. The baseline selected epoch 7; the PSO model selected epoch 8 after six two-epoch candidate evaluations. On all 3,669 official held-out test images, the baseline achieved **0.8084 Dice / 0.7071 foreground IoU**, while the PSO-selected model achieved **0.8547 Dice / 0.7733 foreground IoU**. The PSO model gained **0.0462 Dice** and **0.0662 IoU** in this single-seed run. Its search took **2,441.94 s**, final training **1,766.50 s**, and combined path **2.60 times** the baseline's **1,618.93 s** training time.

The selected PSO settings were learning rate `0.001071759`, bottleneck dropout `0.0282532`, and weight decay `0.000845020`. The full four-row comparison is in [`results/results.csv`](results/results.csv); exact metadata, histories, plots, and example masks are under `results/` and `figures/`. See [`reports/RUN_STATUS.md`](reports/RUN_STATUS.md) for verification and limitations. Checkpoint binaries and raw data are intentionally absent from the public repository.

## Reproducibility

`config.json` is the authoritative default configuration. Dataset membership files, seed, device, actual epochs, runtime, and exact executed configuration are recorded with each run. A fresh model is initialized for every PSO candidate and final training. GPU kernels and data loading can still introduce some nondeterminism; record environment details when comparing independent runs.

## Phase 1 Deliverables

The [technical report draft](reports/Phase1_Report.md) follows the ten required academic sections. Its [PDF export](reports/Phase1_Report.pdf) is for review and can be regenerated after edits with `python reports/_export_pdf.py`. The original course PDF was not present in the workspace when this project was initialized, so students should reconcile this draft with the official handout and course academic-integrity rules before submission.

## Project Status

This README describes the implementation protocol and completed experiment. Download, test, CI, and push verification are tracked in [`reports/RUN_STATUS.md`](reports/RUN_STATUS.md).

## Future Work

Repeat the complete protocol across multiple seeds to measure run-to-run variation, test higher image resolution for fine boundaries, and decide whether the measured accuracy gain justifies the PSO search cost for the intended use.

## References

1. Kennedy, J., & Eberhart, R. (1995). [Particle swarm optimization](https://doi.org/10.1109/ICNN.1995.488968). *Proceedings of ICNN'95*, 1942–1948.
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).
3. Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). [Cats and Dogs](https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/). *CVPR*.
4. Saifullah, S., & Dreżewski, R. (2024). [Automatic Brain Tumor Segmentation Using Convolutional Neural Networks: U-Net Framework with PSO-Tuned Hyperparameters](https://doi.org/10.1007/978-3-031-70071-2_21). *PPSN XVIII*, 333–351.
5. Saifullah, S., & Dreżewski, R. (2025). [PSO-UNet: Particle Swarm-Optimized U-Net Framework for Precise Multimodal Brain Tumor Segmentation](https://arxiv.org/abs/2503.19152). *GECCO '25 Companion*.
