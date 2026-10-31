# Phase 1 Run Status

**Verified on:** 31 October 2026 (Africa/Cairo)

**Project:** DSAI 403 Project 08, PSO-UNet-Image-Segmentation

**Environment:** Windows, Python 3.11.9, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu, Intel Core i5-1135G7, 8 GB RAM, no CUDA.

The complete Phase 1 baseline, PSO search, final model, and held-out comparison have run on the prepared Oxford-IIIT Pet dataset. The named course guide `NIC_Project8_PSO_UNet_Arabic_English.pdf` was not supplied, so the student team must still reconcile the report with the official handout and course AI-assistance rules.

## Dataset and protocol

The Oxford host timed out locally (`WinError 10060`). The original image and annotation archives were downloaded from [this mirror](https://huggingface.co/datasets/Shiym/ViT-FineTune/tree/main/oxford-iiit-pet) and verified against Torchvision's official MD5 values: `5c4f3ee8e5d25df40f4fd59a7f44e54c` and `95a8c909bbe2e81eed6a22bccdf3f68f`. Raw data and checkpoints remain local and are excluded from Git.

The fixed membership files record **2,944 training**, **736 validation**, and **3,669 official test** images, with zero split overlap. Both final models used seed 42, 128 × 128 inputs, batch size 16, the same compact U-Net, masked BCE plus soft Dice loss, AdamW, eight epochs, and best-validation-Dice checkpoint selection. The test split was accessed after both final configurations and checkpoints were fixed; it was not used for PSO fitness or model selection.

## Completed measurements

| Model and split | Best epoch / total | Mean Dice | Mean foreground IoU | Final training time | Search time |
|---|---:|---:|---:|---:|---:|
| Baseline, validation | 7 / 8 | 0.8017 | 0.6946 | 1,618.93 s | — |
| PSO selected, validation | 8 / 8 | 0.8516 | 0.7667 | 1,766.50 s | 2,441.94 s |
| Baseline, official test | 7 / 8 | 0.8084 | 0.7071 | 1,618.93 s | — |
| PSO selected, official test | 8 / 8 | 0.8547 | 0.7733 | 1,766.50 s | 2,441.94 s |

The PSO pilot completed **six** two-epoch candidate evaluations across three particles and two rounds. Its best candidate validation Dice was `0.7092`. The selected learning rate was `0.0010717589973807434`, bottleneck dropout `0.02825320436629486`, and weight decay `0.0008450201302139068`. Evaluation 5 repeated evaluation 2's best vector; no later candidate improved it. The final PSO model was initialized afresh and trained for the same eight-epoch budget as the baseline.

On the 3,669-image held-out test split, the PSO-selected model gained **0.0462 Dice** and **0.0662 foreground IoU** over the baseline (4.62 and 6.62 percentage points). Both test passes processed 230 batches and 3,669 valid images with zero skipped batches. The PSO path took `4,208.44` seconds for search plus final training, **2.60 times** the baseline's `1,618.93` seconds. These are single-seed results, not a confidence interval or proof of a general advantage.

The exact measurements are in `results/experiment_metadata.json`, `results/training_history.csv`, `results/pso_history.csv`, `results/pso_training_history.csv`, and `results/results.csv`. Corresponding plots and examples are in `figures/training_curves/` and `figures/predictions/`. Checkpoint SHA-256 values in metadata were checked against the local files before test evaluation.

## Verification

| Check | Status |
|---|---|
| Original archive checksums, image/mask pairing, and split non-overlap | Passed. |
| Local CPU test suite | **40 passed** after the full experiment and documentation update. |
| Full baseline | Eight history rows, completed metadata, and matching checkpoint hash verified. |
| PSO search | Six candidate history rows and selected vector recorded. |
| Full PSO final model | Eight history rows, completed metadata, matching search run ID, and checkpoint hash verified. |
| Held-out evaluation | Both models evaluated on 3,669 images; four measured comparison rows generated. |
| Notebook | Schema and syntax validated; selected cells executed locally. Full Colab notebook cells were not rerun because the shared CLI experiment was completed locally. |
| Report PDF | Regenerated as **8 A4 pages** and visually inspected after the measured conclusions were added. |
| GitHub Actions | CPU test workflow runs on push; check the latest commit's workflow result in GitHub Actions. |

An exploratory local subset run was also rerun with **128 training / 64 validation images for two epochs**. Its best validation Dice was `0.2982` at epoch 1. It has a different budget and is excluded from the full comparison. The metadata keeps snapshots of two interrupted duplicate training/search launches; their partial compute time is excluded from the finalized run-time measurements above. A duplicate test evaluator was stopped before writing a result; no test score was used to tune either model.

## Interpretation and remaining review

The selected settings improved both validation and test means in this run, at substantial search cost. Example masks still show over-segmentation and missed fine boundaries. The small six-evaluation search repeated one vector and cannot establish a global optimum; one random seed does not measure run-to-run variation. Repeating the complete protocol across seeds would test reliability. The official course handout remains unavailable for final student review.
