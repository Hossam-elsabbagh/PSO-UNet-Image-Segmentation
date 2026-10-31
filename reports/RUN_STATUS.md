# Phase 1 Run Status

**Verified on:** 31 October 2026 (Africa/Cairo); local experiments ran on 30 October.

**Project:** DSAI 403 Project 08, PSO-UNet-Image-Segmentation

**Environment:** Windows, Python 3.11.9, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu, Intel Core i5-1135G7, 8 GB RAM, CUDA unavailable.

This file distinguishes implemented code, local verification, and experiments that still need a practical training run. The named course guide `NIC_Project8_PSO_UNet_Arabic_English.pdf` was absent from the supplied workspace. The implementation follows the attached project brief; students must reconcile this draft with the official handout before submission.

## Dataset provenance and preparation

The default `torchvision.datasets.OxfordIIITPet` download from the Oxford host timed out locally (`WinError 10060`). The optional `python -m scripts.prepare_dataset --mirror huggingface` path fetched the original two archives from [this mirror](https://huggingface.co/datasets/Shiym/ViT-FineTune/tree/main/oxford-iiit-pet), verified their MD5 values against Torchvision's official resource definitions, and extracted them under `data/raw/oxford-iiit-pet/`:

| Archive | Verified MD5 |
|---|---|
| `images.tar.gz` | `5c4f3ee8e5d25df40f4fd59a7f44e54c` |
| `annotations.tar.gz` | `95a8c909bbe2e81eed6a22bccdf3f68f` |

`python -m scripts.prepare_dataset --mirror huggingface` and a subsequent `--no-download` reuse run both passed. Image/trimap pairing and split non-overlap checks passed. The persisted split counts are **2,944 train**, **736 validation**, and **3,669 official test**; all pairwise intersections have size zero. `figures/dataset_samples/examples.png` was generated and visually checked. The held-out test split was used only to record its official membership, not for training, checkpoint selection, or PSO fitness.

## Executed verification

| Check | Observed result |
|---|---|
| `python -m pytest -q` | **40 passed** on local CPU; no warnings on the final run. |
| `python -m scripts.train_baseline --smoke` | One synthetic-data epoch completed; validation Dice `0.6713518317`. This is not an Oxford result. |
| `python -m scripts.run_pso --smoke` | One synthetic candidate evaluation completed; validation Dice `0.6713518317`. This is not the configured six-evaluation pilot. |
| Notebook schema and syntax | `nbformat.validate` passed; all 22 code cells parsed. |
| Selected notebook cells | Environment, device, seed, real dataset preparation/exploration/split checks, trimap example, U-Net shape, loss, metrics, configuration, and PSO import executed locally. Full training cells remain to run in Colab. |
| PDF render | `Phase1_Report.pdf` rendered as **6 A4 pages** and was visually inspected. |
| `python -m scripts.generate_results` before full runs | Correctly declined to create `results.csv` without completed comparable final-model metadata. |

## Measured preliminary Oxford subset

`python -m scripts.preliminary_subset --no-download` trained for **one epoch** on **64 seeded training images** and evaluated **32 seeded validation images** drawn across the official development split. It selected its only epoch by validation Dice. The repeated run produced the same Dice and IoU, with a runtime difference expected from wall-clock measurement. This run is **not** the configured eight-epoch baseline and cannot be fairly compared with PSO.

| Measurement | Value |
|---|---:|
| Training loss | 0.6679 |
| Validation loss | 0.6503 |
| Validation Dice | 0.5150169435 |
| Validation foreground IoU | 0.3681280933 |
| Completed epochs | 1 |
| Training time | 9.06 s |
| Device / seed | CPU / 42 |

Evidence: `results/preliminary_history.csv`, `results/experiment_metadata.json`, `figures/training_curves/preliminary_subset_loss.png`, `figures/training_curves/preliminary_subset_validation_dice.png`, and `figures/predictions/preliminary_validation.png`. The local best checkpoint is excluded from Git. The displayed predictions are nearly all foreground, so the measured overlap does not indicate useful boundary quality.

## Acceptance status

| Component | Status | Evidence or remaining work |
|---|---|---|
| Environment and configuration | **PASS** | Python 3.11 virtual environment installed; `config.json` contains selected defaults; CPU fallback confirmed. |
| Official dataset loading and split records | **PASS** | Checksummed original archives, torchvision loading, paired files, 2,944/736/3,669 split records, zero overlap. |
| Preprocessing and augmentation | **PASS** | Synthetic tests and real sample figure verify dimensions, trimap mapping, ignored pixels, and joint flips. |
| Compact U-Net | **PASS** | Forward/backward and gradient tests passed; notebook output `(1,1,128,128)` with 1,962,625 parameters. |
| Masked losses and foreground metrics | **PASS** | Unit tests cover ignored pixels, zero-valid images/batches, overlap, empty foreground, and per-image aggregation. |
| Baseline training code and smoke path | **PASS** | Shared loop, best-checkpoint rule, incremental CSV, and synthetic smoke run verified. |
| Full eight-epoch Oxford baseline | **IMPLEMENTED, NOT EXECUTED** | Local CPU run would consume a substantial training budget; use Colab GPU or run `python -m scripts.train_baseline --no-download`. No full baseline score or checkpoint is claimed. |
| PSO mechanics and candidate integration | **PASS** | Six-evaluation configuration, bounds/bests/determinism tests, and synthetic candidate run passed. |
| Full PSO pilot and eight-epoch final model | **IMPLEMENTED, NOT EXECUTED** | Six two-epoch candidate runs plus eight final epochs remain; no selected hyperparameters or convergence curve are claimed. |
| Preliminary real-data result | **PASS** | 64/32-image one-epoch measurements and figures above; explicitly non-comparable. |
| Held-out test evaluation and final comparison | **IMPLEMENTED, NOT EXECUTED** | Script requires both completed, matching final runs and blocks repeated test evaluation. `results/results.csv` correctly remains absent. |
| Google Colab notebook | **PASS** | 23 ordered sections, schema/syntax validation and selected local cells passed; full Colab experiment remains unexecuted. |
| Technical report and diagrams | **PASS** | Markdown draft, visually checked six-page PDF, four Mermaid diagrams, real preliminary figures and student-review disclosure. |
| GitHub Actions | **PASS** | [Push run 41126969467](https://github.com/Hossam-elsabbagh/PSO-UNet-Image-Segmentation/actions/runs/41126969467) completed successfully for implementation commit `3b7ec9b`; its CPU dependency and synthetic-test steps passed. |
| Official course PDF review | **BLOCKED** | The named course PDF is absent; student must supply/review it and verify academic-integrity requirements. |
| GitHub push and verification | **PASS** | Initial implementation commit `3b7ec9b05fb453660565948242a03db4bc810fac` was pushed to `main`; local HEAD and `git ls-remote origin refs/heads/main` matched. This status update is a follow-up documentation commit. |

## Next steps

1. Run the full eight-epoch baseline on the prepared dataset, preferably with a GPU.
2. Run the configured six-evaluation PSO pilot and fresh eight-epoch final model.
3. Evaluate both finalized checkpoints once on the held-out test set, then generate `results/results.csv` and comparison figures.
4. Review and revise the academic report against the missing course guide and the course's AI-assistance policy.
