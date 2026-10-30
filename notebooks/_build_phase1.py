"""Build the portable Phase 1 teaching notebook (developer helper)."""

from __future__ import annotations

import json
from pathlib import Path


cells: list[dict] = []


def md(title: str, body: str) -> None:
    cells.append({
        "cell_type": "markdown", "id": f"cell-{len(cells) + 1:02d}", "metadata": {},
        "source": [line + "\n" for line in (f"## {title}\n\n{body}").splitlines()],
    })


def code(source: str) -> None:
    cells.append({
        "cell_type": "code", "id": f"cell-{len(cells) + 1:02d}",
        "execution_count": None, "metadata": {},
        "outputs": [], "source": [line + "\n" for line in source.strip().splitlines()],
    })


md("1. Project Introduction", """**DSAI 403 Project 08.** This notebook runs the repository's shared Python modules for binary Oxford-IIIT Pet segmentation with a compact U-Net and a small Particle Swarm Optimization (PSO) search. A fixed baseline and the PSO-selected final model have the same eight-epoch final budget. Only completed runs produce measured results. The official course PDF was not available when this draft was prepared; review it before submission.""")

md("2. Environment Setup", """Open this notebook in Google Colab and choose **Runtime → Change runtime type → GPU** if one is available. On Colab, this cell clones the public repository. In a local checkout it reuses the current folder. The notebook never assumes a Windows drive path.""")
code("""
from pathlib import Path
import json, os, subprocess, sys

REPO_URL = "https://github.com/Hossam-elsabbagh/PSO-UNet-Image-Segmentation.git"
project_root = Path.cwd()
if not (project_root / "config.json").exists():
    project_root = Path.cwd() / "PSO-UNet-Image-Segmentation"
    if not project_root.exists():
        subprocess.run(["git", "clone", REPO_URL, str(project_root)], check=True)
os.chdir(project_root)
sys.path.insert(0, str(project_root))
print("Project root:", project_root)
""")

md("3. Dependency Installation", """Install the repository's declared dependencies into the current runtime. Colab normally has PyTorch already; `pip` will satisfy the requirements as needed. Restart the runtime if `pip` requests it, then rerun from the top.""")
code("""
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
""")

md("4. GPU Detection", """The same project code works on CUDA or CPU. CPU is suitable for tests and quick checks, while the complete dataset training and six PSO evaluations are much slower.""")
code("""
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__, "Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
""")

md("5. Random Seed Setup", """Use the selected seed throughout the protocol. The scripts also read this seed from `config.json` and record the executed configuration. Some GPU operations can remain nondeterministic.""")
code("""
import random
import numpy as np
config = json.loads(Path("config.json").read_text(encoding="utf-8"))
seed = int(config["project"]["seed"])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
print("Seed:", seed)
""")

md("6. Dataset Download", """Set `PREPARE_DATASET` to `True` to download or reuse the official Oxford-IIIT Pet files, create deterministic split memberships, verify pairing and non-overlap, and generate sample figures. The upstream image and annotation archives are roughly 800 MB. If the Oxford host is unreachable, set `DATA_MIRROR='huggingface'` to fetch archives whose MD5 values match Torchvision's official files. For synthetic smoke mode, set `PREPARE_DATASET=False` and later `SMOKE_MODE=True`. A download failure leaves the rest of this notebook usable for synthetic tests and code inspection.""")
code("""
PREPARE_DATASET = True
DATA_MIRROR = None  # or 'huggingface' if the Oxford server is unreachable
DATA_READY = False
if PREPARE_DATASET:
    command = [sys.executable, "-m", "scripts.prepare_dataset"]
    if DATA_MIRROR:
        command.extend(["--mirror", DATA_MIRROR])
    result = subprocess.run(command, check=False)
    DATA_READY = result.returncode == 0
    if not DATA_READY:
        print("Dataset preparation did not complete. Check network access and the script output above.")
else:
    DATA_READY = all(Path("splits", f"{name}.txt").exists() for name in ("train", "validation", "test"))
print("Dataset prepared:", DATA_READY)
""")

md("7. Dataset Exploration", """Inspect the official development sample and its original trimap. This cell does not alter the training data. The dataset object is created only after preparation succeeds.""")
code("""
if DATA_READY:
    from torchvision.datasets import OxfordIIITPet
    raw_dev = OxfordIIITPet(root=config["dataset"]["root"], split="trainval",
                            target_types="segmentation", download=False)
    raw_image, raw_trimap = raw_dev[0]
    print("Official development:", len(raw_dev), "(test images remain unopened)")
    print("First image:", raw_image.size, "trimap labels:", np.unique(np.asarray(raw_trimap)).tolist())
else:
    print("Dataset exploration skipped until preparation completes.")
""")

md("8. Mask Preprocessing", """The shared pipeline resizes the trimap with nearest-neighbor interpolation, then maps label 1 to pet foreground, label 2 to background, and label 3 to **ignored** pixels. The separate valid mask is passed to both losses and metrics. The small array below illustrates the mapping without replacing the implementation in `src/preprocessing.py`.""")
code("""
toy_trimap = np.array([[1, 2, 3]], dtype=np.uint8)
toy_target = (toy_trimap == 1).astype(np.float32)
toy_valid = (toy_trimap != 3).astype(np.float32)
print("trimap:", toy_trimap.tolist(), "binary:", toy_target.tolist(), "valid:", toy_valid.tolist())
assert toy_target.tolist() == [[1.0, 0.0, 0.0]]
assert toy_valid.tolist() == [[1.0, 1.0, 0.0]]
""")

md("9. Data Splitting", """The training/validation split is drawn from the official development set with seed 42. The official test split is kept separate from PSO fitness and checkpoint selection. Membership files make the split reproducible.""")
code("""
if DATA_READY:
    memberships = {name: set(Path("splits", f"{name}.txt").read_text(encoding="utf-8").splitlines())
                   for name in ("train", "validation", "test")}
    print({name: len(ids) for name, ids in memberships.items()})
    assert not (memberships["train"] & memberships["validation"])
    assert not (memberships["train"] & memberships["test"])
    assert not (memberships["validation"] & memberships["test"])
else:
    print("Split inspection skipped until preparation completes.")
""")

md("10. Data Visualization", """Prepared sample figures show the RGB image, original trimap, binary mask, and valid-pixel mask. They are produced by the dataset preparation script and are displayed here if available.""")
code("""
from IPython.display import Image, display
sample_figures = sorted(Path("figures/dataset_samples").glob("*.png"))
if sample_figures:
    display(Image(filename=str(sample_figures[0])))
else:
    print("No sample figure has been generated yet.")
""")

md("11. U-Net Architecture", """The compact U-Net has 16/32/64/128 encoder channels, a 256-channel bottleneck with configurable dropout, skip connections, and one raw-logit output channel. No pretrained weights are used. This tiny forward pass checks the shape; full training remains in the shared scripts.""")
code("""
from src.unet import CompactUNet
model_args = {k: v for k, v in config["model"].items() if k != "name"}
model = CompactUNet(**model_args)
with torch.no_grad():
    demo_logits = model(torch.zeros(1, 3, config["dataset"]["image_size"],
                                   config["dataset"]["image_size"]))
print("Output shape:", tuple(demo_logits.shape))
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
""")

md("12. Loss Functions", """The shared training criterion is `0.5 × masked BCEWithLogits + 0.5 × masked soft Dice`. Ignored pixels never enter either term. The training code excludes images with no valid pixels and skips optimizer updates for wholly invalid batches.""")
code("""
from src.losses import masked_bce_with_logits, masked_soft_dice_loss, combined_segmentation_loss
demo_logit = torch.tensor([[[[2.0, -2.0, 20.0]]]])
demo_truth = torch.tensor([[[[1.0, 0.0, 0.0]]]])
demo_valid = torch.tensor([[[[1.0, 1.0, 0.0]]]])
print("BCE:", float(masked_bce_with_logits(demo_logit, demo_truth, demo_valid)))
print("Soft Dice loss:", float(masked_soft_dice_loss(demo_logit, demo_truth, demo_valid)))
print("Combined:", float(combined_segmentation_loss(demo_logit, demo_truth, demo_valid)))
""")

md("13. Segmentation Metrics", """At a fixed 0.5 probability threshold, Dice and foreground IoU are computed per image over valid pixels, then averaged across images. An image with no valid pixels is excluded; an empty prediction and empty truth score 1.""")
code("""
from src.metrics import segmentation_metrics
print(segmentation_metrics(demo_logit, demo_truth, demo_valid))
""")

md("14. Baseline Configuration", """The baseline uses AdamW, batch size 16, learning rate 0.001, dropout 0.10, weight decay 0.0001, eight epochs, and best-validation-Dice checkpoint selection. The project configuration is the single source of these defaults.""")
code("""
print(json.dumps({"dataset": config["dataset"], "model": config["model"],
                  "training": config["training"]}, indent=2))
""")

md("15. Baseline Training", """`SMOKE_MODE` exercises a limited integration path; it is **not** a complete baseline result. With `SMOKE_MODE=False`, the script uses the full configuration. Set `RUN_BASELINE=False` to inspect the notebook without training.""")
code("""
RUN_BASELINE = True
SMOKE_MODE = False
if RUN_BASELINE and (DATA_READY or SMOKE_MODE):
    command = [sys.executable, "-m", "scripts.train_baseline"]
    if SMOKE_MODE:
        command.append("--smoke")
    subprocess.run(command, check=True)
elif RUN_BASELINE:
    print("Full baseline training skipped: dataset not prepared. Synthetic --smoke can run without it.")
else:
    print("Baseline training disabled by RUN_BASELINE.")
""")

md("16. Baseline Validation", """Read the measured history generated by the training script. The best checkpoint is selected using validation Dice only. Missing files mean no completed result is available.""")
code("""
import pandas as pd
history_file = Path("results/training_history.csv")
if history_file.exists():
    baseline_history = pd.read_csv(history_file)
    display(baseline_history.tail())
else:
    print("No baseline training history is available.")
""")

md("17. Training Curves", """Show loss and validation-Dice figures only if generated by actual training. A smoke run must remain labeled as such.""")
code("""
curve_files = sorted(Path("figures/training_curves").glob("*.png"))
if curve_files:
    for figure_path in curve_files:
        print(figure_path)
        display(Image(filename=str(figure_path)))
else:
    print("No measured training curves are available.")
""")

md("18. Segmentation Predictions", """Qualitative images help identify background confusion and boundary errors. Display only figures created from a trained checkpoint. Avoid choosing examples using test performance during model development.""")
code("""
prediction_files = sorted(Path("figures/predictions").glob("*.png"))
if prediction_files:
    for figure_path in prediction_files[:3]:
        print(figure_path)
        display(Image(filename=str(figure_path)))
else:
    print("No checkpoint-based prediction figures are available.")
""")

md("19. PSO Implementation", """PSO represents each particle as `[log10(learning_rate), dropout, log10(weight_decay)]`. Its fitness is `1 − mean validation Dice`; every candidate starts from fresh U-Net weights and receives two epochs. The three-particle, two-round pilot makes six evaluations when it completes.""")
code("""
from src import pso
print("PSO implementation:", pso.__file__)
print(json.dumps(config["pso"], indent=2))
""")

md("20. PSO Pilot", """This optional cell can take substantially longer than the baseline because it trains six candidates and then an eight-epoch final model. `--smoke` checks integration only and does not support a full quality claim. The validation split supplies fitness; the test split is untouched.""")
code("""
RUN_PSO_PILOT = False
if RUN_PSO_PILOT and (DATA_READY or SMOKE_MODE):
    command = [sys.executable, "-m", "scripts.run_pso"]
    if SMOKE_MODE:
        command.append("--smoke")
    subprocess.run(command, check=True)
elif RUN_PSO_PILOT:
    print("Full PSO pilot skipped: dataset not prepared. Synthetic --smoke can run without it.")
else:
    print("PSO pilot disabled by RUN_PSO_PILOT.")
""")

md("21. Best Hyperparameters", """Inspect saved metadata and PSO history only when a search has actually run. A two-epoch candidate score is a selection signal, not the eight-epoch final-model score.""")
code("""
metadata_file = Path("results/experiment_metadata.json")
if metadata_file.exists():
    print(json.dumps(json.loads(metadata_file.read_text(encoding="utf-8")), indent=2))
else:
    print("No experiment metadata is available.")
pso_history_file = Path("results/pso_history.csv")
if pso_history_file.exists():
    display(pd.read_csv(pso_history_file).tail())
else:
    print("No PSO search history is available.")
""")

md("22. Experimental Results", """Set `RUN_HELD_OUT_TEST=True` **only after** both final checkpoints exist and their validation selections are fixed. Evaluation opens the official test split for reporting, never for tuning. `generate_results` regenerates visual summaries from measured files. Exported ZIP excludes large checkpoints and raw images.""")
code("""
RUN_HELD_OUT_TEST = False
if RUN_HELD_OUT_TEST and DATA_READY and not SMOKE_MODE:
    subprocess.run([sys.executable, "-m", "scripts.evaluate_models"], check=True)
elif RUN_HELD_OUT_TEST:
    print("Held-out evaluation skipped: full data and final checkpoints are required.")
subprocess.run([sys.executable, "-m", "scripts.generate_results"], check=False)
result_file = Path("results/results.csv")
if result_file.exists():
    display(pd.read_csv(result_file))
else:
    print("No measured comparison table is available.")

from zipfile import ZipFile, ZIP_DEFLATED
export_path = Path("phase1-results.zip")
with ZipFile(export_path, "w", ZIP_DEFLATED) as archive:
    for folder in (Path("results"), Path("figures")):
        if folder.exists():
            for item in folder.rglob("*"):
                if item.is_file() and item.suffix.lower() in {".csv", ".json", ".png", ".mmd"}:
                    archive.write(item)
print("Export ready:", export_path.resolve())
""")

md("23. Conclusions and Limitations", """Write conclusions **after** inspecting completed runs. Compare only final models with matched eight-epoch budgets, include PSO search time, and treat a one-seed pilot as preliminary. The short fitness budget may rank candidates differently from full training; 128-pixel input may lose fine fur details. Check `reports/Phase1_Report.md` and `reports/RUN_STATUS.md` for the current draft and verification status. Review the original course handout and academic-integrity requirements before submission.""")
code("""
print("Notebook complete. Record completed metrics and runtime from actual artifacts only.")
""")

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "phase1.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
output = Path(__file__).with_name("phase1.ipynb")
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(output, "cells:", len(cells))
