"""Download, validate, split, and preview the Oxford-IIIT Pet dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets.utils import download_and_extract_archive

from src.config import load_config, project_path
from src.dataset import PreparedData, prepare_data
from src.preprocessing import preprocess_pair


# Both checksums are the ones required by torchvision's OxfordIIITPet class.
VERIFIED_MIRROR = (
    (
        "https://huggingface.co/datasets/Shiym/ViT-FineTune/resolve/main/"
        "oxford-iiit-pet/annotations.tar.gz?download=true",
        "annotations.tar.gz",
        "95a8c909bbe2e81eed6a22bccdf3f68f",
    ),
    (
        "https://huggingface.co/datasets/Shiym/ViT-FineTune/resolve/main/"
        "oxford-iiit-pet/images.tar.gz?download=true",
        "images.tar.gz",
        "5c4f3ee8e5d25df40f4fd59a7f44e54c",
    ),
)


def prepare_verified_mirror(root: Path) -> None:
    """Fetch exact upstream archives from a mirror when Oxford is unreachable."""
    base = root / "oxford-iiit-pet"
    base.mkdir(parents=True, exist_ok=True)
    for url, filename, md5 in VERIFIED_MIRROR:
        download_and_extract_archive(
            url, download_root=str(base), extract_root=str(base), filename=filename, md5=md5
        )


def save_dataset_samples(prepared: PreparedData, output: Path, count: int = 4) -> Path:
    """Show the original image/trimap and both processed target masks."""

    if count <= 0:
        raise ValueError("count must be positive")
    count = min(count, len(prepared.train_indices))
    fig, axes = plt.subplots(count, 4, figsize=(12, 3 * count), squeeze=False)
    headings = ("Original image", "Original trimap", "Pet foreground", "Valid pixels")
    sample_positions = np.linspace(0, len(prepared.train_indices) - 1, count, dtype=int)
    for row, sample_position in enumerate(sample_positions):
        source_index = prepared.train_indices[int(sample_position)]
        image, trimap = prepared.train_source[source_index]
        _, foreground, valid = preprocess_pair(image, trimap, prepared.image_size)
        panels = (
            np.asarray(image),
            np.asarray(trimap),
            foreground.squeeze(0).numpy(),
            valid.squeeze(0).numpy(),
        )
        for col, panel in enumerate(panels):
            axis = axes[row, col]
            axis.imshow(panel, cmap=None if col == 0 else "gray", vmin=0 if col else None,
                        vmax=3 if col == 1 else (1 if col > 1 else None))
            axis.set_title(headings[col])
            axis.axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-download", action="store_true", help="Use existing dataset files only")
    parser.add_argument("--mirror", choices=["huggingface"],
                        help="Use a checksum-verified mirror when the Oxford host is unreachable")
    parser.add_argument("--samples", type=int, default=4, help="Number of sample rows to visualize")
    args = parser.parse_args()
    if args.no_download and args.mirror:
        parser.error("--no-download and --mirror cannot be combined")
    config = load_config()
    if args.mirror:
        prepare_verified_mirror(project_path(config["dataset"]["root"]))
    prepared = prepare_data(
        config, download=not args.no_download and not args.mirror, include_test=True
    )
    output = Path(__file__).resolve().parents[1] / "figures/dataset_samples/examples.png"
    save_dataset_samples(prepared, output, count=args.samples)
    print(
        f"Validated Oxford-IIIT Pet: train={len(prepared.train_ids)}, "
        f"validation={len(prepared.validation_ids)}, test={len(prepared.test_ids)}. "
        f"Preview: {output}"
    )


if __name__ == "__main__":
    main()
