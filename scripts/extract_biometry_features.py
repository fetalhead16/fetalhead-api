from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


def find_pairs(images_dir: Path, masks_dir: Optional[Path], mask_suffix: str) -> Iterable[tuple[Path, Path]]:
    if masks_dir and masks_dir.exists():
        for image_path in images_dir.rglob("*"):
            if not image_path.is_file():
                continue
            mask_candidate = masks_dir / image_path.name
            if mask_candidate.exists():
                yield image_path, mask_candidate
                continue
            mask_candidate = masks_dir / f"{image_path.stem}{mask_suffix}{image_path.suffix}"
            if mask_candidate.exists():
                yield image_path, mask_candidate
    else:
        for image_path in images_dir.rglob("*"):
            if not image_path.is_file():
                continue
            mask_candidate = image_path.with_name(f"{image_path.stem}{mask_suffix}{image_path.suffix}")
            if mask_candidate.exists():
                yield image_path, mask_candidate


def ellipse_metrics(contour: np.ndarray, spacing_mm: Optional[float]) -> dict[str, float]:
    ellipse = cv2.fitEllipse(contour)
    (_, _), (axis_1, axis_2), _ = ellipse
    major_axis = max(float(axis_1), float(axis_2))
    minor_axis = min(float(axis_1), float(axis_2))
    semi_major = major_axis / 2.0
    semi_minor = minor_axis / 2.0
    scale = spacing_mm if spacing_mm is not None else 1.0

    ofd = 2.0 * semi_major * scale
    bpd = 2.0 * semi_minor * scale
    hc = np.pi * (3.0 * (semi_major + semi_minor) - np.sqrt((3.0 * semi_major + semi_minor) * (semi_major + 3.0 * semi_minor))) * scale
    ci = (bpd / ofd) * 100.0 if ofd else 0.0
    ha = np.pi * semi_major * semi_minor * (scale**2)
    return {"hc": hc, "bpd": bpd, "ofd": ofd, "ci": ci, "ha": ha}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract biometric features from segmentation masks.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--masks-dir")
    parser.add_argument("--mask-suffix", default="_mask")
    parser.add_argument("--spacing-mm", type=float, default=None)
    parser.add_argument("--output", default="datasets/processed/biometry_features.csv")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path, mask_path in find_pairs(images_dir, masks_dir, args.mask_suffix):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            continue
        metrics = ellipse_metrics(contour, args.spacing_mm)
        rows.append(
            {
                "image": image_path.name,
                "mask": mask_path.name,
                **metrics,
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "mask", "hc", "bpd", "ofd", "ci", "ha"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
