from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def compute_features(grayscale: np.ndarray) -> np.ndarray:
    resized = cv2.resize(grayscale, (256, 256), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)

    hist = cv2.calcHist([blurred], [0], None, [16], [0, 256]).flatten().astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()

    edges = cv2.Canny(blurred, 50, 150)
    edge_density = float(edges.mean() / 255.0)
    lap_var = float(cv2.Laplacian(blurred, cv2.CV_64F).var())

    mean = float(blurred.mean())
    std = float(blurred.std())
    p10 = float(np.percentile(blurred, 10))
    p90 = float(np.percentile(blurred, 90))
    contrast = p90 - p10

    return np.concatenate(
        [
            np.array([mean, std, p10, p90, contrast, edge_density, lap_var], dtype=np.float32),
            hist,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract simple image features for RF training.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", default="datasets/processed/roboflow_image_features.csv")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        features = compute_features(image)
        row = {"image": image_path.name}
        for idx, value in enumerate(features):
            row[f"f{idx:02d}"] = float(value)
        rows.append(row)

    if not rows:
        raise SystemExit("No images found to extract features.")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} feature rows to {output_path}")


if __name__ == "__main__":
    main()
