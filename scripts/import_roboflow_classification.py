from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a Roboflow classification export.")
    parser.add_argument("--source", required=True, help="Path to the Roboflow export root.")
    parser.add_argument("--output-images", default="datasets/raw/roboflow_classification/images")
    parser.add_argument("--output-labels", default="datasets/labels/roboflow_labels.csv")
    args = parser.parse_args()

    source_root = Path(args.source)
    output_images = Path(args.output_images)
    output_labels = Path(args.output_labels)
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.parent.mkdir(parents=True, exist_ok=True)

    splits = ["train", "valid", "test"]
    rows: list[str] = ["image,label,class_name,split"]

    for split in splits:
        split_dir = source_root / split
        if not split_dir.exists():
            continue

        classes_csv = split_dir / "_classes.csv"
        if classes_csv.exists():
            with classes_csv.open("r", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if not header:
                    continue
                class_names = [name.strip() for name in header[1:]]
                for row in reader:
                    if not row:
                        continue
                    filename = row[0].strip()
                    scores = [int(value.strip() or 0) for value in row[1:]]
                    if not scores:
                        continue
                    label = 0 if "normal" in class_names and scores[class_names.index("normal")] == 1 else 1
                    class_name = "normal" if label == 0 else "abnormal"

                    image_path = split_dir / filename
                    if not image_path.exists():
                        continue
                    safe_name = f"{split}__{filename}"
                    dest_path = output_images / safe_name
                    shutil.copy2(image_path, dest_path)
                    rows.append(f"{safe_name},{label},{class_name},{split}")
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name.strip()
            for image_path in class_dir.glob("*"):
                if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                    continue
                safe_name = f"{split}__{class_name}__{image_path.name}"
                dest_path = output_images / safe_name
                shutil.copy2(image_path, dest_path)

                label = 0 if class_name.lower() == "normal" else 1
                rows.append(f"{safe_name},{label},{class_name},{split}")

    if len(rows) == 1:
        raise SystemExit("No images found. Ensure the export has train/valid/test folders with class subfolders.")

    output_labels.write_text("\n".join(rows), encoding="utf-8")
    print(f"Imported {len(rows) - 1} images into {output_images}")
    print(f"Wrote labels to {output_labels}")


if __name__ == "__main__":
    main()
