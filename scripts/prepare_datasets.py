from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path


def unzip_to(source_zip: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip, "r") as archive:
        archive.extractall(target_dir)


def try_download_kaggle(dataset: str, output_dir: Path) -> None:
    try:
        import kaggle  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Kaggle API is not installed. Run `pip install kaggle`.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_files(dataset, path=str(output_dir), unzip=True)


def try_download_roboflow(workspace: str, project: str, version: str, api_key: str, output_dir: Path) -> None:
    try:
        from roboflow import Roboflow  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Roboflow SDK is not installed. Run `pip install roboflow`.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    project_ref = rf.workspace(workspace).project(project)
    dataset = project_ref.version(int(version)).download("folder", location=str(output_dir))
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset folders and optionally download sources.")
    parser.add_argument("--kaggle-dataset", default="ankit8467/fetal-head-ultrasound-dataset-for-image-segment")
    parser.add_argument("--roboflow-workspace", default="hritwik-trivedi-gkgrv")
    parser.add_argument("--roboflow-project", default="fetal-brain-abnormalities-ultrasound")
    parser.add_argument("--roboflow-version", default="1")
    parser.add_argument("--roboflow-api-key", default=os.getenv("ROBOFLOW_API_KEY", ""))
    parser.add_argument("--kaggle-download", action="store_true")
    parser.add_argument("--roboflow-download", action="store_true")
    parser.add_argument("--raw-dir", default="datasets/raw")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    kaggle_dir = raw_dir / "kaggle_fetal_head"
    roboflow_dir = raw_dir / "roboflow_fetal_brain"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    roboflow_dir.mkdir(parents=True, exist_ok=True)

    if args.kaggle_download:
        try_download_kaggle(args.kaggle_dataset, kaggle_dir)
        print(f"Kaggle dataset downloaded to {kaggle_dir}")
    else:
        print(f"Place Kaggle zip files in {kaggle_dir} and run with --kaggle-download if API is configured.")

    if args.roboflow_download:
        if not args.roboflow_api_key:
            raise RuntimeError("ROBOFLOW_API_KEY is required for Roboflow downloads.")
        try_download_roboflow(args.roboflow_workspace, args.roboflow_project, args.roboflow_version, args.roboflow_api_key, roboflow_dir)
        print(f"Roboflow dataset downloaded to {roboflow_dir}")
    else:
        print(f"Place Roboflow export in {roboflow_dir} or run with --roboflow-download.")

    print("Dataset directories ready.")


if __name__ == "__main__":
    main()
