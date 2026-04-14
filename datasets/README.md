# Dataset Status

This deployment references the following datasets mentioned by the project team:

- `Kaggle - Fetal ultrasound images`
- `Roboflow - Fetal brain abnormalities`

The live website does not automatically download these datasets.

To move from demo mode to a trained pipeline, add cleaned exports here, for example:

- `datasets/raw/kaggle_fetal_head/`
- `datasets/raw/roboflow_fetal_brain/`

Expected structure (example):

- `datasets/raw/kaggle_fetal_head/images/`
- `datasets/raw/kaggle_fetal_head/masks/`
- `datasets/labels/labels.csv` (image,label)

Next training tasks:

1. Normalize file structure and labels
2. Split train / validation / test sets
3. Train the segmentation model
4. Extract biometric features
5. Train and export `random_forest.joblib`
6. Export `feature_scaler.joblib`

Quick commands once data is available:

```bash
python scripts/prepare_datasets.py
python scripts/extract_biometry_features.py --images-dir datasets/raw/kaggle_fetal_head/images --masks-dir datasets/raw/kaggle_fetal_head/masks
python scripts/train_random_forest.py --features datasets/processed/biometry_features.csv --labels datasets/labels/labels.csv
```
