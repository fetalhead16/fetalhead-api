# Dataset Status

This deployment references the following datasets mentioned by the project team:

- `Kaggle - Fetal ultrasound images`
- `Roboflow - Fetal brain abnormalities`

The live website does not automatically download these datasets.

To move from demo mode to a trained pipeline, add cleaned exports here, for example:

- `datasets/kaggle_fetal_ultrasound_images/`
- `datasets/roboflow_fetal_brain_abnormalities/`

Next training tasks:

1. Normalize file structure and labels
2. Split train / validation / test sets
3. Train the segmentation model
4. Extract biometric features
5. Train and export `random_forest.joblib`
6. Export `feature_scaler.joblib`
