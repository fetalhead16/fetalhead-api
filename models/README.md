# Model Artifacts

Place trained model files in this directory when they are ready:

- `models/random_forest.joblib`
- `models/feature_scaler.joblib`
- `models/image_random_forest.joblib`
- `models/image_feature_scaler.joblib`

The live deployment automatically switches to the trained classifier when these files exist.

Until then, the app uses a geometric fallback and labels itself as `heuristic_demo`.
