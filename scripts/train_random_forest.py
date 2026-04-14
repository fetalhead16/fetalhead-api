from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Random Forest classifier from biometric features.")
    parser.add_argument("--features", default="datasets/processed/biometry_features.csv")
    parser.add_argument("--labels", default="datasets/labels/labels.csv")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    features_path = Path(args.features)
    labels_path = Path(args.labels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    if "image" not in labels_df.columns or "label" not in labels_df.columns:
        raise ValueError("labels.csv must contain columns: image,label")

    merged = features_df.merge(labels_df, on="image", how="inner")
    if merged.empty:
        raise ValueError("No matching images between features and labels.")

    X = merged[["hc", "bpd", "ofd", "ci", "ha"]].values.astype(np.float32)
    y = merged["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=4,
        class_weight="balanced",
        random_state=args.random_state,
    )
    clf.fit(X_train_scaled, y_train)
    preds = clf.predict(X_test_scaled)

    report = classification_report(y_test, preds, digits=3)
    print(report)

    joblib.dump(clf, output_dir / "random_forest.joblib")
    joblib.dump(scaler, output_dir / "feature_scaler.joblib")
    print(f"Saved model artifacts to {output_dir}")


if __name__ == "__main__":
    main()
