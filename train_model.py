"""
train_model.py
---------------
Trains the match-outcome prediction model.

Run directly:
    python train_model.py

This will:
1. Load & clean data/matches.csv
2. Engineer pre-match rolling-form features (no data leakage)
3. Encode categorical variables
4. Train a RandomForestClassifier
5. Evaluate on a held-out test set (accuracy, precision, recall, F1, confusion matrix)
6. Persist the model, encoders, and metrics to /models via joblib

The Streamlit app (app.py) loads the artifacts produced here automatically.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import utils

FEATURE_COLUMNS = [
    "HomeTeamEnc",
    "AwayTeamEnc",
    "LeagueEnc",
    "HomeFormWinRate",
    "AwayFormWinRate",
    "HomeAvgGoalsScored",
    "HomeAvgGoalsConceded",
    "AwayAvgGoalsScored",
    "AwayAvgGoalsConceded",
    "HomeFormPoints",
    "AwayFormPoints",
]


def encode_features(feat_df: pd.DataFrame, encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Label-encode team/league names. Reuses fitted encoders at inference time."""
    feat_df = feat_df.copy()
    fit_new = encoders is None
    encoders = encoders or {}

    def _encode_column(col: str, source_col: str):
        if fit_new:
            enc = LabelEncoder()
            enc.fit(pd.concat([feat_df["HomeTeam"], feat_df["AwayTeam"]]).unique() if source_col in ("HomeTeam", "AwayTeam") else feat_df[source_col].unique())
            encoders[col] = enc
        enc = encoders[col]
        # Handle unseen labels gracefully at inference time
        known = set(enc.classes_)
        feat_df[col] = feat_df[source_col].apply(lambda x: x if x in known else enc.classes_[0])
        feat_df[col] = enc.transform(feat_df[col])

    _encode_column("HomeTeamEnc", "HomeTeam")
    _encode_column("AwayTeamEnc", "AwayTeam")
    _encode_column("LeagueEnc", "League")

    return feat_df, encoders


def train_and_evaluate(feat_df: pd.DataFrame) -> dict:
    """Train the RandomForest model and return a dict of everything the app needs."""
    encoded_df, encoders = encode_features(feat_df)

    X = encoded_df[FEATURE_COLUMNS]
    y = encoded_df["Result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1, 2]),
        "feature_importance": dict(zip(FEATURE_COLUMNS, model.feature_importances_)),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_labels": ["Away Win", "Draw", "Home Win"],
    }

    return {
        "model": model,
        "encoders": encoders,
        "metrics": metrics,
        "feature_columns": FEATURE_COLUMNS,
    }


def main():
    print("Loading dataset...")
    df = utils.load_dataset()
    print(f"Loaded {len(df)} matches across {df['League'].nunique()} league(s).")

    print("Engineering features (this may take a moment)...")
    feat_df = utils.build_feature_table(df)

    print("Training RandomForestClassifier...")
    result = train_and_evaluate(feat_df)

    os.makedirs("models", exist_ok=True)
    joblib.dump(result["model"], utils.MODEL_PATH)
    joblib.dump(
        {"encoders": result["encoders"], "feature_columns": result["feature_columns"]},
        utils.ENCODERS_PATH,
    )
    joblib.dump(result["metrics"], utils.METRICS_PATH)

    m = result["metrics"]
    print("\n===== Model Evaluation =====")
    print(f"Accuracy : {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall   : {m['recall']:.4f}")
    print(f"F1 Score : {m['f1']:.4f}")
    print("Confusion Matrix (rows=actual, cols=predicted) [Away, Draw, Home]:")
    print(m["confusion_matrix"])
    print(f"\nSaved model to {utils.MODEL_PATH}")
    print(f"Saved encoders to {utils.ENCODERS_PATH}")
    print(f"Saved metrics to {utils.METRICS_PATH}")


if __name__ == "__main__":
    main()
