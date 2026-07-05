"""
predict.py
----------
Loads the trained model + encoders and exposes a simple, reusable
`predict_match()` function that the Streamlit app calls.

Also includes a plain-English explanation generator so predictions are
never a "black box" to the end user.
"""

from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd

import utils
from train_model import FEATURE_COLUMNS, encode_features


class ModelNotFoundError(Exception):
    """Raised when the model artifacts don't exist yet (user hasn't trained)."""


def load_model_artifacts():
    """Load model, encoders, and metrics from disk. Raises a friendly error if missing."""
    if not (os.path.exists(utils.MODEL_PATH) and os.path.exists(utils.ENCODERS_PATH)):
        raise ModelNotFoundError(
            "No trained model found. Please run `python train_model.py` first, "
            "or use the 'Train Model' button in the sidebar."
        )
    model = joblib.load(utils.MODEL_PATH)
    enc_bundle = joblib.load(utils.ENCODERS_PATH)
    metrics = joblib.load(utils.METRICS_PATH) if os.path.exists(utils.METRICS_PATH) else None
    return model, enc_bundle["encoders"], enc_bundle["feature_columns"], metrics


def _build_single_feature_row(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    league: Optional[str] = None,
    manual_home_form: Optional[float] = None,
    manual_away_form: Optional[float] = None,
) -> pd.DataFrame:
    """Construct the feature row for one hypothetical upcoming match."""
    home_form = utils.compute_team_form(df, home_team, before_date=None, last_n=5)
    away_form = utils.compute_team_form(df, away_team, before_date=None, last_n=5)

    home_win_rate = manual_home_form if manual_home_form is not None else home_form.win_rate
    away_win_rate = manual_away_form if manual_away_form is not None else away_form.win_rate

    if league is None:
        league_rows = df[(df["HomeTeam"] == home_team)]
        league = league_rows["League"].iloc[0] if not league_rows.empty else df["League"].iloc[0]

    row = pd.DataFrame([{
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "League": league,
        "HomeFormWinRate": home_win_rate,
        "AwayFormWinRate": away_win_rate,
        "HomeAvgGoalsScored": home_form.avg_goals_scored,
        "HomeAvgGoalsConceded": home_form.avg_goals_conceded,
        "AwayAvgGoalsScored": away_form.avg_goals_scored,
        "AwayAvgGoalsConceded": away_form.avg_goals_conceded,
        "HomeFormPoints": home_form.points,
        "AwayFormPoints": away_form.points,
    }])
    return row


def predict_match(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    manual_home_form: Optional[float] = None,
    manual_away_form: Optional[float] = None,
) -> dict:
    """
    Predict the outcome of home_team vs away_team.

    Returns a dict with the predicted label, class probabilities, a confidence
    score, and a plain-English explanation.
    """
    if home_team == away_team:
        raise ValueError("Home team and away team must be different.")
    if not utils.team_exists(df, home_team):
        raise ValueError(f"Unknown team: '{home_team}' is not in the dataset.")
    if not utils.team_exists(df, away_team):
        raise ValueError(f"Unknown team: '{away_team}' is not in the dataset.")

    model, encoders, feature_columns, _ = load_model_artifacts()

    row = _build_single_feature_row(
        df, home_team, away_team,
        manual_home_form=manual_home_form,
        manual_away_form=manual_away_form,
    )
    encoded_row, _ = encode_features(row, encoders=encoders)
    X = encoded_row[feature_columns]

    proba = model.predict_proba(X)[0]  # order matches model.classes_
    class_order = model.classes_  # e.g. array([0, 1, 2])
    proba_by_class = {int(c): float(p) for c, p in zip(class_order, proba)}

    away_p = proba_by_class.get(0, 0.0)
    draw_p = proba_by_class.get(1, 0.0)
    home_p = proba_by_class.get(2, 0.0)

    predicted_class = int(class_order[int(np.argmax(proba))])
    predicted_label = utils.RESULT_LABELS[predicted_class]
    confidence = float(np.max(proba))

    explanation = _generate_explanation(
        home_team, away_team, home_p, draw_p, away_p, row.iloc[0]
    )

    return {
        "predicted_label": predicted_label,
        "home_win_prob": home_p,
        "draw_prob": draw_p,
        "away_win_prob": away_p,
        "confidence": confidence,
        "explanation": explanation,
    }


def _generate_explanation(home_team, away_team, home_p, draw_p, away_p, feat_row) -> str:
    """Turn the raw probabilities + features into a human-readable explanation."""
    parts = []

    if home_p == max(home_p, draw_p, away_p):
        parts.append(
            f"{home_team} are favoured to win at home with a "
            f"{home_p * 100:.1f}% probability."
        )
    elif away_p == max(home_p, draw_p, away_p):
        parts.append(
            f"{away_team} are favoured to win away with a "
            f"{away_p * 100:.1f}% probability."
        )
    else:
        parts.append(f"A draw is the most likely outcome ({draw_p * 100:.1f}%).")

    form_gap = feat_row["HomeFormWinRate"] - feat_row["AwayFormWinRate"]
    if abs(form_gap) > 0.15:
        better = home_team if form_gap > 0 else away_team
        parts.append(f"{better} have noticeably better recent form, which weighs heavily on this prediction.")

    if feat_row["HomeAvgGoalsScored"] > feat_row["AwayAvgGoalsConceded"]:
        parts.append(f"{home_team}'s attacking output looks strong against {away_team}'s recent defensive record.")

    parts.append("Home advantage is also factored in, as home teams historically perform better.")

    return " ".join(parts)
