"""
utils.py
--------
Shared utility functions used across the whole application:

- Robust dataset loading with automatic column-name detection/mapping
- Data cleaning and missing-value handling
- Target variable creation (Home Win / Draw / Away Win)
- Feature engineering (team form, historical stats, encodings)
- Small helper functions reused by multiple pages / modules

Keeping all of this in one module means train_model.py, predict.py,
analytics.py, simulation.py and app.py all stay perfectly consistent
about what a "clean" match record looks like.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DATA_PATH = os.path.join("data", "matches.csv")
MODEL_PATH = os.path.join("models", "football_model.pkl")
ENCODERS_PATH = os.path.join("models", "encoders.pkl")
METRICS_PATH = os.path.join("models", "metrics.pkl")

# Canonical column names the rest of the app expects, mapped from the many
# possible names a real-world dataset might use (e.g. football-data.co.uk
# style "FTHG"/"FTAG", or Kaggle-style "home_goals").
COLUMN_ALIASES: dict[str, list[str]] = {
    "Date": ["date", "match_date", "matchdate", "game_date"],
    "HomeTeam": ["hometeam", "home_team", "home", "team_home", "hteam"],
    "AwayTeam": ["awayteam", "away_team", "away", "team_away", "ateam"],
    "HomeGoals": ["homegoals", "home_goals", "fthg", "home_score", "hg", "goals_home"],
    "AwayGoals": ["awaygoals", "away_goals", "ftag", "away_score", "ag", "goals_away"],
    "League": ["league", "competition", "div", "division"],
    "Season": ["season", "year"],
}

RESULT_LABELS = {0: "Away Win", 1: "Draw", 2: "Home Win"}
RESULT_LABELS_INV = {v: k for k, v in RESULT_LABELS.items()}


class DatasetError(Exception):
    """Raised for any issue with the input dataset that the UI should surface nicely."""


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #

def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-detect and rename columns to the canonical schema, case-insensitively."""
    lower_map = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    rename_map = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in lower_map:
                rename_map[lower_map[alias]] = canonical
                break

    df = df.rename(columns=rename_map)
    return df


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load, validate, and clean the match dataset.

    Raises
    ------
    DatasetError
        If the file is missing, empty, corrupted, or lacks required columns.
    """
    if not os.path.exists(path):
        raise DatasetError(
            f"Dataset not found at '{path}'. Please add a CSV file with columns "
            "similar to: Date, HomeTeam, AwayTeam, HomeGoals, AwayGoals, League, Season."
        )

    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DatasetError(f"The dataset file appears corrupted or unreadable: {exc}") from exc

    if df.empty:
        raise DatasetError("The dataset is empty. Please provide a non-empty CSV file.")

    df = _map_columns(df)

    required = ["HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DatasetError(
            f"Could not find required column(s) {missing} in the dataset "
            f"(even after automatic name-matching). Found columns: {list(df.columns)}"
        )

    # Optional columns - fill with sensible defaults if absent
    if "League" not in df.columns:
        df["League"] = "Unknown League"
    if "Season" not in df.columns:
        df["Season"] = "Unknown Season"
    if "Date" not in df.columns:
        df["Date"] = pd.NaT

    df = clean_dataset(df)
    df = add_target_variable(df)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, bad types, and obviously invalid rows."""
    df = df.copy()

    # Parse date safely (invalid -> NaT, does not crash the app)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Coerce goals to numeric, drop rows where this fails / is negative
    for col in ["HomeGoals", "AwayGoals"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"])
    df = df[(df["HomeGoals"] >= 0) & (df["AwayGoals"] >= 0)]

    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df["League"] = df["League"].fillna("Unknown League").astype(str).str.strip()
    df["Season"] = df["Season"].fillna("Unknown Season").astype(str).str.strip()

    df["HomeGoals"] = df["HomeGoals"].astype(int)
    df["AwayGoals"] = df["AwayGoals"].astype(int)

    # Remove exact duplicate matches
    df = df.drop_duplicates()

    df = df.sort_values("Date", na_position="last").reset_index(drop=True)
    return df


def add_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Create the 3-class target: 0 = Away Win, 1 = Draw, 2 = Home Win."""
    df = df.copy()
    conditions = [
        df["HomeGoals"] > df["AwayGoals"],
        df["HomeGoals"] == df["AwayGoals"],
        df["HomeGoals"] < df["AwayGoals"],
    ]
    choices = [2, 1, 0]
    df["Result"] = np.select(conditions, choices)
    df["ResultLabel"] = df["Result"].map(RESULT_LABELS)
    return df


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

@dataclass
class TeamRollingStats:
    """Rolling form statistics for a single team up to (but excluding) a given match."""

    matches_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_scored: int = 0
    goals_conceded: int = 0

    @property
    def points(self) -> int:
        return self.wins * 3 + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.matches_played if self.matches_played else 0.0

    @property
    def avg_goals_scored(self) -> float:
        return self.goals_scored / self.matches_played if self.matches_played else 0.0

    @property
    def avg_goals_conceded(self) -> float:
        return self.goals_conceded / self.matches_played if self.matches_played else 0.0


def compute_team_form(df: pd.DataFrame, team: str, before_date=None, last_n: int = 5) -> TeamRollingStats:
    """
    Compute a team's rolling form over its last `last_n` matches
    (optionally only matches strictly before `before_date`).
    """
    matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()
    if before_date is not None:
        matches = matches[matches["Date"] < before_date]
    matches = matches.sort_values("Date").tail(last_n)

    stats = TeamRollingStats()
    for _, row in matches.iterrows():
        is_home = row["HomeTeam"] == team
        gf = row["HomeGoals"] if is_home else row["AwayGoals"]
        ga = row["AwayGoals"] if is_home else row["HomeGoals"]

        stats.matches_played += 1
        stats.goals_scored += gf
        stats.goals_conceded += ga
        if gf > ga:
            stats.wins += 1
        elif gf == ga:
            stats.draws += 1
        else:
            stats.losses += 1
    return stats


def build_feature_table(df: pd.DataFrame, form_window: int = 5) -> pd.DataFrame:
    """
    Build the full ML feature table. For every match we compute the pre-match
    rolling form (last `form_window` games) of both teams, so the model never
    "sees the future" (no data leakage).
    """
    df = df.sort_values("Date").reset_index(drop=True)

    feature_rows = []
    for idx, row in df.iterrows():
        home_form = compute_team_form(df, row["HomeTeam"], before_date=row["Date"], last_n=form_window)
        away_form = compute_team_form(df, row["AwayTeam"], before_date=row["Date"], last_n=form_window)

        feature_rows.append({
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "League": row["League"],
            "HomeFormWinRate": home_form.win_rate,
            "AwayFormWinRate": away_form.win_rate,
            "HomeAvgGoalsScored": home_form.avg_goals_scored,
            "HomeAvgGoalsConceded": home_form.avg_goals_conceded,
            "AwayAvgGoalsScored": away_form.avg_goals_scored,
            "AwayAvgGoalsConceded": away_form.avg_goals_conceded,
            "HomeFormPoints": home_form.points,
            "AwayFormPoints": away_form.points,
            "HomeMatchesPlayed": home_form.matches_played,
            "AwayMatchesPlayed": away_form.matches_played,
            "Result": row["Result"],
        })

    return pd.DataFrame(feature_rows)


def get_team_list(df: pd.DataFrame) -> list[str]:
    """Return a sorted, de-duplicated list of every team in the dataset."""
    teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).dropna().unique().tolist()
    return sorted(teams)


def team_exists(df: pd.DataFrame, team: str) -> bool:
    return team in get_team_list(df)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide without ever raising ZeroDivisionError."""
    return numerator / denominator if denominator else default
