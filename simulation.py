"""
simulation.py
--------------
Monte Carlo match simulator. Simulates a fixture thousands of times using
Poisson-distributed goal counts, calibrated from each team's historical
scoring and conceding rates (a standard, well-established approach to
football match simulation).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

import analytics
import utils


def _expected_goals(df: pd.DataFrame, home_team: str, away_team: str) -> tuple[float, float]:
    """
    Estimate expected goals (xG-style) for both teams using the classic
    attack-strength x defense-weakness approach, relative to league averages.
    """
    league_avg_home_goals = df["HomeGoals"].mean() or 1.3
    league_avg_away_goals = df["AwayGoals"].mean() or 1.1

    home_stats = analytics.get_team_stats(df, home_team)
    away_stats = analytics.get_team_stats(df, away_team)

    home_attack_strength = utils.safe_divide(home_stats.avg_goals_scored, league_avg_home_goals, default=1.0)
    home_defense_weakness = utils.safe_divide(home_stats.avg_goals_conceded, league_avg_away_goals, default=1.0)
    away_attack_strength = utils.safe_divide(away_stats.avg_goals_scored, league_avg_away_goals, default=1.0)
    away_defense_weakness = utils.safe_divide(away_stats.avg_goals_conceded, league_avg_home_goals, default=1.0)

    home_xg = home_attack_strength * away_defense_weakness * league_avg_home_goals
    away_xg = away_attack_strength * home_defense_weakness * league_avg_away_goals

    # Guard against degenerate (0 or NaN) expected goals for teams with no history
    home_xg = float(np.clip(home_xg if np.isfinite(home_xg) else 1.3, 0.2, 5.0))
    away_xg = float(np.clip(away_xg if np.isfinite(away_xg) else 1.1, 0.2, 5.0))

    return home_xg, away_xg


def simulate_match(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    n_simulations: int = 1000,
    seed: int | None = 42,
) -> dict:
    """
    Simulate a match `n_simulations` times using Poisson-distributed goals.

    Returns win/draw/loss percentages, the most likely scoreline, the full
    scoreline probability distribution, and raw goal-count histograms
    (for charting in the UI).
    """
    if home_team == away_team:
        raise ValueError("Home team and away team must be different.")
    if not utils.team_exists(df, home_team) or not utils.team_exists(df, away_team):
        raise ValueError("One or both teams are not present in the dataset.")

    rng = np.random.default_rng(seed)
    home_xg, away_xg = _expected_goals(df, home_team, away_team)

    home_goals_sim = rng.poisson(home_xg, size=n_simulations)
    away_goals_sim = rng.poisson(away_xg, size=n_simulations)

    home_wins = int(np.sum(home_goals_sim > away_goals_sim))
    draws = int(np.sum(home_goals_sim == away_goals_sim))
    away_wins = int(np.sum(home_goals_sim < away_goals_sim))

    scorelines = Counter(zip(home_goals_sim.tolist(), away_goals_sim.tolist()))
    most_likely = scorelines.most_common(1)[0]
    top_scorelines = [
        {"scoreline": f"{h}-{a}", "probability": round(count / n_simulations * 100, 2)}
        for (h, a), count in scorelines.most_common(8)
    ]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_xg": round(home_xg, 2),
        "away_xg": round(away_xg, 2),
        "n_simulations": n_simulations,
        "home_win_pct": round(home_wins / n_simulations * 100, 1),
        "draw_pct": round(draws / n_simulations * 100, 1),
        "away_win_pct": round(away_wins / n_simulations * 100, 1),
        "most_likely_scoreline": f"{most_likely[0][0]}-{most_likely[0][1]}",
        "most_likely_scoreline_pct": round(most_likely[1] / n_simulations * 100, 2),
        "top_scorelines": top_scorelines,
        "home_goals_sim": home_goals_sim,
        "away_goals_sim": away_goals_sim,
    }
