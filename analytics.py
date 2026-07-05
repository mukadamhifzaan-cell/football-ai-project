"""
analytics.py
------------
All non-ML analytics logic:

- Per-team statistics (Team Analytics page)
- League-wide dashboard aggregates (Football Dashboard page)
- "Extra features": Dream XI generator, transfer recommendation engine,
  opponent weakness detector, team form analyzer, momentum tracker,
  and match difficulty rating.

Kept free of any Streamlit / Plotly imports so it can be unit-tested and
reused independently of the UI layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import utils


@dataclass
class TeamStats:
    team: str
    matches_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_scored: int = 0
    goals_conceded: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_scored - self.goals_conceded

    @property
    def win_percentage(self) -> float:
        return utils.safe_divide(self.wins, self.matches_played) * 100

    @property
    def points(self) -> int:
        return self.wins * 3 + self.draws

    @property
    def avg_goals_scored(self) -> float:
        return utils.safe_divide(self.goals_scored, self.matches_played)

    @property
    def avg_goals_conceded(self) -> float:
        return utils.safe_divide(self.goals_conceded, self.matches_played)

    def as_dict(self) -> dict:
        return {
            "Team": self.team,
            "Matches Played": self.matches_played,
            "Wins": self.wins,
            "Draws": self.draws,
            "Losses": self.losses,
            "Goals Scored": self.goals_scored,
            "Goals Conceded": self.goals_conceded,
            "Goal Difference": self.goal_difference,
            "Win %": round(self.win_percentage, 1),
            "Points": self.points,
        }


def get_team_stats(df: pd.DataFrame, team: str) -> TeamStats:
    """Compute full career (all-time, in-dataset) statistics for a team."""
    stats = TeamStats(team=team)
    matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]

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


def get_all_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of stats for every team - powers the league dashboard."""
    teams = utils.get_team_list(df)
    records = [get_team_stats(df, t).as_dict() for t in teams]
    return pd.DataFrame(records).sort_values("Points", ascending=False).reset_index(drop=True)


def get_head_to_head(df: pd.DataFrame, team_a: str, team_b: str) -> pd.DataFrame:
    """Return all historical matches between two teams."""
    mask = (
        ((df["HomeTeam"] == team_a) & (df["AwayTeam"] == team_b))
        | ((df["HomeTeam"] == team_b) & (df["AwayTeam"] == team_a))
    )
    return df[mask].sort_values("Date", ascending=False)


def get_radar_metrics(df: pd.DataFrame, team: str) -> dict:
    """Return normalized (0-100) metrics for a radar chart of team strength."""
    stats = get_team_stats(df, team)
    all_stats = get_all_team_stats(df)

    def _percentile(value, series):
        if series.empty or series.max() == series.min():
            return 50.0
        return float((value - series.min()) / (series.max() - series.min()) * 100)

    return {
        "Attack": _percentile(stats.avg_goals_scored, all_stats["Goals Scored"] / all_stats["Matches Played"].replace(0, 1)),
        "Defense": 100 - _percentile(stats.avg_goals_conceded, all_stats["Goals Conceded"] / all_stats["Matches Played"].replace(0, 1)),
        "Consistency": _percentile(stats.win_percentage, all_stats["Win %"]),
        "Form": _percentile(stats.points, all_stats["Points"]),
        "Discipline": 100 - _percentile(stats.losses, all_stats["Losses"]),
    }


# --------------------------------------------------------------------------- #
# League dashboard aggregates
# --------------------------------------------------------------------------- #

def get_top_scoring_teams(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    all_stats = get_all_team_stats(df)
    return all_stats.sort_values("Goals Scored", ascending=False).head(n)


def get_best_defensive_teams(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    all_stats = get_all_team_stats(df)
    all_stats = all_stats[all_stats["Matches Played"] > 0].copy()
    all_stats["Goals Conceded Per Match"] = all_stats["Goals Conceded"] / all_stats["Matches Played"]
    return all_stats.sort_values("Goals Conceded Per Match", ascending=True).head(n)


def get_league_standings(df: pd.DataFrame, league: str | None = None) -> pd.DataFrame:
    data = df if league is None else df[df["League"] == league]
    teams = utils.get_team_list(data)
    records = [get_team_stats(data, t).as_dict() for t in teams]
    standings = pd.DataFrame(records)
    if standings.empty:
        return standings
    standings = standings.sort_values(
        ["Points", "Goal Difference", "Goals Scored"], ascending=False
    ).reset_index(drop=True)
    standings.insert(0, "Position", range(1, len(standings) + 1))
    return standings


def get_goals_per_match_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Average total goals per match, grouped by season - shows scoring trends over time."""
    data = df.copy()
    data["TotalGoals"] = data["HomeGoals"] + data["AwayGoals"]
    trend = data.groupby("Season")["TotalGoals"].mean().reset_index()
    trend.columns = ["Season", "AvgGoalsPerMatch"]
    return trend


# --------------------------------------------------------------------------- #
# Extra features
# --------------------------------------------------------------------------- #

def momentum_tracker(df: pd.DataFrame, team: str, window: int = 5) -> dict:
    """
    Momentum score from -100 (terrible run) to +100 (unstoppable), based on
    weighted recent results (more recent matches count more).
    """
    matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].sort_values("Date").tail(window)
    if matches.empty:
        return {"score": 0.0, "trend": "No data", "matches_considered": 0}

    weights = np.linspace(0.5, 1.5, len(matches))  # later matches weighted higher
    points = []
    for _, row in matches.iterrows():
        is_home = row["HomeTeam"] == team
        gf = row["HomeGoals"] if is_home else row["AwayGoals"]
        ga = row["AwayGoals"] if is_home else row["HomeGoals"]
        if gf > ga:
            points.append(3)
        elif gf == ga:
            points.append(1)
        else:
            points.append(0)

    weighted_score = np.average(points, weights=weights)
    momentum_score = float((weighted_score / 3) * 100 * 2 - 100)  # scale to -100..100
    momentum_score = max(-100, min(100, momentum_score))

    if momentum_score > 40:
        trend = "🔥 Excellent momentum"
    elif momentum_score > 0:
        trend = "📈 Positive momentum"
    elif momentum_score > -40:
        trend = "📉 Negative momentum"
    else:
        trend = "❄️ Poor momentum"

    return {"score": round(momentum_score, 1), "trend": trend, "matches_considered": len(matches)}


def match_difficulty_rating(df: pd.DataFrame, team: str, opponent: str) -> dict:
    """Rate how difficult a fixture is for `team` against `opponent`, 1 (easy) - 10 (very hard)."""
    team_stats = get_team_stats(df, team)
    opp_stats = get_team_stats(df, opponent)

    opp_strength = (opp_stats.win_percentage / 100) * 5 + (opp_stats.avg_goals_scored) * 1.5 - (opp_stats.avg_goals_conceded) * 0.5
    own_strength = (team_stats.win_percentage / 100) * 5 + (team_stats.avg_goals_scored) * 1.5 - (team_stats.avg_goals_conceded) * 0.5

    raw = 5 + (opp_strength - own_strength)
    rating = max(1.0, min(10.0, round(raw, 1)))

    if rating >= 7.5:
        summary = "Very difficult fixture - opponent is significantly stronger."
    elif rating >= 5.5:
        summary = "Challenging fixture - a competitive, evenly-poised match."
    elif rating >= 3.5:
        summary = "Manageable fixture - your team should have an edge."
    else:
        summary = "Favourable fixture - opponent looks comparatively weak."

    return {"rating": rating, "summary": summary}


def opponent_weakness_detector(df: pd.DataFrame, opponent: str) -> dict:
    """Identify the opponent's biggest statistical weaknesses to exploit."""
    stats = get_team_stats(df, opponent)
    weaknesses = []

    if stats.avg_goals_conceded > 1.3:
        weaknesses.append("Leaky defense - concedes frequently, exploit with direct attacking play.")
    if stats.win_percentage < 35:
        weaknesses.append("Poor overall win rate - team appears low on confidence.")

    matches = df[(df["HomeTeam"] == opponent) | (df["AwayTeam"] == opponent)]
    away_matches = matches[matches["AwayTeam"] == opponent]
    if not away_matches.empty:
        away_losses = ((away_matches["HomeGoals"] > away_matches["AwayGoals"])).mean()
        if away_losses > 0.45:
            weaknesses.append("Struggles significantly when playing away from home.")

    if stats.avg_goals_scored < 1.0:
        weaknesses.append("Blunt attack - creates limited goal-scoring threat.")

    if not weaknesses:
        weaknesses.append("No major statistical weakness detected - a well-balanced side.")

    return {"team": opponent, "weaknesses": weaknesses, "stats": stats.as_dict()}


def team_form_analyzer(df: pd.DataFrame, team: str, window: int = 5) -> dict:
    """Detailed recent-form breakdown (last N matches) with a plain-English summary."""
    form = utils.compute_team_form(df, team, before_date=None, last_n=window)
    momentum = momentum_tracker(df, team, window=window)

    if form.win_rate >= 0.6:
        summary = f"{team} are in excellent form, winning {form.wins} of their last {form.matches_played} matches."
    elif form.win_rate >= 0.35:
        summary = f"{team} are showing mixed form over their last {form.matches_played} matches."
    else:
        summary = f"{team} are struggling for form, with only {form.wins} win(s) in their last {form.matches_played} matches."

    return {
        "matches_played": form.matches_played,
        "wins": form.wins,
        "draws": form.draws,
        "losses": form.losses,
        "avg_goals_scored": round(form.avg_goals_scored, 2),
        "avg_goals_conceded": round(form.avg_goals_conceded, 2),
        "momentum": momentum,
        "summary": summary,
    }


# A small illustrative player pool used purely to power the Dream XI / Transfer
# features for demo purposes, since the dataset itself is match-level only.
_POSITION_POOL = {
    "GK": ["Rossi", "Hendricks", "Alves"],
    "DEF": ["Kova\u010di\u0107", "Silva", "Okafor", "Bennett"],
    "MID": ["Torres", "Novak", "Ade\u0301bayo\u0301", "Lindgren"],
    "FWD": ["Ferreira", "Dubois", "Nakamura"],
}


def dream_xi_generator(df: pd.DataFrame, team: str) -> dict:
    """
    Generate an illustrative Dream XI (4-3-3) for the team, driven by the
    team's attacking/defensive statistical profile.

    NOTE: since the dataset is match-level (no player data), player names here
    are illustrative placeholders. Swap `_POSITION_POOL` for a real player
    dataset to make this fully data-driven.
    """
    stats = get_team_stats(df, team)
    formation = "4-3-3" if stats.avg_goals_scored >= stats.avg_goals_conceded else "4-5-1"

    xi = {
        "Goalkeeper": _POSITION_POOL["GK"][0],
        "Defenders": _POSITION_POOL["DEF"],
        "Midfielders": _POSITION_POOL["MID"] if formation == "4-3-3" else _POSITION_POOL["MID"] + [_POSITION_POOL["MID"][0]],
        "Forwards": _POSITION_POOL["FWD"] if formation == "4-3-3" else _POSITION_POOL["FWD"][:1],
    }
    return {"formation": formation, "lineup": xi, "based_on": stats.as_dict()}


def transfer_recommendation_engine(df: pd.DataFrame, team: str) -> dict:
    """Suggest what type of transfer targets a team statistically needs most."""
    stats = get_team_stats(df, team)
    recommendations = []

    if stats.avg_goals_conceded > 1.2:
        recommendations.append("Priority: a commanding centre-back or defensive midfielder to tighten the defense.")
    if stats.avg_goals_scored < 1.1:
        recommendations.append("Priority: a clinical striker or creative winger to boost goal output.")
    if stats.draws / max(stats.matches_played, 1) > 0.3:
        recommendations.append("Consider: an impact substitute forward to convert draws into wins late in matches.")
    if not recommendations:
        recommendations.append("Squad looks statistically balanced - target depth signings for squad rotation.")

    return {"team": team, "recommendations": recommendations, "stats": stats.as_dict()}
