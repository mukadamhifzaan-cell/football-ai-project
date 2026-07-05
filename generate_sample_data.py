"""
generate_sample_data.py
------------------------
Utility script used ONCE to create a realistic synthetic football results
dataset (data/matches.csv) for demo purposes.

Users can DELETE this file and drop in their own real dataset (e.g. from
Kaggle / football-data.co.uk) with similar columns:
Date, HomeTeam, AwayTeam, HomeGoals, AwayGoals, League, Season

The rest of the pipeline (utils.py) auto-detects and maps column names,
so this script is not a hard dependency of the app.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

LEAGUES = {
    "Premier League": [
        "Manchester City", "Arsenal", "Liverpool", "Chelsea", "Manchester United",
        "Tottenham Hotspur", "Newcastle United", "Aston Villa", "Brighton",
        "West Ham United", "Everton", "Wolves", "Fulham", "Crystal Palace",
        "Brentford", "Nottingham Forest", "Bournemouth", "Leicester City",
    ],
    "La Liga": [
        "Real Madrid", "Barcelona", "Atletico Madrid", "Real Sociedad",
        "Villarreal", "Real Betis", "Sevilla", "Athletic Bilbao",
        "Valencia", "Girona", "Celta Vigo", "Osasuna",
    ],
    "Serie A": [
        "Inter Milan", "AC Milan", "Juventus", "Napoli", "AS Roma",
        "Lazio", "Atalanta", "Fiorentina", "Bologna", "Torino",
    ],
}

# Assign each team a hidden "strength" rating (drives realistic outcomes)
TEAM_STRENGTH = {}
for league, teams in LEAGUES.items():
    base = {"Premier League": 80, "La Liga": 78, "Serie A": 76}[league]
    for t in teams:
        TEAM_STRENGTH[t] = base + rng.normal(0, 10)

SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025"]


def simulate_goals(attack_strength: float, defense_strength: float) -> int:
    """Simulate goals scored using a Poisson process driven by team strength."""
    lam = max(0.3, (attack_strength - defense_strength) / 25 + 1.35)
    return int(rng.poisson(lam))


def build_dataset() -> pd.DataFrame:
    rows = []
    for league, teams in LEAGUES.items():
        for season in SEASONS:
            season_start_year = int(season.split("-")[0])
            # Each team plays every other team home & away (round robin)
            for i, home in enumerate(teams):
                for j, away in enumerate(teams):
                    if home == away:
                        continue
                    # Not every fixture is guaranteed - keep it realistic-ish
                    if rng.random() < 0.55:
                        continue
                    match_day = rng.integers(1, 340)
                    date = pd.Timestamp(f"{season_start_year}-08-01") + pd.Timedelta(days=int(match_day))

                    home_strength = TEAM_STRENGTH[home] + 4  # home advantage
                    away_strength = TEAM_STRENGTH[away]

                    home_goals = simulate_goals(home_strength, away_strength)
                    away_goals = simulate_goals(away_strength, home_strength)

                    rows.append({
                        "Date": date.strftime("%Y-%m-%d"),
                        "HomeTeam": home,
                        "AwayTeam": away,
                        "HomeGoals": home_goals,
                        "AwayGoals": away_goals,
                        "League": league,
                        "Season": season,
                    })
    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("data/matches.csv", index=False)
    print(f"Generated {len(df)} matches -> data/matches.csv")
