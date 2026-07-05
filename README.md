# ⚽ AI-Powered Football Match Prediction & Analytics Dashboard

A full-stack football analytics platform that uses machine learning to predict
match outcomes and provides rich, interactive analytics — built as a Final
Year Major Project with a production-quality architecture.

---

## 🌟 Overview

This platform combines a **RandomForest classifier** (trained on pre-match
rolling team form) with an **interactive Streamlit dashboard** to deliver:

- Match outcome predictions (Home Win / Draw / Away Win) with probabilities
- Monte Carlo match simulation (1000+ simulated replays per fixture)
- Deep per-team analytics with radar, bar, pie, and line charts
- A rule-based, fully-explainable AI Coach Assistant for tactical planning
- Explainable AI: feature importance, confusion matrix, model metrics
- A league-wide analytics dashboard (top scorers, best defenses, standings)
- Extra tools: Dream XI generator, transfer recommendations, opponent
  weakness detector, momentum tracker, and match difficulty rating

The dataset ships with a realistic **synthetic** multi-league, multi-season
dataset (`data/matches.csv`) so the app works out of the box. Swap in a real
dataset at any time — column names are auto-detected and mapped.

---

## ✨ Features

| Page | What it does |
|---|---|
| 🏠 Home | Dataset overview, key stats, recent matches |
| 🔮 Match Prediction | ML-powered outcome prediction + explanation |
| 📊 Team Analytics | Full team stats, radar chart, trends |
| 🎲 Match Simulator | 1000x Poisson-based Monte Carlo simulation |
| 🧠 AI Coach Assistant | Formation, tactics, pressing, set pieces + reasoning |
| 🔍 Explainable AI | Accuracy/Precision/Recall/F1, confusion matrix, feature importance |
| 📈 Football Dashboard | Top scorers, best defenses, standings, comparisons |
| 🛠️ Extra AI Tools | Dream XI, transfer engine, weakness detector, momentum, difficulty rating |

---

## 🗂️ Folder Structure

```
football_project/
├── app.py                    # Streamlit application (all UI pages)
├── train_model.py            # Trains & saves the ML model
├── predict.py                # Prediction logic + explanations
├── analytics.py              # Team/league analytics + extra AI tools
├── simulation.py             # Monte Carlo match simulator
├── coach.py                  # Rule-based AI coach engine
├── utils.py                  # Data loading, cleaning, feature engineering
├── generate_sample_data.py   # (Optional) regenerates the demo dataset
├── requirements.txt
├── README.md
├── data/
│   └── matches.csv           # Match dataset (auto column-mapping supported)
├── models/                   # Trained model + encoders + metrics (generated)
│   ├── football_model.pkl
│   ├── encoders.pkl
│   └── metrics.pkl
└── assets/                   # Static assets (screenshots, images, etc.)
```

---

## 🛠️ Installation

1. **Clone / extract the project** and move into the folder:
   ```bash
   cd football_project
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 How to Train the Model

The app can train the model for you automatically from the sidebar the first
time you visit **Match Prediction** or **Explainable AI** (a "Train Model
Now" button appears if no model exists yet).

To train manually from the command line instead:

```bash
python train_model.py
```

This will:
1. Load and clean `data/matches.csv`
2. Engineer pre-match rolling-form features (no data leakage)
3. Train a `RandomForestClassifier`
4. Print Accuracy / Precision / Recall / F1 / Confusion Matrix
5. Save `models/football_model.pkl`, `models/encoders.pkl`, `models/metrics.pkl`

### Using your own dataset

Replace `data/matches.csv` with your own file containing columns similar to:

```
Date, HomeTeam, AwayTeam, HomeGoals, AwayGoals, League, Season
```

Column names don't need to match exactly — `utils.py` automatically detects
common variants (e.g. `FTHG`/`FTAG`, `home_score`/`away_score`, etc.). Then
re-run `python train_model.py`.

---

## ▶️ How to Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

---

## 🖼️ Screenshots

> _Add screenshots of each page here after running the app locally:_

- `assets/screenshot_home.png`
- `assets/screenshot_prediction.png`
- `assets/screenshot_analytics.png`
- `assets/screenshot_simulator.png`
- `assets/screenshot_coach.png`
- `assets/screenshot_explainable_ai.png`
- `assets/screenshot_dashboard.png`

---

## 🧩 Tech Stack

- **Frontend:** Streamlit
- **ML:** scikit-learn (RandomForestClassifier)
- **Data:** Pandas, NumPy
- **Visualization:** Plotly, Matplotlib
- **Model Persistence:** Joblib

---

## 🚀 Future Scope

- Integrate real-time fixture and odds data via a sports API
- Add player-level statistics for a fully data-driven Dream XI generator
- Support gradient boosting (XGBoost/LightGBM) as a selectable model
- Add a database backend (PostgreSQL/SQLite) for persisting user predictions
- Deploy to Streamlit Community Cloud / Docker for public access
- Add authentication for personalized prediction history
- Incorporate injury/suspension data as model features

---

## ⚠️ Notes

- The bundled dataset is **synthetically generated** for demonstration; swap
  in a real dataset (e.g. from football-data.co.uk or Kaggle) for genuine
  predictive accuracy.
- The Dream XI / Transfer engine use illustrative placeholder player names
  since the dataset is match-level, not player-level — replace
  `_POSITION_POOL` in `analytics.py` with real player data to make this
  fully data-driven.
- This is an educational project; predictions should not be used for betting
  or financial decisions.
