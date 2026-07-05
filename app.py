"""
app.py
------
Main Streamlit entry point for the AI-Powered Football Match Prediction
and Analytics Dashboard.

Run with:
    streamlit run app.py

Pages (selectable from the sidebar):
    1. Home
    2. Match Prediction
    3. Team Analytics
    4. Match Simulator
    5. AI Coach Assistant
    6. Explainable AI
    7. Football Dashboard
    8. Extra AI Tools (Dream XI, transfers, weakness detector, momentum...)
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics
import coach
import predict
import simulation
import utils

# --------------------------------------------------------------------------- #
# Page config & global styling
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="AI Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

PITCH_GREEN = "#0B3D2E"
PITCH_GREEN_LIGHT = "#155843"
GOLD = "#D4AF37"
CHALK = "#F5F3EC"
CARD_BG = "#0F4A38"

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {PITCH_GREEN};
    }}
    section[data-testid="stSidebar"] {{
        background-color: #082B20;
    }}
    h1, h2, h3, h4 {{
        color: {CHALK} !important;
        font-family: 'Trebuchet MS', sans-serif;
    }}
    p, span, label, div {{
        color: {CHALK};
    }}
    .metric-card {{
        background-color: {CARD_BG};
        border: 1px solid {GOLD};
        border-radius: 10px;
        padding: 18px;
        text-align: center;
    }}
    .stButton>button {{
        background-color: {GOLD};
        color: #1a1a1a;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1.5em;
    }}
    .stButton>button:hover {{
        background-color: #e8c65a;
        color: #000000;
    }}
    div[data-testid="stMetricValue"] {{
        color: {GOLD};
    }}
    .reasoning-box {{
        background-color: {CARD_BG};
        border-left: 4px solid {GOLD};
        padding: 10px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = go.layout.Template()
PLOTLY_TEMPLATE.layout.paper_bgcolor = PITCH_GREEN_LIGHT
PLOTLY_TEMPLATE.layout.plot_bgcolor = PITCH_GREEN_LIGHT
PLOTLY_TEMPLATE.layout.font = dict(color=CHALK)
COLOR_SEQ = [GOLD, "#5DADE2", "#EC7063", "#58D68D", "#AF7AC5", "#F5B041"]


# --------------------------------------------------------------------------- #
# Cached data / model loading
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def cached_load_dataset():
    return utils.load_dataset()


@st.cache_resource(show_spinner=False)
def cached_load_model():
    return predict.load_model_artifacts()


def get_dataset_or_stop() -> pd.DataFrame | None:
    """Load the dataset, showing a friendly error and stopping the page if it fails."""
    try:
        return cached_load_dataset()
    except utils.DatasetError as e:
        st.error(f"⚠️ Dataset problem: {e}")
        st.info("Add a valid CSV to `data/matches.csv` and reload the app.")
        st.stop()


def ensure_model_or_offer_training():
    """Try loading the model; if missing, offer a one-click training button."""
    try:
        return cached_load_model()
    except predict.ModelNotFoundError:
        st.warning("⚠️ No trained model found yet.")
        if st.button("🚀 Train Model Now"):
            with st.spinner("Training RandomForest model... this can take a minute."):
                result = subprocess.run(
                    [sys.executable, "train_model.py"], capture_output=True, text=True
                )
            if result.returncode == 0:
                st.success("Model trained successfully! Reloading...")
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error(f"Training failed:\n```\n{result.stderr[-2000:]}\n```")
        st.stop()


# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #

st.sidebar.markdown("## ⚽ Football AI")
st.sidebar.caption("AI-Powered Match Prediction & Analytics")

PAGES = [
    "🏠 Home",
    "🔮 Match Prediction",
    "📊 Team Analytics",
    "🎲 Match Simulator",
    "🧠 AI Coach Assistant",
    "🔍 Explainable AI",
    "📈 Football Dashboard",
    "🛠️ Extra AI Tools",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit, scikit-learn & Plotly")


# --------------------------------------------------------------------------- #
# PAGE 1 — HOME
# --------------------------------------------------------------------------- #

def page_home():
    st.title("⚽ AI-Powered Football Match Prediction & Analytics Dashboard")
    st.markdown(
        "A full-stack football analytics platform combining **machine learning "
        "predictions**, **Monte Carlo simulation**, **explainable AI**, and "
        "**tactical coaching insights** in one dashboard."
    )

    df = get_dataset_or_stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Matches", f"{len(df):,}")
    with c2:
        st.metric("Total Teams", f"{len(utils.get_team_list(df))}")
    with c3:
        st.metric("Leagues Covered", f"{df['League'].nunique()}")
    with c4:
        st.metric("Seasons", f"{df['Season'].nunique()}")

    st.markdown("### 📋 Dataset Summary")
    st.dataframe(df.tail(10), use_container_width=True)

    st.markdown("### 🧭 Explore the Platform")
    cols = st.columns(4)
    features = [
        ("🔮 Match Prediction", "ML-powered outcome prediction with probabilities."),
        ("📊 Team Analytics", "Deep-dive stats and radar comparisons per team."),
        ("🎲 Match Simulator", "1000x Monte Carlo simulation of any fixture."),
        ("🧠 AI Coach", "Tactical formation & strategy recommendations."),
    ]
    for col, (title, desc) in zip(cols, features):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.markdown("### 🕓 Recent Matches")
    recent = df.sort_values("Date", ascending=False).head(5)[
        ["Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals", "League"]
    ]
    st.dataframe(recent, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# PAGE 2 — MATCH PREDICTION
# --------------------------------------------------------------------------- #

def page_prediction():
    st.title("🔮 Match Prediction")
    df = get_dataset_or_stop()
    ensure_model_or_offer_training()

    teams = utils.get_team_list(df)
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("🏠 Home Team", teams, index=0)
    with col2:
        away_options = [t for t in teams if t != home_team]
        away_team = st.selectbox("✈️ Away Team", away_options, index=0)

    with st.expander("⚙️ Advanced: manually override recent form (optional)"):
        override = st.checkbox("Override auto-calculated form")
        manual_home_form = st.slider("Home team form (win rate)", 0.0, 1.0, 0.5, 0.05) if override else None
        manual_away_form = st.slider("Away team form (win rate)", 0.0, 1.0, 0.5, 0.05) if override else None

    if st.button("🔮 Predict Match Result", type="primary"):
        try:
            result = predict.predict_match(
                df, home_team, away_team,
                manual_home_form=manual_home_form,
                manual_away_form=manual_away_form,
            )
        except ValueError as e:
            st.error(f"⚠️ {e}")
            return

        st.markdown("### 🏆 Prediction Result")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"🏠 {home_team} Win", f"{result['home_win_prob']*100:.1f}%")
        c2.metric("🤝 Draw", f"{result['draw_prob']*100:.1f}%")
        c3.metric(f"✈️ {away_team} Win", f"{result['away_win_prob']*100:.1f}%")

        st.markdown(f"#### Predicted Outcome: **{result['predicted_label']}**")
        st.progress(result["confidence"])
        st.caption(f"Model confidence: {result['confidence']*100:.1f}%")

        fig = go.Figure(data=[go.Bar(
            x=[f"{home_team} Win", "Draw", f"{away_team} Win"],
            y=[result["home_win_prob"], result["draw_prob"], result["away_win_prob"]],
            marker_color=COLOR_SEQ[:3],
            text=[f"{v*100:.1f}%" for v in (result["home_win_prob"], result["draw_prob"], result["away_win_prob"])],
            textposition="outside",
        )])
        fig.update_layout(template=PLOTLY_TEMPLATE, title="Outcome Probability Distribution", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 💡 Explanation")
        st.info(result["explanation"])


# --------------------------------------------------------------------------- #
# PAGE 3 — TEAM ANALYTICS
# --------------------------------------------------------------------------- #

def page_team_analytics():
    st.title("📊 Team Analytics")
    df = get_dataset_or_stop()
    teams = utils.get_team_list(df)
    team = st.selectbox("Select a team", teams)

    stats = analytics.get_team_stats(df, team)
    d = stats.as_dict()

    st.markdown(f"### {team} — Career Overview")
    cols = st.columns(6)
    labels = ["Matches Played", "Wins", "Draws", "Losses", "Goals Scored", "Goals Conceded"]
    for col, label in zip(cols, labels):
        col.metric(label, d[label])

    c1, c2 = st.columns(2)
    c1.metric("Goal Difference", d["Goal Difference"])
    c2.metric("Win %", f"{d['Win %']}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Results Breakdown")
        fig_pie = px.pie(
            names=["Wins", "Draws", "Losses"],
            values=[d["Wins"], d["Draws"], d["Losses"]],
            color_discrete_sequence=COLOR_SEQ,
        )
        fig_pie.update_layout(template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("#### Goals: Scored vs Conceded")
        fig_bar = go.Figure(data=[
            go.Bar(name="Scored", x=[team], y=[d["Goals Scored"]], marker_color=GOLD),
            go.Bar(name="Conceded", x=[team], y=[d["Goals Conceded"]], marker_color="#EC7063"),
        ])
        fig_bar.update_layout(template=PLOTLY_TEMPLATE, barmode="group")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### 📈 Points Trend Over Matches (chronological)")
    matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].sort_values("Date")
    running_points = []
    cum = 0
    for _, row in matches.iterrows():
        is_home = row["HomeTeam"] == team
        gf = row["HomeGoals"] if is_home else row["AwayGoals"]
        ga = row["AwayGoals"] if is_home else row["HomeGoals"]
        cum += 3 if gf > ga else (1 if gf == ga else 0)
        running_points.append(cum)
    fig_line = px.line(x=range(1, len(running_points) + 1), y=running_points,
                        labels={"x": "Match #", "y": "Cumulative Points"})
    fig_line.update_traces(line_color=GOLD)
    fig_line.update_layout(template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("#### 🕸️ Radar: Team Strength Profile")
    radar = analytics.get_radar_metrics(df, team)
    categories = list(radar.keys())
    values = list(radar.values())
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=categories + [categories[0]],
        fill="toself", line_color=GOLD, name=team,
    ))
    fig_radar.update_layout(template=PLOTLY_TEMPLATE, polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
    st.plotly_chart(fig_radar, use_container_width=True)


# --------------------------------------------------------------------------- #
# PAGE 4 — MATCH SIMULATOR
# --------------------------------------------------------------------------- #

def page_simulator():
    st.title("🎲 Match Simulator")
    st.caption("Simulates the fixture 1000 times using Poisson-distributed goals calibrated on team history.")
    df = get_dataset_or_stop()
    teams = utils.get_team_list(df)

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        home_team = st.selectbox("🏠 Home Team", teams, key="sim_home")
    with col2:
        away_team = st.selectbox("✈️ Away Team", [t for t in teams if t != home_team], key="sim_away")
    with col3:
        n_sims = st.number_input("Simulations", min_value=100, max_value=10000, value=1000, step=100)

    if st.button("🎲 Run Simulation", type="primary"):
        try:
            res = simulation.simulate_match(df, home_team, away_team, n_simulations=int(n_sims))
        except ValueError as e:
            st.error(f"⚠️ {e}")
            return

        st.markdown("### Simulation Results")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"🏠 {home_team} Win", f"{res['home_win_pct']}%")
        c2.metric("🤝 Draw", f"{res['draw_pct']}%")
        c3.metric(f"✈️ {away_team} Win", f"{res['away_win_pct']}%")

        st.markdown(
            f"**Most likely scoreline:** {res['most_likely_scoreline']} "
            f"({res['most_likely_scoreline_pct']}% of simulations) — "
            f"expected goals: {home_team} {res['home_xg']} vs {away_team} {res['away_xg']}"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Top Scoreline Probabilities")
            score_df = pd.DataFrame(res["top_scorelines"])
            fig = px.bar(score_df, x="scoreline", y="probability", color_discrete_sequence=[GOLD])
            fig.update_layout(template=PLOTLY_TEMPLATE, yaxis_title="Probability (%)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Goal Distribution Histogram")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=res["home_goals_sim"], name=home_team, marker_color=GOLD, opacity=0.75))
            fig_hist.add_trace(go.Histogram(x=res["away_goals_sim"], name=away_team, marker_color="#5DADE2", opacity=0.75))
            fig_hist.update_layout(template=PLOTLY_TEMPLATE, barmode="overlay", xaxis_title="Goals", yaxis_title="Frequency")
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### Outcome Probability Distribution")
        fig_pie = px.pie(
            names=[f"{home_team} Win", "Draw", f"{away_team} Win"],
            values=[res["home_win_pct"], res["draw_pct"], res["away_win_pct"]],
            color_discrete_sequence=COLOR_SEQ,
        )
        fig_pie.update_layout(template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_pie, use_container_width=True)


# --------------------------------------------------------------------------- #
# PAGE 5 — AI COACH ASSISTANT
# --------------------------------------------------------------------------- #

def page_coach():
    st.title("🧠 AI Coach Assistant")
    st.caption("Rule-based tactical recommendations, fully explained.")

    col1, col2, col3 = st.columns(3)
    with col1:
        team_form = st.slider("Your team's form (win rate)", 0.0, 1.0, 0.5, 0.05)
    with col2:
        opponent_form = st.slider("Opponent's form (win rate)", 0.0, 1.0, 0.5, 0.05)
    with col3:
        is_home = st.radio("Venue", ["Home", "Away"]) == "Home"

    if st.button("🧠 Generate Tactical Plan", type="primary"):
        rec = coach.generate_recommendation(team_form, opponent_form, is_home)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📐 Formation")
            st.success(rec.formation)
            st.markdown("#### 🎯 Tactical Style")
            st.info(rec.tactical_style)
            st.markdown("#### 🔥 Pressing Strategy")
            st.info(rec.pressing_strategy)
        with c2:
            st.markdown("#### 🛡️ Defensive Advice")
            st.warning(rec.defensive_advice)
            st.markdown("#### ⚔️ Attacking Advice")
            st.warning(rec.attacking_advice)
            st.markdown("#### 🎯 Set-Piece Suggestion")
            st.info(rec.set_piece_suggestion)

        st.markdown("#### 📋 Overall Match Strategy")
        st.success(rec.match_strategy)

        st.markdown("#### 💭 Why these recommendations?")
        for reason in rec.reasoning:
            st.markdown(f"<div class='reasoning-box'>✅ {reason}</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# PAGE 6 — EXPLAINABLE AI
# --------------------------------------------------------------------------- #

def page_explainable_ai():
    st.title("🔍 Explainable AI")
    st.caption("Understand exactly how the prediction model makes its decisions.")
    get_dataset_or_stop()
    model, encoders, feature_columns, metrics = ensure_model_or_offer_training()

    if metrics is None:
        st.warning("No metrics found. Please retrain the model.")
        return

    st.markdown("### 📊 Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    c2.metric("Precision", f"{metrics['precision']*100:.1f}%")
    c3.metric("Recall", f"{metrics['recall']*100:.1f}%")
    c4.metric("F1 Score", f"{metrics['f1']*100:.1f}%")

    st.markdown("### 🧮 Confusion Matrix")
    cm = metrics["confusion_matrix"]
    labels = metrics["class_labels"]
    fig_cm = px.imshow(
        cm, text_auto=True, x=labels, y=labels,
        color_continuous_scale="Greens",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig_cm.update_layout(template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("### 🌟 Feature Importance")
    importance = metrics["feature_importance"]
    imp_df = pd.DataFrame(sorted(importance.items(), key=lambda x: x[1], reverse=True), columns=["Feature", "Importance"])
    fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[GOLD])
    fig_imp.update_layout(template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("### 📖 Model Explanation")
    st.info(
        f"This RandomForestClassifier was trained on {metrics['n_train']} matches and "
        f"tested on {metrics['n_test']} unseen matches. It uses pre-match rolling form "
        "(last 5 games), average goals scored/conceded, and team/league identity as "
        "features - never information from the match itself - to avoid data leakage. "
        f"The most influential feature is **{imp_df.iloc[0]['Feature']}**, meaning the "
        "model relies heavily on recent team form when making predictions."
    )


# --------------------------------------------------------------------------- #
# PAGE 7 — FOOTBALL DASHBOARD
# --------------------------------------------------------------------------- #

def page_dashboard():
    st.title("📈 Football Dashboard")
    df = get_dataset_or_stop()

    leagues = ["All Leagues"] + sorted(df["League"].unique().tolist())
    league_filter = st.selectbox("Filter by league", leagues)
    filtered_df = df if league_filter == "All Leagues" else df[df["League"] == league_filter]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🥇 Top Scoring Teams")
        top_scorers = analytics.get_top_scoring_teams(filtered_df, n=10)
        fig = px.bar(top_scorers, x="Goals Scored", y="Team", orientation="h", color_discrete_sequence=[GOLD])
        fig.update_layout(template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🛡️ Best Defensive Teams")
        best_def = analytics.get_best_defensive_teams(filtered_df, n=10)
        fig = px.bar(best_def, x="Goals Conceded Per Match", y="Team", orientation="h", color_discrete_sequence=["#5DADE2"])
        fig.update_layout(template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🏆 League Standings")
    standings = analytics.get_league_standings(filtered_df)
    st.dataframe(standings, use_container_width=True, hide_index=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### ⚽ Average Goals per Match (by Season)")
        trend = analytics.get_goals_per_match_trend(filtered_df)
        fig = px.line(trend, x="Season", y="AvgGoalsPerMatch", markers=True)
        fig.update_traces(line_color=GOLD)
        fig.update_layout(template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("#### ⚖️ Team Comparison")
        teams = utils.get_team_list(filtered_df)
        if len(teams) >= 2:
            t1 = st.selectbox("Team A", teams, index=0, key="cmp_a")
            t2 = st.selectbox("Team B", [t for t in teams if t != t1], index=0, key="cmp_b")
            s1, s2 = analytics.get_team_stats(filtered_df, t1), analytics.get_team_stats(filtered_df, t2)
            comp_df = pd.DataFrame({
                "Metric": ["Wins", "Draws", "Losses", "Goals Scored", "Goals Conceded", "Points"],
                t1: [s1.wins, s1.draws, s1.losses, s1.goals_scored, s1.goals_conceded, s1.points],
                t2: [s2.wins, s2.draws, s2.losses, s2.goals_scored, s2.goals_conceded, s2.points],
            })
            fig = go.Figure(data=[
                go.Bar(name=t1, x=comp_df["Metric"], y=comp_df[t1], marker_color=GOLD),
                go.Bar(name=t2, x=comp_df["Metric"], y=comp_df[t2], marker_color="#5DADE2"),
            ])
            fig.update_layout(template=PLOTLY_TEMPLATE, barmode="group")
            st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# PAGE 8 — EXTRA AI TOOLS
# --------------------------------------------------------------------------- #

def page_extra_tools():
    st.title("🛠️ Extra AI Tools")
    df = get_dataset_or_stop()
    teams = utils.get_team_list(df)

    tool = st.selectbox(
        "Choose a tool",
        [
            "Team Form Analyzer",
            "Momentum Tracker",
            "Match Difficulty Rating",
            "Opponent Weakness Detector",
            "Dream XI Generator",
            "Transfer Recommendation Engine",
        ],
    )

    if tool == "Team Form Analyzer":
        team = st.selectbox("Team", teams, key="form_team")
        if st.button("Analyze Form"):
            res = analytics.team_form_analyzer(df, team)
            st.success(res["summary"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Wins", res["wins"])
            c2.metric("Draws", res["draws"])
            c3.metric("Losses", res["losses"])
            st.metric("Momentum Score", res["momentum"]["score"], help="-100 (poor) to +100 (excellent)")
            st.caption(res["momentum"]["trend"])

    elif tool == "Momentum Tracker":
        team = st.selectbox("Team", teams, key="mom_team")
        if st.button("Track Momentum"):
            res = analytics.momentum_tracker(df, team)
            st.metric("Momentum Score", res["score"])
            st.info(res["trend"])

    elif tool == "Match Difficulty Rating":
        c1, c2 = st.columns(2)
        team = c1.selectbox("Your Team", teams, key="diff_team")
        opponent = c2.selectbox("Opponent", [t for t in teams if t != team], key="diff_opp")
        if st.button("Rate Difficulty"):
            res = analytics.match_difficulty_rating(df, team, opponent)
            st.metric("Difficulty Rating", f"{res['rating']} / 10")
            st.info(res["summary"])

    elif tool == "Opponent Weakness Detector":
        opponent = st.selectbox("Opponent", teams, key="weak_opp")
        if st.button("Detect Weaknesses"):
            res = analytics.opponent_weakness_detector(df, opponent)
            for w in res["weaknesses"]:
                st.warning(w)

    elif tool == "Dream XI Generator":
        team = st.selectbox("Team", teams, key="xi_team")
        if st.button("Generate Dream XI"):
            res = analytics.dream_xi_generator(df, team)
            st.success(f"Recommended Formation: {res['formation']}")
            for position, players in res["lineup"].items():
                if isinstance(players, list):
                    st.markdown(f"**{position}:** {', '.join(players)}")
                else:
                    st.markdown(f"**{position}:** {players}")
            st.caption("Illustrative lineup based on team's statistical attacking/defensive profile.")

    elif tool == "Transfer Recommendation Engine":
        team = st.selectbox("Team", teams, key="transfer_team")
        if st.button("Get Recommendations"):
            res = analytics.transfer_recommendation_engine(df, team)
            for rec in res["recommendations"]:
                st.info(rec)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

PAGE_ROUTER = {
    "🏠 Home": page_home,
    "🔮 Match Prediction": page_prediction,
    "📊 Team Analytics": page_team_analytics,
    "🎲 Match Simulator": page_simulator,
    "🧠 AI Coach Assistant": page_coach,
    "🔍 Explainable AI": page_explainable_ai,
    "📈 Football Dashboard": page_dashboard,
    "🛠️ Extra AI Tools": page_extra_tools,
}

PAGE_ROUTER[page]()
