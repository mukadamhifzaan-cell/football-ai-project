"""
coach.py
--------
AI Coach Assistant: a transparent, rule-based tactical recommendation engine.

Given team form, opponent form, and home/away status, it recommends a
formation, tactical style, pressing strategy, defensive/attacking advice,
set-piece suggestions, and overall match strategy - always explaining WHY.

Deliberately rule-based (not a black-box model) so every recommendation
is fully explainable, which is exactly what a coaching tool needs to be
trustworthy and useful.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CoachRecommendation:
    formation: str
    tactical_style: str
    pressing_strategy: str
    defensive_advice: str
    attacking_advice: str
    set_piece_suggestion: str
    match_strategy: str
    reasoning: list[str] = field(default_factory=list)


def generate_recommendation(
    team_form: float,
    opponent_form: float,
    is_home: bool,
) -> CoachRecommendation:
    """
    Generate a full tactical game plan.

    Parameters
    ----------
    team_form : float
        0.0 - 1.0 win-rate style form indicator for your team.
    opponent_form : float
        0.0 - 1.0 win-rate style form indicator for the opponent.
    is_home : bool
        Whether your team is playing at home.
    """
    reasoning = []
    form_gap = team_form - opponent_form

    # --- Formation & style -------------------------------------------------
    if form_gap > 0.2:
        formation = "4-3-3"
        tactical_style = "Possession-based, front-foot football"
        reasoning.append(
            f"Your form ({team_form:.0%}) is well above the opponent's ({opponent_form:.0%}), "
            "so an attacking, possession-heavy setup should be used to press the advantage."
        )
    elif form_gap < -0.2:
        formation = "5-4-1"
        tactical_style = "Compact, counter-attacking football"
        reasoning.append(
            f"The opponent's form ({opponent_form:.0%}) is well above yours ({team_form:.0%}), "
            "so a defensively solid, counter-attacking shape is safer than an open game."
        )
    else:
        formation = "4-2-3-1"
        tactical_style = "Balanced, flexible football"
        reasoning.append(
            "Form levels are fairly even, so a balanced formation keeps options open in both phases."
        )

    # --- Home/away adjustment -----------------------------------------------
    if is_home:
        reasoning.append("Playing at home allows a slightly higher defensive line and more front-foot pressing, backed by crowd support.")
    else:
        reasoning.append("Playing away favours more disciplined shape and quicker transitions to avoid being caught out of position.")

    # --- Pressing strategy ---------------------------------------------------
    if form_gap >= 0 and is_home:
        pressing_strategy = "High press - win the ball back in the opponent's half"
        reasoning.append("Good form plus home advantage supports an aggressive high press to force early errors.")
    elif form_gap < -0.2:
        pressing_strategy = "Mid-to-low block - press selectively, protect central areas"
        reasoning.append("Against a stronger side, pressing high risks being exposed in behind - a controlled block is safer.")
    else:
        pressing_strategy = "Mid-block press with triggers on wide areas"
        reasoning.append("A mid-block balances risk while still applying organized pressure on the ball.")

    # --- Defensive advice ------------------------------------------------
    if opponent_form > 0.5:
        defensive_advice = "Keep a compact back line, mark opponent's key playmaker tightly, and avoid unnecessary fouls near the box."
    else:
        defensive_advice = "Maintain a standard defensive shape; the opponent's attacking threat looks manageable this game."

    # --- Attacking advice ------------------------------------------------
    if team_form > 0.5:
        attacking_advice = "Use width to stretch the defense, commit fullbacks forward, and look for early crosses into the box."
    else:
        attacking_advice = "Prioritize quick, direct transitions and shots on sight rather than prolonged build-up, to make the most of limited chances."

    # --- Set pieces ------------------------------------------------
    set_piece_suggestion = (
        "Dedicate practice to near-post corner routines and a designated free-kick "
        "taker, since set pieces are a reliable source of goals against organized defenses."
    )

    # --- Overall strategy ------------------------------------------------
    if form_gap > 0.2:
        match_strategy = "Dominate possession, force the opponent deep, and be patient in breaking down a low block."
    elif form_gap < -0.2:
        match_strategy = "Stay compact, absorb pressure, and look to hit the opponent on the counter with pace in wide areas."
    else:
        match_strategy = "Control tempo, avoid unnecessary risks in your own half, and capitalize on set pieces and moments of transition."

    return CoachRecommendation(
        formation=formation,
        tactical_style=tactical_style,
        pressing_strategy=pressing_strategy,
        defensive_advice=defensive_advice,
        attacking_advice=attacking_advice,
        set_piece_suggestion=set_piece_suggestion,
        match_strategy=match_strategy,
        reasoning=reasoning,
    )
