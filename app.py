"""
Stratagem AI — 2×2 Game Theory & Nash Equilibrium
Bloomberg-style dark theme with Nashpy solver and Plotly heatmaps.
"""

import os
from datetime import datetime
from typing import List, Optional, Tuple

import google.generativeai as genai
import graphviz
import nashpy as nash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Global initialization for scenario manager
if "scenarios" not in st.session_state:
    st.session_state.scenarios = []


# Predefined classic scenarios (2×2 games)
SCENARIOS = {
    "Prisoner's Dilemma": {
        "row_player": "Prisoner 1",
        "col_player": "Prisoner 2",
        "row_strategies": ["Cooperate", "Defect"],
        "col_strategies": ["Cooperate", "Defect"],
        "A": [[3.0, 0.0], [5.0, 1.0]],
        "B": [[3.0, 5.0], [0.0, 1.0]],
    },
    "Pricing War": {
        "row_player": "Firm A",
        "col_player": "Firm B",
        "row_strategies": ["Maintain Price", "Discount 10%"],
        "col_strategies": ["Maintain Price", "Discount 10%"],
        "A": [[20.0, 10.0], [25.0, 12.0]],
        "B": [[20.0, 25.0], [10.0, 12.0]],
    },
    "Market Entry": {
        "row_player": "Incumbent",
        "col_player": "Entrant",
        "row_strategies": ["Stay Out", "Enter"],
        "col_strategies": ["Accommodate", "Fight"],
        "A": [[0.0, 0.0], [15.0, -5.0]],
        "B": [[10.0, 5.0], [-5.0, -10.0]],
    },
}


def apply_scenario_from_state() -> None:
    """Callback used by 'Load Classic Scenario' selectbox to populate session_state."""
    template = st.session_state.get("scenario_template", "Custom")
    cfg = SCENARIOS.get(template)
    if not cfg:
        return

    # Force 2×2 for classic scenarios
    st.session_state["matrix_size"] = "2×2"

    st.session_state["row_player_name"] = cfg["row_player"]
    st.session_state["col_player_name"] = cfg["col_player"]

    for i, s in enumerate(cfg["row_strategies"]):
        st.session_state[f"rs_{i}"] = s
    for j, s in enumerate(cfg["col_strategies"]):
        st.session_state[f"cs_{j}"] = s

    for i, row in enumerate(cfg["A"]):
        for j, val in enumerate(row):
            st.session_state[f"A_{i}_{j}"] = float(val)
            st.session_state[f"B_{i}_{j}"] = float(cfg["B"][i][j])


def save_scenario() -> None:
    """Callback used by 'Capture current scenario' to snapshot the current game into session_state.scenarios."""
    matrix_size = st.session_state.get("matrix_size", "2×2")
    n = 2 if matrix_size == "2×2" else 3
    row_player = st.session_state.get("row_player_name", "Row player")
    col_player = st.session_state.get("col_player_name", "Column player")

    row_strategies = [st.session_state.get(f"rs_{i}", f"Strategy {i + 1}") for i in range(n)]
    col_strategies = [st.session_state.get(f"cs_{j}", f"Strategy {j + 1}") for j in range(n)]

    A = np.zeros((n, n))
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = float(st.session_state.get(f"A_{i}_{j}", 0.0))
            B[i, j] = float(st.session_state.get(f"B_{i}_{j}", 0.0))

    try:
        equilibria = solve_nash(A, B)
    except Exception:
        equilibria = []

    if equilibria:
        sigma_u, sigma_r = equilibria[0]
        eu = float(np.dot(sigma_u, np.dot(A, sigma_r)))
        er = float(np.dot(sigma_r, np.dot(B.T, sigma_u)))
        score = compute_confidence_score(A, B, sigma_u, sigma_r)
    else:
        eu = er = float("nan")
        score = 0

    snapshot = {
        "matrix_size": matrix_size,
        "row_player": row_player,
        "col_player": col_player,
        "row_strategies": row_strategies,
        "col_strategies": col_strategies,
        "A": A.tolist(),
        "B": B.tolist(),
        "row_expected_payoff": eu,
        "col_expected_payoff": er,
        "confidence_score": score,
    }
    st.session_state.scenarios.append(snapshot)

# ——— Page config (Bloomberg aesthetic) ———
st.set_page_config(
    page_title="Stratagem AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ——— McKinsey-style navy & slate theme ———
MCKINSEY_CSS = """
<style>
    :root {
        --navy-900: #020617;
        --navy-800: #02091f;
        --navy-700: #0b1220;
        --slate-700: #1f2937;
        --slate-600: #374151;
        --accent-cyan: #38bdf8;
        --accent-amber: #fbbf24;
    }

    /* Base app shell */
    .stApp {
        background: radial-gradient(circle at top left, #020617 0, #02091f 40%, #020617 85%);
        color: #e5e7eb;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #02091f 50%, #020617 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.4);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #e5e7eb;
    }

    /* Headers */
    h1, h2, h3 {
        color: #e5e7eb !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    /* Body text */
    p, label, span {
        color: #cbd5f5 !important;
    }

    /* Inputs */
    .stTextInput input, .stNumberInput input {
        background: rgba(15, 23, 42, 0.85) !important;
        color: #e5e7eb !important;
        border: 1px solid rgba(148, 163, 184, 0.6) !important;
        border-radius: 6px;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 0 1px var(--accent-cyan);
    }

    /* Primary buttons (Solve / Capture) */
    .stButton > button {
        background: #001233 !important; /* Deep navy */
        color: #ffffff !important;
        border: 1px solid #0f172a !important;
        font-weight: 600;
        border-radius: 6px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.45);
    }
    .stButton > button:hover {
        background: #011845 !important;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.6);
    }

    /* Metric cards: light, consulting-style KPIs */
    [data-testid="stMetric"] {
        background: #f8f9fa !important;
        border-radius: 1px;
        padding: 16px 18px;
        border: 1px solid #dee2e6 !important;
        min-width: 0;
    }
    [data-testid="stMetricLabel"] {
        color: #111827 !important; /* darker for stronger contrast */
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        line-height: 1.25 !important;
        max-width: none !important;
        word-wrap: break-word;
    }
    /* Ensure all nested label elements inherit the dark color and avoid truncation */
    [data-testid="stMetricLabel"] * {
        color: #111827 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        max-width: none !important;
        display: block !important;
    }
    [data-testid="stMetricValue"] {
        color: #212529 !important;
        font-weight: 600;
        font-size: 1.4rem;
    }

    /* Progress bar */
    [role="progressbar"] > div {
        background: linear-gradient(90deg, #22c55e, #facc15, #f97316);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.5);
        border-radius: 8px;
    }

    /* DataFrames */
    .dataframe {
        border: 1px solid rgba(148, 163, 184, 0.5) !important;
        border-radius: 8px;
    }

    /* Reduce top padding so title is closer to top */
    .main .block-container {
        padding-top: 0.5rem;
    }

    /* Group container for payoff matrices */
    .payoff-group {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 14px 16px;
        background: rgba(15, 23, 42, 0.8);
        margin-bottom: 16px;
    }

    /* AI Insights panel */
    .ai-panel {
        background: #eef6ff;
        border-radius: 8px;
        border: 1px solid #bfdbfe;
        padding: 16px 18px;
        margin-top: 8px;
        color: #0f172a !important;
    }

    /* Align sidebar number input boxes within their columns */
    .stNumberInput > div {
        width: 100% !important;
    }

    /* Normalize sidebar payoff labels so inputs align horizontally */
    .stNumberInput label {
        font-size: 0.8rem !important;
        display: flex !important;
        align-items: flex-end !important;
        min-height: 2.6rem !important; /* ensures consistent vertical offset */
    }

    /* Increase contrast for slider labels (Equilibrium behaviour) */
    .stSlider label, .stSlider span {
        color: #e5e7eb !important;
    }

    /* Ghost-style download buttons */
    [data-testid="stDownloadButton"] > button {
        background: transparent !important;
        border: 1px solid #ffffff !important;
        color: #e5e7eb !important;
        border-radius: 6px !important;
        box-shadow: none !important;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background: rgba(255, 255, 255, 0.06) !important;
    }

    /* Center Plotly matrix within its container */
    .payoff-group .js-plotly-plot {
        margin-left: auto !important;
        margin-right: auto !important;
    }

    div[data-testid="stVerticalBlock"] > div {
        border-radius: 10px;
    }
</style>
"""
st.markdown(MCKINSEY_CSS, unsafe_allow_html=True)


def build_payoff_matrices(user_vals: List[List[float]], rival_vals: List[List[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Row player (User) and column player (Rival) payoff matrices (N×M)."""
    A = np.array(user_vals, dtype=float)
    B = np.array(rival_vals, dtype=float)
    return A, B


def solve_nash(A: np.ndarray, B: np.ndarray) -> list:
    """Compute all Nash equilibria (support enumeration)."""
    game = nash.Game(A, B)
    equilibria = list(game.support_enumeration())
    return equilibria


def get_pure_nash_cells(A: np.ndarray, B: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (row, col) cells that are pure-strategy Nash equilibria (best responses)."""
    n_rows, n_cols = A.shape
    nash_cells: List[Tuple[int, int]] = []
    for i in range(n_rows):
        for j in range(n_cols):
            user_best = A[i, j] >= np.max(A[:, j])  # best row response to column j
            rival_best = B[i, j] >= np.max(B[i, :])  # best column response to row i
            if bool(user_best) and bool(rival_best):
                nash_cells.append((i, j))
    return nash_cells


def compute_confidence_score(
    A: np.ndarray,
    B: np.ndarray,
    sigma_u: np.ndarray,
    sigma_r: np.ndarray,
) -> int:
    """
    Heuristic "Strategic Confidence Score" in [0, 100] based on payoff margins.

    - If equilibrium is effectively pure, use the minimum advantage each player has
      over their best deviation (relative to the overall payoff span).
    - If mixed, return a moderate confidence.
    """
    # Detect pure strategies (one strategy ~1.0)
    row_idx = int(np.argmax(sigma_u))
    col_idx = int(np.argmax(sigma_r))
    is_pure_user = bool(np.isclose(sigma_u[row_idx], 1.0))
    is_pure_rival = bool(np.isclose(sigma_r[col_idx], 1.0))

    if is_pure_user and is_pure_rival:
        # If both players have a strictly dominant strategy that defines this cell,
        # treat the equilibrium as maximally stable.
        dom_row, dom_col = find_strictly_dominant_strategies(A, B)
        if dom_row is not None and dom_col is not None:
            return 100

        # Payoff at the equilibrium cell
        a_eq = float(A[row_idx, col_idx])
        b_eq = float(B[row_idx, col_idx])

        # Best deviation for each player holding opponent fixed
        alt_user = float(A[1 - row_idx, col_idx])
        alt_rival = float(B[row_idx, 1 - col_idx])

        user_margin = max(0.0, a_eq - alt_user)
        rival_margin = max(0.0, b_eq - alt_rival)
        min_margin = min(user_margin, rival_margin)

        # Normalise by overall payoff span across both matrices
        combined = np.concatenate([A.flatten(), B.flatten()])
        span = float(combined.max() - combined.min())
        if span <= 0:
            span = 1.0

        normalized = max(0.0, min(1.0, min_margin / span))
        # Map to 50–100 so pure NE are always stronger than mixed cases
        score = int(50 + normalized * 50)
    else:
        # Mixed equilibria tend to be less "stable" in a classical sense
        score = 40

    return score


def find_strictly_dominant_strategies(A: np.ndarray, B: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    """
    Return (row_idx, col_idx) for strictly dominant strategies if they exist, else (None, None).
    Row strategy i strictly dominates k if A[i,j] > A[k,j] for all j.
    Col strategy j strictly dominates l if B[i,j] > B[i,l] for all i.
    """
    n_rows, n_cols = A.shape

    dom_row: Optional[int] = None
    for i in range(n_rows):
        dominates_all = True
        for k in range(n_rows):
            if k == i:
                continue
            if not np.all(A[i, :] > A[k, :]):
                dominates_all = False
                break
        if dominates_all:
            dom_row = i
            break

    dom_col: Optional[int] = None
    for j in range(n_cols):
        dominates_all = True
        for l in range(n_cols):
            if l == j:
                continue
            if not np.all(B[:, j] > B[:, l]):
                dominates_all = False
                break
        if dominates_all:
            dom_col = j
            break

    return dom_row, dom_col


def is_pareto_efficient(A: np.ndarray, B: np.ndarray, cell: Tuple[int, int]) -> bool:
    """True if no other cell improves both players strictly (Pareto efficient)."""
    i, j = cell
    u0, r0 = float(A[i, j]), float(B[i, j])
    n_rows, n_cols = A.shape
    for r in range(n_rows):
        for c in range(n_cols):
            if r == i and c == j:
                continue
            if float(A[r, c]) > u0 and float(B[r, c]) > r0:
                return False
    return True


def format_equilibria_prompt(
    equilibria: list,
    A: np.ndarray,
    B: np.ndarray,
    row_player: str,
    col_player: str,
    row_strategies: List[str],
    col_strategies: List[str],
) -> str:
    """Format full game context (players, strategies, payoffs, equilibria) for the Gemini prompt."""
    n_rows, n_cols = A.shape
    lines: List[str] = []

    lines.append(f"Row player: {row_player}")
    lines.append(f"Column player: {col_player}")
    lines.append("Row player strategies:")
    for i, s in enumerate(row_strategies):
        lines.append(f"  - Row {i + 1}: {s}")
    lines.append("Column player strategies:")
    for j, s in enumerate(col_strategies):
        lines.append(f"  - Column {j + 1}: {s}")
    lines.append("")
    lines.append("Payoff matrix (Row payoff, Column payoff):")
    for i in range(n_rows):
        for j in range(n_cols):
            lines.append(
                f"  - {row_player} '{row_strategies[i]}' vs {col_player} '{col_strategies[j]}': "
                f"({A[i, j]:.2f}, {B[i, j]:.2f})"
            )
    lines.append("")

    if not equilibria:
        lines.append("No Nash equilibrium was found via support enumeration.")
    else:
        for idx, (sigma_u, sigma_r) in enumerate(equilibria, start=1):
            eu = float(np.dot(sigma_u, np.dot(A, sigma_r)))
            er = float(np.dot(sigma_r, np.dot(B.T, sigma_u)))
            row_mix = ", ".join(f"{s}={p * 100:.2f}%" for s, p in zip(row_strategies, sigma_u))
            col_mix = ", ".join(f"{s}={p * 100:.2f}%" for s, p in zip(col_strategies, sigma_r))
            lines.append(f"Equilibrium {idx}:")
            lines.append(f"  - {row_player} mixed strategy: {row_mix}")
            lines.append(f"  - {col_player} mixed strategy: {col_mix}")
            lines.append(f"  - Expected payoffs: {row_player}={eu:.2f}, {col_player}={er:.2f}")
    return "\n".join(lines)


def build_executive_brief_markdown(
    A: np.ndarray,
    B: np.ndarray,
    equilibria: list,
    ai_analysis: str,
    row_player: str = "Row player",
    col_player: str = "Column player",
    row_strategies: Optional[List[str]] = None,
    col_strategies: Optional[List[str]] = None,
) -> str:
    """
    Create a recruiter-ready consulting-style Markdown brief using the
    McKinsey-style Minto Pyramid:
    1. Executive Summary
    2. Situation
    3. Complication
    4. Resolution
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_rows, n_cols = A.shape

    row_strategies = row_strategies or [f"Strategy {i + 1}" for i in range(n_rows)]
    col_strategies = col_strategies or [f"Strategy {j + 1}" for j in range(n_cols)]

    # Helper: generic markdown table for any matrix
    def md_table_from_matrix(mat: np.ndarray, title: str, r_labels: List[str], c_labels: List[str]) -> str:
        header = "|            | " + " | ".join(c_labels) + " |\n"
        sep = "|------------|" + "|".join(["---------"] * len(c_labels)) + "|\n"
        rows = []
        for i, r_lbl in enumerate(r_labels):
            row_vals = " | ".join(f"{float(mat[i, j]):.2f}" for j in range(len(c_labels)))
            rows.append(f"| {r_lbl:<10} | {row_vals} |\n")
        return f"**{title}**\n\n" + header + sep + "".join(rows)

    # Primary equilibrium: first equilibrium if available
    primary_eq = equilibria[0] if equilibria else None
    if primary_eq:
        sigma_u, sigma_r = primary_eq
        eu = float(np.dot(sigma_u, np.dot(A, sigma_r)))
        er = float(np.dot(sigma_r, np.dot(B.T, sigma_u)))
        score = compute_confidence_score(A, B, sigma_u, sigma_r)
        # For narrative, pick most-probable pure strategies
        row_idx = int(np.argmax(sigma_u))
        col_idx = int(np.argmax(sigma_r))
        primary_row_strat = row_strategies[row_idx]
        primary_col_strat = col_strategies[col_idx]
        pure_cells = get_pure_nash_cells(A, B)
        primary_cell = pure_cells[0] if pure_cells else (row_idx, col_idx)
        pareto_flag = "Yes" if is_pareto_efficient(A, B, primary_cell) else "No"
    else:
        eu = er = 0.0
        score = 0
        primary_row_strat = primary_col_strat = "N/A"
        pareto_flag = "N/A"

    # Best joint outcome (for complication section)
    combined = A + B
    best_r, best_c = np.unravel_index(np.argmax(combined), combined.shape)
    best_u, best_rv = float(A[best_r, best_c]), float(B[best_r, best_c])

    analysis_clean = (ai_analysis or "").strip()
    if not analysis_clean:
        analysis_clean = "_No AI analysis generated (missing API key or request failed)._"

    # Section 1: Executive Summary
    exec_summary_lines = [
        "## 1. Executive Summary",
        "",
        f"- **Core recommendation:** {row_player} should primarily pursue **{primary_row_strat}**, "
        f"while {col_player} should primarily pursue **{primary_col_strat}**, aligning on the "
        "primary Nash equilibrium of the game.",
    ]
    if primary_eq:
        exec_summary_lines.append(
            f"- **Primary Nash Equilibrium:** Expected profits of **{eu:.2f}** for {row_player} "
            f"and **{er:.2f}** for {col_player}, with a **Strategic Confidence Score of {score}/100**."
        )
    else:
        exec_summary_lines.append(
            "- **Primary Nash Equilibrium:** No equilibrium identified; treat this market as unstable and "
            "focus on preserving flexibility and option value."
        )

    # Section 2: Situation
    situation_lines = [
        "## 2. Situation",
        "",
        f"- The game is played between **{row_player}** (row player) and **{col_player}** (column player).",
        f"- {row_player} strategies: " + ", ".join(f"**{s}**" for s in row_strategies) + ".",
        f"- {col_player} strategies: " + ", ".join(f"**{s}**" for s in col_strategies) + ".",
        "",
        md_table_from_matrix(A, f"{row_player} payoff matrix (e.g., profit in millions)", row_strategies, col_strategies),
        "",
        md_table_from_matrix(B, f"{col_player} payoff matrix (e.g., profit in millions)", row_strategies, col_strategies),
    ]

    # Section 3: Complication
    complication_lines = [
        "## 3. Complication",
        "",
        "- Payoffs are not fully aligned; each firm has incentives to deviate from cooperative pricing.",
        f"- The **best joint outcome** (maximising {row_player}+{col_player} profit) delivers approximately "
        f"**{best_u + best_rv:.2f}** in combined profit, but may not be self-enforcing.",
    ]
    if primary_eq:
        diff = (best_u + best_rv) - (eu + er)
        if diff > 1e-6:
            complication_lines.append(
                f"- The recommended Nash outcome leaves roughly **{diff:.2f}** of joint profit on the table "
                "relative to the best cooperative point, highlighting a classic tension between "
                "individual rationality and joint value creation."
            )
        complication_lines.append(
            f"- Pareto efficiency of the primary equilibrium: **{pareto_flag}** "
            "(\"No\" implies a strictly better outcome exists for both firms)."
        )
    else:
        complication_lines.append(
            "- The absence of a stable equilibrium implies high sensitivity to assumptions and a risk of "
            "price wars or repeated under-cutting."
        )

    # Section 4: Resolution
    resolution_lines = [
        "## 4. Resolution",
        "",
        "- **Nash Equilibrium math**",
    ]
    if primary_eq:
        sigma_u, sigma_r = primary_eq
        row_mix = ", ".join(f"{s}: {p * 100:.2f}%" for s, p in zip(row_strategies, sigma_u))
        col_mix = ", ".join(f"{s}: {p * 100:.2f}%" for s, p in zip(col_strategies, sigma_r))
        resolution_lines.extend(
            [
                f"  - {row_player} mixed strategy: {row_mix}.",
                f"  - {col_player} mixed strategy: {col_mix}.",
                f"  - Expected payoff at equilibrium: **{row_player} = {eu:.2f}**, **{col_player} = {er:.2f}**.",
                f"  - **Strategic Confidence Score:** {score}/100, based on deviation margins around the equilibrium "
                "(higher = less incentive for either side to deviate unilaterally).",
            ]
        )
    else:
        resolution_lines.append(
            "  - No equilibrium was identified, so recommendations should emphasise guardrails, "
            "triggers for re-evaluating pricing, and scenario planning rather than a single point solution."
        )

    resolution_lines.extend(
        [
            "",
            "### Supporting AI perspective",
            analysis_clean,
        ]
    )

    md = "\n".join(
        [
            "# Executive Brief — Game Theory Strategic Readout",
            f"**Generated:** {now}",
            "",
            *exec_summary_lines,
            "",
            *situation_lines,
            "",
            *complication_lines,
            "",
            *resolution_lines,
            "",
        ]
    )
    return md


def get_static_strategic_tip(A: np.ndarray, B: np.ndarray, equilibria: list) -> str:
    """
    Fallback consulting-style guidance when AI analysis is unavailable.

    Uses simple structure of equilibria to classify into familiar patterns
    (race to the bottom, coordination, or mixed-strategy standoff).
    """
    pure_cells = get_pure_nash_cells(A, B)

    # Pareto-better cells relative to first pure NE (if any)
    pareto_better_cells: List[Tuple[int, int]] = []
    if pure_cells:
        i_ne, j_ne = pure_cells[0]
        u_ne, r_ne = float(A[i_ne, j_ne]), float(B[i_ne, j_ne])
        n_rows, n_cols = A.shape
        for r in range(n_rows):
            for c in range(n_cols):
                if r == i_ne and c == j_ne:
                    continue
                if float(A[r, c]) > u_ne and float(B[r, c]) > r_ne:
                    pareto_better_cells.append((r, c))

    if not equilibria:
        return (
            "- The game has no clear Nash equilibrium, signalling an unstable competitive"
            " environment.\n"
            "- Expect rapid moves and countermoves; emphasise option value, flexibility,"
            " and short planning cycles."
        )

    if len(pure_cells) == 1:
        i, j = pure_cells[0]
        eq_u, eq_r = float(A[i, j]), float(B[i, j])

        # Check if there exists a jointly better outcome for both players
        jointly_better = False
        for r in range(2):
            for c in range(2):
                if r == i and c == j:
                    continue
                if A[r, c] > eq_u and B[r, c] > eq_r:
                    jointly_better = True
                    break
            if jointly_better:
                break

        if jointly_better:
            return (
                "- The unique equilibrium resembles a Prisoner's Dilemma: both sides"
                " converge on a 'race to the bottom' outcome that is dominated by a more"
                " cooperative alternative.\n"
                "- To escape this dynamic, emphasise credible commitments, reputation,"
                " and mechanisms that reward mutual restraint (e.g., tiered pricing,"
                " long-term contracts)."
            )

        return (
            "- The game exhibits a single dominant-position equilibrium, with one"
            " configuration clearly outperforming nearby deviations.\n"
            "- Strategy should focus on either defending this position (if you hold it)"
            " or selectively disrupting the assumptions that make it attractive (if you"
            " are the challenger)."
        )

    if len(pure_cells) > 1:
        return (
            "- Multiple pure-strategy equilibria indicate a coordination problem: several"
            " outcomes are self-reinforcing.\n"
            "- The side that shapes expectations (through signalling, standards, or"
            " ecosystem design) is likely to lock in the favourable equilibrium."
        )

    # No pure equilibria but at least one mixed equilibrium
    return (
        "- The absence of pure-strategy equilibria pushes both players into mixed"
        " strategies, increasing unpredictability.\n"
        "- Winning here depends less on a single move and more on information"
        " advantages, speed of reaction, and the ability to sustain calculated"
        " risk over time."
    )


def build_sensitivity_dataframe(
    A: np.ndarray,
    B: np.ndarray,
    factors: List[float],
) -> pd.DataFrame:
    """
    Run sensitivity on Rival's payoffs by scaling B with each factor.

    Returns a DataFrame with how the equilibrium mixed strategies shift as
    Rival becomes more/less aggressive.
    """
    records = []
    for f in factors:
        B_adj = B * f
        try:
            eqs = solve_nash(A, B_adj)
        except Exception:
            eqs = []

        if eqs:
            sigma_u, sigma_r = eqs[0]
            user_s1 = float(sigma_u[0])
            rival_s1 = float(sigma_r[0])
        else:
            user_s1 = float("nan")
            rival_s1 = float("nan")

        records.append(
            {
                "Rival aggressiveness (%)": int(round(f * 100)),
                "User: Prob(Strategy 1)": user_s1,
                "Rival: Prob(Strategy 1)": rival_s1,
            }
        )

    return pd.DataFrame.from_records(records)


def build_equilibria_dataframe(
    equilibria: list,
    row_player: str,
    col_player: str,
    row_strategies: List[str],
    col_strategies: List[str],
) -> pd.DataFrame:
    """Tabular view of equilibria as probabilities (percent)."""
    records = []
    for idx, (sigma_u, sigma_r) in enumerate(equilibria, start=1):
        for i, s in enumerate(row_strategies):
            records.append(
                {
                    "Equilibrium": idx,
                    "Player": row_player,
                    "Strategy": s,
                    "Probability (%)": float(sigma_u[i]) * 100.0,
                }
            )
        for j, s in enumerate(col_strategies):
            records.append(
                {
                    "Equilibrium": idx,
                    "Player": col_player,
                    "Strategy": s,
                    "Probability (%)": float(sigma_r[j]) * 100.0,
                }
            )
    return pd.DataFrame.from_records(records)


def build_decision_tree_graph(
    row_player: str,
    col_player: str,
    row_strategies: List[str],
    col_strategies: List[str],
    A: np.ndarray,
    B: np.ndarray,
    highlight_cells: List[Tuple[int, int]],
) -> graphviz.Digraph:
    """
    Build a 2×2 sequential decision tree:
    - Root: row player choice
    - Next: column player choice
    - Leaves: labelled with (row payoff, col payoff).
    Pure-strategy Nash cells in `highlight_cells` are drawn in green.
    """
    g = graphviz.Digraph(format="svg")
    g.attr(rankdir="LR", bgcolor="transparent", nodesep="0.7", ranksep="0.8")

    # Only handle 2×2 clearly; fall back otherwise
    if A.shape != (2, 2):
        return g

    # Nodes
    g.node("A", row_player, shape="box", style="rounded,filled", fillcolor="#0f172a", fontcolor="#e5e7eb")

    g.node("A0", row_strategies[0], shape="box", style="rounded,filled", fillcolor="#1e293b", fontcolor="#e5e7eb")
    g.node("A1", row_strategies[1], shape="box", style="rounded,filled", fillcolor="#1e293b", fontcolor="#e5e7eb")

    g.node("B0_0", col_strategies[0], shape="box", style="rounded,filled", fillcolor="#020617", fontcolor="#e5e7eb")
    g.node("B0_1", col_strategies[1], shape="box", style="rounded,filled", fillcolor="#020617", fontcolor="#e5e7eb")
    g.node("B1_0", col_strategies[0], shape="box", style="rounded,filled", fillcolor="#020617", fontcolor="#e5e7eb")
    g.node("B1_1", col_strategies[1], shape="box", style="rounded,filled", fillcolor="#020617", fontcolor="#e5e7eb")

    # Leaves with payoffs
    def leaf_id(i: int, j: int) -> str:
        return f"L_{i}{j}"

    for i in range(2):
        for j in range(2):
            payoff_label = f"{row_strategies[i]} / {col_strategies[j]}\\n({A[i, j]:.2f}, {B[i, j]:.2f})"
            is_ne = (i, j) in highlight_cells
            color = "#16a34a" if is_ne else "#4b5563"
            g.node(
                leaf_id(i, j),
                payoff_label,
                shape="box",
                style="rounded,filled",
                fillcolor="#020617" if is_ne else "#020617",
                fontcolor="#e5e7eb",
                color=color,
                penwidth="2" if is_ne else "1",
            )

    # Edges
    def edge_color(i: int, j: int) -> str:
        return "#16a34a" if (i, j) in highlight_cells else "#64748b"

    g.edge("A", "A0", label=row_strategies[0], color=edge_color(0, 0))
    g.edge("A", "A1", label=row_strategies[1], color=edge_color(1, 0))

    g.edge("A0", "B0_0", label=col_strategies[0], color=edge_color(0, 0))
    g.edge("A0", "B0_1", label=col_strategies[1], color=edge_color(0, 1))
    g.edge("A1", "B1_0", label=col_strategies[0], color=edge_color(1, 0))
    g.edge("A1", "B1_1", label=col_strategies[1], color=edge_color(1, 1))

    g.edge("B0_0", leaf_id(0, 0), color=edge_color(0, 0))
    g.edge("B0_1", leaf_id(0, 1), color=edge_color(0, 1))
    g.edge("B1_0", leaf_id(1, 0), color=edge_color(1, 0))
    g.edge("B1_1", leaf_id(1, 1), color=edge_color(1, 1))

    return g


# Stable Gemini model name for the v1beta API
# 1.5 Flash is no longer available on v1beta for generateContent in your project/region,
# so we use the recommended 2.0 Flash model instead.
GEMINI_MODEL = "gemini-2.0-flash"


def get_gemini_strategic_analysis(api_key: str, equilibria_text: str) -> str:
    """Call Gemini as a management consultant; return a 2-section markdown explanation."""
    genai.configure(api_key=api_key.strip())
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = (
        "Please analyze the following 2-player normal-form game. "
        "Provide your response in exactly two markdown sections:\n"
        'First, an "### Executive Summary" paragraph explaining the equilibrium (why it occurs, in clear business language).\n'
        'Second, a "### Strategic Recommendation" paragraph advising the players on what to do (pricing posture, signalling, investments, etc.).\n'
        "Avoid raw mathematical notation and do not repeat payoff numbers verbatim; summarise them in words instead.\n\n"
        "Game and equilibrium data:\n"
        f"{equilibria_text}"
    )
    response = model.generate_content(prompt)
    return response.text if response.text else "No response generated."


def plot_payoff_heatmap(
    A: np.ndarray,
    B: np.ndarray,
    nash_cells: Optional[List[Tuple[int, int]]] = None,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    row_player: str = "Row player",
    col_player: str = "Column player",
    pareto_better_cells: Optional[List[Tuple[int, int]]] = None,
) -> go.Figure:
    """N×M heatmap: axes Row player (Y), Column player (X); cell text (Row payoff, Col payoff); highlight pure Nash in gold."""
    n_rows, n_cols = A.shape
    x_labels = col_labels or [f"Rival {j + 1}" for j in range(n_cols)]
    y_labels = row_labels or [f"User {i + 1}" for i in range(n_rows)]

    # Cell text: (User Payoff, Rival Payoff)
    text = [[f"({A[i, j]:.2f}, {B[i, j]:.2f})" for j in range(n_cols)] for i in range(n_rows)]

    # z for color scale (e.g. total payoff); use Greys-friendly range
    z = A + B
    z_flat = z.flatten()
    z_min, z_max = float(np.min(z_flat)), float(np.max(z_flat))
    if z_max == z_min:
        z_max = z_min + 1

    combined_label = f"{row_player} + {col_player}" if row_player and col_player else "Row+Column"

    heat = go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        hoverongaps=False,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title=combined_label, tickformat=".2f"),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>(Row, Column): %{customdata}<br>Total: %{z:.2f}<extra></extra>",
        customdata=text,
    )
    fig = go.Figure(data=[heat])

    # Add per-cell annotations with dynamic contrast
    # For Viridis: low values are dark (purple) → light text,
    # high values are bright (yellow/green) → dark text.
    z_norm = (z - float(np.min(z))) / (float(np.max(z)) - float(np.min(z)) + 1e-9)
    annotations = []
    for i in range(n_rows):
        for j in range(n_cols):
            # Use light text on darkest cells, dark text on bright cells
            font_color = "#f9fafb" if z_norm[i, j] < 0.3 else "#111827"
            annotations.append(
                dict(
                    x=x_labels[j],
                    y=y_labels[i],
                    text=text[i][j],
                    showarrow=False,
                    font=dict(color=font_color, size=16),
                )
            )
    fig.update_layout(annotations=annotations)

    # Gold highlight for Nash equilibrium cell(s)
    nash_cells = nash_cells or []
    pareto_better_cells = pareto_better_cells or []
    shapes = []
    for (i, j) in nash_cells:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=j - 0.5,
                x1=j + 0.5,
                y0=i - 0.5,
                y1=i + 0.5,
                line=dict(width=4, color="Gold"),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )
        )
    # Blue highlight for Pareto-better outcomes relative to NE
    for (i, j) in pareto_better_cells:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=j - 0.5,
                x1=j + 0.5,
                y0=i - 0.5,
                y1=i + 0.5,
                line=dict(width=4, color="#0000FF"),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )
        )
    fig.update_layout(shapes=shapes)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(13,17,23,0)",
        plot_bgcolor="rgba(22,27,34,0.9)",
        font=dict(color="#c9d1d9", size=12),
        margin=dict(l=60, r=60, t=50, b=50),
        # Avoid stray 'undefined' rendering in some frontends by explicitly clearing title
        title_text="",
        title_font=dict(size=16, color="#f0f6fc"),
        height=380,
        xaxis=dict(tickfont=dict(color="#8b949e")),
        yaxis=dict(tickfont=dict(color="#8b949e"), autorange="reversed"),
    )
    return fig


def main():
    st.title("📊 Stratagem AI")
    st.caption("2×2 Game Theory — Nash Equilibrium & Payoff Analysis")

    # ——— Sidebar: API Key & Payoff inputs ———
    with st.sidebar:
        st.header("Settings")
        if "history" not in st.session_state:
            st.session_state["history"] = []
        if "scenarios" not in st.session_state:
            st.session_state["scenarios"] = []
        if "last_ai_analysis" not in st.session_state:
            st.session_state["last_ai_analysis"] = ""

        scenario_template = st.selectbox(
            "Load Classic Scenario",
            options=["Custom", "Prisoner's Dilemma", "Pricing War", "Market Entry"],
            index=0,
            help="Auto-populate players, strategies, and payoffs for standard games.",
            key="scenario_template",
            on_change=apply_scenario_from_state,
        )

        st.subheader("Scenario labels")
        row_player = st.text_input("Row player name", value="Firm A", key="row_player_name")
        col_player = st.text_input("Column player name", value="Firm B", key="col_player_name")

        matrix_size = st.selectbox(
            "Matrix size",
            options=["2×2", "3×3"],
            index=0,
            key="matrix_size",
        )
        n = 2 if matrix_size == "2×2" else 3

        st.markdown("**Strategy labels**")
        if n == 2 and scenario_template == "Prisoner's Dilemma":
            default_row_labels = ["Cooperate", "Defect"]
            default_col_labels = ["Cooperate", "Defect"]
        elif n == 2 and scenario_template == "Pricing War":
            default_row_labels = ["Maintain Price", "Discount 10%"]
            default_col_labels = ["Maintain Price", "Discount 10%"]
        elif n == 2 and scenario_template == "Market Entry":
            default_row_labels = ["Stay Out", "Enter"]
            default_col_labels = ["Accommodate", "Fight"]
        elif n == 2:
            default_row_labels = ["Maintain Price", "Discount 10%"]
            default_col_labels = ["Maintain Price", "Discount 10%"]
        else:
            default_row_labels = [f"Strategy {i + 1}" for i in range(n)]
            default_col_labels = [f"Strategy {j + 1}" for j in range(n)]

        row_strategies = [
            st.text_input(
                f"{row_player} strategy {i + 1}",
                value=default_row_labels[i],
                key=f"rs_{i}",
            )
            for i in range(n)
        ]
        col_strategies = [
            st.text_input(
                f"{col_player} strategy {j + 1}",
                value=default_col_labels[j],
                key=f"cs_{j}",
            )
            for j in range(n)
        ]
        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Paste key or set GEMINI_API_KEY",
            help="Get a key at [Google AI Studio](https://aistudio.google.com/apikey). Or set env var GEMINI_API_KEY.",
        )
        # Use sidebar key if non-empty, else fall back to environment
        api_key = (api_key_input or "").strip() or os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
        if api_key:
            st.success("API key set")

        st.divider()
        st.subheader(f"Payoff matrix ({matrix_size})")

        # Default 2×2 business cases (payoffs interpreted as profit, e.g. in millions)
        if n == 2 and scenario_template == "Prisoner's Dilemma":
            # (Cooperate, Cooperate) best joint; (Defect, Defect) is NE
            default_A = np.array([[3.0, 0.0], [5.0, 1.0]])
            default_B = np.array([[3.0, 5.0], [0.0, 1.0]])
        elif n == 2 and scenario_template == "Pricing War":
            # Both discount is NE; mutual margin erosion
            default_A = np.array([[20.0, 10.0], [25.0, 12.0]])
            default_B = np.array([[20.0, 25.0], [10.0, 12.0]])
        elif n == 2 and scenario_template == "Market Entry":
            # Simple entry/accommodation vs. fight structure
            default_A = np.array([[0.0, 0.0], [15.0, -5.0]])
            default_B = np.array([[10.0, 5.0], [-5.0, -10.0]])
        elif n == 2:
            # Custom baseline symmetric pricing game
            default_A = np.array([[20.0, 15.0], [5.0, 10.0]])
            default_B = np.array([[20.0, 15.0], [5.0, 10.0]])
        else:
            default_A = np.zeros((n, n))
            default_B = np.zeros((n, n))

        st.markdown(f"**{row_player} (row) payoffs**")
        user_vals: List[List[float]] = []
        for i in range(n):
            cols = st.columns(n)
            row = []
            for j in range(n):
                with cols[j]:
                    row.append(
                        st.number_input(
                            f"{row_strategies[i]} vs {col_strategies[j]}",
                            value=float(default_A[i, j]),
                            key=f"A_{i}_{j}",
                            format="%.2f",
                        )
                    )
            user_vals.append(row)

        st.markdown(f"**{col_player} (column) payoffs**")
        rival_vals: List[List[float]] = []
        for i in range(n):
            cols = st.columns(n)
            row = []
            for j in range(n):
                with cols[j]:
                    row.append(
                        st.number_input(
                            f"{row_strategies[i]} vs {col_strategies[j]}",
                            value=float(default_B[i, j]),
                            key=f"B_{i}_{j}",
                            format="%.2f",
                        )
                    )
            rival_vals.append(row)

        st.markdown("**Stress testing**")
        enable_sensitivity = st.checkbox(
            "Run sensitivity analysis (Rival ±10%)",
            value=False,
            help="Stress-test how the Nash equilibrium shifts as the Rival becomes more or less aggressive.",
        )

        st.markdown("**Scenario manager**")
        save_history = st.button(
            "Capture current scenario",
            use_container_width=True,
            on_click=save_scenario,
        )
        history_items = st.session_state.get("history", [])
        if history_items:
            labels = [h["label"] for h in history_items]
            selected = st.selectbox("Saved scenarios", options=labels, index=len(labels) - 1)
            load_history = st.button("Load selected scenario", use_container_width=True)
        else:
            selected = None
            load_history = False

        solve_btn = st.button("Solve Nash equilibrium", use_container_width=True)

    # ——— Build matrices and solve ———
    # Load scenario into inputs (if requested)
    if "load_history" in locals() and load_history and selected is not None:
        match = next((h for h in st.session_state["history"] if h["label"] == selected), None)
        if match:
            st.session_state["row_player_name"] = match["row_player"]
            st.session_state["col_player_name"] = match["col_player"]
            for i, s in enumerate(match["row_strategies"]):
                st.session_state[f"rs_{i}"] = s
            for j, s in enumerate(match["col_strategies"]):
                st.session_state[f"cs_{j}"] = s
            for i in range(match["A"].shape[0]):
                for j in range(match["A"].shape[1]):
                    st.session_state[f"A_{i}_{j}"] = float(match["A"][i, j])
                    st.session_state[f"B_{i}_{j}"] = float(match["B"][i, j])
            st.rerun()

    A, B = build_payoff_matrices(user_vals, rival_vals)
    try:
        equilibria = solve_nash(A, B)
    except Exception as e:
        equilibria = []
        st.sidebar.error(f"Nash solver error: {e}")
    # Legacy 'history' kept only for load-selected-scenario; saving now handled by save_scenario()


    # Dominant strategies + Pareto (for first pure NE if present)
    dom_row, dom_col = find_strictly_dominant_strategies(A, B)
    pure_cells = get_pure_nash_cells(A, B)

    pareto_better_cells: List[Tuple[int, int]] = []
    if pure_cells:
        i_ne, j_ne = pure_cells[0]
        u_ne, r_ne = float(A[i_ne, j_ne]), float(B[i_ne, j_ne])
        n_rows, n_cols = A.shape
        for r in range(n_rows):
            for c in range(n_cols):
                if r == i_ne and c == j_ne:
                    continue
                if float(A[r, c]) > u_ne and float(B[r, c]) > r_ne:
                    pareto_better_cells.append((r, c))

    # Dashboard: payoff viz left (1.5), Nash results right (1) — no scroll to see answer
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown('<div class="payoff-group">', unsafe_allow_html=True)
        payoff_header, payoff_info = st.columns([0.9, 0.1])
        with payoff_header:
            st.subheader("Payoff matrices")
        with payoff_info:
            st.markdown(
                f'<div style="padding-top: 6px; text-align: right;">'
                f'<span title="Each cell shows ({row_player} payoff, {col_player} payoff). '
                f'Shading encodes the combined payoff ({row_player} + {col_player}).">'
                f'ℹ️</span></div>',
                unsafe_allow_html=True,
            )
        st.plotly_chart(
            plot_payoff_heatmap(
                A,
                B,
                nash_cells=pure_cells,
                row_labels=row_strategies,
                col_labels=col_strategies,
                row_player=row_player,
                col_player=col_player,
                pareto_better_cells=pareto_better_cells,
            ),
            use_container_width=True,
            config={"displayModeBar": True, "displaylogo": False},
        )
        with st.expander("View raw matrices"):
            st.dataframe(
                pd.DataFrame(np.round(A, 2), index=["User 1", "User 2"], columns=["Rival 1", "Rival 2"]),
                use_container_width=True,
            )
            st.dataframe(
                pd.DataFrame(np.round(B, 2), index=["User 1", "User 2"], columns=["Rival 1", "Rival 2"]),
                use_container_width=True,
            )

        st.markdown("### Strategic Decision Tree")
        if A.shape == (2, 2):
            tree = build_decision_tree_graph(
                row_player=row_player,
                col_player=col_player,
                row_strategies=row_strategies,
                col_strategies=col_strategies,
                A=A,
                B=B,
                highlight_cells=pure_cells,
            )
            st.graphviz_chart(tree, use_container_width=True)
        else:
            st.info("Strategic decision tree is currently available for 2×2 games only.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("Nash equilibrium")
        if dom_row is not None or dom_col is not None:
            dom_lines = []
            if dom_row is not None:
                dom_lines.append(f"- **{row_player} dominant strategy**: {row_strategies[dom_row]}")
            if dom_col is not None:
                dom_lines.append(f"- **{col_player} dominant strategy**: {col_strategies[dom_col]}")
            st.info("\n".join(dom_lines))

        if not equilibria:
            st.info("No Nash equilibrium found (in support enumeration).")
        else:
            for i, (sigma_u, sigma_r) in enumerate(equilibria):
                st.markdown(f"**Equilibrium {i + 1}**")
                df = build_equilibria_dataframe([(sigma_u, sigma_r)], row_player, col_player, row_strategies, col_strategies)
                df["Probability (%)"] = df["Probability (%)"].map(lambda v: f"{v:.2f}%")
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Player": st.column_config.TextColumn(width="medium"),
                    },
                )

                eu = float(np.dot(sigma_u, np.dot(A, sigma_r)))
                er = float(np.dot(sigma_r, np.dot(B.T, sigma_u)))

                # McKinsey-style KPI cards for expected payoffs (2 decimals)
                kpi_col1, kpi_col2 = st.columns(2)
                with kpi_col1:
                    st.metric(f"{row_player} expected payoff", f"{eu:.2f}")
                with kpi_col2:
                    st.metric(f"{col_player} expected payoff", f"{er:.2f}")

                # Strategic Confidence Score progress bar
                score = compute_confidence_score(A, B, sigma_u, sigma_r)
                conf_header, conf_info = st.columns([0.9, 0.1])
                with conf_header:
                    st.markdown("**Strategic Confidence Score**")
                with conf_info:
                    st.markdown(
                        '<div style="padding-top: 2px; text-align: right;">'
                        '<span title="Heuristic 0–100 based on deviation margins at equilibrium. Higher = less incentive to deviate.">'
                        'ℹ️</span></div>',
                        unsafe_allow_html=True,
                    )
                st.progress(score, text=f"Strategic Confidence Score: {score}/100")

            # Pareto efficiency (only meaningful to flag for pure equilibria)
            if pure_cells:
                pe = is_pareto_efficient(A, B, pure_cells[0])
                st.caption(f"Pareto Improvement Available: {'No' if pe else 'Yes'}")

                st.divider()

        # Download report (Markdown) and CSV exports
        ai_text = st.session_state.get("last_ai_analysis", "")
        brief_md = build_executive_brief_markdown(
            A,
            B,
            equilibria,
            ai_text,
            row_player=row_player,
            col_player=col_player,
            row_strategies=row_strategies,
            col_strategies=col_strategies,
        )
        st.download_button(
            "Download Executive Brief (Markdown)",
            data=brief_md,
            file_name="executive_brief.md",
            mime="text/markdown",
            use_container_width=True,
        )

        eq_csv = build_equilibria_dataframe(equilibria, row_player, col_player, row_strategies, col_strategies).to_csv(index=False)
        st.download_button(
            "Download Equilibria (CSV)",
            data=eq_csv,
            file_name="equilibria.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ——— Sensitivity analysis: Rival payoff aggressiveness ———
    if "enable_sensitivity" in locals() and enable_sensitivity:
        st.subheader("Sensitivity analysis — Rival aggressiveness")
        factors = [0.9, 0.95, 1.0, 1.05, 1.1]
        sens_df = build_sensitivity_dataframe(A, B, factors)

        if sens_df[["User: Prob(Strategy 1)", "Rival: Prob(Strategy 1)"]].isna().all().all():
            st.info(
                "Sensitivity analysis could not compute equilibria for these variations."
                " Try adjusting the payoff matrix to avoid degenerate games."
            )
        else:
            fig_sens = px.line(
                sens_df,
                x="Rival aggressiveness (%)",
                y=["User: Prob(Strategy 1)", "Rival: Prob(Strategy 1)"],
                markers=True,
            )
            fig_sens.update_traces(
                hovertemplate="Aggressiveness: %{x}%<br>%{y:.3f}<extra></extra>"
            )
            fig_sens.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(13,17,23,0)",
                plot_bgcolor="rgba(22,27,34,0.9)",
                font=dict(color="#e5e7eb", size=12),
                legend_title_text="Equilibrium behaviour",
                legend=dict(
                    font=dict(color="#f9fafb", size=12),
                ),
            )
            fig_sens.update_yaxes(range=[0, 1], title="Probability of Strategy 1")
            st.plotly_chart(
                fig_sens,
                use_container_width=True,
                config={"displayModeBar": True, "displaylogo": False},
            )
            st.download_button(
                "Download sensitivity results (CSV)",
                data=sens_df.to_csv(index=False),
                file_name="sensitivity_analysis.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ——— AI Strategic Insights (Gemini, when API key provided) ———
    # ——— AI Strategic Insights (Gemini, when Solve is clicked) ———
    # Start from any prior analysis stored in session (so it persists across tweaks)
    ai_analysis = (st.session_state.get("last_ai_analysis", "") or "").strip()

    # Only trigger a new Gemini call when Solve is clicked AND we have an API key AND equilibria exist
    if solve_btn and api_key and api_key.strip() and equilibria:
        with st.spinner("AI Strategic Insights — consulting analysis in progress…"):
            try:
                equilibria_text = format_equilibria_prompt(
                    equilibria,
                    A,
                    B,
                    row_player=row_player,
                    col_player=col_player,
                    row_strategies=row_strategies,
                    col_strategies=col_strategies,
                )
                ai_analysis = get_gemini_strategic_analysis(api_key, equilibria_text)
                st.session_state["last_ai_analysis"] = ai_analysis
            except Exception as e:
                err_msg = str(e)
                lower_msg = err_msg.lower()
                static_tip = get_static_strategic_tip(A, B, equilibria)

                if "api_key_invalid" in lower_msg or "api key not valid" in lower_msg:
                    st.error(
                        "**Invalid Gemini API key.** Get or create a key at "
                        "[Google AI Studio](https://aistudio.google.com/apikey). "
                        "Paste it in the sidebar or set the `GEMINI_API_KEY` environment variable."
                    )
                    ai_analysis = ""
                elif "429" in err_msg or "quota" in lower_msg or "resource_exhausted" in lower_msg:
                    st.warning(
                        "Strategic analysis is cooling down due to high demand."
                        " Please try again in 60 seconds."
                    )
                    ai_analysis = static_tip
                else:
                    st.error(f"Gemini API error: {e}")
                    ai_analysis = static_tip

                st.session_state["last_ai_analysis"] = ai_analysis

    st.markdown("### 🤖 AI Strategic Insights")
    if ai_analysis:
        # Render the model's markdown directly so it can use the required
        # '### Executive Summary' and '### Strategic Recommendation' headings,
        # and visually wrap it in a styled intelligence layer.
        st.markdown('<div class="ai-panel">', unsafe_allow_html=True)
        st.markdown(ai_analysis)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Enter a Gemini API key in the sidebar and click **Solve Nash equilibrium** to generate AI insights here.")


if __name__ == "__main__":
    main()

# Global scenario comparison table at bottom of page
if hasattr(st.session_state, "scenarios") and st.session_state.scenarios:
    st.markdown('---')
    st.header('Scenario Comparison')
    import pandas as pd
    st.dataframe(pd.DataFrame(st.session_state.scenarios), use_container_width=True)
