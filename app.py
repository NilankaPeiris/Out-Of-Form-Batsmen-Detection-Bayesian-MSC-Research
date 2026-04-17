import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from inference import CricketFormInferenceEngine


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ODI Batting Form Tracker",
    page_icon="🏏",
    layout="wide"
)


# -----------------------------------------------------------------------------
# Custom CSS for sporty UI
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #0b172a 0%, #10233f 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: #f5f7fa;
    }

    .hero-box {
        background: linear-gradient(135deg, #1b4332 0%, #081c15 100%);
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        margin-bottom: 1rem;
    }

    .input-box {
        background: rgba(255,255,255,0.05);
        padding: 1rem 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }

    .result-card {
        background: rgba(255,255,255,0.06);
        padding: 1rem 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 0.8rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }

    .status-good {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        padding: 1rem 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }

    .status-warning {
        background: linear-gradient(135deg, #e65100 0%, #ef6c00 100%);
        padding: 1rem 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }

    .status-danger {
        background: linear-gradient(135deg, #b71c1c 0%, #d32f2f 100%);
        padding: 1rem 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }

    .metric-label {
        font-size: 0.95rem;
        color: #d9e2ec;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #ffffff;
    }

    .state-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .caption-text {
        color: #d9e2ec;
        font-size: 0.92rem;
        margin-top: -0.3rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------------
POSTERIOR_PATH = "artifacts/posterior_summary.json"
FEATURE_INFO_PATH = "artifacts/feature_info.pkl"
PLAYERS_DATA_PATH = "data/final_cricket_data.xlsx"   # recommended reference file


# -----------------------------------------------------------------------------
# Load engine + player reference data
# -----------------------------------------------------------------------------
@st.cache_resource
def load_engine():
    with open(POSTERIOR_PATH, "r") as f:
        posterior_summary = json.load(f)

    posterior_summary["last_training_posterior"] = {
        int(k): np.array(v, dtype=float)
        for k, v in posterior_summary["last_training_posterior"].items()
    }

    with open(FEATURE_INFO_PATH, "rb") as f:
        feature_info = pickle.load(f)

    return CricketFormInferenceEngine(posterior_summary, feature_info)


@st.cache_data
def load_player_reference():
    df_players = pd.read_excel(PLAYERS_DATA_PATH)

    required_cols = ["PlayerID", "Player"]
    missing = [c for c in required_cols if c not in df_players.columns]
    if missing:
        raise ValueError(f"Player reference file is missing columns: {missing}")

    player_map = (
        df_players[["PlayerID", "Player"]]
        .drop_duplicates()
        .sort_values("Player")
        .reset_index(drop=True)
    )

    player_names = ["➕ New Player"] + player_map["Player"].tolist()
    name_to_id = dict(zip(player_map["Player"], player_map["PlayerID"]))

    return player_map, player_names, name_to_id


engine = load_engine()
player_map, player_names, name_to_id = load_player_reference()


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def to_percent(arr):
    return np.round(np.array(arr, dtype=float) * 100, 1)


def likelihood_to_percent(likelihoods):
    likes = np.array(likelihoods, dtype=float)
    likes = likes / likes.sum()
    return np.round(likes * 100, 1)


def most_likely_state_label(posterior):
    idx = int(np.argmax(posterior))
    labels = ["Out of Form", "In Normal Rhythm", "In Strong Form"]
    return labels[idx]


def get_status_text(posterior):
    oof, nf, hf = posterior

    if oof >= 0.70:
        return (
            "The batter currently appears to be clearly out of form. "
            "The latest innings profile strongly leans toward a struggling phase."
        ), "danger"

    if hf >= 0.60:
        return (
            "The batter appears to be in strong form right now. "
            "The latest evidence suggests he is batting with confidence and momentum."
        ), "good"

    if nf >= 0.50:
        return (
            "The batter appears to be in a fairly stable and normal batting phase. "
            "He does not currently show strong signs of being out of form."
        ), "good"

    if oof >= 0.45:
        return (
            "The batter shows some warning signs of a dip in form. "
            "He is not firmly classified as out of form yet, but the current pattern deserves attention."
        ), "warning"

    return (
        "The batter looks mixed between normal and high form, with no strong out-of-form signal at the moment."
    ), "good"


def get_starting_belief_text(player_id):
    if player_id is None:
        return "Starting point before this innings"
    return "Recent form background before this innings"


def get_pre_match_text():
    return "Form expectation before today's score"


def get_score_fit_text():
    return "How well today's score matches each form level"


def get_final_text():
    return "Current form assessment after today's innings"


def render_state_block(title, values_percent):
    state_names = ["Out of Form", "Normal Form", "Hot Form"]
    short_names = ["OOF", "NF", "HF"]

    st.markdown(f'<div class="result-card"><div class="state-title">{title}</div>', unsafe_allow_html=True)

    for name, short, val in zip(state_names, short_names, values_percent):
        st.markdown(f"**{name} ({short})** — {val}%")
        st.progress(float(val / 100))

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-box">
    <h1 style="margin-bottom:0.4rem;">🏏 ODI Batting Form Tracker</h1>
    <p style="margin:0;">
        Estimate a batter’s current form using match context, latest score, and the trained Bayesian form model.
    </p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
left, right = st.columns([1, 1.2], gap="large")

with left:
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    st.subheader("Match Inputs")

    selected_player = st.selectbox("Choose Batter", player_names)

    if selected_player != "➕ New Player":
        st.markdown(
            f'<div class="caption-text">Selected player: <b>{selected_player}</b></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="caption-text">Analyzing a new player with no previous history in the training data.</div>',
            unsafe_allow_html=True
        )

    location = st.selectbox("Match Location", ["Home", "Away", "Neutral"])
    opponent_rank = st.number_input("Opponent Ranking", min_value=1, max_value=20, value=5)
    rest_days = st.number_input("Rest Days Before Match", min_value=0, value=7)
    innings = st.number_input("Career Innings Number", min_value=1, value=1)
    score = st.number_input("Score in the Match", min_value=0, value=30)
    not_out = st.selectbox("Was the Batter Not Out?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    if selected_player == "➕ New Player":
        new_player_prior = st.selectbox(
            "Starting Assumption for a New Player",
            ["uniform", "normal_favoring"],
            format_func=lambda x: "Neutral / No Assumption" if x == "uniform" else "Assume Usually in Normal Form"
        )
    else:
        new_player_prior = "uniform"   # ignored for existing players

    run_button = st.button("Analyze Current Form", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


with right:
    st.subheader("Form Analysis")

    if run_button:
        if selected_player == "➕ New Player":
            player_id = None
        else:
            player_id = int(name_to_id[selected_player])

        result = engine.infer_one_innings(
            player_id=player_id,
            location=location,
            opponent_rank=opponent_rank,
            rest_days=rest_days,
            innings=innings,
            score=score,
            not_out=not_out,
            previous_posterior=None,
            use_training_tail=True if player_id is not None else False,
            new_player_prior=new_player_prior,
        )

        previous_pct = to_percent(result["previous_posterior"])
        prior_pct = to_percent(result["prior_today"])
        posterior_pct = to_percent(result["posterior_today"])
        likelihood_pct = likelihood_to_percent(result["likelihoods"])

        status_text, status_type = get_status_text(result["posterior_today"])
        dominant_label = most_likely_state_label(result["posterior_today"])

        status_class = {
            "good": "status-good",
            "warning": "status-warning",
            "danger": "status-danger"
        }[status_type]

        st.markdown(
            f"""
            <div class="{status_class}">
                <h3 style="margin-bottom:0.35rem;">Current Verdict: {dominant_label}</h3>
                <p style="margin:0;">{status_text}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="metric-label">Out of Form Chance</div>
                    <div class="metric-value">{posterior_pct[0]}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="metric-label">Normal Form Chance</div>
                    <div class="metric-value">{posterior_pct[1]}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="metric-label">Hot Form Chance</div>
                    <div class="metric-value">{posterior_pct[2]}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        render_state_block(
            title=get_starting_belief_text(player_id),
            values_percent=previous_pct
        )

        render_state_block(
            title=get_pre_match_text(),
            values_percent=prior_pct
        )

        render_state_block(
            title=get_score_fit_text(),
            values_percent=likelihood_pct
        )

        render_state_block(
            title=get_final_text(),
            values_percent=posterior_pct
        )

        st.markdown(
            """
            <div class="result-card">
                <b>How to read this:</b><br>
                1. <b>Starting point before this innings</b> shows the model’s initial belief about the batter’s recent form.<br>
                2. <b>Form expectation before today's score</b> shows how that belief shifts naturally over time.<br>
                3. <b>How well today's score matches each form level</b> shows whether the innings looks more like poor, normal, or strong form.<br>
                4. <b>Current form assessment</b> is the final view after combining past form and today’s score.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            """
            <div class="result-card">
                Choose a batter, enter the match details on the left, and click <b>Analyze Current Form</b> to see the form assessment.
            </div>
            """,
            unsafe_allow_html=True
        )