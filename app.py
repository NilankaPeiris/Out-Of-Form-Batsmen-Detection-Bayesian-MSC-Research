import json
import pickle
import numpy as np
import streamlit as st

from inference import CricketFormInferenceEngine


@st.cache_resource
def load_engine():
    with open("artifacts/posterior_summary.json", "r") as f:
        posterior_summary = json.load(f)

    # convert lists back where needed
    posterior_summary["last_training_posterior"] = {
        int(k): np.array(v, dtype=float)
        for k, v in posterior_summary["last_training_posterior"].items()
    }

    with open("artifacts/feature_info.pkl", "rb") as f:
        feature_info = pickle.load(f)

    return CricketFormInferenceEngine(posterior_summary, feature_info)


engine = load_engine()

st.title("ODI Batting Form Inference")

player_id_text = st.text_input("Player ID (leave blank for new player)", "")
location = st.selectbox("Location", ["Home", "Away", "Neutral"])
opponent_rank = st.number_input("Opponent Rank", min_value=1, max_value=20, value=5)
rest_days = st.number_input("Rest Days", min_value=0, value=7)
innings = st.number_input("Innings Number", min_value=1, value=1)
score = st.number_input("Score", min_value=0, value=30)
not_out = st.selectbox("Not Out", [0, 1])

new_player_prior = st.selectbox("New Player Prior", ["uniform", "normal_favoring"])

if st.button("Run Inference"):
    player_id = int(player_id_text) if player_id_text.strip() else None

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

    st.subheader("Results")
    st.write("Previous Posterior:", np.round(result["previous_posterior"], 3))
    st.write("Today's Prior:", np.round(result["prior_today"], 3))
    st.write("Likelihoods:", np.round(result["likelihoods"], 6))
    st.write("Today's Posterior:", np.round(result["posterior_today"], 3))
    st.write("Out of Form Flag (> 0.70):", result["out_of_form_flag_0_70"])