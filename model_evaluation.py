# =============================================================================
# MODEL EVALUATION / PERFORMANCE TESTS
# Compatible with the cricket_form training notebook/script
# =============================================================================

import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 0) Safety checks and artifact loading
# -----------------------------------------------------------------------------
OUTPUT_DIR = Path("/content/cricket_form_outputs")

if "posterior_summary" not in globals():
    posterior_json_path = OUTPUT_DIR / "posterior_summary.json"
    if not posterior_json_path.exists():
        raise FileNotFoundError(
            f"Could not find {posterior_json_path}. "
            "Run the training script first."
        )

    with open(posterior_json_path, "r") as f:
        posterior_summary_json = json.load(f)

    posterior_summary = {
        "alpha_state_mean": np.array(posterior_summary_json["alpha_state_mean"], dtype=float),
        "beta_mean": np.array(posterior_summary_json["beta_mean"], dtype=float),
        "u_mean": np.array(posterior_summary_json["u_mean"], dtype=float),
        "sigma_state_mean": np.array(posterior_summary_json["sigma_state_mean"], dtype=float),
        "nu_mean": float(posterior_summary_json["nu_mean"]),
        "init_state_mean": np.array(posterior_summary_json["init_state_mean"], dtype=float),
        "transition_matrix_mean": np.array(posterior_summary_json["transition_matrix_mean"], dtype=float),
        "last_training_posterior": {
            int(k): np.array(v, dtype=float)
            for k, v in posterior_summary_json["last_training_posterior"].items()
        },
    }

if "feature_info" not in globals():
    feature_pkl_path = OUTPUT_DIR / "feature_info.pkl"
    if not feature_pkl_path.exists():
        raise FileNotFoundError(
            f"Could not find {feature_pkl_path}. "
            "Please save feature_info after training using:\n"
            "with open(OUTPUT_DIR / 'feature_info.pkl', 'wb') as f:\n"
            "    pickle.dump(feature_info, f)"
        )

    with open(feature_pkl_path, "rb") as f:
        feature_info = pickle.load(f)

if "df" not in globals():
    filtered_csv_path = OUTPUT_DIR / "filtered_probabilities.csv"
    if not filtered_csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {filtered_csv_path}. "
            "Run the training script first."
        )
    df = pd.read_csv(filtered_csv_path)
    if "StartDate" in df.columns:
        df["StartDate"] = pd.to_datetime(df["StartDate"])

# Recreate engine if needed
if "engine" not in globals():
    engine = CricketFormInferenceEngine(posterior_summary, feature_info)

# Ensure filtered probabilities are present
required_prob_cols = ["P_OOF", "P_NF", "P_HF"]
missing_prob_cols = [c for c in required_prob_cols if c not in df.columns]
if missing_prob_cols:
    raise ValueError(
        f"Missing filtered probability columns: {missing_prob_cols}. "
        "Please ensure filtered_probabilities.csv was created by the training script."
    )

# Ensure required columns exist
required_eval_cols = [
    "PlayerID", "Player", "StartDate", "Innings", "Location",
    "OpponentRank", "RestDays", "FinalScore", "NotOut"
]
missing_eval_cols = [c for c in required_eval_cols if c not in df.columns]
if missing_eval_cols:
    raise ValueError(f"Dataset missing required evaluation columns: {missing_eval_cols}")

# Sort to ensure proper temporal order
df_eval = df.sort_values(["PlayerID", "StartDate", "Innings"]).reset_index(drop=True).copy()

STATE_NAMES = ["OOF", "NF", "HF"]

# -----------------------------------------------------------------------------
# 1) Sampler diagnostics
# -----------------------------------------------------------------------------
print("=" * 80)
print("1) SAMPLER / CONVERGENCE DIAGNOSTICS")
print("=" * 80)

if "fit" in globals():
    fit_summary = fit.summary()

    max_rhat = fit_summary["R_hat"].replace([np.inf, -np.inf], np.nan).max()
    num_bad_rhat = (fit_summary["R_hat"] > 1.01).sum()

    min_ess_bulk = fit_summary["ESS_bulk"].replace([np.inf, -np.inf], np.nan).min()
    min_ess_tail = fit_summary["ESS_tail"].replace([np.inf, -np.inf], np.nan).min()

    print(f"Max R-hat: {max_rhat:.4f}")
    print(f"Parameters with R-hat > 1.01: {int(num_bad_rhat)}")
    print(f"Minimum ESS_bulk: {min_ess_bulk:.1f}")
    print(f"Minimum ESS_tail: {min_ess_tail:.1f}")

    print("\nInterpretation:")
    if max_rhat <= 1.01:
        print("- Convergence looks good based on R-hat.")
    else:
        print("- Some parameters may not have mixed fully. Consider more warmup/chains or stronger priors.")

else:
    print("fit object not found in memory.")
    print("Skipping sampler diagnostics. If you want these diagnostics, run this cell in the same session as training.")


# -----------------------------------------------------------------------------
# 2) One-step-ahead predictive performance
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("2) ONE-STEP-AHEAD PREDICTIVE PERFORMANCE")
print("=" * 80)

records = []

for pid, g in df_eval.groupby("PlayerID", sort=True):
    g = g.sort_values(["StartDate", "Innings"]).copy()

    prev_posterior = engine.init_state.copy()

    for _, row in g.iterrows():
        # Feature vector for this innings
        x = engine.make_feature_vector(
            location=row["Location"],
            opponent_rank=float(row["OpponentRank"]),
            rest_days=float(row["RestDays"]),
            innings=float(row["Innings"])
        )

        # Markov prior
        prior_today = engine.predict_prior(prev_posterior)

        # Likelihood under each state
        likes = engine.score_likelihoods(
            score=float(row["FinalScore"]),
            not_out=int(row["NotOut"]),
            x=x,
            player_id=int(row["PlayerID"])
        )

        # One-step predictive density = mixture density under the PRIOR
        pred_density = float(np.sum(prior_today * likes))
        pred_density = max(pred_density, 1e-12)

        # Posterior update
        posterior_today = prior_today * likes
        posterior_today = posterior_today / posterior_today.sum()

        # Approximate predicted mean on log scale using PRIOR mixture
        # Note: this is an approximation based on state means, not exact Student-t expectation on raw runs.
        player_eff = engine._player_effect(int(row["PlayerID"]))
        mu_by_state = np.array([
            engine.alpha_state[k] + float(np.dot(x, engine.beta)) + player_eff
            for k in range(3)
        ], dtype=float)

        pred_log_runs = float(np.sum(prior_today * mu_by_state))
        pred_runs = max(0.0, np.expm1(pred_log_runs))

        actual_log_runs = np.log1p(float(row["FinalScore"]))

        records.append({
            "PlayerID": int(row["PlayerID"]),
            "Player": row["Player"],
            "StartDate": row["StartDate"],
            "Innings": int(row["Innings"]),
            "ActualRuns": float(row["FinalScore"]),
            "ActualLogRuns": actual_log_runs,
            "PredLogRuns": pred_log_runs,
            "PredRuns": pred_runs,
            "LogPredictiveDensity": float(np.log(pred_density)),
            "PredDensity": pred_density,
            "Prior_OOF": float(prior_today[0]),
            "Prior_NF": float(prior_today[1]),
            "Prior_HF": float(prior_today[2]),
            "Posterior_OOF": float(posterior_today[0]),
            "Posterior_NF": float(posterior_today[1]),
            "Posterior_HF": float(posterior_today[2]),
        })

        # Recursive filtering
        prev_posterior = posterior_today.copy()

eval_df = pd.DataFrame(records)

avg_log_pred_density = eval_df["LogPredictiveDensity"].mean()
neg_log_pred_density = -avg_log_pred_density
rmse_log = np.sqrt(np.mean((eval_df["ActualLogRuns"] - eval_df["PredLogRuns"]) ** 2))
rmse_runs = np.sqrt(np.mean((eval_df["ActualRuns"] - eval_df["PredRuns"]) ** 2))
mae_runs = np.mean(np.abs(eval_df["ActualRuns"] - eval_df["PredRuns"]))

print(f"Average log predictive density: {avg_log_pred_density:.4f}")
print(f"Negative log predictive density: {neg_log_pred_density:.4f}")
print(f"RMSE on log(1+runs) scale: {rmse_log:.4f}")
print(f"RMSE on runs scale: {rmse_runs:.4f}")
print(f"MAE on runs scale: {mae_runs:.4f}")

print("\nInterpretation:")
print("- Higher average log predictive density is better.")
print("- Lower negative log predictive density is better.")
print("- Lower RMSE / MAE means better predictive calibration on observed scores.")


# -----------------------------------------------------------------------------
# 3) State separation and interpretability
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("3) STATE SEPARATION / INTERPRETABILITY CHECKS")
print("=" * 80)

df_eval["MostLikelyState"] = df_eval[["P_OOF", "P_NF", "P_HF"]].values.argmax(axis=1)
df_eval["MostLikelyStateName"] = df_eval["MostLikelyState"].map({
    0: "OOF",
    1: "NF",
    2: "HF"
})

state_summary = (
    df_eval.groupby("MostLikelyStateName")["FinalScore"]
           .agg(["count", "mean", "median", "std", "min", "max"])
           .reindex(["OOF", "NF", "HF"])
)

print(state_summary.round(3))

# Monotonicity check: average score should ideally increase OOF < NF < HF
state_means = state_summary["mean"].to_dict()
monotonic_ok = (
    ("OOF" in state_means) and ("NF" in state_means) and ("HF" in state_means) and
    (state_means["OOF"] < state_means["NF"] < state_means["HF"])
)

print(f"\nMonotonic mean score ordering OOF < NF < HF: {monotonic_ok}")

print("\nInterpretation:")
print("- A good latent-state model should separate innings into progressively stronger score groups.")
print("- If OOF, NF, HF average scores are ordered correctly, that supports state interpretability.")


# -----------------------------------------------------------------------------
# 4) Practical usefulness of OOF detection
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("4) PRACTICAL OUT-OF-FORM SIGNAL USEFULNESS")
print("=" * 80)

# Heuristic low-score event definition
# You can change this threshold if your research prefers a different cricketing definition.
LOW_SCORE_THRESHOLD = 15

df_eval["LowScoreEvent"] = (df_eval["FinalScore"] <= LOW_SCORE_THRESHOLD).astype(int)
df_eval["OOF_Flag_0_70"] = (df_eval["P_OOF"] > 0.70).astype(int)

# Brier score using posterior OOF probability as a soft predictor of a low-score innings
brier_score = np.mean((df_eval["P_OOF"] - df_eval["LowScoreEvent"]) ** 2)

# Precision of the hard OOF flag for low-score innings
flagged = df_eval[df_eval["OOF_Flag_0_70"] == 1]
not_flagged = df_eval[df_eval["OOF_Flag_0_70"] == 0]

flag_precision = flagged["LowScoreEvent"].mean() if len(flagged) > 0 else np.nan
flag_rate = df_eval["OOF_Flag_0_70"].mean()
low_score_rate_flagged = flagged["LowScoreEvent"].mean() if len(flagged) > 0 else np.nan
low_score_rate_not_flagged = not_flagged["LowScoreEvent"].mean() if len(not_flagged) > 0 else np.nan

# Correlation between OOF probability and low-score event
corr = np.corrcoef(df_eval["P_OOF"], df_eval["LowScoreEvent"])[0, 1]

print(f"Low score threshold: <= {LOW_SCORE_THRESHOLD} runs")
print(f"Brier score (P_OOF vs low-score event): {brier_score:.4f}")
print(f"Correlation(P_OOF, low-score event): {corr:.4f}")
print(f"OOF flag rate (>0.70): {flag_rate:.4f}")
print(f"Low-score rate when OOF flagged: {low_score_rate_flagged:.4f}")
print(f"Low-score rate when NOT OOF flagged: {low_score_rate_not_flagged:.4f}")
print(f"OOF-flag precision for low-score event: {flag_precision:.4f}")

print("\nInterpretation:")
print("- Lower Brier score is better.")
print("- Positive correlation means higher OOF probability aligns with poor-score innings.")
print("- If low-score rate is much higher when OOF is flagged, the signal is practically useful.")


# -----------------------------------------------------------------------------
# 5) Transition matrix sanity check
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("5) TRANSITION MATRIX SANITY CHECK")
print("=" * 80)

trans_df = pd.DataFrame(
    posterior_summary["transition_matrix_mean"],
    index=STATE_NAMES,
    columns=STATE_NAMES
)

print(trans_df.round(4))

diag_mean = np.mean(np.diag(trans_df.values))
print(f"\nAverage diagonal persistence: {diag_mean:.4f}")

print("\nInterpretation:")
print("- Higher diagonal values mean stronger persistence in batting form.")
print("- Off-diagonal values show how easily form changes between states.")


# -----------------------------------------------------------------------------
# 6) Save evaluation outputs
# -----------------------------------------------------------------------------
eval_df.to_csv(OUTPUT_DIR / "evaluation_one_step_predictions.csv", index=False)
state_summary.to_csv(OUTPUT_DIR / "evaluation_state_summary.csv")

metrics_summary = {
    "avg_log_predictive_density": float(avg_log_pred_density),
    "negative_log_predictive_density": float(neg_log_pred_density),
    "rmse_log_scale": float(rmse_log),
    "rmse_runs_scale": float(rmse_runs),
    "mae_runs_scale": float(mae_runs),
    "brier_score_oof_vs_low_score": float(brier_score),
    "corr_p_oof_vs_low_score": float(corr),
    "oof_flag_rate": float(flag_rate),
    "low_score_rate_flagged": float(low_score_rate_flagged) if not np.isnan(low_score_rate_flagged) else None,
    "low_score_rate_not_flagged": float(low_score_rate_not_flagged) if not np.isnan(low_score_rate_not_flagged) else None,
    "oof_flag_precision": float(flag_precision) if not np.isnan(flag_precision) else None,
    "average_transition_diagonal": float(diag_mean),
}

with open(OUTPUT_DIR / "evaluation_metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

print("\nSaved evaluation outputs to:")
print(f"- {OUTPUT_DIR / 'evaluation_one_step_predictions.csv'}")
print(f"- {OUTPUT_DIR / 'evaluation_state_summary.csv'}")
print(f"- {OUTPUT_DIR / 'evaluation_metrics.json'}")