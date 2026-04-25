import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = Path(".")
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
OUTPUT_DIR = ROOT_DIR / "model_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POSTERIOR_FILE = ARTIFACTS_DIR / "posterior_summary.json"
FILTERED_PROBS_FILE = ARTIFACTS_DIR / "filtered_probabilities.csv"

FEATURE_INFO_FILE_1 = ARTIFACTS_DIR / "feature_info.pkl"
FEATURE_INFO_FILE_2 = ARTIFACTS_DIR / "feature_infor.pkl"
ENGINE_FILE = ARTIFACTS_DIR / "inference_engine.pkl"

CMDSTAN_CSV_DIR = ARTIFACTS_DIR / "cmdstan_csv"

STATE_NAMES = ["OOF", "NF", "HF"]
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ============================================================
# HELPERS
# ============================================================

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_pickle_if_exists(path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def inverse_log_score(x):
    return np.expm1(x)


def detect_probability_columns(df):
    oof_candidates = [c for c in df.columns if "oof" in c.lower()]
    nf_candidates = [
        c for c in df.columns
        if c.lower() in ["p_nf", "nf_probability", "p_normal", "p_nf_mean"]
        or "p_nf" in c.lower()
    ]
    hf_candidates = [c for c in df.columns if "hf" in c.lower()]

    if not oof_candidates or not nf_candidates or not hf_candidates:
        return None, None, None

    return oof_candidates[0], nf_candidates[0], hf_candidates[0]


def safe_to_markdown(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

print("Loading model artifacts...")

posterior_summary = read_json(POSTERIOR_FILE)

if not FILTERED_PROBS_FILE.exists():
    raise FileNotFoundError(f"Required file not found: {FILTERED_PROBS_FILE}")

filtered_probs = pd.read_csv(FILTERED_PROBS_FILE)

feature_info = read_pickle_if_exists(FEATURE_INFO_FILE_1)
if feature_info is None:
    feature_info = read_pickle_if_exists(FEATURE_INFO_FILE_2)

engine_artifact = read_pickle_if_exists(ENGINE_FILE)

print("Artifacts loaded successfully.")


# ============================================================
# 1. POSTERIOR PARAMETER EVALUATION
# ============================================================

print("Evaluating posterior parameters...")

alpha_state = np.array(posterior_summary["alpha_state_mean"], dtype=float)
beta_mean = np.array(posterior_summary["beta_mean"], dtype=float)
sigma_state = np.array(posterior_summary["sigma_state_mean"], dtype=float)
nu_mean = float(posterior_summary["nu_mean"])
init_state = np.array(posterior_summary["init_state_mean"], dtype=float)
transition_matrix = np.array(posterior_summary["transition_matrix_mean"], dtype=float)

posterior_eval = {
    "alpha_state_mean_log_scale": dict(zip(STATE_NAMES, alpha_state.tolist())),
    "alpha_state_expected_runs": dict(zip(STATE_NAMES, inverse_log_score(alpha_state).tolist())),
    "sigma_state_mean": dict(zip(STATE_NAMES, sigma_state.tolist())),
    "nu_mean": nu_mean,
    "init_state_mean": dict(zip(STATE_NAMES, init_state.tolist())),
    "transition_matrix_mean": transition_matrix.tolist(),
    "average_transition_diagonal": float(np.mean(np.diag(transition_matrix))),
}

if len(beta_mean) == 5:
    posterior_eval["beta_mean"] = {
        "OppStrengthZ": float(beta_mean[0]),
        "Home": float(beta_mean[1]),
        "Away": float(beta_mean[2]),
        "RestDaysZ": float(beta_mean[3]),
        "InningsZ": float(beta_mean[4]),
    }
else:
    posterior_eval["beta_mean"] = beta_mean.tolist()

save_json(posterior_eval, OUTPUT_DIR / "posterior_parameter_evaluation.json")

state_param_df = pd.DataFrame({
    "State": STATE_NAMES,
    "Alpha_LogScale": alpha_state,
    "Approx_Expected_Runs": inverse_log_score(alpha_state),
    "Sigma_State": sigma_state,
})

state_param_df.to_csv(OUTPUT_DIR / "state_parameter_summary.csv", index=False)

if len(beta_mean) == 5:
    beta_df = pd.DataFrame({
        "Covariate": ["OppStrengthZ", "Home", "Away", "RestDaysZ", "InningsZ"],
        "Posterior_Mean_Beta": beta_mean,
    })
else:
    beta_df = pd.DataFrame({
        "Parameter": [f"beta_{i+1}" for i in range(len(beta_mean))],
        "Posterior_Mean_Beta": beta_mean,
    })

beta_df.to_csv(OUTPUT_DIR / "covariate_effect_summary.csv", index=False)

transition_df = pd.DataFrame(
    transition_matrix,
    index=[f"From_{s}" for s in STATE_NAMES],
    columns=[f"To_{s}" for s in STATE_NAMES],
)

transition_df.to_csv(OUTPUT_DIR / "transition_matrix_summary.csv")


# ============================================================
# 2. FILTERED PROBABILITY EVALUATION
# ============================================================

print("Evaluating filtered probabilities...")

oof_col, nf_col, hf_col = detect_probability_columns(filtered_probs)

if oof_col and nf_col and hf_col:
    filtered_probs["prob_sum"] = (
        filtered_probs[oof_col]
        + filtered_probs[nf_col]
        + filtered_probs[hf_col]
    )

    filtered_probs["most_likely_state"] = filtered_probs[[oof_col, nf_col, hf_col]].idxmax(axis=1)
    filtered_probs["most_likely_state"] = filtered_probs["most_likely_state"].map({
        oof_col: "OOF",
        nf_col: "NF",
        hf_col: "HF",
    })

    filtered_probability_eval = {
        "mean_oof_probability": float(filtered_probs[oof_col].mean()),
        "mean_nf_probability": float(filtered_probs[nf_col].mean()),
        "mean_hf_probability": float(filtered_probs[hf_col].mean()),
        "median_oof_probability": float(filtered_probs[oof_col].median()),
        "median_nf_probability": float(filtered_probs[nf_col].median()),
        "median_hf_probability": float(filtered_probs[hf_col].median()),
        "probability_sum_mean": float(filtered_probs["prob_sum"].mean()),
        "probability_sum_min": float(filtered_probs["prob_sum"].min()),
        "probability_sum_max": float(filtered_probs["prob_sum"].max()),
        "most_likely_state_counts": filtered_probs["most_likely_state"].value_counts().to_dict(),
    }

    save_json(filtered_probability_eval, OUTPUT_DIR / "filtered_probability_evaluation.json")

    filtered_probs.to_csv(
        OUTPUT_DIR / "filtered_probabilities_with_state_summary.csv",
        index=False
    )

    for col, label in [
        (oof_col, "OOF Probability"),
        (nf_col, "NF Probability"),
        (hf_col, "HF Probability"),
    ]:
        plt.figure(figsize=(8, 5))
        plt.hist(filtered_probs[col], bins=30)
        plt.xlabel(label)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {label}")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"{label.lower().replace(' ', '_')}_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

else:
    warnings.warn(
        "Could not detect OOF, NF, and HF probability columns in filtered_probabilities.csv. "
        "Filtered probability evaluation skipped."
    )


# ============================================================
# 3. APPROXIMATE POSTERIOR PREDICTIVE CHECK
# ============================================================

print("Running approximate posterior predictive checks...")

n_sim = 5000

sim_states = np.random.choice(
    a=[0, 1, 2],
    size=n_sim,
    p=init_state / init_state.sum()
)

sim_log_scores = np.zeros(n_sim)

for i, state in enumerate(sim_states):
    mu = alpha_state[state]
    sigma = max(sigma_state[state], 1e-6)
    nu = max(nu_mean, 2.01)

    sim_log_scores[i] = mu + sigma * np.random.standard_t(df=nu)

sim_runs = np.maximum(0, inverse_log_score(sim_log_scores))

ppc_metrics = {
    "simulated_mean_runs": float(np.mean(sim_runs)),
    "simulated_median_runs": float(np.median(sim_runs)),
    "simulated_std_runs": float(np.std(sim_runs)),
    "simulated_duck_rate_score_less_than_1": float(np.mean(sim_runs < 1)),
    "simulated_low_score_rate_less_than_10": float(np.mean(sim_runs < 10)),
    "simulated_fifty_plus_rate": float(np.mean(sim_runs >= 50)),
    "simulated_century_plus_rate": float(np.mean(sim_runs >= 100)),
    "simulated_max_runs": float(np.max(sim_runs)),
    "note": (
        "This PPC is approximate because it uses posterior mean parameters only. "
        "A full Bayesian PPC should use draw-level posterior samples."
    ),
}

save_json(ppc_metrics, OUTPUT_DIR / "posterior_predictive_check_summary.json")

plt.figure(figsize=(8, 5))
plt.hist(sim_runs, bins=50)
plt.xlabel("Simulated Runs")
plt.ylabel("Frequency")
plt.title("Approximate Posterior Predictive Distribution of Runs")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "posterior_predictive_runs_distribution.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 4. TRANSITION DYNAMICS EVALUATION
# ============================================================

print("Evaluating transition dynamics...")

transition_rows = []

for i, from_state in enumerate(STATE_NAMES):
    row = {
        "From_State": from_state,
        "Stay_Probability": float(transition_matrix[i, i]),
        "Most_Likely_Next_State": STATE_NAMES[int(np.argmax(transition_matrix[i]))],
        "Most_Likely_Next_State_Probability": float(np.max(transition_matrix[i])),
    }

    for j, to_state in enumerate(STATE_NAMES):
        row[f"To_{to_state}"] = float(transition_matrix[i, j])

    transition_rows.append(row)

transition_dynamics_df = pd.DataFrame(transition_rows)
transition_dynamics_df.to_csv(
    OUTPUT_DIR / "transition_dynamics_evaluation.csv",
    index=False
)

plt.figure(figsize=(6, 5))
plt.imshow(transition_matrix)
plt.xticks(range(3), STATE_NAMES)
plt.yticks(range(3), STATE_NAMES)
plt.xlabel("To State")
plt.ylabel("From State")
plt.title("Posterior Mean Transition Matrix")

for i in range(3):
    for j in range(3):
        plt.text(j, i, f"{transition_matrix[i, j]:.2f}", ha="center", va="center")

plt.colorbar(label="Transition Probability")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "transition_matrix_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 5. MCMC DIAGNOSTICS AND TRACE PLOTS FROM CMDSTAN CSV
# ============================================================

print("Checking MCMC diagnostics from CmdStan CSV files...")

diagnostic_report = {
    "cmdstan_csv_found": False,
    "rhat_available": False,
    "ess_available": False,
    "trace_plots_available": False,
    "message": "",
}

cmdstan_csv_files = sorted(CMDSTAN_CSV_DIR.glob("*.csv"))

if len(cmdstan_csv_files) == 0:
    diagnostic_report["message"] = (
        "CmdStan CSV files were not found in artifacts/cmdstan_csv. "
        "R-hat, ESS, and trace plots cannot be generated. "
        "Update the training script to save CmdStan outputs using output_dir='artifacts/cmdstan_csv'."
    )
    print(diagnostic_report["message"])

else:
    try:
        import arviz as az
        from cmdstanpy import from_csv

        diagnostic_report["cmdstan_csv_found"] = True

        trace_dir = OUTPUT_DIR / "trace_plots"
        trace_dir.mkdir(parents=True, exist_ok=True)

        print(f"Found {len(cmdstan_csv_files)} CmdStan CSV files.")

        fit = from_csv([str(f) for f in cmdstan_csv_files])

        idata = az.from_cmdstanpy(posterior=fit)

        diagnostics_summary = az.summary(idata)
        diagnostics_summary.to_csv(OUTPUT_DIR / "mcmc_rhat_ess_summary.csv")

        diagnostic_report["rhat_available"] = True
        diagnostic_report["ess_available"] = True

        selected_rows = diagnostics_summary[
            diagnostics_summary.index.to_series().str.contains(
                "alpha_state|beta|sigma_state|nu|init_state|transition_matrix|trans",
                regex=True
            )
        ]

        if not selected_rows.empty:
            selected_rows.to_csv(OUTPUT_DIR / "mcmc_key_parameter_diagnostics.csv")

        key_params = [
            "alpha_state",
            "beta",
            "sigma_state",
            "nu",
            "init_state",
            "transition_matrix",
        ]

        successful_trace_plots = []

        for param in key_params:
            try:
                az.plot_trace(idata, var_names=[param])
                plt.tight_layout()

                plot_path = trace_dir / f"trace_{param}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

                successful_trace_plots.append(str(plot_path))
                print(f"Saved trace plot: {plot_path}")

            except Exception as e:
                print(f"Could not generate trace plot for {param}: {e}")

        diagnostic_report["trace_plots_available"] = len(successful_trace_plots) > 0
        diagnostic_report["trace_plot_files"] = successful_trace_plots
        diagnostic_report["message"] = (
            "MCMC diagnostics generated successfully. "
            "R-hat, ESS summaries, and trace plots were saved."
        )

    except Exception as e:
        diagnostic_report["message"] = (
            f"Failed to generate MCMC diagnostics from CmdStan CSV files: {str(e)}"
        )
        print(diagnostic_report["message"])

save_json(diagnostic_report, OUTPUT_DIR / "mcmc_diagnostic_report.json")


# ============================================================
# 6. FINAL MARKDOWN REPORT
# ============================================================

print("Writing final model evaluation report...")

report_path = OUTPUT_DIR / "model_evaluation_report.md"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Model Evaluation Report\n\n")

    f.write("## 1. Evaluation Context\n\n")
    f.write(
        "The true batting form state is latent and cannot be directly observed. "
        "Therefore, the model is not evaluated as a normal supervised classification model. "
        "Metrics such as accuracy, precision, recall, and F1-score are not used as primary "
        "evaluation metrics for the hidden states. Instead, the model is evaluated using "
        "Bayesian model checking methods such as posterior parameter behaviour, convergence "
        "diagnostics, effective sample size, trace plots, transition behaviour, and posterior "
        "predictive checking.\n\n"
    )

    f.write("## 2. Posterior Parameter Summary\n\n")
    f.write(safe_to_markdown(state_param_df))
    f.write("\n\n")

    f.write("## 3. Covariate Effect Summary\n\n")
    f.write(safe_to_markdown(beta_df))
    f.write("\n\n")

    f.write("## 4. Transition Matrix\n\n")
    try:
        f.write(transition_df.to_markdown())
    except Exception:
        f.write(transition_df.to_string())
    f.write("\n\n")
    f.write(
        f"The average diagonal transition probability is "
        f"{posterior_eval['average_transition_diagonal']:.3f}. "
        "This represents the average persistence of the latent form states.\n\n"
    )

    f.write("## 5. Posterior Predictive Check\n\n")
    for key, value in ppc_metrics.items():
        f.write(f"- {key}: {value}\n")
    f.write("\n")

    f.write("## 6. MCMC Diagnostics\n\n")
    f.write(diagnostic_report["message"])
    f.write("\n\n")

    if diagnostic_report.get("trace_plots_available"):
        f.write("Trace plots were saved in:\n\n")
        f.write("```text\n")
        f.write(str(OUTPUT_DIR / "trace_plots"))
        f.write("\n```\n\n")

    f.write("## 7. Limitations of the Evaluation\n\n")
    f.write(
        "- The real form state is not observed, so direct supervised classification metrics "
        "cannot be used as the main evaluation method.\n"
        "- The approximate posterior predictive check uses posterior mean parameters only. "
        "A stronger version should simulate from full posterior draws.\n"
        "- MCMC diagnostics require saved CmdStan CSV files from the training process.\n"
        "- Future work should include full posterior predictive checks, prior sensitivity checks, "
        "and temporal validation using future innings.\n"
    )

print(f"Model evaluation complete. Outputs saved to: {OUTPUT_DIR.resolve()}")