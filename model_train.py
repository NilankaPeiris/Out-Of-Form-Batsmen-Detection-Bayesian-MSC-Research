
"""
Colab-ready implementation of the ODI batting form model.

Why this implementation?
------------------------
Your methodology describes:
1) 3 hidden form states: Out-of-Form, Normal, Hot
2) a first-order HMM transition matrix
3) a Tobit-Student-t observation model
4) Bayesian filtering for inference
5) training with Bayesian priors

This script follows that structure closely:
- Stan + NUTS is used for training the Bayesian HMM.
- The hidden states are marginalized with the forward algorithm, so we can use NUTS.
- The observation model is Student-t for OUT innings and right-censored Student-t survival for NOT OUT innings.
- After training, simple Python helper functions make inference easy.

Expected input file columns
---------------------------
Player, Location, Opposition, OppositionTeam, NotOut, FinalScore, StartDate,
PlayerID, RestDays, Innings, DateYYYYMMDD, OpponentRank, RankDateUsed
"""

# =============================================================================
# 0) Colab setup
# =============================================================================
# In Google Colab, run this once:
# !pip -q install cmdstanpy openpyxl arviz
#
# Then install CmdStan once (can take a few minutes the first time):
# from cmdstanpy import install_cmdstan
# install_cmdstan()

# =============================================================================
# 1) Imports
# =============================================================================
import os
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from cmdstanpy import CmdStanModel
import arviz as az


# =============================================================================
# 2) User settings
# =============================================================================
DATA_PATH = "/content/final_cricket_data (4).xlsx"   # change if needed
SEED = 42

# Stan/NUTS settings
CHAINS = 4
WARMUP = 1000
SAMPLES = 1000
ADAPT_DELTA = 0.95
MAX_TREEDEPTH = 15

# Output settings
OUTPUT_DIR = Path("/content/cricket_form_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 3) Data loading and preprocessing
# =============================================================================
def load_and_prepare_data(path: str):
    """
    Load the uploaded Excel file and create the model-ready design matrix.

    The methodology uses:
    - log(1 + runs) as the response
    - latent states for form
    - covariates affecting the score distribution mean

    We use the following covariates:
    1. OppStrengthZ : stronger opponent => expected score slightly lower
    2. Home         : small positive home effect
    3. Away         : small negative away effect (neutral is baseline)
    4. RestDaysZ    : slightly positive effect of more rest
    5. InningsZ     : small positive experience effect

    Returns
    -------
    df : cleaned/sorted dataframe
    stan_data : dictionary ready for Stan
    feature_info : metadata needed later for inference
    """
    df = pd.read_excel(path).copy()

    required_cols = [
        "Player", "Location", "OppositionTeam", "NotOut", "FinalScore", "StartDate",
        "PlayerID", "RestDays", "Innings", "OpponentRank"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Sort in temporal order within player: essential for any HMM/state-space model.
    df["StartDate"] = pd.to_datetime(df["StartDate"])
    df = df.sort_values(["PlayerID", "StartDate", "Innings"]).reset_index(drop=True)

    # Response transformation from the methodology
    df["y_log_runs"] = np.log1p(df["FinalScore"].astype(float))

    # Covariate engineering
    # Opponent rank: smaller rank => stronger opponent.
    # We negate it so larger OppStrength means stronger opponent.
    df["OppStrength"] = -df["OpponentRank"].astype(float)

    # One-hot location: neutral is the reference/baseline level
    df["Home"] = (df["Location"].str.strip().str.lower() == "home").astype(int)
    df["Away"] = (df["Location"].str.strip().str.lower() == "away").astype(int)

    # Standardize continuous predictors for stable HMC sampling and interpretable priors
    std_cols = ["OppStrength", "RestDays", "Innings"]
    means = {}
    sds = {}
    for col in std_cols:
        means[col] = df[col].mean()
        sds[col] = df[col].std(ddof=0)
        if sds[col] == 0:
            sds[col] = 1.0
        df[f"{col}Z"] = (df[col] - means[col]) / sds[col]

    feature_cols = ["OppStrengthZ", "Home", "Away", "RestDaysZ", "InningsZ"]
    X = df[feature_cols].astype(float).values

    # Player index for random effects
    player_codes, unique_players = pd.factorize(df["PlayerID"], sort=True)
    df["PlayerIndex"] = player_codes + 1  # Stan is 1-based
    J = len(unique_players)

    # Sequence structure: one sequence per player
    seq_lengths = (
        df.groupby("PlayerID", sort=True)
          .size()
          .astype(int)
          .tolist()
    )

    # Prior means from your methodology:
    # OOF ~ 16 runs, NF ~ 36 runs, HF ~ 66 runs, on the log(1+runs) scale
    state_mean_runs = np.array([16.0, 36.0, 66.0], dtype=float)
    state_mean_log = np.log1p(state_mean_runs)

    # Transition prior: sticky diagonal, but still allows switching.
    # Stronger diagonal means we expect form to persist across innings.
    # Rows correspond to previous state OOF/NF/HF; columns to current state OOF/NF/HF.
    alpha_trans = np.array([
        [8.0, 2.0, 1.0],   # OOF tends to stay OOF more than jump to HF
        [2.0, 8.0, 2.0],   # NF is the most stable central state
        [1.0, 2.0, 8.0],   # HF tends to stay HF more than collapse immediately
    ])

    # Initial state prior: most innings are expected to be Normal, but not overwhelmingly so
    alpha_init = np.array([2.0, 6.0, 2.0])

    stan_data = {
        "N": int(len(df)),
        "J": int(J),
        "K": 3,
        "P": int(X.shape[1]),
        "y": df["y_log_runs"].astype(float).values,
        "notout": df["NotOut"].astype(int).values,
        "player": df["PlayerIndex"].astype(int).values,
        "X": X,
        "S": int(len(seq_lengths)),
        "seq_lengths": seq_lengths,
        "alpha_state_prior_mean": state_mean_log.tolist(),
        "alpha_state_prior_sd": [0.40, 0.40, 0.40],  # weakly informative but meaningful
        # Prior means for beta align with cricket intuition:
        # OppStrengthZ negative, Home positive, Away negative, Rest positive, Innings positive
        "beta_prior_mean": [-0.15, 0.10, -0.10, 0.08, 0.06],
        "beta_prior_sd": [0.25, 0.20, 0.20, 0.15, 0.15],
        "alpha_init": alpha_init.tolist(),
        "alpha_trans": alpha_trans.tolist(),
    }

    feature_info = {
        "feature_cols": feature_cols,
        "continuous_cols": std_cols,
        "means": means,
        "sds": sds,
        "player_id_to_index": {int(pid): i + 1 for i, pid in enumerate(unique_players)},
        "index_to_player_id": {i + 1: int(pid) for i, pid in enumerate(unique_players)},
        "player_names": df.groupby("PlayerID")["Player"].first().to_dict(),
        "reference_location": "Neutral",
    }

    return df, stan_data, feature_info


# =============================================================================
# 4) Stan model
# =============================================================================
STAN_CODE = r"""
functions {
  real emission_logp(real y, int notout, real mu, real sigma, real nu) {
    if (notout == 1) {
      // NOT OUT = right-censored score
      return student_t_lccdf(y | nu, mu, sigma);
    } else {
      // OUT = fully observed score
      return student_t_lpdf(y | nu, mu, sigma);
    }
  }
}
data {
  int<lower=1> N;                     // total innings
  int<lower=1> J;                     // total players
  int<lower=1> K;                     // total states = 3
  int<lower=1> P;                     // total covariates
  vector[N] y;                        // log(1 + runs)
  array[N] int<lower=0, upper=1> notout;
  array[N] int<lower=1, upper=J> player;
  matrix[N, P] X;

  int<lower=1> S;                     // number of player sequences
  array[S] int<lower=1> seq_lengths;  // sequence length per player

  vector[K] alpha_state_prior_mean;
  vector<lower=0>[K] alpha_state_prior_sd;

  vector[P] beta_prior_mean;
  vector<lower=0>[P] beta_prior_sd;

  vector<lower=0>[K] alpha_init;
  array[K] vector<lower=0>[K] alpha_trans;
}
parameters {
  ordered[K] alpha_state;             // OOF < NF < HF for identifiability
  vector[P] beta;                     // shared covariate effects
  vector[J] u_raw;                    // player random effects (raw)
  real<lower=0> sigma_u;              // sd of player random effects
  vector<lower=0>[K] sigma_state;     // state-specific scale
  real<lower=2> nu;                   // Student-t degrees of freedom
  simplex[K] init_state;              // initial state probabilities
  array[K] simplex[K] trans;          // transition matrix
}
transformed parameters {
  vector[J] u = sigma_u * u_raw;
}
model {
  // Priors
  alpha_state ~ normal(alpha_state_prior_mean, alpha_state_prior_sd);
  beta ~ normal(beta_prior_mean, beta_prior_sd);

  u_raw ~ normal(0, 1);
  sigma_u ~ normal(0, 0.5);           // half-normal after lower bound
  sigma_state ~ normal(0, 0.6);       // half-normal after lower bound
  nu ~ gamma(2, 0.2);                 // centers on moderate heavy tails

  init_state ~ dirichlet(alpha_init);
  for (k in 1:K) {
    trans[k] ~ dirichlet(alpha_trans[k]);
  }

  // Forward algorithm over each player's innings
  {
    int pos = 1;
    for (s in 1:S) {
      int T = seq_lengths[s];
      vector[K] log_alpha_prev;
      vector[K] log_alpha_curr;

      // First innings of the sequence
      for (k in 1:K) {
        real mu = alpha_state[k] + X[pos] * beta + u[player[pos]];
        log_alpha_prev[k] = log(init_state[k]) + emission_logp(y[pos], notout[pos], mu, sigma_state[k], nu);
      }

      // Remaining innings
      if (T > 1) {
        for (t in 2:T) {
          int n = pos + t - 1;
          for (k in 1:K) {
            vector[K] tmp;
            real mu = alpha_state[k] + X[n] * beta + u[player[n]];
            for (j in 1:K) {
              tmp[j] = log_alpha_prev[j] + log(trans[j][k]);
            }
            log_alpha_curr[k] = emission_logp(y[n], notout[n], mu, sigma_state[k], nu) + log_sum_exp(tmp);
          }
          log_alpha_prev = log_alpha_curr;
        }
      }

      target += log_sum_exp(log_alpha_prev);
      pos += T;
    }
  }
}
generated quantities {
  matrix[N, K] filtered_prob;
  matrix[K, K] transition_matrix;

  for (j in 1:K) {
    for (k in 1:K) {
      transition_matrix[j, k] = trans[j][k];
    }
  }

  // Compute filtered P(state_t | data_1:t)
  {
    int pos = 1;
    for (s in 1:S) {
      int T = seq_lengths[s];
      vector[K] log_alpha_prev;
      vector[K] log_alpha_curr;

      // First innings in sequence
      for (k in 1:K) {
        real mu = alpha_state[k] + X[pos] * beta + u[player[pos]];
        log_alpha_prev[k] = log(init_state[k]) + emission_logp(y[pos], notout[pos], mu, sigma_state[k], nu);
      }
      {
        real z = log_sum_exp(log_alpha_prev);
        for (k in 1:K) {
          filtered_prob[pos, k] = exp(log_alpha_prev[k] - z);
        }
      }

      // Remaining innings
      if (T > 1) {
        for (t in 2:T) {
          int n = pos + t - 1;
          for (k in 1:K) {
            vector[K] tmp;
            real mu = alpha_state[k] + X[n] * beta + u[player[n]];
            for (j in 1:K) {
              tmp[j] = log_alpha_prev[j] + log(trans[j][k]);
            }
            log_alpha_curr[k] = emission_logp(y[n], notout[n], mu, sigma_state[k], nu) + log_sum_exp(tmp);
          }
          log_alpha_prev = log_alpha_curr;
          {
            real z = log_sum_exp(log_alpha_prev);
            for (k in 1:K) {
              filtered_prob[n, k] = exp(log_alpha_prev[k] - z);
            }
          }
        }
      }

      pos += T;
    }
  }
}
"""


def build_stan_model(stan_code: str, model_name: str = "cricket_form_hmm"):
    """
    Write Stan code to a temporary file and compile it.
    """
    stan_dir = OUTPUT_DIR / "stan"
    stan_dir.mkdir(parents=True, exist_ok=True)
    stan_file = stan_dir / f"{model_name}.stan"
    stan_file.write_text(stan_code)
    model = CmdStanModel(stan_file=str(stan_file))
    return model


# =============================================================================
# 5) Training
# =============================================================================
def fit_model(stan_model: CmdStanModel, stan_data: dict, seed: int = 42):
    """
    Fit the Bayesian HMM using NUTS.

    Notes:
    - This is the heaviest step.
    - For a quick first run, 2 chains x (500 warmup + 500 sample) is reasonable in Colab.
    - For a final dissertation run, increase chains and samples.
    """
    fit = stan_model.sample(
        data=stan_data,
        chains=CHAINS,
        parallel_chains=CHAINS,
        iter_warmup=WARMUP,
        iter_sampling=SAMPLES,
        adapt_delta=ADAPT_DELTA,
        max_treedepth=MAX_TREEDEPTH,
        seed=seed,
        show_progress=True,
    )
    return fit


# =============================================================================
# 6) Posterior summarization
# =============================================================================
STATE_NAMES = ["OOF", "NF", "HF"]


def summarize_fit(fit, df, feature_info):
    """
    Extract posterior means and filtered probabilities for easy inference later.
    """
    summary = {}

    summary["alpha_state_mean"] = fit.stan_variable("alpha_state").mean(axis=0)  # (K,)
    summary["beta_mean"] = fit.stan_variable("beta").mean(axis=0)                # (P,)
    summary["u_mean"] = fit.stan_variable("u").mean(axis=0)                      # (J,)
    summary["sigma_state_mean"] = fit.stan_variable("sigma_state").mean(axis=0)  # (K,)
    summary["nu_mean"] = float(fit.stan_variable("nu").mean())
    summary["init_state_mean"] = fit.stan_variable("init_state").mean(axis=0)    # (K,)
    summary["transition_matrix_mean"] = fit.stan_variable("transition_matrix").mean(axis=0)  # (K, K)

    # Filtered probabilities for each innings from generated quantities
    filtered_draws = fit.stan_variable("filtered_prob")   # draws x N x K
    filtered_mean = filtered_draws.mean(axis=0)           # N x K

    filtered_df = df.copy()
    filtered_df[["P_OOF", "P_NF", "P_HF"]] = filtered_mean

    # Keep the last training posterior for each player. This is useful for existing-player inference.
    last_posterior = (
        filtered_df.sort_values(["PlayerID", "StartDate", "Innings"])
                  .groupby("PlayerID")[["P_OOF", "P_NF", "P_HF"]]
                  .tail(1)
    )
    last_player_ids = (
        filtered_df.sort_values(["PlayerID", "StartDate", "Innings"])
                  .groupby("PlayerID")
                  .tail(1)["PlayerID"]
                  .tolist()
    )

    summary["last_training_posterior"] = {
        int(pid): probs.values.astype(float)
        for pid, (_, probs) in zip(last_player_ids, last_posterior.iterrows())
    }
    summary["filtered_df"] = filtered_df
    return summary


# =============================================================================
# 7) Inference helper engine
# =============================================================================
class CricketFormInferenceEngine:
    """
    Lightweight prediction/filtering engine built from posterior mean parameters.

    What this engine does
    ---------------------
    For a new innings:
    1. Create today's PRIOR state belief:
       previous_posterior @ transition_matrix
    2. Evaluate the SCORE LIKELIHOOD under each state:
       - Student-t pdf if OUT
       - Student-t survival probability if NOT OUT
    3. Combine prior x likelihood and normalize:
       posterior ∝ prior * likelihood

    This is exactly the predict-then-update logic described in your methodology.
    """

    def __init__(self, posterior_summary, feature_info):
        self.alpha_state = np.asarray(posterior_summary["alpha_state_mean"], dtype=float)      # (3,)
        self.beta = np.asarray(posterior_summary["beta_mean"], dtype=float)                    # (P,)
        self.u = np.asarray(posterior_summary["u_mean"], dtype=float)                          # (J,)
        self.sigma_state = np.asarray(posterior_summary["sigma_state_mean"], dtype=float)      # (3,)
        self.nu = float(posterior_summary["nu_mean"])
        self.init_state = np.asarray(posterior_summary["init_state_mean"], dtype=float)        # (3,)
        self.trans = np.asarray(posterior_summary["transition_matrix_mean"], dtype=float)      # (3,3)
        self.last_training_posterior = posterior_summary["last_training_posterior"]
        self.feature_info = feature_info

    # -------------------------
    # Feature handling
    # -------------------------
    def _standardize(self, value, col_name):
        mean = self.feature_info["means"][col_name]
        sd = self.feature_info["sds"][col_name]
        return (value - mean) / sd

    def make_feature_vector(self, location, opponent_rank, rest_days, innings):
        """
        Build one row of covariates in the same order used in training.

        Parameters
        ----------
        location : str
            Home / Away / Neutral
        opponent_rank : numeric
            ICC ODI team rank at match time
        rest_days : numeric
            Full days between previous and current match
        innings : numeric
            Running innings index for that player

        Returns
        -------
        x : np.ndarray, shape (P,)
        """
        loc = str(location).strip().lower()
        home = 1 if loc == "home" else 0
        away = 1 if loc == "away" else 0

        opp_strength = -float(opponent_rank)
        x = np.array([
            self._standardize(opp_strength, "OppStrength"),
            home,
            away,
            self._standardize(float(rest_days), "RestDays"),
            self._standardize(float(innings), "Innings"),
        ], dtype=float)
        return x

    # -------------------------
    # Prior handling
    # -------------------------
    def get_starting_posterior(self, player_id=None, use_training_tail=True, new_player_prior="uniform"):
        """
        Get the starting state belief before predicting the next unseen innings.

        Existing player:
            - by default uses the player's last posterior from training
        New player:
            - uniform prior [1/3, 1/3, 1/3]
            - or cricket-informed prior centered on Normal Form
        """
        if player_id is not None and use_training_tail and player_id in self.last_training_posterior:
            return np.array(self.last_training_posterior[player_id], dtype=float)

        if new_player_prior == "uniform":
            return np.array([1/3, 1/3, 1/3], dtype=float)
        elif new_player_prior == "normal_favoring":
            return np.array([0.20, 0.60, 0.20], dtype=float)
        else:
            raise ValueError("new_player_prior must be 'uniform' or 'normal_favoring'")

    def predict_prior(self, previous_posterior):
        """
        Markov step:
            prior_today = previous_posterior @ transition_matrix
        """
        previous_posterior = np.asarray(previous_posterior, dtype=float)
        prior_today = previous_posterior @ self.trans
        prior_today = prior_today / prior_today.sum()
        return prior_today

    # -------------------------
    # Likelihood handling
    # -------------------------
    def _player_effect(self, player_id=None):
        """
        Existing players get their posterior-mean random effect.
        New players get population mean = 0.
        """
        if player_id is None:
            return 0.0
        idx = self.feature_info["player_id_to_index"].get(int(player_id))
        if idx is None:
            return 0.0
        return float(self.u[idx - 1])

    def score_likelihoods(self, score, not_out, x, player_id=None):
        """
        Compute the score likelihood under each hidden state.

        OUT innings:
            use Student-t pdf at observed y = log(1 + score)

        NOT OUT innings:
            use right-tail probability P(Y >= y)
            because the true innings could have continued further
        """
        y = np.log1p(float(score))
        player_eff = self._player_effect(player_id)

        likes = np.zeros(3, dtype=float)
        for k in range(3):
            mu = self.alpha_state[k] + float(np.dot(x, self.beta)) + player_eff
            sigma = self.sigma_state[k]

            if int(not_out) == 1:
                # survival probability = 1 - CDF
                likes[k] = 1.0 - student_t.cdf(y, df=self.nu, loc=mu, scale=sigma)
            else:
                likes[k] = student_t.pdf(y, df=self.nu, loc=mu, scale=sigma)

        # Numerical floor so we never divide by zero later
        likes = np.clip(likes, 1e-12, None)
        return likes

    # -------------------------
    # Full one-innings inference
    # -------------------------
    def infer_one_innings(
        self,
        location,
        opponent_rank,
        rest_days,
        innings,
        score,
        not_out,
        player_id=None,
        previous_posterior=None,
        use_training_tail=True,
        new_player_prior="uniform",
    ):
        """
        Run one full prediction/update cycle.

        Returns
        -------
        dict containing:
            - feature_vector
            - prior_today
            - likelihoods
            - posterior_today
            - out_of_form_flag
        """
        x = self.make_feature_vector(location, opponent_rank, rest_days, innings)

        if previous_posterior is None:
            previous_posterior = self.get_starting_posterior(
                player_id=player_id,
                use_training_tail=use_training_tail,
                new_player_prior=new_player_prior,
            )

        prior_today = self.predict_prior(previous_posterior)
        likelihoods = self.score_likelihoods(score=score, not_out=not_out, x=x, player_id=player_id)

        unnormalized = prior_today * likelihoods
        posterior_today = unnormalized / unnormalized.sum()

        return {
            "feature_vector": x,
            "previous_posterior": previous_posterior,
            "prior_today": prior_today,
            "likelihoods": likelihoods,
            "posterior_today": posterior_today,
            "state_labels": STATE_NAMES,
            "out_of_form_flag_0_70": bool(posterior_today[0] > 0.70),
        }

    def infer_sequence(
        self,
        innings_rows,
        player_id=None,
        use_training_tail=True,
        new_player_prior="uniform",
    ):
        """
        Run recursive filtering for a list of innings dictionaries.

        innings_rows: list of dicts, each containing
            location, opponent_rank, rest_days, innings, score, not_out

        The posterior from innings t becomes the previous posterior for innings t+1.
        """
        results = []
        prev = None
        for row in innings_rows:
            res = self.infer_one_innings(
                location=row["location"],
                opponent_rank=row["opponent_rank"],
                rest_days=row["rest_days"],
                innings=row["innings"],
                score=row["score"],
                not_out=row["not_out"],
                player_id=player_id,
                previous_posterior=prev,
                use_training_tail=use_training_tail,
                new_player_prior=new_player_prior,
            )
            results.append(res)
            prev = res["posterior_today"]
        return results


# =============================================================================
# 8) Main workflow
# =============================================================================
def main():
    # -------------------------
    # Load data
    # -------------------------
    print("Loading and preparing data...")
    df, stan_data, feature_info = load_and_prepare_data(DATA_PATH)
    print(f"Rows: {len(df):,}")
    print(f"Players: {df['PlayerID'].nunique():,}")

    # -------------------------
    # Build/compile Stan model
    # -------------------------
    print("Compiling Stan model...")
    model = build_stan_model(STAN_CODE)

    # -------------------------
    # Train
    # -------------------------
    print("Training model with NUTS...")
    fit = fit_model(model, stan_data, seed=SEED)

    # Save raw fit CSV outputs path info
    print(fit.diagnose())

    # -------------------------
    # Summarize posterior
    # -------------------------
    print("Summarizing posterior...")
    posterior_summary = summarize_fit(fit, df, feature_info)

    # Save filtered probabilities to Excel/CSV for inspection
    posterior_summary["filtered_df"].to_csv(OUTPUT_DIR / "filtered_probabilities.csv", index=False)

    # Save summary JSON (portable)
    json_ready = {
        "alpha_state_mean": posterior_summary["alpha_state_mean"].tolist(),
        "beta_mean": posterior_summary["beta_mean"].tolist(),
        "u_mean": posterior_summary["u_mean"].tolist(),
        "sigma_state_mean": posterior_summary["sigma_state_mean"].tolist(),
        "nu_mean": posterior_summary["nu_mean"],
        "init_state_mean": posterior_summary["init_state_mean"].tolist(),
        "transition_matrix_mean": posterior_summary["transition_matrix_mean"].tolist(),
        "last_training_posterior": {
            str(k): v.tolist() for k, v in posterior_summary["last_training_posterior"].items()
        },
    }
    with open(OUTPUT_DIR / "posterior_summary.json", "w") as f:
        json.dump(json_ready, f, indent=2)

    # Build inference engine and save it
    engine = CricketFormInferenceEngine(posterior_summary, feature_info)
    with open(OUTPUT_DIR / "inference_engine.pkl", "wb") as f:
        pickle.dump({"posterior_summary": json_ready, "feature_info": feature_info}, f)

    # -------------------------
    # Print a readable model summary
    # -------------------------
    print("\nPosterior mean state intercepts on log(1+runs) scale:")
    for s, val in zip(STATE_NAMES, posterior_summary["alpha_state_mean"]):
        print(f"  {s}: {val:.3f}")

    print("\nApproximate expected runs implied by state intercept only:")
    for s, val in zip(STATE_NAMES, posterior_summary["alpha_state_mean"]):
        print(f"  {s}: {np.expm1(val):.1f}")

    print("\nPosterior mean beta coefficients:")
    for name, val in zip(feature_info["feature_cols"], posterior_summary["beta_mean"]):
        print(f"  {name}: {val:.3f}")

    print("\nPosterior mean transition matrix:")
    trans_df = pd.DataFrame(
        posterior_summary["transition_matrix_mean"],
        index=STATE_NAMES,
        columns=STATE_NAMES,
    )
    print(trans_df.round(3))

    # -------------------------
    # Example inference
    # -------------------------
    # Existing player example: use the player's last training posterior automatically
    example_existing_player_id = int(df["PlayerID"].iloc[0])

    existing_result = engine.infer_one_innings(
        player_id=example_existing_player_id,
        location="Home",
        opponent_rank=5,
        rest_days=7,
        innings=35,
        score=42,
        not_out=0,
        previous_posterior=None,       # use last training posterior for this player
        use_training_tail=True,
    )

    print("\nExample inference for an EXISTING player")
    print("Previous posterior:", np.round(existing_result["previous_posterior"], 3))
    print("Today's prior:      ", np.round(existing_result["prior_today"], 3))
    print("Likelihoods:        ", np.round(existing_result["likelihoods"], 6))
    print("Today's posterior:  ", np.round(existing_result["posterior_today"], 3))
    print("Flag OOF > 0.70?:   ", existing_result["out_of_form_flag_0_70"])

    # New player example: use uniform prior and player effect = 0
    new_result = engine.infer_one_innings(
        player_id=999999,               # unseen player
        location="Away",
        opponent_rank=3,
        rest_days=4,
        innings=1,
        score=18,
        not_out=0,
        previous_posterior=None,
        use_training_tail=False,
        new_player_prior="uniform",
    )

    print("\nExample inference for a NEW player")
    print("Previous posterior:", np.round(new_result["previous_posterior"], 3))
    print("Today's prior:      ", np.round(new_result["prior_today"], 3))
    print("Likelihoods:        ", np.round(new_result["likelihoods"], 6))
    print("Today's posterior:  ", np.round(new_result["posterior_today"], 3))
    print("Flag OOF > 0.70?:   ", new_result["out_of_form_flag_0_70"])

    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
