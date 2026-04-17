
import os
import json
import pickle
import tempfile
from pathlib import Path
from scipy.stats import t as student_t
import numpy as np
import pandas as pd
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
        self.STATE_NAMES = ["OOF", "NF", "HF"]

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
            "state_labels": self.STATE_NAMES,
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
