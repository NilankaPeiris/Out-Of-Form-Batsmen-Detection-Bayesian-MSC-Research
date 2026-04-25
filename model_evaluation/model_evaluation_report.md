# Model Evaluation Report

## 1. Evaluation Context

The true batting form state is latent and cannot be directly observed. Therefore, the model is not evaluated as a normal supervised classification model. Metrics such as accuracy, precision, recall, and F1-score are not used as primary evaluation metrics for the hidden states. Instead, the model is evaluated using Bayesian model checking methods such as posterior parameter behaviour, convergence diagnostics, effective sample size, trace plots, transition behaviour, and posterior predictive checking.

## 2. Posterior Parameter Summary

| State   |   Alpha_LogScale |   Approx_Expected_Runs |   Sigma_State |
|:--------|-----------------:|-----------------------:|--------------:|
| OOF     |          1.69049 |                4.42216 |      1.07346  |
| NF      |          3.29152 |               25.8836  |      0.768616 |
| HF      |          3.90757 |               48.7778  |      0.621795 |

## 3. Covariate Effect Summary

| Covariate    |   Posterior_Mean_Beta |
|:-------------|----------------------:|
| OppStrengthZ |            -0.150632  |
| Home         |             0.0936581 |
| Away         |            -0.194701  |
| RestDaysZ    |            -0.0446707 |
| InningsZ     |             0.0681888 |

## 4. Transition Matrix

|          |   To_OOF |    To_NF |    To_HF |
|:---------|---------:|---------:|---------:|
| From_OOF | 0.526425 | 0.199012 | 0.274563 |
| From_NF  | 0.222786 | 0.580523 | 0.196692 |
| From_HF  | 0.413601 | 0.1606   | 0.4258   |

The average diagonal transition probability is 0.511. This represents the average persistence of the latent form states.

## 5. Posterior Predictive Check

- simulated_mean_runs: 33.62776876167538
- simulated_median_runs: 22.97152626508028
- simulated_std_runs: 37.590038552135205
- simulated_duck_rate_score_less_than_1: 0.0422
- simulated_low_score_rate_less_than_10: 0.2476
- simulated_fifty_plus_rate: 0.21
- simulated_century_plus_rate: 0.0506
- simulated_max_runs: 589.0616116557613
- note: This PPC is approximate because it uses posterior mean parameters only. A full Bayesian PPC should use draw-level posterior samples.

## 6. MCMC Diagnostics

MCMC diagnostics generated successfully. R-hat, ESS summaries, and trace plots were saved.

Trace plots were saved in:

```text
model_evaluation/trace_plots
```

## 7. Limitations of the Evaluation

- The real form state is not observed, so direct supervised classification metrics cannot be used as the main evaluation method.
- The approximate posterior predictive check uses posterior mean parameters only. A stronger version should simulate from full posterior draws.
- MCMC diagnostics require saved CmdStan CSV files from the training process.
- Future work should include full posterior predictive checks, prior sensitivity checks, and temporal validation using future innings.
