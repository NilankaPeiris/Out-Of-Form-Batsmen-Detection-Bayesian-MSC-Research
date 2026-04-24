"""
Exploratory Data Analysis for:
A Bayesian State-Space Approach for Detecting Out-of-Form Batters in ODI Cricket

This script is designed to:
1. Read the Excel dataset provided by the researcher.
2. Perform univariate and bivariate EDA for all variables relevant to the Bayesian model.
3. Create publication-ready figures suitable for use in Overleaf.
4. Save summary tables and figures into an organised output folder.
5. Include comments that explain how each analysis can be written up in the thesis.

Expected Bayesian-model-relevant variables in the dataset:
- Player / PlayerID          -> player-specific effect
- Innings                    -> time index for state evolution
- FinalScore                 -> observed batting score
- NotOut                     -> censoring indicator for Tobit-style treatment
- RestDays                   -> recovery / spacing covariate
- Location                   -> contextual categorical covariate
- OpponentRank               -> opposition strength covariate
- StartDate                  -> chronological ordering and trend inspection

The script is defensive about column names and should run without changes on the
uploaded file "final_cricket_data (4).xlsx".
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# -----------------------------
# 1. USER CONFIGURATION
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR.parent/"data"/"final_cricket_data.xlsx"
OUTPUT_DIR = BASE_DIR/"output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# Save both PDF and PNG so the same figure can be used in Overleaf (vector PDF)
# and also previewed quickly outside LaTeX (PNG).
SAVE_PDF = True
SAVE_PNG = True
DPI = 320


# -----------------------------
# 2. PLOTTING STYLE
# -----------------------------
# The goal is to avoid plain default matplotlib sketches.
# These settings create cleaner, presentation-quality charts.
sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams.update(
    {
        "figure.figsize": (12, 7),
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 12,
        "pdf.fonttype": 42,  # better text rendering in vector outputs
        "ps.fonttype": 42,
    }
)


# -----------------------------
# 3. HELPER FUNCTIONS
# -----------------------------
def ensure_directories() -> None:
    """Create output folders."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)



def save_figure(fig: plt.Figure, filename_stem: str) -> None:
    """Save a figure in one or both formats."""
    if SAVE_PNG:
        fig.savefig(FIG_DIR / f"{filename_stem}.png", bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(FIG_DIR / f"{filename_stem}.pdf", bbox_inches="tight")
    plt.close(fig)



def percent_fmt(x, pos):
    return f"{x:.0f}%"



def load_data(path: Path) -> pd.DataFrame:
    """Load the cricket data from Excel."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_excel(path)

    # Standardise column names lightly while preserving the original names.
    df.columns = [c.strip() for c in df.columns]

    required_columns = [
        "Player",
        "Location",
        "NotOut",
        "FinalScore",
        "StartDate",
        "PlayerID",
        "RestDays",
        "Innings",
        "OpponentRank",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            "The following required columns are missing from the dataset: "
            + ", ".join(missing)
        )

    df["StartDate"] = pd.to_datetime(df["StartDate"], errors="coerce")

    # Core derived fields for EDA and Bayesian interpretation.
    df["ScoreLog"] = np.log1p(df["FinalScore"])
    df["OutStatus"] = np.where(df["NotOut"].eq(1), "Not Out", "Out")
    df["Duck"] = np.where(df["FinalScore"].eq(0), 1, 0)
    df["FiftyPlus"] = np.where(df["FinalScore"] >= 50, 1, 0)
    df["CenturyPlus"] = np.where(df["FinalScore"] >= 100, 1, 0)

    # Make opponent-rank bands for easier interpretation in EDA.
    df["OpponentRankBand"] = pd.cut(
        df["OpponentRank"],
        bins=[0, 3, 6, 10, 20],
        labels=["Top 3", "Rank 4-6", "Rank 7-10", "Rank 11+"],
        include_lowest=True,
        right=True,
    )

    # Rest-day bands allow practical EDA while keeping the raw numeric variable for modelling.
    df["RestDaysBand"] = pd.cut(
        df["RestDays"],
        bins=[-1, 3, 7, 14, 30, np.inf],
        labels=["0-3", "4-7", "8-14", "15-30", "31+"],
        include_lowest=True,
    )

    # Order the data for player-wise time analysis.
    df = df.sort_values(["Player", "StartDate", "Innings"]).reset_index(drop=True)

    return df



def save_basic_tables(df: pd.DataFrame) -> None:
    """Save descriptive summary tables for thesis/report usage."""
    missing_tbl = df.isna().sum().rename("MissingCount").reset_index()
    missing_tbl.columns = ["Variable", "MissingCount"]
    missing_tbl.to_csv(TABLE_DIR / "missing_values_summary.csv", index=False)

    numeric_cols = ["FinalScore", "ScoreLog", "RestDays", "Innings", "OpponentRank"]
    numeric_summary = df[numeric_cols].describe().T
    numeric_summary["skew"] = df[numeric_cols].skew()
    numeric_summary["kurtosis"] = df[numeric_cols].kurtosis()
    numeric_summary.to_csv(TABLE_DIR / "numeric_summary.csv")

    categorical_cols = ["Player", "Location", "OutStatus", "OpponentRankBand", "RestDaysBand"]
    for col in categorical_cols:
        out = (
            df[col]
            .value_counts(dropna=False)
            .rename_axis(col)
            .reset_index(name="Count")
        )
        out["Percent"] = 100 * out["Count"] / len(df)
        out.to_csv(TABLE_DIR / f"frequency_{col}.csv", index=False)

    player_summary = (
        df.groupby("Player")
        .agg(
            InningsCount=("FinalScore", "size"),
            MeanScore=("FinalScore", "mean"),
            MedianScore=("FinalScore", "median"),
            StdScore=("FinalScore", "std"),
            NotOutRate=("NotOut", "mean"),
            FiftyPlusRate=("FiftyPlus", "mean"),
            CenturyPlusRate=("CenturyPlus", "mean"),
            DuckRate=("Duck", "mean"),
            MeanRestDays=("RestDays", "mean"),
            MeanOpponentRank=("OpponentRank", "mean"),
        )
        .sort_values("MeanScore", ascending=False)
    )
    player_summary.to_csv(TABLE_DIR / "player_level_summary.csv")

    # Correlation table among numeric variables relevant to modelling.
    corr = df[["FinalScore", "ScoreLog", "RestDays", "Innings", "OpponentRank", "NotOut"]].corr(method="spearman")
    corr.to_csv(TABLE_DIR / "spearman_correlation_matrix.csv")


# -----------------------------
# 4. PLOTS - UNIVARIATE
# -----------------------------
def plot_missingness(df: pd.DataFrame) -> None:
    """Visual confirmation that the dataset is complete or where gaps exist."""
    miss = df.isna().mean().sort_values(ascending=False) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=miss.index, y=miss.values, color="steelblue", ax=ax)
    ax.set_title("Missing Data Percentage by Variable")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Missing data (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(percent_fmt))
    ax.tick_params(axis="x", rotation=45)

    # Thesis note:
    # Describe whether missingness is negligible or substantial.
    # If there are no missing values, state that the data required no imputation,
    # which simplifies the Bayesian modelling stage.
    save_figure(fig, "01_missing_data_percentage")



def plot_score_distribution(df: pd.DataFrame) -> None:
    """Distribution of the raw score, which typically shows zero inflation and right skewness."""
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(df["FinalScore"], bins=30, kde=True, color="steelblue", ax=ax)
    ax.set_title("Distribution of Raw ODI Scores")
    ax.set_xlabel("Final score")
    ax.set_ylabel("Frequency")

    # Thesis note:
    # Describe the concentration of low scores, the long right tail, and the presence
    # of extreme innings. This provides a direct rationale for not relying on a simple
    # Gaussian assumption in the Bayesian observation model.
    save_figure(fig, "02_score_distribution_raw")



def plot_score_distribution_transformed(df: pd.DataFrame) -> None:
    """Distribution of log(1 + score), matching the transformation logic in the methodology."""
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(df["ScoreLog"], bins=30, kde=True, color="steelblue", ax=ax)
    ax.set_title("Distribution of Log-Transformed Score: log(1 + score)")
    ax.set_xlabel("log(1 + FinalScore)")
    ax.set_ylabel("Frequency")

    # Thesis note:
    # Explain that the transformation compresses the high-score tail, stabilises
    # variation, and makes the score process more suitable for a flexible state-space
    # Bayesian model.
    save_figure(fig, "03_score_distribution_log_transformed")



def plot_box_score(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(x=df["FinalScore"], color="lightsteelblue", ax=ax)
    ax.set_title("Boxplot of Raw Scores")
    ax.set_xlabel("Final score")

    # Thesis note:
    # Use this figure to emphasise the heavy-tailed nature of batting scores and the
    # existence of unusually large innings. This motivates the t-distribution style of
    # modelling rather than a thin-tailed normal model.
    save_figure(fig, "04_score_boxplot")



def plot_categorical_frequencies(df: pd.DataFrame) -> None:
    plot_specs = [
        ("Location", "Location Distribution", "05_location_distribution"),
        ("OutStatus", "Out vs Not Out Distribution", "06_outstatus_distribution"),
        ("OpponentRankBand", "Opponent Rank Band Distribution", "07_opponent_rank_band_distribution"),
        ("RestDaysBand", "Rest-Day Band Distribution", "08_rest_days_band_distribution"),
    ]

    for col, title, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(10, 6))
        vc = df[col].value_counts(dropna=False)
        sns.barplot(x=vc.index.astype(str), y=vc.values, color="steelblue", ax=ax)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=20)

        # Thesis note:
        # Use these charts to describe the sample composition. For example, mention if
        # one location dominates the data or if not-out innings are relatively rare.
        # Such imbalances matter when interpreting subsequent model coefficients.
        save_figure(fig, filename)



def plot_player_innings_distribution(df: pd.DataFrame) -> None:
    innings_per_player = df.groupby("Player").size().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(x=innings_per_player.index, y=innings_per_player.values, color="steelblue", ax=ax)
    ax.set_title("Number of Innings Available per Player")
    ax.set_xlabel("Player")
    ax.set_ylabel("Number of innings")
    ax.tick_params(axis="x", rotation=60)

    # Thesis note:
    # Explain the variation in player-specific sample sizes. This is important because
    # players with fewer innings will naturally have more uncertain latent-state and
    # player-effect estimates in the Bayesian model.
    save_figure(fig, "09_player_innings_counts")



def plot_rest_days_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(df["RestDays"], bins=30, kde=True, color="steelblue", ax=ax)
    ax.set_title("Distribution of Rest Days Between Innings")
    ax.set_xlabel("Rest days")
    ax.set_ylabel("Frequency")

    # Thesis note:
    # Explain whether players usually return after short gaps or long gaps. This helps
    # position RestDays as a plausible contextual predictor of performance fluctuations.
    save_figure(fig, "10_rest_days_distribution")



def plot_opposition_rank_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(df["OpponentRank"], bins=17, discrete=True, color="steelblue", ax=ax)
    ax.set_title("Distribution of Opposition Rank Faced")
    ax.set_xlabel("Opponent rank")
    ax.set_ylabel("Frequency")

    # Thesis note:
    # Describe whether the sample is concentrated against stronger or weaker opponents.
    # This matters because opposition quality is one of the main covariates entering the
    # observation model.
    save_figure(fig, "11_opposition_rank_distribution")


# -----------------------------
# 5. PLOTS - BIVARIATE / MULTIVARIATE
# -----------------------------
def plot_score_by_location(df: pd.DataFrame) -> None:
    order = df.groupby("Location")["FinalScore"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.boxplot(data=df, x="Location", y="FinalScore", order=order, color="lightsteelblue", ax=ax)
    sns.stripplot(data=df, x="Location", y="FinalScore", order=order, color="black", alpha=0.25, size=3, ax=ax)
    ax.set_title("Score Distribution by Match Location")
    ax.set_xlabel("Location")
    ax.set_ylabel("Final score")

    # Thesis note:
    # Compare the medians and spread across home, away, and neutral conditions.
    # This gives an initial indication of whether location should contribute meaningfully
    # to the mean structure of the Bayesian observation model.
    save_figure(fig, "12_score_by_location")



def plot_score_by_outstatus(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.violinplot(data=df, x="OutStatus", y="FinalScore", inner="box", cut=0, palette="pastel", ax=ax)
    ax.set_title("Score Distribution by Dismissal Status")
    ax.set_xlabel("Innings outcome")
    ax.set_ylabel("Final score")

    # Thesis note:
    # Explain that not-out innings tend to be structurally different because the innings
    # was not fully observed. This supports the use of censoring-aware likelihoods in the model.
    save_figure(fig, "13_score_by_outstatus")



def plot_score_by_opponent_rank_band(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(data=df, x="OpponentRankBand", y="FinalScore", color="lightsteelblue", ax=ax)
    ax.set_title("Score Distribution by Opposition Strength Band")
    ax.set_xlabel("Opposition rank band")
    ax.set_ylabel("Final score")

    # Thesis note:
    # Discuss whether scores appear lower against top-ranked sides and higher against
    # weaker-ranked sides. Even if the pattern is not perfectly monotonic, the figure
    # helps justify the inclusion of OppositionRank as a covariate.
    save_figure(fig, "14_score_by_opponent_rank_band")



def plot_score_by_rest_days_band(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(data=df, x="RestDaysBand", y="FinalScore", color="lightsteelblue", ax=ax)
    ax.set_title("Score Distribution by Rest-Day Band")
    ax.set_xlabel("Rest-day band")
    ax.set_ylabel("Final score")

    # Thesis note:
    # Use this to discuss whether there is visible evidence that recovery time is related
    # to performance. This is only exploratory and not causal, but it prepares the reader
    # for the regression component in the Bayesian framework.
    save_figure(fig, "15_score_by_rest_days_band")



def plot_player_mean_scores(df: pd.DataFrame) -> None:
    player_mean = (
        df.groupby("Player")["FinalScore"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=player_mean, x="Player", y="FinalScore", color="steelblue", ax=ax)
    ax.set_title("Average Score by Player")
    ax.set_xlabel("Player")
    ax.set_ylabel("Mean final score")
    ax.tick_params(axis="x", rotation=60)

    # Thesis note:
    # Explain that players have clearly different baseline scoring levels, which justifies
    # including a player-specific effect in the model rather than assuming all batters are identical.
    save_figure(fig, "16_player_mean_scores")



def plot_player_notout_rates(df: pd.DataFrame) -> None:
    player_notout = (
        df.groupby("Player")["NotOut"].mean().sort_values(ascending=False).mul(100).reset_index()
    )

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=player_notout, x="Player", y="NotOut", color="steelblue", ax=ax)
    ax.set_title("Not-Out Rate by Player")
    ax.set_xlabel("Player")
    ax.set_ylabel("Not-out rate (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(percent_fmt))
    ax.tick_params(axis="x", rotation=60)

    # Thesis note:
    # Mention that censoring is not evenly distributed across players, which further supports
    # explicit handling of not-out innings instead of treating them the same as completed dismissals.
    save_figure(fig, "17_player_notout_rates")



def plot_time_trend_all_players(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(data=df, x="StartDate", y="ScoreLog", hue="Player", palette="tab20", estimator=None, lw=1.8, alpha=0.65, legend=False, ax=ax)
    ax.set_title("Player-wise Log-Score Trajectories Over Time")
    ax.set_xlabel("Match date")
    ax.set_ylabel("log(1 + FinalScore)")

    # Thesis note:
    # This figure is useful to argue that batting performance is dynamic rather than static.
    # The visible ups and downs across time provide intuitive support for latent state evolution.
    save_figure(fig, "18_time_trend_logscore_all_players")



def plot_player_small_multiples(df: pd.DataFrame, top_n: int = 9) -> None:
    """Small multiples for the players with the most innings."""
    players = df["Player"].value_counts().head(top_n).index.tolist()
    subset = df[df["Player"].isin(players)].copy()

    g = sns.FacetGrid(subset, col="Player", col_wrap=3, sharey=False, height=3.6)
    g.map_dataframe(sns.lineplot, x="Innings", y="ScoreLog", marker="o", color="steelblue")
    g.set_axis_labels("Innings index", "log(1 + FinalScore)")
    g.set_titles("{col_name}")
    g.fig.subplots_adjust(top=0.90)
    g.fig.suptitle("Within-Player Performance Paths (Top 9 Players by Number of Innings)", fontsize=18, fontweight="bold")

    # Thesis note:
    # Use this to show that within-player trajectories fluctuate substantially. That kind of
    # within-subject variation is exactly what a hidden-state approach aims to capture.
    if SAVE_PNG:
        g.savefig(FIG_DIR / "19_small_multiples_player_trajectories.png", bbox_inches="tight")
    if SAVE_PDF:
        g.savefig(FIG_DIR / "19_small_multiples_player_trajectories.pdf", bbox_inches="tight")
    plt.close(g.fig)



def plot_score_vs_covariates(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.regplot(data=df, x="RestDays", y="ScoreLog", color="steelblue", scatter_kws={"alpha": 0.45, "s": 35}, line_kws={"linewidth": 2}, lowess=True, ax=axes[0])
    axes[0].set_title("Log-Score vs Rest Days")
    axes[0].set_xlabel("Rest days")
    axes[0].set_ylabel("log(1 + FinalScore)")

    sns.regplot(data=df, x="OpponentRank", y="ScoreLog", color="steelblue", scatter_kws={"alpha": 0.45, "s": 35}, line_kws={"linewidth": 2}, lowess=True, ax=axes[1])
    axes[1].set_title("Log-Score vs Opposition Rank")
    axes[1].set_xlabel("Opponent rank")
    axes[1].set_ylabel("log(1 + FinalScore)")

    # Thesis note:
    # These are exploratory smooth trends, not final inferential evidence. Use them to describe
    # whether the relationships appear flat, monotonic, or nonlinear, which may inform the final
    # mean specification in the Bayesian model.
    save_figure(fig, "20_logscore_vs_numeric_covariates")



def plot_numeric_correlation_heatmap(df: pd.DataFrame) -> None:
    cols = ["FinalScore", "ScoreLog", "RestDays", "Innings", "OpponentRank", "NotOut"]
    corr = df[cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, ax=ax)
    ax.set_title("Spearman Correlation Matrix for Numeric Model Variables")

    # Thesis note:
    # Explain that this matrix provides a first look at dependence structure among numeric variables.
    # Correlations should be described cautiously because the main model is dynamic and hierarchical,
    # but the figure helps justify variable inclusion and detects obvious collinearity concerns.
    save_figure(fig, "21_spearman_correlation_heatmap")



def plot_notout_probability_by_context(df: pd.DataFrame) -> None:
    rates = (
        df.groupby(["Location", "OpponentRankBand"])["NotOut"]
        .mean()
        .mul(100)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=rates, x="OpponentRankBand", y="NotOut", hue="Location", palette="deep", ax=ax)
    ax.set_title("Not-Out Rate Across Match Contexts")
    ax.set_xlabel("Opposition rank band")
    ax.set_ylabel("Not-out rate (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(percent_fmt))

    # Thesis note:
    # This figure helps show whether censoring may also vary by context, not just by player.
    # That is useful when motivating why not-out innings should be handled carefully in the likelihood.
    save_figure(fig, "22_notout_rate_by_context")



def plot_scoring_milestones(df: pd.DataFrame) -> None:
    milestones = pd.DataFrame(
        {
            "Milestone": ["Duck (0)", "50+", "100+"],
            "Rate": [df["Duck"].mean() * 100, df["FiftyPlus"].mean() * 100, df["CenturyPlus"].mean() * 100],
        }
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=milestones, x="Milestone", y="Rate", color="steelblue", ax=ax)
    ax.set_title("Key Scoring Milestone Rates")
    ax.set_xlabel("Milestone")
    ax.set_ylabel("Rate (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(percent_fmt))

    # Thesis note:
    # This is a simple descriptive summary of batting outcomes. It is especially helpful for
    # communicating to readers that the score distribution is not symmetric and contains both
    # many failures and a smaller number of very high performances.
    save_figure(fig, "23_scoring_milestone_rates")


# -----------------------------
# 6. OPTIONAL THESIS-READY NARRATIVE OUTPUT
# -----------------------------
def write_interpretation_notes(df: pd.DataFrame) -> None:
    """Generate a small text file with empirical observations the researcher can adapt."""
    notes = []
    notes.append("EDA interpretation notes for thesis drafting")
    notes.append("=" * 60)
    notes.append(f"Total innings analysed: {len(df)}")
    notes.append(f"Total players analysed: {df['Player'].nunique()}")
    notes.append(f"Date range: {df['StartDate'].min().date()} to {df['StartDate'].max().date()}")
    notes.append("")

    notes.append("1. Score distribution")
    notes.append(
        f"   The raw score distribution is strongly right-skewed, with a median of {df['FinalScore'].median():.1f} and a maximum of {df['FinalScore'].max():.0f}."
    )
    notes.append(
        "   This supports the use of a transformed score and a heavy-tailed observation model rather than a simple normal model."
    )
    notes.append("")

    notes.append("2. Censoring")
    notes.append(
        f"   Not-out innings account for {df['NotOut'].mean() * 100:.1f}% of observations, indicating that censoring is non-negligible and should be modelled explicitly."
    )
    notes.append("")

    notes.append("3. Contextual covariates")
    notes.append(
        f"   The sample contains {df['Location'].nunique()} match-location categories and opposition ranks ranging from {df['OpponentRank'].min():.0f} to {df['OpponentRank'].max():.0f}."
    )
    notes.append(
        "   These contextual variables are therefore empirically meaningful candidates for the observation equation of the Bayesian model."
    )
    notes.append("")

    notes.append("4. Player heterogeneity")
    notes.append(
        "   Large differences in average score, not-out rate, and innings counts across players suggest that player-specific effects are required."
    )
    notes.append("")

    notes.append("5. Temporal dynamics")
    notes.append(
        "   Player trajectories fluctuate noticeably over innings and calendar time, which supports the idea that batting form is time-varying and can be represented as a latent state process."
    )
    notes.append("")

    notes.append("6. Transition into Bayesian modelling chapter")
    notes.append(
        "   Overall, the EDA shows skewed and heavy-tailed score behaviour, meaningful censoring, heterogeneity across players, and visible temporal variation."
    )
    notes.append(
        "   Together, these findings motivate a Bayesian state-space model with player-level effects, contextual covariates, and censoring-aware score likelihoods."
    )

    (OUTPUT_DIR / "eda_interpretation_notes.txt").write_text("\n".join(notes), encoding="utf-8")


# -----------------------------
# 7. LIGHTWEIGHT STATISTICAL CHECKS
# -----------------------------
def write_statistical_checks(df: pd.DataFrame) -> None:
    """Small optional numeric checks to support the narrative."""
    checks = []
    checks.append("Exploratory statistical checks")
    checks.append("=" * 60)

    rho_rest, p_rest = spearmanr(df["RestDays"], df["FinalScore"])
    rho_rank, p_rank = spearmanr(df["OpponentRank"], df["FinalScore"])
    rho_innings, p_innings = spearmanr(df["Innings"], df["FinalScore"])

    checks.append(f"Spearman correlation: FinalScore vs RestDays      = {rho_rest:.3f} (p={p_rest:.4f})")
    checks.append(f"Spearman correlation: FinalScore vs OpponentRank = {rho_rank:.3f} (p={p_rank:.4f})")
    checks.append(f"Spearman correlation: FinalScore vs Innings      = {rho_innings:.3f} (p={p_innings:.4f})")
    checks.append("")
    checks.append(
        "These are only descriptive checks. In the thesis, avoid treating them as final evidence because the planned model is dynamic, hierarchical, and censoring-aware."
    )

    (OUTPUT_DIR / "exploratory_statistical_checks.txt").write_text("\n".join(checks), encoding="utf-8")


# -----------------------------
# 8. MAIN EXECUTION
# -----------------------------
def main() -> None:
    ensure_directories()
    df = load_data(INPUT_FILE)

    save_basic_tables(df)

    # Univariate analysis
    plot_missingness(df)
    plot_score_distribution(df)
    plot_score_distribution_transformed(df)
    plot_box_score(df)
    plot_categorical_frequencies(df)
    plot_player_innings_distribution(df)
    plot_rest_days_distribution(df)
    plot_opposition_rank_distribution(df)

    # Bivariate / multivariate analysis
    plot_score_by_location(df)
    plot_score_by_outstatus(df)
    plot_score_by_opponent_rank_band(df)
    plot_score_by_rest_days_band(df)
    plot_player_mean_scores(df)
    plot_player_notout_rates(df)
    plot_time_trend_all_players(df)
    plot_player_small_multiples(df, top_n=9)
    plot_score_vs_covariates(df)
    plot_numeric_correlation_heatmap(df)
    plot_notout_probability_by_context(df)
    plot_scoring_milestones(df)

    # Supporting notes
    write_interpretation_notes(df)
    write_statistical_checks(df)

    print("EDA completed successfully.")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
