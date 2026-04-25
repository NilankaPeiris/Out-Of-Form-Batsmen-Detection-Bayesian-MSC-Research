from pathlib import Path
import json
import pandas as pd


# =========================
# CONFIG
# =========================
DATA_FILE = Path("data/final_cricket_data.xlsx")  # change filename.xlsx to your real file name
POSTERIOR_FILE = Path("artifacts/posterior_summary.json")
OUTPUT_DIR = Path("artifacts")

PLAYER_ID_COL = "PlayerID"
PLAYER_NAME_COL = "Player"   # if your dataset has player names
DATE_COL = "StartDate"       # optional, used to identify last innings


# =========================
# LOAD DATA
# =========================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA_FILE)

with open(POSTERIOR_FILE, "r", encoding="utf-8") as f:
    posterior_summary = json.load(f)


# =========================
# VALIDATION
# =========================
if PLAYER_ID_COL not in df.columns:
    raise ValueError(f"Column '{PLAYER_ID_COL}' not found in dataset.")

if "u_mean" not in posterior_summary:
    raise ValueError("'u_mean' not found in posterior_summary.json.")

if "last_training_posterior" not in posterior_summary:
    raise ValueError("'last_training_posterior' not found in posterior_summary.json.")


# =========================
# RECREATE PLAYER INDEX MAPPING
# IMPORTANT:
# This must match the same logic used before modelling.
# If your modelling code used sorted PlayerID, keep this.
# =========================
player_lookup = (
    df[[PLAYER_ID_COL, PLAYER_NAME_COL]]
    .drop_duplicates()
    .sort_values(PLAYER_ID_COL)
    .reset_index(drop=True)
    if PLAYER_NAME_COL in df.columns
    else df[[PLAYER_ID_COL]]
    .drop_duplicates()
    .sort_values(PLAYER_ID_COL)
    .reset_index(drop=True)
)

player_lookup["stan_player_index"] = range(1, len(player_lookup) + 1)


# =========================
# MAP PLAYER RANDOM EFFECTS
# =========================
u_mean = posterior_summary["u_mean"]

if len(u_mean) != len(player_lookup):
    raise ValueError(
        f"Mismatch: posterior has {len(u_mean)} player effects, "
        f"but dataset has {len(player_lookup)} unique players. "
        "Check whether the same filtering was used before modelling."
    )

player_lookup["player_random_effect_u_mean"] = u_mean


# =========================
# MAP LAST TRAINING POSTERIORS
# =========================
last_post = posterior_summary["last_training_posterior"]

posterior_rows = []

for player_id, probs in last_post.items():
    posterior_rows.append({
        PLAYER_ID_COL: int(player_id) if str(player_id).isdigit() else player_id,
        "OOF_probability": probs[0],
        "NF_probability": probs[1],
        "HF_probability": probs[2],
        "OOF_percentage": probs[0] * 100,
        "NF_percentage": probs[1] * 100,
        "HF_percentage": probs[2] * 100,
        "most_likely_state": ["OOF", "NF", "HF"][probs.index(max(probs))],
        "OOF_flag_70_threshold": probs[0] > 0.70
    })

last_posterior_df = pd.DataFrame(posterior_rows)


# =========================
# MERGE FINAL OUTPUT
# =========================
final_output = player_lookup.merge(
    last_posterior_df,
    on=PLAYER_ID_COL,
    how="left"
)


# =========================
# ADD LAST INNINGS DETAILS IF AVAILABLE
# =========================
if DATE_COL in df.columns:
    df_sorted = df.sort_values([PLAYER_ID_COL, DATE_COL])
    last_innings = (
        df_sorted
        .groupby(PLAYER_ID_COL)
        .tail(1)
        .copy()
    )

    useful_cols = [
        PLAYER_ID_COL,
        DATE_COL,
        "FinalScore",
        "OutStatus",
        "Location",
        "OpponentRank",
        "RestDays"
    ]

    useful_cols = [c for c in useful_cols if c in last_innings.columns]

    final_output = final_output.merge(
        last_innings[useful_cols],
        on=PLAYER_ID_COL,
        how="left"
    )


# =========================
# SAVE OUTPUTS
# =========================
csv_output = OUTPUT_DIR / "player_level_model_summary.csv"
xlsx_output = OUTPUT_DIR / "player_level_model_summary.xlsx"
json_output = OUTPUT_DIR / "player_level_model_summary.json"

final_output.to_csv(csv_output, index=False)
final_output.to_excel(xlsx_output, index=False)
final_output.to_json(json_output, orient="records", indent=4)

print("Player-level model summary created successfully.")
print(f"CSV saved to:  {csv_output}")
print(f"Excel saved to: {xlsx_output}")
print(f"JSON saved to: {json_output}")

print("\nPreview:")
print(final_output.head())