import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

historical = pd.read_csv("../data/belgian_gp_historical.csv")


train_data = historical.dropna(subset=["GridPosition"]).copy()
train_data = train_data[train_data["DNF"] == False]   # <-- add this

team_year_avg = (
    historical.groupby(["TeamName", "Year"])["Position"]
    .mean()
    .reset_index()
    .rename(columns={"Position": "TeamYearAvgFinish"})
)

train_data = train_data.merge(team_year_avg, on=["TeamName", "Year"], how="left")
train_data = train_data.dropna(subset=["TeamYearAvgFinish"])

# Final feature set: grid position + team-year form.
# Team one-hot dummies were tested and removed — they made
# performance worse (4.24 vs 4.01 MAE), likely redundant with
# TeamYearAvgFinish and adding noise given the small dataset (~96 rows).
X = train_data[["GridPosition", "TeamYearAvgFinish"]]
y = train_data["Position"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"Mean Absolute Error on test set: {mae:.2f} positions")

# ── Predict on the actual 2026 Belgian GP field ──────────────────

qualifying = pd.read_csv("../data/belgian_gp_2026_qualifying.csv")
features_2026 = pd.read_csv("belgian_gp_features.csv")

# Bring in real grid position for each driver.
predict_data = features_2026.merge(
    qualifying[["Abbreviation", "GridPosition"]], on="Abbreviation", how="left"
)

# Use the current team's season form as our TeamYearAvgFinish
# equivalent — TeamSeasonAvgFinish already captures "how strong is
# this car in 2026," the same concept the model was trained on.
predict_data["TeamYearAvgFinish"] = predict_data["TeamSeasonAvgFinish"]

X_2026 = predict_data[["GridPosition", "TeamYearAvgFinish"]]

predict_data["PredictedFinish"] = model.predict(X_2026)

# Blend model prediction with the driver's actual 2026 season form.
# The model only sees grid + team; this pulls the prediction toward
# how each driver is genuinely performing this season individually.
# ── Bring in practice race pace and blend three signals ──────────

practice = pd.read_csv("../data/belgian_gp_practice_pace.csv")

# Convert practice pace into an implied finishing position (1 = fastest
# race pace). This puts it on the same 1-22 scale as the other signals
# so they can be blended sensibly.
practice = practice.sort_values("PracticePaceSeconds").reset_index(drop=True)
practice["PracticeRank"] = practice.index + 1

predict_data = predict_data.merge(
    practice[["Abbreviation", "PracticeRank"]], on="Abbreviation", how="left"
)

# Some drivers may lack practice data (missed the session, etc.) —
# fall back to their model prediction for those rows so nothing breaks.
predict_data["PracticeRank"] = predict_data["PracticeRank"].fillna(
    predict_data["PredictedFinish"]
)

# Grid position (via the model) anchors the prediction — Spa's chaos
# coefficient (~3.44) is moderate, so most drivers finish near their
# grid slot, and the deviations are better handled as Monte Carlo
# variance than predicted deterministically. Practice pace and season
# form are supporting adjustments, not primary drivers.
MODEL_WEIGHT = 0.55
PRACTICE_WEIGHT = 0.25
FORM_WEIGHT = 0.20

predict_data["PredictedFinish"] = (
    PRACTICE_WEIGHT * predict_data["PracticeRank"]
    + MODEL_WEIGHT * predict_data["PredictedFinish"]
    + FORM_WEIGHT * predict_data["SeasonAvgFinish"]
)

result = predict_data[
    ["Abbreviation", "FullName", "GridPosition", "PredictedFinish"]
].sort_values("PredictedFinish")

result.to_csv("belgian_gp_2026_predictions.csv", index=False)
print(f"\nSaved predictions to belgian_gp_2026_predictions.csv")

print("\n2026 Belgian GP - Predicted Finishing Order:")
print(result.to_string(index=False))