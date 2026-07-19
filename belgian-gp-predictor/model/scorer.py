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

result = predict_data[
    ["Abbreviation", "FullName", "GridPosition", "PredictedFinish"]
].sort_values("PredictedFinish")

result.to_csv("belgian_gp_2026_predictions.csv", index=False)
print(f"\nSaved predictions to belgian_gp_2026_predictions.csv")

print("\n2026 Belgian GP - Predicted Finishing Order:")
print(result.to_string(index=False))