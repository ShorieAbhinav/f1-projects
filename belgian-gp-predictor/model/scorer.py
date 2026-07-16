import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

historical = pd.read_csv("../data/belgian_gp_historical.csv")

train_data = historical.dropna(subset=["GridPosition"]).copy()

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

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"Mean Absolute Error on test set: {mae:.2f} positions")