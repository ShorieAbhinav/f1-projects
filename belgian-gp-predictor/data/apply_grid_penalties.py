"""
apply_grid_penalties.py

FastF1's qualifying data reflects qualifying classification, NOT the
actual starting grid — grid penalties are applied administratively
after qualifying and don't appear in the timing data until the race
session runs. Since we're predicting before the race, we override the
grid here with the real post-penalty starting order.

2026 Belgian GP penalties (from official post-qualifying grid):
  - Norris: 10-place penalty (power unit elements) -> qualified P3, starts P13
  - Stroll: 10-place penalty (power unit elements)
  - Hadjar: back of grid (multiple power unit elements)
  - Alonso: back of grid (multiple power unit elements)
"""

import pandas as pd

# Actual starting grid (post-penalty), by driver abbreviation.
ACTUAL_GRID = {
    "ANT": 1,
    "VER": 2,
    "RUS": 3,
    "LEC": 4,
    "HAM": 5,
    "PIA": 6,
    "LIN": 7,
    "BOR": 8,
    "LAW": 9,
    "GAS": 10,
    "COL": 11,
    "HUL": 12,
    "NOR": 13,
    "SAI": 14,
    "BEA": 15,
    "ALB": 16,
    "OCO": 17,
    "BOT": 18,
    "PER": 19,
    "STR": 20,
    "HAD": 21,
    "ALO": 22,
}

quali = pd.read_csv("belgian_gp_2026_qualifying.csv")

# Overwrite GridPosition with the real starting grid.
quali["GridPosition"] = quali["Abbreviation"].map(ACTUAL_GRID)

# Safety check: make sure every driver got a grid slot.
missing = quali[quali["GridPosition"].isna()]
if len(missing) > 0:
    print("WARNING — these drivers have no grid position mapped:")
    print(missing[["Abbreviation", "FullName"]].to_string(index=False))
else:
    quali = quali.sort_values("GridPosition")
    quali.to_csv("belgian_gp_2026_qualifying.csv", index=False)
    print("Applied post-penalty starting grid. Updated qualifying CSV:")
    print(quali[["Abbreviation", "FullName", "GridPosition"]].to_string(index=False))