import pandas as pd
import numpy as np

# Load the two data sources built earlier this week.
historical = pd.read_csv("../data/belgian_gp_historical.csv")
current_season = pd.read_csv("../data/season_2026_form.csv")

# Reduce historical data down to one row per driver - just the
# recency-weighted Spa stats we computed earlier this week 
historical_summary = historical[
    ["Abbreviation", "SpaWeightedAvgFinish", "SpaWeightedDNFRate", "SpaRaces"]
].drop_duplicates(subset="Abbreviation")

# Same idea for current season data - one row per driver data, plus their
# team, since we'll need TeamName to bring in team-level Spa/season stats.
current_summary =  current_season[
    ["Abbreviation", "FullName", "TeamName", "SeasonAvgFinish", "SeasonAvgPoints",
    "SeasonDNFRate", "SeasonRaces", "TeamSeasonAvgFinish"]
].drop_duplicates(subset="Abbreviation")

# Merge on Abbreviation. how="outer" keeps every driver from either
# source — even a rookie with zero Spa history (who'd be missing from
# historical_summary) still shows up, just with NaN in the Spa columns.
features = current_summary.merge(
    historical_summary, on="Abbreviation", how="outer"
)

# Keep only drivers actually racing in 2026 — anyone with no season
# data isn't on this weekend's grid, so their old Spa history is
# irrelevant to us.
features = features[features["SeasonAvgPoints"].notna()].copy()

# Branching logic: a driver needs a reasonable amount of Spa history
# before we trust their weighted average. Below this threshold, treat
# them as a "rookie" for Spa purposes and rely on current-season form
# instead (using the team's Spa history as a fallback signal for
# "how does this car tend to do here," even if the driver doesn't).
MIN_SPA_RACES = 3

features["IsSpaRookie"] = (
    features["SpaRaces"].isna() | (features["SpaRaces"] < MIN_SPA_RACES)
)

def compute_final_features(df):
    """
    For each driver, produce one blended set of features — combining
    personal Spa history (when trustworthy) with current-season form,
    so a driver whose team changed since their Spa results won't be
    scored on a car they no longer drive.
    """

    # How much to trust personal Spa history vs. current team form.
    # Only applies when the driver actually has Spa history to blend.
    SPA_WEIGHT = 0.6
    TEAM_WEIGHT = 0.4

    blended_avg_finish = (
        SPA_WEIGHT * df["SpaWeightedAvgFinish"]
        + TEAM_WEIGHT * df["TeamSeasonAvgFinish"]
    )

    # Rookies (no usable Spa history): rely fully on current form —
    # their own season average, since there's no personal Spa signal
    # to blend in at all.
    df["FinalAvgFinish"] = np.where(
        df["IsSpaRookie"],
        df["SeasonAvgFinish"],
        blended_avg_finish
    )

    # DNF rate: same blending idea, personal Spa DNF history blended
    # with current-season DNF rate for non-rookies, pure season rate
    # for rookies.
    blended_dnf_rate = (
        SPA_WEIGHT * df["SpaWeightedDNFRate"]
        + TEAM_WEIGHT * df["SeasonDNFRate"]
    )

    df["FinalDNFRate"] = np.where(
        df["IsSpaRookie"],
        df["SeasonDNFRate"],
        blended_dnf_rate
    )

    return df

features = compute_final_features(features)

if __name__ == "__main__":
    features = compute_final_features(features)

    features.to_csv("belgian_gp_features.csv", index=False)
    print(f"\nSaved {len(features)} rows to belgian_gp_features.csv")

    summary = features[
        ["Abbreviation", "FullName", "IsSpaRookie", "FinalAvgFinish", "FinalDNFRate"]
    ].sort_values("FinalAvgFinish")

    print("\nBelgian GP 2026 - blended driver features:")
    print(summary.to_string(index=False))