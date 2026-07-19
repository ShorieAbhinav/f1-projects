import fastf1
import pandas as pd
import numpy as np

fastf1.Cache.enable_cache('../cache')

YEARS = [2019, 2020, 2022, 2023, 2024, 2025]
DECAY = 0.85

def fetch_year(year: int) -> pd.DataFrame:
    """ Pull data for a given year """
    # Ask fastf1 for the session data, R = Racw
    # Q = Qualifying, FP1/2/3 = Practice Sessions
    session = fastf1.get_session(year, "Belgian Grand Prix", "R")

    # This downloads the data for the session
    session.load()

    # Session datas has many columns we only want to keep the ones we need
    results = session.results[["DriverNumber", "Abbreviation","FullName","TeamName", "GridPosition", "Position", "Status", "Points"]].copy()
    results["Year"] = year
    results["DNF"] = ~results["Status"].str.contains("Finished", na=False)

    return results

def build_historical_dataset():
    """ Loop through the YEARS, pull one year at a time, concat the results into a single DataFrame """

    frames = []
    for year in YEARS:
        print(f"Fetching {year} Belgian Grand Prix...")
        try:
            frames.append(fetch_year(year))
        except Exception as e:
            print(f"skipped {year} due to error: {e}")

    return pd.concat(frames, ignore_index=True)

def add_recency_weighted_features(df):
    """
    Compute each driver's recency-weighted average finish and DNF rate at Spa, using exponential decay so recent years count more.
    """

    most_recent_year = df["Year"].max()

    #Assignn each row a weight based on how recent its year is
    df["YearWeight"] = DECAY ** (most_recent_year - df["Year"])

    driver_stats = []

    # Group all rows by driver, so we can compute one summary per driver.
    for driver, group in df.groupby("Abbreviation"):
        # Rows with higher YearWeight count more towards the average
        weighted_avg_finish = np.average(group["Position"], weights=group["YearWeight"])
        weighted_dnf_rate = np.average(group["DNF"].astype(float), weights=group["YearWeight"])

        driver_stats.append({
            "Abbreviation": driver,
            "SpaWeightedAvgFinish": weighted_avg_finish,
            "SpaWeightedDNFRate": weighted_dnf_rate,
            "SpaRaces": len(group),
        })

    stats_df = pd.DataFrame(driver_stats)
    return df.merge(stats_df, on="Abbreviation", how="left")

def compute_chaos_coefficient(df):
    """
    Average position change at Spa across all historical races.
    Higher = more chaotic circuit where grid position matters less.
    Lower = processional race where qualifying position is everything.
    Same definition used in the Chinese GP project: mean absolute
    difference between grid position and finish position.
    """
    position_changes = []

    for _, row in df.iterrows():
        grid = row.get("GridPosition")
        finish = row.get("Position")

        if grid and finish and grid > 0 and finish > 0:
            change = abs(int(finish) - int(grid))
            position_changes.append(change)

    if len(position_changes) == 0:
        print("No valid data - using default chaos coefficient")
        return 3.5

    coefficient = np.mean(position_changes)
    print(f"\nSpa chaos coefficient: {coefficient:.2f}")
    print(f"Based on {len(position_changes)} driver results")
    return coefficient
    
def add_driver_metadata(df):
    """ Add driver metadata from the driver_info.csv file """
    driver_info = pd.read_csv("driver_info.csv")
    df = df.merge(driver_info, on="Abbreviation", how="left")
    return df

if __name__ == "__main__":
    # This block only runs when you execute this file directly
    # (python3 fetch_historical.py), not if this file gets imported
    # elsewhere later. Standard Python convention for a script's entry point.

    raw = build_historical_dataset()
    chaos_coefficient = compute_chaos_coefficient(raw)

    enriched = add_recency_weighted_features(raw)
    enriched.to_csv("belgian_gp_historical.csv", index = False)
    print(f"\nSaved {len(enriched)} rows to belgian_gp_historical.csv")

    # Show one row per driver (drop duplicate rows across years) with
    # just the weighted summary columns, sorted best to worst.

    summary = enriched[
        ["Abbreviation","FullName", "SpaWeightedAvgFinish", "SpaWeightedDNFRate", "SpaRaces"]
    ].drop_duplicates().sort_values(by="SpaWeightedAvgFinish")
    
    print("\nPer-driver Spa summary (recency-weighted):")
    print(summary.to_string(index=False))


