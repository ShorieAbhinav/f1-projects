import fastf1
import pandas as pd
import numpy as np

fastf1.Cache.enable_cache("../cache")

# Belgian GP is round 12, so we want ewvery race that's already
#$ happened this season 1-11

ROUNDS_SO_FAR = list(range(1,12))

def fetch_round(round_number: int)-> pd.DataFrame:
    session = fastf1.get_session(2026, round_number, "R")
    session.load()

    results = session.results[
        ["DriverNumber", "Abbreviation", "TeamName", "GridPosition", "Position",
        "Status", "Points"]
    ].copy()

    results["Round"] = round_number
    results["DNF"] = ~results["Status"].str.contains("Finished", na = False)

    return results

def build_current_dataset() -> pd.DataFrame:
    frames = []
    for round_number in ROUNDS_SO_FAR:
        print(f"Fetching 2026 Round {round_number}...")
        try:
            frames.append(fetch_round(round_number))
        except Exception as e:
            print(f"  Skipped Round {round_number}: {e}")

    return pd.concat(frames, ignore_index = True)

def add_season_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each driver's season-sofar average finish and DNF rate,
    plus each team's season-so-far average finish (the 'how good is the car right now' signal - useful even for rookies with no personal history)
    """

    driver_stats = []
    for driver, group in df.groupby("Abbreviation"):
        driver_stats.append({
            "Abbreviation" : driver,
            "SeasonAvgFinish" : group["Position"].mean(),
            "SeasonAvgPoints": group["Points"].mean(),
            "SeasonDNFRate" : group["DNF"].astype(float).mean(),
            "SeasonRaces" : len(group),
        })
    driver_stats_df = pd.DataFrame(driver_stats)

    team_stats = []
    for team, group in df.groupby("TeamName"):
        team_stats.append({
            "TeamName" : team,
            "TeamSeasonAvgFinish" : group["Position"].mean(),
        })
    team_stats_df = pd.DataFrame(team_stats)

    df = df.merge(driver_stats_df, on="Abbreviation", how ="left")
    df = df.merge(team_stats_df, on = "TeamName", how = "left")

    return df

if __name__ == "__main__":
    raw = build_current_dataset()

    enriched = add_season_form_features(raw)

    enriched.to_csv("season_2026_form.csv", index=False)
    print(f"\nSaved {len(enriched)} rows to season_2026_form.csv")

    driver_summary = enriched[
        ["Abbreviation", "TeamName", "SeasonAvgFinish", "SeasonAvgPoints",
         "SeasonDNFRate", "TeamSeasonAvgFinish"]
    ].drop_duplicates(subset="Abbreviation").sort_values("SeasonAvgPoints", ascending=False)

    print("\n2026 season-so-far summary (per driver):")
    print(driver_summary.to_string(index=False))