import fastf1
import pandas as pd

fastf1.Cache.enable_cache("../cache")

def fetch_qualifying_results():
    session = fastf1.get_session(2026, "Belgium", "Q")
    session.load()

    results = session.results[
        ["DriverNumber", "Abbreviation", "FullName", "TeamName", "Position"]
    ].copy()

    # In qualifying, "Position" is the qualifying classification —
    # this becomes tomorrow's grid position.
    results = results.rename(columns={"Position": "GridPosition"})

    return results


if __name__ == "__main__":
    quali = fetch_qualifying_results()
    quali.to_csv("belgian_gp_2026_qualifying.csv", index=False)
    print(f"Saved {len(quali)} rows to belgian_gp_2026_qualifying.csv")
    print(quali.sort_values("GridPosition").to_string(index=False))