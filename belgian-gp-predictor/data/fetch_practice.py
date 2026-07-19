import fastf1
import pandas as pd
import numpy as np

fastf1.Cache.enable_cache("../cache")

# The three practice sessions for a standard (non-sprint) weekend.
PRACTICE_SESSIONS = ["FP1", "FP2", "FP3"]


def fetch_practice_laps(session_name):
    """Pull all lap data from one practice session."""
    session = fastf1.get_session(2026, "Belgium", session_name)
    session.load()

    laps = session.laps[
        ["Driver", "LapTime", "LapNumber", "Compound",
         "TyreLife", "PitOutTime", "PitInTime"]
    ].copy()

    laps["Session"] = session_name
    return laps


def build_practice_dataset():
    """Pull and combine all three practice sessions."""
    frames = []
    for session_name in PRACTICE_SESSIONS:
        print(f"Fetching {session_name}...")
        try:
            frames.append(fetch_practice_laps(session_name))
        except Exception as e:
            print(f"  Skipped {session_name}: {e}")
    return pd.concat(frames, ignore_index=True)


def compute_race_pace(laps):
    """
    A2 filtering: isolate representative race-pace laps and compute
    a clean pace figure per driver.

    Steps:
    1. Drop laps with no lap time (missing/invalid).
    2. Drop in-laps and out-laps (entering/leaving pits) — these are
       artificially slow and not representative of race pace.
    3. Drop very early tyre-life laps aren't the issue; instead we
       drop each driver's fastest laps as likely qualifying simulations
       (low fuel, one-lap pace), keeping the race-representative meat.
    4. For what remains, take each driver's median lap time — median
       rather than mean so a single anomalous lap doesn't skew it.
    """

    # 1. Drop laps with no recorded lap time.
    laps = laps.dropna(subset=["LapTime"]).copy()

    # Convert LapTime (a timedelta) into seconds for math.
    laps["LapSeconds"] = laps["LapTime"].dt.total_seconds()

    # 2. Drop in-laps and out-laps. On a normal flying lap, both
    #    PitInTime and PitOutTime are null (NaT). If either is set,
    #    this lap involved the pits and isn't representative.
    laps = laps[laps["PitOutTime"].isna() & laps["PitInTime"].isna()]

    # 3. Remove obvious outliers per driver, then strip likely
    #    qualifying-simulation laps (the very fastest, low-fuel runs).
    clean_rows = []
    for driver, group in laps.groupby("Driver"):
        times = group["LapSeconds"]

        # Guard: need enough laps to filter meaningfully.
        if len(times) < 5:
            continue

        # Drop extreme outliers (traffic laps, aborted laps, etc.)
        # using the interquartile range — keep the middle bulk.
        q1 = times.quantile(0.25)
        q3 = times.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered = group[(times >= lower) & (times <= upper)]

        # Of what remains, drop the fastest 20% as likely quali sims
        # (low-fuel one-lap pace), keeping race-representative laps.
        cutoff = filtered["LapSeconds"].quantile(0.20)
        race_pace_laps = filtered[filtered["LapSeconds"] >= cutoff]

        clean_rows.append(race_pace_laps)

    clean = pd.concat(clean_rows, ignore_index=True)
    # Drop reserve/non-race drivers (e.g. FP1-only stand-ins) who
    # aren't in Sunday's field.
    clean = clean[clean["Driver"] != "CRA"]

    # 4. Median race-pace lap time per driver.
    pace = (
        clean.groupby("Driver")["LapSeconds"]
        .median()
        .reset_index()
        .rename(columns={"Driver": "Abbreviation",
                         "LapSeconds": "PracticePaceSeconds"})
    )

    # A relative measure is more useful than raw seconds: how far off
    # the fastest driver each driver is (gap to best practice pace).
    fastest = pace["PracticePaceSeconds"].min()
    pace["PracticePaceGap"] = pace["PracticePaceSeconds"] - fastest

    return pace.sort_values("PracticePaceSeconds")


if __name__ == "__main__":
    raw = build_practice_dataset()
    pace = compute_race_pace(raw)

    pace.to_csv("belgian_gp_practice_pace.csv", index=False)
    print(f"\nSaved practice pace for {len(pace)} drivers "
          f"to belgian_gp_practice_pace.csv")
    print("\n2026 Belgian GP - Practice race pace (gap to fastest):")
    print(pace.to_string(index=False))