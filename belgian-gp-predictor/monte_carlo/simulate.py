import numpy as np
import pandas as pd

N_SIMULATIONS = 100000
CHAOS_COEFFICIENT = 3.44

predictions = pd.read_csv("../model/belgian_gp_2026_predictions.csv")
features = pd.read_csv("../model/belgian_gp_features.csv")

drivers = predictions.merge(
    features[["Abbreviation", "FinalDNFRate"]], on="Abbreviation", how="left"
)

# Race-day rain probability from Sunday's forecast (~20-28%).
RAIN_PROBABILITY = 0.25

# Wet races shuffle the grid far more — multiply chaos when it rains.
WET_CHAOS_MULTIPLIER = 2.0

# Wet races also raise retirement risk (spins, aquaplaning, incidents).
WET_DNF_MULTIPLIER = 1.5


def run_simulation(drivers, n_sims, chaos):
    names = drivers["FullName"].values
    predicted = drivers["PredictedFinish"].values
    dnf_rates = drivers["FinalDNFRate"].values
    n_drivers = len(drivers)

    win_counts = {name: 0 for name in names}
    podium_counts = {name: 0 for name in names}
    wet_race_count = 0

    for _ in range(n_sims):
        # Roll for rain this simulated race.
        is_wet = np.random.random() < RAIN_PROBABILITY

        if is_wet:
            wet_race_count += 1
            effective_chaos = chaos * WET_CHAOS_MULTIPLIER
            effective_dnf = np.minimum(dnf_rates * WET_DNF_MULTIPLIER, 1.0)
        else:
            effective_chaos = chaos
            effective_dnf = dnf_rates

        # Add race-day randomness, scaled by (weather-adjusted) chaos.
        noise = np.random.normal(0, effective_chaos, n_drivers)
        sim_result = predicted + noise

        # Roll for DNFs using the (weather-adjusted) rates.
        dnf_roll = np.random.random(n_drivers) < effective_dnf
        sim_result[dnf_roll] = 999

        order = np.argsort(sim_result)

        win_counts[names[order[0]]] += 1
        for pos in range(3):
            podium_counts[names[order[pos]]] += 1

    print(f"Simulated {wet_race_count:,} wet races out of {n_sims:,} "
          f"({wet_race_count/n_sims*100:.1f}%)")

    return win_counts, podium_counts

if __name__ == "__main__":
    win_counts, podium_counts = run_simulation(
        drivers, N_SIMULATIONS, CHAOS_COEFFICIENT
    )

    # Convert raw counts into probabilities and attach to the table.
    drivers["WinProbability"] = drivers["FullName"].map(win_counts) / N_SIMULATIONS
    drivers["PodiumProbability"] = drivers["FullName"].map(podium_counts) / N_SIMULATIONS

    result = drivers[
        ["FullName", "GridPosition", "PredictedFinish", "WinProbability", "PodiumProbability"]
    ].sort_values("WinProbability", ascending=False)

    result.to_csv("belgian_gp_2026_simulation.csv", index=False)
    print(f"Ran {N_SIMULATIONS:,} simulations\n")

    print("2026 Belgian GP - Monte Carlo Predictions:")
    print(f"{'Driver':<20} {'Grid':>5} {'Win %':>8} {'Podium %':>10}")
    print("-" * 46)
    for _, row in result.iterrows():
        print(f"{row['FullName']:<20} "
              f"P{int(row['GridPosition']):>3} "
              f"{row['WinProbability']*100:>7.2f}% "
              f"{row['PodiumProbability']*100:>9.2f}%")