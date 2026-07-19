# Belgian GP 2026 Predictor

A Monte Carlo race predictor for the 2026 Belgian Grand Prix at Spa-Francorchamps. Pulls historical and current-season data via FastF1, trains a machine learning model to estimate finishing positions, and runs 100,000 weather-aware simulations to produce win and podium probabilities for every driver.

## Prediction

Final prediction (post-qualifying, after grid penalties applied), by win probability:

| Rank | Driver | Grid | Win % | Podium % |
|------|--------|------|-------|----------|
| 1 | Charles Leclerc | P4 | 25.9% | 58.0% |
| 2 | Lewis Hamilton | P5 | 21.3% | 52.9% |
| 3 | Kimi Antonelli | P1 | 14.3% | 42.5% |
| 4 | George Russell | P3 | 11.8% | 33.6% |
| 5 | Max Verstappen | P2 | 6.6% | 25.1% |
| 6 | Oscar Piastri | P6 | 6.2% | 22.5% |

Leclerc emerges as the favorite, combining the fastest practice race pace with a P4 start. Hamilton follows closely, giving Ferrari two cars at the front. Note the grid reflects post-qualifying penalties: Norris qualified P3 but starts P13 (10-place power-unit penalty), which drops him from a would-be favorite to a P7 win chance despite strong underlying pace — a reminder that grid position is the model's dominant signal. Stroll (10-place), Hadjar and Alonso (back of grid) also carry penalties.

## Methodology

The final prediction blends four signals, then feeds the result into a Monte Carlo simulation that models race-day randomness.

### Data sources (via FastF1)
- **Historical Spa races (2019-2025)** - excluding 2021 (rain-shortened to ~1 lap, not representative).
- **2026 season form (rounds 1-9)** - each driver's and team's average finish and points so far this season.
- **2026 Spa qualifying** - Saturday's classification, with a manual override applying the official post-qualifying grid penalties (FastF1 timing data does not reflect administrative penalties before the race runs).
- **2026 practice sessions (FP1-FP3)** - long-run race pace, filtered to exclude in/out laps, outliers, and low-fuel qualifying simulations.

### The model
A Gradient Boosting Regressor (scikit-learn) trained on historical Spa results to predict finishing position from grid position and team-year form. Trained only on drivers who finished (DNFs excluded), since a DNF reflects reliability, not pace - that risk is handled separately in the simulation. Test-set mean absolute error: ~1.85 finishing positions.

### The blend
The model's prediction is combined with two other signals into a final expected finishing position:
- **Model (grid + team), 55%** - grid position anchors the prediction. Spa's chaos coefficient (~3.44 average grid-to-finish position change) is moderate, so most drivers finish near their grid slot; the deviations are better modeled as simulation variance than predicted deterministically.
- **Practice race pace, 25%** - the most current, long-run race-representative signal, capturing this weekend's true pace including any car upgrades.
- **Season form, 20%** - general current-year driver and car quality.

### The simulation
100,000 Monte Carlo simulations. Each race:
- Adds random noise to every driver's expected finish, scaled by Spa's chaos coefficient.
- Rolls for DNFs per driver based on their blended DNF rate.
- Rolls for rain (25% race-day probability per Sunday's forecast); wet races multiply chaos by 2 and DNF risk by 1.5, reflecting how much more a wet Spa shuffles the grid.

Win and podium probabilities are the frequency of each outcome across all simulations.

## Architecture

```
belgian-gp-predictor/
├── data/
│   ├── fetch_historical.py       # Spa races 2019-2025, recency-weighted, chaos coefficient
│   ├── fetch_current_season.py   # 2026 season form (rounds 1-9)
│   ├── fetch_qualifying.py       # Saturday's classification
│   ├── apply_grid_penalties.py   # override with actual post-penalty starting grid
│   └── fetch_practice.py         # FP1-3 race pace, filtered
├── model/
│   ├── build_features.py         # merges sources, rookie/veteran branching
│   └── scorer.py                 # trains model, blends signals, predicts
├── monte_carlo/
│   └── simulate.py               # 100k weather-aware simulations
└── gui/
    └── dashboard.py              # Streamlit dashboard
```

Run order: the data/ scripts (including apply_grid_penalties.py after qualifying), then build_features.py, scorer.py, simulate.py, and finally streamlit run gui/dashboard.py.

## Limitations

Honest notes on where this model is uncertain:
- **Small dataset.** The model trains on ~96 historical finisher rows. Metrics like the 1.85 MAE carry real uncertainty from the small test set, and more features don't reliably help at this scale.
- **Gradient Boosting was chosen without a rigorous side-by-side comparison** against Random Forest, due to time constraints.
- **Several parameters are reasoned estimates, not optimized** - the recency decay, blend weights, rain probability, and wet-race multipliers are set to sensible values but not tuned via holdout validation.
- **Some inputs require manual entry.** Grid penalties and (potentially) car upgrades are real, race-relevant events that FastF1's pre-race data does not capture, so they are applied by hand from official sources.
- **Practice pace is noisy.** Teams run undisclosed fuel loads and engine modes in practice, so long-run pace is indicative, not definitive.
- **F1 is inherently chaotic.** A large share of any race outcome - safety cars, first-lap incidents, strategy, weather - cannot be predicted from pre-race data. The Monte Carlo layer expresses this as probability rather than false precision.

## Stack
Python · FastF1 · scikit-learn · NumPy · pandas · Plotly · Streamlit