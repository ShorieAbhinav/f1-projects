# Belgian GP 2026 Predictor

A Monte Carlo race predictor for the 2026 Belgian Grand Prix at Spa-Francorchamps. Pulls historical and current-season data via FastF1, trains a machine learning model to estimate finishing positions, and runs 100,000 weather-aware simulations to produce win and podium probabilities for every driver.

## Prediction

Final prediction (after Saturday qualifying), by win probability:

| Rank | Driver | Grid | Win % | Podium % |
|------|--------|------|-------|----------|
| 1 | Lando Norris | P3 | 23.0% | 51.8% |
| 2 | Charles Leclerc | P5 | 21.0% | 53.6% |
| 3 | Lewis Hamilton | P6 | 18.4% | 50.1% |
| 4 | Kimi Antonelli | P1 | 12.7% | 40.3% |
| 5 | George Russell | P4 | 9.4% | 29.8% |
| 6 | Max Verstappen | P2 | 6.1% | 23.9% |

A tight three-way fight at the front between Norris, Leclerc, and Hamilton, all of whom combined strong practice race pace with solid grid positions. Leclerc actually leads on podium probability despite Norris leading on wins, reflecting his consistency across simulations.

## Methodology

The final prediction blends four signals, then feeds the result into a Monte Carlo simulation that models race-day randomness.

### Data sources (via FastF1)
- **Historical Spa races (2019-2025)** - excluding 2021 (rain-shortened to ~1 lap, not representative).
- **2026 season form (rounds 1-9)** - each driver's and team's average finish and points so far this season.
- **2026 Spa qualifying** - Saturday's actual grid.
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
│   ├── fetch_qualifying.py       # Saturday's real grid
│   └── fetch_practice.py         # FP1-3 race pace, filtered
├── model/
│   ├── build_features.py         # merges sources, rookie/veteran branching
│   └── scorer.py                 # trains model, blends signals, predicts
├── monte_carlo/
│   └── simulate.py               # 100k weather-aware simulations
└── gui/
    └── dashboard.py              # Streamlit dashboard
```

Run order: the data/ scripts, then build_features.py, scorer.py, simulate.py, and finally streamlit run gui/dashboard.py.

## Limitations

Honest notes on where this model is uncertain:
- **Small dataset.** The model trains on ~96 historical finisher rows. Metrics like the 1.85 MAE carry real uncertainty from the small test set, and more features don't reliably help at this scale.
- **Gradient Boosting was chosen without a rigorous side-by-side comparison** against Random Forest, due to time constraints.
- **Several parameters are reasoned estimates, not optimized** - the recency decay, blend weights, rain probability, and wet-race multipliers are set to sensible values but not tuned via holdout validation.
- **Practice pace is noisy.** Teams run undisclosed fuel loads and engine modes in practice, so long-run pace is indicative, not definitive.
- **F1 is inherently chaotic.** A large share of any race outcome - safety cars, first-lap incidents, strategy, weather - cannot be predicted from pre-race data. The Monte Carlo layer expresses this as probability rather than false precision.

## Stack
Python · FastF1 · scikit-learn · NumPy · pandas · Plotly · Streamlit