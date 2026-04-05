MVM-2: Matchup Volatility Modeling Engine
This repository contains the architecture, data parsing logic, and execution steps for predicting March Madness outcomes. MVM-2 utilizes a Triple-Engine Ensemble (PyTorch Neural Network, XGBoost, and Random Forest) stacked with a Logistic Regression Meta-Learner to project tournament outcomes, run Monte Carlo simulations, and generate diversified bracket portfolios.

System Architecture & Data Flow
Multi-Source Data Ingestion: Base efficiency metrics are loaded from kenpom.csv and torvik.csv. Historical matchup data is sourced from EvanMiya.

Live Tactical & Contextual Scouting: The pipeline actively scrapes Sports-Reference to build contextual profiles for teams. This includes:
# MVM-2: Matchup Volatility Modeling Engine

This repository contains the architecture, data parsing logic, and execution steps for predicting March Madness outcomes. **MVM-2** utilizes a Triple-Engine Ensemble (PyTorch Neural Network, XGBoost, and Random Forest) stacked with a Logistic Regression Meta-Learner to project tournament outcomes, run Monte Carlo simulations, and generate diversified bracket portfolios.

## System Architecture & Data Flow

1. **Multi-Source Data Ingestion**: Base efficiency metrics are loaded from `kenpom.csv` and `torvik.csv`. Historical matchup data is sourced from `EvanMiya.csv`. 
2. **Live Tactical & Contextual Scouting**: The pipeline actively scrapes *Sports-Reference* to build contextual profiles for teams. This includes:
   * **Schedule Engine**: Calculates a `Fragility Score` and `Momentum Scalar` based on regular-season performance and quality/bad losses.
   * **Coach Engine**: Derives Deep Run Experience (`S16`, `E8`, `F4`, `PAKE`/`PASE`) for current head coaches.
   * **Injury Scraper**: Dynamically adjusts offensive and defensive ratings based on player usage, bench depth mitigation, and an *Alpha-Replacement Penalty*.
3. **Tactical Feature Engineering**: Raw stats are modified in real-time by `tactics.py` to account for schematic advantages, including the *Stretch-Big Index* spacing penalty, P4 Athleticism Floor adjustments, and Bracket Buster DNA (elite 3PT volume/turnover forcing).
4. **Triple-Engine Inference**: Features are normalized and fed into three parallel models:
   * **PyTorch NN**: A feed-forward deep neural network with `BatchNorm1d` and `Dropout` layers.
   * **XGBoost Classifier**: Gradient-boosted trees trained with upset-weighted samples.
   * **Random Forest**: High-depth forest for ensemble variance reduction.
5. **Meta-Learner & Monte Carlo Simulation**: The base probabilities are stacked and calibrated using a Logistic Regression Meta-Learner. The `predict.py` engine can then inject Gaussian noise based on team fragility to simulate thousands of bracket permutations, scoring them by Expected Value (EV) to generate a diversified portfolio (Core, Pivot, and Chaos brackets).

## Setup Instructions

1. Ensure your Python virtual environment is activated:
   * **Windows**: `.\venv\Scripts\activate`
   * **Mac/Linux**: `source venv/bin/activate`

2. Install all required dependencies:
   ```bash
   pip install torch pandas numpy scikit-learn xgboost joblib beautifulsoup4 requests rich
Schedule Engine: Calculates a "Fragility Score" and "Momentum Scalar" based on regular-season performance and quality/bad losses.

Coach Engine: Derives "Deep Run Experience" (S16, E8, F4, PAKE/PASE) for current head coaches.

Injury Scraper: Dynamically adjusts offensive and defensive ratings based on player usage, bench depth mitigation, and an "Alpha-Replacement Penalty."

Tactical Feature Engineering: Raw stats are modified in real-time by tactics.py to account for schematic advantages, including the "Stretch-Big Index" spacing penalty, P4 Athleticism Floor adjustments, and Bracket Buster DNA (elite 3PT volume/turnover forcing).

Triple-Engine Inference: Features are normalized and fed into three parallel models:

PyTorch NN: A feed-forward deep neural network with Batch Normalization and Dropout layers.

XGBoost Classifier: Gradient-boosted trees trained with upset-weighted samples.

Random Forest: High-depth forest for ensemble variance reduction.

Meta-Learner & Monte Carlo Simulation: The base probabilities are stacked and calibrated using a Logistic Regression Meta-Learner. The predict.py engine can then inject Gaussian noise based on team fragility to simulate thousands of bracket permutations, scoring them by Expected Value (EV) to generate a diversified portfolio (Core, Pivot, and Chaos brackets).

Setup Instructions
Ensure your Python virtual environment is activated:

Windows: .\venv\Scripts\activate

Mac/Linux: source venv/bin/activate

Install all required dependencies:

pip install torch pandas numpy scikit-learn xgboost joblib beautifulsoup4 requests rich

Structure your local data directories:

Place kenpom.csv, torvik.csv, EvanMiya.csv, and Tournament Matchups.csv into the appropriate data/raw/ directory (or map them in config.py).

Run the data verifier and model training pipeline:

python src/run_pipeline.py

Project Structure
Plaintext
├── src/
│   ├── predict.py           # Core CLI, Matchup inference, and Monte Carlo engine
│   ├── train.py             # Triple-engine training and Meta-Learner stacking
│   ├── run_pipeline.py      # Automated data loading and training trigger
│   ├── data_loader.py       # Parses Torvik/Kenpom/EvanMiya datasets and handles caching
│   ├── model.py             # PyTorch Neural Network architecture
│   ├── tactics.py           # Applies schematic modifiers (Stretch-Big, Upset Markers)
│   ├── schedule_engine.py   # Scrapes schedules to calculate Fragility & Momentum
│   ├── coach_engine.py      # Scrapes and scores Coach PAKE/PASE and Deep Run metrics
│   ├── injuries.py          # Math logic for penalizing team efficiency based on missing players
│   ├── injury_scraper.py    # Rotowire injury report scraper
│   └── utils.py             # Normalization and slug generation tools
├── data/
│   └── raw/                 # Base CSV files (KenPom, Torvik, Historical Matchups)
└── README.md
CLI Usage
The predict.py script offers a rich CLI experience for analyzing single games or simulating the entire tournament field.

Single Matchup Prediction (Live Scout):

Bash
python src/predict.py "Purdue" "Gonzaga"
Force Live Refresh (Bypass Local Cache):

Bash
python src/predict.py "Duke" "North Carolina" --refresh
Run Single Matchup Probabilistic Simulations (e.g., 1000 times):

Bash
python src/predict.py "Connecticut" "Houston" --sim_matchup 1000
Simulate Top 64 Bracket (Deterministic):

Bash
python src/predict.py --simulate
Run Mass Monte Carlo Portfolio Generation:
Generates N brackets using fragility-based stat jittering, scores them by Global EV, and outputs a top-tier diversified portfolio to CSV.

Bash
python src/predict.py --monte_carlo 1000
