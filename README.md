# 🏎️ F1 Race Predictor

Predicts F1 race winners, podium finishers, and DNF probabilities using FastF1 data and three ML models (XGBoost, Random Forest, Neural Net).

## Project Structure

```
f1-predictor/
├── data/
│   └── data_loader.py         # FastF1 data fetching
├── features/
│   └── feature_engineering.py # Feature engineering (16 features)
├── models/
│   ├── train.py               # Train all 3 models
│   ├── predict.py             # Make predictions
│   └── saved/                 # Trained model files (.pkl)
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── cache/                     # FastF1 cache (auto-created)
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone and install
git clone https://github.com/yourusername/f1-predictor
cd f1-predictor
pip install -r requirements.txt

# 2. Load data (this takes ~30–60 min first time due to FastF1 caching)
python data/data_loader.py

# 3. Train models
python models/train.py

# 4. Launch dashboard
streamlit run dashboard/app.py
```

## Features Used (16 total)

| Feature | Description |
|---|---|
| QualiPosition | Grid starting position |
| QualiGapToPole | Gap to pole in quali positions |
| LapTimeDelta | Avg lap time vs field average |
| BestLapDelta | Best lap vs field best |
| TotalPitStops | Number of pit stops |
| RollingAvgPos | Rolling avg position (last 3 races) |
| RollingTeamAvgPos | Rolling avg team position |
| WinRate | Career win rate up to that point |
| TrackFamiliarity | Times driver raced at this track |
| TeamEncoded | Team label encoded |
| DriverEncoded | Driver label encoded |
| AvgTrackTemp | Average track temperature |
| AvgAirTemp | Average air temperature |
| AvgRainfall | Average rainfall |
| AvgWindSpeed | Average wind speed |
| IsWet | Binary wet/dry race flag |

## Models

- **XGBoost** — gradient boosted trees, best overall performer
- **Random Forest** — ensemble method, robust baseline
- **Neural Net** — MLP with 3 hidden layers (128→64→32)

Each model predicts 3 targets: **Winner**, **Podium (top 3)**, **DNF**

## Dashboard Features

- Race selector (year + round)
- Model switcher (XGBoost / RF / Neural Net)
- Predicted podium with confidence %
- DNF risk per driver
- Win probability bar chart (top 10)
- Full grid predictions table
- Model comparison (Accuracy, F1, AUC)

## Tech Stack

`FastF1` · `XGBoost` · `Scikit-learn` · `Pandas` · `Streamlit` · `Plotly`
