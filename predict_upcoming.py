import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))

from features.feature_engineering import engineer_features, get_feature_columns
from models.predict import load_model, load_scaler

# ── 2026 Driver & Team Info ───────────────────────────────────────────────────
DRIVERS_2026 = [
    {'Abbreviation': 'RUS', 'FullName': 'George Russell',      'TeamName': 'Mercedes',          'DriverNumber': '63'},
    {'Abbreviation': 'ANT', 'FullName': 'Kimi Antonelli',      'TeamName': 'Mercedes',          'DriverNumber': '12'},
    {'Abbreviation': 'LEC', 'FullName': 'Charles Leclerc',     'TeamName': 'Ferrari',           'DriverNumber': '16'},
    {'Abbreviation': 'HAM', 'FullName': 'Lewis Hamilton',      'TeamName': 'Ferrari',           'DriverNumber': '44'},
    {'Abbreviation': 'NOR', 'FullName': 'Lando Norris',        'TeamName': 'McLaren',           'DriverNumber': '4'},
    {'Abbreviation': 'PIA', 'FullName': 'Oscar Piastri',       'TeamName': 'McLaren',           'DriverNumber': '81'},
    {'Abbreviation': 'VER', 'FullName': 'Max Verstappen',      'TeamName': 'Red Bull Racing',   'DriverNumber': '1'},
    {'Abbreviation': 'LAW', 'FullName': 'Liam Lawson',         'TeamName': 'Red Bull Racing',   'DriverNumber': '30'},
    {'Abbreviation': 'ALO', 'FullName': 'Fernando Alonso',     'TeamName': 'Aston Martin',      'DriverNumber': '14'},
    {'Abbreviation': 'STR', 'FullName': 'Lance Stroll',        'TeamName': 'Aston Martin',      'DriverNumber': '18'},
    {'Abbreviation': 'GAS', 'FullName': 'Pierre Gasly',        'TeamName': 'Alpine',            'DriverNumber': '10'},
    {'Abbreviation': 'DOO', 'FullName': 'Jack Doohan',         'TeamName': 'Alpine',            'DriverNumber': '7'},
    {'Abbreviation': 'ALB', 'FullName': 'Alexander Albon',     'TeamName': 'Williams',          'DriverNumber': '23'},
    {'Abbreviation': 'SAI', 'FullName': 'Carlos Sainz',        'TeamName': 'Williams',          'DriverNumber': '55'},
    {'Abbreviation': 'HUL', 'FullName': 'Nico Hulkenberg',     'TeamName': 'Audi',              'DriverNumber': '27'},
    {'Abbreviation': 'BOR', 'FullName': 'Gabriel Bortoleto',   'TeamName': 'Audi',              'DriverNumber': '5'},
    {'Abbreviation': 'TSU', 'FullName': 'Yuki Tsunoda',        'TeamName': 'Racing Bulls',      'DriverNumber': '22'},
    {'Abbreviation': 'HAD', 'FullName': 'Isack Hadjar',        'TeamName': 'Racing Bulls',      'DriverNumber': '6'},
    {'Abbreviation': 'OCO', 'FullName': 'Esteban Ocon',        'TeamName': 'Haas F1 Team',      'DriverNumber': '31'},
    {'Abbreviation': 'BEA', 'FullName': 'Oliver Bearman',      'TeamName': 'Haas F1 Team',      'DriverNumber': '87'},
    {'Abbreviation': 'PER', 'FullName': 'Sergio Perez',        'TeamName': 'Cadillac',          'DriverNumber': '11'},
    {'Abbreviation': 'BOT', 'FullName': 'Valtteri Bottas',     'TeamName': 'Cadillac',          'DriverNumber': '77'},
]

# ── 2026 results so far (for rolling form) ────────────────────────────────────
RESULTS_2026 = [
    # Round 1 — Australia
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'RUS', 'Position': 1,  'GridPosition': 1,  'Status': 'Finished', 'Points': 25},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'ANT', 'Position': 2,  'GridPosition': 2,  'Status': 'Finished', 'Points': 18},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'LEC', 'Position': 3,  'GridPosition': 4,  'Status': 'Finished', 'Points': 15},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'NOR', 'Position': 4,  'GridPosition': 3,  'Status': 'Finished', 'Points': 12},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'PIA', 'Position': 5,  'GridPosition': 6,  'Status': 'Finished', 'Points': 10},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'HAM', 'Position': 6,  'GridPosition': 5,  'Status': 'Finished', 'Points': 8},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'VER', 'Position': 7,  'GridPosition': 8,  'Status': 'Finished', 'Points': 6},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'ALO', 'Position': 8,  'GridPosition': 7,  'Status': 'Finished', 'Points': 4},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'SAI', 'Position': 9,  'GridPosition': 9,  'Status': 'Finished', 'Points': 2},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'GAS', 'Position': 10, 'GridPosition': 11, 'Status': 'Finished', 'Points': 1},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'LAW', 'Position': 11, 'GridPosition': 10, 'Status': 'Finished', 'Points': 0},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'TSU', 'Position': 12, 'GridPosition': 13, 'Status': 'Finished', 'Points': 0},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'STR', 'Position': 13, 'GridPosition': 14, 'Status': 'Finished', 'Points': 0},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'HUL', 'Position': 14, 'GridPosition': 15, 'Status': 'Finished', 'Points': 0},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'OCO', 'Position': 15, 'GridPosition': 16, 'Status': 'Finished', 'Points': 0},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'ALB', 'Position': 16, 'GridPosition': 17, 'Status': 'Finished', 'Points': 0},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'BOR', 'Position': 17, 'GridPosition': 18, 'Status': 'Finished', 'Points': 0},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'HAD', 'Position': 18, 'GridPosition': 19, 'Status': 'Finished', 'Points': 0},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'BEA', 'Position': 19, 'GridPosition': 20, 'Status': 'Finished', 'Points': 0},
    {'Year': 2026, 'Round': 1, 'RaceName': 'Australian Grand Prix',
     'Abbreviation': 'PER', 'Position': 20, 'GridPosition': 21, 'Status': 'Finished', 'Points': 0},
]

# ── Japanese GP 2026 — estimated quali (update after March 28 qualifying) ─────
JAPAN_2026_QUALI = [
    {'Abbreviation': 'RUS', 'QualiPosition': 1},
    {'Abbreviation': 'ANT', 'QualiPosition': 2},
    {'Abbreviation': 'NOR', 'QualiPosition': 3},
    {'Abbreviation': 'LEC', 'QualiPosition': 4},
    {'Abbreviation': 'VER', 'QualiPosition': 5},
    {'Abbreviation': 'HAM', 'QualiPosition': 6},
    {'Abbreviation': 'PIA', 'QualiPosition': 7},
    {'Abbreviation': 'SAI', 'QualiPosition': 8},
    {'Abbreviation': 'ALO', 'QualiPosition': 9},
    {'Abbreviation': 'GAS', 'QualiPosition': 10},
    {'Abbreviation': 'LAW', 'QualiPosition': 11},
    {'Abbreviation': 'TSU', 'QualiPosition': 12},
    {'Abbreviation': 'STR', 'QualiPosition': 13},
    {'Abbreviation': 'HUL', 'QualiPosition': 14},
    {'Abbreviation': 'OCO', 'QualiPosition': 15},
    {'Abbreviation': 'ALB', 'QualiPosition': 16},
    {'Abbreviation': 'BOR', 'QualiPosition': 17},
    {'Abbreviation': 'HAD', 'QualiPosition': 18},
    {'Abbreviation': 'BEA', 'QualiPosition': 19},
    {'Abbreviation': 'PER', 'QualiPosition': 20},
    {'Abbreviation': 'DOO', 'QualiPosition': 21},
    {'Abbreviation': 'BOT', 'QualiPosition': 22},
]

# ── Suzuka weather forecast ───────────────────────────────────────────────────
JAPAN_WEATHER = {
    'AvgTrackTemp': 22.0,
    'AvgAirTemp':   16.0,
    'AvgRainfall':  0.0,
    'AvgWindSpeed': 12.0,
}


def build_prediction_input(race_name, round_number, quali_order, weather,
                            historical_df, year=2026):
    """Build input dataframe for prediction using historical + current season data."""

    # Load historical data for feature engineering context
    hist = historical_df.copy()

    # Add 2026 Round 1 results to help rolling form
    results_2026 = pd.DataFrame(RESULTS_2026)
    results_2026['DNF']    = 0
    results_2026['Winner'] = (results_2026['Position'] == 1).astype(int)
    results_2026['Podium'] = (results_2026['Position'] <= 3).astype(int)
    results_2026['FullName'] = results_2026['Abbreviation'].map(
        {d['Abbreviation']: d['FullName'] for d in DRIVERS_2026})
    results_2026['DriverNumber'] = results_2026['Abbreviation'].map(
        {d['Abbreviation']: d['DriverNumber'] for d in DRIVERS_2026})
    results_2026['AvgLapTime']    = hist['AvgLapTime'].median()
    results_2026['BestLapTime']   = hist['BestLapTime'].median()
    results_2026['TotalPitStops'] = 2
    results_2026['QualiPosition'] = results_2026['GridPosition']
    for k, v in weather.items():
        results_2026[k] = v

    combined = pd.concat([hist, results_2026], ignore_index=True)

    # Build upcoming race rows
    quali_df = pd.DataFrame(quali_order)
    rows = []
    for driver in DRIVERS_2026:
        q = next((x for x in quali_order if x['Abbreviation'] == driver['Abbreviation']), None)
        quali_pos = q['QualiPosition'] if q else 20
        rows.append({
            'Year':           year,
            'Round':          round_number,
            'RaceName':       race_name,
            'DriverNumber':   driver['DriverNumber'],
            'Abbreviation':   driver['Abbreviation'],
            'FullName':       driver['FullName'],
            'TeamName':       driver['TeamName'],
            'GridPosition':   quali_pos,
            'QualiPosition':  quali_pos,
            'Position':       np.nan,
            'Status':         'Unknown',
            'Points':         0,
            'DNF':            0,
            'Winner':         0,
            'Podium':         0,
            'AvgLapTime':     combined['AvgLapTime'].median(),
            'BestLapTime':    combined['BestLapTime'].median(),
            'TotalPitStops':  2,
            **weather,
        })

    upcoming_df = pd.DataFrame(rows)
    full_df = pd.concat([combined, upcoming_df], ignore_index=True)
    full_df = engineer_features(full_df)

    # Return only the upcoming race rows
    result = full_df[full_df['Round'] == round_number][full_df['Year'] == year].copy()
    return result


def predict_race(input_df, model_name='XGBoost'):
    """Run predictions for all 3 targets."""
    features = get_feature_columns()
    X = input_df[features]

    results = input_df[['Abbreviation', 'FullName', 'TeamName', 'QualiPosition']].copy()

    for target in ['Winner', 'Podium', 'DNF']:
        model = load_model(model_name, target)
        if model is None:
            print(f"Model not found: {model_name}_{target}. Run train.py first.")
            results[f'{target}_Prob'] = 0.0
            continue

        if model_name == 'NeuralNet':
            scaler = load_scaler(target)
            X_in = scaler.transform(X) if scaler else X
        else:
            X_in = X

        proba = model.predict_proba(X_in)[:, 1]
        results[f'{target}_Prob'] = (proba * 100).round(1)

    results = results.sort_values('Winner_Prob', ascending=False).reset_index(drop=True)
    results.index += 1
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("🏎️  F1 2026 — Japanese GP Prediction")
    print("   Suzuka · March 29, 2026")
    print("=" * 60)

    # Load historical data
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'f1_raw_data.csv')
    if not os.path.exists(data_path):
        print("ERROR: data/f1_raw_data.csv not found. Run data_loader.py first.")
        sys.exit(1)

    hist_df = pd.read_csv(data_path)
    print(f"Loaded {len(hist_df)} historical records (2022–2025)")

    for model_name in ['XGBoost', 'RandomForest', 'NeuralNet']:
        print(f"\n{'─'*60}")
        print(f"Model: {model_name}")
        print(f"{'─'*60}")

        input_df = build_prediction_input(
            race_name='Japanese Grand Prix',
            round_number=3,
            quali_order=JAPAN_2026_QUALI,
            weather=JAPAN_WEATHER,
            historical_df=hist_df,
        )

        preds = predict_race(input_df, model_name=model_name)

        print(f"\n🏆 Predicted Podium:")
        medals = ['🥇', '🥈', '🥉']
        for i, row in preds.head(3).iterrows():
            medal = medals[i-1] if i <= 3 else f'P{i}'
            print(f"  {medal} {row['FullName']:<22} ({row['TeamName']:<18}) "
                  f"Win: {row['Winner_Prob']:>5.1f}%  "
                  f"Podium: {row['Podium_Prob']:>5.1f}%")

        print(f"\n🚨 Top DNF Risks:")
        dnf = preds.sort_values('DNF_Prob', ascending=False).head(3)
        for _, row in dnf.iterrows():
            print(f"  ⚠ {row['Abbreviation']}  DNF: {row['DNF_Prob']:.1f}%")

    print(f"\n{'='*60}")
    print("⚠  NOTE: Quali positions are estimates — update")
    print("   JAPAN_2026_QUALI after March 28 qualifying!")
    print("   2026 regulations are new — treat predictions")
    print("   as indicative, not definitive.")
    print("=" * 60)