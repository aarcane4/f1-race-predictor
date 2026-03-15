"""
F1 2026 Auto Pipeline
=====================
Run this script after each 2026 race weekend.
It will:
  1. Fetch completed 2026 races from FastF1
  2. Append new data to f1_raw_data.csv
  3. Retrain XGBoost + RandomForest models
  4. Print prediction for the next upcoming race

Usage:
    python auto_pipeline.py            # fetch all completed 2026 races + retrain
    python auto_pipeline.py --predict  # just predict next race (no fetch/retrain)
    python auto_pipeline.py --force    # re-fetch all 2026 rounds even if cached
"""

import os, sys, time, argparse
import pandas as pd
import numpy as np
import fastf1
import joblib

sys.path.append(os.path.dirname(__file__))
from data.data_loader     import load_race_results, load_qualifying_results, \
                                  load_lap_data, load_weather_data, DELAY_BETWEEN_ROUNDS
from features.feature_engineering import engineer_features, get_feature_columns
from models.train          import train_models
from models.predict        import load_model, load_scaler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'f1_raw_data.csv')
CACHE_DIR  = os.path.join(BASE_DIR, 'cache')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved')

fastf1.Cache.enable_cache(CACHE_DIR)

# ── 2026 Full Schedule ─────────────────────────────────────────────────────────
SCHEDULE_2026 = [
    {'Round':1,  'RaceName':'Australian Grand Prix',     'Date':'2026-03-15'},
    {'Round':2,  'RaceName':'Chinese Grand Prix',        'Date':'2026-03-22'},
    {'Round':3,  'RaceName':'Japanese Grand Prix',       'Date':'2026-03-29'},
    {'Round':4,  'RaceName':'Bahrain Grand Prix',        'Date':'2026-04-19'},
    {'Round':5,  'RaceName':'Saudi Arabian Grand Prix',  'Date':'2026-04-26'},
    {'Round':6,  'RaceName':'Miami Grand Prix',          'Date':'2026-05-10'},
    {'Round':7,  'RaceName':'Emilia Romagna Grand Prix', 'Date':'2026-05-24'},
    {'Round':8,  'RaceName':'Monaco Grand Prix',         'Date':'2026-05-31'},
    {'Round':9,  'RaceName':'Spanish Grand Prix',        'Date':'2026-06-14'},
    {'Round':10, 'RaceName':'Canadian Grand Prix',       'Date':'2026-06-21'},
    {'Round':11, 'RaceName':'Austrian Grand Prix',       'Date':'2026-07-05'},
    {'Round':12, 'RaceName':'British Grand Prix',        'Date':'2026-07-19'},
    {'Round':13, 'RaceName':'Belgian Grand Prix',        'Date':'2026-08-02'},
    {'Round':14, 'RaceName':'Hungarian Grand Prix',      'Date':'2026-08-23'},
    {'Round':15, 'RaceName':'Dutch Grand Prix',          'Date':'2026-08-30'},
    {'Round':16, 'RaceName':'Italian Grand Prix',        'Date':'2026-09-06'},
    {'Round':17, 'RaceName':'Azerbaijan Grand Prix',     'Date':'2026-09-20'},
    {'Round':18, 'RaceName':'Singapore Grand Prix',      'Date':'2026-10-04'},
    {'Round':19, 'RaceName':'United States Grand Prix',  'Date':'2026-10-18'},
    {'Round':20, 'RaceName':'Mexico City Grand Prix',    'Date':'2026-10-25'},
    {'Round':21, 'RaceName':'Sao Paulo Grand Prix',      'Date':'2026-11-08'},
    {'Round':22, 'RaceName':'Las Vegas Grand Prix',      'Date':'2026-11-22'},
    {'Round':23, 'RaceName':'Qatar Grand Prix',          'Date':'2026-11-29'},
    {'Round':24, 'RaceName':'Abu Dhabi Grand Prix',      'Date':'2026-12-06'},
]

DRIVERS_2026 = [
    {'Abbreviation':'RUS','FullName':'George Russell',    'TeamName':'Mercedes',        'DriverNumber':'63'},
    {'Abbreviation':'ANT','FullName':'Kimi Antonelli',    'TeamName':'Mercedes',        'DriverNumber':'12'},
    {'Abbreviation':'LEC','FullName':'Charles Leclerc',   'TeamName':'Ferrari',         'DriverNumber':'16'},
    {'Abbreviation':'HAM','FullName':'Lewis Hamilton',    'TeamName':'Ferrari',         'DriverNumber':'44'},
    {'Abbreviation':'NOR','FullName':'Lando Norris',      'TeamName':'McLaren',         'DriverNumber':'4'},
    {'Abbreviation':'PIA','FullName':'Oscar Piastri',     'TeamName':'McLaren',         'DriverNumber':'81'},
    {'Abbreviation':'VER','FullName':'Max Verstappen',    'TeamName':'Red Bull Racing', 'DriverNumber':'1'},
    {'Abbreviation':'LAW','FullName':'Liam Lawson',       'TeamName':'Red Bull Racing', 'DriverNumber':'30'},
    {'Abbreviation':'ALO','FullName':'Fernando Alonso',   'TeamName':'Aston Martin',    'DriverNumber':'14'},
    {'Abbreviation':'STR','FullName':'Lance Stroll',      'TeamName':'Aston Martin',    'DriverNumber':'18'},
    {'Abbreviation':'GAS','FullName':'Pierre Gasly',      'TeamName':'Alpine',          'DriverNumber':'10'},
    {'Abbreviation':'DOO','FullName':'Jack Doohan',       'TeamName':'Alpine',          'DriverNumber':'7'},
    {'Abbreviation':'ALB','FullName':'Alexander Albon',   'TeamName':'Williams',        'DriverNumber':'23'},
    {'Abbreviation':'SAI','FullName':'Carlos Sainz',      'TeamName':'Williams',        'DriverNumber':'55'},
    {'Abbreviation':'HUL','FullName':'Nico Hulkenberg',   'TeamName':'Audi',            'DriverNumber':'27'},
    {'Abbreviation':'BOR','FullName':'Gabriel Bortoleto', 'TeamName':'Audi',            'DriverNumber':'5'},
    {'Abbreviation':'TSU','FullName':'Yuki Tsunoda',      'TeamName':'Racing Bulls',    'DriverNumber':'22'},
    {'Abbreviation':'HAD','FullName':'Isack Hadjar',      'TeamName':'Racing Bulls',    'DriverNumber':'6'},
    {'Abbreviation':'OCO','FullName':'Esteban Ocon',      'TeamName':'Haas F1 Team',    'DriverNumber':'31'},
    {'Abbreviation':'BEA','FullName':'Oliver Bearman',    'TeamName':'Haas F1 Team',    'DriverNumber':'87'},
    {'Abbreviation':'PER','FullName':'Sergio Perez',      'TeamName':'Cadillac',        'DriverNumber':'11'},
    {'Abbreviation':'BOT','FullName':'Valtteri Bottas',   'TeamName':'Cadillac',        'DriverNumber':'77'},
]

TEAM_FORM = {
    'Red Bull Racing':1.9,'Ferrari':2.8,'McLaren':3.1,'Mercedes':3.8,
    'Aston Martin':6.2,'Alpine':7.8,'Williams':8.5,'Racing Bulls':8.9,
    'Haas F1 Team':11.2,'Audi':12.1,'Cadillac':13.5,
}
DRIVER_FORM = {
    'VER':1.8,'LEC':3.2,'NOR':3.5,'HAM':3.8,'RUS':4.1,'PIA':4.3,
    'SAI':4.8,'ALO':5.2,'ANT':5.5,'GAS':7.1,'TSU':7.8,'STR':8.2,
    'LAW':8.5,'OCO':9.1,'ALB':9.4,'HUL':9.8,'DOO':11.2,'BOR':11.8,
    'HAD':12.1,'BEA':13.0,'PER':7.5,'BOT':10.2,
}


# ── Step 1: Fetch completed 2026 races ────────────────────────────────────────
def fetch_2026_races(force=False):
    from datetime import date
    today = date.today()

    existing = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else pd.DataFrame()
    already_fetched = set()
    if not existing.empty and 'Year' in existing.columns:
        done = existing[existing['Year']==2026]
        already_fetched = set(done['Round'].unique())

    new_data = []
    completed = [r for r in SCHEDULE_2026
                 if pd.to_datetime(r['Date']).date() < today]

    if not completed:
        print("No 2026 races completed yet.")
        return existing

    print(f"\nFound {len(completed)} completed 2026 races.")
    for race in completed:
        rnd = race['Round']
        if rnd in already_fetched and not force:
            print(f"  Round {rnd:02d} ({race['RaceName']}) — already in dataset, skipping")
            continue

        print(f"  Round {rnd:02d} ({race['RaceName']}) — fetching...", end=' ', flush=True)
        race_df = load_race_results(2026, rnd)
        if race_df.empty:
            print("failed")
            continue

        quali_df = load_qualifying_results(2026, rnd)
        lap_df   = load_lap_data(2026, rnd)
        weather  = load_weather_data(2026, rnd)

        merged = race_df.copy()
        if not quali_df.empty and 'QualiPosition' in quali_df.columns:
            merged = merged.merge(quali_df[['Abbreviation','QualiPosition']],
                                  on='Abbreviation', how='left')
        else:
            merged['QualiPosition'] = merged['GridPosition']

        if not lap_df.empty and 'Abbreviation' in lap_df.columns:
            lap_cols = [c for c in ['Abbreviation','AvgLapTime','BestLapTime','TotalPitStops']
                        if c in lap_df.columns]
            merged = merged.merge(lap_df[lap_cols], on='Abbreviation', how='left')
        else:
            merged['AvgLapTime']    = pd.NA
            merged['BestLapTime']   = pd.NA
            merged['TotalPitStops'] = pd.NA

        for k, v in weather.items():
            merged[k] = v

        new_data.append(merged)
        print("done")
        time.sleep(DELAY_BETWEEN_ROUNDS)

    if not new_data:
        print("No new 2026 data fetched.")
        return existing

    new_df = pd.concat(new_data, ignore_index=True)

    # Remove old 2026 entries if force
    if force and not existing.empty:
        existing = existing[existing['Year'] != 2026]

    # Remove already fetched rounds from existing to avoid dupes
    if not existing.empty and not new_df.empty:
        new_rounds = set(new_df['Round'].unique())
        existing = existing[~((existing['Year']==2026) & (existing['Round'].isin(new_rounds)))]

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values(['Year','Round','Position']).reset_index(drop=True)
    combined.to_csv(DATA_PATH, index=False)

    years = sorted(combined['Year'].unique())
    print(f"\nDataset updated: {len(combined)} rows | Years: {years}")
    return combined


# ── Step 2: Retrain with 2 models only ────────────────────────────────────────
def retrain(df):
    print("\nRetraining XGBoost + RandomForest...")
    train_models(df=df)
    print("Retraining complete!")


# ── Step 3: Predict next upcoming race ────────────────────────────────────────
def get_next_race():
    from datetime import date
    today = date.today()
    upcoming = [r for r in SCHEDULE_2026
                if pd.to_datetime(r['Date']).date() >= today]
    return upcoming[0] if upcoming else None


def predict_next_race(df, model_name='XGBoost'):
    next_race = get_next_race()
    if not next_race:
        print("No upcoming 2026 races.")
        return

    print(f"\n{'='*60}")
    print(f"  PREDICTION: {next_race['RaceName'].upper()}")
    print(f"  Date: {next_race['Date']} · Round {next_race['Round']:02d}/24")
    print(f"{'='*60}")

    # Build driver form from 2026 data so far
    data_2026 = df[df['Year']==2026].copy() if not df.empty else pd.DataFrame()
    driver_recent_form = {}
    if not data_2026.empty:
        form = data_2026.groupby('Abbreviation').agg(
            avg_pos   =('Position','mean'),
            win_rate  =('Winner','mean'),
            podium_rate=('Podium','mean'),
            dnf_rate  =('DNF','mean'),
        ).reset_index()
        for _, r in form.iterrows():
            driver_recent_form[r['Abbreviation']] = r.to_dict()

    # Default quali order (update after Saturday qualifying!)
    quali_positions = {d['Abbreviation']: i+1 for i,d in enumerate(DRIVERS_2026)}

    # Check if quali file exists for this round
    quali_path = os.path.join(BASE_DIR, 'data',
                              f'quali_2026_r{next_race["Round"]:02d}.csv')
    if os.path.exists(quali_path):
        qdf = pd.read_csv(quali_path)
        for _, row in qdf.iterrows():
            if row['Abbreviation'] in quali_positions:
                quali_positions[row['Abbreviation']] = int(row['QualiPosition'])
        print(f"  Using qualifying data from {quali_path}")
    else:
        print(f"  Using estimated qualifying order")
        print(f"  Tip: Save actual quali results to data/quali_2026_r{next_race['Round']:02d}.csv")
        print(f"  Format: Abbreviation,QualiPosition")

    # Score-based prediction
    rows = []
    for d in DRIVERS_2026:
        abbr = d['Abbreviation']
        qpos = quali_positions.get(abbr, 15)
        form = driver_recent_form.get(abbr, {})

        avg_pos    = form.get('avg_pos',    DRIVER_FORM.get(abbr, 12.0))
        win_rate   = form.get('win_rate',   max(0, .05 - DRIVER_FORM.get(abbr,12)*.004))
        podium_rate= form.get('podium_rate',max(0, .25 - DRIVER_FORM.get(abbr,12)*.018))
        dnf_rate   = form.get('dnf_rate',   0.12)
        team_form  = TEAM_FORM.get(d['TeamName'], 10.0)
        pole       = min(quali_positions.values())

        rows.append({
            'Abbreviation':abbr, 'FullName':d['FullName'],
            'TeamName':d['TeamName'],
            'QualiPosition':qpos,
            'score': qpos*3.0 + avg_pos*1.5 + team_form*1.0 +
                     (qpos-pole)*0.8 - win_rate*20.0 - podium_rate*5.0,
            'dnf_rate': dnf_rate,
        })

    result = pd.DataFrame(rows)
    result['snorm'] = (result['score'] - result['score'].min()) / \
                      (result['score'].max() - result['score'].min() + 1e-9)
    result['inv']   = 1.0 / (result['snorm'] + 0.08)
    result['Win%']  = (result['inv'] / result['inv'].sum() * 100).round(1)
    result['Pod%']  = (result['Win%'] * 2.8).clip(upper=88).round(1)
    result['DNF%']  = (result['dnf_rate'] * 100).clip(4, 60).round(1)
    result = result.sort_values('Win%', ascending=False).reset_index(drop=True)

    print(f"\n  {'#':<4} {'Driver':<22} {'Team':<20} {'Win%':>6} {'Pod%':>6} {'DNF%':>6}")
    print(f"  {'-'*64}")
    medals = ['🥇','🥈','🥉']
    for i, row in result.iterrows():
        medal = medals[i] if i < 3 else f"P{i+1:<2}"
        print(f"  {medal}  {row['FullName']:<22} {row['TeamName']:<20} "
              f"{row['Win%']:>5.1f}% {row['Pod%']:>5.1f}% {row['DNF%']:>5.1f}%")

    # Save prediction to file
    pred_path = os.path.join(BASE_DIR, 'data',
                             f'prediction_2026_r{next_race["Round"]:02d}.csv')
    result[['Abbreviation','FullName','TeamName',
            'QualiPosition','Win%','Pod%','DNF%']].to_csv(pred_path, index=False)
    print(f"\n  Prediction saved to {pred_path}")
    print(f"\n  Update quali: data/quali_2026_r{next_race['Round']:02d}.csv")
    print(f"  Then re-run:  python auto_pipeline.py --predict")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='F1 2026 Auto Pipeline')
    parser.add_argument('--predict', action='store_true',
                        help='Only predict next race, skip fetch/retrain')
    parser.add_argument('--force', action='store_true',
                        help='Re-fetch all 2026 data even if already cached')
    args = parser.parse_args()

    print("=" * 60)
    print("  F1 2026 AUTO PIPELINE")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else pd.DataFrame()

    if not args.predict:
        print("\nStep 1: Fetching 2026 race data...")
        df = fetch_2026_races(force=args.force)

        print("\nStep 2: Retraining models...")
        if len(df) > 100:
            retrain(df)
        else:
            print("Not enough data to retrain — using existing models")

    print("\nStep 3: Predicting next race...")
    predict_next_race(df)

    print("\n" + "="*60)
    print("  Done! Launch dashboard: streamlit run dashboard/app.py")
    print("="*60)