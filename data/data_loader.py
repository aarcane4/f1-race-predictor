import fastf1
import pandas as pd
import os
import time

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

DELAY_BETWEEN_ROUNDS = 3
DELAY_ON_RATE_LIMIT  = 70


def _load_with_retry(year, round_number, session_type, retries=3):
    for attempt in range(retries):
        try:
            session = fastf1.get_session(year, round_number, session_type)
            session.load(telemetry=False, weather=(session_type == 'R'), messages=False)
            return session
        except Exception as e:
            msg = str(e)
            if 'rate limit' in msg.lower() or '500 calls' in msg.lower():
                wait = DELAY_ON_RATE_LIMIT * (attempt + 1)
                print(f"\n  Rate limit hit — waiting {wait}s (retry {attempt+1}/{retries})...")
                time.sleep(wait)
            else:
                raise
    return None


def load_race_results(year, round_number):
    try:
        session = _load_with_retry(year, round_number, 'R')
        if session is None:
            return pd.DataFrame()
        results = session.results[['DriverNumber', 'Abbreviation', 'FullName',
                                    'TeamName', 'GridPosition', 'Position',
                                    'Status', 'Points']].copy()
        results['Year']     = year
        results['Round']    = round_number
        results['RaceName'] = session.event['EventName']
        results['DNF']      = results['Status'].apply(
            lambda x: 0 if x in ['Finished', '+1 Lap', '+2 Laps', '+3 Laps'] else 1)
        results['Winner']   = (results['Position'] == 1).astype(int)
        results['Podium']   = (results['Position'] <= 3).astype(int)
        return results
    except Exception as e:
        print(f"Error loading race {year} R{round_number}: {e}")
        return pd.DataFrame()


def load_qualifying_results(year, round_number):
    try:
        session = _load_with_retry(year, round_number, 'Q')
        if session is None:
            return pd.DataFrame()
        quali = session.results[['DriverNumber', 'Abbreviation', 'Position']].copy()
        quali.rename(columns={'Position': 'QualiPosition'}, inplace=True)
        quali['Year']  = year
        quali['Round'] = round_number
        return quali
    except Exception as e:
        print(f"Error loading quali {year} R{round_number}: {e}")
        return pd.DataFrame()


def load_lap_data(year, round_number):
    try:
        session = _load_with_retry(year, round_number, 'R')
        if session is None:
            return pd.DataFrame()
        laps = session.laps[['Driver', 'LapTime', 'PitOutTime']].copy()
        laps = laps[laps['LapTime'].notna()]
        laps['LapTimeSec'] = laps['LapTime'].dt.total_seconds()
        avg_lap = laps.groupby('Driver').agg(
            AvgLapTime=('LapTimeSec', 'mean'),
            BestLapTime=('LapTimeSec', 'min'),
            TotalPitStops=('PitOutTime', lambda x: x.notna().sum()),
        ).reset_index()
        avg_lap.rename(columns={'Driver': 'Abbreviation'}, inplace=True)
        avg_lap['Year']  = year
        avg_lap['Round'] = round_number
        return avg_lap
    except Exception as e:
        print(f"Error loading laps {year} R{round_number}: {e}")
        return pd.DataFrame()


def load_weather_data(year, round_number):
    try:
        session = _load_with_retry(year, round_number, 'R')
        if session is None:
            return {}
        w = session.weather_data
        return {
            'AvgTrackTemp': w['TrackTemp'].mean(),
            'AvgAirTemp':   w['AirTemp'].mean(),
            'AvgRainfall':  w['Rainfall'].mean(),
            'AvgWindSpeed': w['WindSpeed'].mean(),
        }
    except Exception as e:
        print(f"Error loading weather {year} R{round_number}: {e}")
        return {}


def load_full_dataset(years, max_rounds=22):
    all_data = []

    for year in years:
        print(f"\nLoading year {year}...")
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        rounds = [r for r in schedule['RoundNumber'].tolist() if r <= max_rounds]

        for rnd in rounds:
            print(f"  Round {rnd}...", end=' ', flush=True)

            race_df = load_race_results(year, rnd)
            if race_df.empty:
                print("skipped")
                time.sleep(DELAY_BETWEEN_ROUNDS)
                continue

            quali_df = load_qualifying_results(year, rnd)
            lap_df   = load_lap_data(year, rnd)
            weather  = load_weather_data(year, rnd)

            merged = race_df.copy()

            # Safe quali merge
            if (not quali_df.empty
                    and 'Abbreviation' in quali_df.columns
                    and 'QualiPosition' in quali_df.columns):
                merged = merged.merge(
                    quali_df[['Abbreviation', 'QualiPosition']],
                    on='Abbreviation', how='left')
            else:
                merged['QualiPosition'] = merged['GridPosition']

            # Safe lap merge
            if not lap_df.empty and 'Abbreviation' in lap_df.columns:
                lap_cols = [c for c in ['Abbreviation', 'AvgLapTime',
                                        'BestLapTime', 'TotalPitStops']
                            if c in lap_df.columns]
                merged = merged.merge(lap_df[lap_cols], on='Abbreviation', how='left')
            else:
                merged['AvgLapTime']    = pd.NA
                merged['BestLapTime']   = pd.NA
                merged['TotalPitStops'] = pd.NA

            for k, v in weather.items():
                merged[k] = v

            all_data.append(merged)
            print("done")
            time.sleep(DELAY_BETWEEN_ROUNDS)

    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal records loaded: {len(df)}")
        return df
    return pd.DataFrame()


if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'f1_raw_data.csv')

    # Load only 2025
    df_new = load_full_dataset(years=[2025], max_rounds=22)

    # Append to existing CSV instead of overwriting
    if os.path.exists(out_path):
        df_existing = pd.read_csv(out_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(out_path, index=False)
        print(f"Appended 2025 data. Total rows: {len(df_combined)}")
    else:
        df_new.to_csv(out_path, index=False)
        print(f"Saved. Total rows: {len(df_new)}")
