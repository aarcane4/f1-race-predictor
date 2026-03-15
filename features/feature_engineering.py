import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all ML features from raw F1 data."""
    df = df.copy()

    # --- 1. Qualifying gap to pole ---
    pole_times = df.groupby(['Year', 'Round'])['QualiPosition'].transform(
        lambda x: x.min()
    )
    df['QualiGapToPole'] = df['QualiPosition'] - pole_times

    # --- 2. Lap time delta vs field average ---
    field_avg = df.groupby(['Year', 'Round'])['AvgLapTime'].transform('mean')
    df['LapTimeDelta'] = df['AvgLapTime'] - field_avg

    # --- 3. Best lap time delta ---
    field_best = df.groupby(['Year', 'Round'])['BestLapTime'].transform('mean')
    df['BestLapDelta'] = df['BestLapTime'] - field_best

    # --- 4. Rolling driver form (avg position last 3 races) ---
    df = df.sort_values(['Abbreviation', 'Year', 'Round'])
    df['RollingAvgPos'] = (
        df.groupby('Abbreviation')['Position']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # --- 5. Rolling team form ---
    df['RollingTeamAvgPos'] = (
        df.groupby('TeamName')['Position']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # --- 6. Driver win rate (career up to that point) ---
    df['CumulativeWins'] = (
        df.groupby('Abbreviation')['Winner']
        .transform(lambda x: x.shift(1).expanding().sum())
    )
    df['CumulativeRaces'] = (
        df.groupby('Abbreviation')['Winner']
        .transform(lambda x: x.shift(1).expanding().count())
    )
    df['WinRate'] = df['CumulativeWins'] / (df['CumulativeRaces'] + 1e-6)

    # --- 7. Track familiarity (number of times driver raced here) ---
    df['TrackFamiliarity'] = (
        df.groupby(['Abbreviation', 'RaceName']).cumcount()
    )

    # --- 8. Encode team and driver ---
    le_team = LabelEncoder()
    le_driver = LabelEncoder()
    df['TeamEncoded'] = le_team.fit_transform(df['TeamName'].fillna('Unknown'))
    df['DriverEncoded'] = le_driver.fit_transform(df['Abbreviation'].fillna('Unknown'))

    # --- 9. Weather features ---
    df['AvgTrackTemp'] = df['AvgTrackTemp'].fillna(df['AvgTrackTemp'].median())
    df['AvgAirTemp'] = df['AvgAirTemp'].fillna(df['AvgAirTemp'].median())
    df['AvgRainfall'] = df['AvgRainfall'].fillna(0)
    df['AvgWindSpeed'] = df['AvgWindSpeed'].fillna(df['AvgWindSpeed'].median())
    df['IsWet'] = (df['AvgRainfall'] > 0).astype(int)

    # --- 10. Fill missing values ---
    num_cols = ['QualiPosition', 'AvgLapTime', 'BestLapTime', 'TotalPitStops',
                'LapTimeDelta', 'BestLapDelta', 'RollingAvgPos',
                'RollingTeamAvgPos', 'WinRate', 'TrackFamiliarity']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def get_feature_columns() -> list:
    """Return the list of feature columns used for training."""
    return [
        'QualiPosition',
        'QualiGapToPole',
        'LapTimeDelta',
        'BestLapDelta',
        'TotalPitStops',
        'RollingAvgPos',
        'RollingTeamAvgPos',
        'WinRate',
        'TrackFamiliarity',
        'TeamEncoded',
        'DriverEncoded',
        'AvgTrackTemp',
        'AvgAirTemp',
        'AvgRainfall',
        'AvgWindSpeed',
        'IsWet',
    ]


def get_targets() -> list:
    return ['Winner', 'Podium', 'DNF']


if __name__ == '__main__':
    df = pd.read_csv('../data/f1_raw_data.csv')
    df_features = engineer_features(df)
    df_features.to_csv('../data/f1_features.csv', index=False)
    print(f"Features saved. Shape: {df_features.shape}")
    print(f"Feature columns: {get_feature_columns()}")
