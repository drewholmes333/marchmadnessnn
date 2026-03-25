import pandas as pd
from data_loader import *

hist = get_historical_stats()
print("HIST:", len(hist), "Unique Years:", hist['YEAR'].unique())

matchups_path = '../../Tournament Matchups.csv'
tourney = pd.read_csv(matchups_path)
tourney = tourney[tourney['SCORE'].notna()].reset_index(drop=True)
print("TOURNEY GAMES ROWS:", len(tourney))

games_even = tourney.iloc[::2].reset_index(drop=True)
games_odd = tourney.iloc[1::2].reset_index(drop=True)
print("GAMES:", len(games_even))

games_df = pd.DataFrame()
games_df['YEAR'] = games_even['YEAR']
games_df['TeamA_NO'] = games_even['TEAM NO']
games_df['TeamB_NO'] = games_odd['TEAM NO']

m1 = pd.merge(games_df, hist, left_on=['YEAR', 'TeamA_NO'], right_on=['YEAR', 'TEAM NO'], how='inner')
print("MERGE 1:", len(m1))

m2 = pd.merge(m1, hist, left_on=['YEAR', 'TeamB_NO'], right_on=['YEAR', 'TEAM NO'], how='inner', suffixes=('_A', '_B'))
print("MERGE 2:", len(m2))
