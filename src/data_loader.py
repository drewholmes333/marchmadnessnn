import pandas as pd
import numpy as np
import os
import torch
import warnings
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _read_csv_smart(filepath):
    temp_df = pd.read_csv(filepath, header=None, low_memory=False)
    header_idx = temp_df[temp_df.eq('Team').any(axis=1)].index[0]
    df = pd.read_csv(filepath, skiprows=header_idx)
    df = df[df['Team'] != 'Team'].dropna(subset=['Team'])
    return df

def get_base_stats():
    filepath_kp = os.path.join(BASE_DIR, 'data', 'raw', 'kenpom.csv')
    if not os.path.exists(filepath_kp):
        filepath_kp = os.path.join(BASE_DIR, 'kenpom.csv')
        
    filepath_tv = os.path.join(BASE_DIR, 'data', 'raw', 'torvik.csv')
    if not os.path.exists(filepath_tv):
        filepath_tv = os.path.join(BASE_DIR, 'torvik.csv')

    kenpom_df = _read_csv_smart(filepath_kp)
    torvik_df = _read_csv_smart(filepath_tv)
    
    kenpom_df['Seed'] = kenpom_df['Team'].astype(str).str.extract(r'\s+(\d+)$').astype(float)
    kenpom_df['Team'] = kenpom_df['Team'].astype(str).str.replace(r'\s*\d+$', '', regex=True).str.strip()
    torvik_df['Team'] = torvik_df['Team'].astype(str).str.split('\n').str[0].str.strip()
    torvik_df['Team'] = torvik_df['Team'].str.replace(' \u2705', '', regex=False).str.strip()
    
    def clean_torvik_num(x):
        try:
            return float(str(x).split('\n')[0].strip())
        except ValueError:
            return 0.0
            
    features_to_clean = ['AdjOE', 'AdjDE', 'Barthag', 'EFG%', 'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'Adj T.', 'FTR', 'FTRD', '2P%', '2P%D', '3P%', '3P%D', '3PR', '3PRD', 'WAB']
    for f in features_to_clean:
        if f in torvik_df.columns:
            torvik_df[f] = torvik_df[f].apply(clean_torvik_num)
            
    merged_stats = pd.merge(kenpom_df, torvik_df, on='Team', how='inner', suffixes=('_kp', '_tv'))
    return merged_stats

def get_historical_stats():
    em_path = os.path.join(BASE_DIR, 'data', 'raw', 'EvanMiya.csv')
    if not os.path.exists(em_path):
        em_path = os.path.join(BASE_DIR, 'EvanMiya.csv')
        
    f38_path = os.path.join(BASE_DIR, 'data', 'raw', '538 Ratings.csv')
    if not os.path.exists(f38_path):
        f38_path = os.path.join(BASE_DIR, '538 Ratings.csv')
    
    em_df = pd.read_csv(em_path)
    f38_df = pd.read_csv(f38_path)
    
    em_df['TEAM'] = em_df['TEAM'].str.strip()
    f38_df['TEAM'] = f38_df['TEAM'].str.strip()
    
    hist_stats = pd.merge(em_df, f38_df, on=['YEAR', 'TEAM NO'], how='left', suffixes=('', '_f38'))
    # Fill in older historical rows lacking FiveThirtyEight analytics safely.
    if 'POWER RATING' in hist_stats.columns:
        hist_stats['POWER RATING'] = hist_stats['POWER RATING'].fillna(hist_stats['POWER RATING'].median())
    
    base_stats = get_base_stats()
    
    hist_stats = pd.merge(hist_stats, base_stats, left_on='TEAM', right_on='Team', how='left')
    hist_stats = hist_stats.dropna(subset=['Barthag'])
    
    return hist_stats

def load_historical_data():
    hist_stats = get_historical_stats()
    
    matchups_path = os.path.join(BASE_DIR, 'data', 'raw', 'Tournament Matchups.csv')
    if not os.path.exists(matchups_path):
        matchups_path = os.path.join(BASE_DIR, 'Tournament Matchups.csv')
    
    tourney = pd.read_csv(matchups_path)
    tourney = tourney[tourney['SCORE'].notna()].reset_index(drop=True)
    
    games_even = tourney.iloc[::2].reset_index(drop=True)
    games_odd = tourney.iloc[1::2].reset_index(drop=True)
    
    games_df = pd.DataFrame()
    games_df['YEAR'] = games_even['YEAR']
    games_df['TeamA_NO'] = games_even['TEAM NO']
    games_df['TeamB_NO'] = games_odd['TEAM NO']
    
    score_a = pd.to_numeric(games_even['SCORE'], errors='coerce')
    score_b = pd.to_numeric(games_odd['SCORE'], errors='coerce')
    games_df['TeamAwins'] = (score_a > score_b).astype(int)
    
    matchups = pd.merge(games_df, hist_stats, left_on=['YEAR', 'TeamA_NO'], right_on=['YEAR', 'TEAM NO'], how='inner')
    matchups = pd.merge(matchups, hist_stats, left_on=['YEAR', 'TeamB_NO'], right_on=['YEAR', 'TEAM NO'], how='inner', suffixes=('_A', '_B'))
    
    # ---------------------------
    # Data Augmentation: Duplicate and Swap
    # ---------------------------
    matchups_swapped = matchups.copy()
    matchups_swapped['TeamAwins'] = 1 - matchups['TeamAwins']
    
    rename_dict = {}
    for c in matchups.columns:
        if c.endswith('_A'):
            rename_dict[c] = c[:-2] + '_B'
        elif c.endswith('_B'):
            rename_dict[c] = c[:-2] + '_A'
    matchups_swapped = matchups_swapped.rename(columns=rename_dict)
    
    matchups = pd.concat([matchups, matchups_swapped], ignore_index=True)
    
    # ---------------------------
    # Mass Calculate All Difference Features
    # ---------------------------
    feature_cols = []
    
    matchups['Diff_AdjOE_A_vs_DE_B'] = matchups['AdjOE_A'] - matchups['AdjDE_B']
    matchups['Diff_AdjOE_B_vs_DE_A'] = matchups['AdjOE_B'] - matchups['AdjDE_A']
    
    matchups['Diff_Pace'] = abs(matchups['Adj T._A'] - matchups['Adj T._B'])
    matchups['Pace_Trap'] = (matchups['Diff_Pace'] > 7.0).astype(float)
    
    feature_cols.extend(['Diff_AdjOE_A_vs_DE_B', 'Diff_AdjOE_B_vs_DE_A', 'Diff_Pace', 'Pace_Trap'])
    
    whitelist = [
        'AdjOE', 'AdjDE', 'Barthag', 'EFG%', 'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'Adj T.', 
        'FTR', 'FTRD', '2P%', '2P%D', '3P%', '3P%D', '3PR', '3PRD', 'WAB', 'NetRtg', 'Luck',
        'KILLSHOTS MARGIN', 'POWER RATING'
    ]
    
    numeric_cols = hist_stats.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        # STRICT Pre-Game Stat Whitelist for preventing Data Leakage (ROUND, SEED, W/L)
        if col in whitelist:
            col_name = f"Diff_{col}"
            if col_name not in feature_cols:
                matchups[col_name] = matchups[f"{col}_A"] - matchups[f"{col}_B"]
                feature_cols.append(col_name)
    
    # Safely zero-out differential signals for historical datasets missing specific analytical columns
    matchups[feature_cols] = matchups[feature_cols].fillna(0)
    return matchups, feature_cols

def merge_and_create_features():
    matchups, feature_cols = load_historical_data()
    return matchups, feature_cols

def get_dataloaders(batch_size=32):
    from sklearn.model_selection import train_test_split
    matchups, feature_cols = merge_and_create_features()
    
    # Extend bounds strictly spanning 2015-2025 (The Knowledge Base)
    full_df = matchups[(matchups['YEAR'] >= 2015) & (matchups['YEAR'] <= 2025)]
    
    # 15% out-of-sample data independent validation hook
    train_df, test_df = train_test_split(full_df, test_size=0.15, random_state=42)
    
    print(f"Training on {len(train_df)} games. Validating on {len(test_df)} games (Combined 2015-2025 Base).")
    
    X_train_full = train_df[feature_cols].values
    y_train = train_df['TeamAwins'].values
    
    # ---------------------------
    # Random Forest Top 15 Feature Selection
    # ---------------------------
    print("Running Random Forest Regressor to extract Top 15 Features for neural network ingestion...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_full, y_train)
    
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[-15:]
    top_15_features = [feature_cols[i] for i in top_indices]
    
    # Force Volatility Metrics into the array per user requirement
    volatility_metrics = ['Diff_Pace', 'Pace_Trap', 'Diff_3PR']
    for metric in volatility_metrics:
        if metric not in top_15_features and metric in feature_cols:
            # Remove the weakest feature dynamically
            top_15_features.pop(0)
            top_15_features.append(metric)
    
    print("\nSelected Top 15 Features:")
    for f in reversed(top_15_features):
        print(f" - {f}")
        
    joblib.dump(top_15_features, os.path.join(BASE_DIR, 'top_15_features.pkl'))
    
    X_train = train_df[top_15_features].values
    X_test = test_df[top_15_features].values
    y_test = test_df['TeamAwins'].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.pkl'))
    
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, 15

if __name__ == '__main__':
    dl, vl, size = get_dataloaders()
    print(f"DataLoader initialized. Target Input size constraint locked heavily to: {size}")
