import pandas as pd
import numpy as np
import re
import os
import torch
import warnings
import joblib
import difflib
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import TensorDataset, DataLoader

# Internal modules
import config
import coach_engine
from utils.scraper import fetch_page
from utils import team_to_slug

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

def load_coach_data():
    """Load coach_results.csv and compute Deep Run Experience score."""
    coach_path = config.COACH_RESULTS_PATH
    if not os.path.exists(coach_path):
        logger.warning(f"Coach results not found at {coach_path}")
        return pd.DataFrame()
    df = pd.read_csv(coach_path)
    df = df.dropna(subset=['COACH', 'TEAM'])
    for col in ['PAKE', 'PASE', 'S16', 'E8', 'F4', 'CHAMP']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    if 'WIN%' in df.columns:
        df['WIN%'] = df['WIN%'].astype(str).str.replace('%', '', regex=False)
        df['WIN%'] = pd.to_numeric(df['WIN%'], errors='coerce').fillna(0)
        if df['WIN%'].max() > 1:
            df['WIN%'] /= 100.0
    df['Deep_Run'] = df.get('S16', 0) + df.get('E8', 0) * 2 + df.get('F4', 0) * 3
    return df

_SCRAPED_COACH_CACHE = {}

def scrape_coach_live(team_name):
    """Scrape sports-reference.com for a team's current head coach."""
    if team_name in _SCRAPED_COACH_CACHE:
        return _SCRAPED_COACH_CACHE[team_name]
        
    try:
        slug = team_to_slug(team_name)
        url = f"https://www.sports-reference.com/cbb/schools/{slug}/2026.html"
        soup = fetch_page(url)
        if not soup:
            _SCRAPED_COACH_CACHE[team_name] = None
            return None
            
        for p in soup.find_all(['p', 'div']):
            text = p.get_text()
            if 'Coach:' in text:
                link = p.find('a')
                if link and 'coach' in link.get('href', ''):
                    name = link.get_text().strip()
                else:
                    name = text.split('Coach:')[1].strip().split('\n')[0].strip()
                _SCRAPED_COACH_CACHE[team_name] = name
                return name
    except Exception as e:
        logger.error(f"Error scraping coach for {team_name}: {e}")
        
    _SCRAPED_COACH_CACHE[team_name] = None
    return None

def get_coach_for_team(team_name, coach_df, skip_scrape=False):
    """Match a short-form team name to coach_results TEAM column. Falls back to live scraping."""
    coach_name = None
    
    # 1. Check overrides first (High Priority)
    if team_name in config.COACH_TEAM_OVERRIDES:
        coach_name = config.COACH_TEAM_OVERRIDES[team_name]
    
    # 2. Match a short-form team name to coach_results TEAM column
    if not coach_name and not coach_df.empty:
        def _normalize(s):
            return str(s).lower().replace('.', '').replace("'", '').replace('-', ' ').strip()
        
        target_norm = _normalize(team_name)
        teams = coach_df['TEAM'].dropna().unique().tolist()
        
        substring_matches = [t for t in teams if target_norm in _normalize(t)]
        match_found = None
        if len(substring_matches) == 1:
            match_found = coach_df[coach_df['TEAM'] == substring_matches[0]].iloc[0]
        elif len(substring_matches) > 1:
            best = difflib.get_close_matches(team_name, substring_matches, n=1, cutoff=0.3)
            if best:
                match_found = coach_df[coach_df['TEAM'] == best[0]].iloc[0]
        else:
            matches = difflib.get_close_matches(target_norm, [_normalize(t) for t in teams], n=1, cutoff=0.7)
            if matches:
                for t in teams:
                    if _normalize(t) == matches[0]:
                        match_found = coach_df[coach_df['TEAM'] == t].iloc[0]
        
        if match_found is not None:
            coach_name = match_found['COACH']
    
    # Fallback to live name scrape
    if not coach_name and not skip_scrape:
        coach_name = scrape_coach_live(team_name)
    
    # If we have a coach name, get full stats from coach_engine
    if coach_name:
        stats_dict = coach_engine.calculate_coach_stats(coach_name)
        if stats_dict:
            return pd.Series(stats_dict)
            
    return None

def get_base_stats():
    """Load and merge KenPom/Torvik stats using config paths."""
    # Use path discovery from config
    filepath_kp = config.KENPOM_PATH if os.path.exists(config.KENPOM_PATH) else os.path.join(config.DATA_CLEAN_DIR, 'kenpom.csv')
    filepath_tv = config.TORVIK_PATH if os.path.exists(config.TORVIK_PATH) else os.path.join(config.DATA_CLEAN_DIR, 'torvik.csv')

    def _read_csv_smart(filepath):
        temp_df = pd.read_csv(filepath, header=None, low_memory=False)
        # Case-insensitive search for Team/TEAM
        mask = temp_df.apply(lambda row: row.astype(str).str.contains('Team', case=False).any(), axis=1)
        if not mask.any():
            # If no 'Team' found, assume standard CSV with header at row 0
            df = pd.read_csv(filepath)
        else:
            header_idx = temp_df[mask].index[0]
            df = pd.read_csv(filepath, skiprows=header_idx)
            
        # Standardize the team column name to 'Team'
        if 'TEAM' in df.columns and 'Team' not in df.columns:
            df = df.rename(columns={'TEAM': 'Team'})
            
        df = df[df['Team'] != 'Team'].dropna(subset=['Team'])
        
        # Numeric cleaning for all non-string columns
        def _clean_num(x):
            try:
                # Handle Torvik-style multiline entries
                return float(str(x).split('\n')[0].strip())
            except (ValueError, TypeError):
                return np.nan # Use NaN to handle missing values properly
                
        for col in df.columns:
            if col not in ['Team', 'TEAM', 'Conf', 'CONF', 'Conf_kp', 'Conf_tv', 'W-L', 'Rec', 'YEAR', 'Year']:
                df[col] = df[col].apply(_clean_num)
        
        # Fill missing values with 0.0 after cleaning
        df = df.fillna(0.0)
        return df

    kenpom_df = _read_csv_smart(filepath_kp)
    torvik_df = _read_csv_smart(filepath_tv)
    
    # Naming Normalization and Seed Extraction
    def _extract_seed(name):
        # 1. Search for "Team 1" (Old KenPom style)
        match = re.search(r'\s+(\d+)$', str(name))
        if match: return int(match.group(1))
        
        # 2. Search for "1 seed, ✅" (New Torvik multiline style)
        lines = str(name).split('\n')
        if len(lines) > 1:
            match = re.search(r'(\d+)\s+seed', lines[1])
            if match: return int(match.group(1))
            
        return 99

    # Extract seeds from Torvik (contains multiline seeds) and KenPom
    torvik_df['Seed'] = torvik_df['Team'].apply(_extract_seed)
    kenpom_df['Seed'] = kenpom_df['Team'].apply(_extract_seed)
    
    # Strip seeds from names
    kenpom_df['Team'] = kenpom_df['Team'].astype(str).str.replace(r'\s*\d+$', '', regex=True).str.strip()
    kenpom_df['Team'] = kenpom_df['Team'].replace(config.TEAM_NAME_NORMALIZE)
    
    torvik_df['Team'] = torvik_df['Team'].astype(str).str.split('\n').str[0].str.strip()
    torvik_df['Team'] = torvik_df['Team'].replace(config.TEAM_NAME_NORMALIZE)
    
    # Merge and resolve Seed (Prefer KP's extraction, fallback to TV)
    merged = pd.merge(kenpom_df, torvik_df, on='Team', how='inner', suffixes=('_kp', '_tv'))
    merged['Seed'] = merged.apply(lambda r: r['Seed_kp'] if r['Seed_kp'] != 99 else r['Seed_tv'], axis=1)
    
    # 3. Integrate current season EvanMiya data if available
    if os.path.exists(config.EVAN_MIYA_PATH):
        evan_df = _read_csv_smart(config.EVAN_MIYA_PATH)
        # Filter for 2026 data
        if 'YEAR' in evan_df.columns:
            evan_2026 = evan_df[evan_df['YEAR'] == 2026].copy()
            evan_2026['Team'] = evan_2026['Team'].replace(config.TEAM_NAME_NORMALIZE)
            merged = pd.merge(merged, evan_2026, on='Team', how='left')
    
    # SOS Column Normalization
    if 'NetRtg.1' in merged.columns:
        merged = merged.rename(columns={'NetRtg.1': 'SOS AdjEM'})
    
    return merged

def load_historical_data():
    """Load historical tournament matchups and pair them into Team A / Team B matches."""
    path = os.path.join(config.DATA_RAW_DIR, 'Tournament Matchups.csv')
    if not os.path.exists(path):
        path = config.TOURNEY_MATCHUPS_PATH # Fallback
    
    if not os.path.exists(path):
        logger.error(f"Tournament Matchups not found at {path}")
        return pd.DataFrame(), []

    df = pd.read_csv(path)
    # Filter for years we have data for (exclude 2026 which has no scores yet)
    df = df[(df['YEAR'] >= 2015) & (df['YEAR'] <= 2025)]
    
    matchups = []
    # Logic: Adjacent rows in the same YEAR and CURRENT ROUND are opponents
    # Column names in CSV: ['YEAR', 'BY YEAR NO', 'TEAM NO', 'TEAM', 'SEED', 'ROUND', 'CURRENT ROUND', 'SCORE']
    for (year, curr_round), group in df.groupby(['YEAR', 'CURRENT ROUND']):
        # Sort by BY YEAR NO to ensure consistent pairing
        group = group.sort_values('BY YEAR NO', ascending=False)
        rows = group.to_dict('records')
        
        for i in range(0, len(rows) - 1, 2):
            team_a = rows[i]
            team_b = rows[i+1]
            
            # Determine winner (SCORE comparison)
            score_a = float(team_a.get('SCORE', 0))
            score_b = float(team_b.get('SCORE', 0))
            winner_a = 1 if score_a > score_b else 0
            
            matchups.append({
                'YEAR': year,
                'ROUND': curr_round,
                'TEAM_A': team_a['TEAM'],
                'SEED_A': team_a['SEED'],
                'SCORE_A': score_a,
                'TEAM_B': team_b['TEAM'],
                'SEED_B': team_b['SEED'],
                'SCORE_B': score_b,
                'TeamAwins': winner_a
            })
            
    matchup_df = pd.DataFrame(matchups)
    logger.info(f"Loaded {len(matchup_df)} historical matchups.")
    return matchup_df, []

def merge_and_create_features():
    """Merge historical matchups with efficiency metrics and create 'Diff' features."""
    matchups, _ = load_historical_data()
    if matchups.empty:
        return pd.DataFrame(), []
        
    # Load historical stats from EvanMiya (primary source for multi-year)
    stats_path = os.path.join(config.DATA_RAW_DIR, 'EvanMiya.csv')
    if not os.path.exists(stats_path):
        stats_path = config.EVAN_MIYA_PATH
    
    if not os.path.exists(stats_path):
        logger.error("EvanMiya.csv not found for historical merging.")
        return pd.DataFrame(), []
        
    stats_df = pd.read_csv(stats_path)
    stats_df['Team_Clean'] = stats_df['TEAM'].replace(config.TEAM_NAME_NORMALIZE)
    
    # Redefined OFFICIAL FEATURES based on historical availability (EvanMiya 2015-2025)
    # We remove any feature not present in this dataset to avoid the "Zero-Data" trap.
    OFFICIAL_FEATURES = [
        'Diff_AdjOE', 'Diff_AdjDE', 'Diff_NetRtg', 'Diff_POWER RATING', 
        'Diff_Barthag', 'Diff_Pace', 'Diff_KILLSHOTS MARGIN', 'Pace_Trap'
    ]
    
    features = []
    for _, row in matchups.iterrows():
        year = row['YEAR']
        team_a = row['TEAM_A']
        team_b = row['TEAM_B']
        
        stats_a = stats_df[(stats_df['YEAR'] == year) & (stats_df['Team_Clean'] == team_a)]
        stats_b = stats_df[(stats_df['YEAR'] == year) & (stats_df['Team_Clean'] == team_b)]
        
        if stats_a.empty or stats_b.empty:
            continue
            
        sa = stats_a.iloc[0]
        sb = stats_b.iloc[0]
        
        # Calculate Barthag for historical stats
        def _calc_barthag(o, d):
            return (o**10.25) / (o**10.25 + d**10.25)
            
        # Calculate Differences (Team A - Team B)
        diff_pace = abs(sa['PACE ADJUST'] - sb['PACE ADJUST'])
        
        feat_dict = {
            'YEAR': year,
            'TEAM_A': team_a,
            'TEAM_B': team_b,
            'TeamAwins': row['TeamAwins'],
            'ROUND': row['ROUND'],
            'SEED_A': row['SEED_A'],
            'SEED_B': row['SEED_B'],
            'Diff_AdjOE': sa['O RATE'] - sb['O RATE'],
            'Diff_AdjDE': sa['D RATE'] - sb['D RATE'],
            'Diff_NetRtg': sa['RELATIVE RATING'] - sb['RELATIVE RATING'],
            'Diff_POWER RATING': sa['RELATIVE RATING'] - sb['RELATIVE RATING'],
            'Diff_KILLSHOTS MARGIN': sa['KILLSHOTS MARGIN'] - sb['KILLSHOTS MARGIN'],
            'Diff_Barthag': _calc_barthag(sa['O RATE'], sa['D RATE']) - _calc_barthag(sb['O RATE'], sb['D RATE']),
            'Diff_Pace': sa['PACE ADJUST'] - sb['PACE ADJUST'],
            'Pace_Trap': 1.0 if diff_pace > 7.0 else 0.0
        }
        
        features.append(feat_dict)
        
    full_df = pd.DataFrame(features)
    # Final cleanup: Ensure all numeric columns are filled and valid
    full_df = full_df.fillna(0.0)
    
    logger.info(f"Created features for {len(full_df)} matchups using {len(OFFICIAL_FEATURES)} historical columns.")
    
    return full_df, OFFICIAL_FEATURES

def get_dataloaders(batch_size=32):
    """Process features, scale them, and return components for training."""
    df, feature_cols = merge_and_create_features()
    if df.empty:
        return None, None, [], None
        
    X = df[feature_cols].values
    y = df['TeamAwins'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for use in predict.py (sync)
    joblib.dump(scaler, os.path.join(config.PREPROCESSING_DIR, 'scaler.pkl'))
    
    return X_scaled, y, feature_cols, scaler
