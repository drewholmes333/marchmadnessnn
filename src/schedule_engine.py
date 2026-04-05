import pandas as pd
import numpy as np
import logging
import os
import re
import difflib
from datetime import datetime
import config
from utils.scraper import fetch_page
from utils import team_to_slug

logger = logging.getLogger(__name__)

def _normalize_team_name(name):
    """
    Normalizes team names between Sports-Reference and Torvik/KenPom.
    """
    if not name: return ""
    name = name.replace('State', 'St.')
    name = name.replace('NC St.', 'North Carolina St.')
    name = name.replace('UNC', 'North Carolina')
    name = name.replace('LSU', 'Louisiana St.')
    name = name.replace('Mississippi', 'Ole Miss')
    name = name.replace('UConn', 'Connecticut')
    name = name.replace("St. John's", "St. John's (NY)")
    name = name.replace('University', '').replace('Univ.', '').strip()
    return name

# Global RAM Cache to avoid thousands of CSV reads in Monte Carlo
_GLOBAL_SCHEDULE_CACHE = {}

def get_team_schedule(team_name, stats_df, year=2026, force_refresh=False, skip_scrape=False):
    """
    Refined on-demand scraper with RAM-based caching for sub-millisecond lookups.
    """
    slug = team_to_slug(team_name)
    global _GLOBAL_SCHEDULE_CACHE
    
    # 1. Bulk Load from Disk (Once per Process)
    if not _GLOBAL_SCHEDULE_CACHE and os.path.exists(config.CACHE_SCHEDULES_PATH):
        try:
            logger.debug("Bulk loading schedule cache into RAM...")
            cache_df = pd.read_csv(config.CACHE_SCHEDULES_PATH)
            # Filter for latest data only if duplicates exist
            cache_df = cache_df.sort_values('Last_Scraped').drop_duplicates('Slug', keep='last')
            _GLOBAL_SCHEDULE_CACHE = cache_df.set_index('Slug').to_dict('index')
        except Exception as e:
            logger.debug(f"Schedule bulk load error: {e}")

    # 2. Check RAM Cache
    if slug in _GLOBAL_SCHEDULE_CACHE and not force_refresh:
        match = _GLOBAL_SCHEDULE_CACHE[slug]
        last_scraped_str = match.get('Last_Scraped')
        if pd.notna(last_scraped_str):
            try:
                last_scraped = pd.to_datetime(last_scraped_str)
                if (datetime.now() - last_scraped).total_seconds() < 86400: # 24 hrs
                    return match
            except: pass
            
    # Scrape
    if skip_scrape:
        logger.debug(f"Cache miss for {team_name} - [SKIP SCRAPE] active.")
        return {}
        
    url = f"https://www.sports-reference.com/cbb/schools/{slug}/{year}-schedule.html"
    logger.info(f"Scouting schedule for {team_name}: {url}")
    
    soup = fetch_page(url)
    if not soup: return {}
            
    table = soup.find('table', {'id': 'schedule'})
    if not table: return {}
        
    rows = []
    tbody = table.find('tbody')
    for tr in tbody.find_all('tr'):
        if 'thead' in tr.get('class', []): continue
        res_td = tr.find('td', {'data-stat': 'game_result'})
        if not res_td or res_td.get_text() not in ['W', 'L']: continue
            
        row_data = {
            'Date': tr.find('td', {'data-stat': 'date_game'}).get_text() if tr.find('td', {'data-stat': 'date_game'}) else "N/A",
            'Location': tr.find('td', {'data-stat': 'game_location'}).get_text() if tr.find('td', {'data-stat': 'game_location'}) else "",
            'Opponent': tr.find('td', {'data-stat': 'opp_name'}).get_text() if tr.find('td', {'data-stat': 'opp_name'}) else "N/A",
            'Result': res_td.get_text(),
            'PTS': pd.to_numeric(tr.find('td', {'data-stat': 'pts'}).get_text(), errors='coerce') if tr.find('td', {'data-stat': 'pts'}) else 0,
            'OPP_PTS': pd.to_numeric(tr.find('td', {'data-stat': 'opp_pts'}).get_text(), errors='coerce') if tr.find('td', {'data-stat': 'opp_pts'}) else 0,
            'OT': bool(tr.find('td', {'data-stat': 'overtimes'}).get_text()) if tr.find('td', {'data-stat': 'overtimes'}) else False,
        }
        row_data['Opponent'] = re.sub(r'#\d+\s*|\(\d+\)\s*', '', row_data['Opponent']).strip()
        rows.append(row_data)
        
    df = pd.DataFrame(rows)
    if df.empty: return {}
    
    # Filter Regular Season
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[df['Date'] < datetime(year, 3, 18)]
    
    # Calculate metrics
    metrics = calculate_balanced_fragility_v2_5(df, team_name, stats_df)
    save_to_cache(slug, metrics)
    return metrics

def calculate_balanced_fragility_v2_5(df, team_name, stats_df):
    """Balanced Fragility Index."""
    # (Simplified for refactor demo, identical to original logic but using config/logging)
    rank_col = [c for c in stats_df.columns if 'Rk_tv' in c]
    rank_col = rank_col[0] if rank_col else 'Rk_tv'
    
    df['MOV'] = df['PTS'] - df['OPP_PTS']
    season_avg_mov = df['MOV'].mean() if not df.empty else 0.0
    team_rank = float(stats_df.loc[team_name].get(rank_col, 50.0)) if team_name in stats_df.index else 50.0
    
    df = df.tail(15).copy() # 15-game window
    
    bad_losses, quality_losses, elite_fragility, blowout_losses = 0, 0, 0, 0
    valid_teams = stats_df.index
    
    for _, row in df[df['Result'] == 'L'].iterrows():
        opp_norm = _normalize_team_name(row['Opponent'])
        match = difflib.get_close_matches(opp_norm, valid_teams, n=1, cutoff=0.7)
        opp_rank = float(stats_df.loc[match[0]].get(rank_col, 125.0)) if match else 125.0
            
        if opp_rank > 100: bad_losses += 1
        if abs(row['MOV']) > 20: blowout_losses += 1
        
        margin_thresh = 10 if (row['OT'] or opp_rank <= 10) else 8
        if opp_rank <= 25:
            if abs(row['MOV']) <= margin_thresh: quality_losses += 1
            elif (opp_rank - team_rank) >= 25.0: elite_fragility += 1
                
    wins = df[df['Result'] == 'W']
    total_wins = len(wins)
    close_win_ratio = (len(wins[wins['MOV'] <= 5]) / total_wins) if total_wins > 0 else 0.0
    road_games = df[df['Location'] == '@']
    road_win_pct = len(road_games[road_games['Result'] == 'W']) / len(road_games) if not road_games.empty else 0.5
    
    sos_col = 'SOS AdjEM' if 'SOS AdjEM' in stats_df.columns else 'NetRtg.1'
    team_sos = float(stats_df.loc[team_name].get(sos_col, 0.0)) if team_name in stats_df.index else 0.0
    sos_factor = 1.0 / (1.0 + max(0, team_sos / 10.0))
    momentum_scalar = max(min((df.tail(5)['MOV'].mean() - season_avg_mov) * sos_factor, 3.0), -3.0)
    
    b_term, e_term, q_term = min(float(bad_losses)/2.0, 1.0), min(float(elite_fragility)/2.0, 1.0), min(float(quality_losses)/2.0, 1.0)
    fragility_score = (5.0 * b_term) + (2.0 * e_term) + (3.0 * close_win_ratio) - (2.0 * road_win_pct) - (2.0 * q_term) + (3.0 * blowout_losses)
    
    return {
        'Slug': team_to_slug(team_name), 
        'Team': team_name, 
        'Bad_Losses': float(bad_losses),
        'Quality_Losses': float(quality_losses),
        'Elite_Fragility': int(elite_fragility),
        'Blowout_Score': np.nan,
        'Road_Win_Pct': float(road_win_pct),
        'Close_Win_Ratio': float(close_win_ratio),
        'Fragility_Score': float(fragility_score), 
        'Momentum_Scalar': float(momentum_scalar),
        'Last_Scraped': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Blowout_Losses': float(blowout_losses)
    }

def save_to_cache(slug, metrics):
    if not metrics: return
    
    # Update RAM
    global _GLOBAL_SCHEDULE_CACHE
    _GLOBAL_SCHEDULE_CACHE[slug] = metrics
    
    # Update Disk
    new_row = pd.DataFrame([metrics])
    cache_path = config.CACHE_SCHEDULES_PATH
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
            df = pd.concat([df[df['Slug'] != slug], new_row], ignore_index=True)
            df.to_csv(cache_path, index=False)
        except: pd.DataFrame([metrics]).to_csv(cache_path, index=False)
    else: pd.DataFrame([metrics]).to_csv(cache_path, index=False)
