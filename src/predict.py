import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import torch
torch.set_num_threads(1)
import argparse
import warnings
import joblib
import difflib
import glob
import xgboost as xgb
import logging
import random
import csv
import concurrent.futures

# New internal modules
import config
import tactics
import injuries
import coach_engine
import schedule_engine
from model import MarchMadnessNN
from data_loader import get_base_stats, load_coach_data, get_coach_for_team

# UI Imports
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Purge Pickles 
try:
    for f in glob.glob(os.path.join(config.PREPROCESSING_DIR, '*.pkl')):
        if 'inj' in f.lower() or 'roster' in f.lower():
            os.remove(f)
except Exception as e:
    logger.debug(f"Failed to purge pickles: {e}")

warnings.filterwarnings('ignore')

def get_merged_stats():
    """ONLY ingest active 2026 Base Stats (Kenpom/Torvik)."""
    stats = get_base_stats()
    stats = stats.dropna(subset=['Team'])
    stats = stats.drop_duplicates(subset=['Team'])
    return stats.set_index('Team')

_ROSTER_CACHE = {}
_PLAYERS_DF_GLOBAL = None

def get_team_roster(team_name):
    global _PLAYERS_DF_GLOBAL
    if team_name in _ROSTER_CACHE:
        return _ROSTER_CACHE[team_name]
        
    players_csv_path = config.PLAYERS_PATH
    try:
        if _PLAYERS_DF_GLOBAL is None:
            if os.path.exists(players_csv_path):
                _PLAYERS_DF_GLOBAL = pd.read_csv(players_csv_path, low_memory=False)
            else:
                return pd.DataFrame()
        
        if _PLAYERS_DF_GLOBAL.empty: return pd.DataFrame()
        
        teams = _PLAYERS_DF_GLOBAL['Team'].dropna().unique()
        matches = difflib.get_close_matches(team_name, teams, n=1, cutoff=0.5)
        if matches:
            roster = _PLAYERS_DF_GLOBAL[_PLAYERS_DF_GLOBAL['Team'] == matches[0]].copy()
            _ROSTER_CACHE[team_name] = roster
            return roster
    except Exception as e:
        logger.debug(f"Error loading roster for {team_name}: {e}")
    return pd.DataFrame()

def recalculate_aggregate_metrics(s_df):
    """
    Sync aggregate columns (NetRtg, Barthag) with potentially modified AdjOE/AdjDE.
    """
    oe = float(s_df.get('AdjOE', 100.0))
    de = float(s_df.get('AdjDE', 100.0))
    # NetRtg is ORtg - DRtg (AdjOE - AdjDE)
    s_df['NetRtg'] = oe - de
    # Barthag (Pythagorean win probability)
    try:
        s_df['Barthag'] = (oe**10.25) / (oe**10.25 + de**10.25)
    except Exception:
        s_df['Barthag'] = 1.0 if oe > de else 0.0
    return s_df

from utils import TEAM_ALIASES

def fuzzy_match_team(team_name, valid_teams):
    valid_teams = [str(t) for t in valid_teams if pd.notna(t)]
    if team_name is None:
        raise ValueError("team_name is None")
    
    if team_name in TEAM_ALIASES:
        alias = TEAM_ALIASES[team_name]
        if alias in valid_teams:
            return alias
    
    if team_name in valid_teams:
        return team_name
    
    lower_map = {str(t).lower(): t for t in valid_teams if pd.notna(t)}
    if team_name.lower() in lower_map:
        return lower_map[team_name.lower()]
    
    matches = difflib.get_close_matches(team_name, valid_teams, n=1, cutoff=0.5)
    if not matches:
        raise ValueError(f"Team '{team_name}' could not be matched securely.")
    return matches[0]

def get_team_profile(team_name, stats, scout_cache=None, coach_df=None, inj_df=None, is_deep_tourney=False, momentum_multiplier=1.0, silent=True, console_logs=None):
    """
    Extracts all team-specific metrics (Independent of opponent).
    Pre-calculates AdjOE/AdjDE modifiers for performance.
    """
    if console_logs is None:
        console_logs = []
        
    # 1. Clean Name & Base Stats
    team_clean = team_name if team_name in stats.index else fuzzy_match_team(team_name, stats.index)
    s_df = stats.loc[team_clean].to_dict()
    
    # 2. Scouting & Schedule Context
    scout = (scout_cache.get(team_clean, {}) if scout_cache else 
             schedule_engine.get_team_schedule(team_clean, stats, skip_scrape=True))
    
    s_df['Fragility_Score'] = scout.get('Fragility_Score', 0.0)
    
    # 3. Momentum (Round-Scaled)
    mom_scalar = scout.get('Momentum_Scalar', 0.0)
    s_df['AdjOE'] = float(s_df.get('AdjOE', 100.0)) + (mom_scalar * 0.2 * momentum_multiplier)
    
    # 4. Roster & Tactics (Single-Team parts)
    roster = get_team_roster(team_clean)
    sbi = tactics.get_team_stretch_index(team_clean, roster, console_logs)
    
    # 5. Injury Math
    if inj_df is not None:
        inj_res = injuries.find_team_injuries(team_clean, inj_df, roster, console_logs)
        injuries.apply_injury_math(team_clean, s_df, inj_res, config.BENCH_MINUTES_DB, console_logs)
        
    # 6. SOS Baseline (Median correction handled in matchup or pre-calc)
    # 7. Fragility Modifiers
    frag_score = float(scout.get('Fragility_Score', 0.0))
    if frag_score > 3.0:
        penalty = min(5.0, (frag_score - 3.0) * 0.5)
        s_df['AdjDE'] = float(s_df.get('AdjDE', 100.0)) + penalty
        
    ql = float(scout.get('Quality_Losses', 0.0))
    if ql >= 2.0 and frag_score < 0.0:
        bonus = min(2.0, ql * 0.4)
        s_df['AdjOE'] = float(s_df.get('AdjOE', 100.0)) + bonus
        
    # 8. Coach Layer
    coach = get_coach_for_team(team_clean, coach_df, skip_scrape=True)
    if coach is None: coach = {"is_unknown": True}
    
    is_unknown = coach.get('is_unknown', True)
    pake = float(coach.get('PAKE', 0)) if not is_unknown else 0.0
    pase = float(coach.get('PASE', 0)) if not is_unknown else 0.0
    f4 = float(coach.get('F4', 0))
    
    # Power Rating / Barthag logic
    oe = float(s_df.get('AdjOE', 100.0))
    de = float(s_df.get('AdjDE', 100.0))
    try:
        barthag = (oe**10.25) / (oe**10.25 + de**10.25)
    except:
        barthag = 1.0 if oe > de else 0.0
        
    profile = {
        'Team': team_clean,
        'AdjOE': oe,
        'AdjDE': de,
        'NetRtg': oe - de,
        'Barthag': barthag,
        'Adj T.': float(s_df.get('Adj T.', 70.0)),
        'KILLSHOTS MARGIN': float(s_df.get('KILLSHOTS MARGIN', 0.0)),
        'PAKE': pake,
        'PASE': pase,
        'F4': f4,
        'Conf': str(s_df.get('Conf_kp', 'Other')),
        'SOS': float(s_df.get('SOS AdjEM', 0.0) if 'SOS AdjEM' in s_df else s_df.get('NetRtg.1', 0.0)),
        'SBI': sbi,
        'Seed': s_df.get('Seed', 99)
    }
    return profile

def get_matchup_features(team_a, team_b, stats, scaler, injuries_data=None, skip_scrape=False, coach_df=None, inj_df=None, official_features=None, tourney_round="Regular Season", silent=False, profiles=None):
    """
    Extracts 25 features for the model using pre-calculated profiles if available.
    """
    logs = []
    is_deep_tourney = tourney_round in ["Sweet 16", "Elite 8", "Final 4", "Championship"]
    momentum_multiplier = 1.5 if is_deep_tourney else 1.0
    
    if profiles and team_a in profiles and team_b in profiles:
        p_a = profiles[team_a].copy()
        p_b = profiles[team_b].copy()
    else:
        p_a = get_team_profile(team_a, stats, coach_df=coach_df, inj_df=inj_df, is_deep_tourney=is_deep_tourney, momentum_multiplier=momentum_multiplier, silent=silent, console_logs=logs)
        p_b = get_team_profile(team_b, stats, coach_df=coach_df, inj_df=inj_df, is_deep_tourney=is_deep_tourney, momentum_multiplier=momentum_multiplier, silent=silent, console_logs=logs)

    team_a_clean = p_a['Team']
    team_b_clean = p_b['Team']
    
    # Matchup-Dependent Logic (Tactics & P4 Floor)
    # 1. Stretch-Big Index Modifier
    if p_a['SBI'] > 0:
        penalty = min(3.0, p_b['AdjDE'] * 0.025)
        p_b['AdjDE'] += penalty
    if p_b['SBI'] > 0:
        penalty = min(3.0, p_a['AdjDE'] * 0.025)
        p_a['AdjDE'] += penalty
        
    # 2. P4 Athleticism Floor
    is_p4_a = p_a['Conf'] in config.P4_CONFS
    is_p4_b = p_b['Conf'] in config.P4_CONFS
    if is_p4_a != is_p4_b:
        p_p4, p_opp = (p_a, p_b) if is_p4_a else (p_b, p_a)
        gap_reduction = max(0, min(0.2, (p_opp['SOS'] + 5) / 50.0))
        eff_gap = 0.15 - gap_reduction
        p_p4['AdjOE'] += eff_gap
        p_p4['AdjDE'] -= eff_gap

    # Build Diff Vector (X_vals)
    X_vals = [
        p_a['AdjOE'] - p_b['AdjOE'],
        p_b['AdjDE'] - p_a['AdjDE'],
        p_a['NetRtg'] - p_b['NetRtg'],
        p_a['NetRtg'] - p_b['NetRtg'], # Power Rating proxy
        p_a['KILLSHOTS MARGIN'] - p_b['KILLSHOTS MARGIN'],
        p_a['Barthag'] - p_b['Barthag'],
        p_a['Adj T.'] - p_b['Adj T.'],
        float(abs(p_a['Adj T.'] - p_b['Adj T.']) > 7.0)
    ]
    
    X = np.nan_to_num(np.array([X_vals]))
    X_scaled = scaler.transform(X)
    
    return torch.tensor(X_scaled, dtype=torch.float32), team_a_clean, team_b_clean, p_a['F4'], p_b['F4'], X, logs

def predict_matchup(team_a, team_b, model_nn, model_xgb, model_rf, stats, scaler, injuries_data=None, tourney_round="Regular Season", skip_scrape=False, coach_df=None, inj_df=None, official_features=None, meta_learner_obj=None, silent=False, profiles=None, use_seed_bias=True):
    """Symmetric Ensemble prediction. Supports pre-calculated profiles."""
    
    def _get_raw_probs(t1, t2):
        res = get_matchup_features(
            t1, t2, stats, scaler, injuries_data, 
            skip_scrape=skip_scrape, 
            coach_df=coach_df, 
            inj_df=inj_df, 
            official_features=official_features,
            tourney_round=tourney_round,
            silent=silent,
            profiles=profiles
        )
        X_nn, t1_clean, t2_clean, f1, f2, X_raw, logs = res
        
        X_scaled_np = X_nn.numpy()
        model_nn.eval()
        with torch.no_grad():
            nn_p = torch.sigmoid(model_nn(X_nn)).item()
        
        xgb_p = model_xgb.predict_proba(X_scaled_np)[:, 1][0]
        rf_p = model_rf.predict_proba(X_scaled_np)[:, 1][0]
        return nn_p, xgb_p, rf_p, logs, t1_clean, t2_clean

    nn_a, xgb_a, rf_a, logs_a, tA, tB = _get_raw_probs(team_a, team_b)
    nn_b, xgb_b, rf_b, logs_b, _, _ = _get_raw_probs(team_b, team_a)
    
    # Average the probabilities (A winning vs B losing)
    nn_prob = (nn_a + (1.0 - nn_b)) / 2.0
    xgb_prob = (xgb_a + (1.0 - xgb_b)) / 2.0
    rf_prob = (rf_a + (1.0 - rf_b)) / 2.0
    
    # Final Weighted Ensemble (Using Meta-Learner if available)
    if meta_learner_obj is not None:
        X_meta = np.array([[xgb_prob, nn_prob, rf_prob]])
        final_prob_a = meta_learner_obj.predict_proba(X_meta)[:, 1][0]
        X_meta_b = np.array([[1.0 - xgb_prob, 1.0 - nn_prob, 1.0 - rf_prob]])
        final_prob_b = meta_learner_obj.predict_proba(X_meta_b)[:, 1][0]
    else:
        # Fixed weights faster for MC
        final_prob_a = (0.40 * xgb_prob) + (0.35 * nn_prob) + (0.25 * rf_prob)
        final_prob_b = (0.40 * (1.0 - xgb_prob)) + (0.35 * (1.0 - nn_prob)) + (0.25 * (1.0 - rf_prob))

    final_prob = (final_prob_a + (1.0 - final_prob_b)) / 2.0
    unique_logs = list(dict.fromkeys(logs_a + logs_b)) if not silent else []
    
    # --- Seed Prior ---
    if use_seed_bias:
        if profiles and tA in profiles and tB in profiles:
            seed_a = profiles[tA]['Seed']
            seed_b = profiles[tB]['Seed']
        else:
            seed_a = stats.loc[tA].get('Seed', 99) if tA in stats.index else 99
            seed_b = stats.loc[tB].get('Seed', 99) if tB in stats.index else 99
        
        if pd.notna(seed_a) and pd.notna(seed_b) and seed_a != 99 and seed_b != 99:
            favored_a = (seed_a < seed_b)
            high_s, low_s = min(seed_a, seed_b), max(seed_a, seed_b)
            historical_odds = {(1, 16): 0.98, (2, 15): 0.931, (3, 14): 0.85, (4, 13): 0.79, (5, 12): 0.644, (6, 11): 0.61, (7, 10): 0.608, (8, 9): 0.50}
            hist_win_prob = historical_odds.get((high_s, low_s), 0.75 if high_s != low_s else 0.50)
            hist_prob_a = hist_win_prob if favored_a else (1.0 - hist_win_prob)
            final_prob = (final_prob * 0.7) + (hist_prob_a * 0.3)
            if not silent: unique_logs.append(f"[bold magenta][SEED ANCHOR] Blended Model Prob ({hist_prob_a*100:.1f}% Prior)[/bold magenta]")
    
    return final_prob, tA, tB, unique_logs

def run_matchup_monte_carlo(team_a, team_b, model_nn, model_xgb, model_rf, stats, scaler, injuries_data=None, n_sims=1000, skip_scrape=True):
    """Run probabilistic simulations for a single matchup."""
    prob_a, tA, tB, logs = predict_matchup(team_a, team_b, model_nn, model_xgb, model_rf, stats, scaler, injuries_data, skip_scrape=skip_scrape)
    
    # Console for log output
    console = Console()
    for log in logs:
        console.print(log)
        
    wins_a = 0
    for _ in range(n_sims):
        if random.random() < prob_a:
            wins_a += 1
            
    return wins_a, n_sims, tA, tB

def get_tournament_teams(stats):
    """
    Load the 64-team bracket from Tournament Matchups.csv for 2026.
    Ensures pairings are correctly ordered for the simulation loop.
    """
    if os.path.exists(config.TOURNEY_MATCHUPS_PATH):
        try:
            df = pd.read_csv(config.TOURNEY_MATCHUPS_PATH)
            # Filter for 2026 Round of 64
            df_2026 = df[(df['YEAR'] == 2026) & (df['CURRENT ROUND'] == 64)]
            
            ordered_teams = []
            seen = set()
            
            # The CSV is pre-ordered in bracket sequences
            for i in range(len(df_2026)):
                team_name = df_2026.iloc[i]['TEAM']
                
                # Deduplication for First Four (keep the first team/matchup variant found)
                if team_name in seen:
                    continue
                
                ordered_teams.append(team_name)
                seen.add(team_name)
                
                if len(ordered_teams) >= 64:
                    break
            
            if len(ordered_teams) == 64:
                logger.info(f"Successfully loaded 64-team bracket from {config.TOURNEY_MATCHUPS_PATH}")
                return ordered_teams
            else:
                logger.warning(f"Only found {len(ordered_teams)} unique teams in {config.TOURNEY_MATCHUPS_PATH}. Falling back to Seed sorting.")
        except Exception as e:
            logger.error(f"Error parsing tournament matchups: {e}")

    # Fallback Logic (Seed sorting)
    tournament_teams = stats[stats['Seed'] != 99].copy()
    if tournament_teams.empty:
        logger.warning("No seeded teams found in stats. Falling back to Top 64 by Barthag.")
        return stats.sort_values(by='Barthag', ascending=False).head(64).index.tolist()
    
    # Sort by Seed (primary) then Barthag (secondary) to handle selection order
    return tournament_teams.sort_values(by=['Seed', 'Barthag'], ascending=[True, False]).head(64).index.tolist()

def precompute_team_profiles(teams, stats, scout_cache, coach_df, inj_df, is_deep=False, momentum_mult=1.0, silent=True):
    """Bake profiles for all 64 teams to avoid redundant lookups."""
    profiles = {}
    for t in teams:
        profiles[t] = get_team_profile(t, stats, scout_cache, coach_df, inj_df, is_deep, momentum_mult, silent)
    return profiles

def generate_win_matrix(teams, stats, model_nn, model_xgb, model_rf, scaler, meta_learner_obj, profiles, silent=True):
    """
    Generate win probabilities for all possible pairs in the 64-team field.
    Bypasses individual predict calls for massive gain.
    """
    matrix = {}
    # Use standard predict loop for now, batching can be added later if needed.
    # With profiles, this only takes ~1-2 seconds for all 2,000 matches.
    for i, tA in enumerate(teams):
        matrix[tA] = {}
        for j, tB in enumerate(teams):
            if i == j: continue
            prob, _, _, _ = predict_matchup(tA, tB, model_nn, model_xgb, model_rf, stats, scaler, meta_learner_obj=meta_learner_obj, profiles=profiles, silent=True)
            matrix[tA][tB] = prob
    return matrix

def simulate_tournament(model_nn, model_xgb, model_rf, stats, scaler, teams=None, live_injuries=None, probabilistic=False, console=None, skip_scrape=False, coach_df=None, inj_df=None, official_features=None, meta_learner_obj=None, silent=True, win_matrix=None):
    """Execute a full tournament simulation (UI optional)."""
    if teams is None:
        teams = get_tournament_teams(stats)
    
    # Warm up cache if we have a console (indicating interactive use)
    if console:
        warm_up_cache(teams, stats, skip_scrape=skip_scrape)
    
    if console:
        console.print(Panel("[bold cyan] Welcome to the 2026 March Madness AI Simulator [/bold cyan]\n[white]Simulating Top 64 Live Teams into Championship Glory![/white]", border_style="cyan"))
    
    rounds = ["R32 Field", "Sweet 16", "Elite 8", "Final 4", "Championship", "Champion"]
    current_teams = teams.copy()
    # R64 refers to the initial 64 teams
    results = {"R64": teams.copy()}
    
    upset_counts = {'R32 Field': 0, 'Sweet 16': 0, 'Elite 8': 0, 'Final 4': 0, 'Championship': 0, 'Champion': 0, 'TOTAL': 0}
    
    for round_name in rounds:
        if len(current_teams) < 2: break
        
        # Display label for the games being played
        active_round = "Round of 64"
        if len(current_teams) == 32: active_round = "Round of 32"
        elif len(current_teams) == 16: active_round = "Sweet 16"
        elif len(current_teams) == 8: active_round = "Elite 8"
        elif len(current_teams) == 4: active_round = "Final 4"
        elif len(current_teams) == 2: active_round = "National Championship"
        
        if console:
            console.print(f"\n[bold yellow]--- {active_round} ---[/bold yellow]")
        
        next_round = []
        table = Table(box=box.MINIMAL_DOUBLE_HEAD) if console else None
        if table:
            table.add_column("Team A", style="bold green")
            table.add_column("Win Prob", justify="center", style="bold magenta")
            table.add_column("Team B", style="bold red")
            table.add_column("Projected Winner", style="bold yellow")
        
        for i in range(0, len(current_teams), 2):
            team_a = current_teams[i]
            team_b = current_teams[i+1]
            
            if win_matrix and team_a in win_matrix and team_b in win_matrix[team_a]:
                prob_a = win_matrix[team_a][team_b]
                tA, tB = team_a, team_b
                logs = []
            else:
                prob_a, tA, tB, logs = predict_matchup(
                    team_a, team_b, model_nn, model_xgb, model_rf, stats, scaler, 
                    injuries_data=live_injuries, 
                    tourney_round=round_name, 
                    skip_scrape=skip_scrape,
                    coach_df=coach_df,
                    inj_df=inj_df,
                    official_features=official_features,
                    meta_learner_obj=meta_learner_obj,
                    silent=silent
                )
            
            # Probabilistic Decision
            if probabilistic:
                # The NIL Wall Logic (Caps on extreme upsets to prevent MC breakage)
                seed_a = stats.loc[tA].get('Seed', 99) if tA in stats.index else 99
                seed_b = stats.loc[tB].get('Seed', 99) if tB in stats.index else 99
                
                if seed_a == 1 and seed_b == 16:
                    prob_a = max(prob_a, 0.99)
                elif seed_a == 16 and seed_b == 1:
                    prob_a = min(prob_a, 0.01)
                elif seed_a == 2 and seed_b == 15:
                    prob_a = max(prob_a, 0.93)
                elif seed_a == 15 and seed_b == 2:
                    prob_a = min(prob_a, 0.07)
                elif seed_a == 3 and seed_b == 14:
                    prob_a = max(prob_a, 0.85)
                elif seed_a == 14 and seed_b == 3:
                    prob_a = min(prob_a, 0.15)

                winner = tA if random.random() < prob_a else tB
                disp_prob = prob_a if winner == tA else (1 - prob_a)
            else:
                winner = tA if prob_a >= 0.5 else tB
                disp_prob = prob_a if prob_a >= 0.5 else 1 - prob_a
                
            prob_str = f"{disp_prob*100:.1f}%"
            
            if table:
                if winner == tA:
                    table.add_row(f"[bold]{tA}[/bold]", prob_str, tB, tA)
                else:
                    table.add_row(tA, prob_str, f"[bold]{tB}[/bold]", tB)
                
            next_round.append(winner)
            
            # --- Upset Tracking ---
            loser = tB if winner == tA else tA
            seed_w = stats.loc[winner].get('Seed', 99) if winner in stats.index else 99
            seed_l = stats.loc[loser].get('Seed', 99) if loser in stats.index else 99
            
            if pd.notna(seed_w) and pd.notna(seed_l) and seed_w != 99 and seed_l != 99:
                if (seed_w - seed_l) >= 2:
                    upset_counts[round_name] += 1
                    upset_counts['TOTAL'] += 1
            
        if console:
            console.print(table)
            
        current_teams = next_round
        results[round_name] = current_teams.copy()
        
    if current_teams and console:
        champion = current_teams[0]
        console.print(Panel(f"[bold magenta] 2026 TOURNAMENT CHAMPION [/bold magenta]\n\n[bold white]{champion}[/bold white]", expand=False, border_style="yellow"))
    
    results['Upset_Counts'] = upset_counts
    return results

def warm_up_cache(teams, stats, skip_scrape=False):
    """
    Pre-fetch or verify necessary scraping data (coaches, schedules) for the 
    teams in the simulation.
    """
    console = Console()
    missing_teams = []
    
    # Check what's missing
    for team in teams:
        slug = schedule_engine.team_to_slug(team)
        cache_path = config.CACHE_SCHEDULES_PATH
        is_missing = True
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path)
                if not df[df['Slug'] == slug].empty:
                    is_missing = False
            except: pass
        if is_missing:
            missing_teams.append(team)

    if not missing_teams:
        return

    if skip_scrape:
        console.print(f"\n[bold yellow][CACHE MISS][/bold yellow] {len(missing_teams)} teams are missing local schedule data. [SKIP SCRAPE] is active.")
        return

    console.print(f"\n[bold yellow][CACHE WARM-UP][/bold yellow] Pre-fetching data for {len(missing_teams)} new teams...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Crawling Sports-Reference...", total=len(missing_teams))
        for team in missing_teams:
            progress.update(task, description=f"[cyan]Scouting {team}...")
            # Trigger coach and schedule scrapes
            try:
                coach_df = load_coach_data()
                get_coach_for_team(team, coach_df, skip_scrape=False)
                schedule_engine.get_team_schedule(team, stats, skip_scrape=False)
            except Exception as e:
                logger.debug(f"Warm-up error for {team}: {e}")
            progress.update(task, advance=1)
    console.print("[bold green]Cache Warmed![/bold green] Simulation starting now...\n")

def jitter_team_stats(stats_df, teams, scout_cache, scale=1.0):
    """
    Creates a 'Performance Scenario' by adding Gaussian noise to team metrics.
    Vectorized for performance.
    """
    jittered_stats = stats_df.copy()
    
    # Filter teams that exist in index to avoid errors
    valid_teams = [t for t in teams if t in jittered_stats.index]
    
    # Pre-extract fragility and momentum for all tournament teams
    fragilities = np.array([float(scout_cache.get(t, {}).get('Fragility_Score', 5.0)) for t in valid_teams])
    momentums = np.array([float(scout_cache.get(t, {}).get('Momentum_Scalar', 0.0)) for t in valid_teams])
    
    # Vectorized Noise Generation
    std_devs = np.maximum(0.05, scale * (0.5 + (np.maximum(0, fragilities) / 10.0)))
    noises = np.random.normal(0, std_devs, size=len(valid_teams))
    oe_noises = np.random.normal(0, std_devs * 0.5, size=len(valid_teams))
    de_noises = np.random.normal(0, std_devs * 0.5, size=len(valid_teams))
    
    # Apply to the DataFrame in blocks
    if 'AdjEM' in jittered_stats.columns:
        jittered_stats.loc[valid_teams, 'AdjEM'] += noises + (momentums * 0.2)
    if 'AdjOE' in jittered_stats.columns:
        jittered_stats.loc[valid_teams, 'AdjOE'] += oe_noises
    if 'AdjDE' in jittered_stats.columns:
        jittered_stats.loc[valid_teams, 'AdjDE'] += de_noises
            
    return jittered_stats

def generate_win_matrix(teams, stats, model_nn, model_xgb, model_rf, scaler, meta_learner_obj, profiles, silent=True):
    """
    Generate win probabilities for all possible pairs in the 64-team field using BATCH INFERENCE.
    """
    matrix = {t: {} for t in teams}
    n = len(teams)
    
    # 1. Prepare all pairwise matchups (A vs B and B vs A for symmetry)
    matchup_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            matchup_pairs.append((teams[i], teams[j]))
            matchup_pairs.append((teams[j], teams[i]))
            
    # 2. Extract features in batches
    X_raw_list = []
    for t1, t2 in matchup_pairs:
        # Mini-get_features for speed (profiles are already handled)
        p_a, p_b = profiles[t1].copy(), profiles[t2].copy()
        
        # Tactical/P4 logic (Matchup-dependent)
        if p_a['SBI'] > 0: p_b['AdjDE'] += min(3.0, p_b['AdjDE'] * 0.025)
        if p_b['SBI'] > 0: p_a['AdjDE'] += min(3.0, p_a['AdjDE'] * 0.025)
        
        is_p4_a, is_p4_b = p_a['Conf'] in config.P4_CONFS, p_b['Conf'] in config.P4_CONFS
        if is_p4_a != is_p4_b:
            p_p4, p_opp = (p_a, p_b) if is_p4_a else (p_b, p_a)
            eff_gap = 0.15 - max(0, min(0.2, (p_opp['SOS'] + 5) / 50.0))
            p_p4['AdjOE'] += eff_gap
            p_p4['AdjDE'] -= eff_gap

        X_vals = [p_a['AdjOE'] - p_b['AdjOE'], p_b['AdjDE'] - p_a['AdjDE'], p_a['NetRtg'] - p_b['NetRtg'], p_a['NetRtg'] - p_b['NetRtg'], 
                  p_a['KILLSHOTS MARGIN'] - p_b['KILLSHOTS MARGIN'], p_a['Barthag'] - p_b['Barthag'], p_a['Adj T.'] - p_b['Adj T.'], float(abs(p_a['Adj T.'] - p_b['Adj T.']) > 7.0)]
        X_raw_list.append(X_vals)
        
    X_batch = np.nan_to_num(np.array(X_raw_list))
    X_scaled = scaler.transform(X_batch)
    
    # 3. Batch Predict
    model_nn.eval()
    with torch.no_grad():
        nn_probs = torch.sigmoid(model_nn(torch.tensor(X_scaled, dtype=torch.float32))).numpy().flatten()
    
    xgb_probs = model_xgb.predict_proba(X_scaled)[:, 1]
    rf_probs = model_rf.predict_proba(X_scaled)[:, 1]
    
    # 4. Ensemble & Matrix Populate
    if meta_learner_obj:
        X_meta = np.column_stack([xgb_probs, nn_probs, rf_probs])
        final_probs = meta_learner_obj.predict_proba(X_meta)[:, 1]
    else:
        final_probs = (0.40 * xgb_probs) + (0.35 * nn_probs) + (0.25 * rf_probs)
        
    for idx, (t1, t2) in enumerate(matchup_pairs):
        # We handle symmetry by calculating both and averaging later, or just storing
        matrix[t1][t2] = final_probs[idx]
        
    # 5. Final Symmetry & Seed Prior Pass
    historical_odds = {(1, 16): 0.98, (2, 15): 0.931, (3, 14): 0.85, (4, 13): 0.79, (5, 12): 0.644, (6, 11): 0.61, (7, 10): 0.608, (8, 9): 0.50}
    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = teams[i], teams[j]
            # Average the symmetric predictions
            p12 = matrix[t1].get(t2, 0.5)
            p21 = matrix[t2].get(t1, 0.5)
            avg_p = (p12 + (1.0 - p21)) / 2.0
            
            # Seed Prior
            s1, s2 = profiles[t1]['Seed'], profiles[t2]['Seed']
            if pd.notna(s1) and pd.notna(s2) and s1 != 99 and s2 != 99:
                favored_1 = s1 < s2
                h, l = min(s1, s2), max(s1, s2)
                hist_p = historical_odds.get((h, l), 0.75 if h != l else 0.50)
                hist_p1 = hist_p if favored_1 else (1.0 - hist_p)
                avg_p = (avg_p * 0.7) + (hist_p1 * 0.3)
            
            matrix[t1][t2] = avg_p
            matrix[t2][t1] = 1.0 - avg_p
            
    return matrix

def _worker_simulate_bracket(worker_args):
    """Scenario worker: Generates matrix and runs a batch of brackets."""
    stats_df, teams, scout_cache, model_nn, model_xgb, model_rf, scaler_obj, \
    live_injuries, coach_df, inj_df, official_features, meta_learner_obj, batch_size = worker_args
    
    # 1. Scenario Jitter & Profiling
    sim_stats = jitter_team_stats(stats_df, teams, scout_cache, scale=1.0)
    profiles = precompute_team_profiles(teams, sim_stats, scout_cache, coach_df, inj_df, silent=True)
    
    # 2. Scenario Matrix Generation (Bakes the scenario)
    win_matrix = generate_win_matrix(teams, sim_stats, model_nn, model_xgb, model_rf, scaler_obj, meta_learner_obj, profiles, silent=True)
    
    # 3. Batch Run Brackets in this Scenario
    scenario_results = []
    for _ in range(batch_size): 
        results = simulate_tournament(
            model_nn, model_xgb, model_rf, sim_stats, scaler_obj, 
            teams=teams, live_injuries=live_injuries, 
            probabilistic=True, console=None, skip_scrape=True,
            coach_df=coach_df,
            inj_df=inj_df,
            official_features=official_features,
            meta_learner_obj=meta_learner_obj,
            silent=True,
            win_matrix=win_matrix
        )
        scenario_results.append(results)
    return scenario_results

def run_monte_carlo(model_nn, model_xgb, model_rf, stats, scaler, live_injuries=None, n_simulations=100):
    """Run large-scale tournament simulations using performance scenarios (jitter)."""
    console = Console()
    teams = get_tournament_teams(stats)
    
    # Load Historical Upset Constraints
    upset_file = os.path.join(config.BASE_DIR, "data", "raw", "Upset Count.csv")
    min_upsets = {'TOTAL': 0}
    max_upsets = {'TOTAL': 99}
    if os.path.exists(upset_file):
        try:
            udf = pd.read_csv(upset_file)
            min_upsets['TOTAL'] = max(0, udf['TOTAL'].min() - 1)
            max_upsets['TOTAL'] = udf['TOTAL'].max() + 1
        except Exception as e:
            console.print(f"[bold red]Warning: Could not load Upset Count.csv ({e})[/bold red]")
    
    # 1. Warm up the cache for these 64 teams
    warm_up_cache(teams, stats, skip_scrape=True)
    
    # 2. Pre-fetch schedule scout data and rosters to avoid repeated cache misses
    console.print("[cyan]Generating volatility profiles and rosters for tournament field...[/cyan]")
    scout_cache = {}
    for team in teams:
        scout_cache[team] = schedule_engine.get_team_schedule(team, stats, skip_scrape=True)
        get_team_roster(team) # Pre-warm roster cache
    
    # 3. High Performance Cache Loading (Pre-load large DFs once)
    console.print("[cyan]Caching tactical intelligence for simulation...[/cyan]")
    cached_coach_df = load_coach_data()
    cached_inj_df = pd.read_csv(config.INJURIES_PATH) if os.path.exists(config.INJURIES_PATH) else None
    
    meta_path = os.path.join(config.WEIGHTS_DIR, 'meta_learner.joblib')
    cached_meta = joblib.load(meta_path) if os.path.exists(meta_path) else None
    
    try:
        try:
            cached_features = joblib.load(os.path.join(config.PREPROCESSING_DIR, 'feature_list.pkl'))
        except:
            cached_features = None

        champion_counts = {}
        csv_file = os.path.join(config.OUTPUTS_DIR, "monte_carlo_brackets.csv")
        headers = ["Sim_ID", "R32", "S16", "E8", "F4", "Finalists", "Winner"]
        
        console.print(Panel(f"[bold cyan] Initializing Calculated Monte Carlo [/bold cyan]\n[white]Simulating {n_simulations} performance scenarios based on team fragility.[/white]", border_style="cyan"))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Running Scenarios...", total=n_simulations)
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
                valid_brackets = 0
                # Dynamic batch size: Up to 500 brackets per scenario matrix for runs > 50k
                batch_size = max(5, min(500, n_simulations // 100)) 
                required_scenarios = (n_simulations // batch_size) + 1
                
                # Pack static picklable arguments
                worker_args = (stats, teams, scout_cache, model_nn, model_xgb, model_rf, scaler, live_injuries, cached_coach_df, cached_inj_df, cached_features, cached_meta, batch_size)
                
                # Limit max_workers for Windows stability
                max_w = min(os.cpu_count() or 4, 8)
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_w) as executor:
                    futures = set()
                    # Submit based on scenarios needed
                    for _ in range(min(required_scenarios, max_w * 4)):
                        futures.add(executor.submit(_worker_simulate_bracket, worker_args))
                        
                    while valid_brackets < n_simulations and futures:
                        done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                        
                        for future in done:
                            try:
                                batch_results = future.result() 
                                
                                for results in batch_results:
                                    if valid_brackets >= n_simulations: break
                                    
                                    # --- Rejection Sampling ---
                                    total_upsets = results.get('Upset_Counts', {}).get('TOTAL', 0)
                                    final_four = results.get("Final 4", [])
                                    champ = results.get("Champion", [None])[0]
                                    
                                    invalid_seed_run = False
                                    for t in final_four:
                                        s = stats.loc[t].get('Seed', 99) if t in stats.index else 99
                                        if s >= 12: invalid_seed_run = True; break
                                    
                                    if not invalid_seed_run and champ:
                                        s = stats.loc[champ].get('Seed', 99) if champ in stats.index else 99
                                        if s >= 9: invalid_seed_run = True
                                    
                                    if total_upsets < min_upsets['TOTAL'] or total_upsets > max_upsets['TOTAL'] or invalid_seed_run:
                                        continue
                                        
                                    # Success
                                    valid_brackets += 1
                                    if champ: champion_counts[champ] = champion_counts.get(champ, 0) + 1
                                    
                                    row = [valid_brackets, "|".join(results.get("R32 Field", [])), "|".join(results.get("Sweet 16", [])),
                                           "|".join(results.get("Elite 8", [])), "|".join(results.get("Final 4", [])),
                                           "|".join(results.get("Championship", [])), champ]
                                    writer.writerow(row)
                                    
                                    if valid_brackets % 100 == 0: f.flush()
                                    progress.update(task, completed=valid_brackets)
                                    
                                # Replenish
                                if valid_brackets < n_simulations:
                                    futures.add(executor.submit(_worker_simulate_bracket, worker_args))
                                    
                            except Exception as e:
                                progress.console.print(f"[bold red]Worker Error: {e}[/bold red]")
                                futures.add(executor.submit(_worker_simulate_bracket, worker_args))
                    
                    # Target matched. Cancel any lagging over-saturated futures
                    for f in futures: f.cancel()
            
        # --- COMPLETED SIMULATION ---
        if valid_brackets >= 100:
            console.print("[bold cyan]Step B: Running EV Scoring Engine...[/bold cyan]")
            df = pd.read_csv(csv_file)
            df = calculate_ev_scores(df)
            df.to_csv(csv_file, index=False)
            
            console.print("[bold magenta]Step C: Generating Diversified Portfolio (1/10th Pool)...[/bold magenta]")
            portfolio_df = generate_diversified_portfolio(df, stats, scout_cache)
            if not portfolio_df.empty:
                n_top = len(portfolio_df)
                out_path = os.path.join(config.OUTPUTS_DIR, f"top_{n_top}_projected_outcomes.csv")
                portfolio_df.to_csv(out_path, index=False)
                console.print(f"[bold green]Portfolio Generated: top_{n_top}_projected_outcomes.csv with 40/40/20 distribution.[/bold green]")

        # Summary Table
        table = Table(title=f"Monte Carlo Summary - Top 10 Champions ({n_simulations} Runs)", box=box.DOUBLE_EDGE)
        table.add_column("Team", style="bold white")
        table.add_column("Wins", justify="right", style="bold gold1")
        table.add_column("Frequency", justify="right", style="cyan")
        
        sorted_champs = sorted(champion_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for team, count in sorted_champs:
            table.add_row(team, str(count), f"{(count/n_simulations)*100:.1f}%")
            
        console.print(table)
        console.print(f"\n[bold green]Simulation Complete![/bold green] All {n_simulations} probabilistic bracket paths saved to: [underline]{csv_file}[/underline].")
        
    except Exception as e:
        console.print(f"[bold red]Monte Carlo Pipeline Failed: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

def calculate_ev_scores(df):
    """Calculates Global_EV_Score for every bracket based on aggregate frequencies."""
    # Round columns from headers: Winner, Finalists, F4, E8, S16, R32
    round_cols = {
        'Winner': 320, 
        'Finalists': 160, 
        'F4': 80, 
        'E8': 40, 
        'S16': 20, 
        'R32': 10
    }
    
    total_n = len(df)
    freqs = {}
    for col in round_cols:
        # Flatten the column and count frequencies
        # Some columns are list-like (multiple teams separated by |)
        all_teams = "|".join(df[col].astype(str)).split("|")
        from collections import Counter
        counts = Counter(all_teams)
        freqs[col] = {team: (count / total_n) for team, count in counts.items()}
        
    def _score_row(row):
        score = 0.0
        for col, weight in round_cols.items():
            teams = str(row[col]).split("|")
            col_freq = sum(freqs[col].get(t, 0.0) for t in teams)
            score += col_freq * weight
        return round(score, 2)
    
    df['Global_EV_Score'] = df.apply(_score_row, axis=1)
    return df.sort_values(by='Global_EV_Score', ascending=False)

def generate_diversified_portfolio(df, stats, scout_cache):
    """Creates top_1/10th file using Core (40%), Pivot (40%), and Chaos (20%) buckets."""
    n_total = len(df)
    n_target = n_total // 10
    if n_target < 10: return pd.DataFrame()
    
    # 1. Identify Frequency Ranking for Champions
    champ_freq = df['Winner'].value_counts()
    top_champs = champ_freq.index.tolist()
    
    # --- Bucket 1: Core (40%) ---
    n_core = int(n_target * 0.40)
    top_1 = top_champs[0] if top_champs else None
    core_df = df[df['Winner'] == top_1].head(n_core).copy()
    core_df['Portfolio_Group'] = 'Core'
    
    # --- Bucket 2: Pivot (40%) ---
    n_pivot = int(n_target * 0.40)
    pivot_champs = top_champs[1:5] # 2nd, 3rd, 4th, 5th
    pivot_list = []
    if pivot_champs:
        per_pivot = n_pivot // len(pivot_champs)
        for c in pivot_champs:
            pivot_list.append(df[df['Winner'] == c].head(per_pivot))
    pivot_df = pd.concat(pivot_list) if pivot_list else pd.DataFrame()
    if not pivot_df.empty:
        pivot_df['Portfolio_Group'] = 'Pivot'
    
    # --- Bucket 3: Chaos (20%) ---
    n_chaos = n_target - len(core_df) - len(pivot_df) # Remainder
    # Identify Fragile #1 and #2 Seeds
    fragile_seeds = []
    for team, scout in scout_cache.items():
        seed = stats.loc[team].get('Seed', 99) if team in stats.index else 99
        if seed in [1, 2] and float(scout.get('Fragility_Score', 0.0)) > 3.0:
            fragile_seeds.append(team)
            
    # Brackets where at least one fragile seed is NOT in S16
    if fragile_seeds:
        def _is_chaos(row):
            s16_list = str(row['S16']).split("|")
            for fs in fragile_seeds:
                if fs not in s16_list: return True
            return False
        chaos_pool = df[df.apply(_is_chaos, axis=1)]
    else:
        chaos_pool = df # Fallback to top EV if no fragile seeds found
        
    # Exclude already picked brackets
    picked_ids = set(core_df['Sim_ID'].tolist() + pivot_df['Sim_ID'].tolist())
    chaos_pool = chaos_pool[~chaos_pool['Sim_ID'].isin(picked_ids)]
    chaos_df = chaos_pool.head(n_chaos).copy()
    chaos_df['Portfolio_Group'] = 'Chaos'
    
    return pd.concat([core_df, pivot_df, chaos_df])


def main():
    parser = argparse.ArgumentParser(description="Live 2026 March Madness Predictor")
    parser.add_argument('team_a', nargs='?', type=str)
    parser.add_argument('team_b', nargs='?', type=str)
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--sim_matchup', type=int, nargs='?', const=1000, help="Run N probabilistic simulations for a single matchup.")
    parser.add_argument('--monte_carlo', type=int, nargs='?', const=100, help="Run N probabilistic simulations.")
    parser.add_argument('--round', type=str, default="Regular Season")
    parser.add_argument('--refresh', action='store_true', help="Force live scrape for the provided matchup (otherwise uses cache).")
    args = parser.parse_args()
    
    console = Console()
    
    if not (args.team_a and args.team_b) and not args.simulate and args.monte_carlo is None:
        console.print("[bold red]Error: Please provide two teams, use --simulate, or use --monte_carlo [N].[/bold red]")
        return
        
    try:
        stats = get_merged_stats()
        # Pre-load shared data for all modes
        cached_coach_df = load_coach_data()
        cached_inj_df = pd.read_csv(config.INJURIES_PATH) if os.path.exists(config.INJURIES_PATH) else None
        
        meta_path = os.path.join(config.WEIGHTS_DIR, 'meta_learner.joblib')
        cached_meta = joblib.load(meta_path) if os.path.exists(meta_path) else None
        
        try:
            cached_features = joblib.load(os.path.join(config.PREPROCESSING_DIR, 'feature_list.pkl'))
        except:
            cached_features = None

        scaler = joblib.load(os.path.join(config.PREPROCESSING_DIR, 'scaler.pkl'))
        
        # Load Models
        model_nn = MarchMadnessNN(input_size=len(cached_features) if cached_features else 8)
        model_nn.load_state_dict(torch.load(os.path.join(config.WEIGHTS_DIR, 'march_madness_weights.pt')))
        model_nn.eval()
        
        model_xgb = xgb.XGBClassifier()
        model_xgb.load_model(os.path.join(config.WEIGHTS_DIR, 'xgb_model.json'))
        
        model_rf = joblib.load(os.path.join(config.WEIGHTS_DIR, 'rf_model.joblib'))
        
        from injury_scraper import scrape_injuries
        # Simulation modes always skip live injury scraping for speed
        total_skip = (args.monte_carlo is not None) or (args.simulate)
        live_injuries = scrape_injuries(skip_scrape=total_skip)
    except Exception as e:
        console.print(f"[bold red]Initialization Error: {e}[/bold red]")
        return
        
    if args.monte_carlo is not None:
        run_monte_carlo(model_nn, model_xgb, model_rf, stats, scaler, live_injuries, n_simulations=args.monte_carlo)
    elif args.simulate:
        simulate_tournament(
            model_nn, model_xgb, model_rf, stats, scaler, None, live_injuries, 
            probabilistic=False, console=console, skip_scrape=True,
            coach_df=cached_coach_df, inj_df=cached_inj_df, 
            official_features=cached_features, meta_learner_obj=cached_meta
        )
    elif args.sim_matchup is not None:
        run_matchup_monte_carlo(args.team_a, args.team_b, model_nn, model_xgb, model_rf, stats, scaler, live_injuries, n_sims=args.sim_matchup, skip_scrape=True)
    else:
        # Standard One-on-One: Defaults to Cache (skip_scrape=True) for consistency
        # Use --refresh to get live data
        use_skip = not args.refresh
        if args.refresh:
            console.print("[yellow]Live Refresh active: Scraping Sports-Reference for fresh matchup data...[/yellow]")
        
        try:
            prob, tA, tB, logs = predict_matchup(
                args.team_a, args.team_b, model_nn, model_xgb, model_rf, stats, scaler, 
                live_injuries, tourney_round=args.round, skip_scrape=use_skip,
                coach_df=cached_coach_df, inj_df=cached_inj_df, 
                official_features=cached_features, meta_learner_obj=cached_meta,
                use_seed_bias=False
            )
            
            for log in logs:
                console.print(log)
            
            winner = tA if prob >= 0.5 else tB
            win_prob = prob if prob >= 0.5 else 1 - prob
            
            # Robust seed extraction for UI
            def _safe_seed(team):
                if team in stats.index:
                    val = stats.loc[team, 'Seed']
                    if isinstance(val, pd.Series): return val.iloc[0]
                    return val
                return 99

            seed_a = _safe_seed(tA)
            seed_b = _safe_seed(tB)
            
            console.print(Panel.fit(
                f"[bold cyan]Matchup Analysis (Live 2026)[/bold cyan]\n\n"
                f"[white]{tA} (#{(int(seed_a) if pd.notna(seed_a) and seed_a != 99 else '?')}) vs {tB} (#{(int(seed_b) if pd.notna(seed_b) and seed_b != 99 else '?')})[/white]\n\n"
                f"[bold yellow]Predicted Winner: {winner} ({win_prob*100:.1f}%)[/bold yellow]",
                border_style="green"
            ))
        except Exception as e:
            import traceback
            traceback.print_exc()
            console.print(f"[bold red]Prediction Error: {e}[/bold red]")

if __name__ == '__main__':
    main()
