import pandas as pd
import numpy as np
import torch
import argparse
import warnings
import joblib
import difflib
import os
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Purge Pickles 
try:
    for f in glob.glob(os.path.join(BASE_DIR, '*.pkl')):
        if 'inj' in f.lower() or 'roster' in f.lower():
            os.remove(f)
except Exception:
    pass

warnings.filterwarnings('ignore')

from model import MarchMadnessNN
from data_loader import get_base_stats
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

def get_merged_stats():
    # Phase 5 Live Ingestion: ONLY ingest active 2026 Base Stats (Kenpom/Torvik)
    stats = get_base_stats()
    stats = stats.drop_duplicates(subset=['Team'])
    return stats.set_index('Team')

BENCH_MINUTES_DB = {
    "duke": 36.5,
    "st. john's": 28.0,
    "texas tech": 25.5,
    "alabama": 38.0
}

def fuzzy_match_team(team_name, valid_teams):
    matches = difflib.get_close_matches(team_name, valid_teams, n=1, cutoff=0.5)
    if not matches:
        raise ValueError(f"Team '{team_name}' could not be matched securely to any active 2026 Kaggle indexing list.")
    return matches[0]

def get_matchup_features(team_a, team_b, stats, scaler, live_injuries=None, console=None):
    team_a_clean = fuzzy_match_team(team_a, stats.index)
    team_b_clean = fuzzy_match_team(team_b, stats.index)
        
    stats_a = stats.loc[team_a_clean].copy()
    stats_b = stats.loc[team_b_clean].copy()
    
    if console:
        players_csv_path = os.path.join(BASE_DIR, 'data', 'raw', '2026_players.csv')
        injuries_csv_path = os.path.join(BASE_DIR, 'data', 'raw', 'college-basketball-injury-report.csv')
        try:
            players_df = pd.read_csv(players_csv_path, low_memory=False)
            injuries_df = pd.read_csv(injuries_csv_path, low_memory=False)
            print(f"DEBUG: Current Injury List Keys: {injuries_df['Player'].tolist()[:15]}...")
        except Exception:
            players_df = pd.DataFrame()
            injuries_df = pd.DataFrame()

        def get_team_roster(team_name):
            if players_df.empty: return pd.DataFrame()
            teams = players_df['Team'].dropna().unique()
            matches = difflib.get_close_matches(team_name, teams, n=1, cutoff=0.5)
            if matches:
                return players_df[players_df['Team'] == matches[0]]
            return pd.DataFrame()

        def find_inj(name):
            if injuries_df.empty: return []
            inj_teams = injuries_df['Team'].dropna().unique()
            team_matches = difflib.get_close_matches(name, inj_teams, n=1, cutoff=0.5)
            if not team_matches:
                return []
            
            matched_team = team_matches[0]
            team_injuries = injuries_df[injuries_df['Team'] == matched_team]
            team_roster = get_team_roster(name)
            
            if team_roster.empty:
                return []
                
            valid_injuries = []
            for _, row in team_injuries.iterrows():
                player = row['Player']
                status = str(row['Status']).upper()
                
                if status in ["OUT", "SUSPENSION", "DOUBTFUL", "GAME TIME DECISION", "OUT FOR SEASON"]:
                    roster_col = 'Unnamed: 3' if 'Unnamed: 3' in team_roster.columns else 'Player'
                    roster_players = team_roster[roster_col].dropna().tolist()
                    player_matches = difflib.get_close_matches(player, roster_players, n=1, cutoff=0.75)
                    
                    if not player_matches:
                        clean_player = player.replace(".", "").lower()
                        for rp in roster_players:
                            if clean_player == rp.replace(".", "").lower():
                                player_matches = [rp]
                                break
                    
                    if not player_matches:
                        parts = player.split()
                        if len(parts) >= 2:
                            first, last = parts[0].lower(), parts[-1].lower()
                            for rp in roster_players:
                                rp_parts = rp.split()
                                if len(rp_parts) >= 2 and rp_parts[0].lower() == first and rp_parts[-1].lower() == last:
                                    player_matches = [rp]
                                    break
                    
                    if not player_matches:
                        console.print(f"[yellow]player {player} not found on team[/yellow]")
                        continue
                        
                    matched_player = player_matches[0]
                    print(f"DEBUG: Matched {player} to {matched_player}")
                    p_data = team_roster[team_roster[roster_col] == matched_player].iloc[0]
                    
                    usg = pd.to_numeric(p_data.get('Usg', 0), errors='coerce')
                    efg = pd.to_numeric(p_data.get('eFG', 0), errors='coerce')
                    obpm = pd.to_numeric(p_data.get('OBPM', np.nan), errors='coerce')
                    dbpm = pd.to_numeric(p_data.get('DBPM', 0), errors='coerce')
                    
                    if pd.isna(usg): usg = 0
                    if pd.isna(efg): efg = 0
                    if pd.isna(obpm):
                        obpr = (usg * efg) / 100.0
                    else:
                        obpr = obpm
                    
                    dbpr = -dbpm if pd.notna(dbpm) else 0.0
                    
                    valid_injuries.append({
                        "player": matched_player,
                        "status": status,
                        "is_star": True,
                        "Usg": usg,
                        "OBPR": obpr,
                        "DBPR": dbpr
                    })
            return valid_injuries

        for team_lbl, team_clean, stats_df in [("team_a", team_a_clean, stats_a), ("team_b", team_b_clean, stats_b)]:
            console.print(f"\n[cyan]Checking injury status for {team_clean}...[/cyan]")
            inj_res = find_inj(team_clean)
            if inj_res:
                out_players = [p['player'] for p in inj_res]
                console.print(f"[red]Missing Players: {', '.join(out_players)}[/red]")
                
                bench_pct = BENCH_MINUTES_DB.get(team_clean.lower(), 0.0)
                mitigation = 0.5 if bench_pct > 35.0 else 1.0
                
                orig_oe = float(stats_df.get('AdjOE', 0.0))
                orig_de = float(stats_df.get('AdjDE', 0.0))
                deducted_oe, deducted_de = 0.0, 0.0
                
                for p in inj_res:
                    p_usg = pd.to_numeric(p.get("Usg", 0), errors='coerce')
                    if pd.notna(p_usg) and p_usg > 25.0:
                        alpha_deduction_oe = orig_oe * 0.08 * mitigation
                        alpha_deduction_de = -orig_de * 0.04 * mitigation
                        deducted_oe += alpha_deduction_oe
                        deducted_de += alpha_deduction_de
                        console.print(f"[bold red]Alpha-Replacement Penalty triggered for {str(p['player'])} (Usg: {p_usg} > 25%)[/bold red]")
                    else:
                        deducted_oe += float(p["OBPR"]) * mitigation
                        deducted_de += float(p["DBPR"]) * mitigation
                        
                if Deducted_OE_Or_DE_Check := (deducted_oe != 0.0 or deducted_de != 0.0):
                    stats_df['AdjOE'] -= deducted_oe
                    stats_df['AdjDE'] -= deducted_de
                    console.print(Panel(f"[bold red]CRITICAL: Player-Level Injury Math Applied for {team_clean}![/bold red]\n[yellow]Bench Mins: {bench_pct}% | Mitigation: {mitigation}x\nOriginal AdjOE: {orig_oe:.1f} -> Adjusted AdjOE: {stats_df['AdjOE']:.1f}\nOriginal AdjDE: {orig_de:.1f} -> Adjusted AdjDE: {stats_df['AdjDE']:.1f}[/yellow]"))
            else:
                console.print(f"[green]No injuries detected for {team_clean} - Proceeding with full strength.[/green]")
    
    if console:
        tempo_a = pd.to_numeric(stats_a.get('Adj T.', 0.0), errors='coerce')
        tempo_b = pd.to_numeric(stats_b.get('Adj T.', 0.0), errors='coerce')
        if pd.notna(tempo_a) and pd.notna(tempo_b) and abs(tempo_a - tempo_b) > 7.0:
            seed_a = stats_a.get('Seed', 99); seed_a = seed_a if pd.notna(seed_a) else 99
            seed_b = stats_b.get('Seed', 99); seed_b = seed_b if pd.notna(seed_b) else 99
            
            if tempo_a < tempo_b:
                slower_team, slower_stats, slow_name = "Team A", stats_a, team_a_clean
                faster_stats, fast_name = stats_b, team_b_clean
                is_underdog = (seed_a > seed_b)
            else:
                slower_team, slower_stats, slow_name = "Team B", stats_b, team_b_clean
                faster_stats, fast_name = stats_a, team_a_clean
                is_underdog = (seed_b > seed_a)
            
            if is_underdog:
                pace_penalty = faster_stats.get('AdjDE', 0.0) * 0.04
                faster_stats['AdjDE'] += pace_penalty
                console.print(Panel(f"[bold yellow]⚠️ INVERSE PACE TRAP TRIGGERED: |{tempo_a:.1f} - {tempo_b:.1f}| > 7.0[/bold yellow]\n[white]Favorite {fast_name} forced out of rhythm by scrappy underdog {slow_name}. Favorite's AdjDE degraded by 4% (+{pace_penalty:.1f})[/white]", border_style="yellow"))
            else:
                pace_penalty = slower_stats.get('AdjDE', 0.0) * 0.04
                slower_stats['AdjDE'] += pace_penalty
                console.print(Panel(f"[bold yellow]⚠️ PACE TRAP TRIGGERED: |{tempo_a:.1f} - {tempo_b:.1f}| > 7.0[/bold yellow]\n[white]{slow_name} forced into track meet. AdjDE physically degraded by 4% (+{pace_penalty:.1f})[/white]", border_style="yellow"))

    top_15_features = joblib.load(os.path.join(BASE_DIR, 'top_15_features.pkl'))
    X_vals = []
    
    diff_pace = abs(stats_a.get('Adj T.', 0.0) - stats_b.get('Adj T.', 0.0))
    diff_3pr = stats_a.get('3PR', 0.0) - stats_b.get('3PR', 0.0)
    
    if console:
        console.print(Panel(f"[bold cyan]Volatility Check: Pace Diff {diff_pace:.1f} | 3PT Chaos Level {diff_3pr:.1f}[/bold cyan]"))
    
    for f in top_15_features:
        if f == 'Diff_AdjOE_A_vs_DE_B':
            val = stats_a.get('AdjOE', 0.0) - stats_b.get('AdjDE', 0.0)
        elif f == 'Diff_AdjOE_B_vs_DE_A':
            val = stats_b.get('AdjOE', 0.0) - stats_a.get('AdjDE', 0.0)
        elif f == 'Diff_Pace':
            val = diff_pace
        elif f == 'Pace_Trap':
            val = float(diff_pace > 7.0)
        else:
            col = f.replace("Diff_", "", 1)
            
            # Explicit safe dictionary fetches neutralizing historical dependencies like POWER RATING completely to 0 natively.
            val_a = stats_a.get(col, 0.0)
            val_a = val_a if pd.notna(val_a) else 0.0
            
            val_b = stats_b.get(col, 0.0)
            val_b = val_b if pd.notna(val_b) else 0.0
            
            val = float(val_a) - float(val_b)
        X_vals.append(val)
    
    X = np.array([X_vals])
    X_scaled = scaler.transform(X)
    
    return torch.tensor(X_scaled, dtype=torch.float32), team_a_clean, team_b_clean


def predict_matchup(team_a, team_b, model, stats, scaler, live_injuries=None, console=None):
    if live_injuries is None:
        live_injuries = {}
    
    features, tA_clean, tB_clean = get_matchup_features(team_a, team_b, stats, scaler, live_injuries, console)
    with torch.no_grad():
        prob_a = model(features).item()
        
    chaos_thresh = stats['3PR'].quantile(0.9)
    rate_a = stats.loc[tA_clean].get('3PR', 0)
    rate_b = stats.loc[tB_clean].get('3PR', 0)
    
    barthag_spread = abs(stats.loc[tA_clean].get('Barthag', 0) - stats.loc[tB_clean].get('Barthag', 0))
    multiplier = 1.35 if barthag_spread > 0.15 else 1.15
    
    if prob_a < 0.5 and pd.notna(rate_a) and rate_a >= chaos_thresh:
        prob_a = min(0.99, prob_a * multiplier)
        if console: console.print(Panel(f"[bold magenta]🎲 3PT CHAOS ACTIVATED 🎲[/bold magenta]\n[white]Underdog {tA_clean} shoots 3s heavily ({rate_a:.1f} > 90th% threshold). Matchup variance overrides standard gaps! Win Probability increased {multiplier}x.[/white]", border_style="magenta"))
    elif (1 - prob_a) < 0.5 and pd.notna(rate_b) and rate_b >= chaos_thresh:
        prob_b_new = min(0.99, (1 - prob_a) * multiplier)
        prob_a = 1 - prob_b_new
        if console: console.print(Panel(f"[bold magenta]🎲 3PT CHAOS ACTIVATED 🎲[/bold magenta]\n[white]Underdog {tB_clean} shoots 3s heavily ({rate_b:.1f} > 90th% threshold). Matchup variance overrides standard gaps! Win Probability increased {multiplier}x.[/white]", border_style="magenta"))

    return prob_a, tA_clean, tB_clean

def get_top_64_teams(stats):
    top64 = stats.sort_values(by='Barthag', ascending=False).head(64).index.tolist()
    return top64

def simulate_tournament(model, stats, scaler, teams=None, live_injuries=None):
    if teams is None:
        teams = get_top_64_teams(stats)
    
    console = Console()
    console.print(Panel("[bold cyan] Welcome to the 2026 March Madness AI Simulator 🏀[/bold cyan]\n[white]Simulating Top 64 Live Teams into Championship Glory![/white]", border_style="cyan"))
    
    rounds = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]
    current_teams = teams.copy()
    
    import random
    
    for round_name in rounds:
        console.print(f"\n[bold yellow]--- {round_name} ---[/bold yellow]")
        next_round = []
        
        table = Table(box=box.MINIMAL_DOUBLE_HEAD)
        table.add_column("Seed 1 / Team A", style="bold green")
        table.add_column("Win Prob", justify="center", style="bold magenta")
        table.add_column("Seed 2 / Team B", style="bold red")
        table.add_column("Projected Winner", style="bold yellow")
        
        for i in range(0, len(current_teams), 2):
            team_a = current_teams[i]
            team_b = current_teams[i+1]
            
            prob_a, tA, tB = predict_matchup(team_a, team_b, model, stats, scaler, live_injuries, console)
            
            if prob_a >= 0.5:
                winner = tA
            else:
                winner = tB
                prob_a = 1 - prob_a
                
            prob_str = f"{prob_a*100:.1f}%"
            
            if winner == tA:
                table.add_row(f"[bold]{tA}[/bold]", prob_str, tB, tA)
            else:
                table.add_row(tA, prob_str, f"[bold]{tB}[/bold]", tB)
                
            next_round.append(winner)
            
        console.print(table)
        current_teams = next_round
        
    champion = current_teams[0]
    console.print(Panel(f"[bold magenta]🏆 2026 TOURNAMENT CHAMPION 🏆[/bold magenta]\n\n[bold white]{champion}[/bold white]", expand=False, border_style="yellow"))


def main():
    parser = argparse.ArgumentParser(description="Live 2026 March Madness Ingestion Tracker")
    parser.add_argument('team_a', nargs='?', type=str, help='First team natively')
    parser.add_argument('team_b', nargs='?', type=str, help='Second team natively')
    parser.add_argument('--simulate', action='store_true', help='Execute auto 64 bracket sweep array.')
    args = parser.parse_args()
    
    console = Console()
    
    if not args.simulate and not (args.team_a and args.team_b):
        console.print("[bold red]Error: Please array two exact string queries (in quotes) or launch with the --simulate flag.[/bold red]")
        console.print("[dim]Usage Example: python predict.py \"Purdue\" \"North Carolina\"[/dim]")
        return
        
    with console.status("[bold green]Tapping into Live 2026 Kaggle Ingestions & Injury Reports...[/bold green]"):
        try:
            from injury_scraper import scrape_injuries
            live_injuries = scrape_injuries()
        except:
            live_injuries = {}
            
        try:
            stats = get_merged_stats()
            scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
        except Exception as e:
            console.print(f"[bold red]Error loading stats or scaler arrays natively: {e}[/bold red]")
            return
            
        model = MarchMadnessNN(input_size=15)
        try:
            model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'march_madness_weights.pt')))
            model.eval()
        except FileNotFoundError:
            console.print("[bold red]march_madness_weights.pt missing completely. Execute run_pipeline.py strictly.[/bold red]")
            return
            
    if args.simulate:
        simulate_tournament(model, stats, scaler, None, live_injuries)
    else:
        try:
            prob, tA, tB = predict_matchup(args.team_a, args.team_b, model, stats, scaler, live_injuries, console)
            winner = tA if prob >= 0.5 else tB
            win_prob = prob if prob >= 0.5 else 1 - prob
            
            seed_a = stats.loc[tA].get('Seed', 99)
            seed_b = stats.loc[tB].get('Seed', 99)
            
            is_upset = False
            if winner == tA and seed_a > seed_b:
                is_upset = True
            elif winner == tB and seed_b > seed_a:
                is_upset = True

            console.print(Panel.fit(
                f"[bold cyan]Matchup Analysis (Live 2026)[/bold cyan]\n\n"
                f"[white]{tA} (#{(int(seed_a) if not pd.isna(seed_a) and seed_a != 99 else '?')}) vs {tB} (#{(int(seed_b) if not pd.isna(seed_b) and seed_b != 99 else '?')})[/white]\n\n"
                f"[bold yellow]Predicted Winner: {winner}[/bold yellow] ([magenta]{win_prob*100:.1f}%[/magenta])",
                border_style="green"
            ))
            
            if 0.45 <= win_prob <= 0.55 or is_upset:
                console.print(Panel("[bold yellow]⚠️ VOLATILITY WARNING: High Upset Potential ⚠️[/bold yellow]"))
                
        except ValueError as e:
            console.print(f"[bold red]{e}[/bold red]")

if __name__ == '__main__':
    main()
