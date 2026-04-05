import pandas as pd
import numpy as np
import difflib
import logging

logger = logging.getLogger(__name__)

def find_team_injuries(team_name, injuries_df, roster_df, console_logs):
    """
    Locates injured players for a team based on injury report and team roster.
    Returns a list of injury dictionaries.
    """
    if injuries_df.empty:
        return []
    
    inj_teams = injuries_df['Team'].dropna().unique()
    team_matches = difflib.get_close_matches(team_name, inj_teams, n=1, cutoff=0.5)
    
    if not team_matches:
        return []
    
    matched_team = team_matches[0]
    team_injuries = injuries_df[injuries_df['Team'] == matched_team]
    
    if roster_df.empty:
        return []
        
    valid_injuries = []
    for _, row in team_injuries.iterrows():
        player = row['Player']
        status = str(row['Status']).upper()
        
        if status in ["OUT", "SUSPENSION", "DOUBTFUL", "GAME TIME DECISION", "OUT FOR SEASON"]:
            roster_col = 'Unnamed: 3' if 'Unnamed: 3' in roster_df.columns else 'Player'
            roster_players = roster_df[roster_col].dropna().tolist()
            player_matches = difflib.get_close_matches(player, roster_players, n=1, cutoff=0.75)
            
            if not player_matches:
                # Robust Set-Based Matching (Handles "Glenn, Kaleb" vs "Kaleb Glenn")
                def _get_name_set(n):
                    return set(n.replace(",", "").replace(".", "").lower().split())
                
                target_set = _get_name_set(player)
                for rp in roster_players:
                    if _get_name_set(rp) == target_set:
                        player_matches = [rp]
                        break
            
            if not player_matches:
                console_logs.append(f"[yellow]Player {player} not found on team roster[/yellow]")
                continue
                
            matched_player = player_matches[0]
            p_data = roster_df[roster_df[roster_col] == matched_player].iloc[0]
            
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

def apply_injury_math(team_name, stats_df, injuries, bench_minutes_db, console_logs):
    """
    Computes and applies the mathematical penalty for injuries.
    Updates stats_df in-place.
    """
    if not injuries:
        console_logs.append(f"[green]No injuries detected for {team_name} - Proceeding with full strength.[/green]")
        return console_logs
    
    out_players = [p['player'] for p in injuries]
    console_logs.append(f"[red]Missing Players: {', '.join(out_players)}[/red]")
    
    # Bench Mitigation (P4 / Historically Deep Rotations)
    bench_pct = bench_minutes_db.get(team_name.lower(), 0.0)
    mitigation = 0.5 if bench_pct > 35.0 else 1.0
    
    orig_oe = float(stats_df.get('AdjOE', 0.0))
    orig_de = float(stats_df.get('AdjDE', 0.0))
    deducted_oe, deducted_de = 0.0, 0.0
    
    for p in injuries:
        p_usg = pd.to_numeric(p.get("Usg", 0), errors='coerce')
        if pd.notna(p_usg) and p_usg > 25.0:
            # Alpha-Replacement Penalty: High-usage stars docked percentage of team efficiency
            alpha_deduction_oe = orig_oe * 0.08 * mitigation
            alpha_deduction_de = -orig_de * 0.04 * mitigation
            deducted_oe += alpha_deduction_oe
            deducted_de += alpha_deduction_de
            console_logs.append(f"[bold red]Alpha-Replacement Penalty triggered for {str(p['player'])} (Usg: {p_usg} > 25%)[/bold red]")
        else:
            # Normal Replacement: Objections and Defensive ratings deducted
            deducted_oe += float(p["OBPR"]) * mitigation
            deducted_de += float(p["DBPR"]) * mitigation
            
    if deducted_oe != 0.0 or deducted_de != 0.0:
        stats_df['AdjOE'] -= deducted_oe
        stats_df['AdjDE'] -= deducted_de
        # Result summary will be built during reporting
        console_logs.append(f"[bold red]CRITICAL: Player-Level Injury Math Applied for {team_name}![/bold red]")
        console_logs.append(f"[yellow]Bench Mins: {bench_pct}% | Mitigation: {mitigation}x[/yellow]")
        console_logs.append(f"[yellow]Original AdjOE: {orig_oe:.1f} -> Adjusted AdjOE: {stats_df['AdjOE']:.1f}[/yellow]")
        console_logs.append(f"[yellow]Original AdjDE: {orig_de:.1f} -> Adjusted AdjDE: {stats_df['AdjDE']:.1f}[/yellow]")
            
    return console_logs
