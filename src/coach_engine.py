import os
import re
import pandas as pd
import logging
import config
from utils.scraper import fetch_page

logger = logging.getLogger(__name__)

# Standard expected tournament wins by seed
EXPECTED_WINS = config.EXPECTED_WINS_BY_SEED

def _coach_name_to_slug(name):
    overrides = {
        "Rick Pitino": "rick-pitino-1",
        "John Calipari": "john-calipari-1",
        "Tom Izzo": "tom-izzo-1",
    }
    if name in overrides: return overrides[name]
    parts = name.lower().strip().split()
    return f"{parts[0]}-{parts[1]}-1" if len(parts) >= 2 else f"{name.lower().strip()}-1"

_MEMORY_CACHE = {}
_STATIC_DATABASE = {}

def _load_cache():
    global _MEMORY_CACHE, _STATIC_DATABASE
    if _MEMORY_CACHE:
        return _MEMORY_CACHE
        
    # 1. Load Static Database (High Priority)
    if os.path.exists(config.COACH_RESULTS_PATH):
        try:
            static_df = pd.read_csv(config.COACH_RESULTS_PATH)
            for _, row in static_df.iterrows():
                name = str(row.get('COACH', '')).lower()
                if name:
                    # Map the raw CSV columns to expected internal format
                    stats = {
                        'COACH': row.get('COACH'),
                        'PAKE': float(str(row.get('PAKE', 0)).replace('%','')),
                        'PASE': float(str(row.get('PASE', 0)).replace('%','')),
                        'WIN%': float(str(row.get('WIN%', 0)).replace('%','')) / 100.0 if '%' in str(row.get('WIN%','')) else float(row.get('WIN%',0)),
                        'F4': int(row.get('F4', 0)),
                        'CHAMP': int(row.get('CHAMP', 0)),
                        'Deep_Run': int(row.get('S16', 0)) + int(row.get('E8', 0)) * 2 + int(row.get('F4', 0)) * 3 + int(row.get('CHAMP', 0)) * 5,
                        'Tourney_Wins': int(row.get('W', 0)),
                        'Tourney_Apps': int(row.get('Tourney_Apps', 0)) if 'Tourney_Apps' in row else 10, # Fallback
                        'is_unknown': False
                    }
                    _STATIC_DATABASE[name] = stats
                    _MEMORY_CACHE[name] = stats
        except Exception as e:
            logger.debug(f"Static coach results load failure: {e}")

    # 2. Load Dynamic Cache (Low Priority)
    if os.path.exists(config.CACHE_COACHES_PATH):
        try:
            df = pd.read_csv(config.CACHE_COACHES_PATH)
            if not df.empty:
                for _, row in df.iterrows():
                    name = str(row['COACH']).lower()
                    # Only add to memory if not already in static DB (Static wins)
                    if name not in _MEMORY_CACHE:
                        _MEMORY_CACHE[name] = row.to_dict()
            return _MEMORY_CACHE
        except Exception as e:
            logger.error(f"Error loading coach cache: {e}")
    return _MEMORY_CACHE

def _save_to_cache(stats):
    global _MEMORY_CACHE
    _MEMORY_CACHE[stats['COACH'].lower()] = stats
    
    # Still save to disk for persistence
    try:
        cache_file = config.CACHE_COACHES_PATH
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
            df = df[df['COACH'].str.lower() != stats['COACH'].lower()]
            df = pd.concat([df, pd.DataFrame([stats])], ignore_index=True)
        else:
            df = pd.DataFrame([stats])
        df.to_csv(cache_file, index=False)
    except Exception as e:
        logger.error(f"Disk save failure: {e}")

def calculate_coach_stats(coach_name):
    if not coach_name or coach_name == "Unknown": return None
    
    # 1. Historical Overrides
    if coach_name in config.HISTORICAL_COACH_OVERRIDES:
        res = config.HISTORICAL_COACH_OVERRIDES[coach_name].copy()
        res['COACH'] = coach_name
        res['is_unknown'] = False
        return res

    # 2. Disk/Memory Cache
    cache = _load_cache()
    if coach_name.lower() in cache:
        return cache[coach_name.lower()]

    # 3. Scraper
    try:
        slug = _coach_name_to_slug(coach_name)
        url = f"https://www.sports-reference.com/cbb/coaches/{slug}.html"
        logger.info(f"Scraping coach: {url}")
        
        soup = fetch_page(url)
        if not soup: return _build_empty(coach_name)

        tourney_wins, expected_wins = 0, 0.0
        s16, e8, f4, champ = 0, 0, 0, 0
        total_w, total_g, tourney_apps = 0, 0, 0
        
        table = soup.find('table', {'id': 'coaching_record'})
        if table:
            rows = table.find('tbody').find_all('tr', class_=lambda x: x != 'thead')
            for row in rows:
                tds = row.find_all('td')
                if len(tds) < 5: continue
                # Total wins
                try:
                    w, l = int(tds[2].text or 0), int(tds[3].text or 0)
                    total_w += w; total_g += (w+l)
                except: pass
                # Tournament notes
                notes = tds[-1].text
                if "NCAA" in notes:
                    tourney_apps += 1
                    seed_match = re.search(r'#(\d+)', notes)
                    seed = int(seed_match.group(1)) if seed_match else None
                    res_match = re.search(r'\((.*?)\)', notes)
                    res_text = res_match.group(1).lower() if res_match else ""
                    
                    wins = _res_to_score(res_text)
                    tourney_wins += wins
                    if seed in EXPECTED_WINS: expected_wins += EXPECTED_WINS[seed]
                    if any(x in res_text for x in ['sweet 16', 's16']): s16 += 1
                    if any(x in res_text for x in ['elite 8', 'e8']): e8 += 1
                    if any(x in res_text for x in ['final four', 'f4']): f4 += 1
                    if "champion" in res_text: champ += 1

        pake = round(tourney_wins - expected_wins, 2)
        is_unknown = (tourney_apps == 0)
        stats = {
            'COACH': coach_name, 'PAKE': pake, 'PASE': 0.1, 'WIN%': round(total_w/max(total_g, 1), 4),
            'F4': f4, 'Deep_Run': s16 + (e8*2) + (f4*3) + (champ*5), 'Tourney_Wins': tourney_wins,
            'Tourney_Apps': tourney_apps, 'CHAMP': champ, 'Wins_Per_App': round(tourney_wins/max(tourney_apps, 1), 2),
            'is_unknown': is_unknown
        }
        _save_to_cache(stats)
        return stats
    except Exception as e:
        logger.error(f"Error scraping coach {coach_name}: {e}")
        return _build_empty(coach_name)

def _res_to_score(res):
    if "champion" in res: return 6
    if "final" in res or "f4" in res: return 4
    if "elite" in res: return 3
    if "sweet" in res: return 2
    if "2nd round" in res or "32" in res: return 1
    return 0

def _build_empty(coach_name):
    return {
        'COACH': coach_name, 'PAKE': 0.0, 'PASE': 0.0, 'WIN%': 0.0, 'F4': 0, 'Deep_Run': 0, 
        'Tourney_Wins': 0, 'Tourney_Apps': 0, 'Wins_Per_App': 0.0, 'CHAMP': 0, 'is_unknown': True
    }

def get_or_calculate(coach_name):
    return calculate_coach_stats(coach_name)

def compute_system_score(team_stats, coach_name):
    adj_oe = float(team_stats.get('AdjOE', 105))
    return max(0.0, round((adj_oe - 105) * 0.015, 2))
