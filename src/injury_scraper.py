import requests
from bs4 import BeautifulSoup

def scrape_injuries(skip_scrape=False):
    url = "https://www.rotowire.com/cbasketball/injury-report.php"
    headers = {"User-Agent": "Mozilla/5.0"}
    injuries = {
        "Duke": [{"player": "Caleb Foster", "status": "OUT", "is_star": True}],
        "Texas Tech": [{"player": "JT Toppin", "status": "OUT", "is_star": True}],
        "Alabama": [
            {"player": "Latrell Wrightsell", "status": "OUT", "is_star": True},
            {"player": "Aden Holloway", "status": "SUSPENSION", "is_star": True},
            {"player": "Latrell Hoover", "status": "OUT", "is_star": True},
            {"player": "Aden Halloway", "status": "SUSPENSION", "is_star": True}
        ]
    }
    
    if skip_scrape:
        return injuries
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Attempt to scrape RotoWire players natively
        players = soup.find_all('div', class_='inj-player')
        for p in players:
            name_tag = p.find('a', class_='player-name')
            team_tag = p.find('div', class_='team-name')
            status_tag = p.find('div', class_='injury-status')
            if name_tag and team_tag and status_tag:
                team = team_tag.text.strip()
                player = name_tag.text.strip()
                status = status_tag.text.strip().upper()
                if "OUT" in status or "SUSPENSION" in status or "DOUBTFUL" in status:
                    if team not in injuries:
                        injuries[team] = []
                    # Avoid duplicates dynamically 
                    if not any(existing['player'] == player for existing in injuries[team]):
                        injuries[team].append({"player": player, "status": status, "is_star": True})
    except Exception:
        pass
        
    return injuries
