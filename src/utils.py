import re

TEAM_ALIASES = {
    # Manual Overrides for Scraping
    "UConn": "Connecticut",
    "UCLA": "UCLA",
    "BYU": "Brigham Young",
    "FDU": "Fairleigh Dickinson",
    "UNLV": "Nevada-Las Vegas",
    "USC": "Southern California",
    "SMU": "Southern Methodist",
    "LSU": "Louisiana State",
    "NC State": "North Carolina State",
    "Texas A&M": "Texas AM",
    "Miami FL": "Miami FL",
    "Ole Miss": "Mississippi",
    "St. John's": "St. Johns",
    "UCF": "Central Florida",
    "Vanderbilt": "Vanderbilt",
    
    # Matching Overrides from Predict.py
    "McNeese": "McNeese St.",
    "McNeese State": "McNeese St.",
    "NC State": "N.C. State",
    "North Carolina State": "N.C. State",
    "Miami": "Miami FL",
    "Long Beach State": "Long Beach St.",
    "San Diego State": "San Diego St.",
    "Boise State": "Boise St.",
    "Colorado State": "Colorado St.",
    "Fresno State": "Fresno St.",
    "Kansas State": "Kansas St.",
    "Michigan State": "Michigan St.",
    "Mississippi State": "Mississippi St.",
    "Oregon State": "Oregon St.",
    "Penn State": "Penn St.",
    "Ohio State": "Ohio St.",
    "Oklahoma State": "Oklahoma St.",
    "Iowa State": "Iowa St.",
    "Utah State": "Utah St.",
    "Wichita State": "Wichita St.",
    "Wright State": "Wright St.",
    "Kennesaw State": "Kennesaw St.",
    "Long Island University": "LIU",
    "Long Island": "LIU",
}

def team_to_slug(team_name):
    """Convert a short-form team name to a sports-reference URL slug."""
    slug_overrides = {
        "St. John's": "st-johns-ny", "Miami FL": "miami-fl", "Miami (FL)": "miami-fl", "Miami OH": "miami-oh",
        "UCF": "central-florida", "UConn": "connecticut", "UNLV": "nevada-las-vegas",
        "USC": "southern-california", "SMU": "southern-methodist", "BYU": "brigham-young",
        "LSU": "louisiana-state", "Ole Miss": "mississippi", "UNC": "north-carolina",
        "North Carolina": "north-carolina", "UCSB": "uc-santa-barbara", "VCU": "virginia-commonwealth",
        "Pitt": "pittsburgh", "Utah St.": "utah-state", "San Diego St.": "san-diego-state",
        "CSU Fullerton": "cal-state-fullerton", "LIU": "long-island-university",
        "FIU": "florida-international", "TCU": "texas-christian",
        "ETSU": "east-tennessee-state", "UNC Wilmington": "unc-wilmington",
        "UNC Asheville": "unc-asheville", "UNC Greensboro": "unc-greensboro",
        "UMBC": "maryland-baltimore-county", "UAB": "alabama-birmingham",
        "UT Arlington": "texas-arlington", "UTEP": "texas-el-paso",
        "UTSA": "texas-san-antonio", "App State": "appalachian-state",
        "Loyola Chicago": "loyola-il", "Saint Mary's": "saint-marys-ca",
        "Mount St. Mary's": "mount-st-marys", "St. Peter's": "saint-peters",
        "St. Bonaventure": "st-bonaventure", "Saint Louis": "saint-louis",
        "Long Beach St.": "long-beach-state", "Boise St.": "boise-state",
        "Colorado St.": "colorado-state", "Fresno St.": "fresno-state",
        "Kansas St.": "kansas-state", "Michigan St.": "michigan-state",
        "Mississippi St.": "mississippi-state", "Oregon St.": "oregon-state",
        "Penn St.": "penn-state", "Wichita St.": "wichita-state",
        "Wright St.": "wright-state", "Ohio St.": "ohio-state",
        "Oklahoma St.": "oklahoma-state", "Iowa St.": "iowa-state",
        "Texas Tech": "texas-tech", "Texas A&M": "texas-am",
        "Grand Canyon": "grand-canyon", "High Point": "high-point",
        "Northern Iowa": "northern-iowa", "New Mexico": "new-mexico",
        "Kennesaw St.": "kennesaw-state", "Southeast Missouri St.": "southeast-missouri-state",
        "McNeese St.": "mcneese-state", "McNeese": "mcneese-state", "McNeese State": "mcneese-state",
        "N.C. State": "north-carolina-state",
        "Cal Baptist": "california-baptist",
        "St. Thomas": "st-thomas-mn",
        "Sam Houston St.": "sam-houston-state",
        "Cal St. Bakersfield": "cal-state-bakersfield",
        "Grambling St.": "grambling",
    }
    
    if team_name in slug_overrides:
        return slug_overrides[team_name]
    
    # Generic normalization
    s = team_name.lower()
    
    # Convert 'st.' to 'state' specifically at the end of names
    if s.endswith(" st."):
        s = s.replace(" st.", " state")
    elif s.endswith(" st"):
        s = s.replace(" st", " state")
        
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = s.replace(" ", "-")
    return s
