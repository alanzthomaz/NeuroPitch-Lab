# prediction_engine/elo.py
import math

K_FACTOR_WC = 60
HOME_ADV = 75

# Long-run strength priors (Elo anchors) for the 48 finalists.
SEED = {
    "france": 2120, "spain": 2065, "england": 2030, "argentina": 2020, "brazil": 2045,
    "portugal": 1980, "netherlands": 1965, "germany": 1945, "belgium": 1925, "italy": 1915,
    "colombia": 1890, "uruguay": 1875, "croatia": 1870, "morocco": 1840, "switzerland": 1825,
    "usa": 1830, "mexico": 1825, "japan": 1810, "senegal": 1795, "denmark": 1790,
    "ecuador": 1760, "australia": 1735, "south-korea": 1730, "iran": 1720, "poland": 1715,
    "canada": 1700, "serbia": 1695, "wales": 1665, "ghana": 1665, "tunisia": 1655,
    "ivory-coast": 1655, "nigeria": 1645, "saudi-arabia": 1640, "qatar": 1630, "egypt": 1620,
    "algeria": 1615, "scotland": 1610, "cameroon": 1600, "paraguay": 1595, "venezuela": 1590,
    "chile": 1580, "peru": 1575, "czech-republic": 1570, "bosnia-and-herzegovina": 1545,
    "south-africa": 1520, "new-zealand": 1495, "panama": 1480, "jamaica": 1460,
    "honduras": 1440, "jordan": 1420, "haiti": 1380, "el-salvador": 1370,
    "trinidad-and-tobago": 1360, "guatemala": 1345
}

HOSTS = {"mexico", "usa", "canada"}

def base_k(league_name=""):
    n = league_name.lower()
    if "world cup" in n and "qual" not in n:
        return 55
    if "world cup" in n and "qual" in n or "qualification" in n:
        return 40
    if any(x in n for x in ["copa america", "euro championship", "asian cup", "africa cup", "gold cup"]):
        return 50
    if "nations league" in n or "nations cup" in n:
        return 32
    if "friendl" in n:
        return 18
    return 28

def recency(ts_sec, now_sec):
    # 18-month half-life: 30.44 days per month
    return 0.5 ** (((now_sec - ts_sec) / (30.44 * 86400)) / 18)

def expected_score(rating_a, rating_b, home_bonus_a=0):
    return 1.0 / (1.0 + 10.0 ** ((rating_b - (rating_a + home_bonus_a)) / 400.0))

def expected_goals(rating, opponent, home_bonus=0):
    diff = (rating + home_bonus) - opponent
    lam = 1.35 + diff / 400.0
    return max(0.3, min(3.5, lam))

def g_mult(gd):
    d = abs(gd)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11.0 + d) / 8.0

def calibrate_elo(matches, now_sec=None):
    """
    Calibrate Elo ratings based on match results.
    matches: list of dicts with: homeSlug, awaySlug, homeName, awayName, hg, ag, ts, leagueName
    """
    if now_sec is None:
        now_sec = matches[-1]["ts"] if matches else int(1718092800) # Default to 2026-06-11 approx

    ratings_state = {}
    
    def get_r(slug, name):
        k = slug if slug else f"ghost:{name}"
        if k not in ratings_state:
            ratings_state[k] = SEED[slug] if (slug and slug in SEED) else 1500
        return ratings_state[k]
        
    def set_r(slug, name, v):
        k = slug if slug else f"ghost:{name}"
        ratings_state[k] = v

    applied = 0
    for m in matches:
        if m.get("hg") is None or m.get("ag") is None:
            continue
        
        home_slug = m.get("homeSlug")
        away_slug = m.get("awaySlug")
        home_name = m.get("homeName")
        away_name = m.get("awayName")
        
        ra = get_r(home_slug, home_name)
        rb = get_r(away_slug, away_name)
        
        home_bonus = (HOME_ADV / 2.0) if (home_slug and home_slug in HOSTS) else 0.0
        exp = expected_score(ra, rb, home_bonus)
        
        hg, ag = m["hg"], m["ag"]
        score = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
        
        k = base_k(m.get("leagueName", "")) * recency(m["ts"], now_sec) * g_mult(hg - ag)
        delta = k * (score - exp)
        
        set_r(home_slug, home_name, ra + delta)
        set_r(away_slug, away_name, rb - delta)
        applied += 1

    # Apply shrinkage estimator: 70% calibrated + 30% prior
    calibrated_ratings = {}
    for slug, seed_val in SEED.items():
        r_val = ratings_state.get(slug, seed_val)
        calibrated_ratings[slug] = round(0.7 * r_val + 0.3 * seed_val)
        
    return calibrated_ratings, applied
