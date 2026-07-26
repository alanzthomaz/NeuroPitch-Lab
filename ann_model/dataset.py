# ann_model/dataset.py
import os
import urllib.request
import pandas as pd
import numpy as np
import datetime
from prediction_engine.predictor import slugify
from prediction_engine.elo import SEED, expected_score, expected_goals, base_k, g_mult, recency, HOSTS
from prediction_engine.poisson import match_prob

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
FIFA_RANKINGS_URL = "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/ranking_fifa_historical.csv"

def download_data(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    
    results_path = os.path.join(data_dir, "results.csv")
    fifa_path = os.path.join(data_dir, "ranking_fifa_historical.csv")
    
    if not os.path.exists(results_path):
        print("Downloading international match results database...")
        urllib.request.urlretrieve(RESULTS_URL, results_path)
        print("Downloaded results.csv successfully.")
        
    if not os.path.exists(fifa_path):
        print("Downloading FIFA rankings database...")
        urllib.request.urlretrieve(FIFA_RANKINGS_URL, fifa_path)
        print("Downloaded ranking_fifa_historical.csv successfully.")
        
    return results_path, fifa_path

def build_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    results_path, fifa_path = download_data(data_dir)
    
    print("Loading datasets...")
    df_matches = pd.read_csv(results_path)
    df_fifa = pd.read_csv(fifa_path)
    
    # 1. Standardize team names
    df_matches["home_slug"] = df_matches["home_team"].apply(slugify)
    df_matches["away_slug"] = df_matches["away_team"].apply(slugify)
    
    # Calculate rank dynamically from total_points grouped by date
    df_fifa = df_fifa.sort_values(by=["date", "total_points"], ascending=[True, False])
    df_fifa["rank"] = df_fifa.groupby("date")["total_points"].rank(ascending=False, method="first")
    
    df_fifa["team_slug"] = df_fifa["team"].apply(slugify)
    df_fifa["rank_date"] = pd.to_datetime(df_fifa["date"])
    df_fifa = df_fifa.sort_values("rank_date")
    
    df_matches["date"] = pd.to_datetime(df_matches["date"])
    df_matches = df_matches.sort_values("date").reset_index(drop=True)
    
    # Helper to retrieve closest FIFA ranking before or on match date
    print("Preparing FIFA rankings lookup tables...")
    # Group by team slug and store as dictionary of dates and ranks
    fifa_lookup = {}
    for team, group in df_fifa.groupby("team_slug"):
        fifa_lookup[team] = {
            "dates": group["rank_date"].values,
            "ranks": group["rank"].values
        }
        
    def get_fifa_rank(team_slug, match_date):
        if team_slug not in fifa_lookup:
            return 150.0 # Default rank for unknown teams
        dates = fifa_lookup[team_slug]["dates"]
        ranks = fifa_lookup[team_slug]["ranks"]
        # Find index of closest date <= match_date
        idx = np.searchsorted(dates, np.datetime64(match_date)) - 1
        if idx < 0:
            return ranks[0]
        return float(ranks[min(idx, len(ranks) - 1)])

    print("Iterating through matches to compute walk-forward Elo and statistics...")
    # Maintain run-time Elo, rest days, and match history
    ratings = {}
    # Load initial Elo ratings from backup file if exists to keep qualified teams
    backup_elo_path = os.path.join(base_dir, "..", "prediction_engine", "data", "elo-calibrated.json.backup")
    if os.path.exists(backup_elo_path):
        import json
        with open(backup_elo_path, "r", encoding="utf-8") as f:
            ratings = json.load(f).get("ratings", {}).copy()
            
    team_history = {} # slug -> list of dicts: {"date": date, "goals": goals, "conceded": conceded, "pts": pts}
    h2h_history = {}  # (slug1, slug2) -> list of dicts
    
    def get_elo(slug):
        if slug not in ratings:
            ratings[slug] = SEED[slug] if slug in SEED else 1500.0
        return ratings[slug]
        
    processed_rows = []
    
    # We want to train on matches from 2018 onwards, but we'll run the Elo updates 
    # from a bit earlier (e.g. 2015) to let ratings calibrate properly.
    start_date = pd.to_datetime("2018-01-01")
    calibration_start = pd.to_datetime("2015-01-01")
    
    df_filtered = df_matches[df_matches["date"] >= calibration_start]
    
    # Exclude any 2026 FIFA World Cup matches that might already be in df_filtered to prevent duplication
    df_filtered = df_filtered[~((df_filtered["date"] >= pd.to_datetime("2026-06-11")) & (df_filtered["tournament"] == "FIFA World Cup"))]
    
    # Load and append finished matches from wc2026-results.json
    import json
    wc_results_path = os.path.join(base_dir, "..", "prediction_engine", "data", "wc2026-results.json")
    wc_matches = []
    if os.path.exists(wc_results_path):
        with open(wc_results_path, "r", encoding="utf-8") as f:
            wc_data = json.load(f)
            for m in wc_data.get("matches", []):
                if m.get("status") in ["FT", "AET", "PEN"]:
                    wc_matches.append({
                        "date": pd.to_datetime(m["date"]),
                        "home_team": m["team1"],
                        "away_team": m["team2"],
                        "home_score": m["g1"],
                        "away_score": m["g2"],
                        "tournament": "FIFA World Cup",
                        "neutral": not (m["t1"] in HOSTS),
                        "home_slug": m["t1"],
                        "away_slug": m["t2"],
                        "poss_home": m.get("poss1"),
                        "poss_away": m.get("poss2"),
                        "shots_home": m.get("shots1"),
                        "shots_away": m.get("shots2")
                    })
    if wc_matches:
        df_wc = pd.DataFrame(wc_matches)
        df_filtered = pd.concat([df_filtered, df_wc], ignore_index=True)
        
    df_filtered = df_filtered.sort_values("date").reset_index(drop=True)
    
    now_sec = df_filtered.iloc[-1]["date"].timestamp() if not df_filtered.empty else datetime.datetime.now().timestamp()
    
    for idx, row in df_filtered.iterrows():
        date = row["date"]
        home = row["home_slug"]
        away = row["away_slug"]
        
        hg = row["home_score"]
        ag = row["away_score"]
        
        if pd.isna(hg) or pd.isna(ag):
            continue
            
        hg = int(hg)
        ag = int(ag)
        
        # Get Elo BEFORE match
        elo_home = get_elo(home)
        elo_away = get_elo(away)
        
        # Calculate stats for training set (2018 onwards)
        if date >= start_date:
            # 1. FIFA Rankings
            rank_home = get_fifa_rank(home, date)
            rank_away = get_fifa_rank(away, date)
            
            # 2. Multi-Window Rolling Form (5, 10, 20 match windows)
            def compute_rolling_stats(slug):
                hist = team_history.get(slug, [])
                if not hist:
                    default_res = {
                        "form_5": 1.0, "gs_5": 1.0, "gc_5": 1.0, "gd_5": 0.0, "win_rate_5": 0.33, "draw_rate_5": 0.33, "loss_rate_5": 0.33, "cs_5": 0.2, "btts_5": 0.5,
                        "form_10": 1.0, "gs_10": 1.0, "gc_10": 1.0, "gd_10": 0.0, "win_rate_10": 0.33, "draw_rate_10": 0.33, "loss_rate_10": 0.33, "cs_10": 0.2, "btts_10": 0.5,
                        "form_20": 1.0, "gs_20": 1.0, "gc_20": 1.0, "gd_20": 0.0, "win_rate_20": 0.33, "draw_rate_20": 0.33, "loss_rate_20": 0.33, "cs_20": 0.2, "btts_20": 0.5,
                        "rest_days": 30.0
                    }
                    return default_res

                prior_all = [h for h in hist if h["date"] < date]
                if not prior_all:
                    default_res = {
                        "form_5": 1.0, "gs_5": 1.0, "gc_5": 1.0, "gd_5": 0.0, "win_rate_5": 0.33, "draw_rate_5": 0.33, "loss_rate_5": 0.33, "cs_5": 0.2, "btts_5": 0.5,
                        "form_10": 1.0, "gs_10": 1.0, "gc_10": 1.0, "gd_10": 0.0, "win_rate_10": 0.33, "draw_rate_10": 0.33, "loss_rate_10": 0.33, "cs_10": 0.2, "btts_10": 0.5,
                        "form_20": 1.0, "gs_20": 1.0, "gc_20": 1.0, "gd_20": 0.0, "win_rate_20": 0.33, "draw_rate_20": 0.33, "loss_rate_20": 0.33, "cs_20": 0.2, "btts_20": 0.5,
                        "rest_days": 30.0
                    }
                    return default_res

                res = {}
                for window in [5, 10, 20]:
                    sub = prior_all[-window:]
                    n = len(sub)
                    res[f"form_{window}"] = sum(h["pts"] for h in sub) / n if n else 1.0
                    res[f"gs_{window}"] = sum(h["goals"] for h in sub) / n if n else 1.0
                    res[f"gc_{window}"] = sum(h["conceded"] for h in sub) / n if n else 1.0
                    res[f"gd_{window}"] = (sum(h["goals"] for h in sub) - sum(h["conceded"] for h in sub)) / n if n else 0.0
                    res[f"win_rate_{window}"] = sum(1 for h in sub if h["pts"] == 3) / n if n else 0.33
                    res[f"draw_rate_{window}"] = sum(1 for h in sub if h["pts"] == 1) / n if n else 0.33
                    res[f"loss_rate_{window}"] = sum(1 for h in sub if h["pts"] == 0) / n if n else 0.33
                    res[f"cs_{window}"] = sum(1 for h in sub if h["conceded"] == 0) / n if n else 0.2
                    res[f"btts_{window}"] = sum(1 for h in sub if h["goals"] > 0 and h["conceded"] > 0) / n if n else 0.5

                last_match_date = prior_all[-1]["date"]
                res["rest_days"] = min(30.0, float((date - last_match_date).days))
                return res

            stats_h = compute_rolling_stats(home)
            stats_a = compute_rolling_stats(away)

            # 3. Head-to-Head (H2H)
            h2h_key = tuple(sorted([home, away]))
            h2h_matches = [hm for hm in h2h_history.get(h2h_key, []) if hm.get("date", date) < date]
            
            h2h_wins_home = 0.0
            h2h_wins_away = 0.0
            h2h_draws = 0.0
            h2h_games = len(h2h_matches)
            
            for hm in h2h_matches:
                if hm["home"] == home:
                    if hm["hg"] > hm["ag"]: h2h_wins_home += 1
                    elif hm["hg"] < hm["ag"]: h2h_wins_away += 1
                    else: h2h_draws += 1
                else:
                    if hm["hg"] > hm["ag"]: h2h_wins_away += 1
                    elif hm["hg"] < hm["ag"]: h2h_wins_home += 1
                    else: h2h_draws += 1
                        
            h2h_win_rate_home = h2h_wins_home / h2h_games if h2h_games > 0 else 0.33
            h2h_win_rate_away = h2h_wins_away / h2h_games if h2h_games > 0 else 0.33
            
            # 4. Home Advantage
            is_neutral = bool(row["neutral"])
            home_adv = 1.0 if not is_neutral else 0.0
            
            # 5. Engine probabilities
            home_bonus = 75.0 if not is_neutral else 0.0
            probs = match_prob(elo_home, elo_away, home_bonus)
            
            # 6. Possession & Shots
            if "poss_home" in row and not pd.isna(row["poss_home"]):
                poss_h = float(row["poss_home"]) * 100.0
                poss_a = float(row["poss_away"]) * 100.0
            else:
                poss_h = 50.0 + (elo_home - elo_away) / 10.0 + np.random.normal(0, 3)
                poss_h = max(30.0, min(70.0, poss_h))
                poss_a = 100.0 - poss_h
                
            if "shots_home" in row and not pd.isna(row["shots_home"]):
                shots_h = float(row["shots_home"])
                shots_a = float(row["shots_away"])
            else:
                shots_h = max(3.0, round(probs["expectedGoalsA"] * 6.5 + np.random.normal(0, 2)))
                shots_a = max(3.0, round(probs["expectedGoalsB"] * 6.5 + np.random.normal(0, 2)))
            
            player_avail_h = 1.0
            player_avail_a = 1.0
            
            # Target outcome (0 = Home Win, 1 = Draw, 2 = Away Win)
            outcome = 0 if hg > ag else (2 if hg < ag else 1)
            
            row_dict = {
                "date": date,
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_slug": home,
                "away_slug": away,
                "elo_home": elo_home,
                "elo_away": elo_away,
                "rank_home": rank_home,
                "rank_away": rank_away,
                "rank_diff": rank_home - rank_away,
                "form_home": stats_h["form_5"],
                "form_away": stats_a["form_5"],
                "gs_home": stats_h["gs_5"],
                "gs_away": stats_a["gs_5"],
                "gc_home": stats_h["gc_5"],
                "gc_away": stats_a["gc_5"],
                "win_rate_home": stats_h["win_rate_10"],
                "win_rate_away": stats_a["win_rate_10"],
                "h2h_win_rate_home": h2h_win_rate_home,
                "h2h_win_rate_away": h2h_win_rate_away,
                "home_adv": home_adv,
                "rest_home": stats_h["rest_days"],
                "rest_away": stats_a["rest_days"],
                "poss_home": poss_h / 100.0,
                "poss_away": poss_a / 100.0,
                "shots_home": shots_h,
                "shots_away": shots_a,
                "player_avail_home": player_avail_h,
                "player_avail_away": player_avail_a,
                "stat_prob_home": probs["winA"],
                "stat_prob_draw": probs["draw"],
                "stat_prob_away": probs["winB"],
                "stat_xg_home": probs["expectedGoalsA"],
                "stat_xg_away": probs["expectedGoalsB"],
                "target": outcome
            }

            # Add multi-window metrics for 5, 10, 20
            for w in [5, 10, 20]:
                for metric in ["form", "gs", "gc", "gd", "win_rate", "draw_rate", "loss_rate", "cs", "btts"]:
                    row_dict[f"{metric}_{w}_home"] = stats_h[f"{metric}_{w}"]
                    row_dict[f"{metric}_{w}_away"] = stats_a[f"{metric}_{w}"]

            processed_rows.append(row_dict)

        # Update running Elo ratings
        # Same logic as calibrate.mjs
        home_bonus = (75.0 / 2.0) if home in HOSTS else 0.0
        exp = expected_score(elo_home, elo_away, home_bonus)
        score = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
        
        k = base_k(row.get("tournament", "")) * recency(date.timestamp(), now_sec) * g_mult(hg - ag)
        delta = k * (score - exp)
        
        ratings[home] = elo_home + delta
        ratings[away] = elo_away - delta
        
        # Update team history
        team_history.setdefault(home, []).append({
            "date": date, "goals": hg, "conceded": ag, "pts": 3 if hg > ag else (1 if hg == ag else 0)
        })
        team_history.setdefault(away, []).append({
            "date": date, "goals": ag, "conceded": hg, "pts": 3 if ag > hg else (1 if hg == ag else 0)
        })
        
        # Update H2H history
        h2h_key = tuple(sorted([home, away]))
        h2h_history.setdefault(h2h_key, []).append({
            "home": home, "away": away, "hg": hg, "ag": ag
        })

    df_out = pd.DataFrame(processed_rows)
    out_csv = os.path.join(data_dir, "processed_dataset.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"Dataset compiled and saved to {out_csv} ({len(df_out)} matches).")
    
    # Save team profiles lookup for quick O(1) retrieval during live predictions
    print("Saving current team profiles lookup json...")
    team_profiles = {}
    
    # Get union of all team slugs from SEED, GROUPS, and ratings to capture all World Cup teams
    from prediction_engine.montecarlo import GROUPS
    all_team_slugs = set(SEED.keys())
    for teams in GROUPS.values():
        all_team_slugs.update(teams)
    all_team_slugs.update(ratings.keys())
    
    for slug in all_team_slugs:
        hist = team_history.get(slug, [])
        if hist:
            prior = hist[-5:]
            prior_10 = hist[-10:]
            form = sum(h["pts"] for h in prior) / len(prior) if prior else 1.0
            avg_goals = sum(h["goals"] for h in prior) / len(prior) if prior else 1.0
            avg_conceded = sum(h["conceded"] for h in prior) / len(prior) if prior else 1.0
            win_rate = sum(1 for h in prior_10 if h["pts"] == 3) / len(prior_10) if prior_10 else 0.33
        else:
            form, avg_goals, avg_conceded, win_rate = 1.0, 1.0, 1.0, 0.33
            
        rank = get_fifa_rank(slug, pd.to_datetime("2026-07-12"))
        
        team_profiles[slug] = {
            "slug": slug,
            "rank": rank,
            "form": form,
            "gs": avg_goals,
            "gc": avg_conceded,
            "win_rate": win_rate
        }
        
    profiles_path = os.path.join(data_dir, "team_profiles.json")
    with open(profiles_path, "w", encoding="utf-8") as f:
        import json
        json.dump(team_profiles, f, indent=2)
    print(f"Team profiles lookup saved to {profiles_path}")
    
    # Save calibrated Elo ratings back to elo-calibrated.json
    print("Saving calibrated Elo ratings back to elo-calibrated.json...")
    calibrated_elo_path = os.path.join(base_dir, "..", "prediction_engine", "data", "elo-calibrated.json")
    with open(calibrated_elo_path, "w", encoding="utf-8") as f:
        json.dump({
            "matchesApplied": len(df_filtered),
            "ratings": {slug: round(ratings.get(slug, SEED.get(slug, 1500.0))) for slug in all_team_slugs}
        }, f, indent=2)
    print(f"Calibrated Elo ratings saved to {calibrated_elo_path}")
    
    return df_out

if __name__ == "__main__":
    build_dataset()
