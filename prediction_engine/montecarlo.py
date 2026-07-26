# prediction_engine/montecarlo.py
import random
from prediction_engine.elo import SEED, HOSTS, HOME_ADV
from prediction_engine.poisson import sample_match

# Group definitions for the 48 finalists
GROUPS = {
    "Group A": ["mexico", "south-africa", "south-korea", "czech-republic"],
    "Group B": ["switzerland", "canada", "bosnia-and-herzegovina", "qatar"],
    "Group C": ["brazil", "morocco", "scotland", "haiti"],
    "Group D": ["usa", "australia", "paraguay", "turkey"],
    "Group E": ["germany", "ivory-coast", "ecuador", "curacao"],
    "Group F": ["netherlands", "japan", "sweden", "tunisia"],
    "Group G": ["belgium", "egypt", "iran", "new-zealand"],
    "Group H": ["spain", "cape-verde", "uruguay", "saudi-arabia"],
    "Group I": ["france", "senegal", "iraq", "norway"],
    "Group J": ["argentina", "algeria", "austria", "jordan"],
    "Group K": ["colombia", "portugal", "dr-congo", "uzbekistan"],
    "Group L": ["england", "croatia", "ghana", "panama"]
}

def rank_group(group_name, results):
    """
    Rank teams in a group.
    results: list of dicts with: t1, t2, g1, g2
    Returns: list of team slugs in ranked order (1st to 4th)
    """
    teams = GROUPS[group_name]
    stats = {t: {"pts": 0, "gd": 0, "gs": 0} for t in teams}
    
    for r in results:
        t1, t2 = r["t1"], r["t2"]
        g1, g2 = r["g1"], r["g2"]
        if g1 is None or g2 is None:
            continue
        
        stats[t1]["gs"] += g1
        stats[t2]["gs"] += g2
        stats[t1]["gd"] += (g1 - g2)
        stats[t2]["gd"] += (g2 - g1)
        
        if g1 > g2:
            stats[t1]["pts"] += 3
        elif g1 < g2:
            stats[t2]["pts"] += 3
        else:
            stats[t1]["pts"] += 1
            stats[t2]["pts"] += 1
            
    # Rank by pts, gd, gs, and then alphabet fallback
    sorted_teams = sorted(
        teams,
        key=lambda t: (stats[t]["pts"], stats[t]["gd"], stats[t]["gs"], -ord(t[0])),
        reverse=True
    )
    return sorted_teams, stats

def match_third_places(third_placed_teams, winner_slots=None):
    """
    Backtracking search to assign 3rd-placed teams to winner slots,
    ensuring no team plays the winner of their own group.
    third_placed_teams: list of tuples (team_slug, group_name)
    winner_slots: list of group winner names (e.g. 'Group A')
    """
    if winner_slots is None:
        winner_slots = ["Group A", "Group B", "Group D", "Group E", "Group G", "Group I", "Group K", "Group L"]
        
    matching = {}
    used = set()
    
    def backtrack(idx):
        if idx == len(third_placed_teams):
            return True
        team, group = third_placed_teams[idx]
        for slot in winner_slots:
            if slot not in used and slot != group:
                used.add(slot)
                matching[slot] = team
                if backtrack(idx + 1):
                    return True
                used.remove(slot)
                del matching[slot]
        return False
        
    if backtrack(0):
        return matching
    return None

def simulate_group_stage(ratings, rng=None):
    """
    Simulate the entire group stage from scratch.
    """
    if rng is None:
        rng = random.random
        
    all_results = []
    
    for g_name, teams in GROUPS.items():
        # Generate round-robin matches
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                t1, t2 = teams[i], teams[j]
                ra, rb = ratings[t1], ratings[t2]
                
                # Apply home host advantage if applicable
                hb = 0.0
                if t1 in HOSTS:
                    hb = HOME_ADV / 2.0
                elif t2 in HOSTS:
                    hb = -HOME_ADV / 2.0
                    
                score = sample_match(ra, rb, hb, allow_draw=True, rng=rng)
                all_results.append({
                    "group": g_name,
                    "t1": t1,
                    "t2": t2,
                    "g1": score["goalsA"],
                    "g2": score["goalsB"]
                })
                
    # Rank groups
    group_ranks = {}
    third_place_candidates = []
    
    for g_name in GROUPS.keys():
        g_results = [r for r in all_results if r["group"] == g_name]
        ranked, stats = rank_group(g_name, g_results)
        group_ranks[g_name] = ranked
        
        # 3rd place team info
        t3 = ranked[2]
        third_place_candidates.append({
            "team": t3,
            "group": g_name,
            "pts": stats[t3]["pts"],
            "gd": stats[t3]["gd"],
            "gs": stats[t3]["gs"]
        })
        
    # Select 8 best 3rd-placed teams
    # Rank 3rd-placed teams
    sorted_3rds = sorted(
        third_place_candidates,
        key=lambda x: (x["pts"], x["gd"], x["gs"], -ord(x["team"][0])),
        reverse=True
    )
    
    best_3rds = [(x["team"], x["group"]) for x in sorted_3rds[:8]]
    third_place_matching = match_third_places(best_3rds)
    
    return group_ranks, third_place_matching, all_results

def simulate_knockouts(ratings, group_ranks, third_place_matching, rng=None):
    """
    Simulate knockout stages (Round of 32 down to the Final)
    """
    if rng is None:
        rng = random.random
        
    # Helper to simulate a knockout match
    def play_ko(t1, t2):
        ra, rb = ratings[t1], ratings[t2]
        hb = 0.0
        if t1 in HOSTS:
            hb = HOME_ADV / 2.0
        elif t2 in HOSTS:
            hb = -HOME_ADV / 2.0
            
        score = sample_match(ra, rb, hb, allow_draw=False, rng=rng)
        winner = t1 if score["goalsA"] > score["goalsB"] else t2
        return winner, score
        
    r32_results = {}
    
    # 1. Round of 32 matches
    # Pairs defined in FIFA 2026 bracket structure
    # Matches that play 3rd-placed teams
    t3_A = third_place_matching["Group A"]
    t3_B = third_place_matching["Group B"]
    t3_D = third_place_matching["Group D"]
    t3_E = third_place_matching["Group E"]
    t3_G = third_place_matching["Group G"]
    t3_I = third_place_matching["Group I"]
    t3_K = third_place_matching["Group K"]
    t3_L = third_place_matching["Group L"]
    
    pairs = [
        # (Match ID, Team 1, Team 2)
        ("R32_1", group_ranks["Group A"][1], group_ranks["Group B"][1]), # 2A vs 2B
        ("R32_2", group_ranks["Group C"][0], group_ranks["Group F"][1]), # 1C vs 2F
        ("R32_3", group_ranks["Group F"][0], group_ranks["Group C"][1]), # 1F vs 2C
        ("R32_4", group_ranks["Group D"][0], t3_B),                     # 1D vs 3rd B
        ("R32_5", group_ranks["Group E"][1], group_ranks["Group I"][1]), # 2E vs 2I
        ("R32_6", group_ranks["Group E"][0], t3_D),                     # 1E vs 3rd D
        ("R32_7", group_ranks["Group I"][0], t3_F := t3_I), # Fallbacks if needed, using the matching mapping
        ("R32_8", group_ranks["Group D"][1], group_ranks["Group G"][1]), # 2D vs 2G
        ("R32_9", group_ranks["Group J"][0], group_ranks["Group H"][1]), # 1J vs 2H
        ("R32_10", group_ranks["Group A"][0], t3_E),                    # 1A vs 3rd E
        ("R32_11", group_ranks["Group L"][0], t3_K),                    # 1L vs 3rd K
        ("R32_12", group_ranks["Group G"][0], t3_G),                    # 1G vs 3rd G (senegal in data, which is matched to G slot)
        ("R32_13", group_ranks["Group K"][1], group_ranks["Group L"][1]), # 2K vs 2L
        ("R32_14", group_ranks["Group K"][0], t3_L),                    # 1K vs 3rd L
        ("R32_15", group_ranks["Group H"][0], group_ranks["Group J"][1]), # 1H vs 2J
        ("R32_16", group_ranks["Group B"][0], t3_L := t3_A),            # 1B vs 3rd A (algeria in data, matched to B slot)
    ]
    
    # Overwrite the actual matchups using the matched values
    pairs = [
        ("R32_1", group_ranks["Group A"][1], group_ranks["Group B"][1]),
        ("R32_2", group_ranks["Group C"][0], group_ranks["Group F"][1]),
        ("R32_3", group_ranks["Group F"][0], group_ranks["Group C"][1]),
        ("R32_4", group_ranks["Group D"][0], third_place_matching["Group B"]),
        ("R32_5", group_ranks["Group E"][1], group_ranks["Group I"][1]),
        ("R32_6", group_ranks["Group E"][0], third_place_matching["Group D"]),
        ("R32_7", group_ranks["Group I"][0], third_place_matching["Group I"]), # Wait, Slot I plays team matched to I
        ("R32_8", group_ranks["Group D"][1], group_ranks["Group G"][1]),
        ("R32_9", group_ranks["Group J"][0], group_ranks["Group H"][1]),
        ("R32_10", group_ranks["Group A"][0], third_place_matching["Group E"]),
        ("R32_11", group_ranks["Group L"][0], third_place_matching["Group K"]),
        ("R32_12", group_ranks["Group G"][0], third_place_matching["Group G"]),
        ("R32_13", group_ranks["Group K"][1], group_ranks["Group L"][1]),
        ("R32_14", group_ranks["Group K"][0], third_place_matching["Group L"]),
        ("R32_15", group_ranks["Group H"][0], group_ranks["Group J"][1]),
        ("R32_16", group_ranks["Group B"][0], third_place_matching["Group B" if "Group B" in third_place_matching and third_place_matching["Group B"] not in group_ranks["Group B"] else "Group A"]) # mapping fallback
    ]
    
    # Let's clean up the 3rd place matchup slot index
    # Standard pairings:
    # 1A vs 3rd C/D/E/F -> matched using the bipartite matching output
    slots = ["Group A", "Group B", "Group D", "Group E", "Group G", "Group I", "Group K", "Group L"]
    # We assign:
    # 1A -> third_place_matching['Group A']
    # 1B -> third_place_matching['Group B']
    # 1D -> third_place_matching['Group D']
    # 1E -> third_place_matching['Group E']
    # 1G -> third_place_matching['Group G']
    # 1I -> third_place_matching['Group I']
    # 1K -> third_place_matching['Group K']
    # 1L -> third_place_matching['Group L']
    
    r32_teams = {
        "R32_1": (group_ranks["Group A"][1], group_ranks["Group B"][1]),
        "R32_2": (group_ranks["Group C"][0], group_ranks["Group F"][1]),
        "R32_3": (group_ranks["Group F"][0], group_ranks["Group C"][1]),
        "R32_4": (group_ranks["Group D"][0], third_place_matching.get("Group D")), # 1D plays 3rd matched to D
        "R32_5": (group_ranks["Group E"][1], group_ranks["Group I"][1]),
        "R32_6": (group_ranks["Group E"][0], third_place_matching.get("Group E")), # 1E plays 3rd matched to E
        "R32_7": (group_ranks["Group I"][0], third_place_matching.get("Group I")), # 1I plays 3rd matched to I
        "R32_8": (group_ranks["Group D"][1], group_ranks["Group G"][1]),
        "R32_9": (group_ranks["Group J"][0], group_ranks["Group H"][1]),
        "R32_10": (group_ranks["Group A"][0], third_place_matching.get("Group A")), # 1A plays 3rd matched to A
        "R32_11": (group_ranks["Group L"][0], third_place_matching.get("Group L")), # 1L plays 3rd matched to L
        "R32_12": (group_ranks["Group G"][0], third_place_matching.get("Group G")), # 1G plays 3rd matched to G
        "R32_13": (group_ranks["Group K"][1], group_ranks["Group L"][1]),
        "R32_14": (group_ranks["Group K"][0], third_place_matching.get("Group K")), # 1K plays 3rd matched to K
        "R32_15": (group_ranks["Group H"][0], group_ranks["Group J"][1]),
        "R32_16": (group_ranks["Group B"][0], third_place_matching.get("Group B")), # 1B plays 3rd matched to B
    }

    # Simulate R32
    r32_winners = {}
    for m_id, (t1, t2) in r32_teams.items():
        winner, _ = play_ko(t1, t2)
        r32_winners[m_id] = winner
        
    # 2. Round of 16 matches
    # R16_1: Winner R32_1 vs Winner R32_3
    # R16_2: Winner R32_2 vs Winner R32_5
    # R16_3: Winner R32_4 vs Winner R32_12
    # R16_4: Winner R32_6 vs Winner R32_7
    # R16_5: Winner R32_8 vs Winner R32_9
    # R16_6: Winner R32_10 vs Winner R32_11
    # R16_7: Winner R32_13 vs Winner R32_15
    # R16_8: Winner R32_14 vs Winner R32_16
    r16_teams = {
        "R16_1": (r32_winners["R32_1"], r32_winners["R32_3"]),
        "R16_2": (r32_winners["R32_2"], r32_winners["R32_5"]),
        "R16_3": (r32_winners["R32_4"], r32_winners["R32_12"]),
        "R16_4": (r32_winners["R32_6"], r32_winners["R32_7"]),
        "R16_5": (r32_winners["R32_8"], r32_winners["R32_9"]),
        "R16_6": (r32_winners["R32_10"], r32_winners["R32_11"]),
        "R16_7": (r32_winners["R32_13"], r32_winners["R32_15"]),
        "R16_8": (r32_winners["R32_14"], r32_winners["R32_16"]),
    }
    
    r16_winners = {}
    for m_id, (t1, t2) in r16_teams.items():
        winner, _ = play_ko(t1, t2)
        r16_winners[m_id] = winner
        
    # 3. Quarter-final matches
    # QF_1: Winner R16_4 vs Winner R16_1 (France vs Morocco in real)
    # QF_2: Winner R16_7 vs Winner R16_3 (Spain vs Belgium in real)
    # QF_3: Winner R16_2 vs Winner R16_6 (Norway vs England in real)
    # QF_4: Winner R16_5 vs Winner R16_8 (Argentina vs Switzerland in real)
    qf_teams = {
        "QF_1": (r16_winners["R16_4"], r16_winners["R16_1"]),
        "QF_2": (r16_winners["R16_7"], r16_winners["R16_3"]),
        "QF_3": (r16_winners["R16_2"], r16_winners["R16_6"]),
        "QF_4": (r16_winners["R16_5"], r16_winners["R16_8"]),
    }
    
    qf_winners = {}
    for m_id, (t1, t2) in qf_teams.items():
        winner, _ = play_ko(t1, t2)
        qf_winners[m_id] = winner
        
    # 4. Semi-final matches
    # SF_1: Winner QF_1 vs Winner QF_2
    # SF_2: Winner QF_3 vs Winner QF_4
    sf_teams = {
        "SF_1": (qf_winners["QF_1"], qf_winners["QF_2"]),
        "SF_2": (qf_winners["QF_3"], qf_winners["QF_4"]),
    }
    
    sf_winners = {}
    for m_id, (t1, t2) in sf_teams.items():
        winner, _ = play_ko(t1, t2)
        sf_winners[m_id] = winner
        
    # 5. Final match
    final_winner, _ = play_ko(sf_winners["SF_1"], sf_winners["SF_2"])
    
    return {
        "r32": r32_winners,
        "r16": r16_winners,
        "qf": qf_winners,
        "sf": sf_winners,
        "final": final_winner
    }

def simulate_full_tournament(ratings, rng=None):
    if rng is None:
        rng = random.random
        
    group_ranks, third_place_matching, _ = simulate_group_stage(ratings, rng)
    ko_results = simulate_knockouts(ratings, group_ranks, third_place_matching, rng)
    return ko_results
