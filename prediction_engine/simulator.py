# prediction_engine/simulator.py
import json
import os
import random
import math
from collections import defaultdict

# Import the predictor (will be set externally)
# We'll rely on an attribute `self.predictor` set after instantiation.

class TournamentSimulator:
    def __init__(self, data_dir=None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base_dir, "data")
            
        # Load Elo ratings and host info from prediction_engine.elo
        from prediction_engine.elo import SEED, HOSTS
        self.ratings = SEED.copy()
        self.hosts = set(HOSTS)
        self.world_cup_teams = list(self.ratings.keys())
        
        # Path to results for conditioning (optional)
        self.wc_results_path = os.path.join(data_dir, "wc2026-results.json")
        
        # Load finished games if exists (for conditioning)
        self.finished_matches = []
        if os.path.exists(self.wc_results_path):
            with open(self.wc_results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.finished_matches = data.get("matches", [])
        
        # Placeholder for predictor, to be set by main.py
        self.predictor = None

    def _get_match_prediction(self, team1, team2, home_team=None):
        """
        Uses the predictor to get win/draw/loss probabilities and expected goals.
        Returns dict with keys:
            win_prob_team1: probability team1 wins (win + 0.5*draw)
            win_prob_team2: probability team2 wins (win + 0.5*draw)
            expected_goals_team1: expected goals for team1
            expected_goals_team2: expected goals for team2
        """
        if self.predictor is None:
            # Fallback: use simple elo-based win prob (no draw)
            r1 = self.ratings.get(team1, 1500)
            r2 = self.ratings.get(team2, 1500)
            # Simple logistic
            diff = r1 - r2
            p_win = 1.0 / (1.0 + 10 ** (-diff / 400.0))
            # Assume no draw
            return {
                "win_prob_team1": p_win,
                "win_prob_team2": 1.0 - p_win,
                "expected_goals_team1": 1.5,  # dummy
                "expected_goals_team2": 1.5
            }
        # Use predictor
        res = self.predictor.predict_match(
            team1, team2, home_team,
            weights={"stat": 0.61, "ann": 0.39}
        )
        winA = res.get("winA", 0.0)
        draw = res.get("draw", 0.0)
        winB = res.get("winB", 0.0)
        exp1 = res.get("expectedGoalsA", 0.0)
        exp2 = res.get("expectedGoalsB", 0.0)
        # Probability team1 wins in knockout sense (win or half draw)
        p1 = winA + 0.5 * draw
        p2 = winB + 0.5 * draw
        # Normalize just in case (should sum to 1)
        total = p1 + p2
        if total > 0:
            p1 /= total
            p2 /= total
        return {
            "win_prob_team1": p1,
            "win_prob_team2": p2,
            "expected_goals_team1": exp1,
            "expected_goals_team2": exp2
        }
    def _simulate_match(self, team1, team2, home_team=None, knockout=False):
        """
        Simulate a single match between team1 and team2.
        If knockout=True, treat draw as random 50/50 win.
        Returns winner team name.
        Also updates internal statistics via the accumulator passed from caller.
        """
        pred = self._get_match_prediction(team1, team2, home_team)
        p1 = pred["win_prob_team1"]
        p2 = pred["win_prob_team2"]
        exp1 = pred["expected_goals_team1"]
        exp2 = pred["expected_goals_team2"]
        # For simplicity, we simulate goals from Poisson distribution to get a winner
        # but we only need winner for progression; we also need to acknowledge draw handling.
        # We'll sample goals from Poisson with means exp1, exp2.
        try:
            import numpy as np
            g1 = np.random.poisson(exp1)
            g2 = np.random.poisson(exp2)
        except Exception:
            # fallback: binomial approximation
            g1 = 1 if random.random() < min(exp1, 1) else 0
            g2 = 1 if random.random() < min(exp2, 1) else 0
        if g1 > g2:
            winner = team1
        elif g2 > g1:
            winner = team2
        else:
            # draw
            if knockout:
                # random winner
                winner = team1 if random.random() < 0.5 else team2
            else:
                # In group stage, draw is possible; we keep draw as outcome but for
                # advancement we need points etc. We'll handle outside.
                return None  # indicate draw
        # Return winner and goals for possible use
        return winner, g1, g2

    def simulate_knockouts(self, num_sims=10000, bracket_teams=None):
        """
        Run Monte Carlo simulation of knockout stage only.
        bracket_teams: list of 16 team slugs (ordered 1..16) for seeding.
        If None, we will take top 16 by rating.
        Returns dict with:
            - advancement: dict team -> {r32, r16, qf, sf, final, win} probabilities (0-100)
            - avg_expected_goals: dict team -> average expected goals per match played
            - avg_win_prob: dict team -> average win probability per match played
            - bracket_matchups: list of rounds, each list of matchups with
                                {team1, team2, win_prob1, win_prob2,
                                 expected_goals1, expected_goals2}
            - belief_confidence: float 0-100
        """
        if bracket_teams is None:
            # Use only teams available in the predictor to avoid unknown team crashes
            available_teams = self.world_cup_teams
            if self.predictor and hasattr(self.predictor, 'predictor') and self.predictor.predictor.ratings:
                available_teams = [t for t in self.world_cup_teams if t in self.predictor.predictor.ratings]
            
            # Sort by rating descending
            sorted_teams = sorted(available_teams,
                                  key=lambda t: self.ratings.get(t, 0),
                                  reverse=True)
            bracket_teams = sorted_teams[:16]  # take top 16
        # Build bracket: standard seeding 1..16
        # We'll define rounds as lists of matchups (tuples of team indices in bracket_teams list)
        # Round of 32 indices (0-based):
        r32_pairs = [
            (0, 15), (7, 8), (3, 12), (4, 11),
            (1, 14), (6, 9), (2, 13), (5, 10)
        ]
        # Prepare accumulators
        advancement_counts = {t: {"r32": 0, "r16": 0, "qf": 0, "sf": 0, "final": 0, "win": 0} for t in bracket_teams}
        exp_goals_sum = {t: 0.0 for t in bracket_teams}
        win_prob_sum = {t: 0.0 for t in bracket_teams}
        match_count = {t: 0 for t in bracket_teams}
        
        for _ in range(num_sims):
            # Simulate each round
            # Initialize slot mapping: slot i -> team bracket_teams[i]
            slots = bracket_teams[:]  # list length 16
            
            # Round 32
            winners = []
            for i, (idx1, idx2) in enumerate(r32_pairs):
                t1 = slots[idx1]
                t2 = slots[idx2]
                pred = self._get_match_prediction(t1, t2, None)
                p1 = pred["win_prob_team1"]
                p2 = 1.0 - p1  # since we normalized
                # Determine winner
                if random.random() < p1:
                    winner = t1
                else:
                    winner = t2
                winners.append(winner)
                # Update stats for both teams
                exp_goals_sum[t1] += pred["expected_goals_team1"]
                exp_goals_sum[t2] += pred["expected_goals_team2"]
                win_prob_sum[t1] += pred["win_prob_team1"]
                win_prob_sum[t2] += 1.0 - pred["win_prob_team1"]  # which is win_prob_team2
                match_count[t1] += 1
                match_count[t2] += 1
                if winner == t1:
                    advancement_counts[t1]["r32"] += 1
                else:
                    advancement_counts[t2]["r32"] += 1
            # Round 16: pairs of winners
            r16_pairs = [(0,1), (2,3), (4,5), (6,7)]  # indexes in winners list
            r16_winners = []
            for i, (j, k) in enumerate(r16_pairs):
                t1 = winners[j]
                t2 = winners[k]
                pred = self._get_match_prediction(t1, t2, None)
                p1 = pred["win_prob_team1"]
                p2 = 1.0 - p1
                if random.random() < p1:
                    winner = t1
                else:
                    winner = t2
                r16_winners.append(winner)
                # stats
                exp_goals_sum[t1] += pred["expected_goals_team1"]
                exp_goals_sum[t2] += pred["expected_goals_team2"]
                win_prob_sum[t1] += pred["win_prob_team1"]
                win_prob_sum[t2] += 1.0 - pred["win_prob_team1"]
                match_count[t1] += 1
                match_count[t2] += 1
                if winner == t1:
                    advancement_counts[t1]["r16"] += 1
                else:
                    advancement_counts[t2]["r16"] += 1
            # Quarter finals: pairs of r16_winners
            qf_pairs = [(0,1), (2,3)]
            qf_winners = []
            for i, (j, k) in enumerate(qf_pairs):
                t1 = r16_winners[j]
                t2 = r16_winners[k]
                pred = self._get_match_prediction(t1, t2, None)
                p1 = pred["win_prob_team1"]
                p2 = 1.0 - p1
                if random.random() < p1:
                    winner = t1
                else:
                    winner = t2
                qf_winners.append(winner)
                # stats
                exp_goals_sum[t1] += pred["expected_goals_team1"]
                exp_goals_sum[t2] += pred["expected_goals_team2"]
                win_prob_sum[t1] += pred["win_prob_team1"]
                win_prob_sum[t2] += 1.0 - pred["win_prob_team1"]
                match_count[t1] += 1
                match_count[t2] += 1
                if winner == t1:
                    advancement_counts[t1]["qf"] += 1
                else:
                    advancement_counts[t2]["qf"] += 1
            # Semi finals
            sf_pair = (0,1)
            t1 = qf_winners[0]
            t2 = qf_winners[1]
            pred = self._get_match_prediction(t1, t2, None)
            p1 = pred["win_prob_team1"]
            p2 = 1.0 - p1
            if random.random() < p1:
                winner = t1
            else:
                winner = t2
            # stats
            exp_goals_sum[t1] += pred["expected_goals_team1"]
            exp_goals_sum[t2] += pred["expected_goals_team2"]
            win_prob_sum[t1] += pred["win_prob_team1"]
            win_prob_sum[t2] += 1.0 - pred["win_prob_team1"]
            match_count[t1] += 1
            match_count[t2] += 1
            if winner == t1:
                advancement_counts[t1]["sf"] += 1
                advancement_counts[t1]["final"] += 1  # finalist
                advancement_counts[t2]["sf"] += 1
            else:
                advancement_counts[t2]["sf"] += 1
                advancement_counts[t2]["final"] += 1
                advancement_counts[t1]["sf"] += 1
            # Final
            if winner == t1:
                advancement_counts[t1]["win"] += 1
                finalist_loser = t2
            else:
                advancement_counts[t2]["win"] += 1
                finalist_loser = t1
            # loser already got final count above
        
        # Convert counts to percentages
        advancement = {}
        avg_exp_goals = {}
        avg_win_prob = {}
        for t in bracket_teams:
            total_matches = max(1, match_count[t])  # avoid div zero
            advancement[t] = {
                k: (v / num_sims) * 100.0 for k, v in advancement_counts[t].items()
            }
            avg_exp_goals[t] = (exp_goals_sum[t] / total_matches) if total_matches > 0 else 0.0
            avg_win_prob[t] = (win_prob_sum[t] / total_matches) if total_matches > 0 else 0.0
        
        # Build bracket_matchups (deterministic based on seeding and predictor)
        bracket_matchups = []
        # Helper to get deterministic winner (higher rating)
        def get_det_winner(t1, t2):
            return t1 if self.ratings.get(t1, 0) >= self.ratings.get(t2, 0) else t2
        
        # Round 32 matchups
        r32_list = []
        for idx1, idx2 in r32_pairs:
            t1 = bracket_teams[idx1]
            t2 = bracket_teams[idx2]
            pred = self._get_match_prediction(t1, t2, None)
            r32_list.append({
                "team1": t1,
                "team2": t2,
                "win_prob1": round(pred["win_prob_team1"] * 100, 2),
                "win_prob2": round(pred["win_prob_team2"] * 100, 2),
                "expected_goals1": round(pred["expected_goals_team1"], 2),
                "expected_goals2": round(pred["expected_goals_team2"], 2)
            })
        bracket_matchups.append(r32_list)
        # Round 16 matchups: based on winners of r32 assuming higher rating wins
        r32_winners_det = []
        for idx1, idx2 in r32_pairs:
            t1 = bracket_teams[idx1]
            t2 = bracket_teams[idx2]
            w = get_det_winner(t1, t2)
            r32_winners_det.append(w)
        r16_list = []
        r16_pairs_det = [(0,1), (2,3), (4,5), (6,7)]
        for i, (j, k) in enumerate(r16_pairs_det):
            t1 = r32_winners_det[j]
            t2 = r32_winners_det[k]
            pred = self._get_match_prediction(t1, t2, None)
            r16_list.append({
                "team1": t1,
                "team2": t2,
                "win_prob1": round(pred["win_prob_team1"] * 100, 2),
                "win_prob2": round(pred["win_prob_team2"] * 100, 2),
                "expected_goals1": round(pred["expected_goals_team1"], 2),
                "expected_goals2": round(pred["expected_goals_team2"], 2)
            })
        bracket_matchups.append(r16_list)
        # Quarter finals
        qf_winners_det = []
        for i, (j, k) in enumerate(r16_pairs_det):
            w = get_det_winner(r32_winners_det[j], r32_winners_det[k])
            qf_winners_det.append(w)
        qf_list = []
        qf_pairs_det = [(0,1), (2,3)]
        for i, (j, k) in enumerate(qf_pairs_det):
            t1 = qf_winners_det[j]
            t2 = qf_winners_det[k]
            pred = self._get_match_prediction(t1, t2, None)
            qf_list.append({
                "team1": t1,
                "team2": t2,
                "win_prob1": round(pred["win_prob_team1"] * 100, 2),
                "win_prob2": round(pred["win_prob_team2"] * 100, 2),
                "expected_goals1": round(pred["expected_goals_team1"], 2),
                "expected_goals2": round(pred["expected_goals_team2"], 2)
            })
        bracket_matchups.append(qf_list)
        # Semi finals
        sf_winners_det = []
        for i, (j, k) in enumerate(qf_pairs_det):
            w = get_det_winner(qf_winners_det[j], qf_winners_det[k])
            sf_winners_det.append(w)
        sf_list = []
        sf_pairs_det = [(0,1)]
        for i, (j, k) in enumerate(sf_pairs_det):
            t1 = sf_winners_det[j]
            t2 = sf_winners_det[k]
            pred = self._get_match_prediction(t1, t2, None)
            sf_list.append({
                "team1": t1,
                "team2": t2,
                "win_prob1": round(pred["win_prob_team1"] * 100, 2),
                "win_prob2": round(pred["win_prob_team2"] * 100, 2),
                "expected_goals1": round(pred["expected_goals_team1"], 2),
                "expected_goals2": round(pred["expected_goals_team2"], 2)
            })
        bracket_matchups.append(sf_list)
        # Final
        t1 = sf_winners_det[0]
        t2 = sf_winners_det[1]
        pred = self._get_match_prediction(t1, t2, None)
        f_list = [{
            "team1": t1,
            "team2": t2,
            "win_prob1": round(pred["win_prob_team1"] * 100, 2),
            "win_prob2": round(pred["win_prob_team2"] * 100, 2),
            "expected_goals1": round(pred["expected_goals_team1"], 2),
            "expected_goals2": round(pred["expected_goals_team2"], 2)
        }]
        bracket_matchups.append(f_list)
        
        # Compute champion probabilities etc already in advancement['win']
        # Compute confidence from championship distribution
        champ_probs = {t: advancement[t]["win"] / 100.0 for t in bracket_teams}
        total_prob = sum(champ_probs.values())
        if total_prob > 0:
            norm = {t: p/total_prob for t, p in champ_probs.items()}
        else:
            norm = {t: 0.0 for t in champ_probs}
        entropy = 0.0
        for p in norm.values():
            if p > 0:
                entropy -= p * math.log(p)
        max_entropy = math.log(len(bracket_teams)) if len(bracket_teams) > 0 else 1.0
        if max_entropy > 0:
            confidence = (1.0 - entropy / max_entropy) * 100.0
        else:
            confidence = 100.0
        # Ensure within bounds
        confidence = max(0.0, min(100.0, confidence))
        
        result = {
            "advancement": advancement,
            "avg_expected_goals": avg_exp_goals,
            "avg_win_prob": avg_win_prob,
            "belief_confidence": round(confidence, 2),
            "bracket_matchups": bracket_matchups,
            "conditioned": True  # we always condition on using top 16 by rating
        }
        return result

# End of class
