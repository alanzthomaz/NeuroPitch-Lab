# prediction_engine/predictor.py
import os
import json
from prediction_engine.poisson import match_prob

def slugify(team_name):
    if not team_name:
        return ""
    name = team_name.lower().strip()
    name = name.replace(" & ", "-and-")
    name = name.replace(" and ", "-and-")
    name = name.replace(" ", "-")
    name = name.replace("'", "")
    # Remove dots and brackets
    name = name.replace(".", "")
    name = name.replace("(", "").replace(")", "")
    return name

class MatchPredictor:
    def __init__(self, ratings_path=None):
        if ratings_path is None:
            # Default to the copied JSON in the package
            base_dir = os.path.dirname(os.path.abspath(__file__))
            ratings_path = os.path.join(base_dir, "data", "elo-calibrated.json")
            
        self.ratings = {}
        if os.path.exists(ratings_path):
            with open(ratings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.ratings = data.get("ratings", {})
        else:
            # Fallback to importing SEED from elo if file doesn't exist
            from prediction_engine.elo import SEED
            self.ratings = SEED.copy()

    def get_rating(self, team_name):
        slug = slugify(team_name)
        if slug in self.ratings:
            return slug, self.ratings[slug]
        # Direct lookup fallback
        if team_name in self.ratings:
            return team_name, self.ratings[team_name]
        return slug, None

    def predict_match(self, team1, team2, home_team=None):
        """
        Predict match probabilities between team1 and team2.
        home_team can be team1, team2, or None/neutral.
        Returns a dict of probabilities and expected goals.
        """
        slug1, r1 = self.get_rating(team1)
        slug2, r2 = self.get_rating(team2)
        
        if r1 is None:
            raise ValueError(f"Unknown team: {team1}")
        if r2 is None:
            raise ValueError(f"Unknown team: {team2}")
            
        # Determine home bonus (75 if home, -75 if opponent home, 0 if neutral)
        home_bonus = 0.0
        if home_team:
            home_slug = slugify(home_team)
            if home_slug == slug1 or home_team == team1:
                home_bonus = 75.0
            elif home_slug == slug2 or home_team == team2:
                home_bonus = -75.0
                
        prob = match_prob(r1, r2, home_bonus)
        
        return {
            "team1": team1,
            "team2": team2,
            "team1_slug": slug1,
            "team2_slug": slug2,
            "team1_elo": r1,
            "team2_elo": r2,
            "home_bonus": home_bonus,
            "winA": prob["winA"],
            "draw": prob["draw"],
            "winB": prob["winB"],
            "expectedGoalsA": prob["expectedGoalsA"],
            "expectedGoalsB": prob["expectedGoalsB"]
        }
