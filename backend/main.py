# backend/main.py
import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, List
from pydantic import BaseModel

from ensemble.ensemble_predictor import EnsemblePredictor
from prediction_engine.simulator import TournamentSimulator
from prediction_engine.predictor import slugify
from prediction_engine.elo import SEED, HOSTS

# New imports for the Learning Lab
from ann_model.dynamic_model import DynamicFootballANN
from ann_model.dynamic_trainer import train_and_stream, extract_match_features
from ann_model.explainability import compute_saliency_map, compute_integrated_gradients

app = FastAPI(
    title="NeuroPitch Lab - FIFA World Cup 2026 Prediction API",
    description="NeuroPitch Lab - Ensemble Prediction Platform & Deep Learning Lab",
    version="2.0.0"
)

# Enable CORS for frontend React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import threading

# Initialize default models
predictor = EnsemblePredictor()
simulator = TournamentSimulator()
simulator.predictor = predictor

# Global state for the active dynamic model in the laboratory
ACTIVE_LAB_MODEL = {
    "model": None,
    "features": [],
    "scaler": None,
    "activation": "relu",
    "hidden_layers": [128, 64, 32]
}

model_lock = threading.Lock()

# Rebuild the dataset to synchronize latest tournament matches
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_scaler_path = os.path.join(base_dir, "ann_model", "data", "scaler.joblib")
default_profiles_path = os.path.join(base_dir, "ann_model", "data", "team_profiles.json")
wc_results_path = os.path.join(base_dir, "prediction_engine", "data", "wc2026-results.json")
experiments_path = os.path.join(base_dir, "ann_model", "data", "experiments.json")

try:
    print("Rebuilding dataset at server startup...")
    from ann_model.dataset import build_dataset
    build_dataset()
    print("Dataset compiled successfully on startup.")
    
    # Reload profiles and ratings in prediction engine
    default_ratings_path = os.path.join(base_dir, "prediction_engine", "data", "elo-calibrated.json")
    if os.path.exists(default_profiles_path):
        with open(default_profiles_path, "r", encoding="utf-8") as f:
            predictor.profiles = json.load(f)
    if os.path.exists(default_ratings_path):
        with open(default_ratings_path, "r", encoding="utf-8") as f:
            predictor.predictor.ratings = json.load(f).get("ratings", {})
except Exception as e:
    print(f"Failed to compile dataset on startup: {e}")

# Check if dynamic model exists and load it first for persistence
dynamic_config_path = os.path.join(base_dir, "ann_model", "data", "dynamic_config.json")
dynamic_model_path = os.path.join(base_dir, "ann_model", "data", "dynamic_model.pth")
dynamic_scaler_path = os.path.join(base_dir, "ann_model", "data", "dynamic_scaler.joblib")

loaded_dynamic = False
if os.path.exists(dynamic_config_path) and os.path.exists(dynamic_model_path) and os.path.exists(dynamic_scaler_path):
    try:
        with open(dynamic_config_path, "r", encoding="utf-8") as f:
            d_config = json.load(f)
        d_features = d_config.get("features", [])
        d_hidden_layers = d_config.get("hidden_layers", [16, 8])
        d_activation = d_config.get("activation", "relu")
        
        from ann_model.dynamic_model import DynamicFootballANN
        import joblib
        import torch
        
        d_model = DynamicFootballANN(len(d_features), d_hidden_layers, d_activation)
        d_model.load_state_dict(torch.load(dynamic_model_path, map_location="cpu"))
        d_model.eval()
        d_scaler = joblib.load(dynamic_scaler_path)
        
        ACTIVE_LAB_MODEL["model"] = d_model
        ACTIVE_LAB_MODEL["features"] = d_features
        ACTIVE_LAB_MODEL["scaler"] = d_scaler
        ACTIVE_LAB_MODEL["activation"] = d_activation
        ACTIVE_LAB_MODEL["hidden_layers"] = d_hidden_layers
        loaded_dynamic = True
        print("Loaded custom active lab model from disk successfully.")
    except Exception as e:
        print(f"Failed to load dynamic model on startup: {e}. Falling back to default model.")

if not loaded_dynamic and predictor.ann_model is not None:
    ACTIVE_LAB_MODEL["model"] = predictor.ann_model
    ACTIVE_LAB_MODEL["scaler"] = predictor.scaler
    from ann_model.train import FEATURES
    ACTIVE_LAB_MODEL["features"] = FEATURES
    ACTIVE_LAB_MODEL["activation"] = "relu"
    ACTIVE_LAB_MODEL["hidden_layers"] = [128, 64, 32]

# Map of slug to human readable names
TEAM_NAMES = {slug: slug.replace("-", " ").title() for slug in SEED.keys()}
TEAM_NAMES.update({
    "usa": "USA",
    "dr-congo": "DR Congo",
    "south-korea": "South Korea",
    "south-africa": "South Africa",
    "new-zealand": "New Zealand",
    "saudi-arabia": "Saudi Arabia",
    "ivory-coast": "Ivory Coast",
    "cape-verde": "Cape Verde",
    "czech-republic": "Czech Republic",
    "bosnia-and-herzegovina": "Bosnia & Herzegovina",
    "trinidad-and-tobago": "Trinidad & Tobago",
    "el-salvador": "El Salvador"
})

@app.get("/")
def read_root():
    return {
        "status": "online",
        "api": "NeuroPitch Lab - FIFA World Cup 2026 Prediction",
        "supported_endpoints": ["/predict", "/simulate", "/team/{slug}", "/bracket", "/history", "/api/train/stream", "/api/explain", "/api/experiments"]
    }

# =====================================================================
# REST ENDPOINTS FROM ORIGINAL
# =====================================================================

@app.get("/predict")
def predict_match(
    team1: str = Query(..., description="First team name"),
    team2: str = Query(..., description="Second team name"),
    home_team: Optional[str] = Query(None, description="Team playing at home"),
    stat_weight: float = Query(0.61),
    ann_weight: float = Query(0.39)
):
    try:
        # Check if we should predict using the dynamically trained Lab model
        # or fall back to default ensemble predictor
        with model_lock:
            has_model = ACTIVE_LAB_MODEL["model"] is not None and ACTIVE_LAB_MODEL["scaler"] is not None
            
        if has_model:
            with model_lock:
                stat_pred = predictor.predictor.predict_match(team1, team2, home_team)
                
                slug1 = slugify(team1)
                slug2 = slugify(team2)
                
                # Extract features vector
                feat_vec = extract_match_features(
                    team1, team2, home_team, 
                    ACTIVE_LAB_MODEL["features"], 
                    predictor.profiles, 
                    predictor.predictor.ratings
                )
                
                feat_scaled = ACTIVE_LAB_MODEL["scaler"].transform(feat_vec.reshape(1, -1))
                feat_tensor = torch.FloatTensor(feat_scaled)
                
                ACTIVE_LAB_MODEL["model"].eval()
                with torch.no_grad():
                    outputs = ACTIVE_LAB_MODEL["model"](feat_tensor)
                    ann_probs = torch.softmax(outputs, dim=1).numpy()[0]
                    
                ann_h, ann_d, ann_a = float(ann_probs[0]), float(ann_probs[1]), float(ann_probs[2])
                
                # Blended
                ens_h = stat_weight * stat_pred["winA"] + ann_weight * ann_h
                ens_d = stat_weight * stat_pred["draw"] + ann_weight * ann_d
                ens_a = stat_weight * stat_pred["winB"] + ann_weight * ann_a
                
                total = ens_h + ens_d + ens_a
                ens_h /= total
                ens_d /= total
                ens_a /= total
                
                return {
                    **stat_pred,
                    "team1_name": TEAM_NAMES.get(slug1, team1.title()),
                    "team2_name": TEAM_NAMES.get(slug2, team2.title()),
                    "ann_prob_home": ann_h,
                    "ann_prob_draw": ann_d,
                    "ann_prob_away": ann_a,
                    "ensemble_home": ens_h,
                    "ensemble_draw": ens_d,
                    "ensemble_away": ens_a,
                    "weights": {"stat": stat_weight, "ann": ann_weight}
                }
        else:
            pred = predictor.predict_match(team1, team2, home_team, weights={"stat": stat_weight, "ann": ann_weight})
            pred["team1_name"] = TEAM_NAMES.get(pred["team1_slug"], pred["team1"].title())
            pred["team2_name"] = TEAM_NAMES.get(pred["team2_slug"], pred["team2"].title())
            return pred
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/simulate")
def simulate_tournament(
    num_sims: int = Query(2500, ge=10, le=100000),
    condition: bool = Query(True)
):
    try:
        # Use the new simulator method that returns advancement, bracket, confidence
        result = simulator.simulate_knockouts(num_sims=num_sims, bracket_teams=None)
        # Build title_odds list for all teams (we will include only those in bracket_teams for now)
        # For simplicity, we will include all world cup teams, but set zero advancement for those not in bracket.
        # However our simulate_knockouts only returns stats for bracket_teams.
        # We'll get the list of teams that were simulated from result['advancement'] keys.
        simulated_teams = list(result["advancement"].keys())
        # Build list for all teams
        title_odds = []
        for team in simulator.world_cup_teams:
            if team in result["advancement"]:
                adv = result["advancement"][team]
                name = TEAM_NAMES.get(team, team.title())
                rating = predictor.predictor.ratings.get(team, 1500)
                exp_goals = result["avg_expected_goals"].get(team, 0.0)
                avg_win_prob = result["avg_win_prob"].get(team, 0.0)
                title_odds.append({
                    "slug": team,
                    "name": name,
                    "rating": rating,
                    "r32": round(adv["r32"], 2),
                    "r16": round(adv["r16"], 2),
                    "qf": round(adv["qf"], 2),
                    "sf": round(adv["sf"], 2),
                    "final": round(adv["final"], 2),
                    "win": round(adv["win"], 2),
                    "expected_goals": round(exp_goals, 2),
                    "avg_win_prob": round(avg_win_prob, 2)
                })
            else:
                name = TEAM_NAMES.get(team, team.title())
                rating = predictor.predictor.ratings.get(team, 1500)
                title_odds.append({
                    "slug": team,
                    "name": name,
                    "rating": rating,
                    "r32": 0.0,
                    "r16": 0.0,
                    "qf": 0.0,
                    "sf": 0.0,
                    "final": 0.0,
                    "win": 0.0,
                    "expected_goals": 0.0,
                    "avg_win_prob": 0.0
                })
        # Sort by win probability descending
        title_odds.sort(key=lambda x: x["win"], reverse=True)
        # Build bracket data in the format expected by frontend (list of rounds)
        # result["bracket_matchups"] is already list of rounds, each list of matchups with fields:
        # team1, team2, win_prob1, win_prob2, expected_goals1, expected_goals2
        # We need to convert each matchup to objects with nested team objects.
        bracket_rounds = []
        for round_matchups in result["bracket_matchups"]:
            round_list = []
            for m in round_matchups:
                t1_name = TEAM_NAMES.get(m["team1"], m["team1"].title())
                t2_name = TEAM_NAMES.get(m["team2"], m["team2"].title())
                round_list.append({
                    "team1": {"slug": m["team1"], "name": t1_name},
                    "team2": {"slug": m["team2"], "name": t2_name},
                    "win_prob1": m["win_prob1"],
                    "win_prob2": m["win_prob2"],
                    "expected_goals1": m["expected_goals1"],
                    "expected_goals2": m["expected_goals2"]
                })
            bracket_rounds.append(round_list)
        return {
            "winner_freq": title_odds,
            "bracket": bracket_rounds,
            "confidence": result["belief_confidence"]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/team/{slug}")
def get_team_profile(slug: str):
    slug = slugify(slug)
    if slug not in SEED:
        raise HTTPException(status_code=404, detail=f"Team slug '{slug}' not found.")
    rating = predictor.predictor.ratings.get(slug, SEED[slug])
    profile = predictor.profiles.get(slug, {"rank": 150.0, "form": 1.0, "gs": 1.0, "gc": 1.0, "win_rate": 0.33})
    from prediction_engine.montecarlo import GROUPS
    team_group = "Unknown"
    for g_name, teams in GROUPS.items():
        if slug in teams:
            team_group = g_name
            break
    fixtures = []
    if os.path.exists(wc_results_path):
        with open(wc_results_path, "r", encoding="utf-8") as f:
            matches_data = json.load(f).get("matches", [])
            for m in matches_data:
                if m["t1"] == slug or m["t2"] == slug:
                    fixtures.append({
                        "date": m.get("date"),
                        "round": m.get("round"),
                        "group": m.get("group"),
                        "opponent": m["t2"] if m["t1"] == slug else m["t1"],
                        "opponent_name": TEAM_NAMES.get(m["t2"] if m["t1"] == slug else m["t1"], ""),
                        "score": f"{m['g1']} - {m['g2']}" if m.get("status") in ["FT", "AET", "PEN"] else "vs",
                        "status": m.get("status"),
                        "winner": m.get("winner")
                    })
    return {
        "slug": slug,
        "name": TEAM_NAMES.get(slug, slug.title()),
        "elo": rating,
        "fifa_rank": int(profile["rank"]),
        "form": round(profile["form"], 2),
        "avg_goals_scored": round(profile["gs"], 2),
        "avg_goals_conceded": round(profile["gc"], 2),
        "win_rate": round(profile["win_rate"] * 100, 1),
        "group": team_group,
        "is_host": slug in HOSTS,
        "fixtures": fixtures
    }

@app.get("/bracket")
def get_bracket_state():
    try:
        if not os.path.exists(wc_results_path):
            raise HTTPException(status_code=500, detail="Tournament results file missing.")
        with open(wc_results_path, "r", encoding="utf-8") as f:
            all_matches = json.load(f).get("matches", [])
        knockouts = [m for m in all_matches if m.get("group") is None]
        augmented_bracket = []
        for m in knockouts:
            t1, t2 = m["t1"], m["t2"]
            pred = None
            if t1 and t2:
                home_team = t1 if t1 in HOSTS else (t2 if t2 in HOSTS else None)
                pred = predict_match(t1, t2, home_team, stat_weight=0.61, ann_weight=0.39)
            augmented_bracket.append({
                "round": m.get("round"),
                "date": m.get("date"),
                "team1": t1,
                "team1_name": TEAM_NAMES.get(t1, t1.title()) if t1 else "TBD",
                "team2": t2,
                "team2_name": TEAM_NAMES.get(t2, t2.title()) if t2 else "TBD",
                "score": f"{m['g1']} - {m['g2']}" if m.get("status") in ["FT", "AET", "PEN"] else None,
                "status": m.get("status"),
                "winner": m.get("winner"),
                "winner_name": TEAM_NAMES.get(m.get("winner"), "") if m.get("winner") else None,
                "prediction": {
                    "home_win": pred["ensemble_home"] if pred else 0.0,
                    "draw": pred["ensemble_draw"] if pred else 0.0,
                    "away_win": pred["ensemble_away"] if pred else 0.0,
                    "stat_xg_home": pred["expectedGoalsA"] if pred else 0.0,
                    "stat_xg_away": pred["expectedGoalsB"] if pred else 0.0,
                } if pred else None
            })
        return augmented_bracket
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_prediction_history():
    try:
        if not os.path.exists(wc_results_path):
            raise HTTPException(status_code=500, detail="Tournament results file missing.")
        with open(wc_results_path, "r", encoding="utf-8") as f:
            all_matches = json.load(f).get("matches", [])
        finished = [m for m in all_matches if m.get("status") in ["FT", "AET", "PEN"]]
        history = []
        correct = 0
        total = 0
        for m in finished:
            t1, t2 = m["t1"], m["t2"]
            g1, g2 = m["g1"], m["g2"]
            winner = m.get("winner")
            home_team = t1 if t1 in HOSTS else (t2 if t2 in HOSTS else None)
            pred = predict_match(t1, t2, home_team, stat_weight=0.61, ann_weight=0.39)
            if winner:
                actual = 0 if winner == t1 else 2
            else:
                actual = 0 if g1 > g2 else (2 if g2 > g1 else 1)
            probs = [pred["ensemble_home"], pred["ensemble_draw"], pred["ensemble_away"]]
            predicted = np.argmax(probs)
            is_correct = bool(predicted == actual)
            if is_correct:
                correct += 1
            total += 1
            history.append({
                "date": m.get("date"),
                "round": m.get("round"),
                "team1": t1,
                "team1_name": TEAM_NAMES.get(t1, t1.title()),
                "team2": t2,
                "team2_name": TEAM_NAMES.get(t2, t2.title()),
                "score": f"{g1} - {g2}" + (f" ({m['pens1']}-{m['pens2']} p)" if m.get("status") == "PEN" else ""),
                "status": m.get("status"),
                "prediction": {
                    "win1": round(pred["ensemble_home"] * 100, 1),
                    "draw": round(pred["ensemble_draw"] * 100, 1),
                    "win2": round(pred["ensemble_away"] * 100, 1)
                },
                "correct": is_correct
            })
        return {
            "accuracy": round((correct / total) * 100, 1) if total > 0 else 0.0,
            "correct_count": correct,
            "total_count": total,
            "predictions": history[::-1]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# NEW ANN LEARNING LAB ENDPOINTS
# =====================================================================

@app.get("/api/train/stream")
def stream_training_lab(
    hidden_layers: str = Query("16,8", description="Comma separated hidden layers, e.g. 16,8"),
    activation: str = Query("relu"),
    optimizer: str = Query("adam"),
    learning_rate: float = Query(0.005),
    epochs: int = Query(30),
    batch_size: int = Query(64),
    features: str = Query("elo_home,elo_away,rank_diff,home_adv"),
    l2_regularization: float = Query(1e-4),
    weight_init: str = Query("xavier_uniform"),
    grad_clipping: bool = Query(False),
    loss_func: str = Query("crossentropy"),
    scheduler: str = Query("none"),
    momentum: float = Query(0.9),
    weight_decay: float = Query(1e-4),
    early_stopping: int = Query(0),
    random_seed: int = Query(42),
    team1: str = Query("france"),
    team2: str = Query("spain"),
    home_team: str = Query("france")
):
    """
    Server-Sent Events endpoint that trains the ANN and streams updates to the visualizer.
    """
    layers = [int(x) for x in hidden_layers.split(",") if x.strip()]
    feats = [x.strip() for x in features.split(",") if x.strip()]
    
    config = {
        "hidden_layers": layers,
        "activation": activation,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "features": feats,
        "l2_regularization": l2_regularization,
        "weight_init": weight_init,
        "grad_clipping": grad_clipping,
        "loss_func": loss_func,
        "scheduler": scheduler,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "early_stopping": early_stopping,
        "random_seed": random_seed,
        "reference_match": {
            "team1": team1,
            "team2": team2,
            "home_team": home_team
        }
    }
    
    # Intercept training data generator to cache trained scaler & model dynamically in memory
    def stream_and_cache():
        last_event = None
        for chunk in train_and_stream(config):
            # Parse chunk if it is an event state
            if chunk.startswith("data: "):
                try:
                    payload = json.loads(chunk[6:].strip())
                    if "network_state" in payload:
                        last_event = payload
                except:
                    pass
            yield chunk
            
        # Once training is complete, build dynamic model in memory for explainability/sandbox
        if last_event is not None:
            try:
                # Re-load scaling matrix
                scaler_path = os.path.join(base_dir, "ann_model", "data", "dynamic_scaler.joblib")
                scaler = joblib.load(scaler_path)
                
                # Reconstruct trained model
                input_dim = len(feats)
                trained_model = DynamicFootballANN(
                    input_dim, layers, activation, 
                    init_name=weight_init
                )
                
                # Load weights from the temporary state file
                model_pth = os.path.join(base_dir, "ann_model", "data", "dynamic_model.pth")
                trained_model.load_state_dict(torch.load(model_pth, weights_only=True))
                
                with model_lock:
                    ACTIVE_LAB_MODEL["model"] = trained_model
                    ACTIVE_LAB_MODEL["features"] = feats
                    ACTIVE_LAB_MODEL["scaler"] = scaler
                    ACTIVE_LAB_MODEL["activation"] = activation
                    ACTIVE_LAB_MODEL["hidden_layers"] = layers
                print("Active Lab Model successfully loaded in memory.")
            except Exception as e:
                print(f"Failed to cache lab model: {e}")

    return StreamingResponse(stream_and_cache(), media_type="text/event-stream")

@app.get("/api/explain")
def explain_attributions(
    team1: str = Query(..., description="First team name"),
    team2: str = Query(..., description="Second team name"),
    home_team: Optional[str] = Query(None, description="Home team name"),
    target_class: int = Query(0, description="0=Home Win, 1=Draw, 2=Away Win")
):
    """
    Exposes autograd Saliency Maps and Integrated Gradients attributions for explainability.
    """
    with model_lock:
        has_model = ACTIVE_LAB_MODEL["model"] is not None and ACTIVE_LAB_MODEL["scaler"] is not None
        
    if not has_model:
        raise HTTPException(status_code=400, detail="No active ANN model available in Lab. Please train a network first.")
        
    try:
        with model_lock:
            # Extract features vector
            feat_vec = extract_match_features(
                team1, team2, home_team, 
                ACTIVE_LAB_MODEL["features"], 
                predictor.profiles, 
                predictor.predictor.ratings
            )
            
            # Scale
            feat_scaled = ACTIVE_LAB_MODEL["scaler"].transform(feat_vec.reshape(1, -1))[0]
            
            # Run explainers
            saliency = compute_saliency_map(ACTIVE_LAB_MODEL["model"], feat_scaled, target_class)
            attribution = compute_integrated_gradients(ACTIVE_LAB_MODEL["model"], feat_scaled, target_class)
            
            # Format mapping back to features
            explanation = []
            for i, feat in enumerate(ACTIVE_LAB_MODEL["features"]):
                explanation.append({
                    "feature": feat,
                    "label": feat.replace("_", " ").title(),
                    "value": float(feat_vec[i]),
                    "saliency": float(saliency[i]),
                    "attribution": float(attribution[i])
                })
            
        return {
            "team1": team1,
            "team2": team2,
            "target_class": target_class,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability computation failed: {str(e)}")

# =====================================================================
# EXPERIMENT TRACKER ENDPOINTS
# =====================================================================

@app.get("/api/experiments")
def get_experiments():
    if not os.path.exists(experiments_path):
        return []
    try:
        with open(experiments_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

@app.post("/api/experiments")
def save_experiment(experiment: dict = Body(...)):
    experiments = []
    if os.path.exists(experiments_path):
        try:
            with open(experiments_path, "r", encoding="utf-8") as f:
                experiments = json.load(f)
        except:
            pass
            
    # Add new experiment
    experiments.append(experiment)
    
    try:
        with open(experiments_path, "w", encoding="utf-8") as f:
            json.dump(experiments, f, indent=2)
        return {"status": "success", "count": len(experiments)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save experiment: {e}")

@app.delete("/api/experiments")
def clear_experiments():
    try:
        if os.path.exists(experiments_path):
            os.remove(experiments_path)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SandboxPayload(BaseModel):
    features_override: dict
    weight_overrides: Optional[dict] = None

def apply_weight_overrides(model, overrides):
    if not overrides:
        return
    linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]
    for key, val in overrides.items():
        parts = key.split("_")
        if len(parts) == 3:
            l_idx = int(parts[0])
            s_idx = int(parts[1])
            d_idx = int(parts[2])
            if l_idx < len(linear_layers):
                layer = linear_layers[l_idx]
                if d_idx < layer.weight.data.shape[0] and s_idx < layer.weight.data.shape[1]:
                    layer.weight.data[d_idx, s_idx] = float(val)

@app.post("/api/sandbox/predict")
def sandbox_predict(
    payload: SandboxPayload,
    target_class: int = Query(0)
):
    """
    Computes predictions, neuron activations, and explainability attributions
    for custom overridden inputs and weights in the sandbox playground.
    """
    with model_lock:
        has_model = ACTIVE_LAB_MODEL["model"] is not None and ACTIVE_LAB_MODEL["scaler"] is not None
        
    if not has_model:
        raise HTTPException(status_code=400, detail="No active ANN model available in Lab. Train a network first.")
        
    try:
        import copy
        with model_lock:
            features_list = ACTIVE_LAB_MODEL["features"]
            model_copy = copy.deepcopy(ACTIVE_LAB_MODEL["model"])
            
        # Apply weight overrides to the copied model instance
        apply_weight_overrides(model_copy, payload.weight_overrides)
        
        # Compile overridden feature vector
        vector = []
        for feat in features_list:
            val = payload.features_override.get(feat, 0.0)
            vector.append(float(val))
            
        vector_np = np.array(vector, dtype=float)
        with model_lock:
            scaler = ACTIVE_LAB_MODEL["scaler"]
        feat_scaled = scaler.transform(vector_np.reshape(1, -1))[0]
        
        # Forward pass for custom prediction
        x_tensor = torch.FloatTensor(feat_scaled).unsqueeze(0)
        model_copy.eval()
        with torch.no_grad():
            outputs = model_copy(x_tensor)
            probs = torch.softmax(outputs, dim=1).numpy()[0]
            
        # Capture layer-by-layer activations
        activations = model_copy.get_layer_activations(torch.FloatTensor(feat_scaled))
        
        # Explain predictions using Autograd Saliency & Path Attributions
        saliency = compute_saliency_map(model_copy, feat_scaled, target_class)
        attribution = compute_integrated_gradients(model_copy, feat_scaled, target_class)
        
        explanation = []
        for i, feat in enumerate(features_list):
            explanation.append({
                "feature": feat,
                "label": feat.replace("_", " ").title(),
                "value": float(vector[i]),
                "saliency": float(saliency[i]),
                "attribution": float(attribution[i])
            })
            
        return {
            "prediction": {
                "home": float(probs[0]),
                "draw": float(probs[1]),
                "away": float(probs[2])
            },
            "activations": activations,
            "explanation": explanation
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sandbox prediction failed: {str(e)}")

# =====================================================================
# IMPORT STUDIO WEIGHTS ENDPOINT
# =====================================================================
from pydantic import BaseModel

class ModelImportPayload(BaseModel):
    config: dict
    scaler_mean: list[float]
    scaler_var: list[float]
    scaler_scale: list[float]
    state_dict: dict[str, list]

@app.post("/api/model/import")
def import_model_state(payload: ModelImportPayload):
    try:
        from sklearn.preprocessing import StandardScaler
        
        # Rebuild Scaler
        scaler = StandardScaler()
        scaler.mean_ = np.array(payload.scaler_mean)
        scaler.var_ = np.array(payload.scaler_var)
        scaler.scale_ = np.array(payload.scaler_scale)
        
        # Rebuild Model Architecture
        features = payload.config.get("features", [])
        hidden_layers = payload.config.get("hidden_layers", [16, 8])
        activation = payload.config.get("activation", "relu")
        
        model = DynamicFootballANN(
            len(features), hidden_layers, activation
        )
        
        # Load imported weights from state dict
        state_dict_tensors = {}
        for k, v in payload.state_dict.items():
            state_dict_tensors[k] = torch.tensor(v, dtype=torch.float32)
        model.load_state_dict(state_dict_tensors)
        
        # Cache globally
        with model_lock:
            ACTIVE_LAB_MODEL["model"] = model
            ACTIVE_LAB_MODEL["features"] = features
            ACTIVE_LAB_MODEL["scaler"] = scaler
            ACTIVE_LAB_MODEL["activation"] = activation
            ACTIVE_LAB_MODEL["hidden_layers"] = hidden_layers
        
        # Save dynamically to disk for persistence
        data_dir = os.path.join(base_dir, "ann_model", "data")
        torch.save(model.state_dict(), os.path.join(data_dir, "dynamic_model.pth"))
        joblib.dump(scaler, os.path.join(data_dir, "dynamic_scaler.joblib"))
        
        # Save model config for persistence
        with open(os.path.join(data_dir, "dynamic_config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "features": features,
                "hidden_layers": hidden_layers,
                "activation": activation
            }, f, indent=2)
            
        # Re-run predictor profiles and ratings reload
        ratings_path = os.path.join(base_dir, "prediction_engine", "data", "elo-calibrated.json")
        profiles_path = os.path.join(data_dir, "team_profiles.json")
        if os.path.exists(profiles_path):
            with open(profiles_path, "r", encoding="utf-8") as f:
                predictor.profiles = json.load(f)
        if os.path.exists(ratings_path):
            with open(ratings_path, "r", encoding="utf-8") as f:
                predictor.predictor.ratings = json.load(f).get("ratings", {})
                
        # Rebuild network state
        network_state = []
        layer_idx = 0
        for name, layer in model.net.named_children():
            if isinstance(layer, nn.Linear):
                network_state.append({
                    "layer_index": layer_idx,
                    "name": f"Layer {name}",
                    "weights": layer.weight.data.tolist(),
                    "biases": layer.bias.data.tolist() if layer.bias is not None else [],
                    "gradients": []
                })
                layer_idx += 1
                
        # Evaluate activations on reference match
        ref_features_dict = {}
        try:
            feat_vec = extract_match_features(
                "france", "spain", "france", 
                features, 
                predictor.profiles, 
                predictor.predictor.ratings
            )
            for i, f in enumerate(features):
                ref_features_dict[f] = float(feat_vec[i])
            feat_scaled = scaler.transform(feat_vec.reshape(1, -1))[0]
            ref_t = torch.FloatTensor(feat_scaled)
            ref_activations = model.get_layer_activations(ref_t)
        except Exception as e:
            print(f"Failed to calculate activations for imported model: {e}")
            ref_activations = []
            
        print("Model imported and activated successfully in design studio.")
        return {
            "status": "success", 
            "message": "Model imported successfully.",
            "network_state": network_state,
            "ref_activations": ref_activations,
            "ref_features": ref_features_dict
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")

@app.get("/api/model/active")
def get_active_model_details(
    team1: str = Query("france"),
    team2: str = Query("spain"),
    home_team: str = Query("france")
):
    with model_lock:
        has_model = ACTIVE_LAB_MODEL["model"] is not None and ACTIVE_LAB_MODEL["scaler"] is not None
        
    if not has_model:
        return {"status": "none"}
        
    with model_lock:
        model = ACTIVE_LAB_MODEL["model"]
        features = ACTIVE_LAB_MODEL["features"]
        scaler = ACTIVE_LAB_MODEL["scaler"]
        activation = ACTIVE_LAB_MODEL.get("activation", "relu")
        hidden_layers = ACTIVE_LAB_MODEL.get("hidden_layers", [16, 8])
        
        # Calculate network state
        network_state = []
        layer_idx = 0
        for name, layer in model.net.named_children():
            if isinstance(layer, nn.Linear):
                network_state.append({
                    "layer_index": layer_idx,
                    "name": f"Layer {name}",
                    "weights": layer.weight.data.tolist(),
                    "biases": layer.bias.data.tolist() if layer.bias is not None else [],
                    "gradients": []
                })
                layer_idx += 1
                
        # Calculate activations on reference match
        ref_features_dict = {}
        try:
            feat_vec = extract_match_features(
                team1, team2, home_team, 
                features, 
                predictor.profiles, 
                predictor.predictor.ratings
            )
            for i, f in enumerate(features):
                ref_features_dict[f] = float(feat_vec[i])
            feat_scaled = scaler.transform(feat_vec.reshape(1, -1))[0]
            ref_t = torch.FloatTensor(feat_scaled)
            ref_activations = model.get_layer_activations(ref_t)
        except Exception as e:
            print(f"Failed to calculate activations for active model: {e}")
            ref_activations = []
        
    return {
        "status": "active",
        "config": {
            "hidden_layers": hidden_layers,
            "activation": activation,
            "features": features
        },
        "network_state": network_state,
        "ref_activations": ref_activations,
        "ref_features": ref_features_dict,
        "scaler_mean": scaler.mean_.tolist() if hasattr(scaler, 'mean_') and scaler.mean_ is not None else [],
        "scaler_var": scaler.var_.tolist() if hasattr(scaler, 'var_') and scaler.var_ is not None else [],
        "scaler_scale": scaler.scale_.tolist() if hasattr(scaler, 'scale_') and scaler.scale_ is not None else []
    }


@app.get("/api/teams")
def get_teams():
    """Return list of all available teams"""
    try:
        teams_list = []
        for slug in predictor.profiles.keys():
            teams_list.append({
                "slug": slug,
                "name": TEAM_NAMES.get(slug, slug.title()),
                "elo": predictor.predictor.ratings.get(slug, 1500),
                "fifa_rank": 100  # placeholder, could be calculated from Elo
            })
        # Sort by name for consistent ordering
        teams_list.sort(key=lambda x: x["name"])
        return teams_list
    except Exception as e:
        print(f"Error in get_teams: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/match/analysis")
def get_match_analysis(
    team1: str = Query(..., description="First team slug"),
    team2: str = Query(..., description="Second team slug"),
    home_team: Optional[str] = Query(None, description="Home team slug")
):
    """Get match analysis including prediction and explanation"""
    try:
        # Use the same logic as /predict endpoint but return explanation too
        if not team1 or not team2:
            raise HTTPException(status_code=400, detail="Both team1 and team2 are required")
        
        # Get prediction using active laboratory model if available, otherwise fallback
        with model_lock:
            has_model = ACTIVE_LAB_MODEL["model"] is not None and ACTIVE_LAB_MODEL["scaler"] is not None
            
        if has_model:
            with model_lock:
                # Use the active lab model
                stat_pred = predictor.predictor.predict_match(team1, team2, home_team)
                
                slug1 = slugify(team1)
                slug2 = slugify(team2)
                
                # Extract features vector
                feat_vec = extract_match_features(
                    team1, team2, home_team, 
                    ACTIVE_LAB_MODEL["features"], 
                    predictor.profiles, 
                    predictor.predictor.ratings
                )
                
                feat_scaled = ACTIVE_LAB_MODEL["scaler"].transform(feat_vec.reshape(1, -1))
                feat_tensor = torch.FloatTensor(feat_scaled)
                
                # Get ANN prediction
                with torch.no_grad():
                    ann_output = ACTIVE_LAB_MODEL["model"](feat_tensor)
                    ann_probs = torch.softmax(ann_output, dim=1).cpu().numpy()[0]
                    
                # Blend predictions (statistical and ANN)
                # Using weights from config or default 0.61/0.39
                stat_weight = 0.61
                ann_weight = 0.39
                
                home_win = stat_weight * stat_pred.get('winA', 0.0) + ann_weight * ann_probs[0]
                draw = stat_weight * stat_pred.get('draw', 0.0) + ann_weight * ann_probs[1]
                away_win = stat_weight * stat_pred.get('winB', 0.0) + ann_weight * ann_probs[2]
                
                # Normalize to ensure sum = 1
                total = home_win + draw + away_win
                if total > 0:
                    home_win /= total
                    draw /= total
                    away_win /= total
                
                prediction = {
                    "home": float(home_win),
                    "draw": float(draw),
                    "away": float(away_win)
                }
                
                # Get explanation using integrated gradients
                explanation = []
                try:
                    ig_result = compute_integrated_gradients(
                        ACTIVE_LAB_MODEL["model"],
                        feat_scaled[0],
                        target_class=0,  # Explain home win
                        steps=20
                    )
                    
                    feature_names = ACTIVE_LAB_MODEL["features"]
                    attributions = ig_result
                    
                    for i, (feat, attr) in enumerate(zip(feature_names, attributions)):
                        explanation.append({
                            "feature": feat,
                            "contribution": float(attr)
                        })
                        
                    # Sort by absolute contribution
                    explanation.sort(key=lambda x: abs(x["contribution"]), reverse=True)
                except Exception as e:
                    import traceback
                    print(f"Could not generate explanation: {e}")
                    traceback.print_exc()
                    # Fallback to feature importance from model weights
                    explanation = []
                    for i, feat in enumerate(ACTIVE_LAB_MODEL["features"]):
                        explanation.append({
                            "feature": feat,
                            "contribution": 0.0  # placeholder
                        })
        else:
            # Fallback to ensemble predictor
            pred = predictor.predictor.predict_match(team1, team2, home_team)
            prediction = {
                "home": float(pred.get('winA', 0.33)),
                "draw": float(pred.get('draw', 0.33)),
                "away": float(pred.get('winB', 0.34))
            }
            
            # Simple explanation based on feature differences
            explanation = []
            feat_vec = extract_match_features(
                team1, team2, home_team,
                ['elo_home', 'elo_away', 'rank_diff', 'home_adv'],
                predictor.profiles, predictor.predictor.ratings
            )
            feature_names = ['elo_home', 'elo_away', 'rank_diff', 'home_adv']
            for feat, val in zip(feature_names, feat_vec):
                explanation.append({
                    "feature": feat,
                    "contribution": float(val * 0.1)  # simplified
                })
        
        return {
            "prediction": prediction,
            "explanation": explanation
        }
    except Exception as e:
        print(f"Error in get_match_analysis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
