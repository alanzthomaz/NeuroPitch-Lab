# ensemble/ensemble_predictor.py
import os
import json
import joblib
import numpy as np
import torch
from prediction_engine.predictor import MatchPredictor, slugify
from ann_model.model import FootballANN
from ann_model.train import FEATURES

class EnsemblePredictor:
    def __init__(self, data_dir=None, stat_weight=0.61, ann_weight=0.39):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.predictor = MatchPredictor(os.path.join(base_dir, "prediction_engine", "data", "elo-calibrated.json"))
        
        # Paths for ANN files
        ann_data_dir = os.path.join(base_dir, "ann_model", "data")
        model_path = os.path.join(ann_data_dir, "model.pth")
        scaler_path = os.path.join(ann_data_dir, "scaler.joblib")
        profiles_path = os.path.join(ann_data_dir, "team_profiles.json")
        
        self.stat_weight = stat_weight
        self.ann_weight = ann_weight
        
        # Load scaler
        self.scaler = None
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            
        # Load default ANN model
        self.ann_model = None
        if os.path.exists(model_path) and self.scaler is not None:
            try:
                self.ann_model = FootballANN(len(FEATURES))
                self.ann_model.load_state_dict(torch.load(model_path, map_location="cpu"))
                self.ann_model.eval()
            except Exception as e:
                print(f"Notice: PyTorch ANN model load fallback: {e}")
                self.ann_model = None
            
        # Load profiles
        self.profiles = {}
        if os.path.exists(profiles_path):
            with open(profiles_path, "r", encoding="utf-8") as f:
                self.profiles = json.load(f)
                
        # Load tree model and calibrator if available
        self.tree_model = None
        tree_path = os.path.join(ann_data_dir, "tree_model.joblib")
        if os.path.exists(tree_path):
            try:
                from ann_model.tree_models import TreeModelSuite
                self.tree_model = TreeModelSuite.load(tree_path)
            except Exception as e:
                print(f"Notice: Tree model load fallback: {e}")

        self.calibrator = None
        calib_path = os.path.join(ann_data_dir, "calibrator.joblib")
        if os.path.exists(calib_path):
            try:
                from ann_model.calibration import ProbabilityCalibrator
                self.calibrator = ProbabilityCalibrator.load(calib_path)
            except Exception as e:
                print(f"Notice: Calibrator load fallback: {e}")

    def predict_match(self, team1, team2, home_team=None, weights=None):
        """
        Combine statistical engine, PyTorch ANN, and Gradient Boosting Tree models.
        """
        if weights is None:
            w_stat, w_ann = self.stat_weight, self.ann_weight
        else:
            w_stat = weights.get("stat", self.stat_weight)
            w_ann = weights.get("ann", self.ann_weight)
            
        # 1. Run Statistical Prediction
        stat_pred = self.predictor.predict_match(team1, team2, home_team)
        
        # If ANN is not trained, fallback to Statistical only
        if self.ann_model is None or self.scaler is None:
            return {
                **stat_pred,
                "ann_prob_home": stat_pred["winA"],
                "ann_prob_draw": stat_pred["draw"],
                "ann_prob_away": stat_pred["winB"],
                "ensemble_home": stat_pred["winA"],
                "ensemble_draw": stat_pred["draw"],
                "ensemble_away": stat_pred["winB"],
                "weights": {"stat": 1.0, "ann": 0.0}
            }
            
        # 2. Extract Features for ANN & Tree Models
        from ann_model.dynamic_trainer import extract_match_features
        feat_vector = extract_match_features(team1, team2, home_team, FEATURES, self.profiles, self.predictor.ratings)
        feat_scaled = self.scaler.transform(feat_vector.reshape(1, -1))
        
        # ANN Probabilities
        feat_tensor = torch.FloatTensor(feat_scaled)
        with torch.no_grad():
            outputs = self.ann_model(feat_tensor)
            ann_probs = torch.softmax(outputs, dim=1).numpy()[0]

        if self.calibrator:
            ann_probs = self.calibrator.calibrate(ann_probs.reshape(1, -1))[0]
            
        ann_h, ann_d, ann_a = float(ann_probs[0]), float(ann_probs[1]), float(ann_probs[2])

        # Tree Model Probabilities (if fitted)
        tree_probs = [ann_h, ann_d, ann_a]
        if self.tree_model:
            try:
                tree_probs = self.tree_model.predict_proba(feat_scaled)[0]
                if self.calibrator:
                    tree_probs = self.calibrator.calibrate(tree_probs.reshape(1, -1))[0]
            except Exception as e:
                pass
                
        tree_h, tree_d, tree_a = float(tree_probs[0]), float(tree_probs[1]), float(tree_probs[2])
        
        # 3. Hybrid Blended Ensemble Predictions
        # Blend: Stat + Calibrated ANN + Calibrated Tree Model
        w_tree = 0.2 if self.tree_model else 0.0
        eff_stat = w_stat * (1.0 - w_tree)
        eff_ann = w_ann * (1.0 - w_tree)
        
        ens_h = eff_stat * stat_pred["winA"] + eff_ann * ann_h + w_tree * tree_h
        ens_d = eff_stat * stat_pred["draw"] + eff_ann * ann_d + w_tree * tree_d
        ens_a = eff_stat * stat_pred["winB"] + eff_ann * ann_a + w_tree * tree_a
        
        total = ens_h + ens_d + ens_a
        ens_h /= total
        ens_d /= total
        ens_a /= total
        
        return {
            **stat_pred,
            "ann_prob_home": float(ann_h),
            "ann_prob_draw": float(ann_d),
            "ann_prob_away": float(ann_a),
            "tree_prob_home": float(tree_h),
            "tree_prob_draw": float(tree_d),
            "tree_prob_away": float(tree_a),
            "ensemble_home": float(ens_h),
            "ensemble_draw": float(ens_d),
            "ensemble_away": float(ens_a),
            "weights": {"stat": round(eff_stat, 2), "ann": round(eff_ann, 2), "tree": round(w_tree, 2)}
        }
