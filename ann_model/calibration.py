# ann_model/calibration.py
import os
import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from ann_model.walk_forward_validation import calculate_brier_score, calculate_ece

class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, y_true):
        # Optimize temperature T > 0 to minimize log loss on validation logits
        def loss_func(t):
            T = max(1e-3, t[0])
            scaled_logits = logits / T
            # Softmax
            exp_l = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            return log_loss(y_true, probs)
            
        res = minimize(loss_func, [1.0], method='Nelder-Mead')
        self.temperature = max(1e-3, float(res.x[0]))
        return self

    def calibrate(self, logits):
        scaled = logits / self.temperature
        exp_l = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        return probs

class PlattScaler:
    def __init__(self):
        self.calibrators = []

    def fit(self, probs, y_true):
        self.calibrators = []
        for k in range(3):
            y_k = (y_true == k).astype(int)
            p_k = probs[:, k].reshape(-1, 1)
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(p_k, y_k)
            self.calibrators.append(clf)
        return self

    def calibrate(self, probs):
        calibrated_probs = np.zeros_like(probs)
        for k in range(3):
            p_k = probs[:, k].reshape(-1, 1)
            calibrated_probs[:, k] = self.calibrators[k].predict_proba(p_k)[:, 1]
        # Normalize rows to sum to 1
        s = np.sum(calibrated_probs, axis=1, keepdims=True)
        s[s == 0] = 1.0
        return calibrated_probs / s

class IsotonicCalibrator:
    def __init__(self):
        self.calibrators = []

    def fit(self, probs, y_true):
        self.calibrators = []
        for k in range(3):
            y_k = (y_true == k).astype(int)
            p_k = probs[:, k]
            iso = IsotonicRegression(out_of_bounds='clip').fit(p_k, y_k)
            self.calibrators.append(iso)
        return self

    def calibrate(self, probs):
        calibrated_probs = np.zeros_like(probs)
        for k in range(3):
            p_k = probs[:, k]
            calibrated_probs[:, k] = self.calibrators[k].predict(p_k)
        s = np.sum(calibrated_probs, axis=1, keepdims=True)
        s[s == 0] = 1.0
        return calibrated_probs / s

class ProbabilityCalibrator:
    def __init__(self, method="auto"):
        self.method = method
        self.best_method_name = "none"
        self.active_calibrator = None

    def fit(self, val_probs_or_logits, y_val, is_logits=False):
        if is_logits:
            logits = val_probs_or_logits
            exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        else:
            probs = val_probs_or_logits
            logits = np.log(np.clip(probs, 1e-15, 1.0 - 1e-15))

        candidates = {}
        
        # 1. Temperature Scaling
        ts = TemperatureScaler().fit(logits, y_val)
        ts_probs = ts.calibrate(logits)
        candidates["temperature_scaling"] = (ts, calculate_ece(y_val, ts_probs), ts_probs)
        
        # 2. Platt Scaling
        platt = PlattScaler().fit(probs, y_val)
        platt_probs = platt.calibrate(probs)
        candidates["platt_scaling"] = (platt, calculate_ece(y_val, platt_probs), platt_probs)
        
        # 3. Isotonic Regression
        iso = IsotonicCalibrator().fit(probs, y_val)
        iso_probs = iso.calibrate(probs)
        candidates["isotonic_regression"] = (iso, calculate_ece(y_val, iso_probs), iso_probs)

        if self.method in candidates:
            self.best_method_name = self.method
            self.active_calibrator = candidates[self.method][0]
        else:
            # Auto-select calibrator with lowest ECE
            best_name = min(candidates.keys(), key=lambda k: candidates[k][1])
            self.best_method_name = best_name
            self.active_calibrator = candidates[best_name][0]

        return self

    def calibrate(self, probs_or_logits, is_logits=False):
        if self.active_calibrator is None:
            if is_logits:
                exp_l = np.exp(probs_or_logits - np.max(probs_or_logits, axis=1, keepdims=True))
                return exp_l / np.sum(exp_l, axis=1, keepdims=True)
            return probs_or_logits

        if isinstance(self.active_calibrator, TemperatureScaler):
            if not is_logits:
                probs_or_logits = np.log(np.clip(probs_or_logits, 1e-15, 1.0 - 1e-15))
            return self.active_calibrator.calibrate(probs_or_logits)
        else:
            if is_logits:
                exp_l = np.exp(probs_or_logits - np.max(probs_or_logits, axis=1, keepdims=True))
                probs_or_logits = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            return self.active_calibrator.calibrate(probs_or_logits)

    def save(self, filepath):
        joblib.dump({
            "best_method_name": self.best_method_name,
            "active_calibrator": self.active_calibrator
        }, filepath)

    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        cal = cls(method=data["best_method_name"])
        cal.best_method_name = data["best_method_name"]
        cal.active_calibrator = data["active_calibrator"]
        return cal

if __name__ == "__main__":
    np.random.seed(42)
    y_dummy = np.random.choice([0, 1, 2], size=200)
    raw_probs = np.random.dirichlet((1, 1, 1), size=200)
    
    calibrator = ProbabilityCalibrator(method="auto").fit(raw_probs, y_dummy)
    cal_probs = calibrator.calibrate(raw_probs)
    
    print(f"Original ECE: {calculate_ece(y_dummy, raw_probs):.4f}")
    print(f"Selected Calibrator: {calibrator.best_method_name}")
    print(f"Calibrated ECE: {calculate_ece(y_dummy, cal_probs):.4f}")
