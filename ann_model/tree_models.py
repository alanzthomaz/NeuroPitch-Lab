# ann_model/tree_models.py
import os
import joblib
import numpy as np

class TreeModelSuite:
    def __init__(self, model_type="xgboost", params=None):
        self.model_type = model_type.lower()
        self.params = params if params is not None else {}
        self.model = None
        self.is_fitted = False
        self._initialize_model()

    def _initialize_model(self):
        if self.model_type == "xgboost":
            try:
                import xgboost as xgb
                default_params = {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "objective": "multi:softprob",
                    "num_class": 3,
                    "random_state": 42
                }
                default_params.update(self.params)
                self.model = xgb.XGBClassifier(**default_params)
            except ImportError:
                from sklearn.ensemble import HistGradientBoostingClassifier
                self.model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
                
        elif self.model_type == "lightgbm":
            try:
                import lightgbm as lgb
                default_params = {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "objective": "multiclass",
                    "num_class": 3,
                    "random_state": 42,
                    "verbose": -1
                }
                default_params.update(self.params)
                self.model = lgb.LGBMClassifier(**default_params)
            except ImportError:
                from sklearn.ensemble import HistGradientBoostingClassifier
                self.model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
                
        elif self.model_type == "catboost":
            try:
                import catboost as cb
                default_params = {
                    "iterations": 150,
                    "depth": 5,
                    "learning_rate": 0.05,
                    "loss_function": "MultiClass",
                    "random_seed": 42,
                    "verbose": False
                }
                default_params.update(self.params)
                self.model = cb.CatBoostClassifier(**default_params)
            except ImportError:
                from sklearn.ensemble import HistGradientBoostingClassifier
                self.model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(max_iter=100, random_state=42)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        probs = self.model.predict_proba(X)
        probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
        return probs / np.sum(probs, axis=1, keepdims=True)

    def save(self, filepath):
        joblib.dump({"model_type": self.model_type, "model": self.model}, filepath)

    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        suite = cls(model_type=data["model_type"])
        suite.model = data["model"]
        suite.is_fitted = True
        return suite

if __name__ == "__main__":
    X_dummy = np.random.randn(100, 10)
    y_dummy = np.random.choice([0, 1, 2], size=100)
    
    for m in ["xgboost", "lightgbm", "catboost"]:
        suite = TreeModelSuite(m)
        suite.fit(X_dummy, y_dummy)
        probs = suite.predict_proba(X_dummy[:5])
        print(f"{m} probabilities shape: {probs.shape}, row sum: {np.sum(probs[0]):.4f}")
