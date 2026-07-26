# ann_model/walk_forward_validation.py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss

def calculate_brier_score(y_true, probs):
    """
    Computes Brier Score across multiclass probabilities.
    BS = (1/N) * sum_i sum_k (p_ik - y_ik)^2
    """
    y_onehot = np.zeros((len(y_true), 3))
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

def calculate_ece(y_true, probs, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            
    return float(ece)

class WalkForwardValidator:
    def __init__(self, dataset_path=None):
        if dataset_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_path = os.path.join(base_dir, "data", "processed_dataset.csv")
            
        self.df = pd.read_csv(dataset_path)
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df.sort_values("date").reset_index(drop=True)
        
        # Define historical temporal folds
        self.folds = [
            {"name": "2019 Fold", "train_end": "2018-12-31", "test_start": "2019-01-01", "test_end": "2019-12-31"},
            {"name": "2020-21 Fold", "train_end": "2019-12-31", "test_start": "2020-01-01", "test_end": "2021-12-31"},
            {"name": "2022 WC Fold", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
            {"name": "2023-24 Fold", "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2024-12-31"},
            {"name": "2025-26 Fold", "train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2026-12-31"}
        ]

    def evaluate_model(self, model_trainer_func, feature_cols):
        """
        Runs Walk-Forward Validation for a given model training function.
        model_trainer_func(X_train, y_train, X_test) -> predicted probabilities array (N, 3)
        """
        fold_results = []
        
        for fold in self.folds:
            train_mask = self.df["date"] <= pd.to_datetime(fold["train_end"])
            test_mask = (self.df["date"] >= pd.to_datetime(fold["test_start"])) & (self.df["date"] <= pd.to_datetime(fold["test_end"]))
            
            df_train = self.df[train_mask]
            df_test = self.df[test_mask]
            
            if len(df_test) == 0:
                continue
                
            X_train = df_train[feature_cols].values
            y_train = df_train["target"].values
            X_test = df_test[feature_cols].values
            y_test = df_test["target"].values
            
            probs = model_trainer_func(X_train, y_train, X_test)
            # Clip probabilities for log-loss stability
            probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            
            preds = np.argmax(probs, axis=1)
            
            acc = accuracy_score(y_test, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
            ll = log_loss(y_test, probs)
            bs = calculate_brier_score(y_test, probs)
            ece = calculate_ece(y_test, probs)
            
            fold_results.append({
                "fold": fold["name"],
                "n_train": len(df_train),
                "n_test": len(df_test),
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "log_loss": ll,
                "brier_score": bs,
                "ece": ece
            })
            
        # Summary across all folds
        summary = {
            "accuracy": float(np.mean([r["accuracy"] for r in fold_results])),
            "precision": float(np.mean([r["precision"] for r in fold_results])),
            "recall": float(np.mean([r["recall"] for r in fold_results])),
            "f1": float(np.mean([r["f1"] for r in fold_results])),
            "log_loss": float(np.mean([r["log_loss"] for r in fold_results])),
            "brier_score": float(np.mean([r["brier_score"] for r in fold_results])),
            "ece": float(np.mean([r["ece"] for r in fold_results])),
            "fold_details": fold_results
        }
        
        return summary

if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    print("Testing Walk-Forward Validation Pipeline...")
    validator = WalkForwardValidator()
    
    features = ["elo_home", "elo_away", "rank_diff", "form_home", "form_away", "stat_prob_home", "stat_prob_away"]
    
    def baseline_lr_trainer(X_train, y_train, X_test):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=1000).fit(X_tr, y_train)
        return clf.predict_proba(X_te)
        
    res = validator.evaluate_model(baseline_lr_trainer, features)
    print(f"Overall Walk-Forward Accuracy:  {res['accuracy']*100:.2f}%")
    print(f"Overall Walk-Forward Log-Loss:  {res['log_loss']:.4f}")
    print(f"Overall Walk-Forward Brier:     {res['brier_score']:.4f}")
    print(f"Overall Walk-Forward ECE:       {res['ece']:.4f}")
