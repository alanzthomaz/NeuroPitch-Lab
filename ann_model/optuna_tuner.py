# ann_model/optuna_tuner.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ann_model.walk_forward_validation import WalkForwardValidator, calculate_brier_score, calculate_ece
from ann_model.dynamic_model import DynamicFootballANN
from ann_model.tree_models import TreeModelSuite

class OptunaHyperparameterTuner:
    def __init__(self, dataset_path=None):
        self.validator = WalkForwardValidator(dataset_path=dataset_path)

    def tune_ann(self, feature_cols, n_trials=15):
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                n_layers = trial.suggest_int("n_layers", 1, 3)
                hidden_layers = [trial.suggest_int(f"units_l{i}", 16, 128, step=16) for i in range(n_layers)]
                activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu", "selu"])
                dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
                lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
                batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
                epochs = 30

                def model_trainer(X_train, y_train, X_test):
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_train)
                    X_te = scaler.transform(X_test)

                    X_tr_t = torch.FloatTensor(X_tr)
                    y_tr_t = torch.LongTensor(y_train)
                    X_te_t = torch.FloatTensor(X_te)

                    model = DynamicFootballANN(len(feature_cols), hidden_layers, activation, dropout_rate=dropout)
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                    criterion = nn.CrossEntropyLoss()

                    model.train()
                    n_samples = X_tr_t.size(0)
                    for epoch in range(epochs):
                        perm = torch.randperm(n_samples)
                        for i in range(0, n_samples, batch_size):
                            idx = perm[i:i+batch_size]
                            optimizer.zero_grad()
                            loss = criterion(model(X_tr_t[idx]), y_tr_t[idx])
                            loss.backward()
                            optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        logits = model(X_te_t)
                        probs = torch.softmax(logits, dim=1).numpy()
                    return probs

                res = self.validator.evaluate_model(model_trainer, feature_cols)
                return res["log_loss"]

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            print(f"Optuna ANN Best Log-Loss: {study.best_value:.4f}")
            return best_params
        except ImportError:
            print("Optuna not found, returning default ANN parameters.")
            return {"n_layers": 2, "units_l0": 64, "units_l1": 32, "activation": "relu", "dropout": 0.2, "lr": 0.005, "weight_decay": 1e-4, "batch_size": 64}

    def tune_tree_model(self, model_type="xgboost", feature_cols=None, n_trials=15):
        if feature_cols is None:
            feature_cols = ["elo_home", "elo_away", "rank_diff", "form_home", "form_away", "stat_prob_home", "stat_prob_away"]
            
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
                    "max_depth": trial.suggest_int("max_depth", 3, 7),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1)
                }

                def model_trainer(X_train, y_train, X_test):
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_train)
                    X_te = scaler.transform(X_test)

                    suite = TreeModelSuite(model_type=model_type, params=params)
                    suite.fit(X_tr, y_train)
                    return suite.predict_proba(X_te)

                res = self.validator.evaluate_model(model_trainer, feature_cols)
                return res["log_loss"]

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials)
            print(f"Optuna {model_type.upper()} Best Log-Loss: {study.best_value:.4f}")
            return study.best_params
        except ImportError:
            print(f"Optuna not found, returning default {model_type} parameters.")
            return {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}

if __name__ == "__main__":
    tuner = OptunaHyperparameterTuner()
    features = ["elo_home", "elo_away", "rank_diff", "form_home", "form_away", "stat_prob_home", "stat_prob_away"]
    best_ann = tuner.tune_ann(features, n_trials=3)
    print("Best ANN Params:", best_ann)
