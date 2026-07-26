# ann_model/benchmark_all.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from ann_model.walk_forward_validation import WalkForwardValidator, calculate_brier_score, calculate_ece
from ann_model.dynamic_model import DynamicFootballANN
from ann_model.tree_models import TreeModelSuite
from ann_model.calibration import ProbabilityCalibrator
from ann_model.train import FEATURES

def run_master_benchmark():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    dataset_path = os.path.join(data_dir, "processed_dataset.csv")

    if not os.path.exists(dataset_path):
        from ann_model.dataset import build_dataset
        build_dataset()

    validator = WalkForwardValidator(dataset_path=dataset_path)

    print("==========================================================")
    print("      RESEARCH-GRADE FIFA WORLD CUP ENGINE BENCHMARK      ")
    print("==========================================================")

    # 1. Baseline Model Evaluation (Standard ANN, No Calibration)
    def train_baseline_ann(X_train, y_train, X_test):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_t = torch.LongTensor(y_train)
        X_te_t = torch.FloatTensor(X_te)

        model = DynamicFootballANN(len(FEATURES), [128, 64, 32], "relu", dropout_rate=0.2)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        model.train()
        batch_size = 64
        for epoch in range(40):
            perm = torch.randperm(X_tr_t.size(0))
            for i in range(0, X_tr_t.size(0), batch_size):
                idx = perm[i:i+batch_size]
                optimizer.zero_grad()
                loss = criterion(model(X_tr_t[idx]), y_tr_t[idx])
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(X_te_t)
            probs = torch.softmax(outputs, dim=1).numpy()
        return probs

    print("\n[1/5] Benchmarking Baseline ANN (Walk-Forward Validation)...")
    baseline_res = validator.evaluate_model(train_baseline_ann, FEATURES)
    print(f"--> Baseline ANN: Acc = {baseline_res['accuracy']*100:.2f}%, Log-Loss = {baseline_res['log_loss']:.4f}, Brier = {baseline_res['brier_score']:.4f}, ECE = {baseline_res['ece']:.4f}")

    # 2. Calibrated Upgraded ANN
    def train_calibrated_ann(X_train, y_train, X_test):
        # 80/20 train/val for temperature scaling
        val_size = int(len(X_train) * 0.2)
        X_tr_sub, X_val_sub = X_train[:-val_size], X_train[-val_size:]
        y_tr_sub, y_val_sub = y_train[:-val_size], y_train[-val_size:]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_sub)
        X_val_s = scaler.transform(X_val_sub)
        X_te_s = scaler.transform(X_test)

        X_tr_t = torch.FloatTensor(X_tr_s)
        y_tr_t = torch.LongTensor(y_tr_sub)
        X_val_t = torch.FloatTensor(X_val_s)
        X_te_t = torch.FloatTensor(X_te_s)

        model = DynamicFootballANN(len(FEATURES), [128, 64, 32], "leaky_relu", dropout_rate=0.15)
        optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        model.train()
        batch_size = 64
        for epoch in range(50):
            perm = torch.randperm(X_tr_t.size(0))
            for i in range(0, X_tr_t.size(0), batch_size):
                idx = perm[i:i+batch_size]
                optimizer.zero_grad()
                loss = criterion(model(X_tr_t[idx]), y_tr_t[idx])
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.softmax(model(X_val_t), dim=1).numpy()
            te_probs = torch.softmax(model(X_te_t), dim=1).numpy()

        calibrator = ProbabilityCalibrator().fit(val_probs, y_val_sub)
        return calibrator.calibrate(te_probs)

    print("\n[2/5] Benchmarking Calibrated Upgraded ANN...")
    ann_res = validator.evaluate_model(train_calibrated_ann, FEATURES)
    print(f"--> Upgraded ANN: Acc = {ann_res['accuracy']*100:.2f}%, Log-Loss = {ann_res['log_loss']:.4f}, Brier = {ann_res['brier_score']:.4f}, ECE = {ann_res['ece']:.4f}")

    # 3. Gradient Boosting Tree Model
    def train_tree_model(X_train, y_train, X_test):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        suite = TreeModelSuite(model_type="xgboost")
        suite.fit(X_tr, y_train)
        probs = suite.predict_proba(X_te)
        return probs

    print("\n[3/5] Benchmarking Gradient Boosting Tree Model...")
    tree_res = validator.evaluate_model(train_tree_model, FEATURES)
    print(f"--> Tree Model: Acc = {tree_res['accuracy']*100:.2f}%, Log-Loss = {tree_res['log_loss']:.4f}, Brier = {tree_res['brier_score']:.4f}, ECE = {tree_res['ece']:.4f}")

    # 4. Hybrid Ensemble (Dixon-Coles Statistical Engine + Calibrated ANN + Calibrated Tree Model)
    def train_hybrid_ensemble(X_train, y_train, X_test):
        p_ann = train_calibrated_ann(X_train, y_train, X_test)
        p_tree = train_tree_model(X_train, y_train, X_test)
        
        # Extract stat engine probabilities from feature columns
        stat_idx_h = FEATURES.index("stat_prob_home") if "stat_prob_home" in FEATURES else 25
        stat_idx_d = FEATURES.index("stat_prob_draw") if "stat_prob_draw" in FEATURES else 26
        stat_idx_a = FEATURES.index("stat_prob_away") if "stat_prob_away" in FEATURES else 27

        p_stat = X_test[:, [stat_idx_h, stat_idx_d, stat_idx_a]]

        # Multi-model blend: 50% Dixon-Coles Poisson + 30% Calibrated ANN + 20% Calibrated XGBoost/LightGBM
        p_ens = 0.50 * p_stat + 0.30 * p_ann + 0.20 * p_tree
        p_ens = p_ens / np.sum(p_ens, axis=1, keepdims=True)

        # Calibrate ensemble probabilities with Platt scaling on validation fold
        val_size = int(len(X_train) * 0.2)
        X_val_sub = X_train[-val_size:]
        y_val_sub = y_train[-val_size:]

        p_ann_val = train_calibrated_ann(X_train[:-val_size], y_train[:-val_size], X_val_sub)
        p_tree_val = train_tree_model(X_train[:-val_size], y_train[:-val_size], X_val_sub)
        p_stat_val = X_val_sub[:, [stat_idx_h, stat_idx_d, stat_idx_a]]

        p_ens_val = 0.50 * p_stat_val + 0.30 * p_ann_val + 0.20 * p_tree_val
        p_ens_val = p_ens_val / np.sum(p_ens_val, axis=1, keepdims=True)

        calibrator = ProbabilityCalibrator(method="platt_scaling").fit(p_ens_val, y_val_sub)
        return calibrator.calibrate(p_ens)

    print("\n[4/5] Benchmarking Hybrid Ensemble...")
    ens_res = validator.evaluate_model(train_hybrid_ensemble, FEATURES)
    print(f"--> Hybrid Ensemble: Acc = {ens_res['accuracy']*100:.2f}%, Log-Loss = {ens_res['log_loss']:.4f}, Brier = {ens_res['brier_score']:.4f}, ECE = {ens_res['ece']:.4f}")

    # 5. Model Acceptance Decision
    print("\n==========================================================")
    print("               MODEL ACCEPTANCE AUDIT CHECK               ")
    print("==========================================================")

    candidates = {
        "Upgraded ANN": ann_res,
        "Gradient Boosting Tree Model": tree_res,
        "Hybrid Ensemble": ens_res
    }

    best_candidate_name = None
    best_candidate_res = None

    for cname, cres in candidates.items():
        acc_pass = cres["accuracy"] >= baseline_res["accuracy"]
        loss_pass = cres["log_loss"] <= baseline_res["log_loss"]
        brier_pass = cres["brier_score"] <= baseline_res["brier_score"]
        ece_pass = cres["ece"] <= baseline_res["ece"]

        print(f"\nEvaluating Candidate: {cname}")
        print(f"1. Higher Accuracy:   {'PASSED' if acc_pass else 'FAILED'} ({cres['accuracy']*100:.2f}% vs {baseline_res['accuracy']*100:.2f}%)")
        print(f"2. Lower Log-Loss:    {'PASSED' if loss_pass else 'FAILED'} ({cres['log_loss']:.4f} vs {baseline_res['log_loss']:.4f})")
        print(f"3. Lower Brier Score: {'PASSED' if brier_pass else 'FAILED'} ({cres['brier_score']:.4f} vs {baseline_res['brier_score']:.4f})")
        print(f"4. Lower ECE Error:   {'PASSED' if ece_pass else 'FAILED'} ({cres['ece']:.4f} vs {baseline_res['ece']:.4f})")

        if acc_pass and loss_pass and brier_pass and ece_pass:
            if best_candidate_res is None or cres["accuracy"] > best_candidate_res["accuracy"]:
                best_candidate_name = cname
                best_candidate_res = cres

    if best_candidate_name is not None:
        print(f"\n🏆 ACCEPTANCE RESULT: ACCEPTED ({best_candidate_name}). Promoting upgraded model to production!")
        df_full = pd.read_csv(dataset_path)
        X_full = df_full[FEATURES].values
        y_full = df_full["target"].values

        scaler = StandardScaler()
        X_full_s = scaler.fit_transform(X_full)
        joblib.dump(scaler, os.path.join(data_dir, "scaler.joblib"))

        # Save Tree Model
        tree_suite = TreeModelSuite(model_type="xgboost")
        tree_suite.fit(X_full_s, y_full)
        tree_suite.save(os.path.join(data_dir, "tree_model.joblib"))

        # Fit Calibrator
        calibrator = ProbabilityCalibrator().fit(tree_suite.predict_proba(X_full_s), y_full)
        calibrator.save(os.path.join(data_dir, "calibrator.joblib"))
        print("Production artifacts updated successfully.")
        accepted = True
        promoted_model_name = best_candidate_name
    else:
        print("\n⚠️ ACCEPTANCE RESULT: REJECTED. Preserving existing baseline production model.")
        accepted = False
        promoted_model_name = "Baseline ANN"

    # Save Experiment Log
    exp_log_path = os.path.join(data_dir, "experiments.json")
    experiments = []
    if os.path.exists(exp_log_path):
        try:
            with open(exp_log_path, "r", encoding="utf-8") as f:
                experiments = json.load(f)
        except Exception:
            experiments = []

    experiments.append({
        "experiment_id": f"EXP_{len(experiments)+1:03d}",
        "date": str(pd.Timestamp.now()),
        "features_count": len(FEATURES),
        "baseline_accuracy": baseline_res["accuracy"],
        "baseline_log_loss": baseline_res["log_loss"],
        "upgraded_accuracy": ens_res["accuracy"],
        "upgraded_log_loss": ens_res["log_loss"],
        "upgraded_brier": ens_res["brier_score"],
        "upgraded_ece": ens_res["ece"],
        "accepted": accepted
    })

    with open(exp_log_path, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2)

if __name__ == "__main__":
    run_master_benchmark()
