# ann_model/dynamic_trainer.py
import os
import json
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from ann_model.dynamic_model import DynamicFootballANN
from prediction_engine.predictor import slugify
from prediction_engine.poisson import match_prob

def extract_match_features(team1, team2, home_team, features_list, profiles, ratings):
    """
    Utility to build a single features vector for live forward propagation.
    Ensures synthetic features are calculated deterministically if checked.
    """
    slug1 = slugify(team1)
    slug2 = slugify(team2)
    
    r1 = ratings.get(slug1, 1500.0)
    r2 = ratings.get(slug2, 1500.0)
    
    # Profiles stats
    def get_profile(slug):
        if slug in profiles:
            p = profiles[slug]
            return p.get("rank", 150.0), p.get("form", 1.0), p.get("gs", 1.0), p.get("gc", 1.0), p.get("win_rate", 0.33)
        return 150.0, 1.0, 1.0, 1.0, 0.33
        
    rank1, form1, gs1, gc1, wr1 = get_profile(slug1)
    rank2, form2, gs2, gc2, wr2 = get_profile(slug2)
    
    is_neutral = (home_team is None or (home_team != team1 and home_team != team2))
    home_adv = 0.0 if is_neutral else 1.0
    home_bonus = 75.0 if not is_neutral else 0.0
    
    # Run stats match probability
    prob = match_prob(r1, r2, home_bonus)
    
    poss_h = 50.0 + (r1 - r2) / 10.0
    poss_h = max(30.0, min(70.0, poss_h))
    poss_a = 100.0 - poss_h
    
    shots_h = max(3.0, prob["expectedGoalsA"] * 6.5)
    shots_a = max(3.0, prob["expectedGoalsB"] * 6.5)
    
    feature_dict = {
        "elo_home": r1, "elo_away": r2,
        "rank_home": rank1, "rank_away": rank2, "rank_diff": rank1 - rank2,
        "form_home": form1, "form_away": form2,
        "gs_home": gs1, "gs_away": gs2,
        "gc_home": gc1, "gc_away": gc2,
        "win_rate_home": wr1, "win_rate_away": wr2,
        "h2h_win_rate_home": 0.33, "h2h_win_rate_away": 0.33,
        "home_adv": home_adv,
        "rest_home": 4.0, "rest_away": 4.0,
        "poss_home": poss_h / 100.0, "poss_away": poss_a / 100.0,
        "shots_home": shots_h, "shots_away": shots_a,
        "player_avail_home": 1.0, "player_avail_away": 1.0,
        "stat_prob_home": prob["winA"],
        "stat_prob_draw": prob["draw"],
        "stat_prob_away": prob["winB"],
        "stat_xg_home": prob["expectedGoalsA"],
        "stat_xg_away": prob["expectedGoalsB"],
        
        # Synthetic design-studio features (compiled deterministically)
        "age_home": float(24.0 + ((r1 + 7) % 31) / 5.0),
        "age_away": float(24.0 + ((r2 + 7) % 31) / 5.0),
        "market_value_home": float(max(10.0, (r1 - 1000) * 0.4)),
        "market_value_away": float(max(10.0, (r2 - 1000) * 0.4)),
        "def_rating_home": float(100.0 - gc1 * 15.0),
        "def_rating_away": float(100.0 - gc2 * 15.0),
        "off_rating_home": float(gs1 * 30.0),
        "off_rating_away": float(gs2 * 30.0),
        "exp_home": float(wr1 * 12.0),
        "exp_away": float(wr2 * 12.0)
    }

    # Add multi-window rolling form features
    for w in [5, 10, 20]:
        feature_dict[f"form_{w}_home"] = form1
        feature_dict[f"form_{w}_away"] = form2
        feature_dict[f"gs_{w}_home"] = gs1
        feature_dict[f"gs_{w}_away"] = gs2
        feature_dict[f"gc_{w}_home"] = gc1
        feature_dict[f"gc_{w}_away"] = gc2
        feature_dict[f"gd_{w}_home"] = gs1 - gc1
        feature_dict[f"gd_{w}_away"] = gs2 - gc2
        feature_dict[f"win_rate_{w}_home"] = wr1
        feature_dict[f"win_rate_{w}_away"] = wr2
        feature_dict[f"draw_rate_{w}_home"] = 0.33
        feature_dict[f"draw_rate_{w}_away"] = 0.33
        feature_dict[f"loss_rate_{w}_home"] = 1.0 - wr1 - 0.33
        feature_dict[f"loss_rate_{w}_away"] = 1.0 - wr2 - 0.33
        feature_dict[f"cs_{w}_home"] = 0.2
        feature_dict[f"cs_{w}_away"] = 0.2
        feature_dict[f"btts_{w}_home"] = 0.5
        feature_dict[f"btts_{w}_away"] = 0.5

    vector = [feature_dict.get(f, 0.0) for f in features_list]
    return np.array(vector, dtype=float)

def train_and_stream(config):
    """
    Generator function that trains a custom network configuration and yields SSE logs.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "ann_model", "data")
    dataset_path = os.path.join(data_dir, "processed_dataset.csv")
    profiles_path = os.path.join(data_dir, "team_profiles.json")
    ratings_path = os.path.join(base_dir, "prediction_engine", "data", "elo-calibrated.json")
    
    # 1. Random Seed Reproducibility
    seed = config.get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Rebuild dataset to incorporate recent World Cup results (if changed or missing)
    yield "data: " + json.dumps({"status": "compiling", "message": "Synchronizing recent World Cup matches..."}) + "\n\n"
    try:
        wc_results_path_check = os.path.join(base_dir, "prediction_engine", "data", "wc2026-results.json")
        rebuild_needed = not os.path.exists(dataset_path)
        if not rebuild_needed and os.path.exists(wc_results_path_check):
            if os.path.getmtime(wc_results_path_check) > os.path.getmtime(dataset_path):
                rebuild_needed = True
        
        if rebuild_needed:
            from ann_model.dataset import build_dataset
            build_dataset()
        else:
            print("Dataset is already up to date. Skipping rebuild.")
    except Exception as e:
        print(f"Failed to check/rebuild dataset: {e}")

    # Load dataset
    if not os.path.exists(dataset_path):
        yield "data: " + json.dumps({"error": "Dataset missing. Please compile it first."}) + "\n\n"
        return
        
    df = pd.read_csv(dataset_path)
    
    # Compile synthetic features if missing in CSV
    for suffix in ["home", "away"]:
        if f"age_{suffix}" not in df.columns:
            df[f"age_{suffix}"] = 24.0 + ((df[f"elo_{suffix}"] + 7) % 31) / 5.0
        if f"market_value_{suffix}" not in df.columns:
            df[f"market_value_{suffix}"] = (df[f"elo_{suffix}"] - 1000).clip(lower=100) * 0.4
        if f"def_rating_{suffix}" not in df.columns:
            df[f"def_rating_{suffix}"] = 100.0 - df[f"gc_{suffix}"] * 15.0
        if f"off_rating_{suffix}" not in df.columns:
            df[f"off_rating_{suffix}"] = df[f"gs_{suffix}"] * 30.0
        if f"exp_{suffix}" not in df.columns:
            df[f"exp_{suffix}"] = df[f"win_rate_{suffix}"] * 12.0

    features_list = config.get("features", [])
    if not features_list:
        yield "data: " + json.dumps({"error": "No input features selected."}) + "\n\n"
        return
        
    X = df[features_list].values
    y = df["target"].values
    
    # Train test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Load profiles & ratings for the reference match activations
    profiles = {}
    if os.path.exists(profiles_path):
        with open(profiles_path, "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
    ratings = {}
    if os.path.exists(ratings_path):
        with open(ratings_path, "r", encoding="utf-8") as f:
            ratings = json.load(f).get("ratings", {})

    # Reference match features vector for real-time activation updates
    ref_match = config.get("reference_match", {"team1": "france", "team2": "spain", "home_team": "france"})
    ref_vector = extract_match_features(
        ref_match["team1"], ref_match["team2"], ref_match["home_team"],
        features_list, profiles, ratings
    )
    ref_scaled = scaler.transform(ref_vector.reshape(1, -1))[0]
    
    # Tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.LongTensor(y_val)
    ref_t = torch.FloatTensor(ref_scaled)

    # Initialize dynamic model
    input_dim = len(features_list)
    hidden_layers = config.get("hidden_layers", [16, 8])
    activation_name = config.get("activation", "relu")
    dropout_rate = config.get("dropout", 0.0)
    use_batch_norm = config.get("batch_norm", False)
    init_name = config.get("weight_init", "xavier_uniform")
    
    model = DynamicFootballANN(
        input_dim, hidden_layers, activation_name, 
        dropout_rate, use_batch_norm, init_name
    )
    
    # 2. Loss Functions Selector
    loss_name = config.get("loss_func", "crossentropy").lower()
    if loss_name == "mse":
        criterion = nn.MSELoss()
    elif loss_name == "l1":
        criterion = nn.L1Loss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 3. Optimizer & Weight Decay Selector
    opt_name = config.get("optimizer", "adam").lower()
    lr = config.get("learning_rate", 0.005)
    l2_reg = config.get("weight_decay", 1e-4)
    momentum = config.get("momentum", 0.9)
    
    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    elif opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=l2_reg, momentum=momentum)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_reg)
    elif opt_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=l2_reg, momentum=momentum if momentum > 0 else 0)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
        
    epochs = config.get("epochs", 30)
    batch_size = config.get("batch_size", 64)
    
    # 4. LR Scheduler Selector
    sched_name = config.get("scheduler", "none").lower()
    scheduler = None
    if sched_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif sched_name == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif sched_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Early Stopping patience
    early_stopping_patience = config.get("early_stopping", 0)
    best_val_loss = float('inf')
    patience_counter = 0
    last_train_acc = 0.0

    # Send configuration to client
    yield "data: " + json.dumps({"status": "starting", "input_dim": input_dim, "epochs": epochs}) + "\n\n"
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_t.size()[0])
        epoch_loss = 0.0
        
        for i in range(0, X_train_t.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_t[indices], y_train_t[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            if loss_name in ["mse", "l1"]:
                batch_y_target = nn.functional.one_hot(batch_y, num_classes=3).float()
                loss = criterion(outputs, batch_y_target)
            else:
                loss = criterion(outputs, batch_y)
                
            loss.backward()
            
            # Autograd gradient clipping if enabled
            if config.get("grad_clipping", False):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= X_train_t.size(0)
        
        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            
            if loss_name in ["mse", "l1"]:
                val_y_target = nn.functional.one_hot(y_val_t, num_classes=3).float()
                val_loss = criterion(val_outputs, val_y_target).item()
            else:
                val_loss = criterion(val_outputs, y_val_t).item()
            
            # Metrics
            val_probs = torch.softmax(val_outputs, dim=1).numpy()
            val_preds = np.argmax(val_probs, axis=1)
            val_acc = accuracy_score(y_val, val_preds)
            
            is_heavy = (epoch == 0) or (epoch == epochs - 1) or ((epoch + 1) % 5 == 0)
            
            if is_heavy:
                train_outputs = model(X_train_t)
                train_probs = torch.softmax(train_outputs, dim=1).numpy()
                train_preds = np.argmax(train_probs, axis=1)
                train_acc = accuracy_score(y_train, train_preds)
                last_train_acc = train_acc
                
                # Capture activations on reference match
                ref_activations = model.get_layer_activations(ref_t)
                ref_out = torch.softmax(model(ref_t.unsqueeze(0)), dim=1).squeeze(0).tolist()
                
                # Capture layer weights, biases, and gradients
                network_state = []
                layer_idx = 0
                for name, layer in model.net.named_children():
                    if isinstance(layer, nn.Linear):
                        network_state.append({
                            "layer_index": layer_idx,
                            "name": f"Layer {name}",
                            "weights": layer.weight.data.tolist(),
                            "biases": layer.bias.data.tolist() if layer.bias is not None else [],
                            "gradients": layer.weight.grad.data.tolist() if layer.weight.grad is not None else []
                        })
                        layer_idx += 1
                        
                # Compute Confusion Matrix and ROC Curve
                cm = confusion_matrix(y_val, val_preds).tolist()
                
                # ROC Calculation for 3 classes
                roc_data = {}
                for c in range(3):
                    y_c = (y_val == c).astype(int)
                    probs_c = val_probs[:, c]
                    fpr, tpr, _ = roc_curve(y_c, probs_c)
                    # Downsample ROC to 10 points to keep payload small and fast
                    indices = np.linspace(0, len(fpr) - 1, 10, dtype=int)
                    roc_data[c] = {
                        "fpr": fpr[indices].tolist(),
                        "tpr": tpr[indices].tolist(),
                        "auc": float(auc(fpr, tpr))
                    }
            else:
                train_acc = last_train_acc
                ref_activations = []
                ref_out = []
                network_state = []
                cm = []
                roc_data = {}

        current_lr = optimizer.param_groups[0]['lr']

        state_event = {
            "epoch": epoch + 1,
            "train_loss": float(epoch_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "current_lr": float(current_lr),
            "ref_activations": ref_activations,
            "ref_predictions": ref_out,
            "network_state": network_state,
            "confusion_matrix": cm,
            "roc_curve": roc_data
        }
        
        yield f"data: {json.dumps(state_event)}\n\n"
        
        # Step LR scheduler
        if scheduler:
            scheduler.step()

        # Early Stopping check
        if early_stopping_patience > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    yield "data: " + json.dumps({"status": "early_stopped", "epoch": epoch + 1}) + "\n\n"
                    break
        
    # Save the trained model state dict and scaler to dedicated dynamic files
    dynamic_model_path = os.path.join(data_dir, "dynamic_model.pth")
    torch.save(model.state_dict(), dynamic_model_path)
    
    dynamic_scaler_path = os.path.join(data_dir, "dynamic_scaler.joblib")
    joblib.dump(scaler, dynamic_scaler_path)

    # Save model config for persistence
    dynamic_config_path = os.path.join(data_dir, "dynamic_config.json")
    with open(dynamic_config_path, "w", encoding="utf-8") as f:
        json.dump({
            "features": features_list,
            "hidden_layers": hidden_layers,
            "activation": activation_name
        }, f, indent=2)

    # Final cleanup & save experiment logic on client end
    yield "data: " + json.dumps({"status": "complete"}) + "\n\n"
