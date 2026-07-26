# ann_model/feature_selection.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

def compute_permutation_importance(model_predict_proba_func, X_val, y_val, feature_names, metric="log_loss"):
    """
    Calculates Permutation Importance by measuring performance drop when shuffling each feature column.
    """
    baseline_probs = model_predict_proba_func(X_val)
    if metric == "log_loss":
        baseline_score = log_loss(y_val, baseline_probs)
    else:
        baseline_preds = np.argmax(baseline_probs, axis=1)
        baseline_score = accuracy_score(y_val, baseline_preds)

    importances = {}
    n_features = X_val.shape[1]
    np.random.seed(42)

    for i in range(n_features):
        fname = feature_names[i]
        X_permuted = X_val.copy()
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])

        perm_probs = model_predict_proba_func(X_permuted)
        if metric == "log_loss":
            perm_score = log_loss(y_val, perm_probs)
            # Higher log_loss indicates higher feature importance
            importances[fname] = float(perm_score - baseline_score)
        else:
            perm_preds = np.argmax(perm_probs, axis=1)
            perm_score = accuracy_score(y_val, perm_preds)
            # Accuracy drop indicates importance
            importances[fname] = float(baseline_score - perm_score)

    sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    return sorted_importances

def compute_shap_importance(model, X_sample, feature_names):
    """
    Calculates SHAP importance values for tree models or neural networks.
    Falls back gracefully if SHAP library is not present.
    """
    try:
        import shap
        explainer = shap.Explainer(model.predict_proba, X_sample)
        shap_values = explainer(X_sample)

        # Average absolute SHAP value across classes and samples
        if hasattr(shap_values, "values"):
            vals = np.abs(shap_values.values)
            if len(vals.shape) == 3:
                mean_shap = np.mean(vals, axis=(0, 2))
            else:
                mean_shap = np.mean(vals, axis=0)
        else:
            mean_shap = np.mean(np.abs(shap_values), axis=0)

        shap_dict = {feature_names[i]: float(mean_shap[i]) for i in range(len(feature_names))}
        return dict(sorted(shap_dict.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        # Fallback to random feature variance heuristic if SHAP explainer unavailable
        std_vals = np.std(X_sample, axis=0)
        heuristic_dict = {feature_names[i]: float(std_vals[i]) for i in range(len(feature_names))}
        return dict(sorted(heuristic_dict.items(), key=lambda x: x[1], reverse=True))

def select_top_features(importance_dict, min_importance=0.0, max_features=25):
    """
    Filters and returns top performing feature names.
    """
    filtered = [f for f, imp in importance_dict.items() if imp >= min_importance]
    return filtered[:max_features]

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    X_dummy = np.random.randn(200, 10)
    y_dummy = np.random.choice([0, 1, 2], size=200)
    fnames = [f"feature_{i}" for i in range(10)]
    
    rf = RandomForestClassifier(random_state=42).fit(X_dummy, y_dummy)
    imp = compute_permutation_importance(rf.predict_proba, X_dummy, y_dummy, fnames)
    print("Permutation Importance Top 5:", list(imp.items())[:5])
