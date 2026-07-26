# ann_model/explainability.py
import torch
import numpy as np

def compute_saliency_map(model, input_vector, target_class):
    """
    Computes Saliency Map: gradient of target_class score with respect to input_vector.
    input_vector: 1D numpy array
    """
    model.eval()
    x_tensor = torch.FloatTensor(input_vector).unsqueeze(0)
    x_tensor.requires_grad_()
    
    outputs = model(x_tensor)
    score = outputs[0, target_class]
    
    model.zero_grad()
    score.backward()
    
    saliency = x_tensor.grad.data[0].abs().numpy().tolist()
    return saliency

def compute_integrated_gradients(model, input_vector, target_class, steps=25):
    """
    Computes Integrated Gradients along path from zero baseline.
    input_vector: 1D numpy array
    """
    model.eval()
    
    baseline = np.zeros_like(input_vector)
    
    # Calculate step size
    diff = input_vector - baseline
    
    # Accumulate gradients along path
    grads = []
    
    for i in range(steps + 1):
        alpha = float(i) / steps
        interpolated_input = baseline + alpha * diff
        
        x_step = torch.FloatTensor(interpolated_input).unsqueeze(0)
        x_step.requires_grad_()
        
        outputs = model(x_step)
        score = outputs[0, target_class]
        
        model.zero_grad()
        score.backward()
        
        grads.append(x_step.grad.data[0].numpy())
        
    avg_grads = np.mean(grads, axis=0)
    attributions = (diff * avg_grads).tolist()
    
    return attributions

def compute_shap_explanations(input_vector, feature_names, predicted_class, attributions=None):
    """
    Computes top positive and negative feature contributions for an individual prediction.
    Returns ranked dict of positive factors and negative factors with explanations.
    """
    if attributions is None:
        attributions = (input_vector - np.mean(input_vector)) / (np.std(input_vector) + 1e-5)

    positive_factors = []
    negative_factors = []

    for name, val, attr in zip(feature_names, input_vector, attributions):
        impact = float(attr)
        item = {"feature": name, "value": float(val), "impact": impact}
        if impact > 0:
            positive_factors.append(item)
        else:
            negative_factors.append(item)

    positive_factors = sorted(positive_factors, key=lambda x: x["impact"], reverse=True)
    negative_factors = sorted(negative_factors, key=lambda x: x["impact"])

    return {
        "predicted_class": int(predicted_class),
        "top_positive_features": positive_factors[:5],
        "top_negative_features": negative_factors[:5],
        "all_attributions": {f: float(a) for f, a in zip(feature_names, attributions)}
    }
