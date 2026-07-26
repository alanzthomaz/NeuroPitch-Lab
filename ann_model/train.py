# ann_model/train.py
import os
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from ann_model.model import FootballANN

# Define features to use
FEATURES = [
    "elo_home", "elo_away",
    "rank_home", "rank_away", "rank_diff",
    "form_home", "form_away",
    "gs_home", "gs_away",
    "gc_home", "gc_away",
    "win_rate_home", "win_rate_away",
    "h2h_win_rate_home", "h2h_win_rate_away",
    "home_adv",
    "rest_home", "rest_away",
    "poss_home", "poss_away",
    "shots_home", "shots_away",
    "player_avail_home", "player_avail_away",
    "stat_prob_home", "stat_prob_draw", "stat_prob_away",
    "stat_xg_home", "stat_xg_away"
]

def train_ann():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    dataset_path = os.path.join(data_dir, "processed_dataset.csv")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Processed dataset not found at {dataset_path}. Run dataset.py first.")
        
    print("Loading compiled dataset...")
    df = pd.read_csv(dataset_path)
    
    X = df[FEATURES].values
    y = df["target"].values
    
    print(f"Dataset shape: {X.shape}, unique targets: {np.bincount(y)}")
    
    # Split into train/validation sets (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save the scaler
    scaler_path = os.path.join(data_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Convert to PyTorch Tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.LongTensor(y_val)
    
    # Instantiate Model
    input_dim = len(FEATURES)
    model = FootballANN(input_dim)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training Loop
    epochs = 60
    batch_size = 64
    
    print("Training PyTorch Neural Network...")
    for epoch in range(epochs):
        model.train()
        
        permutation = torch.randperm(X_train_t.size()[0])
        epoch_loss = 0.0
        
        for i in range(0, X_train_t.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_t[indices], y_train_t[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= X_train_t.size(0)
        
        # Validation Evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            
            # Accuracy
            val_probs = torch.softmax(val_outputs, dim=1).numpy()
            val_preds = np.argmax(val_probs, axis=1)
            val_acc = accuracy_score(y_val, val_preds)
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%")
            
    # Final Evaluation
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_val_t)
        probs = torch.softmax(final_outputs, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        
    acc = accuracy_score(y_val, preds)
    loss_val = log_loss(y_val, probs)
    
    # Calculate Brier Score
    # Brier Score = 1/N * sum_{i} sum_{k} (p_{ik} - y_{ik})^2
    y_val_onehot = np.zeros((len(y_val), 3))
    y_val_onehot[np.arange(len(y_val)), y_val] = 1.0
    brier = np.mean(np.sum((probs - y_val_onehot) ** 2, axis=1))
    
    print("\n--- Final Model Validation Metrics ---")
    print(f"Accuracy:    {acc*100:.2f}%")
    print(f"Log-loss:    {loss_val:.4f}")
    print(f"Brier score: {brier:.4f}")
    
    # Save the model
    model_path = os.path.join(data_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
if __name__ == "__main__":
    train_ann()
