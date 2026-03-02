import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

from src.minattlstm import DeepMinAttLSTM, evaluate_model, setup_device


def train_l1():
    """Train L1 MinAttLSTM model with 5-fold cross-validation."""
    
    print("=" * 50)
    print("L1 Training: First-level MinAttLSTM")
    print("=" * 50)
    
    X_resampled = pd.read_csv('data/X_resampled_first_level.csv')
    y_resampled = pd.read_csv('data/y_resampled_first_level.csv', header=None).squeeze("columns")
    
    print(f"Data loaded: X shape {X_resampled.shape}, y shape {y_resampled.shape}")
    
    device = setup_device()
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    total_metrics = {
        k: [] for k in [
            "Balanced Accuracy", "Recall", "Log Loss", 
            "Brier Score Loss", "Calibration Error", 
            "F1 Score", "Precision-Recall AUC", "ECE"
        ]
    }
    
    final_model = None
    
    for fold, (train_index, test_index) in enumerate(kf.split(X_resampled)):
        print(f"\n{'='*30}")
        print(f"Fold {fold + 1}/5")
        print(f"{'='*30}")
        
        X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
        y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
        
        y_train = pd.to_numeric(y_train, errors='coerce').fillna(0).astype(float)
        y_test = pd.to_numeric(y_test, errors='coerce').fillna(0).astype(float)
        
        model = DeepMinAttLSTM(
            input_size=X_train.shape[1],
            hidden_size=128,
            output_size=1,
            num_layers=1  # Single layer for L1
        ).to(device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        train_ds = TensorDataset(
            torch.tensor(X_train.values, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.float32)
        )
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        
        # Training loop
        for epoch in range(20):
            model.train()
            epoch_loss = 0
            
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                
                optimizer.zero_grad()
                out = model(bx.unsqueeze(1))
                loss = criterion(out, by.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            scheduler.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/20, Loss: {epoch_loss/len(train_loader):.4f}")
        
        metrics = evaluate_model(model, X_test.values, y_test.values)
        for k, v in metrics.items():
            if k in total_metrics:
                total_metrics[k].append(v)
        
        final_model = model
    
    print("\n" + "="*50)
    print("Overall L1 Results (5-fold CV):")
    print("="*50)
    for k, v in total_metrics.items():
        print(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")


if __name__ == "__main__":
    train_l1()