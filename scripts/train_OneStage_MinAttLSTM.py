import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score, precision_recall_curve, auc

from src.minattlstm import OneStageMinAttLSTM, calculate_ece, setup_device


def train_and_evaluate():
    """Train One-Stage MinAttLSTM with 5-fold cross-validation."""
    
    print("=" * 50)
    print("One-Stage MinAttLSTM Training")
    print("=" * 50)
    
    X = pd.read_csv('data/OneStage_X.csv')
    y = pd.read_csv('data/OneStage_y.csv', header=None).squeeze()
    
    print(f"\n--- Sample Statistics ---")
    print(f"Total samples: {len(y)}")
    print(f"Positive samples (1): {sum(y)}")
    print(f"Negative samples (0): {len(y)-sum(y)}")
    print(f"Positive ratio: {sum(y)/len(y):.2%}")
    
    device = setup_device()
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n{'='*30}")
        print(f"Fold {fold + 1}/5")
        print(f"{'='*30}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        train_ds = TensorDataset(
            torch.tensor(X_train.values, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.float32)
        )
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        model = OneStageMinAttLSTM(
            dyn_input_size=2,
            stat_input_size=X.shape[1]-6,
            hidden_size=64,
            num_heads=4
        ).to(device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(50):
            epoch_loss = 0
            for b_x, b_y in train_loader:
                b_x, b_y = b_x.to(device), b_y.to(device).unsqueeze(1)
                
                # Split dynamic and static features
                x_dyn = b_x[:, :6].view(-1, 3, 2)  # Reshape to (batch, 3 days, 2 features)
                x_stat = b_x[:, 6:]
                
                # Forward pass with multi-task learning
                d_out, s_out = model(x_dyn, x_stat)
                loss = criterion(d_out, b_y) + 0.5 * criterion(s_out, b_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/50, Loss: {epoch_loss/len(train_loader):.4f}")
        
        model.eval()
        with torch.no_grad():
            t_X_te = torch.tensor(X_test.values, dtype=torch.float32).to(device)
            x_dyn_te = t_X_te[:, :6].view(-1, 3, 2)
            x_stat_te = t_X_te[:, 6:]
            
            y_logit, _ = model(x_dyn_te, x_stat_te)
            probs = torch.sigmoid(y_logit).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            y_true = y_test.values
            
            b_acc = balanced_accuracy_score(y_true, preds)
            rec = recall_score(y_true, preds)
            f1 = f1_score(y_true, preds)
            
            precision, recall_vals, _ = precision_recall_curve(y_true, probs)
            pr_auc = auc(recall_vals, precision)
            ece = calculate_ece(y_true, probs)
            
            fold_metrics.append([b_acc, rec, f1, pr_auc, ece])
            
            print(f"\nFold {fold+1} Results:")
            print(f"Balanced Accuracy: {b_acc:.4f}")
            print(f"Recall: {rec:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"PR-AUC: {pr_auc:.4f}")
            print(f"ECE: {ece:.4f}")
    
    mean_m = np.mean(fold_metrics, axis=0)
    print(f"\n[MinAttLSTM Overall Result]")
    headers = ["Acc", "B_Acc", "Recall", "LogLoss", "Brier", "ECE", "F1", "PR-AUC"]
    for h, m in zip(headers, mean_m):
        print(f"{h}: {m:.4f}")


if __name__ == "__main__":
    train_and_evaluate()