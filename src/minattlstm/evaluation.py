import torch
import numpy as np
from sklearn.metrics import (balanced_accuracy_score, recall_score, log_loss, 
                             brier_score_loss, f1_score, precision_recall_curve, auc)


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_idx = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if np.any(bin_idx):
            delta = np.abs(np.mean(y_prob[bin_idx]) - np.mean(y_true[bin_idx]))
            ece += delta * np.sum(bin_idx) / len(y_true)
    
    return ece


def evaluate_model(model: torch.nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> dict:

    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
        y_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
        
        outputs = model(X_tensor).squeeze(1)
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities >= 0.5).float()
        
        y_true = y_tensor.cpu().numpy()
        preds = predictions.cpu().numpy()
        probs = probabilities.cpu().numpy()
        
        balanced_acc = balanced_accuracy_score(y_true, preds)
        recall = recall_score(y_true, preds)
        logloss = log_loss(y_true, probs)
        brier = brier_score_loss(y_true, probs)
        calibration_error = np.abs(np.mean(probs) - np.mean(y_true))
        f1 = f1_score(y_true, preds)
        
        precision, recall_vals, _ = precision_recall_curve(y_true, probs)
        pr_auc = auc(recall_vals, precision)
        ece = calculate_ece(y_true, probs)
        
        metrics = {
            "Balanced Accuracy": balanced_acc,
            "Recall": recall,
            "Log Loss": logloss,
            "Brier Score Loss": brier,
            "Calibration Error": calibration_error,
            "ECE": ece,
            "F1 Score": f1,
            "Precision-Recall AUC": pr_auc
        }
        
        print("\n=== Evaluation Results ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        
        return metrics