import torch
import numpy as np
import pandas as pd


def setup_device() -> torch.device:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


def reshape_data(X_df: pd.DataFrame, dyn_cols: int = 6, seq_len: int = 3) -> tuple:

    vals = X_df.values
    dyn_part = vals[:, :dyn_cols]
    X_dyn = dyn_part.reshape(len(vals), seq_len, dyn_cols // seq_len)
    X_stat = vals[:, dyn_cols:]
    return X_dyn, X_stat


def seed_everything(seed: int = 42):

    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False