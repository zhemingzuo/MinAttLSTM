import torch
import torch.nn as nn
from .cells import MinLSTMCell


class DeepMinAttLSTM(nn.Module):
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 num_heads: int = 4, num_layers: int = 2):
        super(DeepMinAttLSTM, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(MinLSTMCell(input_size, hidden_size))
        for _ in range(num_layers - 1):
            self.layers.append(MinLSTMCell(hidden_size, hidden_size))

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = x
        for layer in self.layers:
            out = layer(out)
        
        attn_out, _ = self.attention(out, out, out)
        out = self.dropout(attn_out + out)
        
        return self.fc(out[:, -1, :])


class OneStageMinAttLSTM(nn.Module):
    
    def __init__(self, dyn_input_size: int, stat_input_size: int, 
                 hidden_size: int, num_heads: int):
        super(OneStageMinAttLSTM, self).__init__()
        
        self.min_lstm = MinLSTMCell(dyn_input_size, hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        self.static_fc = nn.Sequential(
            nn.Linear(stat_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        
        self.head_daily = nn.Linear(hidden_size, 1)      # Main task: daily ice jam risk
        self.head_seasonal = nn.Linear(hidden_size, 1)   # Auxiliary task: seasonal statistics

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor):

        lstm_out = self.min_lstm(x_dyn)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        dyn_feat = attn_out[:, -1, :]  # Take last time step
        
        stat_feat = self.static_fc(x_stat)
        
        combined = torch.cat([dyn_feat, stat_feat], dim=1)
        fused = torch.relu(self.fusion(combined))
        
        return self.head_daily(fused), self.head_seasonal(fused)