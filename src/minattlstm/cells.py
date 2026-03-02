import torch
import torch.nn as nn


class MinLSTMCell(nn.Module):
    
    def __init__(self, input_size: int, hidden_size: int):
        super(MinLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.gate_projection = nn.Linear(input_size, hidden_size * 4)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        gates = self.gate_projection(x)  # (B, S, 4*H)
        f, i, o, z_raw = gates.chunk(4, dim=-1)
        
        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        z = torch.tanh(z_raw)
        
        c = torch.zeros(batch_size, self.hidden_size).to(x.device)
        h_list = []
        
        for t in range(seq_len):
            c = f[:, t, :] * c + i[:, t, :] * z[:, t, :]
            h = o[:, t, :] * torch.tanh(self.ln(c))
            h_list.append(h)
            
        h_seq = torch.stack(h_list, dim=1)
        return h_seq