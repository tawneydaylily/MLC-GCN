import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        s_2i = torch.arange(0, d_model, step=2, device=device).float()
        d_2i = torch.arange(1, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (s_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (d_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()

        return self.encoding[:seq_len, :]

class Dlinear_r(nn.Module):
    def __init__(self, hidden_size=32, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.linear_seasonal = nn.Linear(hidden_size, hidden_size)
        self.linear_trend = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size*4),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_size*4, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        k, b, d = x.shape
        x = x.reshape((b*k, 1, d))
        x = x.permute(0, 2, 1)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        trend = torch.cat([front, x, end], dim=1)
        trend = self.avg(trend.permute(0, 2, 1))
        res = x.permute(0, 2, 1) - trend
        seasonal_output = self.linear_seasonal(res)
        trend_output = self.linear_trend(trend)

        x = x + seasonal_output + trend_output
        x = x.view((b, k, -1))
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(1, 0, 2)

        return x

class Dlinear(nn.Module):
    def __init__(self, hidden_size=32, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.linear_seasonal = nn.Linear(hidden_size, hidden_size)
        self.linear_trend = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size*4),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_size*4, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        k, b, d = x.shape
        x = x.reshape((b*k, 1, d))
        x = x.permute(0, 2, 1)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        trend = torch.cat([front, x, end], dim=1)
        trend = self.avg(trend.permute(0, 2, 1))
        res = x.permute(0, 2, 1) - trend
        seasonal_output = self.linear_seasonal(res)
        trend_output = self.linear_trend(trend)

        x = seasonal_output + trend_output
        x = x.view((b, k, -1))
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(1, 0, 2)

        return x

class STFE_r(nn.Module):
    def __init__(self, hidden_size=32, kernel_size=5, num_head=8, node_num=273):
        super().__init__()
        self.trans = nn.TransformerEncoderLayer(hidden_size, num_head, dim_feedforward=hidden_size*4)
        self.tc = Dlinear(hidden_size, kernel_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        k, b, d = x.shape
        x1 = self.trans(x)
        x2 = self.tc(x)
        x = self.norm(x + F.relu(x1 + x2))
        x = F.dropout(x, 0.2)

        return x

class STFE(nn.Module):
    def __init__(self, hidden_size=32, kernel_size=5, num_head=8, node_num=273):
        super().__init__()
        self.trans = nn.TransformerEncoderLayer(hidden_size, num_head, dim_feedforward=hidden_size*4)
        self.tc = Dlinear(hidden_size, kernel_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        k, b, d = x.shape
        x1 = self.trans(x)
        x2 = self.tc(x)
        x = self.norm(F.relu(x1 + x2))
        x = F.dropout(x, 0.2)

        return x