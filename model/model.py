import copy
import torch
import torch.nn as nn
from torch.nn import Linear as ln
from torch.nn import Dropout as dp
import torch.nn.functional as F
from model.layers import PositionalEncoding, STFE, STFE_r, Dlinear

class Embed2GraphByProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        m = torch.einsum('ijk,ipk->ijp', x, x)
        m = torch.unsqueeze(m, -1)

        return m

class GNNPredictor(nn.Module):
    def __init__(self, node_input_dim, out_size, roi_num=273, hidden_size=256):
        super().__init__()
        inner_dim = hidden_size
        self.roi_num = roi_num
        self.gcn1 = nn.Sequential(
            ln(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            ln(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn2 = nn.Sequential(
            ln(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            ln(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = torch.nn.BatchNorm1d(8)
        self.fcn = nn.Sequential(
            ln(8*roi_num, 256),
            nn.LeakyReLU(negative_slope=0.2),
            ln(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            ln(32, out_size)
        )


    def forward(self, m, node_feature):
        bz = m.shape[0]
        x = torch.einsum('ijk,ijp->ijp', m, node_feature)
        x = self.gcn1(x)
        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn1(x)
        x = x.reshape((bz, self.roi_num, -1))
        x = torch.einsum('ijk,ijp->ijp', m, x)
        x = self.gcn2(x)
        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((bz, self.roi_num, -1))
        x = x.view(bz,-1)

        return self.fcn(x)

class MLCGCN(nn.Module):
    def __init__(self, hidden_size, kernel_size, num_layer, num_head, roi_num, node_feature_dim, time_series, out_size, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.linear = nn.Sequential(
                        ln(time_series*4, hidden_size*4),
                        nn.GELU(),
                        dp(0.2),
                        ln(hidden_size*4, hidden_size),
                        nn.GELU(),
                        dp(0.2),
                        )
        self.emb2graph = Embed2GraphByProduct()
        self.position_embedding = PositionalEncoding(roi_num, hidden_size, device)
        stfe = STFE(hidden_size, kernel_size, num_head)
        predictor = GNNPredictor(node_feature_dim, out_size, roi_num=roi_num)
        self.go = ln(hidden_size,hidden_size)
        self.stfes = nn.ModuleList([copy.deepcopy(stfe) for i in range(num_layer)])
        self.predictors = nn.ModuleList([copy.deepcopy(predictor) for i in range(num_layer + 1)])
        self.output = nn.Sequential(
                        ln(out_size*(num_layer + 1), out_size*(num_layer + 1) // 2),
                        nn.GELU(),
                        dp(0.2),
                        ln(out_size*(num_layer + 1) // 2, out_size)
                        )


    def forward(self, x, nodes):
        b, k, d = x.shape  # b:batch_size k:roi_num(node_feature_size) d:seq_length
        g1 = nodes.view(b, -1)
        min_g1, _ = torch.min(g1, dim=1, keepdim=True)
        max_g1, _ = torch.max(g1, dim=1, keepdim=True)
        g1 = (g1 - min_g1) / (max_g1 - min_g1 + 1e-6)
        g1 = g1.view(b, k, k)
        graphs = [g1]
        outputs = []
        x = self.conv(x.view((b*k, 1, d))).view((b*k, 1, -1))
        x = x.view((b, k, -1))
        x = self.linear(x).permute(0, 2, 1)
        p = self.position_embedding(x)
        x = x + p
        x = x.permute(2, 0, 1)
        for stfe in self.stfes:
            x = stfe(x)
            f = self.go(x.permute(1, 0, 2))
            f = F.softmax(f, dim=-1)
            m = torch.einsum('ijk,ipk->ijp', f, f)
            g = torch.unsqueeze(m, -1)[:, :, :, 0]
            graphs.append(g)
        for g, predictor in zip(graphs, self.predictors):
            m = predictor(g, nodes)
            outputs.append(m)
        m = outputs[0]
        for i in range(1, len(outputs)):
            m = torch.cat((m,outputs[i]), dim=1)
        out = self.output(m)

        return out, graphs

class Model(nn.Module):
    def __init__(self, model_config):
        super(Model, self).__init__()

        self.hidden_size = model_config['embedding_size']
        self.kernel_size = int(model_config['window_size'])
        self.num_layer = model_config['num_trans_layers']
        self.num_head = model_config['num_heads']
        self.roi_num = model_config['roi_num']
        self.node_feature_dim = model_config['node_feature_dim']
        self.time_series = model_config['time_series']
        self.out_size = model_config['out_size']

        self.model = MLCGCN(hidden_size=self.hidden_size, kernel_size=self.kernel_size, num_layer=self.num_layer,
                            num_head=self.num_head, roi_num=self.roi_num, node_feature_dim=self.node_feature_dim,
                            time_series=self.time_series, out_size=self.out_size)

    def forward(self, x, nodes):
        x = self.model(x, nodes)
        return x