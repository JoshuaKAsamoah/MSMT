from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input


class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out


class Conv(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super().__init__()
        self.time = time
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)
        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        # x: [B, input_dim=3, N, T]
        B, _, N, T = x.shape

        # time-of-day embedding
        day_feat = x[:, 1, :, :]  # [B, N, T]
        day_idx = (day_feat[:, :, -1] * self.time).long().clamp(0, self.time - 1)  # [B, N]
        td = self.time_day[day_idx]  # [B, N, C]
        td = td.permute(0, 2, 1).unsqueeze(-1)  # [B, C, N, 1]

        # day-of-week embedding
        week_feat = x[:, 2, :, :]  # [B, N, T]
        wk_idx = week_feat[:, :, -1].long().clamp(0, 6)  # [B, N]
        tw = self.time_week[wk_idx]  # [B, N, C]
        tw = tw.permute(0, 2, 1).unsqueeze(-1)  # [B, C, N, 1]

        # combine and expand
        tem_emb = td + tw  # [B, C, N, 1]
        tem_emb = tem_emb.expand(-1, -1, -1, T)  # [B, C, N, T]

        return tem_emb


class DilatedTConv(nn.Module):
    """Enhanced temporal convolution with dilations to capture longer-range dependencies"""
    def __init__(self, features=128, layer=4, length=12, dropout=0.1):
        super(DilatedTConv, self).__init__()
        self.layers = nn.ModuleList()
        kernel_size = 3
        
        for i in range(layer):
            dilation = 2**i
            padding = dilation
            conv_layer = nn.Conv2d(
                features,
                features,
                kernel_size=(1, kernel_size),
                padding=(0, padding),
                dilation=(1, dilation)
            )
            nn.init.kaiming_normal_(conv_layer.weight, nonlinearity='relu')
            nn.init.zeros_(conv_layer.bias)
            self.layers.append(nn.Sequential(
                conv_layer,
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        self.output_proj = nn.Conv2d(features, features, (1, 1))
            
    def forward(self, x):
        identity = x
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer(x)
            if i > 0:
                x = x + residual
        x = self.output_proj(x)
        x = x + identity
        return x


class AdaptiveMemoryAttention(nn.Module):
    """Enhanced spatial attention with multi-scale processing and adaptive memory"""
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1, memory_size=4):
        super(AdaptiveMemoryAttention, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model
        self.device = device
        self.memory_size = memory_size
        
        self.q = Conv(d_model)
        self.k = Conv(d_model)
        self.v = Conv(d_model)
        self.concat = Conv(d_model)
        
        # memory banks
        self.memory_bank = nn.Parameter(
            torch.randn(memory_size, head, seq_length, num_nodes, self.d_k)
        )
        nn.init.xavier_uniform_(self.memory_bank)
        self.memory_importance = nn.Parameter(torch.ones(memory_size))
        self.attn_weights = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, memory_size),
            nn.Softmax(dim=-1)
        )

        self.weight = nn.Parameter(torch.ones(d_model, num_nodes, seq_length))
        self.bias = nn.Parameter(torch.zeros(d_model, num_nodes, seq_length))

        # graph structure learning
        apt_size = 10
        nodevecs = torch.randn(num_nodes, apt_size), torch.randn(apt_size, num_nodes)
        self.nodevec1, self.nodevec2 = [
            nn.Parameter(n.to(device), requires_grad=True) for n in nodevecs
        ]
        self.scale_weights = nn.Parameter(torch.ones(3))
        self.cross_dim_proj = nn.Conv2d(d_model, d_model, kernel_size=(1, 1))

    def forward(self, input, adj_list=None):
        batch_size = input.shape[0]
        base_adj = F.relu(torch.mm(self.nodevec1, self.nodevec2))
        scale1_adj = torch.softmax(base_adj, dim=-1)
        scale2_adj = torch.softmax(scale1_adj @ scale1_adj, dim=-1)
        scale3_adj = torch.softmax(scale2_adj @ scale1_adj, dim=-1)
        scale_weights = torch.softmax(self.scale_weights, dim=0)

        query, key, value = self.q(input), self.k(input), self.v(input)
        # reshape for attention
        q = query.view(batch_size, -1, self.d_k, self.num_nodes, self.seq_length).permute(0,1,4,3,2)
        k = key.view(batch_size, -1, self.d_k, self.num_nodes, self.seq_length).permute(0,1,4,3,2)
        v = value.view(batch_size, -1, self.d_k, self.num_nodes, self.seq_length).permute(0,1,4,3,2)

        avg_feat = input.mean(dim=(2,3))
        mem_attn = self.attn_weights(avg_feat)  # [B, M]
        mem_w = F.softmax(self.memory_importance * mem_attn, dim=-1)
        sel_mem = torch.einsum('bm,mhlnk->bhlnk', mem_w, self.memory_bank)

        attn_qk = torch.einsum('bhlnk,bhlmk->bhlnm', q, sel_mem) / math.sqrt(self.d_k)
        attn_qk = torch.softmax(attn_qk, dim=-1)
        attn_qkv = torch.einsum('bhlnm,bhlmk->bhlnk', attn_qk, v)

        # graph propagation
        a1 = torch.einsum('nm,bhlnk->bhlmk', scale1_adj, v)
        a2 = torch.einsum('nm,bhlnk->bhlmk', scale2_adj, v)
        a3 = torch.einsum('nm,bhlnk->bhlmk', scale3_adj, v)
        attn_graph = scale_weights[0]*a1 + scale_weights[1]*a2 + scale_weights[2]*a3

        x = attn_qkv + attn_graph
        x = x.permute(0,1,4,3,2).contiguous().view(batch_size, self.d_model, self.num_nodes, self.seq_length)

        x_cross = self.cross_dim_proj(x)
        x = x + x_cross
        x = self.concat(x)
        if self.num_nodes not in [170, 358, 5]:
            x = x * self.weight + self.bias + x
        return x, self.weight, self.bias


class EnhancedEncoder(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1, memory_size=4):
        super(EnhancedEncoder, self).__init__()
        assert d_model % head == 0
        self.attention = AdaptiveMemoryAttention(device, d_model, head, num_nodes, seq_length, dropout, memory_size)
        self.LayerNorm1 = LayerNorm([d_model, num_nodes, seq_length], elementwise_affine=False)
        self.LayerNorm2 = LayerNorm([d_model, num_nodes, seq_length], elementwise_affine=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.glu = GLU(d_model)

    def forward(self, input, adj_list=None):
        x, weight, bias = self.attention(input)
        x = x + input
        x = self.LayerNorm1(x)
        x = self.dropout1(x)

        res = x
        x = self.glu(x)
        x = x + res

        x = x * weight + bias + x
        x = self.LayerNorm2(x)
        x = self.dropout2(x)
        return x


class MultiResolutionOutput(nn.Module):
    """Multi-resolution output module with uncertainty estimation"""
    def __init__(self, in_channels, out_seq_len, num_nodes):
        super().__init__()
        self.main_head       = nn.Conv2d(in_channels, out_seq_len, kernel_size=(1,1))
        self.fine_head       = nn.Conv2d(in_channels, out_seq_len//2, kernel_size=(1,1))
        self.coarse_head     = nn.Conv2d(in_channels, out_seq_len//2, kernel_size=(1,1))
        self.uncertainty_head= nn.Conv2d(in_channels, out_seq_len,   kernel_size=(1,1))
        self.integrator      = nn.Conv2d(in_channels, out_seq_len,   kernel_size=(1,1))
        self.out_seq_len     = out_seq_len

    def forward(self, x):
        main_pred    = self.main_head(x)
        fine_pred    = self.fine_head(x)
        coarse_pred  = self.coarse_head(x)
        uncertainty  = F.softplus(self.uncertainty_head(x))

        # integrate fine/coarse for first half
        batch_size, _, num_nodes, _ = x.shape
        integrated = main_pred.clone()
        w = torch.exp(-uncertainty[:, :self.out_seq_len//2])
        w = w / (w + 1.0)
        integrated[:, :self.out_seq_len//2] = (
            w * fine_pred +
            (1-w) * main_pred[:, :self.out_seq_len//2]
        )
        return main_pred, fine_pred, coarse_pred, uncertainty


class EnhancedSTAMT(nn.Module):
    def __init__(
        self,
        device,
        input_dim=3,
        channels=64,
        num_nodes=170,
        input_len=12,
        output_len=12,
        dropout=0.1,
        memory_size=4,
    ):
        super().__init__()
        self.device      = device
        self.num_nodes   = num_nodes
        self.input_len   = input_len
        self.input_dim   = input_dim
        self.output_len  = output_len

        # choose time size by dataset
        if num_nodes in [170,307,358,883]:
            time = 288
        elif num_nodes in [250,266]:
            time = 48
        elif num_nodes > 200:
            time = 96
        else:
            time = 288

        self.Temb       = TemporalEmbedding(time, channels)
        self.tconv      = DilatedTConv(channels, layer=4, length=input_len)
        self.start_conv = nn.Conv2d(input_dim, channels, kernel_size=(1,1))
        self.network_channel = channels*2

        self.SpatialBlock = EnhancedEncoder(
            device, self.network_channel, head=1,
            num_nodes=num_nodes, seq_length=input_len,
            dropout=dropout, memory_size=memory_size
        )
        self.fc_st        = nn.Conv2d(self.network_channel, self.network_channel, kernel_size=(1,1))
        self.output_module= MultiResolutionOutput(self.network_channel, output_len, num_nodes)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
    
    def forward(self, history_data, return_uncertainty=False):
        # history_data comes in as: [B, C_in, N, T]
        # DON'T PERMUTE - it's already in the right format!
        # print("Input history_data shape:", history_data.shape)
        
        # Just use the input directly - it's already [B, C_in, N, T]
        x = history_data
        
        # Apply your convolutions
        x = self.start_conv(x)  # [B, channels, N, T]
        x = self.tconv(x)       # [B, channels, N, T]
        
        # For temporal embedding
        hist_for_tem = history_data  # Already in [B, C_in, N, T] format
        tem_emb = self.Temb(hist_for_tem)  # [B, channels, N, T]
        
        # Continue with the model flow
        data_st = torch.cat([x, tem_emb], dim=1)  # [B, 2*channels, N, T]
        data_st = self.SpatialBlock(data_st) + self.fc_st(data_st)
        
        main_pred, fine_pred, coarse_pred, uncertainty = self.output_module(data_st)
        # keep only the final slice along the input-time axis
        main_pred = main_pred[..., -1]        # now [B, T_out, N]

        if return_uncertainty:
            return main_pred, uncertainty
        else:
            return main_pred
 

