import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AblationEnhancedSTAMT(nn.Module):
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
        # Ablation flags
        use_temporal_embedding=True,
        use_dilated_tconv=True,
        use_adaptive_memory=True,
        use_multi_resolution_output=True,
        use_cross_dim_projection=True,
        use_graph_learning=True,
        tconv_layers=4,
        dilated_convolution=True,
        multi_scale_graph=True,
        layer_normalization=True,
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        
        # Store ablation flags
        self.use_temporal_embedding = use_temporal_embedding
        self.use_dilated_tconv = use_dilated_tconv
        self.use_adaptive_memory = use_adaptive_memory
        self.use_multi_resolution_output = use_multi_resolution_output
        self.use_cross_dim_projection = use_cross_dim_projection
        self.use_graph_learning = use_graph_learning
        self.layer_normalization = layer_normalization

        # Choose time size by dataset
        if num_nodes in [170,307,358,883]:
            time = 288
        elif num_nodes in [250,266]:
            time = 48
        elif num_nodes > 200:
            time = 96
        else:
            time = 288

        # Temporal embedding (conditional)
        if self.use_temporal_embedding:
            self.Temb = TemporalEmbedding(time, channels)
            self.network_channel = channels * 2
        else:
            self.network_channel = channels

        # Temporal convolution (conditional)
        if self.use_dilated_tconv and dilated_convolution:
            self.tconv = DilatedTConv(channels, layer=tconv_layers, length=input_len)
        elif self.use_dilated_tconv:
            self.tconv = StandardTConv(channels, layer=tconv_layers, length=input_len)
        else:
            self.tconv = nn.Identity()

        self.start_conv = nn.Conv2d(input_dim, channels, kernel_size=(1,1))

        # Spatial block (conditional)
        if self.use_adaptive_memory:
            self.SpatialBlock = AblationEnhancedEncoder(
                device, self.network_channel, head=1,
                num_nodes=num_nodes, seq_length=input_len,
                dropout=dropout, memory_size=memory_size,
                use_graph_learning=use_graph_learning,
                use_cross_dim_projection=use_cross_dim_projection,
                multi_scale_graph=multi_scale_graph,
                layer_normalization=layer_normalization
            )
        else:
            self.SpatialBlock = SimpleEncoder(self.network_channel, dropout)

        self.fc_st = nn.Conv2d(self.network_channel, self.network_channel, kernel_size=(1,1))
        
        # Output module (conditional)
        if self.use_multi_resolution_output:
            self.output_module = MultiResolutionOutput(self.network_channel, output_len, num_nodes)
        else:
            self.output_module = SimpleOutput(self.network_channel, output_len)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
    
    def forward(self, history_data, return_uncertainty=False):
        x = history_data
        x = self.start_conv(x)
        
        if self.use_dilated_tconv:
            x = self.tconv(x)
        
        # Temporal embedding
        if self.use_temporal_embedding:
            tem_emb = self.Temb(history_data)
            data_st = torch.cat([x, tem_emb], dim=1)
        else:
            data_st = x
        
        data_st = self.SpatialBlock(data_st) + self.fc_st(data_st)
        
        if self.use_multi_resolution_output:
            main_pred, fine_pred, coarse_pred, uncertainty = self.output_module(data_st)
            if return_uncertainty:
                return main_pred, uncertainty
            else:
                return main_pred
        else:
            output = self.output_module(data_st)
            return output


class StandardTConv(nn.Module):
    """Standard temporal convolution without dilations"""
    def __init__(self, features=128, layer=4, length=12, dropout=0.1):
        super(StandardTConv, self).__init__()
        self.layers = nn.ModuleList()
        kernel_size = 3
        
        for i in range(layer):
            padding = kernel_size // 2
            conv_layer = nn.Conv2d(
                features, features,
                kernel_size=(1, kernel_size),
                padding=(0, padding)
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


class SimpleEncoder(nn.Module):
    """Simple encoder without attention mechanisms"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(d_model, d_model, (1, 1))
        self.conv2 = nn.Conv2d(d_model, d_model, (1, 1))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x + residual
        return x


class SimpleOutput(nn.Module):
    """Simple output module without multi-resolution"""
    def __init__(self, in_channels, out_seq_len):
        super().__init__()
        self.output_proj = nn.Conv2d(in_channels, out_seq_len, kernel_size=(1,1))
        
    def forward(self, x):
        return self.output_proj(x)


class AblationEnhancedEncoder(nn.Module):
    """Enhanced encoder with ablation support"""
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1, 
                 memory_size=4, use_graph_learning=True, use_cross_dim_projection=True,
                 multi_scale_graph=True, layer_normalization=True):
        super(AblationEnhancedEncoder, self).__init__()
        assert d_model % head == 0
        
        self.use_graph_learning = use_graph_learning
        self.use_cross_dim_projection = use_cross_dim_projection
        self.multi_scale_graph = multi_scale_graph
        self.layer_normalization = layer_normalization
        
        self.attention = AblationAdaptiveMemoryAttention(
            device, d_model, head, num_nodes, seq_length, dropout, memory_size,
            use_graph_learning, use_cross_dim_projection, multi_scale_graph
        )
        
        if layer_normalization:
            self.LayerNorm1 = LayerNorm([d_model, num_nodes, seq_length], elementwise_affine=False)
            self.LayerNorm2 = LayerNorm([d_model, num_nodes, seq_length], elementwise_affine=False)
        else:
            self.LayerNorm1 = nn.Identity()
            self.LayerNorm2 = nn.Identity()
            
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


class AblationAdaptiveMemoryAttention(nn.Module):
    """Adaptive memory attention with ablation support"""
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1, 
                 memory_size=4, use_graph_learning=True, use_cross_dim_projection=True,
                 multi_scale_graph=True):
        super(AblationAdaptiveMemoryAttention, self).__init__()
        assert d_model % head == 0
        
        self.d_k = d_model // head
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model
        self.device = device
        self.memory_size = memory_size
        self.use_graph_learning = use_graph_learning
        self.use_cross_dim_projection = use_cross_dim_projection
        self.multi_scale_graph = multi_scale_graph
        
        self.q = Conv(d_model)
        self.k = Conv(d_model)
        self.v = Conv(d_model)
        self.concat = Conv(d_model)
        
        # Memory banks (conditional)
        if memory_size > 0:
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

        # Graph structure learning (conditional)
        if use_graph_learning:
            apt_size = 10
            nodevecs = torch.randn(num_nodes, apt_size), torch.randn(apt_size, num_nodes)
            self.nodevec1, self.nodevec2 = [
                nn.Parameter(n.to(device), requires_grad=True) for n in nodevecs
            ]
            if multi_scale_graph:
                self.scale_weights = nn.Parameter(torch.ones(3))
            else:
                self.scale_weights = nn.Parameter(torch.ones(1))
        
        if use_cross_dim_projection:
            self.cross_dim_proj = nn.Conv2d(d_model, d_model, kernel_size=(1, 1))

    def forward(self, input, adj_list=None):
        batch_size = input.shape[0]
        
        # Graph learning (conditional)
        if self.use_graph_learning:
            base_adj = F.relu(torch.mm(self.nodevec1, self.nodevec2))
            scale1_adj = torch.softmax(base_adj, dim=-1)
            
            if self.multi_scale_graph:
                scale2_adj = torch.softmax(scale1_adj @ scale1_adj, dim=-1)
                scale3_adj = torch.softmax(scale2_adj @ scale1_adj, dim=-1)
                scale_weights = torch.softmax(self.scale_weights, dim=0)
            else:
                scale_weights = torch.softmax(self.scale_weights, dim=0)

        query, key, value = self.q(input), self.k(input), self.v(input)
        
        # Reshape for attention
        q = query.view(batch_size, -1, self.d_k, self.num_nodes, self.seq_length).permute(0,1,4,3,2)
        k = key.view(batch_size, -1, self.d_k, self.num_nodes, self.seq_length).permute(0,1,4,3,2)
        v = value.view(batch_size, -1, self.d_k, self.num_nodes, self.seq_length).permute(0,1,4,3,2)

        # Memory attention (conditional)
        if self.memory_size > 0:
            avg_feat = input.mean(dim=(2,3))
            mem_attn = self.attn_weights(avg_feat)
            mem_w = F.softmax(self.memory_importance * mem_attn, dim=-1)
            sel_mem = torch.einsum('bm,mhlnk->bhlnk', mem_w, self.memory_bank)
            
            attn_qk = torch.einsum('bhlnk,bhlmk->bhlnm', q, sel_mem) / math.sqrt(self.d_k)
            attn_qk = torch.softmax(attn_qk, dim=-1)
            attn_qkv = torch.einsum('bhlnm,bhlmk->bhlnk', attn_qk, v)
        else:
            # Simple self-attention without memory
            attn_qk = torch.einsum('bhlnk,bhlmk->bhlnm', q, k) / math.sqrt(self.d_k)
            attn_qk = torch.softmax(attn_qk, dim=-1)
            attn_qkv = torch.einsum('bhlnm,bhlmk->bhlnk', attn_qk, v)

        # Graph propagation (conditional)
        if self.use_graph_learning:
            a1 = torch.einsum('nm,bhlnk->bhlmk', scale1_adj, v)
            if self.multi_scale_graph:
                a2 = torch.einsum('nm,bhlnk->bhlmk', scale2_adj, v)
                a3 = torch.einsum('nm,bhlnk->bhlmk', scale3_adj, v)
                attn_graph = scale_weights[0]*a1 + scale_weights[1]*a2 + scale_weights[2]*a3
            else:
                attn_graph = scale_weights[0] * a1
            x = attn_qkv + attn_graph
        else:
            x = attn_qkv

        x = x.permute(0,1,4,3,2).contiguous().view(batch_size, self.d_model, self.num_nodes, self.seq_length)

        # Cross-dimensional projection (conditional)
        if self.use_cross_dim_projection:
            x_cross = self.cross_dim_proj(x)
            x = x + x_cross
            
        x = self.concat(x)
        
        if self.num_nodes not in [170, 358, 5]:
            x = x * self.weight + self.bias + x
            
        return x, self.weight, self.bias


# Import other necessary classes from your original model file
from model import (LayerNorm, GLU, Conv, TemporalEmbedding, 
                   DilatedTConv, MultiResolutionOutput)