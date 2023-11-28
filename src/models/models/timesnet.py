from typing import List, Sequence, Dict, Tuple
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding


class Inception_Block_V1(nn.Module):
    """source: https://github.com/thuml/Time-Series-Library/blob/main/layers/Conv_Blocks.py"""
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]
    
class TimesBlock(nn.Module):
    """Source: https://github.com/thuml/Time-Series-Library/"""
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        # self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        self.seq_len = T
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res
    
class TimesNetEncoder(nn.Module):
    """
    Paper: https://openreview.net/pdf?id=ju_Uqw384Oq
    Implementation: https://github.com/thuml/Time-Series-Library/
    """
    def __init__(self, dim : int = 1, *args, **kwargs) -> None:
        # --- these are the default hyperparameters from Time-Series-Library ---
        configs = {
            "pred_len": 0,
            "enc_in": dim, # Input dimension <- should be 1 for univariate!
            "top_k": 1,
            "d_ff": 32,
            "d_model": 32, # Embedding dimension
            "num_kernels": 4,
            "num_layers": 1, # Number of TimesNet blocks in encoder
            "dropout": 0.1,
        }
        configs = SimpleNamespace(**configs) # make configs accessible by dots per TimesNet requirements (e.g., configs.num_layers)
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.num_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.projection = nn.Linear(configs.d_model*configs.seq_len, configs.d_model)
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
    
    def forward(self, ts: torch.Tensor) -> torch.Tensor:
        # embedding
        enc_out = self.enc_embedding(ts)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
            
        # Output based on a TimesNet Classifier
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output) # Shape: (batch_size, seq_length, d_model)
        
        # output shape: (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, d_model)
        return output