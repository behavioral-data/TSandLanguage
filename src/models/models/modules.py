import math

import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules import dropout

from src.utils import get_logger
logger = get_logger(__name__)

def conv_l_out(l_in,kernel_size,stride,padding=0, dilation=1):
    return int(np.floor((l_in + 2 * padding - dilation * (kernel_size-1)-1)/stride + 1))

def get_final_conv_l_out(l_in,kernel_sizes,stride_sizes,
                        max_pool_kernel_size=None, max_pool_stride_size=None):
    l_out = l_in
    for kernel_size, stride_size in zip(kernel_sizes,stride_sizes):
        l_out = conv_l_out(l_out,kernel_size,stride_size)
        if max_pool_kernel_size and max_pool_kernel_size:
            l_out = conv_l_out(l_out, max_pool_kernel_size,max_pool_stride_size)
    return int(l_out)

def convtrans_l_out(l_in,kernel_size,stride,padding=0, dilation=1):
    return (l_in -1) *  stride - 2 * padding + dilation * (kernel_size - 1) + 1

def max_pool_l_out(l_in,kernel_size, stride, padding=0):
    return (l_in-1)*stride - 2*padding + kernel_size

def get_final_convtrans_l_out(l_in,kernel_sizes,stride_sizes,
                        max_pool_kernel_size=None, max_pool_stride_size=None):
    l_out = l_in
    for kernel_size, stride_size in zip(kernel_sizes,stride_sizes):
        l_out = convtrans_l_out(l_out,kernel_size,stride_size)
        if max_pool_kernel_size and max_pool_kernel_size:
            l_out = max_pool_l_out(l_out, max_pool_kernel_size,max_pool_stride_size)
    return int(l_out)

class CNNEncoder(nn.Module):
    def __init__(self, input_features, n_timesteps,
                kernel_sizes=[1], out_channels = [128], 
                stride_sizes=[1], max_pool_kernel_size = 3,
                max_pool_stride_size=2) -> None:

        n_layers = len(kernel_sizes)
        assert len(out_channels) == n_layers
        assert len(stride_sizes) == n_layers

        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.stride_sizes = stride_sizes
        self.max_pool_stride_size = max_pool_stride_size
        self.max_pool_kernel_size = max_pool_kernel_size
        self.conv_output_sizes = []

        super().__init__()
        self.input_features = input_features
        
        layers = []
        l_out = n_timesteps
        for i in range(n_layers):
            if i == 0:
                in_channels = input_features
            else:
                in_channels = out_channels[i-1]
            layers.append(nn.Conv1d(in_channels = in_channels,
                                    out_channels = out_channels[i],
                                    kernel_size=kernel_sizes[i],
                                    stride = stride_sizes[i]))
            layers.append(nn.ReLU())
            l_out = conv_l_out(l_out,kernel_sizes[i],stride_sizes[i])
            self.conv_output_sizes.append((l_out,))
            if max_pool_stride_size and max_pool_kernel_size:
                l_out = conv_l_out(l_out, max_pool_kernel_size,max_pool_stride_size)
                
                layers.append(nn.MaxPool1d(max_pool_kernel_size, stride=max_pool_stride_size, return_indices=True))
            layers.append(nn.LayerNorm([out_channels[i],l_out]))

        self.layers = nn.ModuleList(layers)
        self.final_output_length = get_final_conv_l_out(n_timesteps,kernel_sizes,stride_sizes, 
                                                        max_pool_kernel_size=max_pool_kernel_size, 
                                                        max_pool_stride_size=max_pool_stride_size)

        # Is used in the max pool decoder:
        self.max_indices = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Since max_indices might be referenced elsewhere we clear the contents 
        # rather than making a new list
        
        self.max_indices[:] = []
        for l in self.layers:
            if isinstance(l, nn.MaxPool1d):
                x, indices = l(x)
                self.max_indices.append(indices)
            else:
                x = l(x)
        return x

class CNNToTransformerEncoder(nn.Module):
    def __init__(self, input_features, num_attention_heads, num_hidden_layers, n_timesteps, kernel_sizes=[5,3,1], out_channels = [256,128,64], 
                stride_sizes=[2,2,2], dropout_rate=0.3, num_labels=2, positional_encoding = False) -> None:
        
        
        super(CNNToTransformerEncoder, self).__init__()

        self.input_dim = (n_timesteps,input_features)
        self.num_labels = num_labels
          

        self.input_embedding = CNNEncoder(input_features, n_timesteps=n_timesteps, kernel_sizes=kernel_sizes,
                                out_channels=out_channels, stride_sizes=stride_sizes)
        
        self.d_model = out_channels[-1]
        final_length = self.input_embedding.final_output_length
        
        self.final_length = final_length
        
        if self.input_embedding.final_output_length < 1:
            raise ValueError("CNN final output dim is <1 ")                                
        
        if positional_encoding:
            self.positional_encoding = PositionalEncoding(self.d_model, final_length)
        else:
            self.positional_encoding = None

        self.blocks = nn.ModuleList([
            EncoderBlock(self.d_model, num_attention_heads, dropout_rate) for _ in range(num_hidden_layers)
        ])
        
        self.provided_train_dataloader = None        
        if num_attention_heads > 0:
            self.name = "CNNToTransformerEncoder"
        else:
            self.name = "CNN"
            
        self.base_model_prefix = self.name


    def forward(self,inputs):
        return self.encode(inputs)

    def encode(self, inputs_embeds):
    
        x = inputs_embeds.transpose(1, 2)
        x = self.input_embedding(x)
        x = x.transpose(1, 2)

        if self.positional_encoding:
            x = self.positional_encoding(x)

        for l in self.blocks:
            x = l(x)
        return x



class CNNDecoder(nn.Module):

    def __init__(self, input_features, input_length,
                kernel_sizes=[1], out_channels = [128], 
                stride_sizes=[1], max_pool_kernel_size = 3,
                max_pool_stride_size=2, max_indices=None,
                unpool_output_sizes=None) -> None:

        n_layers = len(kernel_sizes)
        assert len(out_channels) == n_layers
        assert len(stride_sizes) == n_layers

        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.stride_sizes = stride_sizes
        self.max_pool_stride_size = max_pool_stride_size
        self.max_pool_kernel_size = max_pool_kernel_size
        
        # A symmetric encoder
        self.max_indices = max_indices

        super(CNNDecoder,self).__init__()
        self.input_features = input_features
        logger.warning("The CNN Decoder uses hard-coded features and will probably break if you try anything fancy")
        layers = []
        for i in range(n_layers):
            if i == 0:
                in_channels = input_features
            else:
                in_channels = out_channels[i-1]

            if max_pool_stride_size and max_pool_kernel_size:
                layers.append(nn.MaxUnpool1d(max_pool_kernel_size, stride=max_pool_stride_size))

            if i in (0,):
                output_padding=(1,)
            else:
                output_padding = (0,)

            layers.append(nn.ConvTranspose1d(in_channels = in_channels,
                                    out_channels = out_channels[i],
                                    kernel_size=kernel_sizes[i],
                                    stride = stride_sizes[i],
                                    output_padding=output_padding))
            layers.append(nn.ReLU())
            
        self.layers = nn.ModuleList(layers)
        self.unpool_output_sizes = unpool_output_sizes
        self.final_output_length = get_final_conv_l_out(input_length,kernel_sizes,stride_sizes, 
                                                        max_pool_kernel_size=max_pool_kernel_size, 
                                                        max_pool_stride_size=max_pool_stride_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1,2)
        max_pool_id = 0
        for l in self.layers:
            if isinstance(l,nn.MaxUnpool1d):
                inds = self.max_indices.pop(-1)
                x = l(x,indices=inds, output_size=self.unpool_output_sizes[max_pool_id])
                max_pool_id+=1
            else:
                x = l(x)
        x = x.transpose(1,2)
        return x

    @staticmethod
    def from_inverse_of_encoder(encoder):
            # decoder_out_channels = encoder.out_channels[::-1]
            return CNNDecoder(
                encoder.out_channels[-1],
                encoder.final_output_length,
                out_channels = encoder.out_channels[:-1][::-1] + [encoder.input_features],
                stride_sizes = encoder.stride_sizes[::-1],
                kernel_sizes = encoder.kernel_sizes[::-1],
                max_pool_kernel_size=encoder.max_pool_kernel_size,
                max_pool_stride_size=encoder.max_pool_stride_size,
                max_indices=encoder.max_indices,
                unpool_output_sizes=encoder.conv_output_sizes[::-1]
            )




# All of the following are from SaND:
# -----------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(seq_len, d_model)

        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x) -> torch.Tensor:
        seq_len = x.shape[1]
        x = math.sqrt(self.d_model) * x
        x = x + self.pe[:, :seq_len].requires_grad_(False)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, layer: nn.Module, embed_dim: int, p=0.1) -> None:
        super(ResidualBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(p=p)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [N, seq_len, features]
        :return: [N, seq_len, features]
        """
        if isinstance(self.layer, nn.MultiheadAttention):
            src = x.transpose(0, 1)     # [seq_len, N, features]
            output, self.attn_weights = self.layer(src, src, src)
            output = output.transpose(0, 1)     # [N, seq_len, features]

        else:
            output = self.layer(x)

        output = self.dropout(output)
        output = self.norm(x + output)
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.hidden_size = hidden_size

        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size * 2, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_size * 2, hidden_size, 1)
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.transpose(1, 2)
        tensor = self.conv(tensor)
        tensor = tensor.transpose(1, 2)

        return tensor


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_head: int, dropout_rate=0.1) -> None:
        super(EncoderBlock, self).__init__()
        self.attention = ResidualBlock(
            nn.MultiheadAttention(embed_dim, num_head), embed_dim, p=dropout_rate
        )
        self.ffn = ResidualBlock(PositionWiseFeedForward(embed_dim), embed_dim, p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.ffn(x)
        return x


class DenseInterpolation(nn.Module):
    def __init__(self, seq_len: int, factor: int) -> None:
        """
        :param seq_len: sequence length
        :param factor: factor M
        """
        super(DenseInterpolation, self).__init__()

        W = np.zeros((factor, seq_len), dtype=np.float32)

        for t in range(seq_len):
            s = np.array((factor * (t + 1)) / seq_len, dtype=np.float32)
            for m in range(factor):
                tmp = np.array(1 - (np.abs(s - (1+m)) / factor), dtype=np.float32)
                w = np.power(tmp, 2, dtype=np.float32)
                W[m, t] = w

        W = torch.tensor(W).float().unsqueeze(0)
        self.register_buffer("W", W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.W.repeat(x.shape[0], 1, 1).requires_grad_(False)
        u = torch.bmm(w, x)
        return u.transpose_(1, 2)


class ClassificationModule(nn.Module):
    def __init__(self, d_model: int, factor: int, num_class: int,
                dropout_p:float = 0.0) -> None:
        super(ClassificationModule, self).__init__()
        self.d_model = d_model
        self.factor = factor
        self.num_class = num_class
        self.dropout = nn.Dropout(dropout_p)

        self.fc = nn.Linear(int(d_model * factor), num_class)

        # nn.init.normal_(self.fc.weight, std=0.02)
        # nn.init.normal_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().view(-1, int(self.factor * self.d_model))
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def reset_parameters(self):
        self.fc.reset_parameters()


class DayOfWeekEmbedding(nn.Module):
    """Acts as a lookup table for day of week embeddings."""
    def __init__(self, embed_dim: int) -> None:
        super(DayOfWeekEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.day_of_week_embedding = nn.Embedding(7, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x % 7
        x = self.day_of_week_embedding(x)
        return x