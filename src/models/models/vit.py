from cProfile import label
from turtle import forward
from typing import Tuple
import torch
import math
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from src.models.models.bases import  ClassificationModel
from src.models.models.transformers import Transformer
# helpers


class ViT(nn.Module):
    def __init__(self, *, input_shape: Tuple[int,int], 
                          patch_length: int,
                          dim: int,
                          num_hidden_layers: int,
                          num_attention_heads: int,
                          mlp_dim: int = 64,
                          pool: str = 'cls',
                          dim_head = 64,
                          dropout_rate: float = 0.,
                          emb_dropout_rate: float = 0.):

        super().__init__()
        n_timesteps, n_channels = input_shape
        assert n_timesteps % patch_length == 0, 'Input length must be divisible by the patch size.'


        num_patches = (n_timesteps // n_channels) 
        patch_dim = n_channels * patch_length
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            nn.BatchNorm1d(n_timesteps),
            Rearrange('b (l p) c -> b l (p c)', p = patch_length) ,
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout_rate)

        self.transformer = Transformer(dim, num_hidden_layers, num_attention_heads, dim_head, mlp_dim, dropout_rate)

        self.pool = pool
        self.to_latent = nn.Identity()


    
    def embed(self, x):

        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x
    
class ViTForClassification(ClassificationModel):
    
    def __init__(self, input_shape: Tuple[int,int], 
                       patch_length: int,
                       dim: int,
                       num_hidden_layers: int,
                       num_attention_heads: int,
                       mlp_dim: int = 64,
                       pool: str = 'cls',
                       dim_head = 64,
                       dropout_rate: float = 0.,
                       num_classes: int = 2,
                       emb_dropout_rate: float = 0.,
                       **kwargs):
        
        super().__init__(**kwargs)
        self.encoder = ViT(input_shape = input_shape,
                          patch_length = patch_length,
                          dim = dim,
                          num_hidden_layers = num_hidden_layers,
                          num_attention_heads = num_attention_heads,
                          mlp_dim = mlp_dim,
                          pool = pool,
                          dim_head = dim_head,
                          dropout_rate = dropout_rate,
                          emb_dropout_rate = emb_dropout_rate)

        # If we ever want to make the objective configurable we can do this:
        # https://pytorch-lightning.readthedocs.io/en/1.6.2/common/lightning_cli.html#class-type-defaults
        
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.objective = nn.CrossEntropyLoss()

    def forward(self, inputs_embeds, label, **kwargs):
        encoding = self.encoder.embed(inputs_embeds)
        preds = self.head(encoding)
        loss = self.objective(preds,label)
        return loss, preds

