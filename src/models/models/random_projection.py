from src.models.models.bases import ClassificationModel
from src.models.transforms import NonCollatedTransformRow

from typing import Tuple
import math

import torch
from torch import nn
from torch.nn import functional as F

class TwoDimensionalRandomProjection(nn.Module):
    """Takes a (b x n x m) matrix and compresses it down
       to (b x d) matrix"""

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(' + self.extra_repr() + ')'
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1)
    

class RandomProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

class RandomProjectionForClassifciation(ClassificationModel):
    
    def __init__(self, input_shape: Tuple[int,int], 
                       dim: int = 64,
                       num_classes: int = 2,
                       **kwargs):
        
        super().__init__(**kwargs)
        self.name = "RandomProjection"
        self.encoder = RandomProjection(input_shape, dim)
        
        # If we ever want to make the objective configurable we can do this:
        # https://pytorch-lightning.readthedocs.io/en/1.6.2/common/lightning_cli.html#class-type-defaults
        
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.objective = nn.CrossEntropyLoss()
        self.row_transform_class = NonCollatedTransformRow
        self.save_hyperparameters()

    def forward(self, label, **kwargs):
        encoding = self.encoder.embed(**kwargs)
        preds = self.head(encoding)
        loss = self.objective(preds,label)
        return loss, preds

default_feature_columns = ["heart_rate","steps", 
                           "sleep_classic_1", "sleep_classic_2","sleep_classic_3"]

default_mask_columns = ["missing_heart_rate","missing_steps", 
                        "sleep_classic_0", "sleep_classic_0","sleep_classic_0"]