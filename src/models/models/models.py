"""
====================================================
Architectures For Behavioral Representation Learning     
====================================================
`Project repository available here  <https://github.com/behavioral-data/SeattleFluStudy>`_

This module contains the architectures used for behavioral representation learning in the reference paper. 
Particularly, the two main classes in the module implement a CNN architecture and the novel CNN-Transformer
architecture. 

**Classes**
    :class CNNEncoder: 
    :class CNNToTransformerEncoder:

"""

from copy import copy

from typing import Dict, Tuple,  Union, Any, Optional, List, Callable
import torch
import torch.nn as nn

from sktime.classification.hybrid import HIVECOTEV2 as BaseHIVECOTEV2
import xgboost as xgb

import src.models.models.modules as modules
from src.utils import get_logger
from src.models.loops import DummyOptimizerLoop, NonNeuralLoop
from src.models.models.bases import ClassificationModel, NonNeuralMixin

from src.models.losses import build_loss_fn
from torch.utils.data.dataloader import DataLoader
from wandb.plot.roc_curve import roc_curve

logger = get_logger(__name__)


"""
 Helper functions:
"""

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


       
class CNNToTransformerClassifier(ClassificationModel):

    def __init__(self, num_attention_heads : int = 4, num_hidden_layers: int = 4,  
                kernel_sizes=[5,3,1], out_channels = [256,128,64], 
                stride_sizes=[2,2,2], dropout_rate=0.3, num_labels=2, 
                positional_encoding = False, pretrained_ckpt_path : Optional[str] = None,
                loss_fn="CrossEntropyLoss", pos_clas_weight=1, neg_class_weight=1, **kwargs) -> None:

        super().__init__(**kwargs)

        if num_hidden_layers == 0:
            self.name = "CNNClassifier"
        else:
            self.name = "CNNToTransformerClassifier"
        n_timesteps, input_features = kwargs.get("input_shape")

        self.criterion = build_loss_fn(loss_fn=loss_fn, task_type="classification")


        self.encoder = modules.CNNToTransformerEncoder(input_features, num_attention_heads, num_hidden_layers,
                                                      n_timesteps, kernel_sizes=kernel_sizes, out_channels=out_channels,
                                                      stride_sizes=stride_sizes, dropout_rate=dropout_rate, num_labels=num_labels,
                                                      positional_encoding=positional_encoding)
        
        self.head = modules.ClassificationModule(self.encoder.d_model, self.encoder.final_length, num_labels)

        if pretrained_ckpt_path:
            ckpt = torch.load(pretrained_ckpt_path)
            try:
                self.load_state_dict(ckpt['state_dict'])
            
            #TODO: Nasty hack for reverse compatability! 
            except RuntimeError:
                new_state_dict = {}
                for k,v in ckpt["state_dict"].items():
                    if not "encoder" in k :
                        new_state_dict["encoder."+k] = v
                    else:
                        new_state_dict[k] = v
                self.load_state_dict(new_state_dict, strict=False)

        self.save_hyperparameters()
        
    def forward(self, inputs_embeds, label, **kwargs):
        encoding = self.encoder.encode(inputs_embeds)
        preds = self.head(encoding)
        loss =  self.criterion(preds,label)
        return loss, preds


class MaskedCNNToTransformerClassifier(CNNToTransformerClassifier):

    def __init__(self, mask_train=True, mask_eval=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_train = mask_train
        self.mask_eval = mask_eval

    def training_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:

        x = batch["inputs_embeds"].type(torch.cuda.FloatTensor)
        y = batch["label"]

        if self.mask_train:
            mask = batch["mask"].bool()
            x = x[mask]
            y = y[mask]

        loss,preds = self.forward(x,y)

        self.log("train/loss", loss.item(),on_step=True)
        preds = preds.detach()

        y = y.int().detach()
        self.train_metrics.update(preds,y)

        if self.is_classifier:
            self.train_preds.append(preds.detach().cpu())
            self.train_labels.append(y.detach().cpu())

        return {"loss": loss, "preds": preds, "labels":y}

    def validation_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        x = batch["inputs_embeds"].type(torch.cuda.FloatTensor)
        y = batch["label"]


        if self.mask_eval:
            mask = batch["mask"].bool()
            x = x[mask]
            y = y[mask]

        loss,preds = self.forward(x,y)

        self.log("val/loss", loss.item(),on_step=True,sync_dist=True)


        if self.is_classifier:
            self.val_preds.append(preds.detach())
            self.val_labels.append(y.detach())

        self.val_metrics.update(preds,y)
        return {"loss":loss, "preds": preds, "labels":y}

    def test_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:

        x = batch["inputs_embeds"].type(torch.cuda.FloatTensor)
        y = batch["label"]
        dates = batch["end_date_str"]
        participant_ids = batch["participant_id"]

        if self.mask_eval:
            mask = batch["mask"].bool()
            x = x[mask]
            y = y[mask]

        loss,preds = self.forward(x,y)

        self.log("test/loss", loss.item(),on_step=True,sync_dist=True)


        self.test_preds.append(preds.detach())
        self.test_labels.append(y.detach())
        self.test_participant_ids.append(participant_ids)
        self.test_dates.append(dates)

        self.test_metrics.update(preds,y)
        return {"loss":loss, "preds": preds, "labels":y}



class WeakCNNToTransformerClassifier(CNNToTransformerClassifier):

    def training_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:

        x = batch["inputs_embeds"].type(torch.cuda.FloatTensor)
        y = batch["label"]
        y_bar = batch["weak_label"].float()

        loss,preds = self.forward(x,y_bar)

        self.log("train/loss", loss.item(),on_step=True)
        preds = preds.detach()

        y = y.int().detach()
        self.train_metrics.update(preds,y)

        if self.is_classifier:
            self.train_preds.append(preds.detach().cpu())
            self.train_labels.append(y.detach().cpu())

        return {"loss": loss, "preds": preds, "labels": y}

class TransformerClassifier(ClassificationModel):
    
    def __init__(
        self,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 4,
        dropout_rate: float = 0.,
        num_labels: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.name = "TransformerClassifier"
        n_timesteps, input_features = kwargs.get("input_shape")

        self.criterion = nn.CrossEntropyLoss()
        self.blocks = nn.ModuleList([
            modules.EncoderBlock(input_features, num_attention_heads, dropout_rate) for _ in range(num_hidden_layers)
        ])
        
        self.head = modules.ClassificationModule(input_features, n_timesteps, num_labels)

    
    def forward(self, inputs_embeds,label, **kwargs):
        x = inputs_embeds
        for l in self.blocks:
            x = l(x)

        preds = self.head(x)
        loss =  self.criterion(preds,label)
        return loss, preds



class HIVECOTE2(NonNeuralMixin,ClassificationModel):
    
#     def __init__(
#         self,
#         n_jobs: int = -1,
#         **kwargs,
#     ) -> None:
#         super().__init__(**kwargs)
#         self.base_model = BaseHIVECOTEV2(n_jobs=n_jobs)
#         self.fit_loop = NonNeuralLoop()
#         self.optimizer_loop = DummyOptimizerLoop()
#         self.save_hyperparameters()
    
    def forward(self, inputs_embeds,labels):
        return self.base_model(inputs_embeds)

class XGBoost(xgb.XGBClassifier, NonNeuralMixin,ClassificationModel):

    def __init__(
            self,
            random_state=None,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.fit_loop = NonNeuralLoop()
        self.optimizer_loop = DummyOptimizerLoop()
        self.save_hyperparameters()
        self.name = "XGBoostClassifier"
        self.random_state = random_state

    def forward(self, inputs_embeds,labels):
        raise NotImplementedError
