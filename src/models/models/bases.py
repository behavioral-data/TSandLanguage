from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
from torch import nn
import torchmetrics
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable

from torch import Tensor

from src.models.eval import (TorchMetricClassification, TorchMetricRegression, TorchMetricMultimodal, compute_log_probs)
from src.models.tasks import MultimodalMCQTask

from src.utils import (get_logger,upload_pandas_df_to_wandb)

logger = get_logger(__name__)

class SensingModel(pl.LightningModule):
    '''
    This is the base class for building sensing models.
    All trainable models should subclass this.
    '''

    def __init__(self, metric_class : torchmetrics.MetricCollection, 
                       learning_rate : float = None,
                       val_bootstraps : int = 0,
                       warmup_steps : int = 0,
                       batch_size : int = 800,
                       num_classes: Optional[int] = None,
                       compute_metrics: bool = True,
                       optimizer: OptimizerCallable = torch.optim.Adam,
                       lr_scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
                       test_results_save_path: str = None,
                       input_shape : Optional[Tuple[int,...]] = None,
                       reset_weights_before_fit=False):
        
        super(SensingModel,self).__init__()

        self.val_results = []
        self.test_results = []


        self.train_dataset = None
        self.eval_dataset=None

        self.num_val_bootstraps = val_bootstraps
        
        self.num_classes = num_classes
        self.compute_metrics = compute_metrics
        self.metric_class = metric_class

        assert learning_rate is None, "model.learning_rate is deprecated. Use optimizer.lr instead."

        if self.compute_metrics:
            self.init_metrics()
        else:
            self.train_metrics = torchmetrics.MetricCollection([])
            self.val_metrics = torchmetrics.MetricCollection([])
            self.test_metrics = torchmetrics.MetricCollection([])
            

        self.test_results_save_path = test_results_save_path
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        
        self.optimizer=optimizer
        self.scheduler=lr_scheduler

        self.wandb_id = None
        self.name = None

        self.reset_weights_before_fit = reset_weights_before_fit

        # self.save_hyperparameters()

    @property
    def is_mcq_task(self):
        return isinstance(self.trainer.datamodule, MultimodalMCQTask)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--warmup_steps", type=int, default=0,
                            help="Steps until the learning rate reaches its maximum values")
        parser.add_argument("--batch_size", type=int, default=800,
                            help="Training batch size")      
        parser.add_argument("--num_val_bootstraps", type=int, default=100,
                            help="Number of bootstraps to use for validation metrics. Set to 0 to disable bootstrapping.") 
        return parser
    
    def init_metrics(self):
        self.train_metrics = self.metric_class(bootstrap_samples=0, prefix="train/", num_classes=self.num_classes)
        self.val_metrics = self.metric_class(bootstrap_samples=self.num_val_bootstraps, prefix="val/", num_classes=self.num_classes)
        self.test_metrics = self.metric_class(bootstrap_samples=self.num_val_bootstraps, prefix="test/", num_classes=self.num_classes)
        
    def on_fit_start(self) -> None:
        if self.reset_weights_before_fit:        
            self.reset_weights()
            logger.info("Resetting model weights before fit.")
            
        return super().on_fit_start()
    
    def on_train_start(self) -> None:
        self.train_metrics.apply(lambda x: x.to(self.device))
        self.val_metrics.apply(lambda x: x.to(self.device))
        
        ## Save wandb id for later use:
        if isinstance(self.logger, WandbLogger):
            self.wandb_id = self.logger.experiment.id

        return super().on_train_start()
    
    def handle_options(self, batch):
        if "options" in batch and isinstance(batch["options"][0][0], str):
            options = batch["options"]
            results = []
            for b in options:
                new_options = [x+ self.tokenizer.eos_token for x in b]
                results.append(self.tokenizer(new_options,return_tensors="pt", padding=True)["input_ids"])
            
            batch["options"] = results

    def common_step(self, batch, loss_key):
        loss, preds = self.forward(**batch)
        if loss is None:
            return None
 
        self.handle_options(batch)

        y = batch["label"]
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu()

        # Log the loss
        self.log(f"{loss_key}/loss", loss, on_step=True, prog_bar=True, sync_dist=False, batch_size = len(y))

        if preds is not None:
            # Update metrics
            preds = preds.detach()
            getattr(self, f"{loss_key}_metrics").update(preds, y, **batch)

        if loss_key in ["test","val"]:
            if isinstance(self, MultimodalModel):
                if self.is_mcq_task:
                    options =  batch["options"]
                    label_index = batch["label_index"]
                    log_probs = torch.stack([compute_log_probs(l, o) for l,o in zip(preds,options)])

                    # Unfold the batch back into records
                    preds = preds.flatten()
                    y = batch["label"]
                    
                    for i in range(len(log_probs)):
                        record = {
                            "log_prob": log_probs[i].cpu().numpy(),
                            "label_index": label_index[i],
                            "label": batch["label"][i],
                        }
                        other_keys = ["ts_qid","uuid","category","question","options"]
                        for key in other_keys:
                            if key in batch:
                                record.update({key:batch.get(key)[i]})
        
                        getattr(self, f"{loss_key}_results").append(record)

                else:
                    results = self._generate_after_step(**batch)
                    getattr(self, f"{loss_key}_results").extend(results)
                            
        return {"loss": loss, "labels": y}

    def training_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        return self.common_step(batch, "train")

    def test_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        return self.common_step(batch, "test")

    def validation_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        return self.common_step(batch, "val")
        

    def on_train_epoch_end(self):
    
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True)
        
        # Clean up for next epoch:
        self.train_metrics.reset()
        super().on_train_epoch_end()
    
    def on_train_epoch_start(self):
        self.train_metrics.to(self.device)
    
    def on_test_epoch_end(self):
        # We get a DummyExperiment outside the main process (i.e. global_rank > 0)
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

        # Clean up
        self.test_metrics.reset()

        if len(self.test_results) > 0: 
            if isinstance(self.logger, WandbLogger):
                upload_pandas_df_to_wandb(logger=self.logger,
                                  table_name="test_results",
                                  df=pd.DataFrame(self.test_results))
            if self.test_results_save_path is not None:
                pd.DataFrame(self.test_results).to_csv(self.test_results_save_path,index=False)
                
        self.test_results = []

        super().on_test_epoch_end()
    

    def predict_step(self, batch: Any) -> Any:

        with torch.no_grad():
            loss,logits = self.forward(**batch)
            probs = torch.nn.functional.softmax(logits,dim=1)[:,-1]
        
        y = batch["label"]       
        return {"loss":loss, "preds": logits, "labels":y,
                "participant_id":batch["participant_id"],
                "end_date":batch["end_date_str"] }
    

    def on_validation_epoch_end(self):

        metrics = self.val_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

        # Clean up
        self.val_metrics.reset()

        if len(self.val_results) > 0 and isinstance(self.logger, WandbLogger):
            upload_pandas_df_to_wandb(logger=self.logger,
                                    table_name="val_results",
                                    df=pd.DataFrame(self.val_results))
        
        self.val_results = []
    
    def configure_optimizers(self):
        #TODO: Add support for other optimizers and lr schedules?
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        scheduler = self.scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def optimizer_step( self,
                        epoch: int,
                        batch_idx: int,
                        optimizer,
                        optimizer_closure):
        
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / (self.warmup_steps + 1))
            for pg in optimizer.param_groups:
                default_lr = optimizer.defaults["lr"]
                pg['lr'] = lr_scale * default_lr    

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
    
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    logger.info(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                logger.info(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def upload_predictions_to_wandb(self):
        if hasattr(self,"predictions_df"):
            upload_pandas_df_to_wandb(run_id=self.wandb_id,
                                  table_name="test_predictions",
                                  df=self.predictions_df)


    def freeze_encoder(self):
        if hasattr(self,"encoder"):
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def reset_weights(self):
    
        @torch.no_grad()
        def weight_reset(m: nn.Module):
            # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        self.apply(fn=weight_reset)
    
    def _generate_after_step(self, **batch) -> None:
        if not hasattr(self,"generate"):
            return []
        
        results = self.generate(**batch)
    
        return results
        

class ModelTypeMixin():
    def __init__(self):
        self.is_regressor = False                            
        self.is_classifier = False
        self.is_autoencoder = False
        self.is_double_encoding = False

        self.metric_class = None

class ClassificationModel(SensingModel):
    '''
    Represents classification models 
    '''
    def __init__(self,**kwargs) -> None:
        if "metric_class" not in kwargs:
            kwargs["metric_class"] = TorchMetricClassification

        SensingModel.__init__(self, **kwargs)
        self.is_classifier = True

class RegressionModel(SensingModel,ModelTypeMixin):
    def __init__(self,**kwargs) -> None:
        SensingModel.__init__(self, metric_class = TorchMetricRegression, **kwargs)
        self.is_regressor = True


class MultimodalModel(SensingModel,ModelTypeMixin):
    def __init__(self,**kwargs) -> None:
        SensingModel.__init__(self, metric_class = TorchMetricMultimodal, **kwargs)
        self.is_regressor = True
        self.is_classifier = True

        # Consider moving this into a superclass at some point
    def _prepare_ts(self, ts: List[np.array]) -> torch.Tensor:
        ts = [torch.tensor(x).to(self.device).type(self.dtype).unsqueeze(0) for x in ts]
        return ts

