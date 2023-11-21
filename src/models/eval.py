import os
from typing import Dict, Any, Optional, Callable, AnyStr
import numpy as np
from numpy.core.shape_base import vstack
import pandas as pd
from torchmetrics.classification.auroc import AUROC
from torchmetrics.metric import Metric
from torchmetrics.regression import explained_variance
import wandb
from wandb.plot.roc_curve import roc_curve
from wandb.data_types import Table

import torch


import torchmetrics
from torchmetrics import (BootStrapper, MetricCollection, Metric, CosineSimilarity,
                          ExplainedVariance, PrecisionRecallCurve, AveragePrecision)
                          

import pytorch_lightning as pl


# from sklearn.metrics import (accuracy_score,precision_recall_fscore_support, roc_auc_score,
#                             mean_absolute_error, det_curve, precision_recall_curve, auc)


from functools import partial
from src.utils import check_for_wandb_run

class Support(Metric):
    def __init__(self,  num_classes: int = 1,
                        compute_on_step: bool = True,
                        dist_sync_on_step: bool = False,
                        process_group: Optional[Any] = None,
                        dist_sync_fn: Callable = None) -> None:

        super().__init__(compute_on_step=compute_on_step,
                        dist_sync_on_step=dist_sync_on_step,
                        process_group=process_group,
                        dist_sync_fn=dist_sync_fn)

        self.num_classes = num_classes
        self.add_state("counts", default = torch.zeros(self.num_classes + 1),
                                dist_reduce_fx="sum")

    def update(self, _preds: torch.Tensor, target: torch.Tensor) -> None:
        values = torch.bincount(target, minlength=self.num_classes+1)
        self.counts += values

    def compute(self) -> Dict[AnyStr,torch.Tensor]:
        return {str(i):self.counts[i] for i in range(self.num_classes + 1)}


class TorchMetricRegression(MetricCollection):
    def __init__(self, bootstrap_samples=1000,
                 prefix=""):
        self.add_prefix = prefix
        metrics = {}

        self.bootstrap_samples = bootstrap_samples

        if bootstrap_samples:
            cosine_sim = BootStrapper(CosineSimilarity(),
                                    num_bootstraps=bootstrap_samples)
            explained_variance = BootStrapper(ExplainedVariance(),
                                  num_bootstraps=bootstrap_samples)
        else:    
            cosine_sim = CosineSimilarity()
            explained_variance = ExplainedVariance()

        metrics["cosine_sim"] = cosine_sim
        metrics["explained_variance"] = explained_variance
        

        self.best_metrics = {"cosine_sim":(max,0),
                             "explained_variance":(max,0)}

        super(TorchMetricRegression,self).__init__(metrics)

    
    def compute(self) -> Dict[str, Any]:
        results = super().compute()
        if self.bootstrap_cis:

            cosine_sim = results["cosine_sim"]["mean"] 
            cosine_sim_std = results["cosine_sim"]["std"] 
            results["cosine_sim_ci_high"] = cosine_sim + 2*cosine_sim_std
            results["cosine_sim_ci_low"] = cosine_sim - 2*cosine_sim_std
            results["cosine_sim"] = results["cosine_sim"]["mean"]
        
            explained_variance = results["explained_variance"]["mean"] 
            explained_variance_std = results["explained_variance"]["std"] 
            results["explained_variance_ci_high"] = explained_variance + 2*explained_variance_std
            results["explained_variance_ci_low"] = explained_variance - 2*explained_variance_std
            results["explained_variance"] = results["explained_variance"]["mean"]
        
        for metric , (operator,old_value) in self.best_metrics.items():
            
            gt_max = (operator == max) and (results[metric] >= old_value)
            lt_min = (operator == min) and (results[metric] <= old_value)
            if gt_max or lt_min:
                self.best_metrics[metric] = (operator,results[metric])
                results[f"best_{metric}"] = results[metric]

        if self.add_prefix:
            return add_prefix(results,self.add_prefix)
        else:
            return results
    
    def update(self, preds: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> None:
        return super().update(preds)

class TorchMetricClassification(MetricCollection):
    def __init__(self, bootstrap_samples=200,
                 num_classes=2,
                 prefix=""):
        
        self.add_prefix = prefix
        self.bootstrap_samples = bootstrap_samples
        self.num_classes = num_classes
        self.task = "binary" if num_classes == 2 else "multiclass"

        metrics = {}
        
        roc_auc = torchmetrics.AUROC(task=self.task, num_classes=self.num_classes)  
        pr_auc = AveragePrecision(task=self.task, num_classes=self.num_classes)
        accuracy = torchmetrics.Accuracy(task=self.task, num_classes=self.num_classes)

        if bootstrap_samples:
            roc_auc = BootStrapper(roc_auc, 
                                   quantile=torch.tensor([0.975, 0.025], device="cuda"),
                                   num_bootstraps=bootstrap_samples)
            
            pr_auc = BootStrapper(pr_auc,
                                  quantile=torch.tensor([0.975, 0.025], device="cuda"),
                                  num_bootstraps=bootstrap_samples)
            
            accuracy = BootStrapper(accuracy,
                                    quantile=torch.tensor([0.975, 0.025], device="cuda"),
                                    num_bootstraps=bootstrap_samples)


        metrics["roc_auc"] = roc_auc
        metrics["pr_auc"] = pr_auc
        metrics["accuracy"] = accuracy
        # metrics["support"] = Support() #TODO: Fix this by flattening the dict output

        super(TorchMetricClassification,self).__init__(metrics)

        self.best_metrics = {"pr_auc":(max,0),
                             "roc_auc":(max,0),
                             "accuracy":(max,0)}

    def compute(self) -> Dict[str, Any]:
        results = {k: m.compute() for k, m in self.items(keep_base=True, copy_state=False)}
        results = {self._set_name(k): v for k, v in results.items()}

        if self.bootstrap_samples:


            results["roc_auc_ci_high"] = results["roc_auc"]["quantile"][0]
            results["roc_auc_ci_low"] = results["roc_auc"]["quantile"][1]
            results["roc_auc"] = results["roc_auc"]["mean"]
        

            results["pr_auc_ci_high"] = results["pr_auc"]["quantile"][0]
            results["pr_auc_ci_low"] = results["pr_auc"]["quantile"][1]
            results["pr_auc"] = results["pr_auc"]["mean"]
        
        
            results["accuracy_ci_high"] = results["accuracy"]["quantile"][0]
            results["accuracy_ci_low"] = results["accuracy"]["quantile"][1]
            results["accuracy"] = results["accuracy"]["mean"]

        for metric , (operator,old_value) in self.best_metrics.items():
            
            gt_max = (operator == max) and (results[metric] >= old_value)
            lt_min = (operator == min) and (results[metric] <= old_value)
            if gt_max or lt_min:
                self.best_metrics[metric] = (operator,results[metric])
                results[f"best_{metric}"] = results[metric]

        if self.add_prefix:
            return add_prefix(results,self.add_prefix)
        else:
            return results

    def update(self, preds: torch.Tensor, target: torch.Tensor, **kwargs) -> None:  # type: ignore
        if self.task == "binary":
            probs = torch.nn.functional.softmax(preds,dim=1)[:,-1]
        else:
            probs = torch.nn.functional.softmax(preds)
        
        return super().update(probs, target)

    def compute_wandb_plots(self, preds, labels):
        results =  {f"roc": wandb_roc_curve(preds,labels, task=self.task), 
                        "pr":  wandb_pr_curve(preds,labels, task=self.task)}
        return add_prefix(results,self.add_prefix)

class TorchMetricMultimodal(MetricCollection):
    def __init__(self, *args, **kwargs):
        super().__init__([])

        # Best proxy for knowing if we have an MCQ task
        if kwargs.get("num_classes") is not None:
            self.classification_metrics = TorchMetricClassification(*args, **kwargs)
            self.add_module("classification", self.classification_metrics)
            self.compute_classification = True
        else:
            self.compute_classification = False

    def compute(self) -> Dict[str, Any]:
        if self.compute_classification:
            return self.classification_metrics.compute()
        else:
            return {}
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor,
                **kwargs) -> None:
        
        if self.compute_classification:   
            options = kwargs["options"]
            label_index = torch.Tensor(kwargs["label_index"]).to(logits.device).int()

            log_probs = torch.stack([compute_log_probs(l, o) for l,o in zip(logits,options)])
            # guess = log_probs.argmax(dim=1)
            self.classification_metrics.update(log_probs, label_index)


def add_prefix(results,prefix):
    renamed = {}
    for k,v in results.items():
        renamed[prefix+k] = v
    return renamed


def wandb_pr_curve(preds,labels,thresholds=50, num_classes=1,task="binary"):

    pr_curve = PrecisionRecallCurve(task=task, thresholds=thresholds, num_classes=num_classes).to(preds.device)
    
    probs = torch.nn.functional.softmax(preds,dim=1)[:,-1]
    precision, recall, _thresholds = pr_curve(probs, labels)
    label_markers = ["Positive"] * len(precision)
    table = Table(columns= ["class","precision","recall"], data=list(zip(label_markers,precision,recall)))

    plot = wandb.plot_table(
        "wandb/precision-recall-curve/v0",
        table,
        {"x": "recall", "y": "precision", "class": "class"},
        {
            "title": "Precision Recall Curve",
            "x-axis-title": "Recall",
            "y-axis-title": "Precision",
        },

    )
    return plot
    

def wandb_roc_curve(preds,labels, task="binary",return_table=False,limit=999):
    probs = torch.nn.functional.softmax(preds,dim=1)[:,-1]
    fpr, tpr, _ = torchmetrics.functional.roc(probs, labels, task=task)
    if limit and limit < len(fpr):
        inds = np.random.choice(len(fpr), size=limit,replace=False)
        fpr = fpr[inds]
        tpr = tpr[inds]

    label_markers = ["Positive"] * len(fpr)
    table = Table(columns= ["class","fpr","tpr"], data=list(zip(label_markers,fpr,tpr)))
    if return_table:
        return table
    plot = wandb.plot_table(
        "wandb/area-under-curve/v0",
        table,
        {"x": "fpr", "y": "tpr", "class": "class"},
        {
            "title": "ROC",
            "x-axis-title": "False positive rate",
            "y-axis-title": "True positive rate",
        },

    )
    return plot


# Should I be ignoring the padding tokens?
def compute_log_probs(logits, options, left_pad=True):
    
    q, l = options.shape
    if left_pad:
        logits = logits[-l:]

    l_idx = torch.arange(l).unsqueeze(0).expand(q,-1)

    # Gather the logits associated with each option
    gathered_logits = logits[l_idx, options]

    # Take the sum across the length dimension to get the log probability of each sequence.
    scores = gathered_logits.sum(dim=-1)

    return scores

def get_target_indices(options, targets):
    # Expand dimensions for advanced indexing
    b, l = targets.shape
    b_idx = torch.arange(b).unsqueeze(1).expand(-1, l)
 
    # Find the indices where options match the targets
    match = (options == targets[b_idx, :, None, None]).nonzero(as_tuple=True)

    # Reshape to match the shape of `options` by filling with -1 where there is no match
    result = torch.full_like(options, -1)
    result[match[:-1]] = match[-1]

    return result