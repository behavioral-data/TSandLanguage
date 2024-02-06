import logging
import os
import warnings

warnings.filterwarnings("ignore")
import sys
from pprint import pprint
from typing import Any, Optional

import pandas as pd
import torch
from dotenv import dotenv_values
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer.states import TrainerFn
from torch import autograd

from src.utils import get_logger

torch.set_float32_matmul_precision('medium')

logger = get_logger(__name__)
CONFIG = dotenv_values(".env")


def add_general_args(parent_parser):
    """ Adds arguments that aren't part of pl.Trainer, but are useful
        (e.g.) --no_wandb """
    parent_parser.add_argument("--checkpoint_metric", type=str, default=None,
                               help="Metric to optimize for during training")
    parent_parser.add_argument("--checkpoint_mode", type=str, default="max",
                               help="Metric direction to optimize for during training")
    parent_parser.add_argument("--no_wandb", default=False, action="store_true",
                               help="Run without wandb logging")
    parent_parser.add_argument("--notes", type=str, default=None,
                               help="Notes to be sent to WandB")
    parent_parser.add_argument("--early_stopping_patience", type=int, default=None,
                               help="path to validation dataset")
    parent_parser.add_argument("--gradient_log_interval", default=0, type=int,
                               help = "Interval with which to log gradients to WandB. 0 -> Never")
    parent_parser.add_argument("--load_weights_path", default=None, type=str)
    parent_parser.add_argument("--freeze_encoder", action="store_true", default=False)
                           
    parent_parser.add_argument("--run_name", type=str, default=None,
                               help="run name to use for to WandB")
    parent_parser.add_argument("--pl_seed", type=int, default=2494,
                               help="Pytorch Lightning seed for current experiment")
    parent_parser.add_argument("--no_ckpt", action="store_true", default=False,
                               help="Don't save any model checkpoints")

    return parent_parser

class WandBSaveConfigCallback(SaveConfigCallback):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:

        if isinstance(trainer.logger, WandbLogger):
            # If we're at rank zero and using WandBLogger then we probably want to
            # log the config
            log_dir = trainer.logger.experiment.dir
            fs = get_filesystem(log_dir)

            config_path = os.path.join(log_dir, self.config_filename)
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=True, multifile=self.multifile
            )
        else:
            super().setup(trainer,pl_module,stage=stage)

class CLI(LightningCLI):
    # It's probably possible to use this CLI to train other types of models
    # using custom training loops

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.init_args.batch_size","data.init_args.batch_size",apply_on="parse")
        parser.link_arguments("model.init_args.hf_name_or_path", "data.init_args.hf_name_or_path", apply_on="parse")
        parser.link_arguments("model.init_args.model_base", "data.init_args.model_base", apply_on="parse")
        parser.link_arguments("data.data_shape", "model.init_args.input_shape", apply_on="instantiate")
        parser.link_arguments("data.num_classes", "model.init_args.num_classes", apply_on="instantiate")

        parser.add_optimizer_args(torch.optim.Adam)

        add_general_args(parser)

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        # It's probably possible to do all of this from a config.
        # We could set a default config that contains all of this, 
        # which vould be overridden by CLI args. For now,
        # prefer to have the API work like it did in the last project,
        # where we make an educated guess about what we're intended to
        # do based on the model and task that are passed.
        subcommand = self.config.subcommand
        extra_callbacks = []

        pl_seed = self.config[subcommand]["pl_seed"]
        seed_everything(pl_seed)

        
        checkpoint_metric = self.config[subcommand]["checkpoint_metric"]
        mode = self.config[subcommand]["checkpoint_mode"]
        run_name = self.config[subcommand]["run_name"]
        
        if self.config.subcommand == TrainerFn.FITTING:
            if self.datamodule.val_dataloader() is not None:
            
                if self.datamodule.is_classification:
                    if checkpoint_metric is None:            
                        checkpoint_metric = "val/roc_auc"
                        mode = "max"
                else:
                    if checkpoint_metric is None:            
                        checkpoint_metric = "val/loss"
                        mode = "min"
                
                if self.config["fit"]["early_stopping_patience"]:
                    early_stopping_callback = EarlyStopping(monitor=checkpoint_metric,
                                                            patience=self.config["fit"]["early_stopping_patience"],
                                                            mode=mode)
                    extra_callbacks.append(early_stopping_callback)
            else:
                if checkpoint_metric is None:            
                    checkpoint_metric = "train/loss"
                    mode = "min"
            
            if (not self.config[subcommand]["no_ckpt"]) and (not os.environ.get("NO_CKPT",None)):
                self.checkpoint_callback = ModelCheckpoint(
                                    filename='{epoch}',
                                    save_last=True,
                                    save_top_k=1,
                                    save_on_train_epoch_end = True,
                                    monitor=checkpoint_metric,
                                    every_n_epochs=1,
                                    mode=mode)
            
                extra_callbacks.append(self.checkpoint_callback)

        if not self.config[self.config.subcommand]["no_wandb"]:
            import wandb
            lr_monitor = LearningRateMonitor(logging_interval='step')
            extra_callbacks.append(lr_monitor)

            # kwargs["save_config_callback"] = WandBSaveConfigCallback

            logger_id = self.model.wandb_id if hasattr(self.model, "id") else None
            
            if os.environ.get("WANDB_DIR",None):
                save_dir = os.environ.get("WANDB_DIR")
            else:
                save_dir = "."

            data_logger = WandbLogger(project=CONFIG["WANDB_PROJECT"],
                                entity=CONFIG["WANDB_ENTITY"],
                                name=run_name,
                                config=dict(self.config.as_dict()[self.config.subcommand]),
                                notes=self.config[self.config.subcommand]["notes"],
                                log_model=False, #saves checkpoints to wandb as artifacts, might add overhead 
                                reinit=True,
                                save_dir=save_dir,
                                resume = 'allow',
                                allow_val_change=True,
                                id = self.model.wandb_id)   #id of run to resume from, None if model is not from checkpoint. Alternative: directly use id = model.logger.experiment.id, or try setting WANDB_RUN_ID env variable                
           
            if not callable(data_logger.experiment.summary):
                data_logger.experiment.summary["task"] = self.datamodule.get_name()
                data_logger.experiment.summary["model"] = self.model.name
                data_logger.experiment.config.update(self.model.hparams, allow_val_change=True)
                self.model.save_hyperparameters() 
                
                # Necessary to save config in the right location
                data_logger._save_dir = data_logger.experiment.dir
            
            self.model.wandb_id = data_logger.version 

        else:
            data_logger = None
        kwargs["logger"] = data_logger

        extra_callbacks = extra_callbacks + [self._get(self.config_init, c) for c in self._parser(self.subcommand).callback_keys]
        trainer_config = {**self._get(self.config_init, "trainer"), **kwargs}
        return self._instantiate_trainer(trainer_config, extra_callbacks)

    def before_fit(self):
        # Enables logging of gradients to WandB
        gradient_log_interval = self.config["fit"]["gradient_log_interval"]
        if isinstance(self.trainer.logger, WandbLogger) and gradient_log_interval:
            self.trainer.logger.watch(self.model, log="all", log_freq=gradient_log_interval)
        
        if self.config["fit"]["load_weights_path"]:
            state_dict = torch.load(self.config["fit"]["load_weights_path"])["state_dict"]
            self.model.load_state_dict(state_dict,strict=False)

        if self.config["fit"]["freeze_encoder"]:
            self.model.freeze_encoder()

    def after_fit(self):
        if self.trainer.is_global_zero:
            logger.info(f"Best model score: {self.checkpoint_callback.best_model_score}")
            logger.info(f"Best model path: {self.checkpoint_callback.best_model_path}")
        results = {}

        if self.trainer.state.fn == TrainerFn.FITTING:
            if (
                    self.trainer.checkpoint_callback
                    and self.trainer.checkpoint_callback.best_model_path
            ):
                ckpt_path = self.trainer.checkpoint_callback.best_model_path
                # Disable useless logging
                logging.getLogger("pytorch_lightning.utilities.distributed").setLevel(
                    logging.WARNING
                )
                logging.getLogger("pytorch_lightning.accelerators.gpu").setLevel(
                    logging.WARNING
                )

                self.trainer.callbacks = []

                
                test_dataloader = self.trainer.datamodule.test_dataloader()
                if test_dataloader:
                    fn_kwargs = {
                        "model": self.model,
                        "dataloaders": [test_dataloader],
                        "ckpt_path": ckpt_path,
                        "verbose": False,
                    }
                    results = self.trainer.test(**fn_kwargs)[0]
                else:
                    results = {}

                if hasattr(self.model, "wandb_id") and results:
                    self.model.upload_predictions_to_wandb()

        else:
            results = self.trainer.logged_metrics

        if results:
            pprint(results)

    def set_defaults(self):
        ...

if __name__ == "__main__":
    trainer_defaults = dict(
                        accelerator="cuda",
                        num_sanity_val_steps=0,
                        devices=-1,
                        profiler=None,
              )
    try:
        os.remove("lightning_config.yaml")
    except FileNotFoundError:
        pass
    cli = CLI(trainer_defaults=trainer_defaults,
            parser_kwargs={"parser_mode": "omegaconf"},
            save_config_callback=WandBSaveConfigCallback)

