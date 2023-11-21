Time Series and Language
==============================

# Install
## Making the Environment
1. Clone this repo
2. Build the environment using `make create_environment`.

**Note** If you're on the UW Slurm Cluster you will need to first load CUDA by running `module load cuda/12.2`

## Getting Data
TODO

# Running Jobs:

This project was designed to be run from the command line. Here's an example command:
```bash
python src/models/cli.py fit \
    --model="src.models.models.LLaVA" \
    --model.hf_name_or_path="liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview" \
    --model.model_base="meta-llama/Llama-2-7b-chat-hf" \
    --model.batch_size="1" \
    --data="configs/tasks/llms_and_ts/ts2desc_mcq.yaml" \
    --trainer.max_epochs="10" \
    --trainer.log_every_n_steps="100" \
    --trainer.precision="bf16" \
    --trainer.limit_train_batches="100" \
    --optimizer.lr="0.0001" \
    --early_stopping_patience="10" \
    --checkpoint_metric="val/loss" \
    --checkpoint_mode="min" \
    --no_wandb
```

There's a few things to notice about this command:
1. We're able to pass arguments directly to the model (e.g. `model.hf_name_or_path`). This is possible because we inherit the `LightningModule` class, which plays nicely with the `LightningCLI`.
2. We can also configure the Lightning `Trainer`.
3. The (optional) `--no_wandb` flag runs the experiment without logging to Weights and Biases.



