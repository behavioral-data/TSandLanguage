CFG=$1
CKPT=$2
DATA=$3
PATH_NAME=$4

conda activate TSandLang
CMD="python src/models/cli.py test --config=$CFG --data=$DATA --ckpt_path=$CKPT --model.test_results_save_path=$PATH_NAME"
python slurm-utils/launch_on_slurm.py --dir .  -n 1 -m 40G -p "gpu-a100" -a bdata --num-gpus 1 --num-cpus 4  --command $CMD --exp-name whisper-debug --time "24:00:00"
