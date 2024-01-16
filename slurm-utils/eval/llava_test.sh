MDL_NAME="llava"
CFG=configs/models/llava_llama_matplotlib.yaml

CMD1="python src/models/cli.py test --config=$CFG --data=configs/tasks/llms_and_ts/qa_mcq.yaml  --model.test_results_save_path=reports/results/{$MDL_NAME}_qa_mcq.csv"
python slurm-utils/launch_on_slurm.py --dir .  -n 1 -m 40G -p "gpu-a100" -a bdata --num-gpus 1 --num-cpus 4  --command "$CMD1" --exp-name whisper-debug --time "12:00:00"

CMD2="python src/models/cli.py test --config=$CFG --data=configs/tasks/llms_and_ts/ts2desc_mcq.yaml  --model.test_results_save_path=reports/results/{$MDL_NAME}_desc_mcq.csv"
python slurm-utils/launch_on_slurm.py --dir .  -n 1 -m 40G -p "gpu-a100" -a bdata --num-gpus 1 --num-cpus 4  --command "$CMD2" --exp-name whisper-debug --time "12:00:00"