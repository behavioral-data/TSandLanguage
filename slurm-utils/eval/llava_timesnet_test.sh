MDL_NAME="llava_timesnet"
CFG=configs/models/llava_llama_timesnet.yaml
CKPT=models/llava_timesnet_desc_ofq3x05i.ckpt
zsh slurm-utils/eval/launch_test.sh $CFG $CKPT /gscratch/bdata/mikeam/TSandLanguage/configs/tasks/llms_and_ts/qa_mcq.yaml  reports/results/{$MDL_NAME}_qa_mcq.csv
zsh slurm-utils/eval/launch_test.sh $CFG $CKPT /gscratch/bdata/mikeam/TSandLanguage/configs/tasks/llms_and_ts/ts2desc_mcq.yaml  reports/results/{$MDL_NAME}_desc_mcq.csv