CFG=configs/models/whisper.yaml
CKPT=models/whisper_u67qdw6y.ckpt
zsh slurm-utils/eval/launch_test.sh $CFG $CKPT /gscratch/bdata/mikeam/TSandLanguage/configs/tasks/llms_and_ts/qa_mcq.yaml  reports/results/whisper_qa_mcq.csv
zsh slurm-utils/eval/launch_test.sh $CFG $CKPT /gscratch/bdata/mikeam/TSandLanguage/configs/tasks/llms_and_ts/ts2desc_mcq.yaml  reports/results/whisper_desc_mcq.csv