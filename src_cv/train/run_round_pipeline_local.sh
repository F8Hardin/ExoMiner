#!/bin/bash
# LOCAL: normalize CV round k (README step 3 Part B, rank k), then train it via run_round_local.sh.
# Waits for any running training to finish first. Usage (cmd, keep window open):
#   wsl -d Ubuntu -- bash /mnt/c/Users/fhard/GitHub/ExoMiner/src_cv/train/run_round_pipeline_local.sh 1 baseline
K=${1:?round k}; SETUP=${2:-baseline}
REPO=/mnt/c/Users/fhard/GitHub/ExoMiner
OUT=/home/fhard/exominer_data/eval_normalized
while pgrep -f "src/train/train_model.py|evaluate_model.py|predict_model.py" >/dev/null; do echo "$(date +%H:%M) waiting for current run to finish..."; sleep 300; done
FREE_GB=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9)
echo "C: free: ${FREE_GB} GB"
if [ "$FREE_GB" -lt 160 ]; then echo "ABORT: need ~160 GB free on C: for round $K (142 GB normalized). Delete an old round's norm_data first."; exit 1; fi
source /home/fhard/miniforge3/etc/profile.d/conda.sh; conda activate exoplnt_dl || exit 1
cd "$REPO" && export PYTHONPATH=$REPO
if [ "$(ls $OUT/cv_iter_$K/norm_data 2>/dev/null | wc -l)" -lt 169 ]; then
  echo "$(date +%H:%M) normalizing round $K..."
  bash src_cv/preprocessing/preprocessing_job.sh $K $OUT $REPO/src_cv/preprocessing/preprocess_cv_folds_trecord_dataset.py $REPO/src_cv/preprocessing/config_preprocess_cv_folds_tfrecord_dataset.yaml 5
  N=$(ls $OUT/cv_iter_$K/norm_data | wc -l); echo "$(date +%H:%M) normalized shards: $N"
  [ "$N" -lt 169 ] && { echo "ABORT: normalization incomplete; see $OUT/preprocessing_logs/preprocessing_cv_dataset_$K.log"; exit 1; }
else echo "round $K already normalized"; fi
bash src_cv/train/run_round_local.sh $K $SETUP
