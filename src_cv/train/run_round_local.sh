#!/bin/bash
# LOCAL launcher (not part of NASA's repo): train one CV round for one model setup with the real config
# (src_cv/train/config_cv_train.yaml), via NASA's run_cv_local.sh.
# Usage (cmd, keep the window open; run `wsl --shutdown` first to reset WSL memory):
#   wsl -d Ubuntu -- bash /mnt/c/Users/fhard/GitHub/ExoMiner/src_cv/train/run_round_local.sh <ROUND k> <SETUP name>
#   e.g. ... run_round_local.sh 0 baseline
ROUND=${1:-0}
SETUP=${2:-baseline}
source /home/fhard/miniforge3/etc/profile.d/conda.sh
conda activate exoplnt_dl || { echo "conda activate failed"; exit 1; }
REPO=/mnt/c/Users/fhard/GitHub/ExoMiner
cd "$REPO" || exit 1
export CONFIG_FP=$REPO/src_cv/train/config_cv_train.yaml
export OUTPUT_DIR=$REPO/exominer_data/experiments/$SETUP
export CV_ITER_START=$ROUND
export N_CV_ITERS=$((ROUND + 1))
export N_MODELS_PER_CV_ITER=1
mkdir -p "$OUTPUT_DIR"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
echo "Round $ROUND, setup $SETUP, output $OUTPUT_DIR"; date
bash src_cv/train/run_cv_local.sh 2>&1 | tee -a "$OUTPUT_DIR/run_cv_local_stdout_round$ROUND.log"
echo "RUN_EXIT=${PIPESTATUS[0]}"; date
echo "Logs: $OUTPUT_DIR/cv_iter_$ROUND/cv_iter_${ROUND}_run.log and cv_iter_$ROUND/models/model0/train_model_0.log"
