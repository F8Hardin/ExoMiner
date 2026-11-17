# Run CV experiment locally.

# codebase root directory
export PYTHONPATH=/mnt/c/Users/fhard/GitHub/ExoMiner

# LOCAL: disable cuDNN autotuning; cuDNN 8.6 has a buggy conv algorithm on RTX 50-series (Blackwell) that autotune can pick
# ("Detected cudnn out-of-bounds write in convolution buffer"); default algorithms only
export TF_CUDNN_USE_AUTOTUNE=0

# config file path
CONFIG_FP=${CONFIG_FP:-$PYTHONPATH/src_cv/train/config_cv_train.yaml}  # override with env var, e.g. the smoke config
# job script for running the Python CV iteration script
RUN_SH_SCRIPT=$PYTHONPATH/src_cv/train/run_cv_iter_modular.sh

# output directory
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/c/Users/fhard/GitHub/ExoMiner/exominer_data/experiments/cv_run}  # override with env var
mkdir -p $OUTPUT_DIR

N_CV_ITERS=${N_CV_ITERS:-1}  # number of CV folds/iterations (only cv_iter_0 is normalized so far)
N_MODELS_PER_CV_ITER=${N_MODELS_PER_CV_ITER:-1}  # number of models to train per CV iteration

# number of GPUs to be used by this job array
N_GPUS_TOTAL=1

echo "Started CV experiment $OUTPUT_DIR."

# run CV iterations sequentially
CV_ITER_START=${CV_ITER_START:-0}  # LOCAL: run only rounds CV_ITER_START..N_CV_ITERS-1 (one fold at a time: CV_ITER_START=k N_CV_ITERS=k+1)
for ((CV_ITER=CV_ITER_START; CV_ITER<"$N_CV_ITERS"; CV_ITER++))
do
    # run CV iteration sh script
    echo "Started CV iteration $CV_ITER."
    # FIX: run_cv_iter_modular.sh expects 8 args ($3 = N_JOBS was missing); N_JOBS=1 since iterations run sequentially
    bash $RUN_SH_SCRIPT $CV_ITER 0 1 $CONFIG_FP $OUTPUT_DIR $N_GPUS_TOTAL $N_CV_ITERS $N_MODELS_PER_CV_ITER
    echo "Finished CV iteration $CV_ITER."
done

echo "Finished CV experiment $OUTPUT_DIR."
