# Aim 1: Training setup + smoke test (notes)

Follows the Aim 1 checklist order in *Project Focus Doc*.
Labels: **[NASA]** = documented in the repo (README, script or config); **[LOCAL]** = my fix/workaround; **[SMOKE]** = smoke-test-only setting.
Nothing committed. See every change with `git status` / `git diff`.

---

## Where training is documented (read this first)
- There is **no training README**. `src_cv/README.md` covers only dataset creation (steps 1-3) and inference data.
  - Its "Code" section points to `src_cv/train/`: *"scripts related to running a training/evaluation experiment on a CV dataset."*
- The real instructions are the scripts themselves, read top to bottom:
  1. `src_cv/train/run_cv_local.sh`: entry point ("Run CV experiment locally"). Loops over CV iterations.
  2. `src_cv/train/run_cv_iter_modular.sh`: one CV iteration. Its header comment lists the 9 arguments.
     For each model: `setup_cv_iter.py` → `src/train/train_model.py`.
     Then: `models/create_ensemble_avg_model.py` → `src/evaluate/evaluate_model.py` → `src/predict/predict_model.py`.
  3. `src_cv/train/setup_cv_iter.py`: merges `config_cv_train.yaml` + that iteration's shard paths (from `cv_folds`) + the model config (`model_config_fp`) into one `config_cv.yaml` per model.
  4. `src_cv/train/config_cv_train.yaml`: experiment settings (epochs, batch size, early stopping, label map).

---

## [ ] Configure `src_cv/train/config_cv_train.yaml` (paths, model = `exominer_new.yaml`)
- **`paths.cv_folds`**: authors' cluster path → `/mnt/c/Users/fhard/GitHub/ExoMiner/exominer_data/eval_normalized/cv_iterations.yaml` **[LOCAL]**
  - Why: this is Part C's output (README step 3). `setup_cv_iter.py` → `add_tfrec_dataset_fps_to_config_file()` reads it to get round *k*'s train/val/test shard lists.
- **`paths.model_config_fp`**: authors' path → `/mnt/c/Users/fhard/GitHub/ExoMiner/models/exominer_new.yaml` **[LOCAL]**
  - Why: the model architecture and hyperparameters (lr 4.176e-05, PReLU). Kept the same file the authors used (`exominer_new.yaml`, not `exominer_plusplus.yaml`, which is commented out in the config).
- **Not changed (decide later):**
  - `n_epochs: 300` with early stopping (patience 20 on `val_auc_pr`).
  - `batch_size: 32`: the paper (appendix, p. 20) says 256. Check GPU memory in the smoke test first.
- **New file `src_cv/train/config_cv_train_smoke.yaml`** **[SMOKE]**: identical copy except `n_epochs: 2`. The real config stays untouched for full runs.

## [ ] Smoke test: 1 fold, 1 model, few epochs
- **`src_cv/train/run_cv_local.sh` edits:**
  - `PYTHONPATH`: authors' Mac path → repo path **[LOCAL]**.
  - `CONFIG_FP`, `OUTPUT_DIR`, `N_CV_ITERS`, `N_MODELS_PER_CV_ITER` can now be overridden by env vars **[LOCAL]**.
    - Defaults: the real config, `exominer_data/experiments/cv_run`, 1 iteration, 1 model.
    - Authors' default was 2 models; the paper uses ensembles, so revisit for full runs.
  - **Bug fix** **[LOCAL]**: it called `run_cv_iter_modular.sh` with 7 args, but that script's header defines 8 required (`$1` parallel index, `$2` job-array index, **`$3` N_JOBS**, `$4` config, `$5` output dir, `$6` GPUs/node, `$7` CV iters, `$8` models/iter).
    - With `$3` missing, every argument after the second shifted by one: the config path became N_JOBS, etc.
    - Inserted `1` as N_JOBS (`CV_ITER = $1 + $2*$3` = iteration index).
    - Also call it with `bash` explicitly.
- **New launcher `src_cv/train/run_smoke_local.sh`** **[LOCAL][SMOKE]**: activates `exoplnt_dl`, sets the smoke env vars (smoke config, output `exominer_data/experiments/smoke_cv_iter0`, 1 iter, 1 model), prints the GPU, runs `run_cv_local.sh`, logs stdout.
  - Run (PowerShell, keep window open):
    `wsl -d Ubuntu -- bash /mnt/c/Users/fhard/GitHub/ExoMiner/src_cv/train/run_smoke_local.sh`
- **Where results land** (`exominer_data/` is gitignored, so the outputs won't clutter git):
  - `exominer_data/experiments/smoke_cv_iter0/cv_iter_0/cv_iter_0_run.log`: step-by-step progress.
  - `.../cv_iter_0/models/model0/train_model_0.log`: training output (loss and metrics per epoch).
  - `.../cv_iter_0/models/model0/config_cv.yaml`: the merged config actually used.
  - `.../cv_iter_0/ensemble_model/`: ensemble, evaluation and prediction outputs (a CSV ranking, since `generate_csv_pred: true`).
- **What "pass" looks like:**
  - GPU used.
  - Loss decreases over the 2 epochs.
  - `model.keras` saved.
  - Evaluation and prediction logs have no traceback.
  - Time per epoch noted, to plan the full 5-fold run.

## Environment fixes found by the smoke test (belongs under checklist item "Create conda env in WSL + confirm TensorFlow sees the GPU")
All four were added **once** to the env's activate script `$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`. They apply automatically every time `exoplnt_dl` activates; nothing to run beforehand. **[LOCAL]**
- Not in NASA's docs. The repo targets Linux clusters, not WSL plus an RTX 50-series card. Each error only appeared at a later stage of training, which is why the earlier GPU test passed.
1. **`ImportError: You must install pydot ... graphviz`** at `src/train/train_model.py:120` (`plot_model`, enabled by `plot_model: true` in the config).
   - Fix: `pip install pydot` (in env) + `sudo apt install -y graphviz`.
   - Bonus: produces `models/model0/model.png` (architecture figure for the paper).
2. **`Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file`**: the first convolution.
   - Cause: on WSL the NVIDIA driver lib lives in `/usr/lib/wsl/lib`, which wasn't on `LD_LIBRARY_PATH`.
   - Fix: `echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`
3. **`libdevice not found at ./libdevice.10.bc`**: the first optimizer step. Keras Adam in TF 2.13 compiles its update with XLA, which needs CUDA's `libdevice` (part of the compiler package).
   - Fix (TensorFlow's own Linux install recipe for TF 2.12/2.13):
     - `conda install -y -c "nvidia/label/cuda-11.8.0" cuda-nvcc` (puts `libdevice` in `$CONDA_PREFIX/nvvm/libdevice/`)
     - `echo 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`
4. Earlier (env creation): CUDA 11.8 + cuDNN 8.6 path (`CUDNN_PATH`, `LD_LIBRARY_PATH`), `CUDA_CACHE_MAXSIZE`, `TF_FORCE_GPU_ALLOW_GROWTH`.
   - The RTX 5070 (compute 12.0) runs via PTX JIT; the first run compiles kernels, then the compile is cached.

## Smoke test log
- Run 1 (13:29): failed at `plot_model`, because pydot/graphviz were missing → fix 1.
- Run 2 (13:33): model built, Epoch 1 started, failed on cuDNN → fix 2.
- Run 3 (13:36): passed the forward pass, failed at the Adam step → fix 3.
- Run 4 (13:41): **training works.**
  - 2 epochs at ~7 min each (1534 steps/epoch at batch 32, ~429 s and ~418 s).
  - Train AUC-PR 0.706 → 0.825; val AUC-PR 0.716 → 0.755 (2 epochs only; the paper reports 0.958 after full training).
  - Loss decreasing; model, ensemble and eval steps running.
  - **Warning seen (2x):** `Detected cudnn out-of-bounds write in convolution buffer! ... GPU state may already be corrupted`.
    - TF's autotuner caught a buggy cuDNN 8.6 algorithm on the RTX 5070 (Blackwell, newer than cuDNN 8.6 supports) and skips it afterwards.
    - Results looked sane, but this needs a decision before full runs (see options in chat).
  - **Timing concern:** 7 min/epoch × up to 300 epochs (early stopping, patience 20) could mean several hours per model.
    - That's × 15 trainings (3 configs × 5 folds).
    - Batch 256 (the paper's value) would cut steps/epoch 8x.

## Other changes in `git status` (from earlier steps)
- `src_cv/preprocessing/config_preprocess_cv_folds_tfrecord_dataset.yaml`: `cv_folds_fp` set to my path (README step 3 Part B) **[LOCAL]**.
- `.gitignore`: added `exominer_data/` **[LOCAL]**.
- `nohup.out`: leftover from the first Part B attempt. Safe to delete.

## Changes after the smoke test (decided with Fate)
- **`TF_CUDNN_USE_AUTOTUNE=0`** added to `src_cv/train/run_cv_local.sh` (applies to every run launched through it) **[LOCAL]**.
  - Why: the cuDNN out-of-bounds warning in run 4. cuDNN 8.6 predates Blackwell; this stops TF from auto-picking the buggy convolution algorithm.
- **`training.batch_size: 32 → 256`** in `config_cv_train.yaml` and the smoke copy **[NASA paper value]**.
  - Source: ExoMiner++ 2.0 appendix (p. 20): "Batch size was set to 256 to provide stable statistics for batch normalization."
  - The repo's config had `32  # 256  # 32`, i.e. the authors toggled between them.
  - The learning rate already matches the paper: `exominer_new.yaml` line 198 `lr: 4.176e-05` (paper: 4.18e-5, raised "to compensate batch size increase").
  - Left at 32: `evaluation.batch_size`, `inference.batch_size`. Those don't affect training results.
  - Must verify: does batch 256 fit in the RTX 5070's 12 GB? Smoke run 5 will show.
- **`training.n_epochs: 300 → 100`** in `config_cv_train.yaml` (not the smoke copy) **[LOCAL, compute budget]**.
  - Early stopping unchanged: `patience: 20` on `val_auc_pr`, `restore_best_weights: true`. Runs can still end earlier.
  - Paper justification: one consumer GPU (RTX 5070, 12 GB); 142 GB per normalized fold; Hamilton storage/memory insufficient. All 3 configurations (baseline, transformer, SSM) use the identical budget, so the Aim 2 comparison stays fair.
- **Run plan: one fold at a time.** Normalize fold k → train baseline + transformer + SSM on it → save results (~20 MB per training) → delete `cv_iter_k/norm_data` → next fold.
  - Peak disk: raw shards (73 GB) + one normalized fold (142 GB).

## Speedup: normalized data moved into WSL's own filesystem **[LOCAL]**
- Why: reads from `/mnt/c` (Windows drive via WSL's translation layer) were the bottleneck.
  - Smoke test: ~7 min/epoch; a train-set evaluation pass alone took 712 s.
- `cv_iter_0/` moved with `rsync -a --remove-source-files` (deletes each file after it's copied, so it never needs double the disk) to `/home/fhard/exominer_data/eval_normalized/cv_iter_0/`.
- Part C re-run with the new folder → `/home/fhard/exominer_data/eval_normalized/cv_iterations.yaml`, since that YAML stores absolute paths.
- `cv_folds` in both training configs now points to it.
- Note: the WSL disk file (ext4.vhdx) grows but doesn't shrink on its own. Space freed by deleting a fold inside WSL gets reused by the next fold.

## How a CV round picks its folds (for running one fold at a time)
- You don't choose folds by hand. You choose the **CV round number k (0-4)**; everything else follows from it.
  - Normalization: `preprocessing_job.sh k ...` (the "rank").
  - Training: `CV_ITER_START=k N_CV_ITERS=k+1` env vars for `run_cv_local.sh` **[LOCAL]**. Added `CV_ITER_START`; the original loop always started at 0.
- Fold roles come from Part A, `create_cv_iterations_yaml_for_cv_dataset()` in `src_cv/preprocessing/create_cv_iterations_yaml_files.py`, lines 64-87, `choose_val_method='rotating_fold'`:
  - Round k: test = fold k.
  - Val = the k-th of the remaining folds (wrapping around).
  - Train = the other 3.

| Round k | Test | Val | Train |
|---|---|---|---|
| 0 | 0 | 1 | 2, 3, 4 |
| 1 | 1 | 2 | 0, 3, 4 |
| 2 | 2 | 3 | 0, 1, 4 |
| 3 | 3 | 4 | 0, 1, 2 |
| 4 | 4 | 0 | 1, 2, 3 |

- Every fold is the test set exactly once and the validation set exactly once.
- Each round needs its own normalization, because the statistics come from that round's training folds.
- **Resume behavior:** `run_cv_iter_modular.sh` skips training if `models/modelN/model.keras` already exists. An interrupted run can be resumed, but **use a new OUTPUT_DIR for a fresh run.** That's why the batch-32 smoke result was renamed to `smoke_cv_iter0_bs32`.

## Switch to one config + real-run launcher (Fate's call: test with the same settings we train with)
- **Dropped the smoke config.** `config_cv_train_smoke.yaml` is no longer used; delete it.
  - It was my addition, a 2-epoch copy. A separate test config risked testing settings different from the real runs.
- **`training.batch_size: 128`** in `config_cv_train.yaml` **[LOCAL]**.
  - Batch 256 (the paper's value) crashed at epoch 1: GPU out-of-memory messages, then the WSL VM died.
  - Paper line: "batch size 128 (paper: 256) due to the 12 GB consumer GPU"; learning rate kept at the paper's 4.18e-5.
- **Launcher renamed `run_smoke_local.sh` → `run_round_local.sh <ROUND> <SETUP>`** **[LOCAL]**.
  - Uses the real config; output goes to `exominer_data/experiments/<SETUP>/`.
  - The first run doubles as the test: watch epoch 1 (memory, time). If it's healthy, let it continue as the real round-0 baseline (≤100 epochs, early stopping).
  - Before each run: `wsl --shutdown` (resets WSL memory), and keep the window open.

## Training crashes = Windows memory limit (confirmed) **[LOCAL]**
- Crashes at 15:25, 15:32, 15:35 and 15:38 each match a Windows "Resource-Exhaustion-Detector" low-virtual-memory event.
  - `vmmemWSL` held 21-26 GB; another `vmmem` (non-WSL VM) held 4.3 GB.
- Cause: Windows' hard limit (commit limit) is RAM 32 GB + pagefile ≈ 44 GB, shared by Windows, apps, the other VM and WSL.
  - Linux in WSL fills all its allowed memory with *file cache* while reading the shards. Windows counts that as used.
  - With a 20 GB WSL cap, the total crossed ~44 GB and Windows killed the VM.
  - Moving the data into WSL's own disk made this worse: the cache now lives inside the VM, not on the Windows side.
- Fix: `.wslconfig` → `memory=14GB`, `swap=96GB`, `[experimental] autoMemoryReclaim=dropcache`.
  - Training itself needs only a few GB; the rest was disposable cache.
  - Anything over 14 GB goes to WSL's swap file on disk, which doesn't count against Windows' limit.
- Optional, bigger fix: raise the Windows pagefile (raises the ~44 GB limit itself; needs a reboot).
- **Baseline round 0 is running** (started ~15:44 CST): batch 128, the real config, data in WSL ext4, WSL capped at 14 GB.
  - Epoch 1: **122 s** (384 steps), vs ~7 min/epoch for the batch-32 smoke test on `/mnt/c`.
  - Projection: 30-100 epochs before early stopping ≈ 1-3.5 h per training, plus eval/predict. 15 trainings ≈ 15-50 h total.
- **Slowdown at epochs 10-12** (479 / 657 / 765 s vs ~95-100 s), while VS Code, pip and other apps were open.
  - Diagnosis:
    - GPU at 3% utilization (waiting on data).
    - Inside WSL memory was fine (swap 8 MB).
    - On Windows: committed memory 45/53.7 GB on 32 GB of RAM, disk C: at 100%, "System" (paging) and vmmemWSL on top.
  - Cause: Windows paging to the same drive the shards are read from.
  - Fix: close all apps. **Rule: keep the PC otherwise idle while training** (WSL 14 GB + Windows ~8 GB + apps must stay under 32 GB).
  - Epoch 13 (started 16:30:52 CST, after the apps were closed) is the recovery test.
  - Recovered: epoch 15 = 133 s after the apps were closed (epoch 13 = 400 s while Windows paged memory back in). Restart not needed.
  - 17:44 CST: epoch 33; best val AUC-PR 0.8594 at epoch 22 (early stopping would end at epoch 42 if no improvement). Epochs mostly ~94 s, but intermittent slow ones (28: 306 s, 32: 330 s; 33 still running 5+ min). Likely PC activity → Windows paging again.
- **Round pipeline launcher** `src_cv/train/run_round_pipeline_local.sh <k> <setup>` **[LOCAL]**: waits for any running training, checks C: has ≥160 GB free, normalizes round k into `/home/fhard/exominer_data/eval_normalized` (skips if already done), then trains via `run_round_local.sh`.
- **Rerun after reboot (round 0):** epochs 1-10 ~95 s, then slow again (epochs 11-13: 748/520/347 s) with the PC idle.
  - Diagnosis (19:54 CST):
    - Windows side light (vmmemWSL 12 GB, nothing else big).
    - Inside WSL: no memory pressure, swap 76 KB.
    - **I/O pressure ~80% "full"**: reads collapsed to ~30-40 MB/s, from ~900 MB/s when fast (85 GB of train shards per 95 s epoch).
  - Drive: Patriot P300 2 TB, a budget DRAM-less NVMe, nearly full.
    - Likely cause: sustained-read thermal throttling after ~15 min. That matches the slowdown at epoch ~10 in both runs.
  - Biggest lever: read fewer bytes per epoch.
    - The model uses 56 features (~270 KB/example, mostly the 4 image stacks).
    - Each normalized example is ~1.46 MB (raw + normalized copies of many features).
    - "Slim" shards with only the model features + label/ID fields would cut I/O ~5x and disk per round from 142 GB to ~25-30 GB (estimate, unverified).

## Round 0 baseline: RESULT (finished 23:34 CST, 2026-11-16)
- Early stopping at epoch 55; best weights from **epoch 35**. Total ~4.9 h including eval/predict (slowed by disk throttling).
- Ensemble (1 model) metrics, from `experiments/baseline/cv_iter_0/ensemble_model/loss_and_performance_metrics.csv`:

| Split | PR-AUC | ROC-AUC | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| Train | 0.917 | 0.980 | 0.986 | 0.499 | 0.970 |
| Val | 0.803 | 0.941 | 0.936 | 0.389 | 0.964 |
| **Test** | **0.809** | 0.944 | 0.957 | 0.444 | 0.974 |

- **Well below the paper's 0.958.** Recall at the 0.5 threshold is low (high precision, misses planets). Needs investigation before rounds 1-4. Candidates:
  - 1 model vs the paper's ensemble; batch 128 vs 256.
  - Whether the paper's 0.958 is on the same split/population.
  - Run-to-run variance: the earlier paged run reached val 0.861.
- **Round 1 normalization FAILED** at shard 20/100 of the stats pass: `ResourceExhaustedError ... Cannot allocate memory`.
  - Cause: the WSL cap is now 14 GB. The stats pass holds all values in RAM (~68 GB total at 100 shards); round 0 only succeeded at 20 GB + swap.
  - Fix options: normalize with WSL at 20 GB (pagefile now larger), or cap `max_n_examples_shard`. A cap applied only to rounds 1-4 would make round 0 inconsistent.
- `.wslconfig` memory raised 14 GB → **20 GB** (swap 96 GB, dropcache kept) for the round 1+ normalization; `wsl --shutdown` applied. Pagefile is now larger, so the Windows commit limit should hold. Keep the PC idle during runs.

## Decision (Fate, 2026-11-17): 3 rounds first
- Run **rounds 0, 1, 2** for all 3 architectures (baseline, transformer, SSM) = 9 trainings. Same rounds for every architecture, so comparisons are paired.
- Rounds 3-4 only if there's time over Thanksgiving break.
- Disk: C: had 466 GB free with round 0's normalized data on disk.
  - Rounds 1 + 2 add ~284 GB, so **keep all three rounds normalized** (no deleting) and the transformer/SSM runs reuse them with no re-normalization.
- Paper wording: "3 of 5 CV folds due to compute; mean ± SD over 3 folds."
- **Round 1 normalization (2026-11-17):** failed twice at shard 64-65/100 of the stats pass with `Cannot allocate memory`.
  - First failure had autoMemoryReclaim on; the second had the round-0 config (20 GB + 96 GB swap).
  - Fate chose to keep full normalization (no cap) to match the paper.
  - 3rd attempt (07:46 CST, same config) ran with a memory monitor (`exominer_data/memwatch_round1.log`):
    - RAM filled to 20 GB, then swap engaged (peaked ~49 GB).
    - **Stats pass completed 100/100.**
  - Root cause of the earlier failures still unclear (swap works; no config change since the 2nd failure). If it recurs, check `dmesg_round1.log` and the memwatch log.
- **Batch size set back to 256** (paper value) in `config_cv_train.yaml` for the Expanse runs (V100 32 GB). Local rounds 0-1 used 128.
