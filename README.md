# Bluesky Reposting Prediction

> End-to-end codebase for training and evaluating models to predict reposting behavior on Bluesky across topics. Primary metric: **binary F1 @ threshold 0.5**.

---

## TL;DR
- **Goal:** Predict whether a user will repost a given message on a decentralized platform (Bluesky).
- **Data:** 11 trending topics, raw API dumps (May 15–June 1, 2025) + a processed **1:5** (pos:neg) training set per topic.
- **Models:**
  
  In-distribution: 1. **MTX** 2. **UHM+UHU** 3. **UHU**
  
  Out-of-distribution: 1. **UHU** 2. **MTX**
- **Evaluation:** In-distribution (ID) per-topic and **leave-one-topic-out OOD**, reported as **F1 (positive class) at fixed 0.5**.

---

## Respository Structure
```text
.
├── LICENSE
├── README.md
├── MLP_UHU.py          # UHU (ID training)
├── message_only.py     # MTX (ID training)
├── user_related.py     # UHM+UHU (ID training)
├── message_ood.py      # MTX (leave-one-topic-out OOD)
└── UHU_ood.py          # UHU (leave-one-topic-out OOD)
```
## Data availability
- **1:5 datasets (11 CSVs):** available at https://drive.google.com/drive/folders/1sY2F4M13c8NcbNJlCMKjgR9bwvC5vBaO?usp=drive_link
- **Bluesky Raw Data:** available at https://drive.google.com/drive/folders/1-_zsgOmdrZvTbQPWY6DIROYuZ6GwZpH2?usp=drive_link

## Environment Setup
We use a Conda environment named `mscproject` with **Python 3.12**.
### Create & activate the Conda env
```bash
# 1) Create the environment (Python 3.12)
conda create -n mscproject python=3.12 -y

# 2) Activate it
conda activate mscproject

# Install PyTorch
# CUDA 12.8 (stable; recommended for NVIDIA GPUs)
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio

# Install the remaining packages
pip install transformers pandas numpy scikit-learn evaluate ftfy unidecode tqdm matplotlib
```

## In-Distribution

### 1) MTX
Runs `message_only.py` on one or more processed `*_1to5.csv` files.

**CLI**
```bash
python message_only.py \
  --csvs /path/to/<topic>_1to5.csv [/more/paths/*.csv] \
  --epochs 3 \
  --batch 32
```
Arguments:

- `--csvs` (required): one or more CSV paths (space-separated).
- `--epochs` (default: 3): fine-tuning epochs.
- `--batch` (default: 32): batch size.

Outputs:

For each CSV, results are saved to: `bert_<topic>_simple/`
(topic is taken from the filename prefix before the first _, e.g., `aew_1to5.csv` → `bert_aew_simple/`)

### 2) UHM+UHU
Runs `user_related.py` (phase-1 trains numeric MLP, phase-2 joint fine-tuning).

**CLI**
```bash
python user_related.py \
  --csvs /path/to/<topic>_1to5.csv [/more/paths/*.csv] \
  --joint_epochs 3 \
  --joint_batch 32
```
Arguments:

- `--csvs` (required): one or more CSV paths (space-separated).
- `--epochs` (default: 3): fine-tuning epochs.
- `--batch` (default: 32): batch size.

Outputs:

For each CSV, results are saved to: `bert_<topic>_U/`

### 3) UHU
Runs `MLP_UHU.py` in batch mode over all `*_1to5.csv` files in the current working directory.

**CLI**
```bash
# Put your *_1to5.csv files in the working directory, then:
python MLP_UHU.py
```

Behavior

- Globs `*_1to5.csv` in the current directory and runs them one by one.
- Default training setup inside the script (e.g., `EPOCHS=30, BATCH_SIZE=512, LR=3e-4`).
- Per-topic outputs go to: `mlp_<topic>_UHU/` (with a `plots/` subfolder and logs).

## Out-of-distribution

### 1) MTX OOD
Runs `UHU_ood.py` on one or more processed `*_1to5.csv` files.

**CLI**
```bash
python message_ood.py \
  --csvs *_1to5.csv \
  --ood_topic <topic> \
  --kfolds 1 \
  --epochs 3 \
  --batch 32 \
  --seed 42

# If you want to run all 11 topics:
python message_ood.py \
  --csvs *_1to5.csv \
  --ood_all \
  --kfolds 1 \
  --epochs 3 \
  --batch 32 \
  --seed 42
```
Arguments:

- `--csvs` (required): CSV paths (space-separated).
- `--epochs` (default: 3): fine-tuning epochs.
- `--batch` (default: 32): batch size.
- `--ood_topic` must match the filename prefix: <prefix>_1to5.csv → <prefix>.
- '--kfolds` reflects the number of cross-validation.

Outputs:

For each topic, results are saved to: `message_holdout_<topic>/`

### 2) UHU OOD
Runs `UHU_ood.py` on one or more processed `*_1to5.csv` files.

**CLI**
```bash
python UHU_ood.py \
  --csvs *_1to5.csv \
  --ood_topic <topic> \
  --kfolds 1 \
  --seed 42

# If you want to run all 11 topics:
python message_ood.py \
  --csvs *_1to5.csv \
  --ood_all \
  --kfolds 1 \
  --seed 42
```
Arguments:

- `--csvs` (required): CSV paths (space-separated).
- `--ood_topic` must match the filename prefix: <prefix>_1to5.csv → <prefix>.
- '--kfolds` reflects the number of cross-validation.

Outputs:

For each topic, results are saved to: `mlp_holdout_<topic>/`



