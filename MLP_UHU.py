#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Monte-Carlo training of an MLP on U/HU numeric features (Fixed Threshold)

• Automatically iterates over *_1to5.csv files
• Row-level 70/10/20 split:
    - first split 80/20 (train_val/test)
    - then take val_frac = 0.10 / 0.80 from train_val
• Remove (S,R) pairs that appear in test from train/val
• MinMax scaling
• 512-256-128-64 hidden layers / configurable Dropout
• Cosine LR scheduler
• Fixed 0.5 threshold: uses a standard 0.5 threshold for predictions.
• Reports standard Binary F1 score.
• Per-file outputs saved to mlp_{topic}_UHU/
"""

# ---------------- Imports ----------------
import os, re, json, random, math, glob
import numpy as np, pandas as pd, torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------- Configuration ------------
LABEL_COL   = "is_positive"
EPOCHS      = 30
BATCH_SIZE  = 512
LR          = 3e-4               # initial learning rate
N_RUNS      = 1
BASE_SEED   = 42
PATIENCE    = 8
DROPOUT_P   = 0.0
MLP_OUT_DIM = 64
CSV_GLOB    = "*_1to5.csv"        # pattern to find input CSV files
FIXED_THRESHOLD = 0.5             # Standard threshold for binary classification
device      = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Columns & Regex -------------------
PREFIX_RE    = re.compile(r"^(U\.|HU\.)")
TEXT_COLS    = [
    "U.S.displayName", "U.S.description",
    "U.R.displayName", "U.R.description",
]
EXCLUDE_COLS = ["M.uri", "U.S.did", "U.R.did", "M.record.createdAt"]

# -------- Utility functions --------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -------- Model --------------------------
class UHUMlp(nn.Module):
    """512-256-128-64 MLP with BatchNorm & Dropout."""
    def __init__(self, in_dim: int, p_drop: float = 0.1, out_dim: int = MLP_OUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128,  out_dim),
            nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Dropout(p_drop),
        )
        self.logit = nn.Linear(out_dim, 1)

    def forward(self, x):
        return self.logit(self.net(x)).squeeze(1)

# -------- Single-CSV routine -------------
def run_for_csv(csv_path: str):
    topic   = Path(csv_path).stem.split("_")[0]
    OUT_DIR = f"mlp_{topic}_UHU"
    ensure_dir(OUT_DIR); ensure_dir(f"{OUT_DIR}/plots")

    print(f"\n[INFO] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    df["pair_key"] = df["U.S.did"].astype(str) + "_" + df["U.R.did"].astype(str)
    df = df.drop(columns=[*EXCLUDE_COLS, *TEXT_COLS], errors="ignore")

    num_cols = [c for c in df.columns if PREFIX_RE.match(c)]
    if len(num_cols) == 0:
        raise ValueError(f"No numeric columns matching ^(U\\.|HU\\.) found in {csv_path}")

    df[num_cols]   = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df[num_cols]   = df[num_cols].fillna(df[num_cols].median()).astype(np.float32)
    df             = df.dropna(subset=[LABEL_COL])
    df[LABEL_COL]  = df[LABEL_COL].astype(int)

    overall_f1, overall_acc = [], []

    for run in range(N_RUNS):
        seed = BASE_SEED + run
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        print(f"\n=== [{topic}] Run {run+1}/{N_RUNS}  Seed {seed} ===")

        df_trainval, df_test = train_test_split(
            df, test_size=0.20, stratify=df[LABEL_COL], random_state=seed)
        val_frac = 0.10 / 0.80
        df_train, df_val = train_test_split(
            df_trainval, test_size=val_frac,
            stratify=df_trainval[LABEL_COL], random_state=seed)

        test_pairs = set(df_test["pair_key"].unique())
        df_train = df_train[~df_train["pair_key"].isin(test_pairs)].reset_index(drop=True)
        df_val   = df_val[~df_val["pair_key"].isin(test_pairs)].reset_index(drop=True)

        scaler  = MinMaxScaler().fit(df_train[num_cols])
        X_train = torch.tensor(scaler.transform(df_train[num_cols]), dtype=torch.float32)
        X_val   = torch.tensor(scaler.transform(df_val[num_cols]),   dtype=torch.float32)
        X_test  = torch.tensor(scaler.transform(df_test[num_cols]),  dtype=torch.float32)
        y_train = torch.tensor(df_train[LABEL_COL].values, dtype=torch.float32)
        y_val   = torch.tensor(df_val[LABEL_COL].values,   dtype=torch.float32)
        y_test  = torch.tensor(df_test[LABEL_COL].values,  dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=(device=="cuda"))
        val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=BATCH_SIZE*2)
        test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=BATCH_SIZE*2)

        model = UHUMlp(in_dim=len(num_cols), p_drop=DROPOUT_P).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        loss_log, f1_log = [], []
        best_f1, patience = 0.0, 0

        for epoch in range(EPOCHS):
            model.train(); epoch_loss = 0.0
            for xb, yb in tqdm(train_loader, leave=False, desc=f"{topic} run{run+1} epoch{epoch}"):
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_loader.dataset)
            scheduler.step()

            model.eval()
            with torch.no_grad():
                logits_val = torch.cat([model(xb.to(device)) for xb, _ in val_loader]).cpu()
            
            # Use fixed 0.5 threshold for predictions
            preds_val = (torch.sigmoid(logits_val) > FIXED_THRESHOLD).int()
            f1 = f1_score(y_val.numpy(), preds_val.numpy())

            loss_log.append(epoch_loss); f1_log.append(f1)
            print(f"Epoch {epoch:02d}  train-loss={epoch_loss:.4f}  val-F1={f1:.4f}")

            if f1 > best_f1:
                best_f1, patience = f1, 0
                torch.save(model.state_dict(), f"{OUT_DIR}/tmp_best.pt")
            else:
                patience += 1
                if patience >= PATIENCE:
                    print("Early stopping."); break
        
        model.load_state_dict(torch.load(f"{OUT_DIR}/tmp_best.pt", map_location=device))
        model.eval()
        with torch.no_grad():
            logits_test = torch.cat([model(xb.to(device)) for xb, _ in test_loader]).cpu()

        # Use fixed 0.5 threshold for final predictions
        preds_test = (torch.sigmoid(logits_test) > FIXED_THRESHOLD).int()
        f1  = f1_score(y_test.numpy(), preds_test.numpy())
        acc = accuracy_score(y_test.numpy(), preds_test.numpy())
        overall_f1.append(f1); overall_acc.append(acc)
        print(f"Run {run+1}: F1={f1:.4f}  Acc={acc:.4f}")

        plt.figure(figsize=(6,4))
        plt.plot(loss_log, label="train loss")
        plt.plot(f1_log,   label="val Binary F1") # Changed label
        plt.xlabel("Epoch"); plt.grid(True)
        plt.title(f"{topic} – Run {run+1} Loss & F1")
        plt.legend()
        ensure_dir(f"{OUT_DIR}/plots")
        plt.savefig(f"{OUT_DIR}/plots/run{run+1}_loss_f1.png", dpi=150, bbox_inches="tight")
        plt.close()

    summary = dict(
        n_runs=N_RUNS,
        f1_scores_all_runs=[float(x) for x in overall_f1],
        accuracy_scores_all_runs=[float(x) for x in overall_acc],
        f1_mean=float(np.mean(overall_f1)),
        f1_std=float(np.std(overall_f1)),
        accuracy_mean=float(np.mean(overall_acc)),
        accuracy_std=float(np.std(overall_acc)),
    )
    with open(f"{OUT_DIR}/overall_metrics.json", "w") as fp:
        json.dump(summary, fp, indent=4)

    print("\n=== Monte-Carlo Summary ({}) ===".format(topic))
    print(f"Binary F1: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}") # Changed label
    print(f"Accuracy : {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"✅  Plots saved to {OUT_DIR}/plots/")
    print(f"✅  Metrics saved to {OUT_DIR}/overall_metrics.json")

    try:
        os.remove(f"{OUT_DIR}/tmp_best.pt")
    except OSError:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -------- Batch entrypoint --------------
if __name__ == "__main__":
    files = sorted(glob.glob(CSV_GLOB))
    if not files:
        print(f"[WARN] No files matched pattern: {CSV_GLOB}")
    for f in files:
        run_for_csv(f)