#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP (U/HU numeric features) — Out-of-Distribution Evaluation (F1 only)
=====================================================================
• Leave-one-topic-out: train on ALL OTHER topics, test on the held-out topic
• K-fold CV on the training pool to select a checkpoint via early stopping
• If --kfolds=1: internally make up to 10 stratified splits but run ONLY ONE fold
  (deterministic choice via --fold_id or derived from --seed)

Data handling
-------------
• Numeric features: columns matching ^(U\.|HU\.)
• Row-level split; remove (S,R) pairs in TEST from TRAIN/VAL to avoid leakage
• MinMax scaling fit on TRAIN only; applied to VAL/TEST

Model
-----
• 512-256-128-64 MLP with BatchNorm + Dropout
• BCEWithLogitsLoss with pos_weight (computed from TRAIN labels)
• Cosine LR scheduler
• Fixed 0.5 threshold → standard positive-class Binary F1

Outputs (per held-out topic)
----------------------------
out_dir/
  mlp_holdout_<TOPIC>/
    fold_<i>/
      best.pt
    mlp_holdout_<TOPIC>_per_fold.jsonl   # eval_f1 per fold
    mlp_holdout_<TOPIC>.json             # summary with F1 only
"""

# ------------------------------ Imports ------------------------------ #
import os, re, json, math, glob, random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

# ------------------------- Hyper-parameters -------------------------- #
LABEL_COL   = "is_positive"
EPOCHS_DEF  = 30
BATCH_DEF   = 512
LR_DEF      = 3e-4
PATIENCE    = 8
DROPOUT_DEF = 0.0
MLP_OUT_DIM = 64
OUT_ROOT_DEF = "UHU_ood"
device      = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------- Column choices -------------------------- #
PREFIX_RE    = re.compile(r"^(U\.|HU\.)")
TEXT_COLS    = ["U.S.displayName", "U.S.description", "U.R.displayName", "U.R.description"]
EXCLUDE_COLS = ["M.uri", "M.record.createdAt"]   # keep DIDs for pair_key

# ------------------------------- Utils ------------------------------- #
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def topic_from_path(p: str) -> str:
    return Path(p).stem.split("_")[0]

def build_pair_key(df: pd.DataFrame) -> pd.Series:
    """Build (S,R) pair key for leakage removal."""
    return df["U.S.did"].astype(str) + "_" + df["U.R.did"].astype(str)

def select_numeric_cols(df: pd.DataFrame) -> List[str]:
    """Pick numeric feature columns by prefix regex."""
    return [c for c in df.columns if PREFIX_RE.match(c)]

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    """Coerce to numeric and impute column medians."""
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df[cols] = df[cols].fillna(df[cols].median()).astype(np.float32)

# ------------------------------ Model -------------------------------- #
class UHUMlp(nn.Module):
    """512-256-128-64 MLP with BatchNorm & Dropout, then a 1-logit head."""
    def __init__(self, in_dim: int, p_drop: float = DROPOUT_DEF, out_dim: int = MLP_OUT_DIM):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logit(self.net(x)).squeeze(1)

# --------------------------- Train / Eval ---------------------------- #
def train_one_fold_mlp(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    in_dim: int, epochs: int, batch: int, lr: float, drop: float,
    out_dir: str, seed: int
) -> Tuple[nn.Module, float]:
    """Train one fold with early stopping on validation F1@0.5; return best model + best F1."""
    # Build tensors
    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    Ytr = torch.tensor(y_tr, dtype=torch.float32)
    Xva = torch.tensor(X_va, dtype=torch.float32)
    Yva = torch.tensor(y_va, dtype=torch.float32)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch, shuffle=True,
                              num_workers=4, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=max(256, batch))

    # Model / loss / optim
    model = UHUMlp(in_dim=in_dim, p_drop=drop).to(device)

    # Class imbalance handling: pos_weight = neg_count / pos_count (computed on TRAIN only)
    pos = max(1, int(Ytr.sum().item()))
    neg = max(1, int(len(Ytr) - pos))
    pos_weight = torch.tensor(neg / pos, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Reproducibility
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    best_f1, patience = 0.0, 0
    best_path = os.path.join(out_dir, "best.pt")

    for ep in range(epochs):
        # ---- train ----
        model.train(); epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        scheduler.step()

        # ---- validate ----
        model.eval()
        with torch.no_grad():
            logits_va = torch.cat([model(xb.to(device)) for xb, _ in val_loader]).cpu().numpy()
        preds_va = (1.0 / (1.0 + np.exp(-logits_va)) >= 0.5).astype(int)
        f1_va = f1_score(y_va, preds_va)

        print(f"  epoch {ep:02d}  train-loss={epoch_loss:.4f}  val-F1={f1_va:.4f}")

        if f1_va > best_f1:
            best_f1, patience = f1_va, 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= PATIENCE:
                print("  early stopping")
                break

    # Load best to return
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, best_f1

def evaluate_f1(model: nn.Module, X_te: np.ndarray, y_te: np.ndarray, batch: int = 1024) -> float:
    """Evaluate Binary F1@0.5 on test set."""
    model.eval()
    Xte = torch.tensor(X_te, dtype=torch.float32)
    test_loader = DataLoader(TensorDataset(Xte, torch.zeros(len(Xte))), batch_size=batch)
    with torch.no_grad():
        logits = torch.cat([model(xb.to(device)) for xb, _ in test_loader]).cpu().numpy()
    preds = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(int)
    return f1_score(y_te, preds)

# --------------------------- Data loader ----------------------------- #
def load_topic_df(csv_path: str) -> Tuple[str, pd.DataFrame, List[str]]:
    """
    Load one CSV; keep DIDs for pair_key; prepare numeric columns list.
    Returns (topic, df, num_cols).
    """
    topic = topic_from_path(csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    # Build pair key before dropping columns
    if "U.S.did" in df.columns and "U.R.did" in df.columns:
        df["pair_key"] = build_pair_key(df)
    # Drop non-numeric text cols; keep DIDs to purge leakage later
    df = df.drop(columns=[c for c in EXCLUDE_COLS + TEXT_COLS if c in df.columns], errors="ignore")
    # Numeric selection & preprocessing
    num_cols = select_numeric_cols(df)
    if not num_cols:
        raise ValueError(f"No numeric columns matching ^(U\\.|HU\\.) in {csv_path}")
    coerce_numeric(df, num_cols)
    # Labels
    df = df.dropna(subset=[LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df["__topic__"] = topic
    return topic, df, num_cols

# ----------------------------- OOD runner ---------------------------- #
def run_ood(csv_list: List[str], out_root: str, epochs: int, batch: int,
            kfolds: int, fold_id: int | None, target_topic: str | None,
            seed_base: int, lr: float, dropout: float):
    # Load all topics
    loaded = [load_topic_df(p) for p in csv_list]
    topics  = [t for t, _, _ in loaded]
    if target_topic is not None and target_topic not in topics:
        raise ValueError(f"Held-out topic '{target_topic}' not in {topics}")

    holdouts = [target_topic] if target_topic is not None else topics
    ensure_dir(out_root)

    # Sanity: ensure all files share the same numeric feature set
    common_num_cols = set(loaded[0][2])
    for _, _, cols in loaded[1:]:
        common_num_cols &= set(cols)
    common_num_cols = sorted(list(common_num_cols))
    if len(common_num_cols) == 0:
        raise ValueError("No common numeric columns across CSVs.")
    print(f"[INFO] Using {len(common_num_cols)} common numeric features.")

    for held in holdouts:
        print(f"\n========== OOD: hold out topic = {held} ==========")
        # Build train pool and test set
        df_test = next(df for t, df, _ in loaded if t == held)
        df_train_pool = pd.concat([df for t, df, _ in loaded if t != held], ignore_index=True)

        # Remove (S,R) pairs in TEST from TRAIN/VAL
        if "pair_key" in df_test.columns and "pair_key" in df_train_pool.columns:
            test_pairs = set(df_test["pair_key"].dropna().unique().tolist())
            before = len(df_train_pool)
            df_train_pool = df_train_pool[~df_train_pool["pair_key"].isin(test_pairs)].reset_index(drop=True)
            print(f"[LEAKAGE] Removed {before - len(df_train_pool)} train rows with test pair_key overlap.")

        # Extract features/labels with the common columns
        X_pool = df_train_pool[common_num_cols].values
        y_pool = df_train_pool[LABEL_COL].values.astype(int)
        X_test = df_test[common_num_cols].values
        y_test = df_test[LABEL_COL].values.astype(int)

        # K-fold config
        unique, counts = np.unique(y_pool, return_counts=True)
        min_class = int(counts.min())

        # Output dirs
        hold_dir = os.path.join(out_root, f"mlp_holdout_{held}")
        ensure_dir(hold_dir)
        metrics_path = os.path.join(hold_dir, f"mlp_holdout_{held}_per_fold.jsonl")
        if os.path.exists(metrics_path):
            os.remove(metrics_path)

        if kfolds == 1:
            # Internally make up to 10 splits, but run only one
            target_splits = 10
            n_splits = min(target_splits, max(2, min_class))
            if n_splits < target_splits:
                print(f"[WARN] Using {n_splits} splits (min-class limited; min_class={min_class}).")
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_base)

            # Choose the single fold index
            if fold_id is not None:
                if not (1 <= fold_id <= n_splits):
                    raise ValueError(f"--fold_id must be in [1..{n_splits}], got {fold_id}.")
                chosen_fold = fold_id
            else:
                chosen_fold = (seed_base % n_splits) + 1
            print(f"[MODE] Single-fold from {n_splits} splits → chosen fold = {chosen_fold}")

            run_results = []
            for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_pool)), y_pool), start=1):
                if fold_idx != chosen_fold:
                    continue

                seed = seed_base + fold_idx
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                # Split arrays
                X_tr_raw, X_va_raw = X_pool[tr_idx], X_pool[va_idx]
                y_tr,     y_va     = y_pool[tr_idx], y_pool[va_idx]

                # Scale on TRAIN only
                scaler = MinMaxScaler().fit(X_tr_raw)
                X_tr = scaler.transform(X_tr_raw)
                X_va = scaler.transform(X_va_raw)
                X_te = scaler.transform(X_test)

                # Fold dir
                fold_dir = os.path.join(hold_dir, f"fold_{fold_idx}")
                ensure_dir(fold_dir)

                # Train + evaluate
                model, best_f1 = train_one_fold_mlp(
                    X_tr, y_tr, X_va, y_va,
                    in_dim=len(common_num_cols), epochs=epochs, batch=batch, lr=lr, drop=dropout,
                    out_dir=fold_dir, seed=seed
                )
                f1_te = evaluate_f1(model, X_te, y_test, batch=max(256, batch))
                print(f"[{held}] Single-fold {fold_idx}/{n_splits} → F1={f1_te:.4f}")

                run_results.append({"eval_f1": float(f1_te)})

                with open(metrics_path, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps({
                        "fold": fold_idx,
                        "n_splits": n_splits,
                        "eval_f1": float(f1_te)
                    }) + "\n")

                # Free GPU mem
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Summary
            f1s = [r["eval_f1"] for r in run_results]
            summary = {
                "topic": held,
                "mode": "single-fold-from-10",
                "kfolds_requested": 1,
                "kfolds_effective": n_splits,
                "folds_run": 1,
                "chosen_fold": chosen_fold,
                "f1_all": f1s,
                "f1_mean": float(np.mean(f1s)) if f1s else float('nan'),
                "f1_std":  0.0,
                "n_samples_train_pool": int(len(df_train_pool)),
                "n_samples_test": int(len(df_test)),
                "n_features": int(len(common_num_cols)),
            }
            with open(os.path.join(hold_dir, f"mlp_holdout_{held}.json"), "w", encoding="utf-8") as fp:
                json.dump(summary, fp, indent=2)
            print(f"\n[SUMMARY:{held}] (single-fold) F1 = {summary['f1_mean']:.4f}")

        else:
            # K-fold mode
            n_splits = min(kfolds, max(2, min_class))
            if n_splits < kfolds:
                print(f"[WARN] Reducing kfolds from {kfolds} to {n_splits} due to class counts {dict(zip(unique, counts))}.")
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_base)

            fold_results = []
            for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_pool)), y_pool), start=1):
                seed = seed_base + fold_idx
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                X_tr_raw, X_va_raw = X_pool[tr_idx], X_pool[va_idx]
                y_tr,     y_va     = y_pool[tr_idx], y_pool[va_idx]

                scaler = MinMaxScaler().fit(X_tr_raw)
                X_tr = scaler.transform(X_tr_raw)
                X_va = scaler.transform(X_va_raw)
                X_te = scaler.transform(X_test)

                fold_dir = os.path.join(hold_dir, f"fold_{fold_idx}")
                ensure_dir(fold_dir)

                model, best_f1 = train_one_fold_mlp(
                    X_tr, y_tr, X_va, y_va,
                    in_dim=len(common_num_cols), epochs=epochs, batch=batch, lr=lr, drop=dropout,
                    out_dir=fold_dir, seed=seed
                )
                f1_te = evaluate_f1(model, X_te, y_test, batch=max(256, batch))
                print(f"[{held}] Fold {fold_idx}/{n_splits} → F1={f1_te:.4f}")

                fold_results.append({"eval_f1": float(f1_te)})

                with open(metrics_path, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps({
                        "fold": fold_idx,
                        "n_splits": n_splits,
                        "eval_f1": float(f1_te)
                    }) + "\n")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            f1s = [r["eval_f1"] for r in fold_results]
            summary = {
                "topic": held,
                "mode": "kfold",
                "kfolds_requested": kfolds,
                "kfolds_effective": n_splits,
                "folds_run": n_splits,
                "f1_all": f1s,
                "f1_mean": float(np.mean(f1s)) if f1s else float('nan'),
                "f1_std":  float(np.std(f1s))  if f1s else float('nan'),
                "n_samples_train_pool": int(len(df_train_pool)),
                "n_samples_test": int(len(df_test)),
                "n_features": int(len(common_num_cols)),
            }
            with open(os.path.join(hold_dir, f"mlp_holdout_{held}.json"), "w", encoding="utf-8") as fp:
                json.dump(summary, fp, indent=2)
            print(f"\n[SUMMARY:{held}] F1 = {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")

# --------------------------------- CLI -------------------------------- #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="MLP OOD evaluator on U/HU numeric features (F1 only)")
    ap.add_argument("--csvs", nargs="+", required=True, help="List of CSV files (one per topic).")
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEF, help="Training epochs per fold.")
    ap.add_argument("--batch",  type=int, default=BATCH_DEF, help="Batch size.")
    ap.add_argument("--lr",     type=float, default=LR_DEF,  help="Learning rate.")
    ap.add_argument("--dropout",type=float, default=DROPOUT_DEF, help="Dropout probability.")
    ap.add_argument("--kfolds", type=int, default=10, help="K for OOD. Use 1 to run a single fold chosen from a 10-way split.")
    ap.add_argument("--fold_id",type=int, default=None, help="When --kfolds=1, choose which fold (1..N). If omitted, derived from --seed.")
    ap.add_argument("--ood_topic", type=str, default=None, help="Hold out a specific topic (stem before first '_' of a CSV filename).")
    ap.add_argument("--ood_all", action="store_true", help="Run leave-one-topic-out for ALL topics provided.")
    ap.add_argument("--out_dir", type=str, default=OUT_ROOT_DEF, help=f"Root output directory (default: {OUT_ROOT_DEF}).")
    ap.add_argument("--seed", type=int, default=42, help="Base random seed.")
    args = ap.parse_args()

    if not args.ood_all and args.ood_topic is None:
        raise SystemExit("Please provide --ood_topic <name> or --ood_all.")
    if args.ood_all and args.ood_topic is not None:
        print("[WARN] Both --ood_all and --ood_topic are set; proceeding with --ood_all.")

    # Expand csvs (support globs passed by shell or literal patterns)
    files = []
    for p in args.csvs:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    if not files:
        raise SystemExit("No CSV files found for --csvs.")

    run_ood(
        csv_list=files,
        out_root=args.out_dir,
        epochs=args.epochs,
        batch=args.batch,
        kfolds=args.kfolds,
        fold_id=args.fold_id,
        target_topic=None if args.ood_all else args.ood_topic,
        seed_base=args.seed,
        lr=args.lr,
        dropout=args.dropout
    )
