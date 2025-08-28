#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Message-BERT — Out-of-Distribution (OOD) Evaluation (F1 only)
=============================================================
• Train on ALL OTHER topics; test on the held-out topic.
• K-fold CV inside the training pool (each fold provides validation).
• If --kfolds=1: internally create up to 10 stratified splits, but run ONLY ONE fold
  (deterministic choice via --fold_id or derived from --seed).

Reported metric
---------------
• Binary F1 for the positive class at a fixed 0.5 threshold (primary and only metric)

Outputs per held-out topic
--------------------------
out_dir/
  message_holdout_<TOPIC>/
    fold_<i>/
    message_holdout_<TOPIC>_per_fold.jsonl   # per-fold eval_f1 only
    message_holdout_<TOPIC>.json             # summary with F1 only
    
Usage
python message_ood.py --csvs *_1to5.csv --ood_all --kfolds 1 --epochs 3 --batch 32 --seed 42
"""

# ------------------------------ Imports ------------------------------ #
import os
import re
import json
import argparse
import pathlib
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import (
    BertTokenizerFast, BertModel,
    TrainingArguments, Trainer, EarlyStoppingCallback, get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import evaluate
import ftfy
import unidecode

# ------------------------- Hyper-parameters -------------------------- #
LABEL_COL   = "is_positive"
LR_BERT     = 5e-5
WD_BERT     = 0.01
PATIENCE    = 8
MAX_SEQ_CAP = 512  # effective cap; we choose min(train-pool true max, 512)

# --------------------------- Column choices -------------------------- #
MESSAGE_TEXT_COLS = [
    "M.record.text",
    # Add more text fields if needed:
    # "M.record.embed.external.description",
    # "M.record.embed.external.title",
    # "M.record.tags",
    # "M.embed.record.value.text",
    # "M.embed.record.value.embed.external.description",
    # "M.embed.record.value.embed.external.title",
    # "M.record.embed.media.external.description",
    # "M.record.embed.media.external.title",
]
EXCL_COLS = ["M.uri", "U.S.did", "U.R.did", "M.record.createdAt"]

# ----------------------- Tokeniser & cleaners ------------------------ #
FIELD_SEP = "[unused1]"
URL_RE, TAG_RE = re.compile(r"https?://\S+"), re.compile(r"<[^>]+>")
tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

def clean_txt(s: str) -> str:
    """Fix encoding, strip links/tags, collapse whitespace."""
    s = ftfy.fix_text(str(s))
    s = URL_RE.sub("[URL]", s)
    s = TAG_RE.sub(" ", s)
    s = unidecode.unidecode(s)
    return re.sub(r"\s+", " ", s).strip()

def concat_cols(df: pd.DataFrame, cols: list[str], sep: str = FIELD_SEP) -> pd.Series:
    """Concatenate selected text columns and clean the result."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series([""] * len(df), index=df.index)
    return (
        df[present].fillna("").astype(str)
          .agg(f" {sep} ".join, axis=1)
          .map(clean_txt)
    )

# ------------------------------ Dataset ------------------------------ #
class MessageDS(Dataset):
    """Torch dataset returning dicts compatible with HF Trainer."""
    def __init__(self, ids, masks, labels):
        self.ids   = torch.tensor(ids)
        self.masks = torch.tensor(masks)
        self.lab   = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, idx):
        return {
            "input_ids": self.ids[idx],
            "attention_mask": self.masks[idx],
            "labels": self.lab[idx],
        }

# ------------------------------- Model -------------------------------- #
class BertClassifier(nn.Module):
    """
    BERT encoder with dropout + linear classifier.
    Supports optional class weights for cost-sensitive training.
    """
    def __init__(self, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.bert  = BertModel.from_pretrained("bert-base-uncased")
        hidden     = self.bert.config.hidden_size
        self.drop  = nn.Dropout(0.1)
        self.cls   = nn.Linear(hidden, 2)
        # Register as buffer to keep it on the right device automatically
        self.register_buffer("class_weights", class_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        logits = self.cls(self.drop(pooled))
        if labels is not None:
            if self.class_weights is not None:
                loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

# ------------------------------- Metrics ------------------------------ #
metric_f1 = evaluate.load("f1")  # positive-class F1 via average="binary"

def compute_metrics(eval_pred):
    """
    Report only: binary-F1@0.5 (positive).
    For a 2-logit head, p_pos >= 0.5 is equivalent to (logit1 - logit0) >= 0.
    """
    if isinstance(eval_pred, tuple):
        logits, labels = eval_pred
    else:
        logits, labels = eval_pred.predictions, eval_pred.label_ids

    if logits.ndim == 2 and logits.shape[1] == 2:
        z = logits[:, 1] - logits[:, 0]       # logit difference
        preds = (z >= 0).astype(int)          # fixed 0.5 threshold
    else:
        preds = np.argmax(logits, axis=-1)

    f1_pos = metric_f1.compute(predictions=preds, references=labels, average="binary")["f1"]
    return {"f1": f1_pos}

# --------------------------- Helpers (train) -------------------------- #
def compute_class_weights_from_train_labels(y: np.ndarray) -> torch.Tensor:
    """
    Build inverse-frequency class weights from training labels and normalize to mean=1
    to keep the loss scale stable across folds/topics.
    """
    neg = max(1, int((y == 0).sum()))
    pos = max(1, int((y == 1).sum()))
    w0, w1 = 1.0 / neg, 1.0 / pos
    s = (w0 + w1) / 2.0
    return torch.tensor([w0 / s, w1 / s], dtype=torch.float)

# --------------------------- Train one fold --------------------------- #
def train_one_fold(model, ds_tr, ds_va, epochs, batch_size, out_dir, seed, fold_idx):
    """Fine-tune with AdamW + linear scheduler + early stopping."""
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": WD_BERT},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, lr=LR_BERT)

    steps_per_epoch = max(1, math.ceil(len(ds_tr) / batch_size))
    total_steps     = max(1, epochs * steps_per_epoch)
    warmup_steps    = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    args = TrainingArguments(
        output_dir=f"{out_dir}/fold_{fold_idx}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",   # primary metric: positive-class binary F1
        greater_is_better=True,
        dataloader_drop_last=False,        # do not drop last batch for train/eval
        seed=seed,
        report_to="none",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )
    trainer.train()
    return trainer

# ------------------------ Utility: token stats ------------------------ #
def max_tokens_on(series: pd.Series, tokenizer: BertTokenizerFast) -> int:
    """Compute the true max token length on a corpus (no truncation)."""
    if len(series) == 0:
        return 0
    toks = tokenizer(series.tolist(), truncation=False, padding=False)["input_ids"]
    return max(len(x) for x in toks) if toks else 0

def encode(series: pd.Series, tokenizer: BertTokenizerFast, max_len: int):
    """Tokenise with padding to max_len and truncation."""
    out = tokenizer(series.tolist(), truncation=True, padding="max_length", max_length=max_len)
    return out["input_ids"], out["attention_mask"]

# ------------------------ Build topic dataframe ----------------------- #
def load_topic_df(csv_path: str) -> tuple[str, pd.DataFrame]:
    """Load a CSV and return (topic_name, processed_df). Topic is stem before first '_'."""
    topic = pathlib.Path(csv_path).stem.split("_")[0]
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.drop(columns=[c for c in EXCL_COLS if c in df.columns], errors="ignore")
    df = df.dropna(subset=[LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df["MSG_text"] = concat_cols(df, MESSAGE_TEXT_COLS)
    df["__topic__"] = topic
    return topic, df

# ----------------------------- OOD runner ----------------------------- #
def run_ood(csv_list: list[str], out_root: str, epochs: int, batch_size: int,
            kfolds: int, fold_id: int | None, target_topic: str | None, seed_base: int):
    """Perform leave-one-topic-out OOD evaluation with K-fold or single-fold mode."""
    # Load all topics
    topic_dfs = []
    for p in csv_list:
        t, d = load_topic_df(p)
        topic_dfs.append((t, d))
    topics = [t for t, _ in topic_dfs]

    if target_topic is not None and target_topic not in topics:
        raise ValueError(f"Held-out topic '{target_topic}' not found in provided CSVs: {topics}")

    holdouts = [target_topic] if target_topic is not None else topics
    os.makedirs(out_root, exist_ok=True)

    for held in holdouts:
        print(f"\n========== OOD: hold out topic = {held} ==========")
        # Build train pool and test set
        df_train_pool = pd.concat([d for t, d in topic_dfs if t != held], ignore_index=True)
        df_test       = next(d for t, d in topic_dfs if t == held)

        # Compute max token length on training pool only (avoid leakage)
        tr_max = max_tokens_on(df_train_pool["MSG_text"], tok)
        max_len = min(max(8, tr_max), MAX_SEQ_CAP)
        print(f"[STATS] Train-pool true max tokens = {tr_max} → using max_len = {max_len}")

        # Prepare fixed test dataset
        ids_te, mask_te = encode(df_test["MSG_text"], tok, max_len)
        ds_te = MessageDS(ids_te, mask_te, df_test[LABEL_COL])

        # K-fold over the training pool
        y = df_train_pool[LABEL_COL].values
        unique, counts = np.unique(y, return_counts=True)
        min_class = int(counts.min())

        # --------- Mode selection ---------
        if kfolds == 1:
            target_splits = 10
            n_splits = min(target_splits, max(2, min_class))
            if n_splits < target_splits:
                print(f"[WARN] Using {n_splits} splits (class-count limited; min-class={min_class}).")
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_base)

            # Pick exactly ONE fold to run
            if fold_id is not None:
                if not (1 <= fold_id <= n_splits):
                    raise ValueError(f"--fold_id must be in [1..{n_splits}], got {fold_id}.")
                chosen_fold = fold_id
            else:
                chosen_fold = (seed_base % n_splits) + 1
            print(f"[MODE] Single-fold from {n_splits} splits → chosen fold = {chosen_fold}")

            hold_dir = os.path.join(out_root, f"message_holdout_{held}")
            os.makedirs(hold_dir, exist_ok=True)
            metrics_path = os.path.join(hold_dir, f"message_holdout_{held}_per_fold.jsonl")
            if os.path.exists(metrics_path):
                os.remove(metrics_path)

            run_results = []
            for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
                if fold_idx != chosen_fold:
                    continue

                seed = seed_base + fold_idx
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                df_tr = df_train_pool.iloc[tr_idx]
                df_va = df_train_pool.iloc[va_idx]

                ids_tr, mask_tr = encode(df_tr["MSG_text"], tok, max_len)
                ids_va, mask_va = encode(df_va["MSG_text"], tok, max_len)

                ds_tr = MessageDS(ids_tr, mask_tr, df_tr[LABEL_COL])
                ds_va = MessageDS(ids_va, mask_va, df_va[LABEL_COL])

                # ----- class weights from training split -----
                class_weights = compute_class_weights_from_train_labels(df_tr[LABEL_COL].astype(int).values)
                model = BertClassifier(class_weights=class_weights)

                trainer = train_one_fold(
                    model, ds_tr, ds_va,
                    epochs=epochs, batch_size=batch_size,
                    out_dir=hold_dir, seed=seed, fold_idx=fold_idx
                )

                res = trainer.evaluate(ds_te)
                print(f"[{held}] Single-fold {fold_idx}/{n_splits} → F1={res['eval_f1']:.4f}")
                run_results.append(res)

                with open(metrics_path, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps({
                        "fold": fold_idx,
                        "n_splits": n_splits,
                        "eval_f1": float(res["eval_f1"])
                    }) + "\n")

                torch.cuda.empty_cache()

            # Summarise (single line, but keep structure consistent)
            f1s  = [r["eval_f1"] for r in run_results]
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
                "max_len_used": max_len,
            }
            summary_path = os.path.join(hold_dir, f"message_holdout_{held}.json")
            with open(summary_path, "w", encoding="utf-8") as fp:
                json.dump(summary, fp, indent=2)

            print(f"\n[SUMMARY:{held}] (single-fold) F1 = {summary['f1_mean']:.4f}")
            print(f"Saved: {summary_path}")

        else:
            # Standard K-fold mode
            n_splits = min(kfolds, max(2, min_class))  # ≥2 and ≤ min-class
            if n_splits < kfolds:
                print(f"[WARN] Reducing kfolds from {kfolds} to {n_splits} due to class counts {dict(zip(unique, counts))}.")
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_base)

            hold_dir = os.path.join(out_root, f"message_holdout_{held}")
            os.makedirs(hold_dir, exist_ok=True)
            metrics_path = os.path.join(hold_dir, f"message_holdout_{held}_per_fold.jsonl")
            if os.path.exists(metrics_path):
                os.remove(metrics_path)

            fold_results = []
            for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
                seed = seed_base + fold_idx
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                df_tr = df_train_pool.iloc[tr_idx]
                df_va = df_train_pool.iloc[va_idx]

                ids_tr, mask_tr = encode(df_tr["MSG_text"], tok, max_len)
                ids_va, mask_va = encode(df_va["MSG_text"], tok, max_len)

                ds_tr = MessageDS(ids_tr, mask_tr, df_tr[LABEL_COL])
                ds_va = MessageDS(ids_va, mask_va, df_va[LABEL_COL])

                # ----- class weights from training split -----
                class_weights = compute_class_weights_from_train_labels(df_tr[LABEL_COL].astype(int).values)
                model = BertClassifier(class_weights=class_weights)

                trainer = train_one_fold(
                    model, ds_tr, ds_va,
                    epochs=epochs, batch_size=batch_size,
                    out_dir=hold_dir, seed=seed, fold_idx=fold_idx
                )

                res = trainer.evaluate(ds_te)
                print(f"[{held}] Fold {fold_idx}/{n_splits} → F1={res['eval_f1']:.4f}")
                fold_results.append(res)

                with open(metrics_path, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps({
                        "fold": fold_idx,
                        "n_splits": n_splits,
                        "eval_f1": float(res["eval_f1"])
                    }) + "\n")

                torch.cuda.empty_cache()

            # Summarise across folds
            f1s  = [r["eval_f1"] for r in fold_results]
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
                "max_len_used": max_len,
            }
            summary_path = os.path.join(hold_dir, f"message_holdout_{held}.json")
            with open(summary_path, "w", encoding="utf-8") as fp:
                json.dump(summary, fp, indent=2)

            print(f"\n[SUMMARY:{held}] F1 = {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
            print(f"Saved: {summary_path}")

# --------------------------------- CLI -------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Message-BERT OOD evaluator (F1 only)")
    ap.add_argument("--csvs", nargs="+", required=True, help="List of CSV files (one per topic).")
    ap.add_argument("--epochs", type=int, default=3, help="Fine-tuning epochs per fold.")
    ap.add_argument("--batch",  type=int, default=32, help="Batch size.")
    ap.add_argument("--kfolds", type=int, default=10,
                    help="K for OOD. Use 1 to run a single fold chosen from a 10-way split.")
    ap.add_argument("--fold_id", type=int, default=None,
                    help="When --kfolds=1, choose which fold (1..N) to run. If omitted, derived from --seed.")
    ap.add_argument("--ood_topic", type=str, default=None,
                    help="Hold out a specific topic (stem before first '_' of a CSV filename).")
    ap.add_argument("--ood_all", action="store_true",
                    help="Run leave-one-topic-out for ALL topics provided.")
    ap.add_argument("--out_dir", type=str, default="message_ood",
                    help="Root output directory (default: message_ood).")
    ap.add_argument("--seed", type=int, default=42, help="Base random seed.")
    args = ap.parse_args()

    if not args.ood_all and args.ood_topic is None:
        raise SystemExit("Please provide --ood_topic <name> or --ood_all.")
    if args.ood_all and args.ood_topic is not None:
        print("[WARN] Both --ood_all and --ood_topic are set; proceeding with --ood_all.")

    run_ood(
        csv_list=args.csvs,
        out_root=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        kfolds=args.kfolds,
        fold_id=args.fold_id,
        target_topic=None if args.ood_all else args.ood_topic,
        seed_base=args.seed
    )

if __name__ == "__main__":
    main()
