#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triple-BERT + MLP — Two-Phase Training
======================================
Phase-1: train numeric MLP (+ classifier) for N1 epochs, BERT frozen
Phase-2: jointly fine-tune BERT + MLP for N2 epochs

CLI example:
    python user_related.py \
        --csvs mytopic_1to5.csv \
        --joint_epochs 3 \
        --joint_batch 32
"""

# ------------------------------------------------------------------ #
# Imports                                                            #
# ------------------------------------------------------------------ #
import os
import re
import json
import argparse
import pathlib
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast, BertModel,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import evaluate
import ftfy
import unidecode
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import math

# ------------------------------------------------------------------ #
# Hyper-parameters (defaults)                                        #
# ------------------------------------------------------------------ #
LABEL_COL            = "is_positive"
N_RUNS               = 1
BASE_SEED            = 42

# Phase-1 (MLP-only)
EPOCHS_MLP           = 30
MLP_BATCH            = 512
LR_MLP               = 3e-4

# Phase-2 (joint)
EPOCHS_JOINT_DEF     = 3          # CLI override: --joint_epochs
JOINT_BATCH_DEF      = 32         # CLI override: --joint_batch
LR_BERT              = 5e-5
WD_BERT              = 0.01

# Misc
PATIENCE             = 8
MLP_HIDDEN           = [512, 256, 128, 64]
MLP_DROP             = 0.0

# ------------------------------------------------------------------ #
# Column definitions                                                 #
# ------------------------------------------------------------------ #
U_TEXT_COLS   = ["U.S.displayName", "U.S.description",
                 "U.R.displayName", "U.R.description"]
HMS_TEXT_COLS = ["UHM.S.record.text"]
HMR_TEXT_COLS = ["UHM.R.record.text"]
EXCL_COLS     = ["M.uri", "U.S.did", "U.R.did", "M.record.createdAt"]
NUM_RE        = re.compile(r"^(U\.|HU\.)")  # numeric U.* / HU.* columns

# ------------------------------------------------------------------ #
# Tokenizer & text utils                                             #
# ------------------------------------------------------------------ #
FIELD_SEP = "[unused1]"
URL_RE, TAG_RE = re.compile(r"https?://\S+"), re.compile(r"<[^>]+>")
tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

def clean_txt(s: str) -> str:
    """Basic text cleaning."""
    s = ftfy.fix_text(str(s))
    s = URL_RE.sub("[URL]", s)
    s = TAG_RE.sub(" ", s)
    s = unidecode.unidecode(s)
    return re.sub(r"\s+", " ", s).strip()

def concat(df: pd.DataFrame, cols: list[str], sep: str = FIELD_SEP) -> pd.Series:
    """Concatenate columns with separator and clean."""
    return (
        df[cols].fillna("").astype(str)
          .agg(f" {sep} ".join, axis=1)
          .map(clean_txt)
    )

# ------------------------------------------------------------------ #
# Dataset                                                             #
# ------------------------------------------------------------------ #
class DictDS(Dataset):
    """Return dicts ready for HuggingFace Trainer."""
    def __init__(self, idsU, maskU, idsS, maskS, idsR, maskR, num, labels):
        self.idsU,  self.maskU = torch.tensor(idsU), torch.tensor(maskU)
        self.idsS,  self.maskS = torch.tensor(idsS), torch.tensor(maskS)
        self.idsR,  self.maskR = torch.tensor(idsR), torch.tensor(maskR)
        self.num               = torch.tensor(num, dtype=torch.float32)
        self.lab               = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, idx):
        return dict(
            ids_U   = self.idsU[idx],   mask_U   = self.maskU[idx],
            ids_HMS = self.idsS[idx],   mask_HMS = self.maskS[idx],
            ids_HMR = self.idsR[idx],   mask_HMR = self.maskR[idx],
            num     = self.num[idx],
            labels  = self.lab[idx]
        )

# ------------------------------------------------------------------ #
# Model                                                               #
# ------------------------------------------------------------------ #
class MLP(nn.Module):
    """Feed-forward network for numeric features."""
    def __init__(self, in_dim: int):
        super().__init__()
        dims   = [in_dim] + MLP_HIDDEN
        layers = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(MLP_DROP),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)          # [batch, 64]

class TripleBert(nn.Module):
    """Three BERT towers + numeric MLP + classifier."""
    def __init__(self, n_num: int):
        super().__init__()
        self.bert_U = BertModel.from_pretrained("bert-base-uncased")
        self.bert_S = BertModel.from_pretrained("bert-base-uncased")
        self.bert_R = BertModel.from_pretrained("bert-base-uncased")
        self.mlp    = MLP(n_num)
        hidden      = self.bert_U.config.hidden_size
        self.cls    = nn.Linear(3 * hidden + MLP_HIDDEN[-1], 2)
        self.drop   = nn.Dropout(0.1)

    def forward(self, ids_U, mask_U,
                      ids_HMS, mask_HMS,
                      ids_HMR, mask_HMR,
                      num, labels=None):
        pU = self.bert_U(ids_U,   attention_mask=mask_U  ).pooler_output
        pS = self.bert_S(ids_HMS, attention_mask=mask_HMS).pooler_output
        pR = self.bert_R(ids_HMR, attention_mask=mask_HMR).pooler_output
        pN = self.mlp(num)
        logits = self.cls(self.drop(torch.cat([pU, pS, pR, pN], dim=1)))
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

# ------------------------------------------------------------------ #
# Metrics                                                             #
# ------------------------------------------------------------------ #
metric_acc = evaluate.load("accuracy")
metric_f1  = evaluate.load("f1")

def compute_metrics(pred):
    """Computes accuracy and standard binary F1 for the positive class."""
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    return dict(
        accuracy = metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        # MODIFIED: Changed from 'macro' to 'binary'
        f1       = metric_f1.compute(predictions=preds, references=labels, average="binary")["f1"],
    )

# ------------------------------------------------------------------ #
# Training helpers                                                   #
# ------------------------------------------------------------------ #
def train_mlp_only(model, ds_tr, ds_va, out_dir, seed):
    """Phase-1: train only MLP + classifier, BERT frozen."""
    for n, p in model.named_parameters():
        if n.startswith("bert_"):
            p.requires_grad = False
        else:
            p.requires_grad = True

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_MLP
    )

    args = TrainingArguments(
        output_dir             = f"{out_dir}/phase1_mlp",
        num_train_epochs       = EPOCHS_MLP,
        per_device_train_batch_size = MLP_BATCH,
        eval_strategy          = "epoch",
        save_strategy          = "epoch",
        save_total_limit       = 1,
        load_best_model_at_end = True,
        metric_for_best_model  = "eval_f1",
        dataloader_drop_last   = True,
        seed                   = seed,
        report_to              = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = ds_tr,
        eval_dataset    = ds_va,
        compute_metrics = compute_metrics,
        optimizers      = (optimizer, None),
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
    )

    trainer.train()
    print("Phase-1 (MLP-only) finished.")
    return trainer

LR_MLP_PHASE2 = 5e-5   # learning rate for the MLP branch in phase-2

def train_joint(model, ds_tr, ds_va,
                joint_epochs, joint_batch,
                out_dir, seed):
    """Phase-2: fine-tune BERT and MLP together."""
    for p in model.parameters():
        p.requires_grad = True

    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if n.startswith("bert") and not any(nd in n for nd in no_decay)],
            "lr": LR_BERT, "weight_decay": WD_BERT
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if n.startswith("bert") and any(nd in n for nd in no_decay)],
            "lr": LR_BERT, "weight_decay": 0.0
        },
        {
            "params": [p for n, p in model.named_parameters() if not n.startswith("bert")],
            "lr": LR_MLP_PHASE2, "weight_decay": 0.0
        }
    ]

    optimizer = AdamW(param_groups)
    steps_per_epoch = math.ceil(len(ds_tr) / joint_batch)
    total_steps     = steps_per_epoch * joint_epochs
    warmup_steps    = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    args = TrainingArguments(
        output_dir             = f"{out_dir}/phase2_joint",
        num_train_epochs       = joint_epochs,
        per_device_train_batch_size = joint_batch,
        eval_strategy          = "epoch",
        save_strategy          = "epoch",
        save_total_limit       = 1,
        load_best_model_at_end = True,
        metric_for_best_model  = "eval_f1",
        dataloader_drop_last   = True,
        seed                   = seed,
        report_to              = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = ds_tr,
        eval_dataset    = ds_va,
        compute_metrics = compute_metrics,
        optimizers      = (optimizer, scheduler),
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
    )

    trainer.train()
    print("Phase-2 (joint) finished.")
    return trainer

# ------------------------------------------------------------------ #
# Routine per topic                                                  #
# ------------------------------------------------------------------ #
def run_for_topic(csv_file: str, out_dir: str,
                  joint_epochs: int, joint_batch: int):
    df = pd.read_csv(csv_file, low_memory=False)
    df["pair_key"] = df["U.S.did"].astype(str) + "_" + df["U.R.did"].astype(str)
    df = df.drop(columns=[c for c in EXCL_COLS if c in df.columns], errors="ignore")

    cand_num = [c for c in df.columns if NUM_RE.match(c)]
    num_df   = df[cand_num].apply(pd.to_numeric, errors="coerce")
    for col in cand_num:
        if num_df[col].notna().any():
            num_df[col] = num_df[col].fillna(num_df[col].median())
        else:
            num_df[col] = 0.0
    df[cand_num] = num_df.astype(np.float32)
    num_cols = cand_num

    df["U_text"]   = concat(df, [c for c in U_TEXT_COLS   if c in df.columns])
    df["HMS_text"] = concat(df, [c for c in HMS_TEXT_COLS if c in df.columns])
    df["HMR_text"] = concat(df, [c for c in HMR_TEXT_COLS if c in df.columns])

    df = df.dropna(subset=[LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    def get_max_token_stats(text_series: pd.Series, tokenizer):
        if text_series.empty: return 0, -1
        tokenized_output = tokenizer(text_series.tolist(), truncation=False, padding=False)["input_ids"]
        token_lengths = [len(ids) for ids in tokenized_output]
        if not token_lengths: return 0, -1
        max_pos_idx = np.argmax(token_lengths)
        return token_lengths[max_pos_idx], text_series.index[max_pos_idx]

    print("\nCalculating max token lengths and their locations...")
    U_max, U_idx = get_max_token_stats(df["U_text"], tok)
    HMS_max, HMS_idx = get_max_token_stats(df["HMS_text"], tok)
    HMR_max, HMR_idx = get_max_token_stats(df["HMR_text"], tok)
    print(f"[STATS] U_text max tokens   = {U_max} (at DataFrame index: {U_idx})")
    print(f"[STATS] HMS_text max tokens = {HMS_max} (at DataFrame index: {HMS_idx})")
    print(f"[STATS] HMR_text max tokens = {HMR_max} (at DataFrame index: {HMR_idx})")
    print("------------------------------------------------------------")

    os.makedirs(out_dir, exist_ok=True)
    results = []

    for run in range(N_RUNS):
        seed = BASE_SEED + run
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        print(f"\n=== Run {run + 1}/{N_RUNS} (seed {seed}) ===")

        df_trainval, df_test = train_test_split(
            df, test_size=0.20, stratify=df[LABEL_COL], random_state=seed
        )
        df_train, df_val = train_test_split(
            df_trainval, test_size=0.125, stratify=df_trainval[LABEL_COL], random_state=seed
        )
        test_pairs = set(df_test["pair_key"].unique())
        df_train = df_train[~df_train["pair_key"].isin(test_pairs)]
        df_val   = df_val[~df_val["pair_key"].isin(test_pairs)]

        scaler = MinMaxScaler().fit(df_train[num_cols])
        Xtr, Xva, Xte = (
            scaler.transform(d[num_cols]).astype(np.float32)
            for d in (df_train, df_val, df_test)
        )

        def max_len(series):
            if series.empty: return 0
            return min(max(len(x) for x in tok(series.tolist(), padding=False)["input_ids"]), 512)
        L_U, L_S, L_R = map(max_len, (df_train["U_text"], df_train["HMS_text"], df_train["HMR_text"]))

        def encode(series, ml):
            out = tok(series.tolist(), truncation=True, padding="max_length", max_length=ml)
            return out["input_ids"], out["attention_mask"]

        idsU_tr, maskU_tr = encode(df_train["U_text"], L_U)
        idsS_tr, maskS_tr = encode(df_train["HMS_text"], L_S)
        idsR_tr, maskR_tr = encode(df_train["HMR_text"], L_R)

        idsU_va, maskU_va = encode(df_val["U_text"], L_U)
        idsS_va, maskS_va = encode(df_val["HMS_text"], L_S)
        idsR_va, maskR_va = encode(df_val["HMR_text"], L_R)

        idsU_te, maskU_te = encode(df_test["U_text"], L_U)
        idsS_te, maskS_te = encode(df_test["HMS_text"], L_S)
        idsR_te, maskR_te = encode(df_test["HMR_text"], L_R)

        ds_tr = DictDS(idsU_tr, maskU_tr, idsS_tr, maskS_tr, idsR_tr, maskR_tr, Xtr, df_train[LABEL_COL])
        ds_va = DictDS(idsU_va, maskU_va, idsS_va, maskS_va, idsR_va, maskR_va, Xva, df_val[LABEL_COL])
        ds_te = DictDS(idsU_te, maskU_te, idsS_te, maskS_te, idsR_te, maskR_te, Xte, df_test[LABEL_COL])

        model = TripleBert(len(num_cols))
        trainer1 = train_mlp_only(model, ds_tr, ds_va, out_dir=out_dir, seed=seed)
        trainer2 = train_joint(
            model, ds_tr, ds_va, joint_epochs=joint_epochs,
            joint_batch=joint_batch, out_dir=out_dir, seed=seed
        )

        res = trainer2.evaluate(ds_te)
        print(f"Run {run + 1}: F1={res['eval_f1']:.4f}  Acc={res['eval_accuracy']:.4f}")
        results.append(res)
        torch.cuda.empty_cache()

    f1s  = [r["eval_f1"] for r in results]
    accs = [r["eval_accuracy"] for r in results]
    summary = dict(
        n_runs   = N_RUNS, f1_all   = f1s, acc_all  = accs,
        f1_mean  = float(np.mean(f1s)), f1_std   = float(np.std(f1s)),
        acc_mean = float(np.mean(accs)), acc_std  = float(np.std(accs)),
    )
    with open(f"{out_dir}/overall_metrics.json", "w") as fp:
        json.dump(summary, fp, indent=4)

    print("\n=== Monte-Carlo Summary ===")
    # MODIFIED: Changed label from "Macro-F1" to "Binary F1"
    print(f"Binary F1: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"Accuracy : {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}")
    print(f"✅  Metrics saved to {out_dir}/overall_metrics.json")

# ------------------------------------------------------------------ #
# CLI                                                                #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Two-phase Triple-BERT trainer (Binary F1)")
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="List of CSV files (e.g., aew_1to5.csv)")
    ap.add_argument("--joint_epochs", type=int, default=EPOCHS_JOINT_DEF,
                    help=f"Joint phase epochs (default: {EPOCHS_JOINT_DEF})")
    ap.add_argument("--joint_batch", type=int, default=JOINT_BATCH_DEF,
                    help=f"Joint phase batch size (default: {JOINT_BATCH_DEF})")
    args = ap.parse_args()

    for csv_path in args.csvs:
        topic   = pathlib.Path(csv_path).stem.split("_")[0]
        out_dir = f"bert_{topic}_U"
        run_for_topic(csv_path, out_dir,
                      joint_epochs=args.joint_epochs,
                      joint_batch=args.joint_batch)