#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Message-BERT — Single-Phase Fine-Tuning (Binary F1)
==================================================
• Fine-tune BERT together with a linear classification head
• Uses a fixed 0.5 threshold for predictions (via argmax)
• Reports standard Binary F1 score
CLI example
-----------
python message_only.py \
    --csvs aew_1to5.csv \
    --epochs 3 \
    --batch 32
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
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast, BertModel,
    TrainingArguments, Trainer, EarlyStoppingCallback, get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import evaluate
import ftfy
import unidecode
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------ #
# Hyper-parameters (defaults)                                        #
# ------------------------------------------------------------------ #
LABEL_COL         = "is_positive"
N_RUNS            = 1
BASE_SEED         = 42

EPOCHS_DEF        = 3          # CLI override: --epochs
BATCH_DEF         = 32         # CLI override: --batch
LR_BERT           = 5e-5
WD_BERT           = 0.01
PATIENCE          = 8

# ------------------------------------------------------------------ #
# Column definitions                                                 #
# ------------------------------------------------------------------ #
MESSAGE_TEXT_COLS = [
    "M.record.text",
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
NUM_RE    = re.compile(r"^M\.")

# ------------------------------------------------------------------ #
# Tokenizer & text utilities                                         #
# ------------------------------------------------------------------ #
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

def concat(df: pd.DataFrame, cols: list[str], sep: str = FIELD_SEP) -> pd.Series:
    """Concatenate selected columns and clean the result."""
    return (
        df[cols].fillna("").astype(str)
          .agg(f" {sep} ".join, axis=1)
          .map(clean_txt)
    )

# ------------------------------------------------------------------ #
# Dataset                                                             #
# ------------------------------------------------------------------ #
class MessageDS(Dataset):
    """Dataset that returns dicts for HuggingFace Trainer."""
    def __init__(self, ids, masks, labels):
        self.ids   = torch.tensor(ids)
        self.masks = torch.tensor(masks)
        self.lab   = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, idx):
        return {
            "input_ids"      : self.ids[idx],
            "attention_mask" : self.masks[idx],
            "labels"         : self.lab[idx]
        }

# ------------------------------------------------------------------ #
# Model                                                               #
# ------------------------------------------------------------------ #
class BertClassifier(nn.Module):
    """BERT encoder with dropout and linear classifier."""
    def __init__(self):
        super().__init__()
        self.bert  = BertModel.from_pretrained("bert-base-uncased")
        hidden     = self.bert.config.hidden_size
        self.drop  = nn.Dropout(0.1)
        self.cls   = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask, labels=None):
        pooled = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output
        logits = self.cls(self.drop(pooled))
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

# ------------------------------------------------------------------ #
# Metrics                                                             #
# ------------------------------------------------------------------ #
metric_acc = evaluate.load("accuracy")
metric_f1  = evaluate.load("f1", keep_in_memory=True)

def compute_metrics(pred):
    """Computes accuracy and standard binary F1 for the positive class."""
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        # MODIFIED: Changed from 'macro' to 'binary'
        "f1"      : metric_f1.compute(predictions=preds, references=labels, average="binary")["f1"],
    }

# ------------------------------------------------------------------ #
# Training helper                                                    #
# ------------------------------------------------------------------ #
def train_single_phase(model, ds_tr, ds_va,
                       epochs, batch_size,
                       out_dir, seed):
    """Fine-tune BERT with AdamW + linear scheduler + early stopping."""
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params"       : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay" : WD_BERT,
        },
        {
            "params"       : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay" : 0.0,
        }
    ]

    optimizer = AdamW(param_groups, lr=LR_BERT)
    steps_per_epoch = math.ceil(len(ds_tr) / batch_size)
    total_steps     = epochs * steps_per_epoch
    warmup_steps    = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    args = TrainingArguments(
        output_dir             = f"{out_dir}/finetune",
        num_train_epochs       = epochs,
        per_device_train_batch_size = batch_size,
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
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    trainer.train()
    print("Fine-tuning finished.")
    return trainer

# ------------------------------------------------------------------ #
# Routine per topic                                                  #
# ------------------------------------------------------------------ #
def run_for_topic(csv_file: str, out_dir: str,
                  epochs: int, batch_size: int):
    df = pd.read_csv(csv_file, low_memory=False)
    df = df.drop(columns=[c for c in EXCL_COLS if c in df.columns], errors="ignore")

    text_cols_present = [c for c in MESSAGE_TEXT_COLS if c in df.columns]
    df["MSG_text"]    = concat(df, text_cols_present)
    df = df.dropna(subset=[LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    def get_max_token_stats(text_series, tokenizer):
        """Return maximum token length and DataFrame index."""
        tok_out       = tokenizer(text_series.tolist(), truncation=False, padding=False)["input_ids"]
        token_lengths = [len(ids) for ids in tok_out]
        if not token_lengths: return 0, -1
        max_idx       = int(np.argmax(token_lengths))
        return token_lengths[max_idx], text_series.index[max_idx]

    print("\nCalculating max token length (this may take a moment)...")
    true_max_len, max_len_idx = get_max_token_stats(df["MSG_text"], tok)
    print(f"[STATS] MSG_text max tokens = {true_max_len} (index {max_len_idx})")
    print("------------------------------------------------------------")

    max_len = min(true_max_len, 512)

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
            df_trainval, test_size=0.125,
            stratify=df_trainval[LABEL_COL], random_state=seed
        )

        def encode(series, length):
            out = tok(series.tolist(), truncation=True, padding="max_length", max_length=length)
            return out["input_ids"], out["attention_mask"]

        ids_tr, mask_tr = encode(df_train["MSG_text"], max_len)
        ids_va, mask_va = encode(df_val["MSG_text"],   max_len)
        ids_te, mask_te = encode(df_test["MSG_text"],  max_len)

        ds_tr = MessageDS(ids_tr, mask_tr, df_train[LABEL_COL])
        ds_va = MessageDS(ids_va, mask_va, df_val[LABEL_COL])
        ds_te = MessageDS(ids_te, mask_te, df_test[LABEL_COL])

        model   = BertClassifier()
        trainer = train_single_phase(
            model, ds_tr, ds_va,
            epochs=epochs,
            batch_size=batch_size,
            out_dir=out_dir,
            seed=seed
        )

        res = trainer.evaluate(ds_te)
        print(f"Run {run + 1}: F1={res['eval_f1']:.4f}  Acc={res['eval_accuracy']:.4f}")
        results.append(res)
        torch.cuda.empty_cache()

    f1s  = [r["eval_f1"]       for r in results]
    accs = [r["eval_accuracy"] for r in results]
    summary = {
        "n_runs"  : N_RUNS, "f1_all"  : f1s, "acc_all" : accs,
        "f1_mean" : float(np.mean(f1s)), "f1_std"  : float(np.std(f1s)),
        "acc_mean": float(np.mean(accs)), "acc_std" : float(np.std(accs)),
    }
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
    ap = argparse.ArgumentParser(description="Message-BERT single-phase fine-tuner (Binary F1)")
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="List of CSV files (e.g., aew_1to5.csv)")
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEF,
                    help=f"Fine-tuning epochs (default: {EPOCHS_DEF})")
    ap.add_argument("--batch", type=int, default=BATCH_DEF,
                    help=f"Batch size (default: {BATCH_DEF})")
    args = ap.parse_args()

    for csv_path in args.csvs:
        topic   = pathlib.Path(csv_path).stem.split("_")[0]
        out_dir = f"bert_{topic}_simple"
        run_for_topic(csv_path, out_dir,
                      epochs=args.epochs,
                      batch_size=args.batch)