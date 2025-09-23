#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning DP (mono-label) avec split **par classe**.

Option 3 (recommandée si certaines classes sont rares):
- Pour chaque classe c:
  * si n_c == 1  -> sample unique en TRAIN (exclu de la val)
  * si n_c >= 2  -> au moins 1 échantillon en VAL (reste en TRAIN)
- Respecte approx. une fraction globale --val-frac (par défaut 0.1) sans vider le train.

Sauvegarde:
- output_dir/
    - label_map.json
    - train.csv / val.csv
    - config_train.json
    - checkpoints/ ...

Exemple:
python train_finetune_dp.py \
  --input-csv dp_dataset.csv \
  --text-col text --label-col code_dp \
  --output-dir checkpoints/camembert_dp_ft \
  --pretrained almanach/camembert-bio-base \
  --epochs 3 --batch-size 8 --lr 2e-5 --fp16
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from torch import nn
from torch.utils.data import Dataset
from dataclasses import dataclass


# =============== Utils I/O ===============

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============== Datasets HF ===============

class TextLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, label_col: str, label2id: Dict[str, int], tokenizer, max_length: int = 512):
        self.texts = df[text_col].astype(str).tolist()
        self.labels = [label2id[l] for l in df[label_col].tolist()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        txt = self.texts[i]
        enc = self.tokenizer(
            txt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        enc["labels"] = self.labels[i]
        return enc


# =============== Model wrapper avec class weights ===============

class WeightedCELossModel(nn.Module):
    """
    Enveloppe AutoModelForSequenceClassification pour appliquer une CrossEntropy pondérée.
    """
    def __init__(self, base_model: AutoModelForSequenceClassification, class_weights: torch.Tensor | None):
        super().__init__()
        self.base = base_model
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, **kwargs):
        labels = kwargs.get("labels", None)
        outputs = self.base(**{k: v for k, v in kwargs.items() if k != "labels"})
        logits = outputs.logits
        loss = None
        if labels is not None:
            if self.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


# =============== Split par classe (Option 3) ===============

def split_by_class(
    df: pd.DataFrame,
    label_col: str,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Split manuel par classe:
      - si n_c == 1: tout en TRAIN
      - si n_c >= 2: au moins 1 en VAL; on vise ~val_frac globalement, mais sans vider train
    Retourne (train_df, val_df, stats)
    """
    rng = np.random.default_rng(seed)
    classes = df[label_col].unique().tolist()
    grouped = {c: df[df[label_col] == c].sample(frac=1.0, random_state=seed) for c in classes}

    total = len(df)
    target_val = int(round(val_frac * total))
    picked_val = 0

    train_rows = []
    val_rows = []

    stats = {}

    for c, g in grouped.items():
        n = len(g)
        if n == 1:
            # train only
            train_rows.append(g)
            stats[c] = {"train": 1, "val": 0, "total": 1}
            continue

        # nombre cible pour cette classe si on proportionne sur val_frac
        cand_val = int(math.floor(n * val_frac))
        cand_val = max(cand_val, 1)  # au moins 1 en val
        cand_val = min(cand_val, n - 1)  # laisser au moins 1 en train

        # Ajustement global: si on a déjà dépassé la cible globale, on réduit localement
        # (sans descendre en dessous de 1 pour n>=2)
        if picked_val + cand_val > target_val:
            # reste possible globalement
            remaining = max(target_val - picked_val, 0)
            if remaining >= 1:
                cand_val = max(1, min(cand_val, remaining))
            # sinon si remaining == 0, mettre le minimum (1), mais ça dépassera légèrement la cible globale
            # on garde 1 pour respecter la règle "au moins 1 en val"
        val_rows.append(g.iloc[:cand_val])
        train_rows.append(g.iloc[cand_val:])

        picked_val += cand_val
        stats[c] = {"train": n - cand_val, "val": cand_val, "total": n}

    train_df = pd.concat(train_rows, axis=0, ignore_index=True)
    val_df = pd.concat(val_rows, axis=0, ignore_index=True) if len(val_rows) else pd.DataFrame(columns=df.columns)

    # Shuffle final
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Rapport synthétique
    nb_val_zero = sum(1 for c, s in stats.items() if s["val"] == 0)
    print(f"[SPLIT] Total={total} | target_val≈{target_val} | picked_val={picked_val} | classes sans val={nb_val_zero}")

    return train_df, val_df, stats


# =============== Metrics ===============

def compute_metrics(eval_pred, id2label: Dict[int, str]):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_micro": f1_micro}


# =============== Main ===============

def main():
    parser = argparse.ArgumentParser("Fine-tune DP (option 3: split par classe)")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--text-col", required=True, help="Nom de la colonne texte")
    parser.add_argument("--label-col", required=True, help="Nom de la colonne DP (mono-label)")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--pretrained", default="almanach/camembert-bio-base")
    parser.add_argument("--max-length", type=int, default=512)

    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction globale visée pour la validation")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--weighting", choices=["none", "inverse_freq", "effective_num"], default="inverse_freq",
                        help="Pondération de la CE: none | inverse_freq | effective_num")

    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--grad-accum", type=int, default=1)

    args = parser.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Lecture
    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(f"Colonnes manquantes. Trouvé: {df.columns.tolist()}")

    # Nettoyage basique
    df = df[[args.text_col, args.label_col]].dropna().reset_index(drop=True)
    df[args.text_col] = df[args.text_col].astype(str).str.strip()
    df[args.label_col] = df[args.label_col].astype(str).str.strip()

    # 2) Split par classe (option 3)
    train_df, val_df, stats = split_by_class(df, args.label_col, val_frac=args.val_frac, seed=args.seed)

    # 3) Label map (sur TOUT l'ensemble)
    labels_sorted = sorted(df[args.label_col].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels_sorted)}
    id2label = {i: lab for lab, i in label2id.items()}

    save_json({"label2id": label2id, "id2label": id2label}, out / "label_map.json")
    save_json({"split_stats": stats}, out / "split_stats.json")
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)

    # 4) Tokenizer & config/model
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, use_fast=True)
    config = AutoConfig.from_pretrained(
        args.pretrained,
        num_labels=len(label2id),
        id2label={i: l for i, l in enumerate(labels_sorted)},
        label2id={l: i for i, l in enumerate(labels_sorted)},
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(args.pretrained, config=config)

    # 5) Poids de classe
    class_weights = None
    if args.weighting != "none":
        # comptages sur TRAIN uniquement
        counts = train_df[args.label_col].value_counts().reindex(labels_sorted, fill_value=0).astype(float).values
        if args.weighting == "inverse_freq":
            w = 1.0 / np.maximum(counts, 1.0)
            w = (w / w.mean()).astype(np.float32)
        else:
            # effective number (Cui et al. 2019)
            beta = 0.999
            effective_num = 1.0 - np.power(beta, counts)
            w = (1.0 - beta) / np.maximum(effective_num, 1e-9)
            w = (w / w.mean()).astype(np.float32)
        class_weights = torch.tensor(w)

        print("[CLASS WEIGHTS]", dict(zip(labels_sorted, w.round(3))))

    model = WeightedCELossModel(base_model, class_weights)

    # 6) Datasets
    train_ds = TextLabelDataset(train_df, args.text_col, args.label_col, label2id, tokenizer, args.max_length)
    eval_ds = TextLabelDataset(val_df, args.text_col, args.label_col, label2id, tokenizer, args.max_length) if len(val_df) else None

    # 7) TrainingArguments
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=args.fp16,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True if eval_ds is not None else False,
        metric_for_best_model="f1_macro" if eval_ds is not None else None,
        warmup_ratio=args.warmup_ratio,
        report_to="none",
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 8) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=(lambda p: compute_metrics(p, id2label)) if eval_ds is not None else None,
    )

    # 9) Fit
    print(f"[INFO] Train size={len(train_ds)} | Val size={0 if eval_ds is None else len(eval_ds)} | Labels={len(label2id)}")
    trainer.train()

    # 10) Sauvegardes finales
    (out / "final").mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out / "final"))  # config + model
    tokenizer.save_pretrained(str(out / "final"))

    cfg_dump = vars(args).copy()
    cfg_dump["n_train"] = len(train_ds)
    cfg_dump["n_val"] = 0 if eval_ds is None else len(eval_ds)
    save_json(cfg_dump, out / "config_train.json")

    print(f"[OK] Entraînement terminé. Modèle + tokenizer enregistrés dans: {out / 'final'}")


if __name__ == "__main__":
    main()
