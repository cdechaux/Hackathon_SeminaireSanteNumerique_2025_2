#!/usr/bin/env python3
# train_finetune_dp.py
"""
Fine-tuning DP (single-label) avec Transformers + Trainer custom (poids de classes).
- Modèle HF standard (AutoModelForSequenceClassification) => save_pretrained OK
- Pondération des classes dans `compute_loss()`
- Split "par classe" robuste (évite l'erreur de stratification quand classe rare)
- Sauvegarde: checkpoints/, final/ (config.json, model.safetensors, tokenizer*), label_map.json
"""

from __future__ import annotations
import argparse, json, math, os, random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# ----------------- Utils -----------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def per_class_split(
    df: pd.DataFrame, label_col: str, val_frac: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split "par classe" robuste :
      - si une classe a 1 seul échantillon: tout en train (0 en val)
      - si >=2: on met au moins 1 en val (si possible), le reste en train
    """
    rng = np.random.default_rng(seed)
    groups = []
    for label, sub in df.groupby(label_col):
        idx = sub.index.to_list()
        rng.shuffle(idx)
        n = len(idx)
        if n == 1:
            train_idx, val_idx = idx, []
        else:
            n_val = max(1, int(round(n * val_frac)))
            n_val = min(n_val, n - 1)  # laisse au moins 1 en train
            val_idx = idx[:n_val]; train_idx = idx[n_val:]
        groups.append((train_idx, val_idx))
    train_ids = [i for tr, _ in groups for i in tr]
    val_ids   = [i for _, va in groups for i in va]
    train_df = df.loc[train_ids].sample(frac=1, random_state=seed)
    val_df   = df.loc[val_ids].sample(frac=1, random_state=seed)
    return train_df, val_df

class SimpleTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class WeightedTrainer(Trainer):
    """Applique des poids de classes dans la cross-entropy."""
    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights  # tensor sur le bon device dans training_step
        self._loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # [B, C]
        if self.class_weights is not None:
            # déplacer sur device du logits
            cw = self.class_weights.to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=cw, reduction="mean")
        else:
            loss = self._loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ----------------- Main -----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True, help="CSV avec colonnes texte & label")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="code_dp")
    p.add_argument("--output-dir", default="checkpoints/camembert_dp_ft")
    p.add_argument("--pretrained", default="almanach/camembert-bio-base")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # 1) Data
    df = pd.read_csv(args.input_csv)
    assert args.text_col in df.columns and args.label_col in df.columns, "Cols manquantes dans le CSV"
    df = df[[args.text_col, args.label_col]].dropna().reset_index(drop=True)

    # 2) Label map (fixe, pour reproducibilité)
    labels_sorted = sorted(df[args.label_col].astype(str).unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels_sorted)}
    id2label = {i: lab for lab, i in label2id.items()}
    save_json(label2id, out / "label_map.json")

    # 3) Split "par classe" (robuste aux classes rares)
    train_df, val_df = per_class_split(df, args.label_col, args.val_frac, args.seed)
    train_df.to_csv(out / "data/train.csv", index=False)
    val_df.to_csv(out / "data/val.csv", index=False)

    # 4) Tokenizer + Datasets
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, use_fast=True)
    train_ds = SimpleTextDataset(
        texts=train_df[args.text_col].astype(str).tolist(),
        labels=[label2id[x] for x in train_df[args.label_col].astype(str).tolist()],
        tokenizer=tokenizer, max_length=args.max_length,
    )
    eval_ds = SimpleTextDataset(
        texts=val_df[args.text_col].astype(str).tolist(),
        labels=[label2id[x] for x in val_df[args.label_col].astype(str).tolist()],
        tokenizer=tokenizer, max_length=args.max_length,
    ) if len(val_df) else None

    # 5) Config + Model (HF standard)
    config = AutoConfig.from_pretrained(
        args.pretrained,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained, config=config
    )

    # 6) Poids de classes (inverse freq sur le TRAIN)
    counts = train_df[args.label_col].value_counts().reindex(labels_sorted, fill_value=0)
    freq = counts.values.astype(np.float32)
    inv = 1.0 / np.maximum(freq, 1.0)
    class_weights = torch.tensor(inv / inv.mean(), dtype=torch.float32)  # normalisés

    # 7) TrainingArguments
    train_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_every if eval_ds is not None else None,
        logging_steps=100,
        save_steps=args.eval_every if eval_ds is not None else 1000,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=bool(eval_ds is not None),
        metric_for_best_model="eval_loss",
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=[],
    )

    # 8) Data collator (padding dynamique)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 9) Trainer custom (compute_loss pondéré)
    trainer = WeightedTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        class_weights=class_weights,
    )

    # 10) Train
    trainer.train()

    # 11) Sauvegardes finales (format HF complet)
    final_dir = out / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))       # => config.json + model.safetensors
    tokenizer.save_pretrained(str(final_dir))

    # 12) Sauvegarde config entraînement
    cfg = vars(args).copy()
    cfg["n_train"] = len(train_ds)
    cfg["n_val"] = 0 if eval_ds is None else len(eval_ds)
    save_json(cfg, out / "config_train.json")

    # 13) Stats split
    split_stats = {
        "train_counts": train_df[args.label_col].value_counts().to_dict(),
        "val_counts":   (val_df[args.label_col].value_counts().to_dict() if len(val_df) else {}),
        "labels": labels_sorted,
    }
    save_json(split_stats, out / "split_stats.json")

    print(f"[OK] Fine-tuning terminé. Modèle + tokenizer sauvegardés dans: {final_dir}")

if __name__ == "__main__":
    main()
