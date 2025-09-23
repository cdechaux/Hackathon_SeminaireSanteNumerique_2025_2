#!/usr/bin/env python3
# train_finetune_dp.py
from __future__ import annotations
import argparse, os, json, math, random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments)
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# --------------------------- Helpers ---------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_label_map(codes: List[str]) -> Tuple[Dict[str,int], Dict[int,str]]:
    uniq = sorted(set(codes))
    label2id = {c:i for i,c in enumerate(uniq)}
    id2label = {i:c for c,i in label2id.items()}
    return label2id, id2label

def chunk_encode(
    texts: List[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    stride: int
):
    """Tokenize en autorisant le débordement (overflow) pour générer des chunks."""
    enc = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_overflowing_tokens=True,
        stride=stride,
    )
    return enc

# --------------------------- Main ---------------------------

def main():
    p = argparse.ArgumentParser("Fine-tune DP classifier (HF) with chunking + doc-level eval")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="code_dp")
    p.add_argument("--output-dir", required=True)  # dossier HF à créer (checkpoint)
    p.add_argument("--pretrained", default="almanach/camembert-bio-base")

    # chunking/tokenization
    p.add_argument("--max-length", type=int, default=384)       # en tokens
    p.add_argument("--stride", type=int, default=64)            # en tokens

    # training
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--eval-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")

    # class imbalance
    p.add_argument("--class-weights", action="store_true", help="Active weighted CE (inverse freq)")

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # --------- Load data ----------
    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(f"Colonnes attendues absentes. Trouvées: {df.columns.tolist()}")

    # Label mapping
    label2id, id2label = build_label_map(df[args.label_col].tolist())

    # Split train/eval stratifié sur DP
    train_df, eval_df = train_test_split(
        df, test_size=args.eval_size, random_state=args.seed,
        stratify=df[args.label_col]
    )
    train_df = train_df.reset_index(drop=True)
    eval_df  = eval_df.reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, use_fast=True)

    # On stocke doc_ids pour l'éval doc-level
    train_doc_ids = list(range(len(train_df)))
    eval_doc_ids  = list(range(len(eval_df)))

    # --------- Build chunked datasets ----------
    def build_chunked(ds_df: pd.DataFrame, doc_ids: List[int]):
        texts = ds_df[args.text_col].astype(str).tolist()
        labels = [label2id[c] for c in ds_df[args.label_col].tolist()]

        enc = chunk_encode(texts, tokenizer, args.max_length, args.stride)

        # map overflow_to_sample_mapping pour savoir quel chunk vient de quel doc
        doc_map = enc.pop("overflow_to_sample_mapping")
        # doc_id par chunk + label par chunk
        chunk_doc_ids = [doc_ids[i] for i in doc_map]
        chunk_labels  = [labels[i]  for i in doc_map]

        # dataset HF
        data_dict = {k: enc[k] for k in ["input_ids", "attention_mask"]}
        data_dict["labels"]   = chunk_labels
        data_dict["doc_id"]   = chunk_doc_ids   # pour compute_metrics
        return Dataset.from_dict(data_dict), doc_ids

    train_hf, _  = build_chunked(train_df, train_doc_ids)
    eval_hf,  _  = build_chunked(eval_df,  eval_doc_ids)

    dsd = DatasetDict({"train": train_hf, "validation": eval_hf})

    # --------- Model ----------
    num_labels = len(label2id)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained,
        num_labels=num_labels,
        id2label={i: id2label[i] for i in range(num_labels)},
        label2id={k: v for k, v in label2id.items()},
        problem_type="single_label_classification",
    )

    # Option: class weights
    ce_weight = None
    if args.class_weights:
        # compute weights on train set labels
        y = np.array(train_hf["labels"])
        uniq, counts = np.unique(y, return_counts=True)
        inv_freq = {u: (1.0 / c) for u, c in zip(uniq, counts)}
        # normalize
        s = sum(inv_freq.values())
        inv_freq = {k: v / s * len(inv_freq) for k, v in inv_freq.items()}
        weight_vec = np.zeros(num_labels, dtype=np.float32)
        for k,v in inv_freq.items():
            weight_vec[k] = v
        ce_weight = torch.tensor(weight_vec)

        # wrap default loss fn inside trainer later (via model hook)
        def compute_loss_with_weights(model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=ce_weight.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # --------- Metrics (doc-level) ----------
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred.predictions, eval_pred.label_ids   # chunk-level
        # On récupère doc_ids dans l'ordre de l'eval set
        doc_ids = eval_hf["doc_id"]

        # Agrégation des logits par doc_id
        # (on fait une moyenne simple; max/median/attention-pooling possible)
        by_doc: Dict[int, List[np.ndarray]] = {}
        for logit, did in zip(logits, doc_ids):
            by_doc.setdefault(did, []).append(logit)

        y_true_doc, y_pred_doc = [], []
        for did, chunks in by_doc.items():
            agg = np.mean(np.stack(chunks, axis=0), axis=0)
            pred = int(np.argmax(agg))
            # label "or" au niveau doc : on récupère le label du doc via le premier chunk trouvé
            # (tous les chunks d'un doc partagent le même label ici)
            # On prend le label correspondant au premier chunk du doc:
            idx_first_chunk = next(i for i, d in enumerate(doc_ids) if d == did)
            gold = int(labels[idx_first_chunk])
            y_true_doc.append(gold); y_pred_doc.append(pred)

        p, r, f1, _ = precision_recall_fscore_support(y_true_doc, y_pred_doc, average="macro", zero_division=0)
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_true_doc, y_pred_doc, average="micro", zero_division=0)

        return {
            "macro_p": float(p),
            "macro_r": float(r),
            "macro_f1": float(f1),
            "micro_p": float(p_micro),
            "micro_r": float(r_micro),
            "micro_f1": float(f1_micro),
        }

    # --------- Trainer ----------
    args_train = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=50,
        save_total_limit=2,
        dataloader_num_workers=2,
        report_to=[],  # pas de wandb par défaut
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if args.class_weights:
        # hack léger pour injecter la loss pondérée
        trainer.compute_loss = compute_loss_with_weights  # type: ignore

    # --------- Train ----------
    trainer.train()

    # --------- Save (config contient id2label/label2id) ----------
    trainer.save_model(args.output_dir)   # model + config
    tokenizer.save_pretrained(args.output_dir)

    # Sauvegarde des colonnes utilisées (pratique pour l'inf)
    meta = {
        "text_col": args.text_col,
        "label_col": args.label_col,
        "id2label": {str(k): v for k, v in id2label.items()},
        "label2id": label2id,
        "max_length": args.max_length,
        "stride": args.stride,
    }
    with open(os.path.join(args.output_dir, "dp_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Checkpoint sauvegardé dans: {args.output_dir}")

if __name__ == "__main__":
    main()
