#!/usr/bin/env python3
# create_rewrite_dataset.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from medkit.core.pipeline import Pipeline, PipelineStep
from medkit.core.text import TextDocument

# importe tes opérations
from operations import RewriteOp, RewriteConfig, MetricsTextOp, MetricsTextConfig

METRIC_KEYS = ["len_chars", "len_words", "sent_len_avg", "lexicon_size", "n_sections", "n_abbr"]

def save_json(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build_pipeline(args) -> Pipeline:
    steps = []
    # Metrics BEFORE
    steps.append(PipelineStep(
        MetricsTextOp(MetricsTextConfig(
            text_field=args.text_col,   # lit directement la colonne d'entrée
            metrics_root="metrics",
            phase="before"
        )),
        ["docs"], ["docs_b"]
    ))
    # Rewrite
    steps.append(PipelineStep(
        RewriteOp(RewriteConfig(
            enabled=True,
            target_words=args.target_words,
            llm_model=args.llm_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )),
        ["docs_b"], ["docs_r"]
    ))
    # Metrics AFTER (sur text_rw)
    steps.append(PipelineStep(
        MetricsTextOp(MetricsTextConfig(
            text_field="text_rw",
            metrics_root="metrics",
            phase="after"
        )),
        ["docs_r"], ["docs_out"]
    ))
    return Pipeline(steps=steps, input_keys=["docs"], output_keys=["docs_out"], name="rewrite_metrics")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--out-text-col", default="text_rw")  # nom de colonne du texte réécrit dans le CSV de sortie

    # Rewrite params
    ap.add_argument("--llm-model", required=True, help="Modèle HF causal (ex: mistralai/Mistral-7B-Instruct-v0.3)")
    ap.add_argument("--target-words", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top_p", type=float, default=0.95)

    # Fichiers de métriques
    ap.add_argument("--metrics-json", default=None, help="Chemin du JSON récapitulatif (default: <output>.metrics.json)")
    args = ap.parse_args()

    inp = Path(args.input_csv)
    out_csv = Path(args.output_csv)
    metrics_json = Path(args.metrics_json) if args.metrics_json else out_csv.with_suffix(".metrics.json")

    df = pd.read_csv(inp)
    assert args.text_col in df.columns, f"Colonne '{args.text_col}' absente de {inp}"

    # Construire docs
    docs = []
    for i, row in df.iterrows():
        txt = str(row[args.text_col]) if pd.notna(row[args.text_col]) else ""
        d = TextDocument(text=txt)
        # pour Metrics(before), on veut lire text_col depuis metadata (sinon il lira d.text)
        d.metadata[args.text_col] = txt
        docs.append(d)

    # Pipeline
    pipe = build_pipeline(args)
    docs = pipe.run(docs)

    # Construire nouveau DataFrame
    out_rows = []
    agg_before = {k: 0.0 for k in METRIC_KEYS}
    agg_after  = {k: 0.0 for k in METRIC_KEYS}

    for src_row, d in zip(df.to_dict("records"), docs):
        metrics = d.metadata.get("metrics", {})
        before = metrics.get("before", {})
        after  = metrics.get("after", {})
        rw_txt = d.metadata.get("text_rw", d.text)

        row = dict(src_row)  # conserve toutes les colonnes originelles
        row[args.out_text_col] = rw_txt

        # Ajoute colonnes métriques _before/_after
        for k in METRIC_KEYS:
            row[f"{k}_before"] = before.get(k, None)
            row[f"{k}_after"]  = after.get(k, None)
            if before.get(k) is not None:
                agg_before[k] += float(before[k])
            if after.get(k) is not None:
                agg_after[k] += float(after[k])

        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)

    n = len(out_rows) if out_rows else 1
    means_before = {k: round(agg_before[k] / n, 3) for k in METRIC_KEYS}
    means_after  = {k: round(agg_after[k] / n, 3) for k in METRIC_KEYS}

    recap = {
        "input_csv": str(inp),
        "output_csv": str(out_csv),
        "n_rows": len(out_rows),
        "rewrite": {
            "llm_model": args.llm_model,
            "target_words": args.target_words,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "metrics_means": {
            "before": means_before,
            "after": means_after,
        }
    }
    save_json(recap, metrics_json)

    print(f"[OK] Nouveau dataset écrit: {out_csv}")
    print(f"[OK] Récap métriques: {metrics_json}")
    print("Moyennes BEFORE:", means_before)
    print("Moyennes AFTER :", means_after)

if __name__ == "__main__":
    main()
