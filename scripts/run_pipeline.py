#!/usr/bin/env python3
# run_pipeline.py
# Pipeline medkit complet (DP uniquement) avec backend Transformers :
#  - Normalize → (Rewrite?) → Chunk → Embed → (TransformerDPHeadOp | LLMDPInferenceOp)
# Entrée : CSV avec colonnes configurables (code, text, code_patient, code_sejour)
# Sortie : CSV minimal (code_sejour, dp_predit) avec noms de colonnes configurables.

from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from medkit.core.pipeline import Pipeline, PipelineStep
from medkit.core.text import TextDocument

from operations import (
    NormalizeConfig, NormalizeOp,
    RewriteConfig, RewriteOp,
    ChunkingConfig, ChunkingOp, AggregateChunksOp,
    EmbedConfig, TransformerEmbedOp,
    DPHeadConfig, TransformerDPHeadOp,
    HFDocPredictConfig, HFDocClassifierOp,
    LLMDPConfig, LLMDPInferenceOp,
)

# ----------------------- I/O helpers -----------------------

def load_docs_from_csv(csv_path: Path, col_text: str, col_dp: str | None, col_patient: str, col_sejour: str):
    df = pd.read_csv(csv_path)
    docs: List[TextDocument] = []
    for _, row in df.iterrows():
        txt = str(row[col_text]) if pd.notna(row[col_text]) else ""
        d = TextDocument(text=txt)
        # ID utile : code_sejour
        if col_sejour in row and pd.notna(row[col_sejour]):
            d.metadata["id_sejour"] = str(row[col_sejour])
        if col_patient in row and pd.notna(row[col_patient]):
            d.metadata["id_patient"] = str(row[col_patient])
        # gold DP si dispo (mode train)
        if col_dp and (col_dp in row) and pd.notna(row[col_dp]):
            d.metadata["gold_dp"] = str(row[col_dp]).strip()
        docs.append(d)
    return docs

def write_preds_to_csv(docs: List[TextDocument], out_csv: Path, out_col_sejour: str, out_col_pred: str):
    rows = []
    for d in docs:
        sid = d.metadata.get("id_sejour", "")
        pred = d.metadata.get("pred_dp", "")
        # >>> forcer "code" si dict
        if isinstance(pred, dict):
            pred = pred.get("code", "")
        rows.append({out_col_sejour: sid, out_col_pred : pred})

    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[out_col_sejour, out_col_pred])
        w.writeheader()
        w.writerows(rows)


# ----------------------- Pipeline builder -----------------------

def build_pipeline(args: argparse.Namespace) -> Pipeline:
    steps: List[PipelineStep] = []

    # 1) Normalisation
    steps.append(PipelineStep(
        NormalizeOp(NormalizeConfig(
            strip=True,
            collapse_spaces=True,
            normalize_newlines=True,
            lower=False,
            keep_accents=True,
        )),
        ["docs_in"], ["docs_norm"]
    ))
    last_docs="docs_norm"

    # 2) Rewrite (optionnel)
    if args.rewrite:
        steps.append(PipelineStep(
            RewriteOp(RewriteConfig(
                enabled=True,
                target_words=args.rewrite_target_words,
                llm_model=args.rewrite_llm_model,
                max_new_tokens=args.rewrite_max_new_tokens,
                temperature=args.rewrite_temperature,
                top_p=args.rewrite_top_p,
            )),
            [last_docs], ["docs_rewrite"]
        ))
        last_docs="docs_rewrite"
    else:
        # même sans rewrite, on va utiliser text_rw = text_norm via NormalizeOp+RewriteOp?
        # Ici on laisse Chunking lire "text_rw" OU sinon "text_norm" si text_rw n'existe pas.
        pass

    # 3) Chunking
    
    # 4) Embeddings (pour backend=transformer)
    if args.backend == "transformer":
        steps.append(PipelineStep(
            ChunkingOp(ChunkingConfig(
                hf_model=args.hf_model,
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap,
                field_in="text_rw",   # ChunkingOp tombera sur text_norm si text_rw absent
                field_out="chunks",
            )),
            [last_docs], ["docs_chunk"]
        ))

        steps.append(PipelineStep(
            TransformerEmbedOp(EmbedConfig(
                hf_model=args.hf_model,
                device="auto" if args.device == "auto" else args.device,
                max_length=args.max_length,
                pooling=args.pooling,
                cls_layers=args.cls_layers,
                chunks_field="chunks",
                emb_field="emb",
            )),
            ["docs_chunk"], ["docs_tr"]
        ))

        steps.append(PipelineStep(
            AggregateChunksOp(strategy=args.aggregate),
            ["docs_tr"], ["docs_ag"]
        ))


        # 5) DP Head
        steps.append(PipelineStep(
            TransformerDPHeadOp(DPHeadConfig(
                bundle_path=str(args.bundle_path),
                mode=args.mode,
                C=args.lr_C,
                max_iter=args.lr_max_iter,
                emb_field="emb",
                gold_field="gold_dp",
                pred_field="pred_dp",
            )),
            ["docs_ag"], ["docs_out"]
        ))

    elif args.backend == "hf_finetuned":
        if not args.hf_checkpoint:
            raise ValueError("Veuillez fournir --hf-checkpoint (dossier HF sauvegardé par train_finetune_dp.py)")
        # ----- modèle HF fine-tuné pour le DP -----
        steps.append(PipelineStep(
            HFDocClassifierOp(HFDocPredictConfig(
                checkpoint_dir=args.hf_checkpoint,
                device=args.device if args.device in ("cpu","cuda") else "auto",
                max_length=args.max_length,
                stride=args.chunk_overlap,      # pas utilisé ici, sécurité
                chunks_field="chunks",
                pred_field="pred_dp",
                aggregate=args.aggregate_hf,
                return_proba=True,
            )),
            [last_docs], ["docs_out"]
        ))

    elif args.backend == "llm":
        # 4) LLM direct (pas d'embeddings)
        steps.append(PipelineStep(
            LLMDPInferenceOp(LLMDPConfig(
                hf_model=args.llm_model,
                max_new_tokens=args.llm_max_new_tokens,
                temperature=args.llm_temperature,
                top_p=args.llm_top_p,
                field_in="text_rw",          # retombe sur text_norm si absent
                pred_field="pred_dp",
            )),
            [last_docs], ["docs_out"]
        ))
    else:
        raise ValueError("backend doit être 'transformer' ou 'llm'")

    return Pipeline(steps=steps, input_keys=["docs_in"], output_keys=["docs_out"], name="dp_pipeline")


# ----------------------- CLI -----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Pipeline DP (Transformers local, sans Docker)")
    # I/O
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)

    # Colonnes d'entrée
    p.add_argument("--col-text", default="text")
    p.add_argument("--col-dp", default=None, help="Nom de la colonne gold DP (requis en mode train)")
    p.add_argument("--col-patient", default="code_patient")
    p.add_argument("--col-sejour", default="code_sejour")

    # Colonnes de sortie
    p.add_argument("--out-col-sejour", default="code_sejour")
    p.add_argument("--out-col-pred", default="dp_predit")

    # Backend & mode
    p.add_argument("--backend", choices=["transformer", "hf_finetuned", "llm"], default="transformer")
    p.add_argument("--mode", choices=["train", "predict"], default="predict")

    # Preprocessing
    p.add_argument("--rewrite", action="store_true", help="Activer la réécriture (LLM requis si tu veux paraphraser)")
    p.add_argument("--rewrite-llm-model", default=None)
    p.add_argument("--rewrite-target-words", type=int, default=300)
    p.add_argument("--rewrite-max-new-tokens", type=int, default=128)
    p.add_argument("--rewrite-temperature", type=float, default=0.3)
    p.add_argument("--rewrite-top-p", type=float, default=0.95)

    p.add_argument("--chunk-size", type=int, default=480)
    p.add_argument("--chunk-overlap", type=int, default=64)

    # Modèle encoder (transformer)
    p.add_argument("--hf-model", default="almanach/camembert-bio-base")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    p.add_argument("--cls-layers", type=int, default=1)
    p.add_argument("--bundle-path", type=Path, default=Path("assets/dp_transformer/bundle.joblib"))
    p.add_argument("--lr-C", type=float, default=1.0)
    p.add_argument("--lr-max-iter", type=int, default=200)
    p.add_argument("--aggregate", choices=["mean", "max", "first"], default="mean",help="Stratégie d’agrégation des embeddings de chunks en un seul vecteur doc.")

    # Transformer finetune
    p.add_argument("--hf-checkpoint", type=str, help="Dossier checkpoint HF fine-tuné")
    p.add_argument("--aggregate_hf", choices=["mean", "max", "median"], default="mean")


    # LLM (Transformers)
    p.add_argument("--llm-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--llm-max-new-tokens", type=int, default=64)
    p.add_argument("--llm-temperature", type=float, default=0.0)
    p.add_argument("--llm-top-p", type=float, default=1.0)

    return p.parse_args()


def main():
    args = parse_args()

    # Charger CSV
    docs = load_docs_from_csv(
        args.input_csv,
        col_text=args.col_text,
        col_dp=args.col_dp if args.mode == "train" else None,
        col_patient=args.col_patient,
        col_sejour=args.col_sejour,
    )
    if not docs:
        raise SystemExit("Aucun document lu depuis le CSV d'entrée.")

    # Vérifs de mode/colonnes
    if args.backend == "transformer" and args.mode == "train" and args.col_dp is None:
        raise SystemExit("En mode train (backend=transformer), --col-dp est requis pour les labels gold.")

    # Construire pipeline
    pipe = build_pipeline(args)

    # Exécuter
    docs_out = pipe.run(docs)

    # Écrire résultats
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_preds_to_csv(docs_out, args.output_csv, args.out_col_sejour, args.out_col_pred)

    # Affichage bref
    n_pred = sum(1 for d in docs_out if d.metadata.get("pred_dp"))
    print(f"[OK] Documents: {len(docs_out)} | DP prédits: {n_pred} | sortie: {args.output_csv}")

if __name__ == "__main__":
    main()
