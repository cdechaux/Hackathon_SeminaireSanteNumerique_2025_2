"""
Pipeline simple pour prediction du Diagnostic Principal (DP) à partir de comptes rendus hospitaliers.

Étapes principales :
1. Lecture d’un CSV d’entrée et création de TextDocument (medkit).
2. Normalisation légère du texte (espaces, sauts de ligne).
3. Découpage en chunks de tokens (avec overlap).
4. Encodage des chunks avec un modèle Transformer (CamemBERT-Bio par défaut).
5. Agrégation des embeddings de chunks en un vecteur par document.
6. Classification multiclasse avec une régression logistique (tête simple DP).

Deux modes pour backend transformer :
- train   : ajuste la tête de classification (LogReg) et sauvegarde un bundle .joblib
- predict : recharge un bundle existant et applique le modèle aux nouveaux textes

Entrées :
- CSV avec colonnes texte, séjour, patient, (et éventuellement DP gold)
- Paramètres configurables via la CLI (modèle HF, chunk size, pooling, etc.)

Sorties :
- CSV avec pour chaque séjour le DP prédit

Ce script sert de pipeline *de base* pour le hackathon.
Les participants peuvent l’utiliser tel quel ou le modifier (autres modèles,
ajout de réécriture, métriques, fine-tuning complet, etc.).
"""

from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from medkit.core.pipeline import Pipeline, PipelineStep
from medkit.core.text import TextDocument

from operations import (
    NormalizeConfig, NormalizeOp,
    ChunkingConfig, ChunkingOp, AggregateChunksOp,
    EmbedConfig, TransformerEmbedOp,
    DPHeadConfig, TransformerDPHeadOp,

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


# ----------------------- Construction du pipeline  -----------------------

def build_pipeline(args: argparse.Namespace) -> Pipeline:
    steps: List[PipelineStep] = []

    # 1) Normalisation des textes

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

    # 2) Chunking des textes

    steps.append(PipelineStep(
        ChunkingOp(ChunkingConfig(
            hf_model=args.hf_model,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
            field_in="text_norm",   
            field_out="chunks",
        )),
        ["docs_norm"], ["docs_chunk"]
    ))

    # 3) Calcul des embeddings des textes

    steps.append(PipelineStep(
        TransformerEmbedOp(EmbedConfig(
            hf_model=args.hf_model,
            device="auto" if args.device == "auto" else args.device,
            max_length=args.max_length,
            pooling=args.pooling,
            cls_layers=args.cls_layers,
            chunks_field="chunks",
            emb_field="emb",
            batch_size=args.embed_batch_size,
        )),
        ["docs_chunk"], ["docs_tr"]
    ))

    # 4) Aggrégation des embeddings des différents chunks d'un même texte

    steps.append(PipelineStep(
        AggregateChunksOp(strategy=args.aggregate),
        ["docs_tr"], ["docs_ag"]
    ))


    # 5) Tete de classification avec Transformer gele

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

    # Chunking
    p.add_argument("--chunk-size", type=int, default=480)
    p.add_argument("--chunk-overlap", type=int, default=64)

    # Modèle encoder (transformer)
    p.add_argument("--mode", choices=["train", "predict"], default="predict")
    p.add_argument("--hf-model", default="almanach/camembert-bio-base")
    p.add_argument("--embed-batch-size", type=int, default=16)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    p.add_argument("--cls-layers", type=int, default=1)
    p.add_argument("--bundle-path", type=Path, default=Path("assets/dp_transformer/bundle.joblib"))
    p.add_argument("--lr-C", type=float, default=1.0)
    p.add_argument("--lr-max-iter", type=int, default=200)
    p.add_argument("--aggregate", choices=["mean", "max", "first"], default="mean",help="Stratégie d’agrégation des embeddings de chunks en un seul vecteur doc.")


    return p.parse_args()


def main():
    args = parse_args()

    # Charger CSV
    docs = load_docs_from_csv(
        args.input_csv,
        col_text=args.col_text,
        col_dp=args.col_dp,
        col_patient=args.col_patient,
        col_sejour=args.col_sejour,
    )
    if not docs:
        raise SystemExit("Aucun document lu depuis le CSV d'entrée.")

    # Construire le pipeline
    pipe = build_pipeline(args)

    # Exécuter
    docs_out = pipe.run(docs)

    # Écrire les résultats
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_preds_to_csv(docs_out, args.output_csv, args.out_col_sejour, args.out_col_pred)
   
    # Affichage 
    n_pred = sum(1 for d in docs_out if d.metadata.get("pred_dp"))
    print(f"[OK] Documents: {len(docs_out)} | DP prédits: {n_pred} | sortie: {args.output_csv}")

if __name__ == "__main__":
    main()
