"""
run_pipeline.py — Pipeline modulaire de prédiction du Diagnostic Principal (DP)

But
----
Chaîner différentes opérations NLP Medkit (normalisation, réécriture, chunking, embeddings,
classification ...) pour entraîner ou inférer des codes DP à partir de comptes rendus
hospitaliers. 

Entrée / Sortie
---------------
- Entrée : CSV avec colonnes configurables (texte, DP gold, id patient/séjour)
- Sortie : CSV avec au minimum (code_sejour, dp_predit).
  Optionnellement : métriques BEFORE/AFTER si --rewrite + --with-metrics.

Backends
--------
- transformer    : embeddings HF gelé + tête LogReg (train/predict)
- hf_finetuned   : modèle HF fine-tuné (sequence classification)
- llm            : inférence DP par génération + extraction CIM-10

Options clés
------------
--normalisation        Activer nettoyage du texte
--rewrite              Réécriture par LLM (avec métriques possibles)
--chunk-size/overlap   Contrôle des fenêtres pour embeddings
--bundle-path          Sauvegarde/chargement de la tête LogReg
--hf-checkpoint        Dossier checkpoint d’un modèle HF fine-tuné
--llm-model            Modèle causal pour inférence directe

Exemples
--------
# Entraînement tête LogReg
python run_pipeline.py --backend transformer --mode train --input-csv data/train.csv --col-dp code_dp

# Inférence LLM avec réécriture + métriques
python run_pipeline.py --backend llm --rewrite --with-metrics --input-csv data/dev.csv
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
    MetricsTextOp, MetricsTextConfig,
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


# ----------------------- Construction du pipeline -----------------------

def build_pipeline(args: argparse.Namespace) -> Pipeline:
    steps: List[PipelineStep] = []

    last_docs="docs_in"

    # Normalisation
    if args.normalisation :
        steps.append(PipelineStep(
            NormalizeOp(NormalizeConfig(
                strip=True,
                collapse_spaces=True,
                normalize_newlines=True,
                lower=False,
                keep_accents=True,
            )),
            [last_docs], ["docs_norm"]
        ))
        last_docs="docs_norm"

    # Reecriture via LLM
    # --- Calcul des metriques de texte avant reecriture
    if args.rewrite and args.with_metrics:    
        steps.append(PipelineStep(
            MetricsTextOp(MetricsTextConfig(
                text_field=args.col_text,      # ou "text_norm" si on veut travailler sur le texte normalisé à une précédente étape
                metrics_root="metrics",
                phase="before"
            )),
            [last_docs], ["docs_rw_b"]
        ))
        last_docs ="docs_rw_b"

    # --- Reecriture
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
            [last_docs], ["docs_rw"]
        ))
        last_docs ="docs_rw"

    # --- Calcul des metriques de texte apres reecriture
    if args.rewrite and args.with_metrics:
        steps.append(PipelineStep(
            MetricsTextOp(MetricsTextConfig(
                text_field="text_rw",
                metrics_root="metrics",
                phase="after"
            )),
            [last_docs], ["docs_rw_a"]
        ))
        last_docs ="docs_rw_a"


    else:
        pass

    # Classification via Transformer gele
    if args.backend == "transformer":
        # Chunking
        steps.append(PipelineStep(
            ChunkingOp(ChunkingConfig(
                hf_model=args.hf_model,
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap,
                field_in="text_rw",   # ChunkingOp tombera sur text_norm si text_rw absent, et sut texte initial si pas de normalisation
                field_out="chunks",
            )),
            [last_docs], ["docs_chunk"]
        ))
        # Calcul des embeddings par chunk
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
        # Aggregation des embeddings
        steps.append(PipelineStep(
            AggregateChunksOp(strategy=args.aggregate),
            ["docs_tr"], ["docs_ag"]
        ))

        # Tete de classification
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

    # Classification via Transformer finetune
    elif args.backend == "hf_finetuned":
        if not args.hf_checkpoint:
            raise ValueError("Veuillez fournir --hf-checkpoint (dossier HF sauvegardé par train_finetune_dp.py)")

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
                batch_size=args.hf_batch_size,
            )),
            [last_docs], ["docs_out"]
        ))
    # Prediction via LLM 
    elif args.backend == "llm":
        steps.append(PipelineStep(
            LLMDPInferenceOp(LLMDPConfig(
                hf_model=args.llm_model,
                max_new_tokens=args.llm_max_new_tokens,
                temperature=args.llm_temperature,
                top_p=args.llm_top_p,
                field_in="text_rw",          # retombe sur text_norm si absent, texte initial si pas de normalisation
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
    p.add_argument("--col-text", default="text")   # "text_rw" si on tourne sur dataset reecrit 
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
    p.add_argument("--normalisation", action="store_true", help="Activer la normalisation du texte en entrée de pipeline")
    p.add_argument("--with-metrics", action="store_true", help="Calcule les métriques BEFORE/AFTER autour de la réécriture et les ajoute au CSV de sortie.")
    p.add_argument("--rewrite", action="store_true", help="Activer la réécriture (LLM requis si tu veux paraphraser)")
    p.add_argument("--rewrite-llm-model", default=None)
    p.add_argument("--rewrite-target-words", type=int, default=300)
    p.add_argument("--rewrite-max-new-tokens", type=int, default=128)
    p.add_argument("--rewrite-temperature", type=float, default=0.3)
    p.add_argument("--rewrite-top-p", type=float, default=0.95)

    #Chunking
    p.add_argument("--chunk-size", type=int, default=480)
    p.add_argument("--chunk-overlap", type=int, default=64)

    # Modèle transformer gele
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

    # Transformer finetune
    p.add_argument("--hf-checkpoint", type=str, help="Dossier checkpoint HF fine-tuné")
    p.add_argument("--hf-batch-size", type=int, default=16)
    p.add_argument("--aggregate_hf", choices=["mean", "max", "median"], default="mean")


    # LLM 
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

    # Écrire résultats (+ métriques si demandé)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    use_metrics = getattr(args, "with_metrics", False) and getattr(args, "rewrite", False)

    if not use_metrics:
        write_preds_to_csv(docs_out, args.output_csv, args.out_col_sejour, args.out_col_pred)
    else:
        # ---- Construction CSV avec métriques ----
        import pandas as pd

        def _metrics_keys(d):
            mb = d.metadata.get("metrics_before") or {}
            ma = d.metadata.get("metrics_after") or {}
            return sorted(set(mb.keys()) | set(ma.keys()))

        # Récupère l’ensemble des clés de métriques présentes dans tout le lot
        all_keys = set()
        for d in docs_out:
            all_keys.update(_metrics_keys(d))
        all_keys = sorted(all_keys)

        rows = []
        for d in docs_out:
            row = {
                args.out_col_sejour: d.metadata.get("id_sejour", ""),
                args.out_col_pred: d.metadata.get("pred_dp", ""),
            }
            mb = d.metadata.get("metrics_before") or {}
            ma = d.metadata.get("metrics_after") or {}
            for k in all_keys:
                row[f"{k}_before"] = mb.get(k)
                row[f"{k}_after"]  = ma.get(k)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(args.output_csv, index=False)

        # ---- Affichage des moyennes globales ----
        numeric_cols = [c for c in df.columns if c.endswith("_before") or c.endswith("_after")]
        means = df[numeric_cols].mean(numeric_only=True)

        print("\n[MÉTRIQUES — moyennes globales]")
        for k in all_keys:
            b_col, a_col = f"{k}_before", f"{k}_after"
            b_mean = means.get(b_col, float("nan"))
            a_mean = means.get(a_col, float("nan"))
            print(f"  - {k:20s}  before={b_mean:.3f}  |  after={a_mean:.3f}")

    # Affichage final
    n_pred = sum(1 for d in docs_out if d.metadata.get("pred_dp"))
    print(f"[OK] Documents: {len(docs_out)} | DP prédits: {n_pred} | sortie: {args.output_csv}")

if __name__ == "__main__":
    main()
