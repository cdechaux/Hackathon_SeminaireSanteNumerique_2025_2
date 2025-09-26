"""
Évalue des prédictions de Diagnostic Principal (DP) (un seul code par séjour) en comparant
un CSV "gold" (de départ) et un CSV "pred" (de sortie du pipeline).

- Jointure sur un identifiant commun (ex: code_sejour).
- Colonnes DP paramétrables.
- Normalisation optionnelle (majuscules, retrait des points) désactivée par défaut.
- Calcule précision, rappel, F1 (micro/macro/weighted) + accuracy.

Usage:
  python eval_dp.py \
    --gold-csv data/dp_dataset.csv --gold-id code_sejour --gold-col code_dp \
    --pred-csv data/pred.csv       --pred-id code_sejour --pred-col dp_predit \
    --average macro

Options utiles:
  --average {micro,macro,weighted}  (par défaut: macro)
  --normalize-upper                 (met en MAJUSCULES)
  --strip-dot                       (retire les points "." des codes)
  --print-report                    (affiche le classification_report)
"""

import argparse
import pandas as pd
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)
import sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gold-csv", required=True, help="CSV de vérité terrain")
    p.add_argument("--pred-csv", required=True, help="CSV des prédictions")
    p.add_argument("--gold-id", required=True, help="Nom de la colonne ID dans le CSV gold")
    p.add_argument("--pred-id", required=True, help="Nom de la colonne ID dans le CSV pred")
    p.add_argument("--gold-col", required=True, help="Nom de la colonne DP dans le CSV gold")
    p.add_argument("--pred-col", required=True, help="Nom de la colonne DP prédite dans le CSV pred")
    p.add_argument("--average", choices=["micro", "macro", "weighted"], default="macro",
                   help="Type de moyenne pour le F1 (défaut: macro)")
    p.add_argument("--normalize-upper", action="store_true", help="Passe les codes en MAJUSCULES")
    p.add_argument("--strip-dot", action="store_true", help="Retire les '.' dans les codes")
    p.add_argument("--print-report", action="store_true", help="Affiche le classification_report détaillé")
    return p.parse_args()

def _norm_code(x: str, upper: bool, strip_dot: bool) -> str:
    if not isinstance(x, str):
        return ""
    s = x.strip()
    if upper:
        s = s.upper()
    if strip_dot:
        s = s.replace(".", "")
    return s

def main():
    args = parse_args()

    try:
        df_g = pd.read_csv(args.gold_csv)
        df_p = pd.read_csv(args.pred_csv)
    except Exception as e:
        print(f"[ERR] Impossible de lire les CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Conserver uniquement les colonnes utiles
    miss_g = [c for c in [args.gold_id, args.gold_col] if c not in df_g.columns]
    miss_p = [c for c in [args.pred_id, args.pred_col] if c not in df_p.columns]
    if miss_g:
        print(f"[ERR] Colonnes manquantes dans gold: {miss_g}", file=sys.stderr)
        sys.exit(1)
    if miss_p:
        print(f"[ERR] Colonnes manquantes dans pred: {miss_p}", file=sys.stderr)
        sys.exit(1)

    # Normalisation des codes (optionnelle)
    df_g["_dp_gold"] = df_g[args.gold_col].apply(
        lambda x: _norm_code(x, args.normalize_upper, args.strip_dot)
    )
    df_p["_dp_pred"] = df_p[args.pred_col].apply(
        lambda x: _norm_code(x, args.normalize_upper, args.strip_dot)
    )

    # Jointure interne sur l'ID
    left = df_g[[args.gold_id, "_dp_gold"]].rename(columns={args.gold_id: "_ID"})
    right = df_p[[args.pred_id, "_dp_pred"]].rename(columns={args.pred_id: "_ID"})
    merged = left.merge(right, on="_ID", how="inner")

    if merged.empty:
        print("[ERR] Aucune ligne commune après jointure sur l'ID.", file=sys.stderr)
        sys.exit(1)

    # Filtrer les lignes valides (codes non vides)
    merged = merged[(merged["_dp_gold"].astype(str).str.len() > 0) &
                    (merged["_dp_pred"].astype(str).str.len() > 0)]

    if merged.empty:
        print("[ERR] Après nettoyage, plus de paires (gold/pred) valides.", file=sys.stderr)
        sys.exit(1)

    y_true = merged["_dp_gold"].tolist()
    y_pred = merged["_dp_pred"].tolist()

    # Calcul des métriques
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=args.average, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    print(f"N appariés: {len(merged)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision ({args.average}): {prec:.4f}")
    print(f"Recall    ({args.average}): {rec:.4f}")
    print(f"F1        ({args.average}): {f1:.4f}")

    if args.print_report:
        print("\nClassification report (par classe):")
        print(classification_report(y_true, y_pred, zero_division=0))

if __name__ == "__main__":
    main()
