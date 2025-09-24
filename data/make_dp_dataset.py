#!/usr/bin/env python3
# make_dp_dataset.py
# Usage:
#   python make_dp_dataset.py --in data_1000_response.csv --out dp_dataset.csv --code-col icd_primary_code

import argparse
from pathlib import Path
import re
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser("Convertit un CSV (colonne 'response') en dataset DP")
    p.add_argument("--in",  dest="in_csv",  type=Path, required=True, help="CSV source avec la colonne 'response'")
    p.add_argument("--out", dest="out_csv", type=Path, required=True, help="CSV de sortie (code_patient, code_sejour, text, code_dp)")
    p.add_argument("--response-col", default="response", help="Nom de la colonne contenant le blob 'JSON-like'")
    p.add_argument("--code-col", default="icd_primary_code", help="Colonne CSV qui contient le DP à copier vers 'code_dp'")
    p.add_argument("--fallback-code-cols", nargs="*", default=["dp", "primary_icd10", "primary_icd", "code"],
                   help="Colonnes alternatives si --code-col absente ou vide")
    return p.parse_args()

# Regex non-gourmande, multi-ligne :
# capture tout ce qu'il y a entre "CR": " ... " et la clé suivante "formulations"
RE_CR = re.compile(r'"CR"\s*:\s*"(.*?)"\s*,\s*\n\s*"formulations"', re.S)

def extract_cr_text(cell: str) -> str | None:
    """Extrait le bloc CR (markdown) sans tenter de parser du JSON invalide."""
    if not isinstance(cell, str) or not cell.strip():
        return None

    # Si la cellule est entourée de fences ```...```, on les ignore sans supposer leur présence
    # On cherche directement le motif "CR": " .... " , "formulations"
    m = RE_CR.search(cell)
    if not m:
        # Variante: parfois il n'y a pas 'formulations' derrière; on coupe jusqu'à la prochaine clé JSON-like.
        m = re.search(r'"CR"\s*:\s*"(.*?)"\s*,\s*\n\s*"', cell, re.S)
    if not m:
        return None

    raw = m.group(1)
    # Certains CSV doublent les guillemets à l'intérieur des strings : "" -> "
    raw = raw.replace('""', '"')
    return raw.strip()

def main():
    args = parse_args()
    df = pd.read_csv(args.in_csv)

    if args.response_col not in df.columns:
        raise SystemExit(f"Colonne '{args.response_col}' introuvable dans {args.in_csv}")

    have_code_col = args.code_col in df.columns
    rows = []

    for i, row in df.iterrows():
        # 1) Texte CR depuis la colonne response (extraction par regex)
        resp_raw = row.get(args.response_col)
        text_val = extract_cr_text(resp_raw) or ""

        # 2) Code DP depuis colonne(s) CSV
        code_dp = ""
        if have_code_col:
            v = row.get(args.code_col)
            if isinstance(v, str) and v.strip():
                code_dp = v.strip()
        if not code_dp:
            for c in args.fallback_code_cols:
                if c in df.columns:
                    v = row.get(c)
                    if isinstance(v, str) and v.strip():
                        code_dp = v.strip()
                        break

        # 3) IDs synthétiques
        code_patient = f"PAT{100000 + i}"
        code_sejour  = f"SEJ{200000 + i}"

        rows.append({
            "code_patient": code_patient,
            "code_sejour": code_sejour,
            "text": text_val,
            "code_dp": code_dp or "",
        })

    out = pd.DataFrame(rows, columns=["code_patient", "code_sejour", "text", "code_dp"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"✅ Écrit {len(out)} lignes → {args.out_csv}")

if __name__ == "__main__":
    main()
