#!/usr/bin/env python3
# create_rewrite_dataset.py
"""
Réécrit un dataset de comptes rendus + calcule des métriques avant/après.
- Optionnel: réécriture via un modèle Causal LM (HF Transformers) local.
- target_words: ENTIER (≈ nombre de mots visés), ou None pour désactiver la contrainte.
- Sections détectées par blocs séparés par >= 2 retours ligne.
- Abréviations approximées: mots en MAJ >=2 lettres OU formes A.B.C.
- Sorties:
  * CSV: colonnes originales + text_rewritten + métriques _before/_after
  * JSON: moyennes globales et configuration (incluant target_words)
"""

from __future__ import annotations
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd

# Dépendances HF seulement si LLM activé
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None


# ----------------------------- Métriques -----------------------------

_RE_TOKEN = re.compile(r"\b\w+\b", flags=re.UNICODE)
_RE_SENT_SPLIT = re.compile(r"[\.!?]+|\n+", flags=re.UNICODE)  # phrases grossières
_RE_ABBR = re.compile(r"\b([A-Z]{2,}|(?:[A-Za-z]\.){2,})\b")  # MAJUSCULES ou A.B.C

def sections_by_blanklines(text: str) -> int:
    """Nombre de sections = blocs séparés par >= 2 sauts de ligne."""
    if not text:
        return 0
    blocks = [b for b in re.split(r"\n{2,}", text.strip()) if b.strip()]
    return len(blocks)

def simple_tokens(text: str):
    return _RE_TOKEN.findall(text.lower()) if text else []

def vocab_size(text: str) -> int:
    toks = simple_tokens(text)
    return len(set(toks))

def len_chars(text: str) -> int:
    return len(text or "")

def len_words(text: str) -> int:
    return len(simple_tokens(text))

def mean_sentence_len_words(text: str) -> float:
    if not text:
        return 0.0
    # split grossier en phrases
    parts = [p.strip() for p in _RE_SENT_SPLIT.split(text) if p.strip()]
    if not parts:
        return 0.0
    lens = [len(simple_tokens(p)) for p in parts if p]
    return float(np.mean(lens)) if lens else 0.0

def abbr_count(text: str) -> int:
    return len(_RE_ABBR.findall(text or ""))

def compute_metrics(text: str) -> Dict[str, float | int]:
    return {
        "len_chars": len_chars(text),
        "len_words": len_words(text),
        "mean_sent_len_words": round(mean_sentence_len_words(text), 4),
        "vocab_size": vocab_size(text),
        "n_sections": sections_by_blanklines(text),
        "abbr_count": abbr_count(text),
    }


# ----------------------------- Rewrite Op -----------------------------

@dataclass
class RewriteConfig:
    enabled: bool = False
    target_words: Optional[int] = None          # ENTIER ≈ longueur cible
    llm_model: Optional[str] = None            # modèle HF causal (local)
    max_new_tokens: int = 128
    temperature: float = 0.3
    top_p: float = 0.95

class RewriteOp:
    """
    Réécriture simple:
      - Si LLM fourni: paraphrase/condense en visant ~target_words (si donné)
      - Sinon: copie le texte tel quel
    """
    def __init__(self, cfg: RewriteConfig):
        self.cfg = cfg
        self._tok = None
        self._lm = None
        if self.cfg.enabled and self.cfg.llm_model:
            if AutoTokenizer is None or AutoModelForCausalLM is None:
                raise RuntimeError("Transformers non disponible. Installez transformers/torch.")
            self._tok = AutoTokenizer.from_pretrained(self.cfg.llm_model)
            self._lm = AutoModelForCausalLM.from_pretrained(self.cfg.llm_model, device_map="auto")
            self._lm.eval()

    @torch.no_grad() if torch else (lambda f: f)
    def _rewrite_with_llm(self, text: str) -> str:
        if not text:
            return ""
        assert self._tok and self._lm
        target = f"\nLongueur cible: ~{self.cfg.target_words} mots." if self.cfg.target_words else ""
        prompt = (
            "Réécris et résume le compte rendu hospitalier ci-dessous en gardant le sens clinique, "
            "en français clair, sans inventer d'informations, en visant l’extraction du diagnostic principal."
            f"{target}\n\nTexte:\n{ text.strip() }"
        )
        tok = self._tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
        tok = {k: v.to(self._lm.device) for k, v in tok.items()}
        out = self._lm.generate(
            **tok,
            do_sample=True,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_new_tokens=self.cfg.max_new_tokens,
            pad_token_id=self._tok.eos_token_id,
        )
        gen = self._tok.decode(out[0], skip_special_tokens=True)
        # Si le modèle recopie le prompt, on ne garde que la partie après "Texte:"
        parts = gen.split("Texte:")
        rewritten = parts[-1].strip() if parts else gen.strip()
        return rewritten

    def run_one(self, text: str) -> str:
        base = text or ""
        if self.cfg.enabled and self._lm is not None:
            return self._rewrite_with_llm(base)
        return base


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True, help="CSV d’entrée")
    ap.add_argument("--output-csv", required=True, help="CSV de sortie réécrit + métriques")
    ap.add_argument("--summary-json", required=False, default=None,
                    help="Chemin JSON récapitulatif (moyennes + config). Par défaut: <output-csv>.json")
    ap.add_argument("--text-col", default="text", help="Nom de la colonne texte d’entrée")
    ap.add_argument("--rewrite", action="store_true", help="Activer la réécriture LLM")
    ap.add_argument("--llm-model", default=None, help="Modèle causal HF (local), ex: 'mistral-small' (gguf converti), 'NousResearch/Meta-Llama-3-8B-Instruct' (si dispo local)")
    ap.add_argument("--target-words", type=int, default=None, help="Nombre de mots visés (~). Laisse vide pour désactiver.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)
    sum_path = Path(args.summary_json) if args.summary_json else out_path.with_suffix(out_path.suffix + ".json")

    df = pd.read_csv(in_path)
    if args.text_col not in df.columns:
        raise ValueError(f"Colonne '{args.text_col}' absente du CSV.")

    # Config + Op
    cfg = RewriteConfig(
        enabled=bool(args.rewrite),
        target_words=args.target_words,
        llm_model=args.llm_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    op = RewriteOp(cfg)

    # Colonnes métriques à ajouter
    metrics_names = ["len_chars", "len_words", "mean_sent_len_words", "vocab_size", "n_sections", "abbr_count"]
    for m in metrics_names:
        df[f"{m}_before"] = np.nan
        df[f"{m}_after"] = np.nan

    # Réécriture + métriques
    rewritten_texts = []
    for i, row in df.iterrows():
        txt = str(row.get(args.text_col, "") or "")

        m_before = compute_metrics(txt)
        for k, v in m_before.items():
            df.at[i, f"{k}_before"] = v

        txt_rw = op.run_one(txt)
        rewritten_texts.append(txt_rw)

        m_after = compute_metrics(txt_rw)
        for k, v in m_after.items():
            df.at[i, f"{k}_after"] = v

    df["text_rewritten"] = rewritten_texts

    # Sauvegarde CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Moyennes globales
    means = {}
    for m in metrics_names:
        means[f"{m}_before_mean"] = float(np.nanmean(df[f"{m}_before"].astype(float)))
        means[f"{m}_after_mean"]  = float(np.nanmean(df[f"{m}_after"].astype(float)))

    # Impression console
    print("\n--- Moyennes globales (avant/après) ---")
    for m in metrics_names:
        print(f"{m:>20}: before={means[f'{m}_before_mean']:.3f} | after={means[f'{m}_after_mean']:.3f}")

    # JSON résumé (inclut target_words & config)
    summary: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "config": {
            "rewrite_enabled": cfg.enabled,
            "llm_model": cfg.llm_model,
            "target_words": cfg.target_words,     # ENTIER ou None
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "section_rule": "sections = blocs séparés par >= 2 sauts de ligne",
            "abbr_rule": "mots MAJ >=2 lettres ou formes A.B.C détectées par regex",
        },
        "metrics_means": means,
    }
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Écrit: {out_path}")
    print(f"[OK] Résumé: {sum_path}")


if __name__ == "__main__":
    main()
