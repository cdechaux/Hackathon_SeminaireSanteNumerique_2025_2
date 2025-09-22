#!/usr/bin/env python3
# operations.py
# Opérations medkit pour un pipeline DP-only, via Transformers.
# - NormalizeOp : nettoyage léger du texte
# - RewriteOp   : réécriture optionnelle (via LLM HF ou no-op)
# - ChunkingOp  : découpe en fenêtres de tokens (overlap)
# - TransformerEmbedOp : embeddings document (agrégation de chunks)
# - TransformerDPHeadOp : entraînement / prédiction d’un DP multiclasse (LogReg)
# - LLMDPInferenceOp    : prédiction DP via LLM HF (génération + extraction regex)
#
# Aucune troncature/modification des codes : les points sont conservés.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Optional, Dict, Any, Tuple
import os
import re
import json
import time
import joblib
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

from medkit.core.operation import Operation
from medkit.core.text import TextDocument

# --------------------------- Utils génériques ---------------------------

ICD10_REGEX = re.compile(r"\b[A-TV-Z][0-9]{2}(?:\.[0-9A-TV-Z]{1,4})?\b")

def first_icd10(text: str) -> Optional[str]:
    m = ICD10_REGEX.search(text or "")
    return m.group(0) if m else None


# --------------------------- 1) Normalize -------------------------------

@dataclass
class NormalizeConfig:
    strip: bool = True
    collapse_spaces: bool = True
    normalize_newlines: bool = True
    lower: bool = False      # False par défaut (les acronymes FR comptent)
    keep_accents: bool = True

class NormalizeOp(Operation):
    """Nettoyage léger du texte du CRH (aucune agressivité)."""
    def __init__(self, cfg: Optional[NormalizeConfig] = None):
        super().__init__()
        self.cfg = cfg or NormalizeConfig()

    def _normalize(self, txt: str) -> str:
        if txt is None:
            return ""
        t = txt
        if self.cfg.strip:
            t = t.strip()
        if self.cfg.normalize_newlines:
            # uniformiser les sauts de ligne triples → doubles
            t = re.sub(r"\n{3,}", "\n\n", t)
        if self.cfg.collapse_spaces:
            # compacter espaces multiples hors nouvelle ligne
            t = re.sub(r"[ \t]{2,}", " ", t)
        if self.cfg.lower:
            t = t.lower()
        return t

    def run(self, docs: Sequence[TextDocument]):
        for d in docs:
            d.metadata["text_norm"] = self._normalize(d.text)
        return docs


# --------------------------- 2) Rewrite (optionnelle) -------------------

@dataclass
class RewriteConfig:
    enabled: bool = False
    target_words: Optional[int] = None
    llm_model: Optional[str] = None            # modèle HF causal LM pour paraphrase
    max_new_tokens: int = 128
    temperature: float = 0.3
    top_p: float = 0.95

class RewriteOp(Operation):
    """Réécriture simple : si LLM fourni → paraphrase condensée; sinon copie le texte normalisé."""
    def __init__(self, cfg: RewriteConfig):
        super().__init__()
        self.cfg = cfg
        self._tok = None
        self._lm = None
        if self.cfg.enabled and self.cfg.llm_model:
            self._tok = AutoTokenizer.from_pretrained(self.cfg.llm_model)
            self._lm = AutoModelForCausalLM.from_pretrained(self.cfg.llm_model, device_map="auto")
            self._lm.eval()

    @torch.no_grad()
    def _rewrite_with_llm(self, text: str) -> str:
        if not text:
            return ""
        assert self._tok and self._lm
        target = f"\nLongueur cible: ~{self.cfg.target_words} mots." if self.cfg.target_words else ""
        prompt = (
            "Réécris le compte rendu hospitalier ci-dessous en gardant le sens clinique, "
            "en français clair, sans inventer d'informations." + target +
            "\nTexte:\n" + text.strip()
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
        # Récupère la partie après "Texte:" si le modèle recopie le prompt
        parts = gen.split("Texte:")
        return parts[-1].strip() if parts else gen.strip()

    def run(self, docs: Sequence[TextDocument]):
        for d in docs:
            base = d.metadata.get("text_norm", d.text)
            if self.cfg.enabled and self._lm is not None:
                d.metadata["text_rw"] = self._rewrite_with_llm(base)
            else:
                d.metadata["text_rw"] = base
        return docs


# --------------------------- 3) Chunking (token) ------------------------

@dataclass
class ChunkingConfig:
    hf_model: str                      # tokenizer pour compter les tokens
    chunk_size: int = 480              # en tokens
    overlap: int = 64                  # en tokens
    field_in: str = "text_rw"          # ou "text_norm" si pas de rewrite
    field_out: str = "chunks"          # liste de str

class ChunkingOp(Operation):
    """Découpage token-based avec overlap, en gardant un mapping doc → chunks (texte brut)."""
    def __init__(self, cfg: ChunkingConfig):
        super().__init__()
        self.cfg = cfg
        self._tok = AutoTokenizer.from_pretrained(self.cfg.hf_model, use_fast=True)

    def _to_chunks(self, text: str) -> List[str]:
        if not text:
            return []
        enc = self._tok(text, add_special_tokens=False)
        ids = enc["input_ids"]
        chunks = []
        step = self.cfg.chunk_size - self.cfg.overlap
        for start in range(0, len(ids), step if step > 0 else self.cfg.chunk_size):
            end = start + self.cfg.chunk_size
            sub_ids = ids[start:end]
            if not sub_ids:
                continue
            chunks.append(self._tok.decode(sub_ids))
            if end >= len(ids):
                break
        return chunks

    def run(self, docs: Sequence[TextDocument]):
        fi, fo = self.cfg.field_in, self.cfg.field_out
        for d in docs:
            text = d.metadata.get(fi) or d.text
            d.metadata[fo] = self._to_chunks(text)
        return docs


# --------------------------- 4) Embeddings Transformer ------------------

@dataclass
class EmbedConfig:
    hf_model: str
    device: str = "auto"        # "cuda" | "cpu" | "auto"
    max_length: int = 512
    pooling: str = "cls"        # "cls" ou "mean"
    cls_layers: int = 1         # nb de derniers layers à moyenner (si pooling=cls)
    chunks_field: str = "chunks"
    emb_field: str = "emb"

class TransformerEmbedOp(Operation):
    """Encode chaque chunk puis agrège (moyenne) → un embedding par document."""
    def __init__(self, cfg: EmbedConfig):
        super().__init__()
        self.cfg = cfg
        self._tok = AutoTokenizer.from_pretrained(cfg.hf_model)
        self._enc = AutoModel.from_pretrained(cfg.hf_model, output_hidden_states=True)
        device = cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._enc.to(self._device).eval()

    @torch.no_grad()
    def _emb_one(self, text: str) -> np.ndarray:
        t = self._tok(text, padding=True, truncation=True, max_length=self.cfg.max_length, return_tensors="pt")
        t = {k: v.to(self._device) for k, v in t.items()}
        out = self._enc(**t)
        if self.cfg.pooling == "cls":
            hs = out.hidden_states  # list of layers: (B, T, D)
            sel = torch.stack(hs[-self.cfg.cls_layers:])[:, :, 0, :].mean(0)  # B×D sur [CLS]
        else:
            sel = out.last_hidden_state.mean(1)  # moyenne temporelle
        return sel[0].detach().cpu().numpy()

    def run(self, docs: Sequence[TextDocument]):
        for d in docs:
            chunks = d.metadata.get(self.cfg.chunks_field) or [d.metadata.get("text_rw") or d.metadata.get("text_norm") or d.text]
            vecs = []
            for ch in chunks:
                try:
                    vecs.append(self._emb_one(ch))
                except Exception:
                    # si un chunk déconne (trop long après troncation improbable), on ignore
                    continue
            if not vecs:
                # vecteur zéro de secours (dimension = D du modèle)
                with torch.no_grad():
                    dummy = self._emb_one((d.metadata.get("text_rw") or d.metadata.get("text_norm") or d.text)[:128])
                vecs = [dummy * 0]
            emb = np.mean(np.stack(vecs, axis=0), axis=0)
            d.metadata[self.cfg.emb_field] = emb
        return docs


# --------------------------- 5) Tête DP (Transformer) -------------------

@dataclass
class DPHeadConfig:
    bundle_path: str                     # chemin .joblib
    mode: str = "predict"                # "train" | "predict"
    # hyperparams LR
    C: float = 1.0
    max_iter: int = 200
    # champs d'I/O
    emb_field: str = "emb"
    gold_field: str = "gold_dp"          # gold str par doc si train
    pred_field: str = "pred_dp"          # sortie: code DP

class TransformerDPHeadOp(Operation):
    """Multiclasse DP sur embeddings : entraînement (LogReg) ou prédiction."""
    def __init__(self, cfg: DPHeadConfig):
        super().__init__()
        self.cfg = cfg
        self._LR = LogisticRegression
        self._bundle: Optional[Dict[str, Any]] = None

    def _load_bundle(self) -> Optional[dict]:
        if os.path.exists(self.cfg.bundle_path):
            return joblib.load(self.cfg.bundle_path)
        return None

    def _save_bundle(self, bundle: dict):
        os.makedirs(os.path.dirname(self.cfg.bundle_path), exist_ok=True)
        joblib.dump(bundle, self.cfg.bundle_path)

    def _fit(self, docs: Sequence[TextDocument]) -> dict:
        embs, labels = [], []
        for d in docs:
            emb = d.metadata.get(self.cfg.emb_field)
            gold = d.metadata.get(self.cfg.gold_field)
            if emb is None or not gold:
                continue
            embs.append(emb); labels.append(gold)
        if not embs:
            raise RuntimeError("Aucune donnée exploitable pour l'entraînement (embeddings/gold manquants)")

        X = np.vstack(embs)
        classes = sorted(list(set(labels)))
        y = np.array([classes.index(c) for c in labels], dtype=int)

        clf = self._LR(C=self.cfg.C, max_iter=self.cfg.max_iter, solver="lbfgs", multi_class="auto")
        clf.fit(X, y)
        bundle = {
            "meta": {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "classes": classes},
            "clf": clf,
        }
        return bundle

    def _predict(self, docs: Sequence[TextDocument], bundle: dict) -> Sequence[TextDocument]:
        clf = bundle["clf"]
        classes = bundle["meta"]["classes"]
        X = np.vstack([d.metadata[self.cfg.emb_field] for d in docs])
        proba = clf.predict_proba(X)
        top = np.argmax(proba, axis=1)
        for d, j in zip(docs, top):
            d.metadata[self.cfg.pred_field] = classes[j]
        return docs

    def run(self, docs: Sequence[TextDocument]):
        mode = self.cfg.mode
        if mode == "train":
            bundle = self._fit(docs)
            self._save_bundle(bundle)
            self._bundle = bundle
            return self._predict(docs, bundle)
        elif mode == "predict":
            bundle = self._load_bundle()
            if bundle is None:
                raise FileNotFoundError(f"Bundle DP introuvable: {self.cfg.bundle_path}")
            self._bundle = bundle
            return self._predict(docs, bundle)
        else:
            raise ValueError("mode doit être 'train' ou 'predict'")


# --------------------------- 6) LLM DP (Transformers) -------------------

@dataclass
class LLMDPConfig:
    hf_model: str
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    field_in: str = "text_rw"
    pred_field: str = "pred_dp"

class LLMDPInferenceOp(Operation):
    """Prédiction DP via LLM HF local : prompt court + extraction du premier code ICD-10 plausible."""
    def __init__(self, cfg: LLMDPConfig):
        super().__init__()
        self.cfg = cfg
        self._tok = AutoTokenizer.from_pretrained(self.cfg.hf_model)
        self._lm = AutoModelForCausalLM.from_pretrained(self.cfg.hf_model, device_map="auto")
        self._lm.eval()

    @torch.no_grad()
    def _infer_dp(self, text: str) -> Optional[str]:
        if not text:
            return None
        prompt = (
            "Tu es un codeur hospitalier expert. Lis le texte clinique et renvoie UNIQUEMENT le code CIM-10 "
            "du diagnostic principal (DP). Pas d'explication, rien d'autre.\n\n"
            f"{text.strip()}\n\nDP:"
        )
        tok = self._tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
        tok = {k: v.to(self._lm.device) for k, v in tok.items()}
        out = self._lm.generate(
            **tok,
            do_sample=(self.cfg.temperature > 0.0),
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_new_tokens=self.cfg.max_new_tokens,
            pad_token_id=self._tok.eos_token_id,
        )
        gen = self._tok.decode(out[0], skip_special_tokens=True)
        # Récupérer après le dernier "DP:"
        part = gen.split("DP:")[-1].strip()
        code = first_icd10(part) or first_icd10(gen)
        return code

    def run(self, docs: Sequence[TextDocument]):
        for d in docs:
            text = d.metadata.get(self.cfg.field_in) or d.metadata.get("text_norm") or d.text
            d.metadata[self.cfg.pred_field] = self._infer_dp(text) or ""
        return docs
