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

ICD10_REGEX = re.compile(r"\b[A-TV-Z][0-9]{2}(?:\.[0-9A-TV-Z]{0,4})?\b")

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
            "Réécris et résume le compte rendu hospitalier ci-dessous en gardant le sens clinique, "
            "en français clair, sans inventer d'informations, sachant que le but final sera de pouvoir éxtraire le diagnostic principal. " + target +
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
    def __init__(self, cfg: ChunkingConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.hf_model, use_fast=True)

    def run(self, docs: Sequence[TextDocument]):
        for d in docs:
            text = (d.metadata.get(self.cfg.field_in) or d.text or "").strip()
            if not text:
                d.metadata[self.cfg.field_out] = []
                continue

            enc = self.tok(
                text,
                return_overflowing_tokens=True,
                truncation=True,
                max_length=self.cfg.chunk_size,
                stride=self.cfg.overlap,           # <-- overlap
                return_offsets_mapping=False,
                add_special_tokens=True,
            )

            chunks = []
            for ids in enc["input_ids"]:
                chunk_txt = self.tok.decode(ids, skip_special_tokens=True)
                chunks.append(chunk_txt)

            d.metadata[self.cfg.field_out] = chunks
        return list(docs)

class AggregateChunksOp(Operation):
    def __init__(self, strategy: str = "mean",
                 chunks_emb_field: str = "chunk_embs",
                 emb_field: str = "emb"):
        super().__init__()
        self.strategy = strategy
        self.chunks_emb_field = chunks_emb_field
        self.emb_field = emb_field

    def run(self, docs: Sequence[TextDocument]):
        for d in docs:
            arrs = d.metadata.get(self.chunks_emb_field) or []
            if len(arrs) == 0:
                d.metadata[self.emb_field] = None
                continue
            X = np.vstack(arrs)
            if self.strategy == "mean":
                emb = X.mean(0)
            else:
                emb = X.mean(0)  # fallback
            d.metadata[self.emb_field] = emb.astype(np.float32)
        return list(docs)



# --------------------------- 4) Embeddings Transformer ------------------

@dataclass
class EmbedConfig:
    hf_model: str
    device: str = "cpu"         # "cpu" | "cuda" | "auto"
    max_length: int = 512
    pooling: str = "cls4"
    cls_layers: int = 4
    chunks_field: str = "chunks"
    emb_field: str = "emb"

class TransformerEmbedOp(Operation):
    def __init__(self, cfg: EmbedConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.hf_model, use_fast=True)
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(cfg.hf_model, output_hidden_states=True)
        if cfg.device == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev = cfg.device
        self.device = dev
        self.enc.to(self.device).eval()

    @torch.no_grad()
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            D = self.enc.config.hidden_size
            return np.zeros((0, D), dtype=np.float32)
        batch = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt"
        ).to(self.device)
        out = self.enc(**batch)

        if self.cfg.pooling == "cls4":
            hs = torch.stack(out.hidden_states[-self.cfg.cls_layers:])  # (L,B,T,D)
            embs = hs[:, :, 0, :].mean(0)  # (B,D)
        else:
            last = out.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            embs = (last * mask).sum(1) / mask.sum(1).clamp(min=1)

        embs = torch.nn.functional.normalize(embs, dim=1)
        return embs.cpu().numpy()

    def run(self, docs: Sequence[TextDocument]):
        for d in docs:
            chunks = d.metadata.get(self.cfg.chunks_field)
            if not chunks:
                chunks = [ (d.metadata.get("text_rw") or d.text or "") ]
            embs = self._embed_texts(chunks)
            # Agrégation plus tard → on stocke les embs de chunks pour l’instant ?
            # Si tu veux déjà agrégér ici, tu peux faire une moyenne directement.
            d.metadata["chunk_embs"] = [e for e in embs]
        return list(docs)



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
                raise FileNotFoundError(f"Bundle DP introuvable: {self.cfg.bundle_path}, veuillez train un model avant de predict.")
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
