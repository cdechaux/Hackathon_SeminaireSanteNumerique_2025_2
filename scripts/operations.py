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
import regex as re
import json
import time
import joblib
import numpy as np
import torch
import unicodedata

from sklearn.linear_model import LogisticRegression

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification

from medkit.core.operation import Operation
from medkit.core.text import TextDocument

# --------------------------- Utils génériques ---------------------------

ICD10_REGEX = re.compile(r"\b[A-TV-Z][0-9]{2}(?:\.[0-9A-TV-Z]{0,4})?\b")

def first_icd10(text: str) -> Optional[str]:
    m = ICD10_REGEX.search(text or "")
    return m.group(0) if m else None


_RE_WORD = re.compile(r"\b[\p{L}\p{N}][\p{L}\p{N}\-_/]*\b", re.UNICODE)
_RE_UPPER_ABBR = re.compile(r"\b[A-ZÀ-Ý0-9]{2,}\b")
_RE_ABBR_DOTTED = re.compile(r"\b(?:[A-Za-z]\.){2,}[A-Za-z]?\b")

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def _tokenize_words(text: str) -> list[str]:
    return _RE_WORD.findall(text)

def _split_sentences(text: str) -> list[str]:
    # split grossière par ponctuation forte + sauts de ligne
    return [s.strip() for s in re.split(r"[\.!?;\n]+", text) if s.strip()]

def _count_sections_by_blanklines(text: str) -> int:
    # sections = blocs séparés par ≥2 sauts de ligne
    blocks = [b for b in re.split(r"\n{2,}", text) if b.strip()]
    return len(blocks)

def _count_abbr(text: str) -> int:
    return len(_RE_UPPER_ABBR.findall(text)) + len(_RE_ABBR_DOTTED.findall(text))



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


# --------------------------- 2) Rewrite --------------------------------


@dataclass
class MetricsTextConfig:
    text_field: str = "text"          # clé de lecture du texte
    metrics_root: str = "metrics"     # ou ecrire les metriques, ex :d.metadata["metrics"][phase] = {...}
    phase: str = "before"             # "before" | "after"
    lowercase: bool = True
    remove_accents: bool = True

class MetricsTextOp(Operation):
    """Calcule des métriques de texte et les écrit dans d.metadata['metrics'][phase]."""
    def __init__(self, cfg: MetricsTextConfig):
        super().__init__()
        self.cfg = cfg

    def _normalize_for_vocab(self, tokens: list[str]) -> list[str]:
        out = tokens
        if self.cfg.lowercase:
            out = [t.lower() for t in out]
        if self.cfg.remove_accents:
            out = [_strip_accents(t) for t in out]
        return out

    def _compute(self, text: str) -> Dict[str, float]:
        if not text:
            return {
                "len_chars": 0,
                "len_words": 0,
                "sent_len_avg": 0.0,
                "lexicon_size": 0,
                "n_sections": 0,
                "n_abbr": 0,
            }
        words = _tokenize_words(text)
        sents = _split_sentences(text)
        norm_words = self._normalize_for_vocab(words)

        len_chars = len(text)
        len_words = len(words)
        sent_len_avg = (sum(len(_tokenize_words(s)) for s in sents) / len(sents)) if sents else 0.0
        lexicon_size = len(set(norm_words))
        n_sections = _count_sections_by_blanklines(text)
        n_abbr = _count_abbr(text)

        return {
            "len_chars": int(len_chars),
            "len_words": int(len_words),
            "sent_len_avg": float(round(sent_len_avg, 3)),
            "lexicon_size": int(lexicon_size),
            "n_sections": int(n_sections),
            "n_abbr": int(n_abbr),
        }

    def run(self, docs: Sequence[TextDocument]):
        for d in docs:
            text = d.metadata.get(self.cfg.text_field, d.text or "")
            stats = self._compute(text)
            d.metadata.setdefault(self.cfg.metrics_root, {})
            d.metadata[self.cfg.metrics_root][self.cfg.phase] = stats
        return list(docs)


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

        assert self._tok is not None and self._lm is not None

        # 1) Messages structurés (spécifiques au modèle Instruct)
        target = f" (~{self.cfg.target_words} mots)" if self.cfg.target_words else ""
        sys_msg = (
            "Tu es un assistant clinique. Réécris et condense un compte rendu hospitalier "
            "en français clair, sans inventer d'informations, en préservant les diagnostics, "
            "pathologies et conduites thérapeutiques, pour faciliter l’extraction du Diagnostic principal. Commence toujours par la conclusion s'il y en a une."
        )
        user_msg = (
            f"Réécris le texte suivant{target}. Ne copie pas mot à mot, synthétise :\n\n{text.strip()}"
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": user_msg},
        ]

        # 2) Prompt via le chat template DU TOKENIZER DU MODÈLE
        # (Très important : template propre à Mistral-Instruct)
        prompt_ids = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        prompt_ids = prompt_ids.to(self._lm.device)
        input_len = prompt_ids.shape[1]

        # 3) Génération
        out_ids = self._lm.generate(
            prompt_ids,
            do_sample=True,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_new_tokens=self.cfg.max_new_tokens,
            eos_token_id=self._tok.eos_token_id,
            pad_token_id=self._tok.eos_token_id,
            repetition_penalty=1.1,   # aide à éviter le copiage
        )

        # 4) NE GARDER QUE LA RÉPONSE (nouveaux tokens)
        new_tokens = out_ids[0, input_len:]
        gen = self._tok.decode(new_tokens, skip_special_tokens=True).strip()

        # 5) Nettoyage léger
        return gen


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
        

# --------------------------- 6) Transformer finetune ------------------

@dataclass
class HFDocPredictConfig:
    checkpoint_dir: str                # dossier du checkpoint HF (celui de train_finetune_dp.py)
    device: str = "auto"               # "cuda" / "cpu" / "auto"
    max_length: int = 384              # par sécurité si chunks absents
    stride: int = 64                   # idem
    chunks_field: str = "chunks"       # fourni par ChunkingOp (list[str])
    pred_field: str = "pred_dp"        # sortie
    aggregate: str = "mean"            # "mean" | "max" | "median"
    return_proba: bool = True

class HFDocClassifierOp(Operation):
    """Inférence DP avec un modèle HF fine-tuné (doc-level via agrégation des logits de chunks)."""

    def __init__(self, cfg: HFDocPredictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.checkpoint_dir)
        self.model.eval()
        dev = cfg.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        self.model.to(self.device)

        # id2label depuis config
        self.id2label = self.model.config.id2label if hasattr(self.model.config, "id2label") else {}
        # fallback si jamais non présent
        if not self.id2label or not isinstance(list(self.id2label.keys())[0], int):
            # HF peut sauver id2label avec clés str; normalisons
            try:
                self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}  # type: ignore
            except Exception:
                pass

    def _aggregate(self, logits: np.ndarray) -> np.ndarray:
        if self.cfg.aggregate == "mean":
            return logits.mean(axis=0)
        if self.cfg.aggregate == "max":
            return logits.max(axis=0)
        if self.cfg.aggregate == "median":
            return np.median(logits, axis=0)
        return logits.mean(axis=0)

    @torch.no_grad()
    def run(self, docs: Sequence[TextDocument]):  # type: ignore[override]
        for d in docs:
            chunks: Optional[List[str]] = d.metadata.get(self.cfg.chunks_field)  # type: ignore
            if not chunks:
                chunks = [d.text]

            chunk_logits = []
            for ch in chunks:
                enc = self.tokenizer(
                    ch, truncation=True, padding=False,
                    max_length=self.cfg.max_length, return_tensors="pt"
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc)
                logit = out.logits.detach().cpu().numpy()  # (1, num_labels)
                chunk_logits.append(logit[0])

            logits = np.stack(chunk_logits, axis=0)          # (n_chunks, num_labels)
            agg = self._aggregate(logits)                     # (num_labels,)
            pred_id = int(np.argmax(agg))
            label = self.id2label.get(pred_id, str(pred_id))

            if self.cfg.return_proba:
                proba = float(torch.softmax(torch.tensor(agg), dim=-1)[pred_id].item())
                d.metadata[self.cfg.pred_field] = {"code": label, "score": proba}
            else:
                d.metadata[self.cfg.pred_field] = {"code": label}

        return list(docs)



# --------------------------- 7) LLM DP (Transformers) -------------------

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
