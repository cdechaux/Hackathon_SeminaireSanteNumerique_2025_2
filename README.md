# Hackathon_SeminaireSanteNumerique_2025_2

Outils pour coder le **Diagnostic Principal (DP)** à partir de comptes-rendus hospitaliers, avec un pipeline modulaire basé sur **medkit** et **Transformers**.
Le repo propose :

* un **pipeline simple** de base (embeddings + régression logistique),
* un **pipeline complet** et modulaire (normalisation, réécriture LLM, chunking, embeddings, plusieurs backends de classification),
* un **script de fine-tuning** d’un modèle HF,
* un **script de réécriture de dataset** avec métriques d'évaluation de texte,
* un **script d’évaluation**,
* des **jobs SLURM** prêts à l’emploi.

---

## Sommaire

* [Arborescence](#arborescence)
* [Prérequis & installation](#prérequis--installation)
* [Jeux de données (format attendu)](#jeux-de-données-format-attendu)
* [Démarrages rapides](#démarrages-rapides)

  * [Pipeline simple](#pipeline-simple)
  * [Pipeline complet et modulaire](#pipeline-complet-et-modulaire)
  * [Fine-tuning HF](#fine-tuning-hf)
  * [Réécriture d’un dataset + métriques](#réécriture-dun-dataset--métriques)
  * [Évaluation](#évaluation)
* [Backends de classification](#backends-de-classification)
* [Options principales](#options-principales)
* [Jobs SLURM](#jobs-slurm)
* [Bonnes pratiques & conseils](#bonnes-pratiques--conseils)
* [Dépannage](#dépannage)
* [Licence](#licence)

---

## Arborescence

```
.
├── assets/                         # (optionnel) poids, bundles, etc.
├── checkpoints/                    # modèles entraînés & artefacts (créés à l'exécution)
├── data/                           # jeux de données (CSV) & sorties
├── data/                           # Logs d'exectution des jobs
├── README.md
├── requirements.txt
└── scripts/
    ├── create_rewrite_dataset.py   # réécriture LLM + métriques texte (before/after)
    ├── eval_dp.py                  # évaluation gold vs prédictions (F1, etc.)
    ├── job_scripts/                # jobs SLURM
    │   ├── job_eval.batch
    │   ├── job_finetuning.batch
    │   ├── job_pipeline.batch
    │   └── job_rewrite.batch
    ├── operations.py               # opérations medkit (Normalize, Rewrite, Chunk, Embed, …)
    ├── run_pipeline.py             # pipeline complet (modulaire, plusieurs backends)
    ├── run_pipeline_simple.py      # pipeline minimal (exemple)
    └── train_finetune_dp.py        # fine-tuning Transformer HF (sequence classification)
```

---

## Prérequis & installation

* **Python 3.10 – 3.12** recommandé
* GPU NVIDIA (optionnel mais recommandé) + CUDA pour l’inférence/entraînement rapides

Installation (dans un venv/conda) :

```bash
pip install -r requirements.txt
```
---

## Jeux de données (format attendu)

CSV avec au minimum :

* `text` : texte du CR (ou `text_rw` si vous utilisez un dataset déjà réécrit),
* `code_sejour` : identifiant du séjour (restitué dans la sortie),
* (optionnel en train) `code_dp` : code DP gold.

Exemple minimal :

```csv
code_sejour,text,code_dp
SEJ001,"Texte clinique...",C34.1
SEJ002,"Un autre CR...",I21.4
```

---

## Démarrages rapides

### Pipeline simple

Embeddings Transformer gelé + régression logistique multiclasse :

```bash
python scripts/run_pipeline_simple.py \
  --input-csv data/dp_dataset.csv \
  --output-csv data/pred.csv \
  --mode predict \
  --hf-model almanach/camembert-bio-base
```

En mode **train** (entraîne la tête logistique puis prédit) :

```bash
python scripts/run_pipeline_simple.py \
  --input-csv data/dp_dataset.csv \
  --output-csv data/pred.csv \
  --mode train \
  --col-dp code_dp
```

---

### Pipeline complet et modulaire

Ajoutez/supprimez des étapes (normalisation, réécriture LLM, chunking, etc.).


Backend **Transformer** (prediction avec modele Transformer gelé) :

```bash
python scripts/run_pipeline.py \
  --input-csv data/dp_dataset.csv \
  --output-csv data/pred.csv \
  --backend transformer \
  --mode predict \
  --normalisation \
  --chunk-size 480 --chunk-overlap 64
```

Backend **HF finetuné** (modèle entraîné avec `train_finetune_dp.py`) :

```bash
python scripts/run_pipeline.py \
  --input-csv data/dp_dataset.csv \
  --output-csv data/pred_hf.csv \
  --backend hf_finetuned \
  --hf-checkpoint checkpoints/camembert_dp_ft/final
```


Réécriture LLM + métriques before/after intégrées au CSV :

```bash
python scripts/run_pipeline.py \
  --input-csv data/dp_dataset.csv \
  --output-csv data/pred_with_metrics.csv \
  --backend transformer \
  --mode predict \
  --normalisation \
  --rewrite --with-metrics \
  --rewrite-llm-model mistralai/Mistral-7B-Instruct-v0.3 \
  --rewrite-target-words 300 --rewrite-max-new-tokens 256
```

Backend **LLM** (prédiction DP directe) :

```bash
python scripts/run_pipeline.py \
  --input-csv data/dp_dataset.csv \
  --output-csv data/pred_llm.csv \
  --backend llm \
  --llm-model mistralai/Mistral-7B-Instruct-v0.3
```


---

### Fine-tuning HF

Entraîne un modèle de classification de séquence (CamemBERT-bio par défaut) :

```bash
python scripts/train_finetune_dp.py \
  --input-csv data/dp_dataset.csv \
  --text-col text \
  --label-col code_dp \
  --output-dir checkpoints/camembert_dp_ft \
  --pretrained almanach/camembert-bio-base \
  --epochs 5 --batch-size 16 --lr 2e-5 --fp16
```

Sorties (dans `--output-dir`) :

* `final/` : **config.json**, **model.safetensors**, **tokenizer** → utilisable par `run_pipeline.py --backend hf_finetuned`
* `data/train.csv`, `data/val.csv`, `label_map.json`, `split_stats.json`, `config_train.json`

> Le split est **par classe** (robuste aux classes rares) : au moins 1 échantillon en validation quand c’est possible, sinon uniquement en train.

---

### Réécriture d’un dataset + métriques

Génère un nouveau CSV avec `text_rw` (réécrit) et calcule des métriques **avant/après** :

```bash
python scripts/create_rewrite_dataset.py \
  --input-csv data/dp_dataset.csv \
  --output-csv data/dp_dataset_rw.csv \
  --text-col text \
  --llm-model mistralai/Mistral-7B-Instruct-v0.3 \
  --target-words 300 \
  --max-new-tokens 256 --temperature 0.3 --top_p 0.95
```

Produit aussi `data/dp_dataset_rw.metrics.json` avec les moyennes globales :

* `len_chars`, `len_words`, `sent_len_avg`, `lexicon_size`, `n_sections`, `n_abbr`.

---

### Évaluation

Comparer `code_dp` (gold) et `dp_predit` (prédictions) :

```bash
python scripts/eval_dp.py \
  --gold-csv data/dp_dataset.csv \
  --pred-csv data/pred.csv \
  --gold-col code_dp \
  --pred-col dp_predit
```

Affiche micro/macro F1, accuracy, etc.

---

## Backends de classification

* **transformer** : encodeur gelé (ex. CamemBERT) → embeddings → **LogisticRegression** multiclasse.
* **hf_finetuned** : modèle **entièrement finetuné** (SequenceClassification) → prédiction doc-level (avec agrégation si chunking).
* **llm** : LLM de génération (prompt court) → extraction du code ICD-10 via regex.

---

## Options principales

Quelques drapeaux utiles (voir `--help` pour le détail) :

* **I/O** : `--input-csv`, `--output-csv`, `--col-text`, `--col-dp`, `--col-sejour`
* **Prétraitement** : `--normalisation`, `--rewrite`, `--with-metrics`
* **Réécriture** : `--rewrite-llm-model`, `--rewrite-target-words`, `--rewrite-max-new-tokens`, `--rewrite-temperature`, `--rewrite-top-p`
* **Chunking** : `--chunk-size` (≈ 480), `--chunk-overlap` (≈ 64)
* **Embeddings** : `--hf-model`, `--pooling` (`cls`/`mean`), `--max-length`
* **Tête LR** : `--lr-C`, `--lr-max-iter`, `--bundle-path`
* **HF finetuné** : `--hf-checkpoint`, `--aggregate_hf`
* **LLM** : `--llm-model`, `--llm-max-new-tokens`, `--llm-temperature`, `--llm-top-p`

---

## Jobs SLURM

Des scripts de job slurm se trouvent dans `scripts/job_scripts/` :

* `job_pipeline.batch` : exécuter un pipeline
* `job_finetuning.batch` : entraîner un modèle HF
* `job_rewrite.batch` : réécrire un dataset
* `job_eval.batch` : évaluer des prédictions

Soumission :

```bash
sbatch scripts/job_scripts/job_pipeline.batch
```

Suivi :

```bash
squeue -u $USER
tail -f logs/<jobname>_<jobid>.out
```

> Adaptez `--partition`, `--gres=gpu:1`, `--mem`, `--time`, activation d’environnement, etc., au cluster.

---


