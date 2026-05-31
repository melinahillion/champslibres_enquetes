# champslibres-pipeline

Pipeline semi-automatisé pour le traitement des **champs libres** (réponses
textuelles courtes) des enquêtes statistiques : correction du texte,
déduplication, embeddings, exploration par clustering (BERTopic), construction
d'un *label book*, classification par grand modèle de langage, puis évaluation.

Conçu pour le **SSP Cloud** (Onyxia), avec les modèles de `llm.lab`. Tout
fonctionne en **CPU par défaut** ; le GPU n'est pas nécessaire.

> **État** : socle + étapes 01 à 04 (cœurs) construits et testés.
> Étapes 05–07, câblage S3 (étape 00 + API `Project`) et interface Streamlit à venir.

---

## 1. Prérequis

- Un service du SSP Cloud (**VSCode-python**, ou **VSCode-pytorch** pour l'étape 04).
- Python ≥ 3.12 (fourni par le service).
- `uv` (gestionnaire de paquets). S'il n'est pas présent : `pip install uv`.
- Une clé d'API `llm.lab` placée dans Vault (voir §2).

---

## 2. Variables d'environnement (secrets)

Le pipeline ne stocke **aucun secret** dans le code : il lit des variables
d'environnement. On ne met dans le YAML que le *nom* de ces variables.

| Variable | Rôle | Comment l'obtenir |
|---|---|---|
| `LLM_LAB_API_KEY` | clé de l'API llm.lab | **à ajouter dans Vault** (« Mes secrets »), puis à injecter au lancement du service |
| `AWS_S3_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` | accès au stockage S3 | **injectées automatiquement** par le SSP Cloud |
| `MLFLOW_TRACKING_URI` | serveur MLflow (étape 04) | injectée automatiquement si le service MLflow est activé ; sinon les runs vont en local dans `./mlruns` |
| `HF_TOKEN` | (optionnel) modèle d'embedding « fermé » depuis HuggingFace | à ajouter dans Vault si besoin |

Vérifier que les variables sont bien présentes :

```bash
echo $AWS_S3_ENDPOINT
[ -n "$LLM_LAB_API_KEY" ] && echo "clé LLM présente"
```

---

## 3. Installation avec uv

```bash
pip install uv                 # si uv n'est pas déjà présent

uv sync                        # installe le COEUR (config, LLM, S3, dédup, éval)
uv sync --extra nlp            # AJOUTE embeddings + clustering BERTopic + MLflow (étapes 03-04)
```

Extras disponibles :

| Commande | Ce qu'elle ajoute | Quand |
|---|---|---|
| `uv sync` | cœur (CPU, léger) | toujours |
| `uv sync --extra nlp` | BERTopic, UMAP, HDBSCAN, sentence-transformers, MLflow | étape 04 (et embedding HuggingFace optionnel) |
| `uv sync --extra ui` | Streamlit | interface (à venir) |

> `--extra nlp` installe `torch` (dépendance de BERTopic) : c'est volumineux,
> prévois quelques Go d'espace disque. Le GPU n'est **pas** nécessaire.

---

## 4. « Tout est-il déjà dans .venv grâce à uv ? »

En partie — et c'est pourquoi ces commandes restent documentées :

- `uv.lock` (versionné) **fige les versions** exactes des paquets : reproductibilité garantie.
- `.venv/` (l'environnement installé) **n'est pas versionné** (il est dans `.gitignore`).
  Sur un nouveau service, ou pour un clone du dépôt, il n'existe pas : il faut le recréer.
- `uv sync` **recrée `.venv`** à partir de `pyproject.toml` + `uv.lock`. Les extras
  sont **optionnels** : `uv sync` seul n'installe **pas** `[nlp]` ; il faut `uv sync --extra nlp`.

Raccourci : `uv run` synchronise le cœur automatiquement avant d'exécuter. Pour une
commande qui a besoin de l'extra `nlp` sans `uv sync --extra nlp` préalable :

```bash
uv run --extra nlp python scripts/test_step04.py
```

---

## 5. Tests / utilisation

```bash
uv run python scripts/test_socle.py     # socle : config + connexion llm.lab + modèles dispo
uv run python scripts/test_step01.py    # correction du texte (none / light / llm)
uv run python scripts/test_step02.py    # déduplication
uv run python scripts/test_step03.py    # embeddings (llm.lab par défaut, repli HuggingFace)

uv sync --extra nlp                      # requis pour l'étape 04
uv run python scripts/test_step04.py     # clustering BERTopic + MLflow
```

Les sections « RÉEL » de ces tests (appels llm.lab) ne s'exécutent que si
`LLM_LAB_API_KEY` est présente ; sinon elles sont ignorées proprement.

---

## 6. Avant de pousser sur GitHub

Le `.gitignore` exclut déjà `.venv/`, les caches, `mlruns/` et tout fichier de secrets.
Avant le premier `git push`, vérifier l'absence de secrets en clair et ce que git suit :

```bash
grep -rInE "sk-[A-Za-z0-9]{8,}|hf_[A-Za-z0-9]{8,}|(password|secret|api_key|token)[[:space:]]*[:=]" . --exclude-dir=.venv --exclude-dir=.git
git ls-files | grep -iE "\.env|secret|\.pem|\.key" || echo "aucun fichier sensible suivi"
```

**Données** : `data/` peut contenir de vraies réponses d'enquête. Vérifier les règles
de **secret statistique** avant publication (dépôt privé recommandé en cas de doute),
ou remplacer par un jeu fictif.

---

## 7. Structure du projet

```
champslibres_enquetes/
├── pyproject.toml              # dépendances + config du projet
├── uv.lock                     # versions figées (versionné)
├── README.md
├── .gitignore
├── champslibres_pipeline/      # le package (code)
│   ├── config.py               # config unifiée (YAML -> objet validé)
│   ├── llm/                    # couche LLM (llm.lab)
│   ├── prompts/                # prompts par défaut, éditables
│   └── steps/                  # étapes 01..04 (puis 05..07)
├── configs/
│   └── exemple.yaml            # un fichier de config par projet
├── scripts/                    # scripts de test
└── data/                       # données (voir §6 : confidentialité)
```
