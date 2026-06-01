# champslibres-pipeline

Pipeline semi-automatisé pour le traitement des **champs libres** (réponses
textuelles courtes) des enquêtes statistiques : statistiques descriptives,
exploration par clustering, construction d'un *label book*, classification par
grand modèle de langage (LLM), puis évaluation des accords (humains / modèles).

Conçu pour le **SSP Cloud** (Onyxia), avec les modèles de `llm.lab` et le
stockage **S3** (MinIO). Tout fonctionne en CPU.

## 1. Installation (uv)

```bash
pip install uv            # si uv n'est pas déjà présent
uv sync --extra nlp       # coeur + clustering BERTopic, embeddings, MLflow, figures
uv run champslibres --help
```

> La commande s'appelle **`champslibres`** (pas le nom du dossier du dépôt).
> Elle n'existe qu'après `uv sync`, et se lance via `uv run champslibres …`.

## Structure du dépôt

```
champslibres_enquetes/            (racine du dépôt)
├── pyproject.toml                déclare les dépendances et la commande `champslibres`
├── README.md
├── .gitignore
├── champslibres_pipeline/        le package Python
│   ├── __init__.py  config.py  s3io.py  paths.py  project.py  cli.py
│   ├── llm/          client llm.lab (OpenAI-compatible)
│   ├── prompts/      prompts éditables (gec, label, classf)
│   └── steps/        step00_describe … step08_report
├── configs/          exemple.yaml, auteur_isi2026.yaml  (TA configuration)
├── scripts/          list_models.py, demo_*.py, make_eval_dataset.py, test_*.py
└── data/             échantillons locaux (NON versionnés : secret statistique)
```

Le **label book révisé par un humain** est une donnée : il ne va pas dans le
dépôt mais sur S3 (voir §6).

## 2. Secrets et variables d'environnement (SSP Cloud)

Injectées automatiquement par Onyxia : `AWS_S3_ENDPOINT`, `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` (accès S3) et, si le service MLflow
est allumé, `MLFLOW_TRACKING_URI`.

À ajouter une fois dans **Vault → « Mes secrets »** puis à injecter au lancement
du service : `LLM_LAB_API_KEY` (clé de l'API llm.lab).

```bash
uv run champslibres list-models        # vérifie l'accès et liste les modèles llm.lab
```

## 3. Configuration d'un projet

Toute la configuration tient dans **un fichier YAML** (voir `configs/exemple.yaml`
et `configs/auteur_isi2026.yaml`). Tu peux en générer un :

```bash
uv run champslibres init --project AUTEUR_ISI2026 \
    --source projet-champs-libres-vrs/AUTEUR/data/anon_AUTEUR_22_25_Type1.csv
```

Champs principaux : `project` (nom du projet), `bucket`, `io.source_csv` (CSV
brut : `id` + `texte`), `columns` (SEQ_ID / text_brut), `survey` (question +
modalités Rxx éventuelles), `label.grouping` (regroupement Fxx), `classf`
(classification), `eval` (jeu annoté, colonnes humaines, modèles à comparer).

## 4. Organisation des sorties sur S3

Chaque projet écrit ses artefacts sous `{bucket}/{outputs_prefix}/{project}/` :

```
00_describe/   stats descriptives (nb de mots)
01_gec/        texte corrigé/normalisé
02_dedup/      réponses dédupliquées + table de correspondance
03_embed/      embeddings (cache) + ids
04_cluster/    clusters.csv, keywords.csv, representative.csv
05_label/      label_book.json (auto), label_book_human.json (révisé), labels.csv, dendrogramme.html
06_classify/   classifications.csv
07_eval/       pairwise.csv, by_type.csv, fleiss.json, distribution_par_classe.csv/.png, accords.csv
config.yaml    copie de la config utilisée (traçabilité)
```

Les **données brutes** et le **jeu annoté** restent à leur emplacement S3 ; on
n'écrit que les artefacts ici.

## 5. Commandes (CLI)

```bash
uv run champslibres list-models                       # modèles llm.lab disponibles
uv run champslibres scaffold        -c configs/auteur_isi2026.yaml  # crée l'arborescence S3 du projet
uv run champslibres describe        -c configs/auteur_isi2026.yaml  # stats descriptives
uv run champslibres build-labelbook -c configs/auteur_isi2026.yaml  # étapes 00->05 (+ dendrogramme)
uv run champslibres classify        -c configs/auteur_isi2026.yaml  # étape 06 (label book auto ou humain)
uv run champslibres eval            -c configs/auteur_isi2026.yaml  # étape 07-08 (accords + tables/figure)
uv run champslibres run             -c configs/auteur_isi2026.yaml  # 00->06 d'un coup (label book auto)
uv run champslibres rerun           -c configs/auteur_isi2026.yaml  # re-cluster/label/classif depuis le cache embeddings
```

## 6. Workflow avec label book révisé par un humain

1. `champslibres scaffold -c config.yaml`
   → crée l'arborescence du projet sur S3 (dossiers visibles dans la console MinIO).
2. `champslibres build-labelbook -c config.yaml`
   → produit `05_label/label_book.json` (automatique) et `05_label/dendrogramme.html`.
3. Un expert **révise** le label book (fusion/renommage des Cxx, super-catégories
   Rxx/Fxx). Le fichier révisé peut être au format « long » (`Super_cat`,
   `Super_label`, `Super_description`, `Sub_cat`, `Sub_label`, `Sub_description`)
   ou au format interne imbriqué ; **les deux sont acceptés**.
4. Dépose le fichier révisé sur S3 en `05_label/label_book_human.json` et renseigne
   son chemin dans **`classf.label_book`** (le seul champ qui pilote le label book
   utilisé en aval).
5. `champslibres classify -c config.yaml` → classe avec le **label book humain**.
6. `champslibres eval -c config.yaml` → accords humain-humain / humain-modèle /
   modèle-modèle + tables et figure.

Le LLM classe toujours au niveau des **feuilles** (Cxx ou Rxx-feuille) ; le
mapping vers les **super-catégories** (Rxx/Fxx) se fait via le label book, et
c'est à ce niveau que se font toutes les comparaisons.

## 7. API Projet (Python / notebook / Streamlit)

```python
from champslibres_pipeline import Project
p = Project("configs/auteur_isi2026.yaml")
p.build_label_book()     # 00 -> 05
# ... révision humaine du label book, dépôt sur S3, classf.label_book renseigné ...
p.classify()             # 06
res = p.evaluate()       # 07 -> 08 ; res["accords"], res["distribution"]
```

## 8. Suivi MLflow

Si le service MLflow est allumé (`MLFLOW_TRACKING_URI` injectée), le clustering
et l'évaluation sont journalisés dans une **expérience portant le nom du projet**
(paramètres, métriques d'accord, tables et figure en artefacts).

## 9. Tests (hors-ligne)

```bash
uv run python scripts/test_step05.py    # label book + regroupement
uv run python scripts/test_step06.py    # classification
uv run python scripts/test_step07.py    # métriques d'accord
uv run python scripts/test_wiring.py    # S3 + API Projet + CLI (simulé en local)
```

> Données réelles = secret statistique : ne pas committer `data/` ni les sorties
> dans git ; tout reste sur S3.