from __future__ import annotations

"""
Étape 04 — Clustering thématique avec BERTopic (CPU) + journalisation MLflow.

On regroupe les réponses par proximité de sens à partir des embeddings de
l'étape 03. On s'appuie sur le framework BERTopic, qui enchaîne :
  embeddings -> réduction de dimension (UMAP) -> regroupement (HDBSCAN)
  -> mots-clés par cluster (c-TF-IDF).

Tous les réglages UMAP/HDBSCAN viennent de la config (section `cluster`) et
sont donc modifiables. À chaque exécution, on enregistre dans MLflow les
paramètres utilisés et la qualité du résultat (nombre de clusters, part de
"bruit"...), pour pouvoir comparer les essais.

Nécessite l'extra [nlp]  ->  uv sync --extra nlp

Coeur réutilisable :
  cluster_embeddings(texts, embeddings, ids, cfg) -> dict de résultats
  log_to_mlflow(cfg, result, embedding_dim)       -> identifiant du run (ou None)
"""

import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import ProjectConfig

# Petite liste de mots vides français : rend les mots-clés c-TF-IDF plus lisibles.
# (c-TF-IDF atténue déjà les mots trop fréquents, mais retirer les mots-outils aide.)
FRENCH_STOPWORDS: List[str] = [
    "à", "au", "aux", "avec", "ce", "ces", "cette", "dans", "de", "des", "du",
    "elle", "en", "et", "eux", "il", "ils", "je", "la", "le", "les", "leur",
    "lui", "ma", "mais", "me", "mes", "moi", "mon", "ne", "nos", "notre", "nous",
    "on", "ou", "où", "par", "pas", "pour", "qu", "que", "qui", "sa", "se", "ses",
    "son", "sur", "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "vos",
    "votre", "vous", "c", "d", "j", "l", "m", "n", "s", "t", "y", "été", "être",
    "avoir", "fait", "faire", "est", "sont", "a", "ont", "plus", "très", "cela",
    "comme", "aussi", "autre", "autres", "non", "oui", "si", "leurs", "lors",
]


# ---------------------------------------------------------------------------
# Construction du modèle BERTopic à partir de la config
# ---------------------------------------------------------------------------
def build_topic_model(cfg: ProjectConfig):
    """
    Construit un BERTopic avec les réglages UMAP/HDBSCAN de la config.
    Les imports sont locaux : l'extra [nlp] n'est requis que pour cette étape.
    """
    try:
        from bertopic import BERTopic
        from bertopic.vectorizers import ClassTfidfTransformer
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as e:
        raise RuntimeError(
            "L'étape 04 nécessite l'extra [nlp]. Installe-le avec : "
            "uv sync --extra nlp"
        ) from e

    u = cfg.cluster.umap
    h = cfg.cluster.hdbscan

    umap_model = UMAP(
        n_neighbors=u.n_neighbors,
        n_components=u.n_components,
        min_dist=u.min_dist,
        metric=u.metric,
        random_state=u.random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=h.min_cluster_size,
        min_samples=h.min_samples,
        metric=h.metric,
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(stop_words=FRENCH_STOPWORDS)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        top_n_words=cfg.cluster.top_n_words,
        # IMPORTANT : "english" (défaut de BERTopic) supprime les caractères non-ASCII
        # des mots-clés (donc les accents). On passe la langue de la config pour
        # préserver les accents français. Les embeddings étant fournis, aucun
        # modèle d'embedding n'est téléchargé ici.
        language=cfg.cluster.language,
        calculate_probabilities=False,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Exemples représentatifs : plus proches du centroïde dans l'espace d'embedding
# ORIGINAL (et non l'espace réduit par UMAP), comme dans l'article.
# ---------------------------------------------------------------------------
def select_representative_docs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ids: Sequence[str],
    texts: Sequence[str],
    top_n: int,
) -> Dict[int, List[Tuple[str, str]]]:
    emb = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embn = emb / norms  # vecteurs normalisés -> produit scalaire = cosinus

    reps: Dict[int, List[Tuple[str, str]]] = {}
    for c in sorted(set(int(x) for x in labels)):
        if c == -1:  # -1 = bruit (réponses atypiques), on ignore
            continue
        idx = np.where(labels == c)[0]
        centroid = embn[idx].mean(axis=0)
        cnorm = np.linalg.norm(centroid) or 1.0
        sims = embn[idx] @ (centroid / cnorm)
        order = idx[np.argsort(-sims)][:top_n]
        reps[c] = [(str(ids[i]), str(texts[i])) for i in order]
    return reps


# ---------------------------------------------------------------------------
# Coeur : lancer le clustering
# ---------------------------------------------------------------------------
def cluster_embeddings(
    texts: Sequence[str],
    embeddings: np.ndarray,
    ids: Sequence[str],
    cfg: ProjectConfig,
) -> Dict[str, Any]:
    """
    Lance BERTopic sur les embeddings fournis et renvoie un dictionnaire :
      - labels               : cluster de chaque texte (-1 = bruit)
      - keywords             : {cluster: [mots-clés c-TF-IDF]}
      - representative_docs  : {cluster: [(id, texte), ...]}
      - n_clusters, n_noise, noise_ratio
      - topic_model          : l'objet BERTopic (pour dendrogramme, etc.)
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    topic_model = build_topic_model(cfg)
    labels, _ = topic_model.fit_transform(list(texts), embeddings)
    labels = np.asarray(labels)

    keywords: Dict[int, List[str]] = {}
    for c in sorted(set(int(x) for x in labels)):
        if c == -1:
            continue
        topic = topic_model.get_topic(c) or []
        keywords[c] = [w for w, _ in topic][: cfg.cluster.top_n_words]

    reps = select_representative_docs(
        embeddings, labels, ids, list(texts), cfg.cluster.nr_representative_docs
    )

    n_noise = int((labels == -1).sum())
    n_clusters = len([c for c in set(int(x) for x in labels) if c != -1])

    return {
        "labels": labels,
        "keywords": keywords,
        "representative_docs": reps,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": float(n_noise) / max(1, len(labels)),
        "topic_model": topic_model,
    }


# ---------------------------------------------------------------------------
# Tableaux pratiques (pour affichage / export ultérieur)
# ---------------------------------------------------------------------------
def clusters_dataframe(ids, texts, labels) -> pd.DataFrame:
    return pd.DataFrame({"id": list(map(str, ids)), "text": list(texts), "cluster": list(labels)})


def keywords_dataframe(keywords: Dict[int, List[str]]) -> pd.DataFrame:
    rows = [{"cluster": c, "keywords": ", ".join(words)} for c, words in keywords.items()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Journalisation MLflow
# ---------------------------------------------------------------------------
def log_to_mlflow(cfg: ProjectConfig, result: Dict[str, Any], embedding_dim: int):
    """
    Enregistre paramètres + métriques du clustering dans MLflow.
    Ne fait rien si la journalisation est désactivée ou si MLflow n'est pas installé.
    Renvoie l'identifiant du run, ou None.
    """
    if not (cfg.mlflow.enabled and cfg.cluster.mlflow.enabled):
        return None
    try:
        import mlflow
    except ImportError:
        print("[mlflow] non installé (extra [nlp]) : journalisation ignorée.")
        return None

    uri = os.environ.get(cfg.mlflow.tracking_uri_env)
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg.cluster.mlflow.experiment)

    u = cfg.cluster.umap
    h = cfg.cluster.hdbscan
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "embedding_backend": cfg.embed.backend,
                "embedding_model": cfg.embed.model_name or "auto",
                "embedding_dim": embedding_dim,
                "umap_n_neighbors": u.n_neighbors,
                "umap_n_components": u.n_components,
                "umap_min_dist": u.min_dist,
                "umap_metric": u.metric,
                "hdbscan_min_cluster_size": h.min_cluster_size,
                "hdbscan_min_samples": h.min_samples,
                "hdbscan_metric": h.metric,
                "top_n_words": cfg.cluster.top_n_words,
            }
        )
        mlflow.log_metrics(
            {
                "n_clusters": result["n_clusters"],
                "n_noise": result["n_noise"],
                "noise_ratio": round(result["noise_ratio"], 4),
                "n_docs": int(len(result["labels"])),
            }
        )
        return run.info.run_id
