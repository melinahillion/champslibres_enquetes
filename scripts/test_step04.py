from __future__ import annotations

"""
Test de l'étape 04 (clustering BERTopic + MLflow), à lancer :

    uv sync --extra nlp                 # une fois, pour installer BERTopic & co.
    uv run python scripts/test_step04.py

Deux vérifications :
  1. sélection des exemples représentatifs (numpy seul, toujours) ;
  2. clustering BERTopic complet + journalisation MLflow, sur données
     synthétiques (3 groupes) — si l'extra [nlp] est installé, sinon ignoré.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champslibres_pipeline.config import load_config
from champslibres_pipeline.steps.step04_cluster import (
    cluster_embeddings,
    keywords_dataframe,
    log_to_mlflow,
    select_representative_docs,
)


def test_representative_docs() -> None:
    print("\n1) Sélection des exemples représentatifs (numpy seul)")
    # trois centres nets ; deux points par centre
    embeddings = np.array(
        [[10, 0], [9, 1], [0, 10], [1, 9], [-10, 0], [-9, -1]], dtype=np.float32
    )
    labels = np.array([0, 0, 1, 1, 2, 2])
    ids = ["a", "b", "c", "d", "e", "f"]
    texts = ["t0", "t1", "t2", "t3", "t4", "t5"]
    reps = select_representative_docs(embeddings, labels, ids, texts, top_n=2)
    print("   clusters :", sorted(reps.keys()))
    assert set(reps.keys()) == {0, 1, 2}
    assert all(len(v) == 2 for v in reps.values())
    print("[OK ] un top-N d'exemples par cluster, bruit (-1) ignoré")


def test_full_clustering(cfg) -> None:
    print("\n2) Clustering BERTopic complet + MLflow")
    try:
        import bertopic  # noqa: F401
    except ImportError:
        print("[--] BERTopic non installé. Lance d'abord : uv sync --extra nlp")
        return

    from sklearn.datasets import make_blobs

    # 3 groupes synthétiques d'embeddings (dim 64)
    X, y = make_blobs(n_samples=120, n_features=64, centers=3, cluster_std=1.5, random_state=0)
    X = X.astype(np.float32)
    # un vocabulaire distinct par groupe -> mots-clés c-TF-IDF lisibles
    vocab = {
        0: "voisin bruit tapage quartier",
        1: "collègue bureau travail réunion",
        2: "ami soirée fête sortie",
    }
    texts = [vocab[int(lab)] for lab in y]
    ids = [f"id{i}" for i in range(len(X))]

    # réglages adaptés à ce petit jeu de test
    cfg.cluster.umap.n_neighbors = 10
    cfg.cluster.umap.n_components = 5
    cfg.cluster.hdbscan.min_cluster_size = 5
    cfg.cluster.hdbscan.min_samples = 3

    res = cluster_embeddings(texts, X, ids, cfg)
    print(f"   clusters = {res['n_clusters']}, bruit = {res['n_noise']} "
          f"(ratio {res['noise_ratio']:.2f})")
    print(keywords_dataframe(res["keywords"]).to_string(index=False))
    assert res["n_clusters"] >= 2, "on attend au moins 2 clusters sur ce jeu synthétique"

    run_id = log_to_mlflow(cfg, res, embedding_dim=X.shape[1])
    print("   run MLflow :", run_id or "(journalisation désactivée/indisponible)")
    print("[OK ] BERTopic a regroupé les données et MLflow a journalisé l'essai")


def main() -> int:
    cfg = load_config("configs/exemple.yaml")
    test_representative_docs()
    test_full_clustering(cfg)
    print("\nÉtape 04 vérifiée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
