from __future__ import annotations

"""
Démonstration de bout en bout sur tes données (data/exemple_input.csv ou autre).

    uv sync --extra nlp
    uv run python scripts/demo_pipeline.py

Enchaîne : correction légère -> déduplication -> embeddings (llm.lab)
-> clustering BERTopic -> label book (avec regroupement Fxx selon la config),
et ÉCRIT tous les résultats dans ./output/ (CSV + JSON + dendrogramme si demandé).

Nécessite l'extra [nlp] et LLM_LAB_API_KEY.
Adapte les seuils UMAP/HDBSCAN à la taille de TON échantillon.
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champslibres_pipeline.config import load_config
from champslibres_pipeline.steps.step01_gec import apply_gec
from champslibres_pipeline.steps.step02_dedup import dedup
from champslibres_pipeline.steps.step03_embed import embed_texts
from champslibres_pipeline.steps.step04_cluster import cluster_embeddings, cluster_code
from champslibres_pipeline.steps.step05_label import run_label_book

SAMPLE = "data/exemple_input.csv"
OUTDIR = "output"


def collect_cxx(book):
    """Récupère tous les Cxx (à plat ou sous leurs super-catégories)."""
    rows = []
    for cat in book["categories"]:
        if cat["code"].startswith("C"):
            rows.append({"category_id": cat["code"], "title": cat["label"], "description": cat["description"]})
        for ch in cat.get("children", []):
            rows.append({"category_id": ch["category_id"], "title": ch.get("title", ""),
                         "description": ch.get("description", "")})
    return rows


def main() -> int:
    if not os.environ.get("LLM_LAB_API_KEY"):
        print("[!] LLM_LAB_API_KEY absente : ajoute-la dans Vault puis relance.")
        return 1
    try:
        import bertopic  # noqa: F401
    except ImportError:
        print("[!] BERTopic non installé. Lance : uv sync --extra nlp")
        return 1

    from champslibres_pipeline.llm import get_llm_client

    os.makedirs(OUTDIR, exist_ok=True)
    cfg = load_config("configs/exemple.yaml")
    client = get_llm_client(cfg.llm)
    if not cfg.label.model:
        cfg.label.model = "gemma4-26b-moe"

    # >>> seuils à ADAPTER à la taille de l'échantillon <<<
    cfg.cluster.umap.n_neighbors = 8
    cfg.cluster.umap.n_components = 5
    cfg.cluster.hdbscan.min_cluster_size = 3
    cfg.cluster.hdbscan.min_samples = 2

    print("1/5  Lecture + correction légère + déduplication...")
    df = pd.read_csv(SAMPLE, dtype=str)
    cfg.gec.mode = "light"
    df = apply_gec(df, cfg)
    dedup_df, _ = dedup(df, cfg)
    ids = dedup_df[cfg.columns.id].tolist()
    texts = dedup_df[cfg.columns.clean].tolist()
    print(f"     {len(df)} réponses -> {len(dedup_df)} uniques")

    print("2/5  Embeddings via llm.lab...")
    emb = embed_texts(texts, cfg, llm_client=client)
    print(f"     matrice {emb.shape}")

    print("3/5  Clustering BERTopic...")
    res = cluster_embeddings(texts, emb, ids, cfg)
    print(f"     {res['n_clusters']} clusters, {res['n_noise']} 'bruit'")

    print(f"4/5  Label book (regroupement Fxx = '{cfg.label.grouping.method}')...")
    dendro = f"{OUTDIR}/dendrogramme"
    book = run_label_book(res["keywords"], res["representative_docs"], cfg,
                          llm_client=client, topic_model=res["topic_model"], docs=texts,
                          dendro_path=dendro)

    print("5/5  Écriture des résultats...")
    pd.DataFrame(
        [{"cluster": cluster_code(c), "n_docs": int((res["labels"] == c).sum()),
          "keywords": ", ".join(words)} for c, words in res["keywords"].items()]
    ).to_csv(f"{OUTDIR}/keywords.csv", index=False)
    pd.DataFrame({
        "id": ids, "text": texts,
        "cluster": [cluster_code(c) if c != -1 else "bruit" for c in res["labels"]],
    }).to_csv(f"{OUTDIR}/clusters.csv", index=False)
    pd.DataFrame(
        [{"cluster": cluster_code(c), "id": i, "text": t}
         for c, docs in res["representative_docs"].items() for (i, t) in docs]
    ).to_csv(f"{OUTDIR}/representative.csv", index=False)
    pd.DataFrame(collect_cxx(book)).to_csv(f"{OUTDIR}/labels.csv", index=False)
    with open(f"{OUTDIR}/label_book.json", "w", encoding="utf-8") as f:
        json.dump(book, f, ensure_ascii=False, indent=2)

    print(f"\nTerminé. Résultats dans ./{OUTDIR}/ :")
    for n in ("keywords.csv", "clusters.csv", "representative.csv", "labels.csv", "label_book.json"):
        print(f"   - {OUTDIR}/{n}")
    if cfg.label.grouping.method == "dendrogram":
        for ext in (".png", ".html"):
            if os.path.exists(dendro + ext):
                print(f"   - {dendro}{ext}  (dendrogramme à visualiser pour choisir le seuil)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
