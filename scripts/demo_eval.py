from __future__ import annotations

"""
Démonstration de l'étape 07. À lancer :  uv run python scripts/demo_eval.py

Lit le jeu annoté (eval.annotated_csv), classe les réponses avec chaque modèle
de eval.models (llm.lab), mappe Cxx -> Rxx/Fxx via le label book, puis calcule
les accords humain-humain / humain-modèle / modèle-modèle. Sans modèles (ou sans
clé llm.lab), l'évaluation porte sur les seuls annotateurs humains.
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champslibres_pipeline.config import load_config
from champslibres_pipeline.steps.step07_eval import classify_models, run_evaluation


def main():
    cfg = load_config("configs/exemple.yaml")
    id_col = cfg.eval.id_column or cfg.columns.id
    text_col = cfg.eval.text_column or cfg.columns.text

    annotated = cfg.eval.annotated_csv or "data/exemple_annotated.csv"
    if not os.path.exists(annotated):
        print(f"[!] {annotated} introuvable. Génère-le : uv run python scripts/make_eval_dataset.py")
        return 1
    df = pd.read_csv(annotated, dtype=str)

    book_path = cfg.classf.label_book or "output/label_book.json"
    if os.path.exists(book_path):
        book = json.load(open(book_path, encoding="utf-8"))
    else:
        print(f"[i] Pas de label book ({book_path}) : mapping Cxx->super en identité.")
        book = {"categories": []}

    models = list(cfg.eval.models)
    if models and os.environ.get("LLM_LAB_API_KEY"):
        from champslibres_pipeline.llm import get_llm_client
        client = get_llm_client(cfg.llm)
        print(f"Classification par {len(models)} modèle(s) : {models}")
        preds = classify_models(df[text_col].tolist(), df[id_col].tolist(), book, cfg,
                                llm_client=client, models=models)
        df = df.merge(preds, on=id_col, how="left")
    elif models:
        print("[i] LLM_LAB_API_KEY absente : évaluation sur les humains seuls.")
        cfg.eval.models = []

    res = run_evaluation(df, book, cfg)
    print("\n=== Accords par paire ===")
    print(res["pairwise"].to_string(index=False))
    print("\n=== Synthèse par type ===")
    print(res["by_type"].to_string(index=False))
    print("\n=== Kappa de Fleiss ===", res["fleiss"])
    print("\nJuges :", res["raters"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
