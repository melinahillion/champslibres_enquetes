from __future__ import annotations

"""
Génère un jeu de données annoté par des humains SYNTHÉTIQUES, au bon format pour
l'étape 07. À lancer :

    uv run python scripts/make_eval_dataset.py --humans 2 --agreement 0.8

Produit data/exemple_annotated.csv avec :
  - les colonnes id / texte (columns.id, columns.text de la config),
  - une colonne par annotateur (humain1, humain2, ...) contenant une
    super-catégorie (Rxx/Fxx).

Les annotations sont synthétiques : une "vérité" latente tirée au hasard parmi
les super-catégories, puis bruitée indépendamment pour chaque annotateur
(paramètre --agreement = probabilité de coller à la vérité). C'est un GABARIT
qui montre le format attendu et permet de tester les métriques ; remplace-le par
de vraies annotations pour des résultats interprétables.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champslibres_pipeline.config import load_config


def supers_from_book(path):
    if path and os.path.exists(path):
        book = json.load(open(path, encoding="utf-8"))
        codes = []
        for cat in book.get("categories", []):
            if cat.get("children"):           # parent Rxx/Fxx
                codes.append(cat["code"])
        if codes:
            return codes + ["OTHER"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exemple.yaml")
    ap.add_argument("--humans", type=int, default=2, help="nombre d'annotateurs humains")
    ap.add_argument("--agreement", type=float, default=0.8, help="probabilité de coller à la vérité latente")
    ap.add_argument("--book", default="output/label_book.json", help="label book (pour récupérer les Rxx/Fxx)")
    ap.add_argument("--supers", default="", help="codes super-catégories séparés par des virgules (sinon: du book ou défaut)")
    ap.add_argument("--out", default="data/exemple_annotated.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    id_col, text_col = cfg.columns.id, cfg.columns.text
    df = pd.read_csv(cfg.io.source_csv, dtype=str)[[id_col, text_col]].copy()

    if args.supers.strip():
        supers = [s.strip() for s in args.supers.split(",") if s.strip()]
    else:
        supers = supers_from_book(args.book) or ["F01", "F02", "F03", "OTHER"]
    print(f"Super-catégories utilisées : {supers}")

    rng = np.random.default_rng(args.seed)
    truth = rng.choice(supers, size=len(df))
    for k in range(1, args.humans + 1):
        keep = rng.random(len(df)) < args.agreement
        noise = rng.choice(supers, size=len(df))
        df[f"humain{k}"] = np.where(keep, truth, noise)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} lignes, {args.humans} annotateur(s) -> {args.out}")
    print(df.head(6).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
