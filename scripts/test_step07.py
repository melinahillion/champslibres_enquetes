from __future__ import annotations

"""
Test de l'étape 07 (évaluation), à lancer :  uv run python scripts/test_step07.py
Hors-ligne : exactitude des métriques (valeurs connues), mapping Cxx->super,
orchestrateur sur données synthétiques, adaptativité (N humains / M modèles).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champslibres_pipeline.config import load_config
from champslibres_pipeline.steps.step07_eval import (
    cohen_kappa, cxx_to_super, fleiss_kappa, map_to_super, percent_agreement, run_evaluation)

BOOK = {"categories": [
    {"code": "F01", "label": "A", "description": "", "source": "generated",
     "children": [{"category_id": "C00", "title": "a"}, {"category_id": "C01", "title": "b"}]},
    {"code": "F02", "label": "B", "description": "", "source": "generated",
     "children": [{"category_id": "C02", "title": "c"}]},
    {"code": "OTHER", "label": "Autre", "description": "", "source": "residual", "children": []}]}


def test_metrics():
    print("\n1) Métriques (valeurs connues)")
    # exemple classique : Po=0.7, kappa=0.4
    x = np.array(["o"] * 25 + ["n"] * 25)
    y = np.array(["o"] * 20 + ["n"] * 5 + ["o"] * 10 + ["n"] * 15)
    assert abs(percent_agreement(x, y) - 0.7) < 1e-9
    assert abs(cohen_kappa(x, y) - 0.4) < 1e-9
    # accord parfait -> kappa 1
    z = np.array(["a", "b", "a", "c"])
    assert cohen_kappa(z, z) == 1.0
    # Fleiss parfait -> 1
    perfect = pd.DataFrame({"r1": ["a", "b", "a"], "r2": ["a", "b", "a"], "r3": ["a", "b", "a"]})
    assert fleiss_kappa(perfect, ["r1", "r2", "r3"]) == 1.0
    print(f"[OK ] Po=0.7, kappa Cohen=0.4, accords parfaits=1.0")


def test_mapping():
    print("\n2) Mapping Cxx -> super")
    m = cxx_to_super(BOOK)
    assert m == {"C00": "F01", "C01": "F01", "C02": "F02", "OTHER": "OTHER"}
    s = map_to_super(pd.Series(["C00", "C02", "OTHER", "C01"]), m)
    assert list(s) == ["F01", "F02", "OTHER", "F01"]
    print("[OK ]", m)


def _synthetic_df():
    return pd.DataFrame({
        "SEQ_ID": [f"id{i}" for i in range(8)],
        "humain1":     ["F01", "F01", "F02", "F02", "F01", "F02", "OTHER", "F01"],
        "humain2":     ["F01", "F01", "F02", "F01", "F01", "F02", "OTHER", "F01"],
        "modelA__cxx": ["C00", "C01", "C02", "C02", "C00", "C02", "OTHER", "C01"],
        "modelB__cxx": ["C00", "C00", "C02", "C00", "C01", "C02", "C02",   "C01"],
    })


def test_orchestrator(cfg):
    print("\n3) Orchestrateur : 2 humains + 2 modèles")
    cfg.eval.human_columns = ["humain1", "humain2"]
    cfg.eval.models = ["modelA", "modelB"]
    cfg.eval.bootstrap_B = 200
    res = run_evaluation(_synthetic_df(), BOOK, cfg)
    pw = res["pairwise"]
    counts = pw["type"].value_counts().to_dict()
    print(pw[["juge_a", "juge_b", "type", "n", "Po", "kappa"]].to_string(index=False))
    print("   Fleiss:", res["fleiss"])
    assert counts == {"humain-modèle": 4, "humain-humain": 1, "modèle-modèle": 1}
    assert set(res["fleiss"]) == {"humains", "tous_juges"}
    # IC encadrent l'estimation
    row = pw.iloc[0]
    assert row["Po_IC95_bas"] <= row["Po"] <= row["Po_IC95_haut"]
    print("[OK ] 6 paires, types corrects, IC cohérents, Fleiss calculé")


def test_adaptivity(cfg):
    print("\n4) Adaptativité")
    df = _synthetic_df()
    # 3 humains, 0 modèle -> 3 paires humain-humain
    df3 = df.copy(); df3["humain3"] = ["F01", "F02", "F02", "F02", "F01", "F01", "OTHER", "F01"]
    cfg.eval.human_columns = ["humain1", "humain2", "humain3"]; cfg.eval.models = []
    cfg.eval.bootstrap_B = 100
    res = run_evaluation(df3, BOOK, cfg)
    assert len(res["pairwise"]) == 3 and set(res["pairwise"]["type"]) == {"humain-humain"}
    print("   3 humains, 0 modèle ->", len(res["pairwise"]), "paires H-H ; Fleiss humains =", res["fleiss"]["humains"])
    # 1 seul juge -> erreur
    cfg.eval.human_columns = ["humain1"]; cfg.eval.models = []
    try:
        run_evaluation(df, BOOK, cfg); raise AssertionError("aurait dû lever ValueError")
    except ValueError:
        print("   1 seul juge -> ValueError (attendu)")
    print("[OK ] s'adapte au nombre de juges")


def main():
    cfg = load_config("configs/exemple.yaml")
    test_metrics(); test_mapping(); test_orchestrator(cfg); test_adaptivity(cfg)
    print("\nÉtape 07 vérifiée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
