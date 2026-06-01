from __future__ import annotations

"""
Test de l'étape 05 (label book + regroupement Fxx), à lancer :

    uv run python scripts/test_step05.py

Hors-ligne (faux client) :
  1. (i) labellisation ;
  2. AVEC Rxx -> (ii)+(iii) ;
  3. SANS Rxx, method="none" ;
  4. SANS Rxx, method="dendrogram" (+ écriture d'un PNG) ;
  5. SANS Rxx, method="llm".
Puis test réel sur llm.lab si la clé est présente.
"""

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champslibres_pipeline.config import load_config
from champslibres_pipeline.llm.base import LLMClient
from champslibres_pipeline.steps.step05_label import label_clusters, run_label_book

KEYWORDS = {
    0: ["voisin", "bruit", "tapage", "quartier"],
    1: ["assurance", "franchise", "constat", "déclaration"],
    2: ["police", "plainte", "gendarmerie", "dépôt"],
}
REP_DOCS = {
    0: [("a", "mon voisin fait du bruit")],
    1: [("c", "constat amiable pour l'assurance")],
    2: [("e", "j'ai déposé plainte")],
}


class FakeLLM(LLMClient):
    def complete(self, messages, *, model, temperature=0.0, max_tokens=1024, json_schema=None) -> str:
        c = messages[-1]["content"]
        if "Catégorie analysée" in c:
            cid = re.search(r"Catégorie analysée\s*:\s*(C\d{2})", c).group(1)
            return json.dumps({"category_id": cid, "title": f"titre {cid}", "description": f"desc {cid}."})
        if "Regroupe-les" in c:  # regroupement LLM
            cids = sorted(set(re.findall(r"C\d{2}", c)))
            half = max(1, len(cids) // 2)
            return json.dumps({"supercategories": [
                {"super_cat": "F01", "super_label": "Groupe A", "super_description": "...", "members": cids[:half]},
                {"super_cat": "F02", "super_label": "Groupe B", "super_description": "...", "members": cids[half:]}]})
        if "liste vide" in c:
            return json.dumps({"supercategories": [
                {"super_cat": "F01", "super_label": "Démarches judiciaires", "super_description": "..."}]})
        if "rattacher" in c.lower():
            cids = sorted(set(re.findall(r"C\d{2}", c.split("rattacher")[-1])))
            parents = ["R01", "F01", "OTHER"]
            return json.dumps({"mappings": [{"sub_cat": x, "super_cat": parents[i % 3]} for i, x in enumerate(cids)]})
        return "{}"

    def complete_batch(self, batch, **kw):
        return [self.complete(m, **kw) for m in batch]

    def embed(self, texts, *, model):
        raise NotImplementedError

    def list_models(self):
        raise NotImplementedError


def _codes(book):
    return [cat["code"] for cat in book["categories"]]


def test_label(cfg):
    print("\n1) (i) Labellisation")
    cfg.label.model = "faux"
    labels = label_clusters(KEYWORDS, REP_DOCS, cfg, llm_client=FakeLLM(), question="Q", context="C")
    assert [l["category_id"] for l in labels] == ["C00", "C01", "C02"]
    print("[OK ]", [l["category_id"] for l in labels])


def test_with_rxx(cfg):
    print("\n2) AVEC Rxx")
    cfg.label.model = "faux"
    cfg.survey.modalities = [{"code": "R01", "label": "Assurance", "description": "..."}]
    book = run_label_book(KEYWORDS, REP_DOCS, cfg, llm_client=FakeLLM())
    print("   ", _codes(book))
    assert book["has_existing_modalities"] and "R01" in _codes(book) and "F01" in _codes(book)
    cfg.survey.modalities = []
    print("[OK ]")


def test_none(cfg):
    print("\n3) SANS Rxx, method='none'")
    cfg.label.model = "faux"; cfg.label.grouping.method = "none"
    book = run_label_book(KEYWORDS, REP_DOCS, cfg, llm_client=FakeLLM())
    print("   ", _codes(book))
    assert _codes(book) == ["C00", "C01", "C02", "OTHER"]
    print("[OK ]")


def test_dendrogram_cut():
    print("\n4) Dendrogramme BERTopic : coupe par union-find (cœur testable hors BERTopic)")
    import pandas as pd
    from champslibres_pipeline.steps.step05_label import (
        cut_hierarchy, hierarchy_to_merges, _supertopic_desc)
    # arbre synthétique : (0,1) fusionnent à 0.5, puis {0,1} et 2 à 1.0
    hier = pd.DataFrame([
        {"Parent_ID": 3, "Child_Left_ID": 0, "Child_Right_ID": 1, "Distance": 0.5, "Topics": [0, 1]},
        {"Parent_ID": 4, "Child_Left_ID": 3, "Child_Right_ID": 2, "Distance": 1.0, "Topics": [0, 1, 2]},
    ])
    leaves, merges = hierarchy_to_merges(hier)
    assert leaves == [0, 1, 2]
    a_thr = cut_hierarchy(leaves, merges, threshold=0.7, n_max=None)
    a_k2 = cut_hierarchy(leaves, merges, threshold=None, n_max=2)
    a_k3 = cut_hierarchy(leaves, merges, threshold=None, n_max=3)
    print("   seuil 0.7 :", a_thr, "| n_max 2 :", a_k2, "| n_max 3 :", a_k3)
    assert a_thr[0] == a_thr[1] != a_thr[2]        # {0,1} ensemble, {2} à part
    assert a_k2[0] == a_k2[1] != a_k2[2]
    assert len(set(a_k3.values())) == 3            # 3 groupes distincts
    desc = _supertopic_desc(["C00", "C01"], {0: ["banque", "remboursement"], 1: ["remboursement", "litige"]})
    print("   description Fxx :", desc)
    assert desc.startswith("Thème regroupé autour de : banque, remboursement")
    print("[OK ] coupe distance/n_max + mots-clés Fxx")


def test_dendrogram_guard(cfg):
    print("\n4b) Dendrogramme : garde-fou si topic_model/docs manquants")
    cfg.label.model = "faux"; cfg.label.grouping.method = "dendrogram"
    try:
        run_label_book(KEYWORDS, REP_DOCS, cfg, llm_client=FakeLLM())
        raise AssertionError("aurait dû lever ValueError")
    except ValueError:
        print("[OK ] ValueError levée (topic_model + docs requis)")
    finally:
        cfg.label.grouping.method = "none"


def test_llm_grouping(cfg):
    print("\n5) SANS Rxx, method='llm'")
    cfg.label.model = "faux"; cfg.label.grouping.method = "llm"; cfg.label.grouping.n_max = 2
    book = run_label_book(KEYWORDS, REP_DOCS, cfg, llm_client=FakeLLM())
    fcodes = [c for c in _codes(book) if c.startswith("F")]
    print("   super-catégories :", fcodes)
    assert "F01" in fcodes and "F02" in fcodes
    cfg.label.grouping.method = "none"
    print("[OK ]")


def test_real(cfg):
    print("\n6) RÉEL sur llm.lab (regroupement LLM, max 5 Fxx)")
    if not os.environ.get("LLM_LAB_API_KEY"):
        print("[--] clé absente : ignoré"); return
    from champslibres_pipeline.llm import get_llm_client, list_models
    m = list_models()
    if not m["generation"]:
        print("[--] aucun modèle : ignoré"); return
    cfg.label.model = m["generation"][0]
    cfg.label.grouping.method = "llm"; cfg.label.grouping.n_max = 5
    cfg.survey.question = "Quelles suites après l'incident ?"
    book = run_label_book(KEYWORDS, REP_DOCS, cfg, llm_client=get_llm_client(cfg.llm))
    print(json.dumps(book, ensure_ascii=False, indent=2))
    print("[OK ]")


def main():
    cfg = load_config("configs/exemple.yaml")
    test_label(cfg); test_with_rxx(cfg); test_none(cfg)
    test_dendrogram_cut(); test_dendrogram_guard(cfg); test_llm_grouping(cfg); test_real(cfg)
    print("\nÉtape 05 vérifiée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
