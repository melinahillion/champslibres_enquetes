from __future__ import annotations

"""
Test de l'étape 06 (classification), à lancer :

    uv run python scripts/test_step06.py

Hors-ligne (faux client) : feuilles assignables, mode "sans", mode "humain",
mode "llm" (vote + file de reprise), validation des codes. Puis test réel si
LLM_LAB_API_KEY est présente.
"""

import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champslibres_pipeline.config import load_config
from champslibres_pipeline.llm.base import LLMClient
from champslibres_pipeline.steps.step06_classify import assignable_labels, run_classification

FLAT_BOOK = {"question": "Q", "context": "", "has_existing_modalities": False, "categories": [
    {"code": "C00", "label": "Voisinage", "description": "bruit", "source": "cluster", "children": []},
    {"code": "C01", "label": "Assurance", "description": "constat", "source": "cluster", "children": []},
    {"code": "C02", "label": "Police", "description": "plainte", "source": "cluster", "children": []},
    {"code": "OTHER", "label": "Autre", "description": "résiduel", "source": "residual", "children": []}]}

HIER_BOOK = {"question": "Q", "context": "", "has_existing_modalities": False, "categories": [
    {"code": "F01", "label": "Démarches", "description": "", "source": "generated", "children": [
        {"category_id": "C01", "title": "Assurance", "description": ""},
        {"category_id": "C02", "title": "Police", "description": ""}]},
    {"code": "F02", "label": "Voisinage", "description": "", "source": "generated", "children": [
        {"category_id": "C00", "title": "Bruit", "description": ""}]},
    {"code": "OTHER", "label": "Autre", "description": "", "source": "residual", "children": []}]}


def _rule(text):
    t = text.lower()
    if "mystère" in t:
        return None  # non classé -> teste la file de reprise -> OTHER
    if "voisin" in t or "bruit" in t:
        return "C00"
    if "assur" in t or "constat" in t:
        return "C01"
    if "plainte" in t or "police" in t:
        return "C02"
    return "OTHER"


class FakeLLM(LLMClient):
    def complete(self, messages, **kw):
        return "{}"

    def complete_batch(self, batch, **kw):
        outs = []
        for msgs in batch:
            c = msgs[-1]["content"]
            pairs = re.findall(r"- \[(.+?)\] (.+)", c)
            assigns = [{"id": i, "code": _rule(t)} for i, t in pairs if _rule(t) is not None]
            outs.append(json.dumps({"assignments": assigns}))
        return outs

    def embed(self, texts, *, model):
        raise NotImplementedError

    def list_models(self):
        raise NotImplementedError


def _df():
    return pd.DataFrame({
        "SEQ_ID": [1, 2, 3, 4, 5],
        "text_clean": ["mon voisin fait du bruit", "constat amiable assurance",
                       "j'ai déposé plainte police", "remboursement banque", "xyz mystère"],
    })


def test_leaves():
    print("\n1) Feuilles assignables")
    flat = assignable_labels(FLAT_BOOK)
    hier = assignable_labels(HIER_BOOK)
    assert [l["code"] for l in flat] == ["C00", "C01", "C02", "OTHER"]
    by = {l["code"]: l for l in hier}
    assert by["C01"]["parent_code"] == "F01" and by["C00"]["parent_code"] == "F02"
    print("[OK ] flat:", [l["code"] for l in flat], "| hier parents:",
          {k: by[k]["parent_code"] for k in ("C00", "C01", "C02")})


def test_sans(cfg):
    print("\n2) mode 'sans' (depuis le clustering)")
    cfg.classf.mode = "sans"
    out = run_classification(_df(), FLAT_BOOK, cfg, clusters=[0, 1, 2, -1, 5])
    print("   ", dict(zip(out["id"], out["code"])))
    assert list(out["code"]) == ["C00", "C01", "C02", "OTHER", "OTHER"]
    print("[OK ]")


def test_humain(cfg):
    print("\n3) mode 'humain' (codes fournis + validation)")
    cfg.classf.mode = "humain"
    df = _df(); df["code_humain"] = ["C00", "C01", "BADCODE", "OTHER", "C02"]
    out = run_classification(df, FLAT_BOOK, cfg)
    print("   ", list(out["code"]))
    assert list(out["code"]) == ["C00", "C01", "OTHER", "OTHER", "C02"]  # BADCODE -> OTHER
    print("[OK ]")


def test_llm(cfg):
    print("\n4) mode 'llm' (lots + vote + reprise)")
    cfg.classf.mode = "llm"; cfg.classf.model = "faux"
    cfg.classf.n_iterations = 3; cfg.classf.batch_size = 2
    out = run_classification(_df(), HIER_BOOK, cfg, llm_client=FakeLLM())
    res = dict(zip(out["id"], out["code"]))
    parents = dict(zip(out["id"], out["parent_code"]))
    print("   codes :", res)
    print("   parents :", parents)
    assert res == {"1": "C00", "2": "C01", "3": "C02", "4": "OTHER", "5": "OTHER"}
    assert parents["1"] == "F02" and parents["2"] == "F01"
    assert float(out.loc[out["id"] == "5", "accord"].iloc[0]) == 0.0  # résolu par reprise
    print("[OK ] vote + reprise OK ; accord(id5)=0.0")


def test_real(cfg):
    print("\n5) RÉEL sur llm.lab")
    if not os.environ.get("LLM_LAB_API_KEY"):
        print("[--] clé absente : ignoré"); return
    from champslibres_pipeline.llm import get_llm_client, list_models
    m = list_models()
    if not m["generation"]:
        print("[--] aucun modèle : ignoré"); return
    cfg.classf.mode = "llm"; cfg.classf.model = m["generation"][0]
    cfg.classf.n_iterations = 1; cfg.classf.batch_size = 5
    cfg.survey.question = "Quelles suites après l'incident ?"
    out = run_classification(_df(), HIER_BOOK, cfg, llm_client=get_llm_client(cfg.llm))
    print(out[["id", "text", "code", "parent_code"]].to_string(index=False))
    print("[OK ]")


def main():
    cfg = load_config("configs/exemple.yaml")
    test_leaves(); test_sans(cfg); test_humain(cfg); test_llm(cfg); test_real(cfg)
    print("\nÉtape 06 vérifiée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
