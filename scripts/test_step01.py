from __future__ import annotations

"""
Test de l'étape 01 (correction du texte), à lancer :

    uv run python scripts/test_step01.py

Quatre vérifications :
  1. mode "none"  : 'clean' = copie exacte du brut ;
  2. mode "light" : normalisation simple (sans LLM ni réseau) ;
  3. mode "llm" avec un FAUX client (hors-ligne) pour valider la mécanique ;
  4. mode "llm" RÉEL sur llm.lab, si la clé est présente (sinon ignoré).
"""

import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champslibres_pipeline.config import load_config
from champslibres_pipeline.llm.base import LLMClient
from champslibres_pipeline.steps.step01_gec import apply_gec

SAMPLE = "data/exemple_input.csv"


# --- 1) mode none ------------------------------------------------------------
def test_none(cfg) -> None:
    print("\n1) Mode 'none' (copie exacte, aucune modification)")
    df = pd.read_csv(SAMPLE, dtype=str)
    cfg.gec.mode = "none"
    out = apply_gec(df, cfg)
    assert out[cfg.columns.clean].tolist() == df[cfg.columns.text].tolist()
    print("[OK ] 'clean' identique à 'text_brut'")


# --- 2) mode light -----------------------------------------------------------
def test_light(cfg) -> None:
    print("\n2) Mode 'light' (normalisation simple, sans LLM)")
    df = pd.read_csv(SAMPLE, dtype=str)
    cfg.gec.mode = "light"
    out = apply_gec(df, cfg)
    print(out[[cfg.columns.text, cfg.columns.clean]].head(4).to_string(index=False))
    assert cfg.columns.clean in out.columns
    print("[OK ] colonne 'clean' produite, normalisation appliquée")


# --- 3) mode llm avec un faux client (hors-ligne) ----------------------------
class FakeLLM(LLMClient):
    """Faux client : renvoie corr<tmp_id> pour chaque entrée. Sert à tester la
    mécanique (lots, schéma, remontée par tmp_id) SANS réseau."""

    def complete(self, messages, *, model, temperature=0.0, max_tokens=1024, json_schema=None) -> str:
        content = messages[-1]["content"]
        ids = re.findall(r'"tmp_id":\s*"(\d+)"', content)
        items = [{"tmp_id": i, "text_corrige": f"corr{i}"} for i in ids]
        return json.dumps({"corrections": items}, ensure_ascii=False)

    def complete_batch(self, batch_messages, **kw):
        return [self.complete(m, **kw) for m in batch_messages]

    def embed(self, texts, *, model):
        raise NotImplementedError

    def list_models(self):
        raise NotImplementedError


def test_llm_fake(cfg) -> None:
    print("\n3) Mode 'llm' avec un FAUX client (hors-ligne)")
    df = pd.read_csv(SAMPLE, dtype=str).head(5)
    cfg.gec.mode = "llm"
    cfg.gec.model = "faux-modele"
    cfg.gec.batch_size = 2  # force plusieurs lots
    out = apply_gec(df, cfg, llm_client=FakeLLM())
    got = out[cfg.columns.clean].tolist()
    expected = [f"corr{i}" for i in range(len(df))]
    print("   résultat :", got)
    assert got == expected, f"attendu {expected}, obtenu {got}"
    print("[OK ] lots + remontée par tmp_id corrects")


# --- 4) mode llm RÉEL sur llm.lab (si clé présente) --------------------------
def test_llm_real(cfg) -> None:
    print("\n4) Mode 'llm' RÉEL sur llm.lab")
    if not os.environ.get("LLM_LAB_API_KEY"):
        print("[--] clé LLM absente : test réel ignoré")
        return
    from champslibres_pipeline.llm import get_llm_client, list_models

    models = list_models()
    if not models["generation"]:
        print("[--] aucun modèle de génération : test réel ignoré")
        return

    model = models["generation"][0]
    cfg.gec.mode = "llm"
    cfg.gec.model = model
    cfg.gec.batch_size = 8

    df = pd.DataFrame(
        {
            cfg.columns.id: ["1", "2", "3"],
            cfg.columns.text: ["ca vient d'arrivé", "constat amiable", "non conaissance du coupable"],
        }
    )
    client = get_llm_client(cfg.llm)
    out = apply_gec(df, cfg, llm_client=client)
    print(f"   modèle = {model}")
    for brut, clean in zip(df[cfg.columns.text], out[cfg.columns.clean]):
        print(f"   {brut!r:45s} -> {clean!r}")
    print("[OK ] correction LLM réelle effectuée")


def main() -> int:
    cfg = load_config("configs/exemple.yaml")
    test_none(cfg)
    test_light(cfg)
    test_llm_fake(cfg)
    test_llm_real(cfg)
    print("\nÉtape 01 vérifiée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
