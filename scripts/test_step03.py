from __future__ import annotations

"""
Test de l'étape 03 (embeddings), à lancer :

    uv run python scripts/test_step03.py

Trois vérifications :
  1. choix du modèle (résolution) selon la config ;
  2. calcul des embeddings avec un FAUX client (hors-ligne) : forme + normalisation ;
  3. calcul RÉEL sur llm.lab si la clé est présente (affiche la liste des
     modèles d'embedding disponibles, puis la dimension obtenue).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champslibres_pipeline.config import load_config
from champslibres_pipeline.llm.base import LLMClient
from champslibres_pipeline.steps.step03_embed import (
    embed_texts,
    list_embedding_models,
    resolve_embedding_model,
)


class FakeEmbedLLM(LLMClient):
    """Faux client : renvoie un vecteur déterministe de dimension 4 par texte."""

    def __init__(self, models=("fake-embed-1", "fake-embed-2")):
        self._models = list(models)

    def complete(self, *a, **k):
        raise NotImplementedError

    def complete_batch(self, *a, **k):
        raise NotImplementedError

    def embed(self, texts, *, model):
        return [[float(len(t)), float(t.count("a")), float(t.count("e")), 1.0] for t in texts]

    def list_models(self):
        return {"all": self._models, "generation": [], "embedding": self._models}


def test_resolution(cfg) -> None:
    print("\n1) Choix du modèle d'embedding")
    fake = FakeEmbedLLM()

    # llm_lab + model_name vide -> 1er modèle dispo
    cfg.embed.backend = "llm_lab"
    cfg.embed.model_name = None
    assert resolve_embedding_model(cfg, fake) == "fake-embed-1"

    # llm_lab + model_name renseigné -> ce modèle
    cfg.embed.model_name = "fake-embed-2"
    assert resolve_embedding_model(cfg, fake) == "fake-embed-2"

    # huggingface -> id HF (défaut bge-m3 si vide)
    cfg.embed.backend = "huggingface"
    cfg.embed.model_name = None
    assert resolve_embedding_model(cfg) == "BAAI/bge-m3"

    print("[OK ] auto (1er dispo), choix explicite, et repli HuggingFace")


def test_embed_fake(cfg) -> None:
    print("\n2) Calcul des embeddings (faux client, hors-ligne)")
    cfg.embed.backend = "llm_lab"
    cfg.embed.model_name = None
    cfg.embed.normalize = True
    texts = ["constat amiable", "non connaissance du coupable", "assurance"]
    arr = embed_texts(texts, cfg, llm_client=FakeEmbedLLM())
    print(f"   forme de la matrice = {arr.shape}  (3 textes attendus)")
    norms = np.linalg.norm(arr, axis=1)
    print(f"   normes des vecteurs = {np.round(norms, 3)}")
    assert arr.shape[0] == 3
    assert np.allclose(norms, 1.0, atol=1e-5), "les vecteurs devraient être normalisés"
    print("[OK ] forme correcte et vecteurs L2-normalisés")


def test_embed_real(cfg) -> None:
    print("\n3) Calcul RÉEL sur llm.lab")
    if not os.environ.get("LLM_LAB_API_KEY"):
        print("[--] clé LLM absente : test réel ignoré")
        return
    from champslibres_pipeline.llm import get_llm_client

    client = get_llm_client(cfg.llm)
    dispo = list_embedding_models(client)
    print("   Modèles d'embedding disponibles :", ", ".join(dispo) or "(aucun)")
    if not dispo:
        print("[--] aucun modèle d'embedding : test réel ignoré")
        return

    cfg.embed.backend = "llm_lab"
    cfg.embed.model_name = None  # -> 1er dispo
    model = resolve_embedding_model(cfg, client)
    arr = embed_texts(["constat amiable", "voisin", "inconnu"], cfg, llm_client=client)
    print(f"   modèle utilisé = {model}")
    print(f"   forme = {arr.shape}  (dimension du vecteur = {arr.shape[1]})")
    print("[OK ] embeddings réels calculés via llm.lab")


def main() -> int:
    cfg = load_config("configs/exemple.yaml")
    test_resolution(cfg)
    test_embed_fake(cfg)
    test_embed_real(cfg)
    print("\nÉtape 03 vérifiée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
