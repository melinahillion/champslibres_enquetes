from __future__ import annotations

"""
Étape 03 — Embeddings (vecteurs de sens).

Transforme chaque texte unique (issu de l'étape 02) en un vecteur dense, où la
proximité géométrique reflète la proximité de sens. C'est ce qui permettra à
l'étape 04 de regrouper les réponses par thème.

Deux backends, au choix dans la config (section `embed`) :

  - backend "llm_lab" (défaut) : on appelle l'API llm.lab.
      * si embed.model_name est vide -> on prend AUTOMATIQUEMENT le premier
        modèle d'embedding disponible sur l'instance ;
      * sinon -> on utilise le modèle nommé.
      Pour voir les modèles disponibles : list_embedding_models(client), ou
      list_models() (et, plus tard, la liste déroulante de l'interface).

  - backend "huggingface" (option) : on télécharge un modèle d'embedding depuis
    HuggingFace (ex. BAAI/bge-m3) et on calcule les vecteurs EN LOCAL.
    Nécessite l'extra [nlp]  ->  uv sync --extra nlp

Coeur réutilisable : embed_texts(texts, cfg, llm_client=...) -> matrice numpy
(une ligne par texte, une colonne par dimension du vecteur).
"""

from typing import List, Optional

import numpy as np

from ..config import ProjectConfig
from ..llm.base import LLMClient


def list_embedding_models(llm_client: LLMClient) -> List[str]:
    """Liste des modèles d'embedding disponibles sur llm.lab."""
    return llm_client.list_models().get("embedding", [])


def resolve_embedding_model(
    cfg: ProjectConfig, llm_client: Optional[LLMClient] = None
) -> str:
    """
    Détermine quel modèle utiliser, selon le backend et embed.model_name.
    """
    embed = cfg.embed

    if embed.backend == "huggingface":
        return embed.model_name or "BAAI/bge-m3"

    if embed.backend == "llm_lab":
        if embed.model_name:
            return embed.model_name  # choix explicite de l'utilisateur
        # sinon : premier modèle d'embedding disponible
        if llm_client is None:
            raise ValueError(
                "backend 'llm_lab' : un client LLM est requis pour lister les modèles."
            )
        available = list_embedding_models(llm_client)
        if not available:
            raise RuntimeError(
                "Aucun modèle d'embedding disponible sur llm.lab. "
                "Renseigne embed.model_name, ou passe embed.backend='huggingface'."
            )
        return available[0]

    raise ValueError(
        f"embed.backend inconnu : {embed.backend!r} (attendu 'llm_lab' ou 'huggingface')."
    )


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """Ramène chaque vecteur à une longueur de 1 (utile pour la similarité cosinus)."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # évite la division par zéro sur un vecteur nul
    return arr / norms


def _embed_huggingface(texts: List[str], cfg: ProjectConfig) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "Le backend 'huggingface' nécessite l'extra [nlp]. "
            "Installe-le avec : uv sync --extra nlp"
        ) from e

    model_name = cfg.embed.model_name or "BAAI/bge-m3"
    model = SentenceTransformer(model_name, device=cfg.embed.device)
    arr = model.encode(
        list(texts),
        batch_size=cfg.embed.batch_size,
        normalize_embeddings=cfg.embed.normalize,
        show_progress_bar=True,
    )
    return np.asarray(arr, dtype=np.float32)


def embed_texts(
    texts: List[str],
    cfg: ProjectConfig,
    *,
    llm_client: Optional[LLMClient] = None,
) -> np.ndarray:
    """
    Renvoie la matrice des embeddings (shape = [nombre de textes, dimension]).
    """
    backend = cfg.embed.backend

    if backend == "llm_lab":
        if llm_client is None:
            raise ValueError(
                "backend 'llm_lab' : il faut fournir un client LLM (llm_client=...)."
            )
        model = resolve_embedding_model(cfg, llm_client)
        vecs = llm_client.embed(list(texts), model=model)
        arr = np.asarray(vecs, dtype=np.float32)
        if cfg.embed.normalize:
            arr = _l2_normalize(arr)
        return arr

    if backend == "huggingface":
        # ici la normalisation est gérée directement par encode(normalize_embeddings=...)
        return _embed_huggingface(list(texts), cfg)

    raise ValueError(
        f"embed.backend inconnu : {backend!r} (attendu 'llm_lab' ou 'huggingface')."
    )
