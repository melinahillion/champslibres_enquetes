from __future__ import annotations

"""
Client LLM qui parle à une API compatible OpenAI.
Par défaut : le service llm.lab du SSP Cloud (OpenWebUI + vLLM).

Trois capacités :
  - complete()        : génération de texte (avec sortie JSON stricte en option)
  - embed()           : calcul d'embeddings (si un modèle d'embedding est exposé)
  - list_models()     : liste des modèles disponibles (récupérée dynamiquement)
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from .base import LLMClient, Message

# Indices pour deviner si un modèle sert aux embeddings : son nom le trahit
# souvent (bge, e5, gte...). Ce n'est qu'une aide pour pré-trier les menus ;
# l'utilisateur garde le dernier mot dans l'interface.
_EMBED_HINTS = ("embed", "bge", "e5", "gte", "minilm", "sentence", "nomic")


def _looks_like_embedding_model(model_id: str) -> bool:
    name = model_id.lower()
    return any(hint in name for hint in _EMBED_HINTS)


class OpenAICompatibleClient(LLMClient):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_sec: int = 120,
        max_concurrency: int = 8,
        max_retries: int = 3,
    ):
        # Import local : la dépendance 'openai' n'est utile que pour ce backend,
        # on ne la charge donc que lorsqu'on en a besoin.
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_sec)
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_retries = max(1, int(max_retries))

    # ------------------------------------------------------------------ génération
    @staticmethod
    def _response_format(json_schema: Optional[Dict[str, Any]]):
        """Traduit un schéma JSON en paramètre 'response_format' pour l'API."""
        if json_schema is None:
            return None
        return {
            "type": "json_schema",
            "json_schema": {
                "name": json_schema.get("title", "reponse"),
                "schema": json_schema,
                "strict": True,
            },
        }

    def complete(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        response_format = self._response_format(json_schema)
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                kwargs: Dict[str, Any] = dict(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if response_format is not None:
                    kwargs["response_format"] = response_format
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - on réessaie après une pause
                last_err = e
                # Repli : si le serveur refuse 'json_schema', on retente en
                # 'json_object' (moins strict mais plus largement accepté).
                if response_format is not None and response_format.get("type") == "json_schema":
                    response_format = {"type": "json_object"}
                time.sleep(1.0 + attempt)

        raise RuntimeError(
            f"Échec de l'appel LLM après {self.max_retries} tentatives : {last_err}"
        )

    def complete_batch(
        self,
        batch_messages: List[List[Message]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        results: List[str] = [""] * len(batch_messages)

        def _one(indexed):
            i, msgs = indexed
            text = self.complete(
                msgs,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_schema=json_schema,
            )
            return i, text

        # Plusieurs appels réseau en parallèle, mais bornés par max_concurrency
        # pour rester courtois avec une plateforme mutualisée.
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            for i, text in pool.map(_one, list(enumerate(batch_messages))):
                results[i] = text
        return results

    # ------------------------------------------------------------------ embeddings
    def embed(self, texts: List[str], *, model: str) -> List[List[float]]:
        out: List[List[float]] = []
        chunk = 64  # on envoie les textes par paquets pour éviter les requêtes géantes
        for start in range(0, len(texts), chunk):
            part = texts[start : start + chunk]
            resp = self._client.embeddings.create(model=model, input=part)
            out.extend(item.embedding for item in resp.data)
        return out

    # ------------------------------------------------------------------ modèles
    def list_models(self) -> Dict[str, List[str]]:
        models = self._client.models.list()
        ids = sorted(m.id for m in models.data)
        embedding = [m for m in ids if _looks_like_embedding_model(m)]
        embedding_set = set(embedding)
        generation = [m for m in ids if m not in embedding_set]
        return {"all": ids, "generation": generation, "embedding": embedding}
