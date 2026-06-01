from __future__ import annotations

"""
Interface commune à tous les "clients LLM".

Le reste du pipeline ne parle qu'à cette interface : il demande "complète ce
message" ou "donne-moi les embeddings de ces textes", SANS savoir quel modèle
ni quel fournisseur est derrière. C'est une COUCHE D'ABSTRACTION : elle isole
le pipeline des détails techniques du fournisseur.

Avantage concret : pour passer de llm.lab à un vLLM local (ou l'inverse), on
ne touche qu'à UN SEUL fichier (l'implémentation), pas à tout le pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Un "message" suit le format OpenAI :
#   {"role": "system" | "user" | "assistant", "content": "..."}
Message = Dict[str, str]


class LLMClient(ABC):
    """Contrat que tout client LLM doit respecter (méthodes obligatoires)."""

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Envoie une conversation et renvoie la réponse texte du modèle.

        Si json_schema est fourni, on demande au modèle de répondre STRICTEMENT
        selon ce schéma JSON. Derrière llm.lab, c'est vLLM qui garantit le
        format (sortie structurée fiable).
        """
        ...

    @abstractmethod
    def complete_batch(
        self,
        batch_messages: List[List[Message]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: Optional[Dict[str, Any]] = None,
        progress: bool = False,
        progress_desc: str = "",
        unit_weights: Optional[List[int]] = None,
    ) -> List[str]:
        """Comme complete(), mais pour une liste de conversations, en parallèle.
        Si progress=True, affiche une barre de progression (avec estimation du
        temps restant) ; unit_weights permet de compter en nombre de réponses."""
        ...

    @abstractmethod
    def embed(self, texts: List[str], *, model: str) -> List[List[float]]:
        """Renvoie un vecteur (liste de nombres) par texte fourni."""
        ...

    @abstractmethod
    def list_models(self) -> Dict[str, List[str]]:
        """
        Liste les modèles disponibles.
        Renvoie un dict : {"all": [...], "generation": [...], "embedding": [...]}.
        """
        ...
