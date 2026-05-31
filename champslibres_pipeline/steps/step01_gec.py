from __future__ import annotations

"""
Étape 01 — Correction / normalisation du texte (OPTIONNELLE).

Trois comportements, pilotés par `gec.mode` dans la config :
  - gec.mode = "none"   -> on ne touche à rien : 'clean' = copie exacte du brut.
  - gec.mode = "light"  -> normalisation typographique simple (CPU, sans LLM).
  - gec.mode = "llm"    -> correction orthographe/grammaire par un LLM (llm.lab).

Le coeur réutilisable est la fonction apply_gec(df, cfg, llm_client=...), qui
travaille sur un DataFrame en mémoire — donc facile à tester. La lecture/écriture
S3 (run) sera ajoutée quand on câblera l'étape 00 et l'API Project.
"""

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import ProjectConfig
from ..llm.base import LLMClient
from ..prompts import load_text
from ..prompts.gec import GEC_SYSTEM, GEC_USER


# ---------------------------------------------------------------------------
# Normalisation légère (sûre, sans LLM)
# ---------------------------------------------------------------------------
def normalize_light(value: Any) -> str:
    """
    Normalisation typographique minimale et sans risque :
      - normalisation unicode NFC,
      - apostrophes typographiques -> apostrophe simple,
      - espaces multiples réduits à un seul, espaces de début/fin supprimés.
    """
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFC", value)
    text = text.replace("\u2019", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Correction par LLM
# ---------------------------------------------------------------------------
def _gec_schema() -> Dict[str, Any]:
    """Schéma JSON imposé à la réponse du LLM (sortie structurée, fiable)."""
    return {
        "title": "corrections",
        "type": "object",
        "properties": {
            "corrections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tmp_id": {"type": "string"},
                        "text_corrige": {"type": "string"},
                    },
                    "required": ["tmp_id", "text_corrige"],
                },
            }
        },
        "required": ["corrections"],
    }


def _parse_corrections(raw: str) -> List[Dict[str, str]]:
    """Transforme la réponse texte du LLM en liste d'objets {tmp_id, text_corrige}."""
    text = (raw or "").strip()
    if text.startswith("```"):  # au cas où le modèle entoure le JSON de ```
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :]
    try:
        data = json.loads(text)
    except Exception:
        return []
    if isinstance(data, dict) and isinstance(data.get("corrections"), list):
        return data["corrections"]
    if isinstance(data, list):
        return data
    return []


def correct_with_llm(
    texts: List[str],
    *,
    client: LLMClient,
    model: str,
    system_prompt: str,
    user_template: str,
    temperature: float = 0.0,
    batch_size: int = 32,
) -> List[str]:
    """
    Corrige une liste de textes par lots. En cas de réponse manquante ou
    invalide pour un texte, on conserve le texte d'entrée (jamais de perte).
    """
    results: List[str] = list(texts)  # par défaut : inchangé

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        # tmp_id = position GLOBALE dans la liste -> remontée fiable du résultat
        items = [{"tmp_id": str(start + i), "text": t} for i, t in enumerate(chunk)]
        user = user_template.format(items=json.dumps(items, ensure_ascii=False))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]
        # marge de tokens proportionnelle à la taille du lot
        max_tokens = 256 + 64 * len(chunk)
        raw = client.complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=_gec_schema(),
        )
        for obj in _parse_corrections(raw):
            try:
                idx = int(obj["tmp_id"])
            except (KeyError, ValueError, TypeError):
                continue
            if 0 <= idx < len(texts):
                corrected = str(obj.get("text_corrige", "") or "")
                # si le modèle renvoie du vide, on garde l'entrée d'origine
                results[idx] = corrected if corrected else texts[idx]
    return results


# ---------------------------------------------------------------------------
# Coeur réutilisable : applique la correction à un DataFrame
# ---------------------------------------------------------------------------
def apply_gec(
    df: pd.DataFrame,
    cfg: ProjectConfig,
    *,
    llm_client: Optional[LLMClient] = None,
) -> pd.DataFrame:
    """
    Renvoie une copie du DataFrame avec la colonne 'clean' (cfg.columns.clean)
    remplie selon la configuration `gec`.
    """
    text_col = cfg.columns.text
    clean_col = cfg.columns.clean
    gec = cfg.gec

    if text_col not in df.columns:
        raise KeyError(f"Colonne texte introuvable : {text_col!r}")

    out = df.copy()
    texts = out[text_col].astype(str).tolist()
    mode = gec.mode

    # "none" : on ne touche à rien (clean = copie exacte du brut)
    if mode == "none":
        out[clean_col] = out[text_col]
        return out

    # "light" : normalisation typographique simple (CPU, sans LLM)
    if mode == "light":
        out[clean_col] = [normalize_light(t) for t in texts]
        return out

    # "llm" : correction orthographe/grammaire par un modèle de llm.lab
    if mode == "llm":
        if llm_client is None:
            raise ValueError(
                "gec.mode = 'llm' : il faut fournir un client LLM (llm_client=...)."
            )
        if not gec.model:
            raise ValueError("gec.mode = 'llm' : précisez gec.model dans la config.")

        system = GEC_SYSTEM  # prompt système par défaut
        user = load_text(GEC_USER, gec.prompt_file)  # surchargé si gec.prompt_file
        base = [normalize_light(t) for t in texts]  # entrée déjà un peu nettoyée
        out[clean_col] = correct_with_llm(
            base,
            client=llm_client,
            model=gec.model,
            system_prompt=system,
            user_template=user,
            temperature=gec.temperature,
            batch_size=gec.batch_size,
        )
        return out

    raise ValueError(f"gec.mode inconnu : {mode!r} (attendu 'none', 'light' ou 'llm').")
