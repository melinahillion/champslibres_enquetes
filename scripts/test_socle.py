from __future__ import annotations

"""
Petit test du SOCLE, à lancer dans un service VSCode-python du SSP Cloud :

    uv run python scripts/test_socle.py

Ce script ne traite aucune donnée. Il vérifie juste que les fondations
fonctionnent : variables d'environnement présentes, configuration valide,
connexion à llm.lab, liste des modèles, et un petit appel de génération.
"""

import os
import sys

# On peut lancer le script depuis la racine du projet sans installer le package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champslibres_pipeline.config import load_config
from champslibres_pipeline.llm import get_llm_client, list_models


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "OK " if ok else "!! "
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))


def main() -> int:
    print("=" * 60)
    print(" Test du socle champslibres_pipeline")
    print("=" * 60)

    # --- 1) Variables d'environnement -------------------------------------
    # On ne montre JAMAIS la valeur d'un secret, juste s'il est présent.
    print("\n1) Variables d'environnement")
    has_key = bool(os.environ.get("LLM_LAB_API_KEY"))
    check("LLM_LAB_API_KEY présente", has_key,
          "" if has_key else "à ajouter dans Vault puis relancer le service")
    s3_endpoint = os.environ.get("AWS_S3_ENDPOINT", "")
    check("AWS_S3_ENDPOINT (injectée par le SSP Cloud)", bool(s3_endpoint), s3_endpoint)

    if not has_key:
        print("\nClé LLM manquante : on s'arrête ici. Ajoute le secret dans Vault.")
        return 1

    # --- 2) Chargement de la configuration --------------------------------
    print("\n2) Configuration")
    try:
        cfg = load_config("configs/exemple.yaml")
        check("configs/exemple.yaml chargée et validée", True,
              f"projet='{cfg.project}', colonnes={cfg.columns.id}/{cfg.columns.text}")
    except Exception as e:  # noqa: BLE001
        check("chargement de la config", False, str(e).splitlines()[0])
        return 1

    # --- 3) Liste des modèles disponibles sur llm.lab ---------------------
    print("\n3) Modèles disponibles sur llm.lab")
    try:
        models = list_models()
        check("connexion à llm.lab + /models", True,
              f"{len(models['all'])} modèles au total")
        print("    Génération :", ", ".join(models["generation"][:8]) or "(aucun)")
        print("    Embedding  :", ", ".join(models["embedding"][:8]) or "(aucun détecté)")
    except Exception as e:  # noqa: BLE001
        check("connexion à llm.lab", False, str(e).splitlines()[0])
        return 1

    # --- 4) Un petit appel de génération ----------------------------------
    print("\n4) Mini appel de génération")
    if not models["generation"]:
        check("génération", False, "aucun modèle de génération détecté")
    else:
        model = models["generation"][0]
        client = get_llm_client(cfg.llm)
        try:
            answer = client.complete(
                [{"role": "user", "content": "Réponds par un seul mot : bonjour."}],
                model=model, max_tokens=10,
            )
            check(f"appel à '{model}'", True, f"réponse = {answer!r}")
        except Exception as e:  # noqa: BLE001
            check(f"appel à '{model}'", False, str(e).splitlines()[0])
            return 1

    # --- 5) (optionnel) un mini embedding ---------------------------------
    if models["embedding"]:
        print("\n5) Mini appel d'embedding")
        emb_model = models["embedding"][0]
        client = get_llm_client(cfg.llm)
        try:
            vecs = client.embed(["maison", "domicile"], model=emb_model)
            check(f"embedding via '{emb_model}'", True,
                  f"dimension = {len(vecs[0])}")
        except Exception as e:  # noqa: BLE001
            check(f"embedding via '{emb_model}'", False, str(e).splitlines()[0])

    print("\nSocle vérifié. Tu peux passer à l'étape 01.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
