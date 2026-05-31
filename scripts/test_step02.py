from __future__ import annotations

"""
Test de l'étape 02 (déduplication), à lancer :

    uv run python scripts/test_step02.py

Deux vérifications :
  1. un petit cas fabriqué, pour voir précisément ce qui fusionne ou non ;
  2. l'effet sur le vrai CSV d'exemple (combien de lignes uniques).
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champslibres_pipeline.config import load_config
from champslibres_pipeline.steps.step01_gec import apply_gec
from champslibres_pipeline.steps.step02_dedup import dedup

SAMPLE = "data/exemple_input.csv"


def test_petit_cas(cfg) -> None:
    print("\n1) Petit cas fabriqué")
    # A et B : identiques à un espace en trop près -> doivent fusionner.
    # C : même mots mais Majuscule -> reste distinct (casse conservée).
    # D : texte différent.
    df = pd.DataFrame(
        {
            cfg.columns.id: ["A", "B", "C", "D"],
            cfg.columns.text: [
                "constat amiable",
                "constat  amiable",   # double espace
                "Constat amiable",    # majuscule
                "autre chose",
            ],
        }
    )
    cfg.gec.mode = "light"
    df = apply_gec(df, cfg)                 # produit la colonne 'clean'
    dedup_df, mapping = dedup(df, cfg)

    print("   Lignes conservées :")
    print(dedup_df[[cfg.columns.id, cfg.columns.clean, "dup_count"]].to_string(index=False))
    print("   Correspondance (origine -> conservé) :")
    print(mapping.to_string(index=False))

    assert len(dedup_df) == 3, f"attendu 3 lignes uniques, obtenu {len(dedup_df)}"
    # A regroupe A et B
    a_count = dedup_df.loc[dedup_df[cfg.columns.id] == "A", "dup_count"].iloc[0]
    assert a_count == 2, f"dup_count de A attendu 2, obtenu {a_count}"
    # B est rattaché à A ; C reste lui-même
    assert mapping.loc[mapping["original_id"] == "B", "kept_id"].iloc[0] == "A"
    assert mapping.loc[mapping["original_id"] == "C", "kept_id"].iloc[0] == "C"
    print("[OK ] B fusionné dans A (dup_count=2), C resté distinct (casse conservée)")


def test_sample(cfg) -> None:
    print("\n2) Effet sur le CSV d'exemple")
    df = pd.read_csv(SAMPLE, dtype=str)
    cfg.gec.mode = "light"
    df = apply_gec(df, cfg)
    dedup_df, mapping = dedup(df, cfg)
    print(f"   {len(df)} réponses -> {len(dedup_df)} uniques "
          f"({len(df) - len(dedup_df)} doublons retirés)")
    assert len(mapping) == len(df), "la correspondance doit couvrir toutes les réponses"
    print("[OK ] correspondance complète, tableau dédupliqué produit")


def main() -> int:
    cfg = load_config("configs/exemple.yaml")
    test_petit_cas(cfg)
    test_sample(cfg)
    print("\nÉtape 02 vérifiée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
