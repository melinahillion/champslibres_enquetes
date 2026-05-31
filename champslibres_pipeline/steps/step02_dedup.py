from __future__ import annotations

"""
Étape 02 — Déduplication (sans LLM).

But : alléger les étapes suivantes (embeddings, clustering, classification) en
ne gardant qu'UNE occurrence de chaque réponse identique, tout en conservant
le moyen de revenir au corpus complet.

Deux sorties :
  - un tableau "dédupliqué" : une ligne par texte unique, avec un compteur
    'dup_count' (combien de réponses identiques ont été regroupées) ;
  - une table de correspondance : pour CHAQUE identifiant d'origine, l'identifiant
    de la ligne conservée. C'est ce qui permettra, en fin de pipeline, de
    réattribuer la classe trouvée à toutes les réponses identiques.

Comme à l'étape 01, le coeur dedup(df, cfg) travaille sur un DataFrame en
mémoire (facile à tester). La lecture/écriture S3 (run) viendra au câblage.

L'égalité entre deux textes est jugée APRÈS normalisation légère
(normalize_light, réutilisée de l'étape 01) : seules les différences de
typographie (espaces, apostrophes, accents unicode) sont ignorées. La casse
(MAJUSCULE/minuscule) est, elle, conservée — deux textes qui ne diffèrent que
par la casse restent donc distincts.
"""

from typing import Tuple

import pandas as pd

from ..config import ProjectConfig
from .step01_gec import normalize_light  # une seule source de vérité

# Noms des colonnes produites
DUP_COUNT_COL = "dup_count"
MAP_ORIGINAL_COL = "original_id"
MAP_KEPT_COL = "kept_id"
_NORM_COL = "_norm"  # colonne de travail interne (non exportée)


def dedup(df: pd.DataFrame, cfg: ProjectConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Renvoie (dedup_df, mapping_df).

      dedup_df : une ligne par texte unique, colonnes
                 [id, text, clean, dup_count].
      mapping_df : une ligne par réponse d'origine, colonnes
                 [original_id, kept_id].
    """
    id_col = cfg.columns.id
    text_col = cfg.columns.text
    clean_col = cfg.columns.clean

    for col in (id_col, text_col, clean_col):
        if col not in df.columns:
            raise KeyError(
                f"Colonne {col!r} absente. L'étape 02 attend la sortie de "
                f"l'étape 01 (colonne '{clean_col}'). Lance d'abord l'étape 01."
            )

    work = df.copy()
    # clé de regroupement : texte nettoyé puis normalisé légèrement
    work[_NORM_COL] = work[clean_col].astype(str).map(normalize_light)

    # combien de réponses partagent chaque texte normalisé
    dup_counts = work.groupby(_NORM_COL, sort=False).size()

    # on garde la PREMIÈRE occurrence de chaque texte normalisé
    kept = work.drop_duplicates(subset=_NORM_COL, keep="first").copy()
    kept[DUP_COUNT_COL] = kept[_NORM_COL].map(dup_counts).astype(int)

    # table de correspondance : tout id d'origine -> id conservé pour ce texte
    norm_to_kept = dict(zip(kept[_NORM_COL], kept[id_col].astype(str)))
    mapping = pd.DataFrame(
        {
            MAP_ORIGINAL_COL: work[id_col].astype(str).values,
            MAP_KEPT_COL: work[_NORM_COL].map(norm_to_kept).values,
        }
    )

    dedup_df = kept[[id_col, text_col, clean_col, DUP_COUNT_COL]].reset_index(drop=True)
    return dedup_df, mapping.reset_index(drop=True)
