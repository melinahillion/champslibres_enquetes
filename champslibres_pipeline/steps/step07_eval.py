from __future__ import annotations

"""
Étape 07 — Évaluation des accords.

Principe (conforme à la méthode) :
  - les ANNOTATEURS HUMAINS annotent directement en super-catégories (Rxx/Fxx) ;
  - chaque MODÈLE LLM classe en Cxx ; on mappe ensuite Cxx -> Rxx/Fxx via le
    label book ; c'est à CE niveau (Rxx/Fxx) que l'on compare.

On compare donc, au niveau des super-catégories :
  - humain  vs humain   (fiabilité inter-annotateurs)
  - humain  vs modèle   (qualité des modèles)
  - modèle  vs modèle   (cohérence entre modèles)

Le nombre d'annotateurs humains (1..N) et de modèles (1..M) est libre : tout
s'adapte aux colonnes présentes. Métriques : accord observé (Po), kappa de
Cohen (par paire), kappa de Fleiss (multi-juges), avec intervalles de confiance
par bootstrap. Pur pandas/numpy.
"""

import itertools
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import ProjectConfig


# ===========================================================================
# Mapping Cxx -> super-catégorie (Rxx/Fxx) via le label book
# ===========================================================================
def cxx_to_super(book: Dict[str, Any]) -> Dict[str, str]:
    """{code Cxx -> code de sa super-catégorie}. Pour un label book à plat
    (sans Rxx/Fxx), chaque Cxx est sa propre super-catégorie (identité)."""
    mapping: Dict[str, str] = {}
    for cat in book.get("categories", []):
        children = cat.get("children", [])
        if children:
            for ch in children:
                mapping[ch["category_id"]] = cat["code"]
        else:
            mapping[cat["code"]] = cat["code"]
    return mapping


def map_to_super(series: pd.Series, cxx2super: Dict[str, str]) -> pd.Series:
    """Remplace chaque Cxx par sa super-catégorie ; valeurs inconnues (ex. OTHER
    ou codes déjà au niveau super) conservées telles quelles."""
    return series.astype(str).map(lambda v: cxx2super.get(v, v))


# ===========================================================================
# Métriques d'accord (pur numpy)
# ===========================================================================
def percent_agreement(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.asarray(x) == np.asarray(y))) if len(x) else float("nan")


def cohen_kappa(x: Sequence, y: Sequence) -> float:
    """Kappa de Cohen (deux juges, catégories nominales)."""
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    if n == 0:
        return float("nan")
    cats = sorted(set(x.tolist()) | set(y.tolist()))
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for xi, yi in zip(x, y):
        m[idx[xi], idx[yi]] += 1
    po = np.trace(m) / n
    pe = float(((m.sum(1) / n) * (m.sum(0) / n)).sum())
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(df: pd.DataFrame, cols: Sequence[str]) -> float:
    """Kappa de Fleiss sur les items notés par TOUS les juges de `cols`."""
    sub = df[list(cols)].dropna()
    n_raters = len(cols)
    if sub.shape[0] == 0 or n_raters < 2:
        return float("nan")
    sub = sub.astype(str)
    cats = sorted(set(sub.to_numpy().ravel().tolist()))
    cidx = {c: i for i, c in enumerate(cats)}
    table = np.zeros((sub.shape[0], len(cats)))
    for r, (_, row) in enumerate(sub.iterrows()):
        for v in row:
            table[r, cidx[v]] += 1
    p_j = table.sum(0) / (sub.shape[0] * n_raters)
    P_i = (np.square(table).sum(1) - n_raters) / (n_raters * (n_raters - 1))
    pbar = float(P_i.mean())
    pe = float(np.square(p_j).sum())
    if pe >= 1.0:
        return 1.0 if pbar >= 1.0 else 0.0
    return (pbar - pe) / (1 - pe)


def _bootstrap_ci(x, y, stat_fn, B, rng, alpha=0.05):
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    x = np.asarray(x); y = np.asarray(y)
    stats = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        stats[b] = stat_fn(x[idx], y[idx])
    return (float(np.nanpercentile(stats, 100 * alpha / 2)),
            float(np.nanpercentile(stats, 100 * (1 - alpha / 2))))


# ===========================================================================
# Accords par paire + synthèse
# ===========================================================================
def _pretty(col: str) -> str:
    return col[:-7] if col.endswith("__super") else (col[:-5] if col.endswith("__cxx") else col)


_PAIR_LABEL = {"HH": "humain-humain", "HM": "humain-modèle", "MM": "modèle-modèle"}


def pairwise_agreement(df, rater_cols, rater_types, *, bootstrap_B=2000, random_state=42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    for a, b in itertools.combinations(rater_cols, 2):
        sub = df[[a, b]].dropna()
        x = sub[a].astype(str).to_numpy()
        y = sub[b].astype(str).to_numpy()
        n = len(x)
        po, kappa = percent_agreement(x, y), cohen_kappa(x, y)
        po_lo, po_hi = _bootstrap_ci(x, y, percent_agreement, bootstrap_B, rng)
        k_lo, k_hi = _bootstrap_ci(x, y, cohen_kappa, bootstrap_B, rng)
        t = "".join(sorted([rater_types[a], rater_types[b]]))
        rows.append({"juge_a": _pretty(a), "juge_b": _pretty(b), "type": _PAIR_LABEL.get(t, t),
                     "n": n, "Po": round(po, 3), "Po_IC95_bas": round(po_lo, 3), "Po_IC95_haut": round(po_hi, 3),
                     "kappa": round(kappa, 3), "kappa_IC95_bas": round(k_lo, 3), "kappa_IC95_haut": round(k_hi, 3)})
    return pd.DataFrame(rows)


def by_type_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pairwise
    return (pairwise.groupby("type")
            .agg(n_paires=("type", "size"), Po_moyen=("Po", "mean"), kappa_moyen=("kappa", "mean"))
            .round(3).reset_index())


# ===========================================================================
# Construction des colonnes-modèles (classification Cxx par modèle)
# ===========================================================================
def classify_models(texts, ids, book, cfg: ProjectConfig, *, llm_client, models=None, progress=True):
    """Classe les réponses avec chaque modèle de `models` (mode llm) et renvoie
    un DataFrame [id, '<modèle>__cxx' par modèle]."""
    from .step06_classify import run_classification

    models = list(models if models is not None else cfg.eval.models)
    base = pd.DataFrame({cfg.columns.id: [str(i) for i in ids], cfg.columns.clean: list(texts)})
    out = pd.DataFrame({cfg.columns.id: base[cfg.columns.id]})
    for mname in models:
        c = cfg.model_copy(deep=True)
        c.classf.mode = "llm"
        c.classf.model = mname
        res = run_classification(base, book, c, llm_client=llm_client, progress=progress)
        out[f"{mname}__cxx"] = res["code"].to_numpy()
    return out


# ===========================================================================
# Orchestrateur
# ===========================================================================
def run_evaluation(df, book, cfg: ProjectConfig, *, model_cxx_columns: Optional[List[str]] = None):
    """
    Évalue les accords au niveau des super-catégories.
    `df` contient : la colonne id, les colonnes d'annotateurs humains
    (cfg.eval.human_columns, déjà en Rxx/Fxx) et une colonne Cxx par modèle.
    Renvoie {pairwise, by_type, fleiss, raters}.
    """
    df = df.copy()
    human_cols = [c for c in cfg.eval.human_columns if c in df.columns]

    if model_cxx_columns is None:
        model_cxx_columns = [f"{m}__cxx" for m in cfg.eval.models if f"{m}__cxx" in df.columns]

    cxx2super = cxx_to_super(book)
    model_super_cols: List[str] = []
    for col in model_cxx_columns:
        base = col[:-5] if col.endswith("__cxx") else col
        scol = f"{base}__super"
        df[scol] = map_to_super(df[col], cxx2super)
        model_super_cols.append(scol)

    rater_cols = human_cols + model_super_cols
    rater_types = {**{c: "H" for c in human_cols}, **{c: "M" for c in model_super_cols}}
    if len(rater_cols) < 2:
        raise ValueError("Évaluation : il faut au moins 2 juges (humains et/ou modèles).")

    pairwise = pairwise_agreement(df, rater_cols, rater_types,
                                  bootstrap_B=cfg.eval.bootstrap_B, random_state=cfg.eval.random_state)
    fleiss = {}
    if len(human_cols) >= 2:
        fleiss["humains"] = round(fleiss_kappa(df, human_cols), 3)
    if len(rater_cols) >= 2:
        fleiss["tous_juges"] = round(fleiss_kappa(df, rater_cols), 3)

    display = {c: f"Humain {i + 1}" for i, c in enumerate(human_cols)}
    display.update({c: _pretty(c) for c in model_super_cols})

    return {"pairwise": pairwise, "by_type": by_type_summary(pairwise), "fleiss": fleiss,
            "raters": {"humains": human_cols, "modeles": [_pretty(c) for c in model_super_cols]},
            "table": df, "rater_cols": rater_cols, "human_cols": human_cols,
            "model_super_cols": model_super_cols, "display": display}