from __future__ import annotations

"""
Étape 06 — Classification des réponses dans le label book.

Trois modes (classf.mode) :
  - "sans"   : on réutilise l'affectation du clustering (chaque réponse -> son Cxx ;
               le 'bruit' -> OTHER). Aucun appel LLM. Sert de base de comparaison.
  - "llm"    : classification zéro-shot par le LLM, par lots, répétée n_iterations
               fois -> vote majoritaire, avec une file de reprise pour les cas non
               résolus. C'est la méthode principale.
  - "humain" : on lit des codes fournis par un humain (colonne classf.human_column)
               et on les valide contre le label book.

On classe au niveau des FEUILLES du label book (les Cxx, ou les catégories de
premier niveau sans enfants, plus OTHER) ; la super-catégorie parente est
reportée dans le résultat pour permettre l'agrégation.

Entrée : un DataFrame avec les colonnes columns.id et columns.clean.
Sortie : un DataFrame [id, text, code, label, parent_code, parent_label, accord].
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import ProjectConfig
from ..llm.base import LLMClient
from ..prompts import load_text
from ..prompts.classf import CLASSIFY_SYSTEM, CLASSIFY_USER
from .step05_label import _loads_loose, cluster_code


# ===========================================================================
# Catégories assignables (feuilles du label book)
# ===========================================================================
def assignable_labels(book: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Aplati le label book en feuilles classables, en conservant le parent."""
    leaves: List[Dict[str, Any]] = []
    for cat in book.get("categories", []):
        children = cat.get("children", [])
        if children:
            for ch in children:
                leaves.append({"code": ch["category_id"], "label": ch.get("title", ""),
                               "description": ch.get("description", ""),
                               "parent_code": cat["code"], "parent_label": cat.get("label", "")})
        else:
            leaves.append({"code": cat["code"], "label": cat.get("label", ""),
                           "description": cat.get("description", ""),
                           "parent_code": None, "parent_label": None})
    return leaves


def _format_labels(leaves: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(f"- {l['code']} : {l['label']} — {l['description']}" for l in leaves)


# ===========================================================================
# Classification LLM (par lots)
# ===========================================================================
def _classify_schema() -> Dict[str, Any]:
    return {"title": "assignments", "type": "object",
            "properties": {"assignments": {"type": "array", "items": {"type": "object",
                "properties": {"id": {"type": "string"}, "code": {"type": "string"}},
                "required": ["id", "code"]}}},
            "required": ["assignments"]}


def _chunks(seq: List[Any], n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _one_pass(items, labels_block, valid, cfg, *, llm_client, question, context, temperature):
    """Un passage complet de classification (par lots). Renvoie {id: code valide}."""
    user_t = load_text(CLASSIFY_USER, cfg.classf.prompt_file)
    batch_msgs = []
    for ch in _chunks(items, max(1, cfg.classf.batch_size)):
        responses = "\n".join(f"- [{i}] {t}" for i, t in ch)
        user = user_t.format(context=context or "(non précisé)", question=question or "(non précisée)",
                             labels=labels_block, responses=responses)
        batch_msgs.append([{"role": "system", "content": CLASSIFY_SYSTEM}, {"role": "user", "content": user}])
    schema = _classify_schema() if cfg.classf.enforce_json_schema else None
    raws = llm_client.complete_batch(batch_msgs, model=cfg.classf.model, temperature=temperature,
                                     max_tokens=64 + 24 * cfg.classf.batch_size, json_schema=schema)
    out: Dict[str, str] = {}
    for raw in raws:
        try:
            data = _loads_loose(raw)
            arr = data.get("assignments", []) if isinstance(data, dict) else data
            for a in arr:
                i = str(a.get("id", "")).strip()
                code = str(a.get("code", "")).strip()
                if i and code in valid:
                    out[i] = code
        except Exception:
            continue
    return out


def _vote(maps: List[Dict[str, str]], ids: List[str]) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Vote majoritaire par id ; égalité -> on préfère un code non-OTHER (ordre stable)."""
    winner: Dict[str, str] = {}
    agree: Dict[str, float] = {}
    for i in ids:
        votes = [m[i] for m in maps if i in m]
        if not votes:
            continue
        counts = Counter(votes).most_common()
        top_n = counts[0][1]
        cands = [code for code, n in counts if n == top_n]
        if len(cands) > 1:
            non_other = [x for x in cands if x != "OTHER"]
            chosen = sorted(non_other or cands)[0]
        else:
            chosen = cands[0]
        winner[i] = chosen
        agree[i] = top_n / len(maps)
    return winner, agree


# ===========================================================================
# Orchestrateur
# ===========================================================================
def run_classification(df, book, cfg: ProjectConfig, *, llm_client: Optional[LLMClient] = None,
                       clusters: Optional[Sequence[int]] = None):
    import pandas as pd

    leaves = assignable_labels(book)
    valid = {l["code"] for l in leaves}
    by_code = {l["code"]: l for l in leaves}
    ids = [str(x) for x in df[cfg.columns.id].tolist()]
    texts = df[cfg.columns.clean].astype(str).tolist()
    mode = cfg.classf.mode

    if mode == "sans":
        if clusters is None:
            raise ValueError("mode 'sans' : 'clusters' requis (labels du clustering).")
        codes = {}
        for i, c in zip(ids, clusters):
            cc = cluster_code(int(c)) if int(c) != -1 else "OTHER"
            codes[i] = cc if cc in valid else "OTHER"
        agree = {i: 1.0 for i in ids}

    elif mode == "humain":
        col = cfg.classf.human_column
        if col not in df.columns:
            raise ValueError(f"mode 'humain' : colonne '{col}' absente du DataFrame.")
        codes = {}
        for i, v in zip(ids, df[col].astype(str).tolist()):
            v = v.strip()
            codes[i] = v if v in valid else "OTHER"
        agree = {i: 1.0 for i in ids}

    elif mode == "llm":
        if llm_client is None or not cfg.classf.model:
            raise ValueError("mode 'llm' : llm_client + classf.model requis.")
        labels_block = _format_labels(leaves)
        items = list(zip(ids, texts))
        q, ctx = cfg.survey.question, cfg.survey.context
        maps = [_one_pass(items, labels_block, valid, cfg, llm_client=llm_client,
                          question=q, context=ctx, temperature=cfg.classf.temperature)
                for _ in range(max(1, cfg.classf.n_iterations))]
        codes, agree = _vote(maps, ids)
        missing = [(i, t) for i, t in items if i not in codes]
        if missing:  # file de reprise, à température 0
            retry = _one_pass(missing, labels_block, valid, cfg, llm_client=llm_client,
                              question=q, context=ctx, temperature=0.0)
            for i, _t in missing:
                codes[i] = retry.get(i, "OTHER")
                agree.setdefault(i, 0.0)

    else:
        raise ValueError(f"classf.mode inconnu : {mode!r} (attendu : sans | llm | humain).")

    rows = []
    for i, t in zip(ids, texts):
        code = codes.get(i, "OTHER")
        leaf = by_code.get(code, {})
        rows.append({"id": i, "text": t, "code": code, "label": leaf.get("label", ""),
                     "parent_code": leaf.get("parent_code"), "parent_label": leaf.get("parent_label"),
                     "accord": round(agree.get(i, 0.0), 3)})
    return pd.DataFrame(rows)
