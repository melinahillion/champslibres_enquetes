from __future__ import annotations

"""
Test du câblage S3 + API Projet + CLI, en LOCAL (fsspec 'file' simule S3) :
  uv run python scripts/test_wiring.py
"""

import json
import os
import sys
import tempfile

import fsspec
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champslibres_pipeline import s3io
from champslibres_pipeline.config import load_config
from champslibres_pipeline.paths import ProjectPaths
from champslibres_pipeline.project import Project

LFS = fsspec.filesystem("file")


def test_s3io_roundtrip(tmp):
    print("\n1) I/O (CSV / JSON / NPY) via fsspec local")
    df = pd.DataFrame({"a": [1, 2], "b": ["x,y", "z"]})
    s3io.write_csv(LFS, df, f"{tmp}/sub/t.csv")
    assert s3io.read_csv(LFS, f"{tmp}/sub/t.csv").shape == (2, 2)
    s3io.write_json(LFS, {"k": [1, 2]}, f"{tmp}/sub/t.json")
    assert s3io.read_json(LFS, f"{tmp}/sub/t.json")["k"] == [1, 2]
    s3io.save_npy(LFS, np.arange(6).reshape(2, 3), f"{tmp}/sub/e.npy")
    assert s3io.load_npy(LFS, f"{tmp}/sub/e.npy").shape == (2, 3)
    print("[OK ] aller-retour CSV/JSON/NPY + création des dossiers parents")


def test_paths(tmp):
    print("\n2) Plan de nommage")
    cfg = load_config("configs/auteur_isi2026.yaml")
    cfg.bucket = f"{tmp}/bucket"
    p = ProjectPaths(cfg)
    assert p.root == f"{tmp}/bucket/champslibres/AUTEUR_ISI2026"
    assert p.clusters.endswith("04_cluster/clusters.csv")
    assert p.label_book.endswith("05_label/label_book.json")
    print("[OK ]", p.root)


def test_describe(tmp):
    print("\n3) Project.describe (local, sans LLM)")
    cfg = load_config("configs/auteur_isi2026.yaml")
    cfg.bucket = f"{tmp}/bucket"
    cfg.io.source_csv = f"{tmp}/input.csv"
    pd.DataFrame({"SEQ_ID": ["a", "b", "c"],
                  "text_brut": ["un deux trois", "", "quatre cinq"]}).to_csv(cfg.io.source_csv, index=False)
    proj = Project(cfg, fs=LFS)
    desc = proj.describe()
    print(desc.to_string(index=False))
    assert int(desc["nb_sequences_non_vides"].iloc[0]) == 2
    assert LFS.exists(proj.paths.describe("input"))
    print("[OK ] stats écrites sous 00_describe/")


def test_evaluate(tmp):
    print("\n4) Project.evaluate (local, 2 humains, 0 modèle)")
    cfg = load_config("configs/auteur_isi2026.yaml")
    cfg.bucket = f"{tmp}/bucket"
    cfg.eval.models = []
    cfg.eval.human_columns = ["CODE1", "CODE2"]
    cfg.eval.annotated_csv = f"{tmp}/annot.csv"
    cfg.eval.bootstrap_B = 100
    pd.DataFrame({
        "SEQ_ID": [f"i{i}" for i in range(6)],
        "text_brut": ["t"] * 6,
        "CODE1": ["F01", "F01", "F02", "F02", "OTHER", "F01"],
        "CODE2": ["F01", "F01", "F02", "F01", "OTHER", "F01"],
    }).to_csv(cfg.eval.annotated_csv, index=False)
    book = {"categories": [
        {"code": "F01", "label": "A", "children": [{"category_id": "C00", "title": "a"}]},
        {"code": "F02", "label": "B", "children": [{"category_id": "C01", "title": "b"}]},
        {"code": "OTHER", "label": "Autre", "children": []}]}
    proj = Project(cfg, fs=LFS)
    s3io.write_json(LFS, book, proj.paths.label_book)
    res = proj.evaluate()
    print(res["pairwise"][["juge_a", "juge_b", "type", "Po", "kappa"]].to_string(index=False))
    assert LFS.exists(proj.paths.eval_pairwise) and LFS.exists(proj.paths.eval_fleiss)
    assert len(res["pairwise"]) == 1 and res["pairwise"]["type"].iloc[0] == "humain-humain"
    print("[OK ] éval écrite sous 07_eval/")


def test_cli():
    print("\n5) CLI")
    from typer.testing import CliRunner
    from champslibres_pipeline.cli import app
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ("run", "rerun", "describe", "eval", "list-models", "init"):
        assert cmd in r.output
    print("[OK ] commandes :", "run / rerun / describe / eval / list-models / init")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        test_s3io_roundtrip(tmp); test_paths(tmp); test_describe(tmp); test_evaluate(tmp)
    test_cli()
    print("\nCâblage vérifié (local).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
