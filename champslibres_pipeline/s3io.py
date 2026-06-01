from __future__ import annotations

"""
Accès S3 (MinIO de SSP Cloud) et lecture/écriture de fichiers.

Construit un système de fichiers s3fs à partir des variables d'environnement
injectées par Onyxia (AWS_S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
AWS_SESSION_TOKEN) — exactement le schéma éprouvé des notebooks existants.

Toutes les fonctions d'I/O acceptent un `fs` (n'importe quel système de fichiers
fsspec), ce qui permet de tester en local sans S3.
"""

import io
import json
import os
from typing import Any, Optional

import pandas as pd

from .config import ProjectConfig


def make_s3_filesystem(cfg: ProjectConfig):
    """Crée le système de fichiers S3 (s3fs) depuis les variables d'environnement."""
    import s3fs

    s = cfg.s3
    endpoint = os.environ[s.endpoint_env]
    return s3fs.S3FileSystem(
        anon=False,
        key=os.environ.get(s.access_key_env),
        secret=os.environ.get(s.secret_key_env),
        token=os.environ.get(s.token_env),
        client_kwargs={"endpoint_url": f"https://{endpoint}"},
        config_kwargs={"signature_version": "s3v4"},
    )


# --------------------------------------------------------------------------- I/O
def read_csv(fs, path: str, **kw) -> pd.DataFrame:
    with fs.open(path, "rb") as f:
        return pd.read_csv(f, **kw)


def write_csv(fs, df: pd.DataFrame, path: str, **kw) -> str:
    _ensure_parent(fs, path)
    with fs.open(path, "w") as f:
        df.to_csv(f, index=False, **kw)
    return path


def read_json(fs, path: str) -> Any:
    with fs.open(path, "r") as f:
        return json.load(f)


def write_json(fs, obj: Any, path: str) -> str:
    _ensure_parent(fs, path)
    with fs.open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def write_text(fs, text: str, path: str) -> str:
    _ensure_parent(fs, path)
    with fs.open(path, "w") as f:
        f.write(text)
    return path


def write_bytes(fs, data: bytes, path: str) -> str:
    _ensure_parent(fs, path)
    with fs.open(path, "wb") as f:
        f.write(data)
    return path


def upload_file(fs, local_path: str, dest: str) -> str:
    """Téléverse un fichier local (ex. dendrogramme.html) vers fs."""
    with open(local_path, "rb") as f:
        return write_bytes(fs, f.read(), dest)


def save_npy(fs, array, path: str) -> str:
    """Sérialise un tableau numpy (embeddings) en .npy sur fs."""
    import numpy as np
    buf = io.BytesIO()
    np.save(buf, np.asarray(array))
    return write_bytes(fs, buf.getvalue(), path)


def load_npy(fs, path: str):
    import numpy as np
    with fs.open(path, "rb") as f:
        return np.load(io.BytesIO(f.read()))


def exists(fs, path: str) -> bool:
    try:
        return fs.exists(path)
    except Exception:
        return False


def _ensure_parent(fs, path: str) -> None:
    """Crée le dossier parent si le système de fichiers le requiert (local)."""
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        try:
            fs.makedirs(parent, exist_ok=True)
        except Exception:
            pass
