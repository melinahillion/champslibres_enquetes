from __future__ import annotations

"""
Interface en ligne de commande (CLI) de champslibres.

    champslibres run      --config configs/auteur_isi2026.yaml   # pipeline complet
    champslibres rerun    --config ...        # re-cluster/label/classif depuis embeddings en cache
    champslibres describe --config ...        # stats descriptives des champs libres
    champslibres eval     --config ...         # accords humains/modèles
    champslibres list-models                   # modèles disponibles sur llm.lab
    champslibres init     --project NOM --bucket B --source s3://...  # config de départ

Toutes les commandes (sauf list-models/init) s'appuient sur la classe Project.
"""

from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="Pipeline de classification de champs libres (INSEE/SSMSI).")


def _project(config: str):
    from .project import Project
    return Project(config)


@app.command()
def scaffold(config: str = typer.Option(..., "--config", "-c")):
    """Crée l'arborescence du projet sur S3 (dossiers + notes), pour y déposer le label book."""
    _project(config).scaffold()


@app.command()
def run(config: str = typer.Option(..., "--config", "-c", help="chemin du fichier YAML de config")):
    """Pipeline complet : entrée brute -> clusters -> label book -> classification (écrit sur S3)."""
    _project(config).run_all()


@app.command("build-labelbook")
def build_labelbook(config: str = typer.Option(..., "--config", "-c")):
    """Étapes 00->05 : décrit, clusterise et produit le label book + le dendrogramme."""
    _project(config).build_label_book()


@app.command()
def classify(config: str = typer.Option(..., "--config", "-c")):
    """Étape 06 : classe les réponses (label book humain si classf.label_book est défini)."""
    _project(config).classify()


@app.command()
def rerun(config: str = typer.Option(..., "--config", "-c")):
    """Re-exécute clustering/label/classification depuis les embeddings en cache (S3)."""
    _project(config).rerun_from_embeddings()


@app.command()
def describe(config: str = typer.Option(..., "--config", "-c")):
    """Statistiques descriptives des champs libres (nombre de mots)."""
    print(_project(config).describe().to_string(index=False))


@app.command()
def eval(config: str = typer.Option(..., "--config", "-c")):
    """Accords humain-humain / humain-modèle / modèle-modèle sur le jeu annoté."""
    res = _project(config).evaluate()
    print(res["pairwise"].to_string(index=False))
    print("\nFleiss:", res["fleiss"])


@app.command("list-models")
def list_models_cmd(config: Optional[str] = typer.Option(None, "--config", "-c")):
    """Liste les modèles llm.lab (génération / embeddings)."""
    from .llm import list_models
    if config:
        from .config import load_config
        c = load_config(config).llm
        m = list_models(base_url=c.base_url, api_key_env=c.api_key_env)
    else:
        m = list_models()
    print("Génération :", ", ".join(m.get("generation", [])) or "(aucun)")
    print("Embeddings :", ", ".join(m.get("embedding", [])) or "(aucun)")


@app.command()
def init(
    project: str = typer.Option(..., help="nom du projet, ex. AUTEUR_ISI2026"),
    bucket: str = typer.Option("projet-champs-libres-vrs", help="bucket S3"),
    source: str = typer.Option(..., help="chemin S3 du CSV brut (id + texte)"),
    out: str = typer.Option("configs/mon_projet.yaml", help="où écrire la config"),
):
    """Crée un fichier de configuration de départ à compléter."""
    import os
    from .config import load_config, dump_config
    base = load_config("configs/exemple.yaml")
    base.project = project
    base.bucket = bucket
    base.io.source_csv = source
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    dump_config(base, out)
    print(f"Config créée : {out}\nÉdite-la (modèles, grouping, eval) puis : champslibres run -c {out}")


def main():
    app()


if __name__ == "__main__":
    main()
