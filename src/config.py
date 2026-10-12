"""Lecture de config.yaml.

Pourquoi un module dédié ? Tous les seuils et chemins sont centralisés dans
config.yaml pour qu'un jury (ou un expert) puisse les relire et les modifier
sans toucher au code. Ce module est le seul endroit qui lit ce fichier.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

# Racine du projet = dossier parent de src/
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Charge config.yaml et renvoie un dictionnaire Python."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def squat_data_dir(config: dict | None = None) -> Path:
    """Renvoie le dossier du dataset squat.

    La variable d'environnement SQUAT_DATA_DIR est prioritaire : elle permet de
    lancer le projet sur une autre machine sans modifier config.yaml.
    """
    config = config or load_config()
    env_var = config["dataset"]["env_var"]
    return Path(os.environ.get(env_var) or config["dataset"]["squat_dir"])
