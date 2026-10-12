"""Indexation du dataset vidéo.

Pourquoi ? Avant toute analyse, on veut un tableau (data/index.csv) qui liste
chaque vidéo avec ses métadonnées : sujet, qualité (good/bad), vue, durée,
fps, résolution. Ce tableau sert ensuite à :
  - vérifier que le dataset est complet (25 sujets x 2 qualités x 3 vues) ;
  - faire le découpage leave-one-subject-out (il faut connaître le sujet) ;
  - séparer les résultats par vue.
Les informations viennent du NOM du fichier (sujet, qualité, vue) et de
l'en-tête de la vidéo lue par OpenCV (fps, nombre d'images, résolution).
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import pandas as pd

INDEX_COLUMNS = [
    "subject", "exercise", "quality", "view", "label",
    "fps", "n_frames", "duration_s", "width", "height",
    "folder", "filename", "path",
]


def parse_filename(filename: str, pattern: str) -> dict | None:
    """Extrait sujet / exercice / qualité / vue du nom de fichier.

    Renvoie None si le nom ne respecte pas le format attendu : on préfère
    ignorer (et signaler) un fichier inconnu plutôt que deviner son label.
    """
    match = re.match(pattern, filename)
    return match.groupdict() if match else None


def read_video_metadata(path: Path) -> dict:
    """Lit fps, nombre d'images et résolution depuis l'en-tête de la vidéo.

    Limite : le nombre d'images annoncé par l'en-tête peut différer
    légèrement du nombre réellement décodé. C'est suffisant pour un index.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"fps": None, "n_frames": None, "duration_s": None, "width": None, "height": None}
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = n_frames / fps if fps else None
    return {
        "fps": round(fps, 3) if fps else None,
        "n_frames": n_frames,
        "duration_s": round(duration, 2) if duration else None,
        "width": width,
        "height": height,
    }


def build_index(data_dir: Path, pattern: str) -> tuple[pd.DataFrame, list[str]]:
    """Parcourt good/ et bad/ et construit l'index.

    Renvoie (index, fichiers_ignorés). Le label numérique vaut 1 pour "bad"
    car c'est la classe que l'on cherche à détecter (classe "positive").
    """
    rows, skipped = [], []
    for folder in ("good", "bad"):
        for path in sorted((data_dir / folder).glob("*")):
            info = parse_filename(path.name, pattern)
            # On vérifie aussi que le label du nom correspond au dossier :
            # une incohérence serait une erreur d'étiquetage à signaler.
            if info is None or info["quality"] != folder:
                skipped.append(str(path))
                continue
            rows.append({
                **info,
                "label": 1 if info["quality"] == "bad" else 0,
                **read_video_metadata(path),
                "folder": folder,
                "filename": path.name,
                "path": str(path),
            })
    index = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    return index, skipped


def missing_combinations(index: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Liste les combinaisons (sujet, qualité, vue) absentes.

    Utile pour savoir si chaque sujet a bien ses 6 vidéos : sinon les
    comparaisons par vue ou le découpage par sujet seraient déséquilibrés.
    """
    expected = {
        (s, q, v)
        for s in index["subject"].unique()
        for q in ("good", "bad")
        for v in ("front", "side", "diagonal")
    }
    present = set(zip(index["subject"], index["quality"], index["view"], strict=True))
    return sorted(expected - present)
