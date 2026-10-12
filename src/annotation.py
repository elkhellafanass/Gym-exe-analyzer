"""Annotation manuelle en aveugle des vidéos de profil.

Pourquoi ? Le dataset ne dit pas QUELLE erreur contient une vidéo "bad".
Sans cette information, on ne peut pas savoir si une règle (ex. "talons
décollés") détecte vraiment ce qu'elle prétend. L'auteur annote donc lui-même.

Pourquoi "en aveugle" ?
  - Label caché : si l'annotateur sait que la vidéo est "bad", il cherchera
    un défaut (biais de confirmation). On mélange donc good et bad, et le nom
    de fichier (qui contient le label) n'est jamais affiché.
  - Aucun résultat des règles ni squelette affiché : l'annotation doit être
    indépendante de ce qu'elle sert à évaluer.
  - Ordre aléatoire mais FIXE (graine) : reproductible, et good/bad d'un même
    sujet ne se suivent pas forcément.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pandas as pd

# Types d'erreur annotables. Les définitions opérationnelles (affichées dans
# l'outil) reprennent les critères de Myer et al. (2014), voir docs/SOURCES.md.
ERROR_TYPES = {
    "depth": ("Profondeur insuffisante",
              "Au point bas, la hanche reste nettement au-dessus du genou."),
    "heel_rise": ("Talons décollés",
                  "Le talon quitte le sol (appui sur l'avant du pied)."),
    "trunk_lean": ("Tronc trop penché / dos arrondi",
                   "Le tronc est nettement plus penché que le tibia, ou le dos s'arrondit."),
    "head": ("Tête / cou penchés",
             "La tête est baissée ou projetée vers l'avant (cou non vertical)."),
    "knees_forward": ("Genoux très en avant",
                      "Les genoux dépassent nettement la pointe des pieds."),
    "other": ("Autre", "Autre défaut (préciser en commentaire)."),
}
# Fréquence de chaque erreur dans la vidéo (une colonne par type d'erreur)
FREQUENCIES = ["absent", "quelques répétitions", "la plupart des répétitions"]
# Avis global de l'annotateur : permet de mesurer son accord avec le label du dataset
OVERALL = ["bonne", "mauvaise", "incertain"]
ANNOT_COLUMNS = ["anon_id", "filename", "overall", *ERROR_TYPES, "comment", "annotated_at"]


def blind_order(index: pd.DataFrame, view: str = "side", seed: int = 42) -> pd.DataFrame:
    """Vidéos d'une vue, mélangées avec une graine fixe, avec un identifiant
    anonyme (V01, V02...). Le label reste dans le tableau pour l'analyse,
    mais l'interface ne l'affiche jamais."""
    subset = index[index["view"] == view].sort_values("filename").reset_index(drop=True)
    order = np.random.default_rng(seed).permutation(len(subset))
    out = subset.iloc[order].reset_index(drop=True)
    out.insert(0, "anon_id", [f"V{i + 1:02d}" for i in range(len(out))])
    return out


def load_annotations(path: Path) -> pd.DataFrame:
    if Path(path).exists():
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=ANNOT_COLUMNS)


def save_annotation(path: Path, record: dict) -> pd.DataFrame:
    """Ajoute ou remplace l'annotation d'une vidéo (clé : anon_id)."""
    path = Path(path)
    table = load_annotations(path)
    table = table[table["anon_id"] != record["anon_id"]]
    record = {**record, "annotated_at": datetime.now().isoformat(timespec="seconds")}
    table = pd.concat([table, pd.DataFrame([record])], ignore_index=True)
    table = table.sort_values("anon_id")[ANNOT_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def ensure_h264(src: Path, cache_dir: Path) -> Path:
    """Convertit une vidéo en H.264 (lisible par les navigateurs), sans le son.

    Les vidéos du dataset sont en HEVC, que la plupart des navigateurs ne
    lisent pas. Le son est retiré : inutile pour l'analyse, et il peut
    contenir des voix (données personnelles). Le cache reste dans data/ (non versionné).
    Le nom du fichier converti est anonyme pour ne pas révéler le label.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # hash stable (hash() de Python change à chaque lancement)
    out = cache_dir / f"{hashlib.sha1(Path(src).name.encode()).hexdigest()[:12]}.mp4"
    if not out.exists():
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
             "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
             "-vf", "scale=-2:'min(960,ih)'", str(out)],
            check=True)
    return out
