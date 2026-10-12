"""Orchestrateur : enchaîne les 4 étapes du pipeline sur une vidéo.

    vidéo -> pose.py (points) -> angles.py (indicateurs) -> reps.py
          (répétitions) -> rules.py (erreurs) -> résultat

Ce même code sert à l'app Streamlit ET aux scripts d'entraînement /
d'évaluation : ce qui est évalué dans le mémoire est exactement ce que
l'utilisateur voit dans l'app.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.angles import compute_indicators
from src.config import ROOT
from src.pose import PoseSeries, clean_pose, ensure_model, extract_pose
from src.reps import segment_reps
from src.rules import evaluate_reps, summarize_video

VIEWS = ("front", "side", "diagonal")


@dataclass
class Analysis:
    view: str
    pose: PoseSeries                 # points nettoyés et lissés
    indicators: pd.DataFrame         # une ligne par image
    info: dict                       # côté utilisé, unités de longueur
    reps: pd.DataFrame               # une ligne par répétition
    rep_results: pd.DataFrame        # une ligne par (répétition, règle)
    summary: pd.DataFrame            # une ligne par règle
    warnings: list[str] = field(default_factory=list)

    def rep_table(self) -> pd.DataFrame:
        """Tableau large : une ligne par répétition, une colonne par règle."""
        if self.rep_results.empty:
            return self.reps.copy()
        values = self.rep_results.pivot(index="rep", columns="rule", values="value")
        errors = self.rep_results.pivot(index="rep", columns="rule", values="error")
        errors.columns = [f"{c}_error" for c in errors.columns]
        table = self.reps.set_index("rep").join(values).join(errors).reset_index()
        table["tempo_ratio"] = table["descent_s"] / table["ascent_s"]
        return table


def analyze_pose(raw: PoseSeries, view: str, config: dict,
                 exercise: str = "squat") -> Analysis:
    """Analyse à partir des points bruts (déjà extraits)."""
    if view not in VIEWS:
        raise ValueError(f"Vue inconnue : {view} (attendu : {VIEWS})")
    pc, rc = config["pose"], config["reps"]
    ex = config["exercises"][exercise]
    pose = clean_pose(raw, pc["min_visibility"], pc["max_gap_frames"],
                      pc["smooth_window"], pc["smooth_polyorder"])
    indicators, info = compute_indicators(pose)
    reps = segment_reps(indicators["hip_y"].to_numpy(), pose.fps,
                        rc["min_travel"], rc["min_interval_s"], rc["edge_fraction"])
    rep_results = evaluate_reps(indicators, reps, view, ex["rules"])
    summary = summarize_video(rep_results, ex["rules"], ex["min_reps_for_video_error"])
    return Analysis(view, pose, indicators, info, reps, rep_results, summary,
                    quality_warnings(raw, reps, view))


def quality_warnings(raw: PoseSeries, reps: pd.DataFrame, view: str) -> list[str]:
    """Avertissements affichés à l'utilisateur : quand le résultat est fragile."""
    warnings = []
    rate = float(np.mean(raw.detected()))
    if rate < 0.9:
        warnings.append(f"Personne détectée sur seulement {rate:.0%} des images.")
    if raw.fps < 25:
        warnings.append(f"Vidéo à {raw.fps:.0f} images/s : les durées (tempo) sont "
                        f"approximatives (±{1 / raw.fps:.2f} s).")
    if len(reps) == 0:
        warnings.append("Aucune répétition détectée : vérifiez que tout le corps est visible.")
    elif len(reps) < 2:
        warnings.append("Une seule répétition détectée : aucun défaut ne peut être retenu "
                        "(il faut au moins 2 répétitions concernées).")
    if view == "diagonal":
        warnings.append("Vue diagonale : les mesures de face (genoux, bassin) sont des "
                        "approximations déformées par la perspective ; les mesures de "
                        "profil ne sont pas évaluées.")
    return warnings


def model_path(config: dict) -> Path:
    """Chemin du modèle MediaPipe (téléchargé au premier usage)."""
    return ensure_model(ROOT / config["pose"]["model_path"], config["pose"]["model_url"])


def load_or_extract(video_path: Path, config: dict, cache_dir: Path | None = None,
                    progress=None) -> PoseSeries:
    """Extrait les points, avec cache disque optionnel.

    Pourquoi un cache ? MediaPipe prend ~9 s par vidéo de 30 s ; les scripts
    d'entraînement relisent les 150 vidéos : sans cache, chaque essai
    coûterait ~25 min. Le cache ne contient que des coordonnées de points
    (données dérivées), jamais d'images.
    """
    video_path = Path(video_path)
    cache = Path(cache_dir) / f"{video_path.stem}.npz" if cache_dir else None
    if cache and cache.exists():
        return PoseSeries.load(cache)
    raw = extract_pose(video_path, model_path(config), config["pose"]["max_people"], progress)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        raw.save(cache)
    return raw


def analyze_video(video_path: Path, view: str, config: dict,
                  cache_dir: Path | None = None, progress=None) -> Analysis:
    raw = load_or_extract(video_path, config, cache_dir, progress)
    return analyze_pose(raw, view, config)
