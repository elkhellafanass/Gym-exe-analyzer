"""Lecture du jeu externe B (Dai 2026) : points MediaPipe en CSV.

Problème : les coordonnées sont NORMALISÉES (x divisé par la largeur de la
vidéo, y par la hauteur), et la résolution de la vidéo d'origine n'est pas
documentée. Or un angle ne se calcule que si x et y ont la même échelle
(voir src/pose.py). Sans le rapport largeur/hauteur, une cuisse inclinée à
45° peut apparaître à 30° ou à 60°.

Ce module ne choisit PAS ce rapport en silence : il est toujours passé en
paramètre (hypothèse explicite). Il fournit aussi un moyen de le TESTER :
la longueur réelle d'un os ne change pas pendant le squat. Avec le bon
rapport, la longueur mesurée de la cuisse et du tibia reste à peu près
constante quand ils pivotent ; avec un mauvais rapport, elle varie selon
l'angle. On cherche le rapport qui rend ces longueurs les plus stables.
Limite : cela suppose un mouvement dans le plan de l'image (vue de profil).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.pose import LM, PoseSeries

ASPECT_GRID = np.round(np.arange(0.30, 4.01, 0.01), 2)


def read_normalized(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSV de Dai -> (xy normalisés (n, 33, 2), visibilité (n, 33), numéros d'image)."""
    df = pd.read_csv(path)
    xy = np.stack([np.c_[df[f"x_{i}"], df[f"y_{i}"]] for i in range(33)], axis=1)
    vis = np.stack([df[f"v_{i}"].to_numpy() for i in range(33)], axis=1)
    return xy.astype(float), vis.astype(float), df["Frame"].to_numpy()


def to_pose_series(xy_norm: np.ndarray, vis: np.ndarray, aspect: float,
                   fps: float, height: int = 1000) -> PoseSeries:
    """Coordonnées normalisées -> pixels, sous l'hypothèse aspect = largeur / hauteur.

    La hauteur (1000) est arbitraire : tous les indicateurs du pipeline sont
    divisés par une longueur du corps, seul le RAPPORT compte.
    """
    width = int(round(aspect * height))
    xy = xy_norm * np.array([width, height], dtype=float)
    return PoseSeries(xy, vis, float(fps), width, height)


def length_instability(xy_norm: np.ndarray, vis: np.ndarray, aspect: float,
                       min_visibility: float = 0.5) -> float:
    """Variation relative (écart-type / moyenne) des longueurs cuisse + tibia.

    Calculée sur la jambe la plus visible. Plus c'est petit, plus le rapport
    `aspect` est plausible.
    """
    side = "l" if vis[:, [LM["hip_l"], LM["knee_l"], LM["ankle_l"]]].mean() >= \
        vis[:, [LM["hip_r"], LM["knee_r"], LM["ankle_r"]]].mean() else "r"
    total = 0.0
    for a, b in (("hip", "knee"), ("knee", "ankle")):
        ia, ib = LM[f"{a}_{side}"], LM[f"{b}_{side}"]
        ok = (vis[:, ia] >= min_visibility) & (vis[:, ib] >= min_visibility)
        if ok.sum() < 5:
            return np.nan
        d = xy_norm[ok, ia] - xy_norm[ok, ib]
        length = np.hypot(d[:, 0] * aspect, d[:, 1])
        total += float(np.std(length) / np.mean(length))
    return total


def estimate_aspect(xy_norm: np.ndarray, vis: np.ndarray) -> float:
    """Rapport largeur/hauteur qui rend cuisse et tibia les plus stables (grille 0,30-4,00)."""
    scores = [length_instability(xy_norm, vis, r) for r in ASPECT_GRID]
    if np.all(np.isnan(scores)):
        return np.nan
    return float(ASPECT_GRID[int(np.nanargmin(scores))])
