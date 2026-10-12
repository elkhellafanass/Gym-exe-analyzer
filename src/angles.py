"""Étape 2 du pipeline : points du squelette -> indicateurs biomécaniques.

Tout est calculé en 2D, dans le plan de l'image. Conséquence importante :
un angle n'a de sens que si le mouvement se fait dans le plan de la caméra.
  - Vue de PROFIL (side) : plan sagittal -> angles genou/hanche, inclinaison
    du tronc et du tibia, profondeur, talons, tête.
  - Vue de FACE (front) : plan frontal -> genoux vers l'intérieur (valgus),
    bascule du bassin. La vue diagonale est une approximation (perspective).
Un angle de genou mesuré de face n'est PAS l'angle réel du genou (c'est une
projection). C'est pour cela que chaque règle déclare ses vues valides.

Convention image : y augmente vers le BAS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pose import LM, PoseSeries


def angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC en degrés (sommet B), pour des tableaux (n, 2).

    180° = segments alignés (jambe tendue), plus petit = articulation fléchie.
    """
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, axis=-1) / (
        np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def angle_to_vertical(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Inclinaison d'un segment (bas -> haut) par rapport à la verticale, en degrés.

    0° = segment vertical ; 90° = horizontal. Valeur absolue : on mesure
    "combien c'est penché", quel que soit le sens où regarde la personne.
    """
    v = top - bottom
    return np.degrees(np.arctan2(np.abs(v[..., 0]), -v[..., 1]))


def choose_side(series: PoseSeries) -> str:
    """Choisit la jambe la mieux vue ("l" ou "r").

    En vue de profil, la jambe éloignée de la caméra est cachée : ses points
    sont estimés, moins fiables. On garde le côté à la meilleure visibilité
    moyenne (hanche + genou + cheville).
    """
    def mean_vis(s):
        idx = [LM[f"{j}_{s}"] for j in ("hip", "knee", "ankle")]
        return float(np.mean(series.vis[:, idx]))
    return "l" if mean_vis("l") >= mean_vis("r") else "r"


def _median_length(a: np.ndarray, b: np.ndarray) -> float:
    """Longueur médiane d'un segment sur la vidéo (sert d'unité de mesure).

    Pourquoi normaliser ? Une distance en pixels dépend de la distance à la
    caméra et de la résolution (4 résolutions dans le dataset). Diviser par la
    longueur de la cuisse ou du pied rend les mesures comparables.
    """
    return float(np.nanmedian(np.linalg.norm(a - b, axis=-1)))


def compute_indicators(series: PoseSeries) -> tuple[pd.DataFrame, dict]:
    """Calcule tous les indicateurs image par image.

    Renvoie (tableau par image, informations : côté utilisé, unités).
    Tous les indicateurs sont calculés ; c'est la règle (rules.py) qui décide
    s'ils sont interprétables dans la vue filmée.
    """
    s = choose_side(series)
    p = series.point
    hip, knee, ankle = p(f"hip_{s}"), p(f"knee_{s}"), p(f"ankle_{s}")
    shoulder, ear = p(f"shoulder_{s}"), p(f"ear_{s}")
    heel, foot = p(f"heel_{s}"), p(f"foot_{s}")

    thigh_len = _median_length(hip, knee)
    foot_len = _median_length(heel, foot)

    df = pd.DataFrame({
        "time_s": np.arange(series.n_frames) / series.fps,
        # --- plan sagittal (vue de profil) ---
        "knee_angle": angle_3pts(hip, knee, ankle),
        "hip_angle": angle_3pts(shoulder, hip, knee),
        "trunk_lean": angle_to_vertical(hip, shoulder),
        "tibia_lean": angle_to_vertical(ankle, knee),
        # Profondeur : > 0 quand la hanche est plus basse que le genou
        "depth": (hip[:, 1] - knee[:, 1]) / thigh_len,
        # Inclinaison du cou (épaule -> oreille) par rapport à la verticale
        "neck_lean": angle_to_vertical(shoulder, ear),
        # Hauteur verticale de la hanche (pour la segmentation des répétitions)
        "hip_y": (p("hip_l")[:, 1] + p("hip_r")[:, 1]) / 2 / thigh_len,
    })
    # Tronc / tibia : Myer et al. (2014) demandent un tronc parallèle au tibia.
    # > 0 : le tronc est plus penché que le tibia.
    df["trunk_minus_tibia"] = df["trunk_lean"] - df["tibia_lean"]

    # Talons : montée du talon par rapport à sa position "au sol". Le sol est
    # estimé par le 90e centile de la hauteur du talon (le talon est le plus
    # bas quand il est posé). Hypothèse : caméra fixe. Unité : longueur de pied.
    heel_floor = np.nanpercentile(heel[:, 1], 90)
    df["heel_rise"] = (heel_floor - heel[:, 1]) / foot_len

    # --- plan frontal (vue de face) ---
    knee_w = np.abs(p("knee_l")[:, 0] - p("knee_r")[:, 0])
    ankle_w = np.abs(p("ankle_l")[:, 0] - p("ankle_r")[:, 0])
    # Ratio de séparation genoux / chevilles (Mizner et al., 2012) :
    # < 1 = genoux plus serrés que les chevilles.
    # (0/0 en vue de profil, jambes superposées -> NaN : "non mesurable", voulu)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["knee_ankle_ratio"] = knee_w / ankle_w
    # Bascule du bassin : inclinaison de la ligne des hanches / horizontale
    hl, hr = p("hip_l"), p("hip_r")
    df["hip_tilt"] = np.degrees(np.arctan2(np.abs(hl[:, 1] - hr[:, 1]),
                                           np.abs(hl[:, 0] - hr[:, 0])))
    # Asymétrie gauche/droite de l'angle du genou (projection 2D en vue de face :
    # utile pour comparer les deux jambes, pas comme angle absolu)
    knee_l = angle_3pts(p("hip_l"), p("knee_l"), p("ankle_l"))
    knee_r = angle_3pts(p("hip_r"), p("knee_r"), p("ankle_r"))
    df["knee_asymmetry"] = np.abs(knee_l - knee_r)

    info = {"side": s, "thigh_len_px": thigh_len, "foot_len_px": foot_len}
    return df, info
