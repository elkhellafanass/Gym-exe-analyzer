"""Générateur de squats synthétiques (vue de profil) pour les tests.

On choisit les angles (inclinaison du tibia, de la cuisse, du tronc...) et on
calcule la position des points par cinématique directe. On connaît donc la
"vérité" et on peut vérifier que le pipeline la retrouve.
"""

import numpy as np

from src.pose import LM, PoseSeries

SHANK, THIGH, TRUNK, NECK, FOOT = 200.0, 200.0, 250.0, 40.0, 80.0


def phase_signal(n_reps=5, fps=15, descent_s=1.0, ascent_s=0.5, pause_s=0.6):
    """Phase 0 (debout) -> 1 (point bas), répétée. Descente plus lente que la remontée."""
    parts = [np.zeros(int(pause_s * fps))]
    for _ in range(n_reps):
        parts.append((1 - np.cos(np.linspace(0, np.pi, int(descent_s * fps)))) / 2)
        parts.append((1 + np.cos(np.linspace(0, np.pi, int(ascent_s * fps)))) / 2)
        parts.append(np.zeros(int(pause_s * fps)))
    return np.concatenate(parts)


def squat_pose(s, fps=15, tibia_max=30, thigh_max=100, trunk_max=40, neck_max=10,
               heel_lift=0.0):
    """PoseSeries d'un squat de profil (personne tournée vers la droite).

    s : phase (n,) entre 0 et 1. thigh_max : angle cuisse/verticale au point bas
    (> 90° = hanche sous le genou). heel_lift : montée du talon au point bas,
    en longueurs de pied.
    """
    s = np.asarray(s, dtype=float)
    n = len(s)
    rad = np.radians
    ankle = np.tile([300.0, 900.0], (n, 1))
    t, th, ph, ps = (rad(s * a) for a in (tibia_max, thigh_max, trunk_max, neck_max))
    knee = ankle + SHANK * np.c_[np.sin(t), -np.cos(t)]
    hip = knee + THIGH * np.c_[-np.sin(th), -np.cos(th)]
    shoulder = hip + TRUNK * np.c_[np.sin(ph), -np.cos(ph)]
    ear = shoulder + NECK * np.c_[np.sin(ps), -np.cos(ps)]
    heel = ankle + np.c_[np.full(n, -20.0), 20.0 - s * heel_lift * FOOT]
    foot = ankle + np.c_[np.full(n, 60.0), np.full(n, 20.0)]

    xy = np.full((n, 33, 2), np.nan)
    for side in ("l", "r"):
        for name, arr in [("ankle", ankle), ("knee", knee), ("hip", hip),
                          ("shoulder", shoulder), ("ear", ear), ("heel", heel),
                          ("foot", foot)]:
            xy[:, LM[f"{name}_{side}"]] = arr
    vis = np.where(np.isnan(xy[..., 0]), 0.0, 1.0)
    return PoseSeries(xy, vis, float(fps), 810, 1440)
