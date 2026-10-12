"""Étape 3 du pipeline : découper la vidéo en répétitions.

Idée : pendant un squat, la hanche descend puis remonte. Sur la courbe de
hauteur de hanche (y vers le bas), chaque répétition fait donc un "pic" :
le point bas du squat. On détecte ces pics avec scipy.signal.find_peaks.

Paramètres (config.yaml, section reps) : ce sont des paramètres TECHNIQUES de
détection, pas des critères de posture :
  - min_travel : descente minimale de la hanche (en longueurs de cuisse) pour
    compter une répétition ; évite de compter une petite oscillation ;
  - min_interval_s : temps minimal entre deux points bas ;
  - edge_fraction : pour trouver le début et la fin du mouvement (voir plus bas).

Limite : à 15 images/s, une durée est connue à ±1 image près (±0,067 s).
Les durées de descente/montée (~0,5 à 1,5 s) ont donc une incertitude
relative de l'ordre de 5 à 15 %.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def segment_reps(hip_y: np.ndarray, fps: float, min_travel: float,
                 min_interval_s: float, edge_fraction: float) -> pd.DataFrame:
    """Renvoie une ligne par répétition : start, bottom, end (indices d'image)
    et durées de descente / remontée en secondes.

    Le début (start) n'est pas simplement le point le plus haut entre deux
    répétitions : si la personne fait une pause debout, la "descente" durerait
    artificiellement longtemps. On prend donc la dernière image, avant le point
    bas, où la hanche est encore proche du haut (à moins de edge_fraction de
    l'amplitude) ; même logique pour la fin.
    """
    columns = ["rep", "start", "bottom", "end", "descent_s", "ascent_s", "travel"]
    signal = pd.Series(hip_y).interpolate(limit_direction="both").to_numpy()
    if np.all(np.isnan(signal)) or len(signal) < 3:
        return pd.DataFrame(columns=columns)

    bottoms, _ = find_peaks(signal, prominence=min_travel,
                            distance=max(1, int(min_interval_s * fps)))
    rows = []
    for k, b in enumerate(bottoms):
        left = bottoms[k - 1] if k > 0 else 0
        right = bottoms[k + 1] if k + 1 < len(bottoms) else len(signal) - 1
        top_before = left + int(np.argmin(signal[left:b + 1]))
        top_after = b + int(np.argmin(signal[b:right + 1]))
        travel = signal[b] - max(signal[top_before], signal[top_after])
        level_start = signal[top_before] + edge_fraction * (signal[b] - signal[top_before])
        level_end = signal[top_after] + edge_fraction * (signal[b] - signal[top_after])
        # dernière image "encore en haut" avant le point bas
        start = top_before + int(np.flatnonzero(signal[top_before:b + 1] <= level_start)[-1])
        # première image "revenue en haut" après le point bas
        end = b + int(np.flatnonzero(signal[b:top_after + 1] <= level_end)[0])
        rows.append({
            "rep": k + 1, "start": start, "bottom": int(b), "end": end,
            "descent_s": (b - start) / fps, "ascent_s": (end - b) / fps,
            "travel": float(travel),
        })
    return pd.DataFrame(rows, columns=columns)


def rep_at_frame(reps: pd.DataFrame, frame: int) -> int | None:
    """Numéro de la répétition en cours à une image donnée (None si aucune)."""
    hit = reps[(reps["start"] <= frame) & (reps["end"] >= frame)]
    return int(hit["rep"].iloc[0]) if len(hit) else None
