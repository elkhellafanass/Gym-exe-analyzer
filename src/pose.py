"""Étape 1 du pipeline : la vidéo devient une série de points du squelette.

Pourquoi MediaPipe Pose ? Il donne 33 points du corps (épaules, hanches,
genoux, chevilles, talons, pointes de pieds...) sur chaque image, tourne sur un
simple processeur et est conçu pour le fitness (voir sa fiche modèle,
docs/SOURCES.md). On utilise l'API "Tasks" (PoseLandmarker), la seule
disponible depuis MediaPipe 1.0.

Ce module fait trois choses :
  1. extraire les points image par image (coordonnées en PIXELS : les angles
     doivent être calculés dans un repère où x et y ont la même échelle,
     ce qui n'est pas le cas des coordonnées normalisées 0-1 d'une vidéo
     portrait) ;
  2. gérer les images sans détection et les points peu visibles (-> NaN) ;
  3. lisser les trajectoires, car la détection "tremble" d'une image à l'autre.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from scipy.signal import savgol_filter

# Indices MediaPipe des points utilisés (fiche modèle BlazePose GHUM 3D)
LM = {
    "nose": 0, "ear_l": 7, "ear_r": 8,
    "shoulder_l": 11, "shoulder_r": 12, "hip_l": 23, "hip_r": 24,
    "knee_l": 25, "knee_r": 26, "ankle_l": 27, "ankle_r": 28,
    "heel_l": 29, "heel_r": 30, "foot_l": 31, "foot_r": 32,
    "elbow_l": 13, "elbow_r": 14, "wrist_l": 15, "wrist_r": 16,
}

# Segments dessinés sur la vidéo annotée
SKELETON = [
    ("shoulder_l", "shoulder_r"), ("hip_l", "hip_r"),
    ("shoulder_l", "hip_l"), ("shoulder_r", "hip_r"),
    ("shoulder_l", "elbow_l"), ("elbow_l", "wrist_l"),
    ("shoulder_r", "elbow_r"), ("elbow_r", "wrist_r"),
    ("hip_l", "knee_l"), ("knee_l", "ankle_l"), ("ankle_l", "heel_l"),
    ("heel_l", "foot_l"), ("ankle_l", "foot_l"),
    ("hip_r", "knee_r"), ("knee_r", "ankle_r"), ("ankle_r", "heel_r"),
    ("heel_r", "foot_r"), ("ankle_r", "foot_r"),
]


@dataclass
class PoseSeries:
    """Points du squelette sur toute la vidéo.

    xy  : (n_images, 33, 2) en pixels, NaN si absent
    vis : (n_images, 33) visibilité 0-1 donnée par MediaPipe
    """

    xy: np.ndarray
    vis: np.ndarray
    fps: float
    width: int
    height: int

    @property
    def n_frames(self) -> int:
        return len(self.xy)

    def detected(self) -> np.ndarray:
        """Vrai pour chaque image où une personne a été détectée."""
        return ~np.isnan(self.xy[:, LM["hip_l"], 0])

    def point(self, name: str) -> np.ndarray:
        """Trajectoire (n_images, 2) d'un point, par son nom."""
        return self.xy[:, LM[name], :]

    def save(self, path: Path) -> None:
        np.savez_compressed(path, xy=self.xy, vis=self.vis,
                            meta=np.array([self.fps, self.width, self.height]))

    @classmethod
    def load(cls, path: Path) -> PoseSeries:
        d = np.load(path)
        fps, w, h = d["meta"]
        return cls(d["xy"], d["vis"], float(fps), int(w), int(h))


def ensure_model(path: Path, url: str) -> Path:
    """Télécharge le modèle MediaPipe s'il est absent (ex. Streamlit Cloud)."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def _pick_person(poses):
    """Choisit LA personne analysée quand MediaPipe en voit plusieurs.

    Pourquoi ? Dans les salles de sport filmées, d'autres personnes passent en
    arrière-plan. La fiche du modèle classe "plusieurs personnes" hors du
    périmètre prévu. Règle simple : on garde le squelette le plus grand en
    hauteur, c'est-à-dire la personne la plus proche de la caméra.
    """
    def body_height(p):
        ys = [q.y for q in p]
        return max(ys) - min(ys)
    return max(poses, key=body_height)


def extract_pose(video_path: Path, model_path: Path, max_people: int = 3,
                 progress=None) -> PoseSeries:
    """Lance MediaPipe sur chaque image de la vidéo.

    Mode VIDEO : MediaPipe suit la personne d'une image à l'autre (plus stable
    et plus rapide que de traiter chaque image indépendamment).
    `progress` : fonction optionnelle appelée avec une fraction 0-1 (barre de
    progression dans l'app).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo : {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=max_people,
    )
    xy, vis = [], []
    width = height = 0
    with PoseLandmarker.create_from_options(options) as landmarker:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(image, int(round(i * 1000 / fps)))
            if result.pose_landmarks:
                p = _pick_person(result.pose_landmarks)
                xy.append([[q.x * width, q.y * height] for q in p])
                vis.append([q.visibility for q in p])
            else:
                xy.append(np.full((33, 2), np.nan))
                vis.append(np.zeros(33))
            i += 1
            if progress and i % 10 == 0:
                progress(min(i / total, 1.0))
    cap.release()
    if not xy:
        raise ValueError("La vidéo ne contient aucune image lisible.")
    return PoseSeries(np.array(xy, dtype=float), np.array(vis, dtype=float),
                      float(fps), width, height)


def clean_pose(series: PoseSeries, min_visibility: float, max_gap_frames: int,
               smooth_window: int, smooth_polyorder: int) -> PoseSeries:
    """Nettoie les trajectoires avant de calculer des angles.

    1. Point peu visible (visibilité < seuil) -> NaN : MediaPipe "devine"
       la position des points cachés ; on préfère ne pas mesurer que mesurer
       une supposition.
    2. Trous courts (<= max_gap_frames) comblés par interpolation linéaire :
       une détection manquée sur 1-2 images ne doit pas casser une répétition.
       Les trous plus longs restent NaN (on ne fabrique pas de données).
    3. Lissage Savitzky-Golay : réduit le tremblement image par image tout en
       conservant mieux les pics (point bas du squat) qu'une moyenne glissante.
    """
    xy = series.xy.copy()
    xy[series.vis < min_visibility] = np.nan
    n, n_points, _ = xy.shape
    flat = pd.DataFrame(xy.reshape(n, -1))
    filled = flat.interpolate(limit_area="inside")
    # pandas `limit` remplirait le DÉBUT d'un trou long : on remet NaN partout
    # où le trou dépasse max_gap_frames.
    gap_len = flat.isna().apply(_run_lengths)
    out = filled.mask(gap_len > max_gap_frames).to_numpy(copy=True)
    if n >= smooth_window:
        for col in range(out.shape[1]):
            out[:, col] = _smooth_with_nans(out[:, col], smooth_window, smooth_polyorder)
    return PoseSeries(out.reshape(n, n_points, 2), series.vis, series.fps,
                      series.width, series.height)


def _run_lengths(is_nan: pd.Series) -> pd.Series:
    """Pour chaque image, longueur du trou (suite de NaN) qui la contient ; 0 sinon."""
    run_id = (is_nan != is_nan.shift()).cumsum()
    return is_nan.groupby(run_id).transform("sum").where(is_nan, 0)


def _smooth_with_nans(signal: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Lisse chaque morceau continu (sans NaN) séparément."""
    out = signal.copy()
    valid = ~np.isnan(signal)
    # Découpe en segments continus de valeurs valides
    edges = np.flatnonzero(np.diff(np.r_[0, valid.astype(int), 0]))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        if stop - start >= window:
            out[start:stop] = savgol_filter(signal[start:stop], window, polyorder)
    return out
