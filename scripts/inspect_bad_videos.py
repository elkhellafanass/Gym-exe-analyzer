"""PHASE 0 : inspection visuelle, que signifie vraiment le label "bad" ?

Le dataset ne documente pas le type d'erreur des vidéos "bad". Avant d'écrire
des règles, on regarde donc les vidéos. Méthode :
  1. pour chaque vidéo, MediaPipe repère la hanche sur chaque image ;
  2. le point le plus bas de chaque répétition = maximum local de la hauteur
     de hanche (en image, y augmente vers le bas) ;
  3. on garde l'image du point bas de la répétition MÉDIANE (représentative,
     ni la meilleure ni la pire) ;
  4. on assemble des planches : même sujet, même vue, good (haut) / bad (bas).
On calcule aussi quelques indicateurs bruts au point bas, À TITRE EXPLORATOIRE
(le pipeline propre, testé et lissé sera fait en phase A).

Sorties (jamais versionnées, personnes identifiables) :
  docs/inspection/<vue>_sujets_<a>-<b>.png
  outputs/inspection_stats.csv

Usage : python scripts/inspect_bad_videos.py
"""

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from scipy.signal import find_peaks, savgol_filter

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "pose_landmarker_full.task"
OUT_IMG = ROOT / "docs" / "inspection"
OUT_CSV = ROOT / "outputs" / "inspection_stats.csv"
THUMB_H = 360  # hauteur des vignettes dans les planches

# Indices MediaPipe (33 points) utilisés ici
SH_L, SH_R, HIP_L, HIP_R, KNEE_L, KNEE_R, ANK_L, ANK_R = 11, 12, 23, 24, 25, 26, 27, 28


def angle(a, b, c):
    """Angle ABC en degrés (sommet en B), en 2D image."""
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def run_pose(path):
    """Renvoie landmarks[n,33,3] (x, y en pixels + visibilité), une ligne par image."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=RunningMode.VIDEO,
    )
    lms = []
    with PoseLandmarker.create_from_options(options) as landmarker:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(image, int(i * 1000 / fps))
            if res.pose_landmarks:
                p = res.pose_landmarks[0]
                lms.append([[q.x * w, q.y * h, q.visibility] for q in p])
            else:
                lms.append(np.full((33, 3), np.nan).tolist())
            i += 1
    cap.release()
    return np.array(lms)


def read_frame(path, idx):
    """Relit une seule image (on ne garde pas toute la vidéo en mémoire)."""
    cap = cv2.VideoCapture(str(path))
    for _ in range(idx + 1):
        ok, frame = cap.read()
    cap.release()
    return frame


def bottom_frames(lms, fps):
    """Indices des points bas (un par répétition) d'après la hauteur de hanche."""
    hip_y = pd.Series(np.nanmean(lms[:, [HIP_L, HIP_R], 1], axis=1)).interpolate().bfill().ffill()
    smooth = savgol_filter(hip_y.to_numpy(), 7, 2)
    amplitude = np.percentile(smooth, 95) - np.percentile(smooth, 5)
    peaks, _ = find_peaks(smooth, prominence=0.4 * amplitude, distance=int(fps))
    return peaks


def stats_at(lms, idx, view):
    """Indicateurs bruts au point bas (exploratoires, non lissés)."""
    p = lms[idx]
    # Côté le plus visible (en vue de profil, l'autre jambe est cachée)
    side = (HIP_L, KNEE_L, ANK_L, SH_L) if p[KNEE_L, 2] >= p[KNEE_R, 2] else \
        (HIP_R, KNEE_R, ANK_R, SH_R)
    hip, knee, ank, sh = (p[j, :2] for j in side)
    trunk = sh - hip
    thigh = np.linalg.norm(hip - knee)
    out = {
        "knee_angle": angle(hip, knee, ank),
        "hip_angle": angle(sh, hip, knee),
        "trunk_lean_deg": float(np.degrees(np.arctan2(abs(trunk[0]), -trunk[1]))),
        # > 0 : hanche plus basse que le genou (profondeur "sous la parallèle")
        "hip_below_knee_rel": float((hip[1] - knee[1]) / thigh),
    }
    if view in ("front", "diagonal"):
        knee_w = abs(p[KNEE_L, 0] - p[KNEE_R, 0])
        ank_w = abs(p[ANK_L, 0] - p[ANK_R, 0])
        out["knee_ankle_width_ratio"] = float(knee_w / (ank_w + 1e-9))
    return out


def thumb(frame, label):
    h, w = frame.shape[:2]
    t = cv2.resize(frame, (int(w * THUMB_H / h), THUMB_H))
    cv2.rectangle(t, (0, 0), (t.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(t, label, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return t


def main():
    index = pd.read_csv(ROOT / "data" / "index.csv", dtype={"subject": str})
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows, thumbs = [], {}
    for n, r in enumerate(index.itertuples(), 1):
        print(f"[{n}/{len(index)}] {r.filename}", flush=True)
        lms = run_pose(r.path)
        peaks = bottom_frames(lms, r.fps)
        detected = float(np.mean(~np.isnan(lms[:, HIP_L, 0])))
        for k, idx in enumerate(peaks):
            rows.append({"subject": r.subject, "quality": r.quality, "view": r.view,
                         "rep": k, "frame": int(idx), "detection_rate": detected,
                         **stats_at(lms, idx, r.view)})
        if len(peaks):
            mid = peaks[np.argsort(lms[peaks, HIP_L, 1])[len(peaks) // 2]]
            thumbs[(r.subject, r.quality, r.view)] = thumb(
                read_frame(r.path, mid), f"S{r.subject} {r.quality} reps={len(peaks)}")
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    # Planches : 5 sujets par planche, ligne du haut = good, ligne du bas = bad
    subjects = sorted(index["subject"].unique())
    for view in ("side", "front", "diagonal"):
        for g in range(0, len(subjects), 5):
            group = subjects[g:g + 5]
            lines = []
            for q in ("good", "bad"):
                cells = [thumbs.get((s, q, view), np.zeros((THUMB_H, 200, 3), np.uint8))
                         for s in group]
                lines.append(np.hstack(cells))
            width = max(line.shape[1] for line in lines)
            lines = [cv2.copyMakeBorder(line, 0, 0, 0, width - line.shape[1],
                                        cv2.BORDER_CONSTANT) for line in lines]
            cv2.imwrite(str(OUT_IMG / f"{view}_sujets_{group[0]}-{group[-1]}.png"),
                        np.vstack(lines))
    print(f"Planches -> {OUT_IMG}\nIndicateurs -> {OUT_CSV}")


if __name__ == "__main__":
    sys.exit(main())
