"""Vidéo annotée : squelette, angles et répétition en cours dessinés sur la vidéo.

Pourquoi ? Un chiffre seul ("tronc - tibia = 22°") est abstrait. Voir le
squelette superposé permet aussi de VÉRIFIER la mesure : si le squelette est
mal placé (mauvaise personne, jambe cachée), l'utilisateur le voit.

Encodage H.264 via imageio-ffmpeg : c'est le format lu par les navigateurs
(st.video). OpenCV n'écrit pas de H.264 de façon fiable sous Windows.
Texte sans accents : OpenCV ne sait pas dessiner les caractères accentués.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from src.analysis import Analysis
from src.pose import LM, SKELETON
from src.reps import rep_at_frame

GREEN, RED, WHITE, GREY = (80, 200, 80), (60, 60, 230), (255, 255, 255), (170, 170, 170)


def ascii_text(text: str) -> str:
    """Retire les accents (é -> e) pour cv2.putText."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _even_size(width: int, height: int, max_height: int) -> tuple[int, int]:
    """Taille de sortie : hauteur limitée et dimensions paires (exigence H.264)."""
    scale = min(1.0, max_height / height)
    return int(width * scale) // 2 * 2, int(height * scale) // 2 * 2


def draw_frame(frame: np.ndarray, analysis: Analysis, i: int, scale: float) -> np.ndarray:
    """Dessine le squelette et les informations sur UNE image (BGR)."""
    xy = analysis.pose.xy[i] * scale
    side = analysis.info["side"]
    for a, b in SKELETON:
        pa, pb = xy[LM[a]], xy[LM[b]]
        if np.isnan(pa).any() or np.isnan(pb).any():
            continue
        # Côté mesuré en vert, l'autre côté en gris
        color = GREEN if a.endswith(f"_{side}") or b.endswith(f"_{side}") else GREY
        cv2.line(frame, tuple(pa.astype(int)), tuple(pb.astype(int)), color, 2, cv2.LINE_AA)

    rep = rep_at_frame(analysis.reps, i)
    ind = analysis.indicators.iloc[i]
    lines = [(f"Rep {rep}/{len(analysis.reps)}" if rep else f"- /{len(analysis.reps)} reps", WHITE)]
    if analysis.view == "side":
        text = f"Genou {ind['knee_angle']:.0f} deg  Tronc {ind['trunk_lean']:.0f} deg"
        lines.append((text, WHITE))
    else:
        lines.append((f"Genoux/chevilles {ind['knee_ankle_ratio']:.2f}", WHITE))
    if rep:
        errors = analysis.rep_results[(analysis.rep_results["rep"] == rep)
                                      & (analysis.rep_results["error"] == True)]  # noqa: E712
        for label in errors["label"]:
            lines.append((ascii_text(label), RED))
        if len(errors):
            # Surlignage du moment fautif : cadre rouge pendant toute la répétition
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, max(4, h // 120))

    font_scale = max(0.4, frame.shape[0] / 1400)
    line_h = int(28 * font_scale / 0.6)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], line_h * len(lines) + 8), (0, 0, 0), -1)
    for k, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (8, line_h * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 1, cv2.LINE_AA)
    return frame


def render_annotated(video_path: Path, analysis: Analysis, out_path: Path,
                     max_height: int = 960, progress=None, hide_image: bool = False) -> Path:
    """Écrit la vidéo annotée (mp4 H.264) et renvoie son chemin.

    hide_image=True : squelette seul sur fond noir. Pourquoi ? Pour partager
    une analyse sans montrer le visage ni le décor de la personne filmée.
    """
    cap = cv2.VideoCapture(str(video_path))
    width, height = analysis.pose.width, analysis.pose.height
    out_w, out_h = _even_size(width, height, max_height)
    scale = out_w / width
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(out_path), (out_w, out_h), fps=analysis.pose.fps, codec="libx264",
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=2)
    writer.send(None)  # initialise le générateur
    n = analysis.pose.n_frames
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (out_w, out_h))
        if hide_image:
            frame = np.zeros_like(frame)
        frame = draw_frame(frame, analysis, i, scale)
        writer.send(np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if progress and i % 10 == 0:
            progress(i / n)
    writer.close()
    cap.release()
    return out_path
