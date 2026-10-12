"""Tests du nettoyage des trajectoires et de la vidéo annotée."""

import cv2
import numpy as np

from src.analysis import analyze_pose
from src.config import load_config
from src.pose import LM, PoseSeries, clean_pose
from src.video import ascii_text, render_annotated
from tests.synthetic import phase_signal, squat_pose

PARAMS = {"min_visibility": 0.5, "max_gap_frames": 3, "smooth_window": 7,
          "smooth_polyorder": 2}


def _series(signal, vis=None):
    n = len(signal)
    xy = np.zeros((n, 33, 2))
    xy[:, :, 1] = signal[:, None]
    vis = np.ones((n, 33)) if vis is None else vis
    return PoseSeries(xy, vis, 15.0, 100, 200)


def test_point_peu_visible_supprime_et_trou_court_comble():
    signal = np.linspace(0, 1, 40)
    vis = np.ones((40, 33))
    vis[10:12] = 0.1     # trou court (2 images) -> interpolé
    vis[20:30] = 0.1     # trou long (10 images) -> reste NaN
    out = clean_pose(_series(signal, vis), **PARAMS)
    y = out.xy[:, LM["hip_l"], 1]
    assert not np.isnan(y[10:12]).any()
    assert np.allclose(y[10:12], signal[10:12], atol=1e-6)
    assert np.isnan(y[20:30]).all()


def test_lissage_reduit_le_bruit():
    rng = np.random.default_rng(1)
    clean = np.sin(np.linspace(0, 4 * np.pi, 120))
    noisy = clean + rng.normal(0, 0.05, 120)
    out = clean_pose(_series(noisy), **PARAMS).xy[:, 0, 1]
    assert np.std(out - clean) < np.std(noisy - clean)


def test_ascii_text():
    assert ascii_text("Talons décollés") == "Talons decolles"


def test_video_annotee(tmp_path):
    """Vidéo synthétique (fond noir) -> vidéo annotée lisible."""
    pose = squat_pose(phase_signal(n_reps=2))
    src = tmp_path / "in.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 15, (810, 1440))
    for _ in range(pose.n_frames):
        writer.write(np.zeros((1440, 810, 3), np.uint8))
    writer.release()
    analysis = analyze_pose(pose, "side", load_config())
    out = render_annotated(src, analysis, tmp_path / "out.mp4", max_height=480)
    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    assert ok and frame.shape[0] == 480
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == pose.n_frames
