"""Tests de la segmentation des répétitions."""

import numpy as np
import pytest

from src.reps import rep_at_frame, segment_reps
from tests.synthetic import phase_signal

PARAMS = {"min_travel": 0.3, "min_interval_s": 0.8, "edge_fraction": 0.1}


def test_compte_et_tempo():
    fps = 15
    hip_y = phase_signal(n_reps=5, fps=fps, descent_s=1.0, ascent_s=0.5)
    reps = segment_reps(hip_y, fps, **PARAMS)
    assert len(reps) == 5
    # À 15 fps, une durée est connue à ±1 image (0,067 s) près, plus le seuil de 10 %
    assert reps["descent_s"].median() == pytest.approx(0.8, abs=0.15)
    assert reps["ascent_s"].median() == pytest.approx(0.4, abs=0.15)
    assert (reps["descent_s"] > reps["ascent_s"]).all()


def test_robuste_au_bruit_et_aux_trous():
    rng = np.random.default_rng(0)
    hip_y = phase_signal(n_reps=6) + rng.normal(0, 0.02, size=len(phase_signal(n_reps=6)))
    hip_y[20:23] = np.nan  # détection perdue sur 3 images
    assert len(segment_reps(hip_y, 15, **PARAMS)) == 6


def test_petites_oscillations_ignorees():
    """Une oscillation plus petite que min_travel n'est pas une répétition."""
    t = np.arange(150) / 15
    hip_y = 0.1 * np.sin(2 * np.pi * t)
    assert len(segment_reps(hip_y, 15, **PARAMS)) == 0


def test_signal_vide():
    assert segment_reps(np.full(30, np.nan), 15, **PARAMS).empty


def test_rep_at_frame():
    reps = segment_reps(phase_signal(n_reps=2), 15, **PARAMS)
    first = reps.iloc[0]
    assert rep_at_frame(reps, int(first["bottom"])) == 1
    assert rep_at_frame(reps, 0) is None
