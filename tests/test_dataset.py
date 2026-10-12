"""Tests de l'indexation du dataset (sans dépendre du vrai dataset)."""

import cv2
import numpy as np

from src.config import load_config
from src.dataset import build_index, missing_combinations, parse_filename

PATTERN = load_config()["dataset"]["filename_pattern"]


def test_parse_filename_valide():
    info = parse_filename("subject_007_squat_bad_diagonal.mp4", PATTERN)
    assert info == {"subject": "007", "exercise": "squat", "quality": "bad", "view": "diagonal"}


def test_parse_filename_invalide():
    assert parse_filename("subject_7_squat_bad_top.mp4", PATTERN) is None
    assert parse_filename("notes.txt", PATTERN) is None


def _write_video(path, n_frames=30, fps=15, size=(64, 112)):
    """Crée une petite vidéo synthétique (images noires) pour le test."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(n_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


def test_build_index_et_incoherence_dossier(tmp_path):
    (tmp_path / "good").mkdir()
    (tmp_path / "bad").mkdir()
    _write_video(tmp_path / "good" / "subject_001_squat_good_front.mp4")
    _write_video(tmp_path / "bad" / "subject_001_squat_bad_front.mp4")
    # Fichier "good" rangé dans bad/ : doit être ignoré, pas indexé.
    _write_video(tmp_path / "bad" / "subject_002_squat_good_side.mp4")

    index, skipped = build_index(tmp_path, PATTERN)

    assert len(index) == 2
    assert len(skipped) == 1
    assert set(index["label"]) == {0, 1}
    assert index.loc[index["quality"] == "bad", "label"].item() == 1
    row = index.iloc[0]
    assert row["fps"] == 15
    assert (row["width"], row["height"]) == (64, 112)
    assert row["n_frames"] == 30
    assert abs(row["duration_s"] - 2.0) < 0.01


def test_missing_combinations(tmp_path):
    (tmp_path / "good").mkdir()
    (tmp_path / "bad").mkdir()
    _write_video(tmp_path / "good" / "subject_001_squat_good_front.mp4", n_frames=2)
    index, _ = build_index(tmp_path, PATTERN)
    missing = missing_combinations(index)
    # 1 sujet x 2 qualités x 3 vues = 6 attendues, 1 présente -> 5 manquantes
    assert len(missing) == 5
    assert ("001", "bad", "front") in missing
