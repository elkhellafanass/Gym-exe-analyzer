"""Tests des calculs d'angles et d'indicateurs."""

import numpy as np
import pytest

from src.angles import angle_3pts, angle_to_vertical, compute_indicators
from tests.synthetic import FOOT, squat_pose


def test_angle_3pts_cas_connus():
    b = np.array([[0.0, 0.0]])
    assert angle_3pts(np.array([[1.0, 0]]), b, np.array([[0.0, 1]]))[0] == pytest.approx(90)
    assert angle_3pts(np.array([[1.0, 0]]), b, np.array([[-1.0, 0]]))[0] == pytest.approx(180)
    assert angle_3pts(np.array([[1.0, 0]]), b, np.array([[1.0, 1]]))[0] == pytest.approx(45)


def test_angle_to_vertical():
    bottom = np.array([[0.0, 10.0]])
    assert angle_to_vertical(bottom, np.array([[0.0, 0.0]]))[0] == pytest.approx(0)
    assert angle_to_vertical(bottom, np.array([[10.0, 0.0]]))[0] == pytest.approx(45)
    # même inclinaison quel que soit le sens (personne tournée à gauche ou à droite)
    assert angle_to_vertical(bottom, np.array([[-10.0, 0.0]]))[0] == pytest.approx(45)


def test_indicateurs_squat_synthetique():
    s = np.array([0.0, 1.0])  # debout, puis point bas
    pose = squat_pose(s, tibia_max=30, thigh_max=100, trunk_max=40, neck_max=10,
                      heel_lift=0.25)
    df, info = compute_indicators(pose)
    debout, bas = df.iloc[0], df.iloc[1]
    assert debout["knee_angle"] == pytest.approx(180)
    # Genou : 180 - (inclinaison tibia + inclinaison cuisse) = 180 - 130
    assert bas["knee_angle"] == pytest.approx(50)
    assert bas["trunk_lean"] == pytest.approx(40)
    assert bas["tibia_lean"] == pytest.approx(30)
    assert bas["trunk_minus_tibia"] == pytest.approx(10)
    assert bas["neck_lean"] == pytest.approx(10)
    # Cuisse à 100° de la verticale : hanche sous le genou de cos(80°) cuisse
    assert bas["depth"] == pytest.approx(np.cos(np.radians(80)), abs=1e-6)
    assert debout["depth"] == pytest.approx(-1.0)
    # Talon monté de 0,25 longueur de pied (à la longueur médiane du pied près)
    assert bas["heel_height"] - debout["heel_height"] == pytest.approx(0.25, rel=0.05)
    assert info["foot_len_px"] == pytest.approx(FOOT, rel=0.05)


def test_ratio_genoux_chevilles_vue_de_face():
    """Genoux serrés (valgus) -> ratio < 1 ; genoux dans l'axe -> ratio 1."""
    pose = squat_pose(np.array([0.0, 0.0]))
    from src.pose import LM
    xy = pose.xy
    for i, knee_dx in enumerate((50.0, 25.0)):  # image 0 : dans l'axe ; image 1 : serrés
        xy[i, LM["ankle_l"]] = [400.0, 900.0]
        xy[i, LM["ankle_r"]] = [300.0, 900.0]
        xy[i, LM["knee_l"]] = [350.0 + knee_dx, 700.0]
        xy[i, LM["knee_r"]] = [350.0 - knee_dx, 700.0]
    df, _ = compute_indicators(pose)
    assert df["knee_ankle_ratio"].iloc[0] == pytest.approx(1.0)
    assert df["knee_ankle_ratio"].iloc[1] == pytest.approx(0.5)
