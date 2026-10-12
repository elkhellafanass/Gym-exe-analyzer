"""Tests des règles et du pipeline complet sur squats synthétiques."""

import numpy as np
import pandas as pd

from src.analysis import analyze_pose
from src.config import load_config
from src.rules import evaluate_reps, summarize_video
from tests.synthetic import phase_signal, squat_pose

CONFIG = load_config()
RULES = CONFIG["exercises"]["squat"]["rules"]


def _verdicts(analysis):
    return dict(zip(analysis.summary["rule"], analysis.summary["video_error"], strict=True))


def test_aucun_seuil_sans_source_ni_statut():
    """Garde-fou : chaque règle déclare une source et un statut connu."""
    allowed = {"sourcé", "critère sourcé, seuil à valider par un expert",
               "à valider par un expert"}
    for rule in RULES.values():
        assert rule["source"]
        assert rule["status"] in allowed


def test_squat_correct_aucune_erreur():
    pose = squat_pose(phase_signal(n_reps=5), thigh_max=100, trunk_max=35, tibia_max=30)
    analysis = analyze_pose(pose, "side", CONFIG)
    assert len(analysis.reps) == 5
    assert not any(_verdicts(analysis).values())


def test_squat_partiel_et_talons():
    pose = squat_pose(phase_signal(n_reps=5), thigh_max=70, heel_lift=0.4)
    verdicts = _verdicts(analyze_pose(pose, "side", CONFIG))
    assert verdicts["depth"]
    assert verdicts["heel_rise"]
    assert not verdicts["trunk_vs_tibia"]


def test_talon_qui_descend_au_point_bas_nest_pas_une_erreur():
    """Cas observé sur données réelles : le point talon descend un peu au point
    bas. La mesure relative au début de répétition ne doit pas le signaler."""
    pose = squat_pose(phase_signal(n_reps=5), heel_lift=-0.3)
    assert not _verdicts(analyze_pose(pose, "side", CONFIG))["heel_rise"]


def test_tronc_penche():
    pose = squat_pose(phase_signal(n_reps=5), trunk_max=60, tibia_max=25)
    assert _verdicts(analyze_pose(pose, "side", CONFIG))["trunk_vs_tibia"]


def test_regles_filtrees_par_vue():
    pose = squat_pose(phase_signal(n_reps=3))
    side = analyze_pose(pose, "side", CONFIG).summary
    front = analyze_pose(pose, "front", CONFIG).summary
    assert "knee_valgus" not in set(side["rule"])
    assert "depth" not in set(front["rule"])
    diag = analyze_pose(pose, "diagonal", CONFIG)
    assert set(diag.summary["reliability"]) == {"reduced"}
    assert any("diagonale" in w for w in diag.warnings)


def _fake(values, rule="knee_valgus"):
    n = len(values)
    indicators = pd.DataFrame({"hip_y": np.r_[0, np.ones(n - 2), 0],
                               RULES[rule]["indicator"]: values})
    reps = pd.DataFrame([{"rep": 1, "start": 0, "bottom": n // 2, "end": n - 1}])
    return indicators, reps


def test_mesure_impossible_donne_nan_pas_de_verdict():
    indicators, reps = _fake(np.full(6, np.nan))
    res = evaluate_reps(indicators, reps, "front", {"knee_valgus": RULES["knee_valgus"]})
    assert np.isnan(res["value"].iloc[0])
    assert pd.isna(res["error"].iloc[0])


def test_defaut_retenu_seulement_sur_deux_repetitions():
    rep_results = pd.DataFrame({
        "rep": [1, 2, 3], "rule": "depth", "label": "x", "value": 0.0,
        "threshold": 0.0, "op": "<", "error": [True, False, False],
        "reliability": "valid", "status": "sourcé"})
    summary = summarize_video(rep_results, RULES, min_reps_for_error=2)
    assert not summary["video_error"].iloc[0]
    rep_results.loc[1, "error"] = True
    assert summarize_video(rep_results, RULES, 2)["video_error"].iloc[0]
