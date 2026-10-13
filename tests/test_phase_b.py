"""Tests de la phase B : anti-fuite LOSO, métriques, bootstrap, verdict."""

import numpy as np
import pandas as pd
import pytest

from src.classifier import loso_predict, make_models, verdict
from src.config import load_config
from src.features import feature_names
from src.metrics import binary_metrics, bootstrap_by_subject, kappa

CONFIG = load_config()
RULES = CONFIG["exercises"]["squat"]["rules"]


def _table(n_subjects=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"subject": f"{s:03d}", "label": q, "x": q + rng.normal(0, 0.3)}
            for s in range(n_subjects) for q in (0, 1)]
    return pd.DataFrame(rows)


class _Spy:
    """Faux modèle qui mémorise les sujets vus à l'entraînement."""
    seen: list = []

    def fit(self, X, y):
        _Spy.seen.append(set(X[:, 1].astype(int)))
        return self

    def predict_proba(self, X):
        return np.c_[1 - X[:, 0] / 10, X[:, 0] / 10]


def test_loso_le_sujet_teste_n_est_jamais_dans_l_entrainement():
    table = _table()
    table["sid"] = table["subject"].astype(int)
    _Spy.seen = []
    proba, _ = loso_predict(table, ["x", "sid"], _Spy)
    assert len(_Spy.seen) == 8
    for k, seen in enumerate(_Spy.seen):       # pli k = sujet k retiré
        assert k not in seen and len(seen) == 7
    assert not np.isnan(proba).any()


def test_normalisation_apprise_dans_le_pli():
    """Le scaler fait partie du Pipeline : réappris à chaque pli."""
    for factory in make_models(CONFIG).values():
        steps = [type(s).__name__ for s in factory().named_steps.values()]
        assert steps[0] == "SimpleImputer"
    lr = make_models(CONFIG)["logreg"]()
    assert "standardscaler" in lr.named_steps


def test_loso_signal_fort_est_retrouve():
    proba, _ = loso_predict(_table(), ["x"], make_models(CONFIG)["logreg"])
    assert binary_metrics(_table()["label"], proba > 0.5)["accuracy"] >= 0.9


def test_metriques_a_la_main():
    m = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5
    # rien n'est prédit "bad" : précision non définie (NaN), pas 0
    assert np.isnan(binary_metrics([1, 0], [0, 0])["precision"])


def test_kappa():
    assert kappa([0, 1, 0, 1], [0, 1, 0, 1]) == pytest.approx(1.0)
    assert kappa([0, 1, 0, 1], [1, 0, 1, 0]) == pytest.approx(-1.0)


def test_bootstrap_reechantillonne_les_sujets():
    table = _table(n_subjects=10)
    ci = bootstrap_by_subject(table, lambda d: {"n": len(d)}, n=50, seed=1)
    # chaque tirage garde 10 sujets x 2 vidéos : toujours 20 lignes
    assert ci.iloc[0]["ci_low"] == ci.iloc[0]["ci_high"] == 20


def test_verdict_zone_incertaine():
    zone = CONFIG["evaluation"]["uncertain_zone"]
    assert verdict(0.5, zone) == "incertain"
    assert verdict(0.9, zone) == "bad" and verdict(0.1, zone) == "good"
    assert verdict(np.nan, zone) == "non analysable"


def test_features_par_vue():
    assert feature_names("side", RULES) == ["depth", "heel_rise", "trunk_vs_tibia",
                                            "head_position", "tempo_ratio"]
    assert feature_names("front", RULES) == ["knee_valgus", "hip_shift", "tempo_ratio"]
    assert feature_names("diagonal", RULES) == feature_names("front", RULES)


def test_rapport_largeur_hauteur_retrouve_sur_squat_synthetique():
    """Coordonnées normalisées d'un squat de profil filmé en 16:9 -> on retrouve ~1,78."""
    from src.external import estimate_aspect, to_pose_series
    from tests.synthetic import phase_signal, squat_pose

    pose = squat_pose(phase_signal(n_reps=3))
    width, height = 1920.0, 1080.0
    xy_norm = pose.xy / np.array([width, height])
    vis = np.ones(pose.vis.shape)
    assert abs(estimate_aspect(xy_norm, vis) - width / height) < 0.03
    # le retour en pixels respecte l'hypothèse donnée
    back = to_pose_series(xy_norm, vis, aspect=width / height, fps=30)
    assert back.width / back.height == pytest.approx(width / height, abs=0.01)
