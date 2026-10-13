"""Tests de l'application (AppTest) et du rapport de correction.

Aucun test n'utilise de vidéo du dataset : l'analyse vient d'un squat
synthétique (tests/synthetic.py).
"""

import re

import pytest
from streamlit.testing.v1 import AppTest

from src.analysis import analyze_pose
from src.config import ROOT, load_config
from src.report import build_report, reliability_lines, to_html
from tests.synthetic import phase_signal, squat_pose

CONFIG = load_config()
PAGES = ["home", "analyze", "results", "training", "evaluation", "method", "data"]
ALLOWED_STATUS = re.compile(r"^(concluant|indicatif|descriptif|non retenu|non classé)")


def _analysis():
    pose = squat_pose(phase_signal(n_reps=5), thigh_max=70, trunk_max=60, tibia_max=25)
    return analyze_pose(pose, "side", CONFIG)


def _page(name):
    return AppTest.from_file(str(ROOT / "app" / "pages" / f"{name}.py"), default_timeout=60)


@pytest.mark.parametrize("name", PAGES)
def test_chaque_page_s_affiche_avec_avertissement(name):
    at = _page(name).run()
    assert not at.exception
    text = " ".join(w.value for w in at.warning)
    assert "pas un dispositif médical" in text
    assert "utilité non évaluée" in text


def test_navigation_principale():
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception


def test_analyse_demande_le_consentement_et_accepte_mp4_mov():
    at = _page("analyze").run()
    assert any("consentement" in c.label for c in at.checkbox)
    source = (ROOT / "app" / "pages" / "analyze.py").read_text(encoding="utf-8")
    assert 'type=["mp4", "mov"]' in source


def test_resultats_sans_analyse_invite_a_envoyer_une_video():
    at = _page("results").run()
    assert any("Aucune analyse" in i.value for i in at.info)


def test_pages_rapport_avec_une_analyse():
    analysis = _analysis()
    report = build_report(analysis, CONFIG, "test.mp4")
    for name in ("analyze", "results"):
        at = _page(name)
        at.session_state["analysis"] = analysis
        at.session_state["report"] = report
        at.session_state["annotated"] = b"\x00"
        at.session_state["report_html"] = "<html></html>"
        at.run()
        assert not at.exception
        shown = " ".join(m.value for m in at.markdown)
        assert "fiabilité" in shown.lower()


def test_verdict_tendance_faible_sans_probabilite():
    report = build_report(_analysis(), CONFIG, "test.mp4")
    v = report.verdict
    assert v.tendency in {"bad", "good", "aucune"}
    assert "Tendance faible" in v.title or "Aucune tendance" in v.title
    # aucune probabilité brute (nombre décimal ou pourcentage) dans le verdict
    assert not re.search(r"\d[.,]\d|%", v.title + v.explanation)
    assert any("tempo" in n and "15 images/s" in n for n in v.notes)


def test_fiabilite_avec_provenance_et_statut():
    for rule, view in [("depth", "side"), ("trunk_vs_tibia", "side"), ("knee_valgus", "front")]:
        lines = reliability_lines(rule, view)
        assert lines
        for line in lines:
            assert line["provenance"]
            assert ALLOWED_STATUS.match(str(line["status"])), line
    # la profondeur porte le seul résultat externe concluant (fausses alertes sur Dai)
    depth = reliability_lines("depth", "side")
    assert any(x["status"] == "concluant" and "fausses alertes 5 %" in x["text"] for x in depth)
    # les fausses alertes des règles d'angle sur Dai sont « non retenues » (format)
    trunk = reliability_lines("trunk_vs_tibia", "side")
    assert any(str(x["status"]).startswith("non retenu") for x in trunk)


def test_chaque_regle_a_un_resume_de_fiabilite():
    for rule in CONFIG["exercises"]["squat"]["rules"]:
        assert CONFIG["reliability_summary"][rule]["level"]
        assert CONFIG["reliability_summary"][rule]["text"]


def test_corrections_citent_la_source_et_l_expert():
    report = build_report(_analysis(), CONFIG, "test.mp4")
    assert report.corrections, "le squat synthétique doit déclencher au moins un défaut"
    for c in report.corrections:
        assert "Myer" in c["source"]
        if c["status"] != "sourcé":
            assert c["expert"] == "À valider par un expert."


def test_export_html_autonome():
    analysis = _analysis()
    report = build_report(analysis, CONFIG, "test.mp4")
    page = to_html(report, analysis, CONFIG)
    assert page.startswith("<!doctype html>")
    assert "pas un dispositif médical" in page
    assert "10.17632/kgbb3yn47p.2" in page            # attribution du dataset
    assert "Données transformées" in page
    assert "<img" not in page                          # aucune image de la vidéo
    assert "cdn" not in page.split("<body>")[0]        # graphiques intégrés, pas de CDN


def test_badges_de_statut():
    from app.common import status_badge
    assert status_badge("concluant").startswith(":green")
    assert status_badge("indicatif").startswith(":orange")
    assert status_badge("non retenu (dépend du format)").startswith(":red")
    assert status_badge("non classé (phase B)").startswith(":gray")
