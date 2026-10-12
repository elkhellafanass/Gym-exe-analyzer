"""Tests de l'outil d'annotation en aveugle."""

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.annotation import ANNOT_COLUMNS, blind_order, load_annotations, save_annotation
from src.config import ROOT


def _index():
    rows = [{"subject": f"{s:03d}", "quality": q, "view": v,
             "filename": f"subject_{s:03d}_squat_{q}_{v}.mp4", "path": "x"}
            for s in range(1, 6) for q in ("good", "bad") for v in ("side", "front")]
    return pd.DataFrame(rows)


def test_ordre_aveugle_deterministe_et_melange():
    a, b = blind_order(_index()), blind_order(_index())
    assert a["filename"].tolist() == b["filename"].tolist()        # reproductible
    assert len(a) == 10 and set(a["view"]) == {"side"}             # profil uniquement
    assert a["anon_id"].tolist()[:2] == ["V01", "V02"]
    # mélangé : pas simplement trié par nom de fichier
    assert a["filename"].tolist() != sorted(a["filename"])
    # l'identifiant anonyme ne contient pas le label
    assert not a["anon_id"].str.contains("good|bad").any()


def test_enregistrement_remplace_l_annotation(tmp_path):
    path = tmp_path / "a.csv"
    record = {"anon_id": "V01", "filename": "f.mp4", "overall": "mauvaise",
              "depth": "absent", "heel_rise": "la plupart des répétitions",
              "trunk_lean": "absent", "head": "absent", "knees_forward": "absent",
              "other": "absent", "comment": ""}
    save_annotation(path, record)
    save_annotation(path, {**record, "overall": "bonne"})
    table = load_annotations(path)
    assert len(table) == 1
    assert table["overall"].iloc[0] == "bonne"
    assert list(table.columns) == ANNOT_COLUMNS


def test_page_accueil_affiche_avertissement():
    at = AppTest.from_file(str(ROOT / "app" / "pages" / "home.py")).run()
    assert not at.exception
    text = " ".join(w.value for w in at.warning)
    assert "pas un dispositif médical" in text
    assert "utilité non évaluée" in text
