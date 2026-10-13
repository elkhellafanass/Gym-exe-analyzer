"""Éléments partagés par les pages : avertissement, badges de statut, documents.

Pourquoi lire les documents (docs/*.md, docs/results/*.csv) au lieu de recopier
les chiffres dans le code ? Une seule source de vérité : si une évaluation est
relancée, l'app affiche automatiquement les nouveaux résultats.
"""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitPageNotFoundError

from src.config import ROOT, load_config

DOCS = ROOT / "docs"
RESULTS = DOCS / "results"
FIGURES = DOCS / "figures"
STATUS_COLORS = {"concluant": "green", "indicatif": "orange", "descriptif": "gray"}


def disclaimer() -> None:
    """Avertissement affiché en haut de chaque page."""
    config = load_config()
    st.warning(f"**{config['app']['disclaimer']}** Conçu comme aide pédagogique ; "
               "utilité non évaluée.", icon="⚠️")


def status_badge(status: str) -> str:
    """Badge Markdown coloré pour un statut (concluant / indicatif / descriptif...)."""
    status = str(status)
    color = "red" if status.startswith("non retenu") else STATUS_COLORS.get(status.split()[0],
                                                                             "gray")
    return f":{color}-badge[{status}]"


@st.cache_data
def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


@st.cache_data
def read_doc(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


def doc_section(name: str, heading: str) -> str:
    """Section d'un document Markdown, du titre `heading` au titre suivant de même niveau."""
    text = read_doc(name)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^(#+)\s+(.*)$", line)
        if m and m.group(2).startswith(heading):
            level = len(m.group(1))
            out = []
            for nxt in lines[i + 1:]:
                n = re.match(r"^(#+)\s", nxt)
                if n and len(n.group(1)) <= level:
                    break
                out.append(nxt)
            return "\n".join(out).strip()
    return f"*(section « {heading} » introuvable dans {name})*"


def fmt_ci(estimate, low, high, pct: bool = False) -> str:
    """'0,61 [0,55 ; 0,67]' (ou en %)."""
    if pd.isna(estimate):
        return "–"
    f = (lambda x: f"{100 * x:.1f} %") if pct else (lambda x: f"{x:.2f}")
    if pd.isna(low) or pd.isna(high):
        return f(estimate).replace(".", ",")
    return f"{f(estimate)} [{f(low)} ; {f(high)}]".replace(".", ",")


def page_link(path: str, label: str, icon: str | None = None) -> None:
    """Lien vers une autre page ; simple texte si la page est exécutée seule (tests)."""
    try:
        st.page_link(path, label=label, icon=icon)
    except StreamlitPageNotFoundError:
        st.caption(f"→ {label} (menu de gauche)")
