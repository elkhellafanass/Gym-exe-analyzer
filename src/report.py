"""Rapport de correction : ce que l'utilisateur lit après l'analyse d'une vidéo.

Pourquoi un module séparé de l'interface ? Le même rapport est affiché dans
l'app ET exporté en HTML : le construire une seule fois garantit que les deux
disent la même chose. Et on peut le tester sans lancer Streamlit.

Principes (issus des évaluations, docs/PIPELINE.md) :
  - le verdict du modèle est une TENDANCE FAIBLE, jamais une probabilité brute :
    ses probabilités ne sont pas mieux calibrées qu'un « 50 % » constant ;
  - chaque règle affiche sa fiabilité RÉELLE (docs/results/rule_reliability.csv),
    avec la provenance et le statut (concluant / indicatif / descriptif) ;
  - seules les métriques « concluantes » sont présentées comme des résultats ;
  - chaque correction cite sa source et dit « à valider par un expert » quand
    le seuil n'est pas validé. Aucun conseil médical.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.analysis import Analysis
from src.config import ROOT
from src.features import video_features

RESULTS = ROOT / "docs" / "results"
MODEL_PATH = ROOT / "models" / "squat_classifier.joblib"
VIEW_LABELS = {"side": "profil", "front": "face", "diagonal": "diagonale"}
TEMPO_NOTE = ("Le modèle s'appuie en partie sur le tempo (rapport descente / remontée). "
              "Dans les données d'entraînement, il est mesuré à 15 images/s (±0,07 s) : "
              "c'est une source d'imprécision.")


@dataclass
class Verdict:
    tendency: str          # "bad", "good", "aucune", "non analysable"
    title: str             # phrase affichée
    explanation: str
    performance: str       # performance mesurée en LOSO pour la vue
    notes: list[str] = field(default_factory=list)


def load_model(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def loso_performance(view: str) -> str:
    """Exactitude équilibrée LOSO de la régression logistique pour la vue (avec IC)."""
    perf = pd.read_csv(RESULTS / "performance.csv")
    row = perf[(perf["method"] == "Régression logistique") & (perf["view"] == view)
               & (perf["metric"] == "balanced_accuracy")]
    if row.empty:
        return "performance non disponible"
    r = row.iloc[0]
    return (f"exactitude équilibrée {r['estimate']:.2f} [IC 95 % {r['ci_low']:.2f} ; "
            f"{r['ci_high']:.2f}] en leave-one-subject-out (hasard = 0,50)").replace(".", ",")


def classifier_verdict(analysis: Analysis, config: dict, bundle: dict | None = None) -> Verdict:
    """Tendance du modèle, SANS probabilité brute.

    La probabilité sert seulement à choisir le libellé : hors de la zone
    « incertain » (config.yaml), on parle de tendance faible ; dedans, d'absence
    de tendance. Pourquoi ne pas l'afficher ? Le score de Brier en LOSO (0,251)
    est égal à celui d'un « 50 % » constant : le chiffre ferait croire à une
    précision qui n'existe pas.
    """
    bundle = bundle or load_model()
    view = analysis.view
    perf = loso_performance(view)
    if len(analysis.reps) == 0:
        return Verdict("non analysable", "Pas de verdict : aucune répétition détectée.",
                       "Le modèle a besoin d'au moins une répétition.", perf)
    rules = config["exercises"]["squat"]["rules"]
    model = bundle["views"][view]
    row = video_features(analysis, rules)
    x = np.array([[row.get(f, np.nan) for f in model["features"]]], dtype=float)
    proba = float(model["model"].predict_proba(x)[0, 1])
    low, high = bundle["uncertain_zone"]
    source = "le label « good / bad » du dataset Mendeley"
    if proba > high:
        verdict = Verdict("bad", "Tendance faible : exécution plutôt « à corriger »",
                          f"Le modèle penche légèrement vers {source} « bad ».", perf)
    elif proba < low:
        verdict = Verdict("good", "Tendance faible : exécution plutôt « correcte »",
                          f"Le modèle penche légèrement vers {source} « good ».", perf)
    else:
        verdict = Verdict("aucune", "Aucune tendance : le modèle ne peut pas trancher",
                          "C'est le cas d'environ 2 vidéos sur 3 dans l'évaluation.", perf)
    verdict.notes = [
        "Ce verdict reproduit un label de dataset (juges non experts, critère non documenté), "
        "pas un avis de coach.",
        "Ses probabilités ne sont pas fiables (score de Brier égal à celui d'un « 50 % » "
        "constant) : aucune probabilité n'est affichée.",
        TEMPO_NOTE,
    ]
    if analysis.pose.fps >= 25:
        verdict.notes.append(f"Votre vidéo est à {analysis.pose.fps:.0f} images/s, le modèle a "
                             "été entraîné à 15 images/s : le tempo est mesuré différemment.")
    return verdict


def reliability_rows(rule: str, view: str) -> pd.DataFrame:
    """Lignes de rule_reliability.csv pour une règle : la vue filmée + les jeux externes."""
    rel = pd.read_csv(RESULTS / "rule_reliability.csv")
    rel = rel[rel["rule"] == rule]
    mendeley = rel["provenance"].str.startswith("Mendeley")
    return rel[(mendeley & (rel["view"] == view)) | ~mendeley].reset_index(drop=True)


def _pct(x) -> str:
    return "–" if pd.isna(x) else f"{100 * x:.0f} %"


def _ci(row: pd.Series, metric: str) -> str:
    lo, hi = row.get(f"{metric}_ci_low"), row.get(f"{metric}_ci_high")
    if pd.isna(lo) or pd.isna(hi):
        return ""
    return f" [IC {100 * lo:.0f}-{100 * hi:.0f} %]"


def reliability_lines(rule: str, view: str) -> list[dict]:
    """Une ligne lisible par provenance. Seul le statut « concluant » est un résultat."""
    lines = []
    for _, r in reliability_rows(rule, view).iterrows():
        prov = r["provenance"]
        if prov.startswith("Mendeley"):
            if pd.notna(r.get("detection")):
                text = (f"contre l'annotation en aveugle (50 vidéos) : détection "
                        f"{_pct(r['detection'])}, fausses alertes {_pct(r['false_alert'])}")
            else:
                text = (f"signalée sur {_pct(r['flag_rate_good'])} des vidéos good et "
                        f"{_pct(r['flag_rate_bad'])} des bad ; non évaluée contre annotation")
            lines.append({"provenance": prov, "text": text,
                          "status": "non classé (phase B, effectif : 25 sujets)"})
            continue
        for metric, name in [("detection", "détection"), ("false_alert", "fausses alertes")]:
            if pd.isna(r.get(metric)):
                continue
            status = r.get(f"{metric}_status")
            conclusion = str(r.get(f"{metric}_conclusion", ""))
            if conclusion.startswith("non retenue"):
                status = "non retenu (dépend du format d'image)"
            text = f"{name} {_pct(r[metric])}{_ci(r, metric)} ({r['evaluated_against']})"
            lines.append({"provenance": prov, "text": text, "status": status})
    return lines


def correction_text(rule_name: str, config: dict) -> dict:
    """Correction écrite : message, source, statut du seuil. Aucun conseil médical."""
    rule = config["exercises"]["squat"]["rules"][rule_name]
    src = config["sources"][rule["source"]]
    expert = "" if rule["status"] == "sourcé" else "À valider par un expert."
    return {"label": rule["label"], "message": rule["message"],
            "source": f"{src['short']} ({src['url']})", "status": rule["status"],
            "expert": expert}


@dataclass
class Report:
    video_name: str
    view: str
    created: str
    verdict: Verdict
    warnings: list[str]
    rules: pd.DataFrame          # une ligne par règle applicable
    reps: pd.DataFrame           # une ligne par répétition
    reliability: dict            # règle -> lignes de fiabilité
    corrections: list[dict]      # défauts retenus seulement


def build_report(analysis: Analysis, config: dict, video_name: str,
                 bundle: dict | None = None) -> Report:
    summary = analysis.summary.copy()
    levels = config["reliability_summary"]
    summary["reliability_level"] = summary["rule"].map(lambda r: levels[r]["level"])
    summary["reliability_text"] = summary["rule"].map(lambda r: levels[r]["text"])
    retained = summary[summary["video_error"].astype(bool)]
    return Report(
        video_name=video_name, view=analysis.view,
        created=datetime.now().strftime("%d/%m/%Y %H:%M"),
        verdict=classifier_verdict(analysis, config, bundle),
        warnings=list(analysis.warnings),
        rules=summary, reps=analysis.rep_table(),
        reliability={r: reliability_lines(r, analysis.view) for r in summary["rule"]},
        corrections=[correction_text(r, config) for r in retained["rule"]],
    )


def rep_figure(analysis: Analysis) -> go.Figure:
    """Frise : hauteur de hanche dans le temps, répétitions surlignées (rouge = défaut)."""
    ind, reps, res = analysis.indicators, analysis.reps, analysis.rep_results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ind["time_s"], y=-ind["hip_y"], mode="lines",
                             name="hauteur de hanche", line={"color": "#4C72B0"}))
    t = ind["time_s"].to_numpy()
    for rep in reps.itertuples():
        errs = res[(res["rep"] == rep.rep) & (res["error"] == True)]  # noqa: E712
        color = "rgba(220,60,60,0.25)" if len(errs) else "rgba(80,180,80,0.18)"
        label = ", ".join(errs["label"]) if len(errs) else "aucun défaut"
        fig.add_vrect(x0=t[rep.start], x1=t[rep.end], fillcolor=color, line_width=0,
                      annotation_text=f"R{rep.rep}", annotation_position="top left")
        fig.add_trace(go.Scatter(x=[t[rep.bottom]], y=[-ind["hip_y"].iloc[rep.bottom]],
                                 mode="markers", marker={"color": "black", "size": 6},
                                 showlegend=False, hovertext=f"Répétition {rep.rep} : {label}",
                                 hoverinfo="text"))
    fig.update_layout(height=280, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      xaxis_title="temps (s)", yaxis_title="hanche (haut = debout)",
                      title="Répétitions (vert : aucun défaut ; rouge : au moins un défaut)")
    return fig


def rule_figure(analysis: Analysis, rule: str) -> go.Figure:
    """Valeur mesurée à chaque répétition, avec le seuil figé."""
    res = analysis.rep_results[analysis.rep_results["rule"] == rule]
    colors = ["#999999" if pd.isna(e) else "#DC3C3C" if bool(e) else "#50B450"
              for e in res["error"]]
    fig = go.Figure(go.Bar(x=[f"R{r}" for r in res["rep"]], y=res["value"],
                           marker_color=colors, name="valeur"))
    if len(res):
        fig.add_hline(y=res["threshold"].iloc[0], line_dash="dash", line_color="black",
                      annotation_text=f"seuil {res['op'].iloc[0]} {res['threshold'].iloc[0]:g}")
    fig.update_layout(height=220, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      title=res["label"].iloc[0] if len(res) else rule, showlegend=False)
    return fig


def to_html(report: Report, analysis: Analysis, config: dict) -> str:
    """Rapport HTML autonome (graphiques intégrés). Aucune image de la vidéo."""
    e = html.escape
    v = report.verdict
    parts = [
        "<!doctype html><html lang='fr'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Rapport squat - {e(report.video_name)}</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:960px;margin:auto;padding:16px;"
        "color:#222;background:#fff}table{border-collapse:collapse;width:100%;margin:8px 0}"
        "td,th{border:1px solid #ccc;padding:4px 6px;font-size:14px;text-align:left}"
        ".warn{background:#fff3cd;padding:10px;border-radius:6px}"
        ".badge{padding:1px 6px;border-radius:4px;font-size:12px;background:#eee}"
        ".concluant{background:#d4edda}.indicatif{background:#ffe8cc}"
        ".descriptif{background:#e2e3e5}</style></head><body>",
        "<h1>Rapport de correction : squat</h1>",
        f"<p class='warn'><b>{e(config['app']['disclaimer'])}</b> Conçu comme aide pédagogique ; "
        "utilité non évaluée. Ce rapport ne remplace pas un professionnel.</p>",
        f"<p>Vidéo : {e(report.video_name)} · vue déclarée : {VIEW_LABELS[report.view]} · "
        f"rapport créé le {report.created}</p>",
        f"<h2>Verdict global</h2><p><b>{e(v.title)}</b><br>{e(v.explanation)}</p>",
        f"<p>Performance mesurée pour cette vue : {e(v.performance)}.</p><ul>",
        *[f"<li>{e(n)}</li>" for n in v.notes], "</ul>",
    ]
    if report.warnings:
        parts += ["<h2>Avertissements sur la vidéo</h2><ul>",
                  *[f"<li>{e(w)}</li>" for w in report.warnings], "</ul>"]
    parts.append("<h2>Répétitions</h2>")
    parts.append(rep_figure(analysis).to_html(full_html=False, include_plotlyjs="inline"))
    parts.append("<h2>Règles</h2><table><tr><th>Règle</th><th>Répétitions concernées</th>"
                 "<th>Défaut retenu (≥ 2 rép.)</th><th>Fiabilité de la règle</th></tr>")
    for r in report.rules.itertuples():
        parts.append(f"<tr><td>{e(r.label)}</td><td>{r.reps_with_error}/{r.reps_measured}</td>"
                     f"<td>{'oui' if r.video_error else 'non'}</td>"
                     f"<td><b>{e(r.reliability_level)}</b> : {e(r.reliability_text)}</td></tr>")
    parts.append("</table>")
    for rule in report.rules["rule"]:
        parts.append(rule_figure(analysis, rule).to_html(full_html=False,
                                                         include_plotlyjs=False))
        rows = "".join(f"<tr><td>{e(x['provenance'])}</td><td>{e(x['text'])}</td>"
                       f"<td><span class='badge {e(str(x['status']).split()[0])}'>"
                       f"{e(str(x['status']))}</span></td></tr>"
                       for x in report.reliability[rule])
        parts.append("<table><tr><th>Provenance</th><th>Mesure de fiabilité</th>"
                     f"<th>Statut</th></tr>{rows}</table>")
    parts.append("<h2>Corrections</h2>")
    if not report.corrections:
        parts.append("<p>Aucun défaut retenu (un défaut doit apparaître sur au moins 2 "
                     "répétitions).</p>")
    for c in report.corrections:
        parts.append(f"<h3>{e(c['label'])}</h3><p>{e(c['message'])}</p>"
                     f"<p>Source : {e(c['source'])}. Statut du seuil : {e(c['status'])}. "
                     f"<b>{e(c['expert'])}</b></p>")
    ds = config["dataset"]
    parts += [
        "<h2>Sources et données</h2>",
        f"<p>Critères : {e(config['sources']['myer2014']['citation'])} "
        f"{e(config['sources']['myer2014']['url'])}</p>",
        f"<p>Modèle entraîné sur : {e(ds['citation'])}. Licence {e(ds['license'])} "
        f"({e(ds['license_url'])}). Données transformées (extraction de landmarks et "
        "d'angles).</p>",
        "<p>Limites : mesures 2D, une seule caméra, règles peu informatives sauf la profondeur, "
        "modèle proche du hasard (voir la page « Méthode et limites » de l'application).</p>",
        "</body></html>",
    ]
    return "\n".join(parts)
