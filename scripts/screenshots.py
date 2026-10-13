"""Captures d'écran de l'application (Playwright) -> docs/figures/app_*.png.

Pourquoi un script ? Les captures du mémoire doivent être reproductibles et
montrer l'app telle qu'elle est. Confidentialité : l'option « squelette seul »
est TOUJOURS cochée, donc aucune image de la personne filmée n'apparaît
(seulement le squelette sur fond noir, les courbes et les tableaux). Le nom
de fichier envoyé est remplacé par « demo.mp4 ».

Étapes :
  1) lancer l'app en local (streamlit run, port 8600)
  2) capturer les pages statiques
  3) envoyer la vidéo, lancer l'analyse, capturer le rapport et le détail
  4) arrêter l'app

Usage : python scripts/screenshots.py CHEMIN_VIDEO.mp4 --view side
(installation : pip install -r requirements-dev.txt ; python -m playwright install chromium)
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
PORT = 8600
URL = f"http://localhost:{PORT}"
VIEWS = {"side": "Profil", "front": "Face", "diagonal": "Diagonale"}

parser = argparse.ArgumentParser()
parser.add_argument("video", type=Path)
parser.add_argument("--view", default="side", choices=list(VIEWS))
args = parser.parse_args()

print("=== 1) Lancement de l'app ===")
server = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", str(PORT),
     "--server.headless", "true"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(60):
    try:
        urllib.request.urlopen(URL, timeout=1)
        break
    except OSError:
        time.sleep(1)


def wait_app(page) -> None:
    """Attend la fin de l'exécution du script Streamlit (plus d'indicateur « running »)."""
    page.wait_for_selector("[data-testid='stApp']")
    page.wait_for_timeout(1500)
    page.wait_for_function(
        "!document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=600_000)
    page.wait_for_timeout(1000)


def shot(page, name: str) -> None:
    """Capture de toute la page : Streamlit fait défiler un conteneur interne, donc on
    agrandit la fenêtre à la hauteur du contenu avant la capture."""
    height = page.evaluate(
        "document.querySelector('[data-testid=\"stMain\"]').scrollHeight")
    page.set_viewport_size({"width": 1400, "height": min(int(height) + 40, 9000)})
    page.wait_for_timeout(1500)
    page.screenshot(path=OUT / f"{name}.png")
    page.set_viewport_size({"width": 1400, "height": 1000})
    print(f"  {name}.png")


def goto(page, title: str) -> None:
    page.get_by_role("link", name=title).first.click()
    wait_app(page)


try:
    with sync_playwright() as p, tempfile.TemporaryDirectory() as tmp:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.goto(URL)
        wait_app(page)

        print("=== 2) Pages statiques ===")
        shot(page, "app_accueil")
        for title, name in [("Entraînement", "app_entrainement"), ("Évaluation", "app_evaluation"),
                            ("Méthode et limites", "app_methode"),
                            ("Données et licences", "app_donnees")]:
            goto(page, title)
            shot(page, name)

        print("=== 3) Analyse d'une vidéo (squelette seul) ===")
        demo = Path(tmp) / "demo.mp4"
        shutil.copy(args.video, demo)
        goto(page, "Analyse d'une vidéo")
        page.get_by_text(VIEWS[args.view], exact=True).click()
        page.get_by_text("squelette seul").click()
        page.get_by_text("consentement de chaque personne").click()
        page.locator("input[type='file']").set_input_files(str(demo))
        page.get_by_role("button", name="Analyser la vidéo").click()
        page.wait_for_selector("text=Verdict global", timeout=900_000)
        wait_app(page)
        shot(page, "app_analyse")
        goto(page, "Résultats par répétition")
        shot(page, "app_resultats")
        browser.close()
finally:
    print("=== 4) Arrêt de l'app ===")
    server.terminate()
