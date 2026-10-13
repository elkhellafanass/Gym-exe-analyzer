# Gym-exe-analyzer

> **Outil pédagogique d'aide à l'analyse technique.** Ce n'est pas un dispositif médical et il ne fait
> aucune prédiction de blessure. Conçu comme aide pédagogique ; utilité non évaluée.

Application Streamlit qui analyse la technique d'un exercice de musculation à partir d'une vidéo
(MediaPipe Pose) et signale des erreurs de posture. Premier exercice : **le squat**.
Projet de mémoire de Master 2 (IPSSI).

## Installation (Windows)

Prérequis : Python 3.12 (`py -3.12 --version`).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # application seule
pip install -r requirements-dev.txt      # + tests, ruff, Playwright (développement)
```

Le modèle MediaPipe (`models/pose_landmarker_full.task`, 9 Mo) est téléchargé automatiquement à la première
analyse. Le classifieur entraîné (`models/squat_classifier.joblib`, 4 Ko, aucune image) est versionné.

## Données

Le dataset n'est **pas** inclus dans le dépôt. Indiquez son dossier `squat` dans `config.yaml` ou via une
variable d'environnement :

```powershell
$env:SQUAT_DATA_DIR = "C:\chemin\vers\Dataset Exercise Quality-wise\squat"
python scripts/prepare_dataset.py   # -> data/index.csv
```

Dataset : Sai Prashanth, Duddela; Kamath, Priya; Bhandary, Rakshith; Pai, S Ashlesh; PN, Pratham; P, Advith;
Mashetty, Abhishek (2026), "A Multi-View Raw Video Dataset of Seven Fitness Exercises with Good/Bad Form
Labels", Mendeley Data, V2, doi: 10.17632/kgbb3yn47p.2. Licence CC BY 4.0. Données transformées (extraction de
landmarks et d'angles). Détails : [docs/DATA_LICENSES.md](docs/DATA_LICENSES.md).

## Lancer l'application

```powershell
streamlit run streamlit_app.py           # puis http://localhost:8501
```

| Page | Contenu |
|---|---|
| Accueil et avertissements | ce que fait l'outil, ce qu'il ne fait pas, ce qu'ont montré les évaluations |
| Analyse d'une vidéo | upload mp4/mov, vue déclarée, consentement, verdict « tendance faible », vidéo annotée (option « squelette seul ») et rapport HTML téléchargeables |
| Résultats par répétition | mesures par répétition, règle déclenchée ou non, fiabilité de chaque règle (provenance + statut), corrections avec source |
| Entraînement | méthode, paramètres et code de `scripts/train.py`, résultats réels |
| Évaluation | Mendeley (phase B) et validation externe, statut *concluant / indicatif / descriptif* de chaque métrique |
| Méthode et limites | lues depuis `docs/PIPELINE.md` et `docs/SOURCES.md` |
| Données et licences | attribution exacte, règles de confidentialité |
| Annotation (aveugle) | **local uniquement** (visible seulement si `data/index.csv` existe) |

La vidéo envoyée est traitée dans un dossier temporaire supprimé après l'analyse. Le rapport HTML est autonome
(graphiques intégrés) et ne contient aucune image de la vidéo ; pour un PDF, ouvrez-le puis « Imprimer > PDF ».

Captures d'écran (`docs/figures/app_*.png`, option « squelette seul », aucun visage) :

```powershell
python -m playwright install chromium
python scripts/screenshots.py CHEMIN\video_consentie.mp4 --view side
```

## Déploiement sur Streamlit Community Cloud

Préparé dans le dépôt : `requirements.txt` (versions épinglées, `opencv-python-headless`), `packages.txt`
(bibliothèques système `libgl1`, `libglib2.0-0`, `libportaudio2` : MediaPipe installe aussi
`opencv-contrib-python`, qui en a besoin), `.streamlit/config.toml` (upload 200 Mo). **Aucune vidéo de dataset
n'est dans le dépôt** : l'app en ligne analyse uniquement les vidéos envoyées.

1. Pousser la branche `main` sur GitHub (`git push origin main`).
2. Sur https://share.streamlit.io : *Create app* → dépôt `elkhellafanass/Gym-exe-analyzer`, branche `main`,
   fichier `streamlit_app.py`.
3. *Advanced settings* → **Python 3.12**. Aucun secret n'est nécessaire.
4. *Deploy*, puis lire les journaux : si une bibliothèque de `packages.txt` est introuvable (nom différent selon la
   version de Debian), la remplacer (par exemple `libglib2.0-0t64`) et pousser.
5. Tester avec une vidéo filmée avec consentement : la première analyse télécharge le modèle MediaPipe.

## Entraînement et évaluation (phase B)

```powershell
python scripts/extract_landmarks.py     # une fois (~35 min) : points MediaPipe en cache
python scripts/train.py                 # LOSO + modèle models/squat_classifier.joblib
python scripts/evaluate.py              # analyse principale (~4 min)
python scripts/evaluate_option_b.py     # analyse secondaire : seuils recalibrés en LOSO
```

Résultats : `docs/results/` (tableaux) et `docs/figures/` (figures), commentés dans
[docs/PIPELINE.md](docs/PIPELINE.md). En bref : la régression logistique fait un peu mieux que le hasard
(AUC 0,60 [0,54 ; 0,67]), mais ses probabilités ne sont pas mieux calibrées qu'un 50 % constant.

## Validation externe (jeux ouverts, hors dépôt)

Deux jeux CC BY 4.0, à télécharger depuis Zenodo et à extraire hors du dépôt (`config.yaml` → `external.root`,
ou `$env:EXTERNAL_DATA_DIR`) : Teng 2025 (images, DOI 10.5281/zenodo.17558630) et Dai 2026 v2 (points MediaPipe,
DOI 10.5281/zenodo.19550070). Attribution complète : [docs/DATA_LICENSES.md](docs/DATA_LICENSES.md).

```powershell
python scripts/inspect_external.py      # description des jeux (~10 min la première fois)
python scripts/evaluate_external.py     # règles figées, protocole committé avant exécution (~8 min)
```

En bref : sur Teng (dos très penché) et sur Shallow (squat peu profond), les règles signalent, à titre
indicatif, les erreurs marquées et mises en scène. La profondeur ne donne que 4,7 % de fausses alertes sur des
squats « standard » d'un autre laboratoire (résultat concluant, indépendant du format d'image). C'est compatible avec l'hypothèse que le label
Mendeley est peu fiable, sans la prouver (laboratoires, caméras et sujets différents). Les effectifs (6 sujets
Shallow, ~10 scènes) ne permettent pas de conclure sur la détection. Détails et limites :
[docs/PIPELINE.md](docs/PIPELINE.md).

## Documentation

- [docs/PIPELINE.md](docs/PIPELINE.md) : schéma de bout en bout et limites déjà identifiées
- [docs/SOURCES.md](docs/SOURCES.md) : source et statut de chaque seuil
- [docs/INSPECTION.md](docs/INSPECTION.md) : inspection des vidéos « bad » et vérification des règles

## Tests

```powershell
pytest            # 54 tests : pipeline, LOSO sans fuite, métriques, rapport, pages (AppTest)
ruff check .
```

## État

- [x] Phase 0 : mise en place, indexation du dataset
- [x] Phase A : pipeline cœur (pose, angles, répétitions, règles)
- [x] Phase B : évaluation sur données réelles (résultats faibles, voir docs/PIPELINE.md)
- [x] Validation externe : Teng 2025 et Dai 2026 v2 (protocole figé avant exécution)
- [x] Phase C : application, rapport de correction, tests AppTest, captures, déploiement préparé
