# Gym-exe-analyzer

> **Outil pédagogique d'aide à l'analyse technique.** Ce n'est pas un dispositif médical et il ne fait
> aucune prédiction de blessure. Conçu comme aide pédagogique ; utilité non évaluée.

Application Streamlit qui analyse la technique d'un exercice de musculation à partir d'une vidéo
(MediaPipe Pose) et signale des erreurs de posture. Premier exercice : **le squat**.
Projet de mémoire de Master 2 (IPSSI).

## Installation (Windows)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

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
streamlit run streamlit_app.py
```

Page **Annotation (aveugle)** : annotation locale des 50 vidéos de profil du dataset, avec label masqué.
Résultats dans `data/annotations/side_blind.csv` (non versionné).

## Documentation

- [docs/PIPELINE.md](docs/PIPELINE.md) : schéma de bout en bout et limites déjà identifiées
- [docs/SOURCES.md](docs/SOURCES.md) : source et statut de chaque seuil
- [docs/INSPECTION.md](docs/INSPECTION.md) : inspection des vidéos « bad » et vérification des règles

## Tests

```powershell
pytest
ruff check .
```

## État

- [x] Phase 0 : mise en place, indexation du dataset
- [x] Phase A : pipeline cœur (pose, angles, répétitions, règles)
- [ ] Phase B : évaluation sur données réelles
- [ ] Phase C : interface, déploiement
