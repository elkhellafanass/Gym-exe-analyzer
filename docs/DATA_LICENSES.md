# Données et licences

## 1. Dataset utilisé : vidéos de squat (Mendeley Data)

| Champ | Valeur |
|---|---|
| Citation | Sai Prashanth, Duddela; Kamath, Priya; Bhandary, Rakshith; Pai, S Ashlesh; PN, Pratham; P, Advith; Mashetty, Abhishek (2026), "A Multi-View Raw Video Dataset of Seven Fitness Exercises with Good/Bad Form Labels", Mendeley Data, V2, doi: 10.17632/kgbb3yn47p.2 |
| URL | https://data.mendeley.com/datasets/kgbb3yn47p/2 |
| DOI | [10.17632/kgbb3yn47p.2](https://doi.org/10.17632/kgbb3yn47p.2) |
| Licence | CC BY 4.0 : https://creativecommons.org/licenses/by/4.0/ (vérifiée sur la page Mendeley) |
| Date de consultation | 12/10/2026 |
| Sous-ensemble utilisé | `Dataset Exercise Quality-wise/squat` (good/ et bad/), 150 vidéos |

**Modifications (exigence CC BY 4.0) :** données transformées (extraction de landmarks et d'angles).
Les vidéos originales ne sont pas modifiées. Le projet en tire :
- un index de métadonnées (`data/index.csv`) ;
- des points du squelette (landmarks MediaPipe), des angles articulaires et des indicateurs par répétition ;
- des métriques agrégées (tableaux CSV, figures) publiées dans `docs/figures/`.

**Non-publication des vidéos :** la licence autoriserait leur redistribution, mais les vidéos montrent des personnes
identifiables. Elles ne sont **jamais** versionnées ni publiées (dépôt GitHub, app en ligne), ni sous forme
d'images extraites (`docs/inspection/` est dans `.gitignore`). Seules des données dérivées et agrégées sont publiées.

## 2. Modèle de détection de pose

| Champ | Valeur |
|---|---|
| Outil | MediaPipe Pose Landmarker (Google), modèle BlazePose GHUM 3D *full*, 33 points |
| Bibliothèque | `mediapipe==1.1.0` (PyPI), Apache 2.0 |
| Fichier modèle | `pose_landmarker_full.task`, téléchargé depuis https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task (SHA-256 `4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad`, 12/10/2026). Non versionné |
| Licence du modèle | Apache 2.0, d'après la fiche modèle : https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf |
| Date de consultation | 12/10/2026 |

## 3. Vidéos personnelles (`data/own/`)

Filmées par l'auteur, avec le consentement écrit de chaque personne filmée
(modèle : [`CONSENT_TEMPLATE.md`](CONSENT_TEMPLATE.md)). Ces vidéos ne sont jamais versionnées.

## 4. Datasets explicitement non utilisés

- Fitness-AQA : non utilisé (choix du projet).
