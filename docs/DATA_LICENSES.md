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

**Ce que la licence ne couvre pas :** la CC BY 4.0, section 2(b)(1), exclut explicitement les droits de la
personnalité : "Moral rights [...] are not licensed under this Public License, nor are publicity, privacy, and/or
other similar personality rights" (https://creativecommons.org/licenses/by/4.0/legalcode.en). Le README du dataset
ne mentionne pas le consentement des participants.

**Non-publication des vidéos :** la licence autoriserait leur redistribution, mais les vidéos montrent des personnes
identifiables. Elles ne sont **jamais** versionnées ni publiées (dépôt GitHub, app en ligne), ni sous forme
d'images extraites (`docs/inspection/` est dans `.gitignore`). Seules des données dérivées et agrégées sont publiées.

## 1 bis. Jeux de données externes (validation externe, inspection faite le 13/10/2026)

Les deux jeux sont extraits **hors de git** (`config.yaml` → `external.root`, défaut `data/external/`, surchargeable par
`EXTERNAL_DATA_DIR`). Ils ne sont jamais commités. Les zip d'origine (conservés hors du dépôt)
ne sont ni modifiés ni supprimés : leur MD5 est identique à celui publié sur Zenodo, et leur SHA-256 est le même
avant et après l'extraction. Les métadonnées ci-dessous viennent de l'API Zenodo (`/api/records/<id>`),
consultée le 13/10/2026.

### A. Teng 2025 : Squat Dataset (images)

| Champ | Valeur |
|---|---|
| Attribution | Teng, Christopher (Binus University) (2025). *Squat Dataset* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17558630 |
| DOI | [10.5281/zenodo.17558630](https://doi.org/10.5281/zenodo.17558630) (concept : 10.5281/zenodo.17558629 ; une seule version) |
| Date de publication | 08/11/2025 |
| Licence | CC BY 4.0 (champ licence de Zenodo) : https://creativecommons.org/licenses/by/4.0/ |
| Fichier | `Dataset.zip`, 824 064 094 octets, MD5 `cebeb32ac04dab34d4abe9686a9fc5d3` (vérifié) |
| Date de consultation | 13/10/2026 |
| Modifications | **données transformées** (extraction de landmarks et d'angles par MediaPipe) ; images d'origine non modifiées |

Description Zenodo (résumé) : images de profil de personnes qui font des squats, 3 classes : Good, Bad Back
(« spinal misalignment »), Bad Heel (« improper foot positioning »). Format annoncé : 1:1.

**Ce que la page ne documente pas :** le nombre de sujets, l'identifiant du sujet par image, le protocole
d'étiquetage (qui a jugé, selon quels critères), le consentement des personnes, l'origine des images (images
extraites de vidéos ?) et la règle du découpage train/test.
**Personnes identifiables : oui.** Les visages sont visibles (vérifié sur une planche de 24 images, gardée hors du
dépôt). Comme pour le dataset Mendeley, aucune image n'est publiée, et la clause CC BY 4.0 2(b)(1) s'applique.

### B. Dai 2026 : points MediaPipe (CSV), deux versions

**Version utilisée pour l'évaluation : v2 (13/04/2026).** La v1 est gardée seulement pour comparaison.

| Champ | v2 (UTILISÉE) | v1 (comparaison) |
|---|---|---|
| Attribution | Dai, Zhiqiang (2026). *Dataset for: Non-standard Squat Posture Detection Method Using Human Skeleton* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.19550070 | idem, https://doi.org/10.5281/zenodo.18859797 |
| DOI | [10.5281/zenodo.19550070](https://doi.org/10.5281/zenodo.19550070) | [10.5281/zenodo.18859797](https://doi.org/10.5281/zenodo.18859797) |
| DOI concept (toutes versions) | 10.5281/zenodo.18859796 | 10.5281/zenodo.18859796 |
| Date de publication | 13/04/2026 | 04/03/2026 |
| Licence | CC BY 4.0 (champ licence de Zenodo) : https://creativecommons.org/licenses/by/4.0/ | CC BY 4.0 |
| Sujets annoncés sur Zenodo | 35 | 22 |
| Fichiers téléchargés | `00_Standard_v2.zip` (28 027 541 o, MD5 `f7b318a7a3d7ac0b9d1838da55d0deef`), `01_Err_Shallow_v2.zip` (32 732 769 o, `9a77b1856f0bd604421e9e40c640b8ec`), `02_Err_GoodMorning_v2.zip` (905 665 o, `5c9667ae5cfb66550b2d946ae4acfee7`) : tailles et MD5 identiques à Zenodo. Renommés `_v2` par l'auteur du projet. `all-kinds-deep-learning.zip` et `our-result.zip` non téléchargés | `00_Standard.zip` (`7d7c2c037139fbdbd58121f8cf17742d`), `01_Err_Shallow.zip` (`9bffec67df8ae2f8f3e6b1eb5cb2c63c`), `02_Err_GoodMorning.zip` (`9f0d0c36cfd8b849f04f6fbb0899e9f8`) : vérifiés |
| Dossier d'extraction (hors dépôt) | `extracted\B_dai_squat_v2\` | `extracted\B_dai_squat\` |
| Date de consultation | 13/10/2026 | 13/10/2026 |
| Modifications | **données transformées** (calcul d'angles et d'indicateurs à partir des points fournis) | idem |

Description Zenodo (résumé, identique dans les deux versions à part le nombre de sujets) : coordonnées 2D de points
du squelette et angles de hanche et de genou, extraits par MediaPipe ; 3 postures (squat standard, squat peu
profond, « good morning ») ; vidéos brutes exclues pour protéger la vie privée. Aucune personne n'est identifiable
dans les fichiers (coordonnées seulement).

**Ce que contiennent vraiment les fichiers (vérifié par `scripts/inspect_external.py`) :**

| | v1 | v2 |
|---|---|---|
| CSV extraits / fichiers uniques (copies identiques retirées) | 442 / 226 | 650 / 344 |
| Sujets distincts (annoncés) | 17 (22) | **28 (35)** : 01 à 29 sans 14 |
| Standard : fichiers / sujets | 193 / 17 | 296 / 28 |
| Shallow : fichiers / sujets | 23 / 4 | 38 / 6 |
| GoodMorning : fichiers / sujets | 10 / 1 | **10 / 1** (sujet 01 seulement) |
| Colonnes | `Frame` + 33 × (x, y, z, v) | idem, **aucune colonne d'angle** |
| Images par fichier | 56 à 133 | 49 à 133 |
| Numérotation `Frame` avec trous | 0 fichier | 5 fichiers |

- Les 226 fichiers de la v1 sont **tous présents à l'identique** dans la v2 (même MD5) ; la v2 ajoute 118 fichiers
  et les sujets 19 à 29 (Standard seulement, plus les sujets 05-06 en Shallow).
- Les zip se recoupent : `01_Err_Shallow_v2.zip` contient aussi des copies identiques de Standard et GoodMorning
  (v1 : c'était `02_Err_GoodMorning.zip`). Les copies sont retirées par empreinte MD5.

**Ce que la page ne documente pas :** la fréquence d'images, la résolution et l'orientation des vidéos d'origine
(indispensables pour remettre x et y à la même échelle), l'angle de vue de la caméra, la version de MediaPipe, le
protocole d'étiquetage, le consentement, et pourquoi les fichiers comptent moins de sujets qu'annoncé.

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
