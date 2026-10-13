# Pipeline de bout en bout

> Outil pédagogique d'aide à l'analyse technique. Ce n'est pas un dispositif médical et il ne prédit aucune
> blessure. **Conçu comme aide pédagogique ; utilité non évaluée.**

## Schéma

```
 vidéo (mp4/mov)
   │
   ▼  src/pose.py ─ MediaPipe Pose Landmarker (33 points / image, en pixels)
   │                · plusieurs personnes → on garde la plus grande
   │                · visibilité < 0,5 → point absent ; trous ≤ 3 images interpolés
   │                · lissage Savitzky-Golay (7 images)
   ▼  src/angles.py ─ indicateurs image par image (2D)
   │                · profil : genou, hanche, tronc, tibia, profondeur, talon, cou
   │                · face   : ratio genoux/chevilles, bascule du bassin, asymétrie
   ▼  src/reps.py ─ répétitions = pics de la hauteur de hanche (find_peaks) + tempo
   │
   ▼  src/rules.py ─ règles de config.yaml (source + statut), filtrées par vue
   │                · défaut retenu si présent sur ≥ 2 répétitions (Myer et al., 2014)
   ▼
 résultats par répétition + synthèse vidéo + vidéo annotée (src/video.py)
```

Le même code (`src/analysis.py`) sert à l'application et aux scripts d'évaluation.

**Évaluation (phase B, décidée le 12/10/2026) :**
- **Analyse principale (option A) :** les seuils des règles restent figés (littérature). Un classifieur
  (régression logistique, forêt aléatoire) apprend le label good/bad à partir des **valeurs continues** des
  indicateurs. Il est validé en *leave-one-subject-out* : les 6 vidéos d'un sujet sont toujours dans le même pli,
  la normalisation est apprise dans chaque pli, et les hyperparamètres sont fixés à l'avance.
- **Analyse secondaire (option B), séparée :** les seuils sont recalibrés à l'intérieur de chaque pli
  leave-one-subject-out. Ces seuils sont étiquetés « calibrés sur le dataset », jamais « sourcés ».
- **Annotation en aveugle** des 50 vidéos de profil par l'auteur (label caché, ordre aléatoire fixe, aucune
  détection affichée). Elle permet d'évaluer chaque règle contre un type d'erreur.

## Limites déjà identifiées

1. **Contamination partielle de la conception.** Les indicateurs et leur façon de mesurer ont été conçus *après*
   avoir exploré les 25 sujets (phase A, `scripts/explore_rules.py`). En particulier, le passage à une mesure du
   talon relative au début de chaque répétition a été décidé en voyant que la mesure absolue se déclenchait sur
   100 % des vidéos good. Aucune donnée « jamais vue » ne reste pour tester ces choix. Les résultats de la phase B
   peuvent donc être légèrement optimistes, et toute recalibration (option B) est en partie contaminée.
2. **Désaccord de référentiel sur la profondeur.** La règle suit Myer et al. (2014) : hanche au moins au niveau du
   genou. La mesure a été vérifiée visuellement : elle est correcte. Mais les squats étiquetés « good » du dataset
   ne respectent souvent pas ce critère (92 % des vidéos good signalées). Le dataset applique un critère **non
   documenté**. Ce n'est pas une erreur de mesure, mais deux standards différents. Aucun des deux n'est une vérité
   absolue.
3. **Talons non mesurables de façon fiable à ces résolutions.**
   - Mesure absolue : le point talon de MediaPipe descend au point bas du squat, ce qui produisait des faux
     signaux.
   - Mesure relative au début de répétition : elle ne voit pas une personne déjà sur la pointe des pieds (sujet
     016 bad, vérifié visuellement), et elle se déclenche sur 80 % des good contre 76 % des bad.
   - Angle du pied (talon → pointe) : il ne sépare pas good et bad (p = 0,17).

   Résolutions du dataset : de 348x624 à 810x1440.
4. **Aucun écart good/bad en vue de profil.** Profondeur, talons, tronc − tibia et tête ne séparent pas good et
   bad. La seule différence significative (tronc − tibia, p = 0,013, sur 8 tests) va dans le sens **inverse** de
   la règle : les vidéos bad ont les genoux plus en avant. Seules les règles de face montrent un écart (valgus
   16 % contre 28 %, bassin 0 % contre 16 % des vidéos), non significatif.

5. **Annotation seulement en partie aveugle.** Avant d'annoter, l'auteur a vu les planches d'inspection et la
   description de certains sujets (ex. 016 bad sur la pointe des pieds, 004 et 008 good pas assez profonds). Il peut
   donc reconnaître certaines personnes malgré l'anonymisation (V01 à V50).

Autres limites, détaillées dans `docs/INSPECTION.md` et `docs/SOURCES.md` :
- 15 fps : tempo à ±0,07 s ;
- plusieurs personnes à l'image, cas hors du périmètre prévu par la fiche du modèle ;
- mesures 2D (projections) ;
- 25 sujets seulement ;
- un label par vidéo, sans type d'erreur ;
- inspection visuelle faite par un non-expert.

## Plan des phases B et C (mis à jour le 12/10/2026, cahier des charges « rapport de correction »)

L'utilisateur envoie une vidéo de squat (importée ou filmée sur son téléphone) et reçoit un rapport de correction
complet. Aucune vidéo « idéale » n'est générée. **Toute fiabilité affichée vient de l'évaluation réelle**
(leave-one-subject-out et annotation en aveugle), jamais d'une estimation a priori.

**Phase B : évaluation (faite le 12/10/2026, résultats plus bas)**
1. `src/features.py` : un seul module construit le vecteur d'indicateurs d'une vidéo (médianes par répétition). Il
   sert à l'entraînement ET à l'app, pour garantir les mêmes calculs.
2. `scripts/train.py` / `scripts/evaluate.py`, en étapes numérotées :
   - régression logistique et forêt aléatoire, en leave-one-subject-out ;
   - résultats par vue ;
   - précision, rappel, F1, matrice de confusion, IC par bootstrap **par sujet**.
3. **Probabilités et incertitude.** On évalue la calibration des probabilités en LOSO (score de Brier, courbe de
   fiabilité). Une zone « incertain » autour de 0,5 est proposée : sa largeur est un choix d'ingénierie, déclaré
   comme tel.
4. **Fiabilité de chaque règle** → `docs/results/rule_reliability.csv`, lu par l'app :
   - règles de profil : sensibilité et spécificité contre l'annotation en aveugle (n = 50 vidéos, avec IC) ;
   - règles de face et de diagonale : pas d'annotation, donc seulement les taux good/bad. Elles sont affichées
     « fiabilité non évaluée contre annotation ».
5. **Analyse secondaire (option B),** clairement séparée : seuils recalibrés en LOSO.
6. **Modèle publié dans le dépôt** (`models/*.joblib`, quelques Ko, appris sur des indicateurs dérivés, sans
   image). C'est nécessaire pour la version en ligne : `.gitignore` sera ajusté pour ce seul fichier.

**Phase C : application et rapport**
1. **Verdict global :** probabilité « bad » + zone d'incertitude + performance LOSO mesurée pour la vue choisie.
   Jamais formulé comme une certitude. La vue (face/profil/diagonale) est déclarée par l'utilisateur.
2. **Détail par répétition :** indicateurs, règles déclenchées, fiabilité de chaque règle (table de la phase B, plus
   les mentions « fiabilité réduite en diagonale » et « talons : indicateur peu fiable »).
   Le modèle donne un verdict **par vidéo** seulement : le dataset n'a pas de label par répétition.
3. **Correction écrite pour chaque défaut :**
   - texte en français simple, tiré de `config.yaml`, avec la source du seuil, son statut (« à valider par un
     expert » si besoin) et sa fiabilité mesurée ;
   - aucun conseil médical.
4. **Vidéo annotée :**
   - squelette et angles ;
   - surlignage des répétitions fautives (bandeau rouge pendant la répétition) ;
   - frise chronologique Plotly des répétitions et des défauts sous la vidéo.
5. **Comparaison côte à côte avec une vidéo good du dataset** (local uniquement, masquée si le dataset est absent,
   donc toujours en ligne) :
   - même vue ;
   - images du point bas mises côte à côte ;
   - avertissement : une vidéo « good » du dataset n'est pas un modèle idéal (par exemple, beaucoup ne respectent
     pas la profondeur de Myer).
6. **Export du rapport :**
   - **HTML autonome** en priorité (graphiques intégrés) ;
   - PDF via l'impression du navigateur, ou en option si le temps le permet ;
   - le rapport ne contient jamais d'image du dataset ;
   - la vidéo de l'utilisateur est traitée dans un dossier temporaire puis supprimée.
7. **Filmer dans l'app** (webcam, streamlit-webrtc) reste un bonus. Le parcours prévu est : filmer avec le téléphone,
   puis importer.

**Ce qui est prioritaire si le temps manque (délai : 2 semaines) :** 1 à 4 et l'export HTML sont indispensables.
La comparaison côte à côte, le PDF natif et la webcam sont optionnels.

## Protocole de la phase B (figé et committé AVANT la première exécution)

Ce protocole a été écrit et committé avant d'exécuter `train.py` et `evaluate.py` sur les données réelles. Le
commit le prouve (historique git). Toute modification ultérieure serait signalée comme telle.

| Élément | Choix fixé d'avance |
|---|---|
| Unité d'analyse | la vidéo (un label par vidéo) ; une ligne = médiane des répétitions de chaque indicateur |
| Indicateurs | ceux des règles applicables à la vue + rapport de tempo (`src/features.py`) |
| Modèles | un par vue ; régression logistique (C = 1) et forêt aléatoire (500 arbres, 3 vidéos min. par feuille) |
| Modèle de l'app | régression logistique (explicabilité), quel que soit le résultat ; la forêt sert de comparaison |
| Validation | leave-one-subject-out, 25 plis, les 6 vidéos d'un sujet dans le même pli |
| Prétraitement | imputation (médiane) et normalisation dans un Pipeline, apprises dans chaque pli |
| Hyperparamètres | aucune recherche ; valeurs de `config.yaml` (`evaluation`) |
| Décision | « bad » si probabilité > 0,5 ; zone « incertain » [0,35 ; 0,65] (choix d'ingénierie, non optimisé) |
| Détecteur à seuils (option A) | vidéo « bad » si ≥ 1 règle applicable à la vue retient un défaut (≥ 2 répétitions) |
| Vidéo sans répétition | « non analysable », exclue des métriques et comptée |
| IC | bootstrap percentile 95 %, 2000 tirages **de sujets** (graine 42) |
| Références | hasard (pile ou face, 10 000 tirages), « toujours bad », « toujours good », probabilité constante 0,5 (Brier = 0,25) |
| Test de permutation | régression logistique, 1000 permutations good/bad à l'intérieur de chaque paire (sujet, vue) |
| Calibration | score de Brier ; courbe de fiabilité en 5 groupes de taille égale |
| Accord annotateur / label | kappa de Cohen ; principal : « incertain » exclu ; sensibilité : « incertain » = désaccord / = accord |
| Règle vs annotation | défaut « présent » = noté sur quelques répétitions ou la plupart (principal) ; « la plupart » seulement (secondaire) ; correspondance : depth↔depth, heel_rise↔heel_rise, trunk_vs_tibia↔trunk_lean, head_position↔head |
| Option B (secondaire, script séparé) | seuil choisi dans chaque pli par l'indice de Youden, sens de la règle inchangé, égalité → seuil le plus proche de la littérature |

## Résultats de la phase B (12/10/2026)

Commandes : `python scripts/train.py`, puis `python scripts/evaluate.py` (≈ 4 min, dont le test de permutation),
puis `python scripts/evaluate_option_b.py`. Tableaux dans `docs/results/`, figures dans `docs/figures/`.
Aucune modification du protocole après exécution. Une vidéo est non analysable (0 répétition détectée :
sujet 007, good, diagonale) : 149 vidéos évaluées.

**Lecture rapide : les résultats sont faibles.** Le classifieur fait un peu mieux que le hasard, de façon
statistiquement détectable, mais ses probabilités ne sont pas plus fiables qu'un « 50 % » constant. Les règles à
seuils figés ne séparent pas good et bad, et ne détectent pas les défauts que l'annotateur voit.

### 1. Accord de l'annotateur avec le label du dataset (50 vidéos de profil)

| | label good | label bad |
|---|---|---|
| annoté « bonne » | 2 | 0 |
| annoté « mauvaise » | 11 | 24 |
| annoté « incertain » | 12 | 1 |

| Convention | n | Kappa de Cohen [IC 95 %] | Accord brut |
|---|---|---|---|
| « incertain » exclu (principale) | 37 | **0,19 [0,00 ; 0,44]** | 0,70 |
| « incertain » = désaccord (pire cas) | 50 | 0,04 [−0,08 ; 0,16] | 0,52 |
| « incertain » = accord (meilleur cas, artificiel) | 50 | 0,56 [0,36 ; 0,76] | 0,78 |

L'accord est faible. L'annotateur juge presque toutes les vidéos imparfaites (35 « mauvaise », 2 « bonne ») :
il a un standard plus strict que le dataset. Son doute se concentre sur les vidéos good (12 « incertain » sur 13).
Ni l'annotateur ni le dataset ne sont une vérité de référence : ce sont deux juges non experts (pour le dataset,
le protocole d'étiquetage n'est pas documenté).

Mais l'annotateur voit **plus de défauts dans les vidéos bad** (`annotation_by_label.csv`, sur 25 vidéos) :
profondeur 14 contre 7, talons 9 contre 3, tronc 19 contre 10, genoux en avant 20 contre 11, tête 22 contre 20.
Une différence visible à l'œil existe donc, au moins en partie (annotation seulement en partie aveugle, voir
limite 5).

### 2. Règles de profil contre l'annotation en aveugle (seuils figés)

Défaut « présent » = noté sur quelques répétitions ou la plupart. Détection = part des vidéos où l'annotateur voit
le défaut et où la règle le signale. Fausses alertes = part des vidéos où l'annotateur ne voit rien et où la règle
signale quand même.

| Règle | Présent (annotation) | Détection [IC 95 %] | Fausses alertes [IC 95 %] | Kappa |
|---|---|---|---|---|
| depth | 21/50 | 0,95 [0,85 ; 1,00] | 0,97 [0,90 ; 1,00] | −0,01 |
| heel_rise | 12/50 | 0,75 [0,40 ; 1,00] | 0,79 [0,67 ; 0,89] | −0,02 |
| trunk_vs_tibia | 29/50 | 0,62 [0,41 ; 0,81] | 0,57 [0,31 ; 0,81] | 0,05 |
| head_position | 42/50 | 0,24 [0,10 ; 0,39] | 0,00 (0/8)* | 0,09 |

\* Le bootstrap donne [0 ; 0] quand aucun événement n'est observé : c'est un artefact. Avec 0 sur 8 vidéos, la
borne haute exacte (Clopper-Pearson, sans tenir compte des sujets) est 0,37.

**Aucune règle n'est en accord avec l'annotateur** (kappa entre −0,02 et 0,09). Trois règles signalent presque
autant quand le défaut est absent que quand il est présent : elles ne transmettent aucune information. La règle de
la tête est l'inverse : elle ne se trompe pas quand elle signale (0/8), mais elle manque 3 défauts sur 4.

Attention : les définitions de l'annotation ne sont pas exactement celles des règles. Exemple : profondeur =
« hanche *nettement* au-dessus du genou » dans l'outil d'annotation, « hanche au moins au niveau du genou » chez
Myer. Une partie des fausses alertes de `depth` vient de cet écart de standard. Résultats avec la définition
secondaire (« la plupart des répétitions ») : mêmes conclusions (`rule_vs_annotation.csv`).

### 3. Performances contre le label du dataset (leave-one-subject-out)

Toutes vues confondues (149 vidéos), IC 95 % par bootstrap sur les sujets :

| Méthode | Précision | Rappel | F1 | Exactitude équilibrée | AUC |
|---|---|---|---|---|---|
| Régression logistique (modèle de l'app) | 0,62 [0,55 ; 0,71] | 0,57 [0,47 ; 0,68] | 0,60 [0,52 ; 0,67] | **0,61 [0,55 ; 0,67]** | **0,60 [0,54 ; 0,67]** |
| Forêt aléatoire | 0,51 [0,45 ; 0,57] | 0,55 [0,43 ; 0,67] | 0,53 [0,45 ; 0,60] | 0,51 [0,45 ; 0,57] | 0,55 [0,48 ; 0,63] |
| Détecteur à seuils (règles, option A) | 0,55 [0,51 ; 0,59] | 0,76 [0,69 ; 0,83] | 0,64 [0,59 ; 0,68] | 0,56 [0,51 ; 0,62] | – |
| Trivial : toujours « bad » | 0,50 | 1,00 | **0,67** | 0,50 | – |
| Trivial : toujours « good » | non définie | 0,00 | 0,00 | 0,50 | – |
| Hasard (pile ou face) | 0,50 [0,42 ; 0,58] | 0,49 [0,39 ; 0,61] | 0,50 [0,41 ; 0,59] | 0,50 [0,42 ; 0,58] | 0,50 |

Par vue (exactitude équilibrée [IC 95 %]) :

| Vue | Régression logistique | Forêt aléatoire | Détecteur à seuils | Matrice du détecteur (TN FP / FN TP) |
|---|---|---|---|---|
| Profil (n = 50) | 0,58 [0,48 ; 0,68] | 0,46 [0,34 ; 0,58] | 0,52 [0,50 ; 0,56] | 1 24 / 0 25 |
| Face (n = 50) | 0,64 [0,54 ; 0,74] | 0,56 [0,44 ; 0,70] | 0,60 [0,48 ; 0,72] | 21 4 / 16 9 |
| Diagonale (n = 49) | 0,61 [0,50 ; 0,72] | 0,51 [0,39 ; 0,63] | 0,56 [0,48 ; 0,66] | 5 19 / 2 23 |

Matrices complètes : `docs/figures/confusion_matrices.png`.

Ce qu'il faut retenir :
- **Le F1 trompe ici.** « Toujours bad » obtient F1 = 0,67, mieux que tous les modèles, sans rien regarder. Avec
  des classes équilibrées, l'exactitude équilibrée et l'AUC sont les comparaisons honnêtes (référence : 0,5).
- **Régression logistique : au-dessus du hasard, mais de peu.** Test de permutation (1000 permutations
  good/bad à l'intérieur de chaque paire sujet-vue, LOSO refait à chaque fois) : AUC observée 0,60 contre 0,57 au
  95e centile sous l'hypothèse nulle, p = 0,002 ; exactitude équilibrée 0,61, p = 0,002. Par vue, les IC touchent
  ou incluent 0,5 : on ne peut pas dire quelle vue fonctionne le mieux.
- **La forêt aléatoire ne fait pas mieux que le hasard** (IC incluant 0,5). Avec 48 vidéos par entraînement, un
  modèle plus flexible apprend surtout le bruit.
- **Le détecteur à seuils de profil équivaut à « toujours bad »** : il signale 24 vidéos good sur 25. De face,
  il est spécifique (21/25 good non signalées) mais manque 16 bad sur 25.

**Ce que la régression logistique a appris** (`feature_importance.png`, coefficients standardisés, moyenne des
25 plis) : le tempo est l'indicateur le plus utilisé dans les trois vues (descente relativement plus courte →
« bad »). De profil : moins de profondeur → « bad » (sens de Myer) ; tronc moins penché que le tibia → « bad »
(sens **inverse** de la règle, comme en phase A) ; tête → « bad » (sens de Myer). Le valgus change de signe entre
face (+0,13) et diagonale (−0,14) : il n'apporte pas d'information stable. Les coefficients varient peu d'un pli à
l'autre, mais c'est attendu (deux plis partagent 23 sujets sur 24) : cela ne prouve pas qu'ils se généralisent.

### 4. Fiabilité des probabilités

| Modèle | Brier (toutes vues) [IC 95 %] | Référence « 0,5 constant » | Vidéos en zone « incertain » [0,35 ; 0,65] | Exactitude hors zone |
|---|---|---|---|---|
| Régression logistique | 0,251 [0,229 ; 0,274] | 0,250 | 94/149 (63 %) | 0,64 (n = 55) |
| Forêt aléatoire | 0,263 [0,243 ; 0,284] | 0,250 | 90/149 (60 %) | 0,59 (n = 59) |

**Les probabilités ne sont pas plus fiables que de répondre « 50 % » à chaque vidéo** (Brier égal à la
référence). La courbe de fiabilité (`reliability_curve.png`) n'est pas monotone : le groupe prédit à 0,49 contient
31 % de bad, celui prédit à 0,42 en contient 37 %, celui à 0,30 en contient 50 %. Conséquence pour l'app : la
probabilité ne doit pas être présentée comme « X % de chances d'être mauvais ». Le verdict sera « incertain »
pour environ 2 vidéos sur 3, et même hors de la zone, il se trompe environ 1 fois sur 3.

### 5. Analyse secondaire, option B (seuils recalibrés en LOSO) : séparée de l'analyse principale

Seuils choisis dans chaque pli (indice de Youden), appliqués au sujet retiré. **Ces seuils sont « calibrés sur le
dataset », pas « sourcés », et ils ne vont pas dans `config.yaml`.** Ils sont aussi en partie contaminés (limite 1).

| Vue | Méthode | Exactitude équilibrée [IC 95 %] | Seuil Myer → seuil calibré (médiane des plis) |
|---|---|---|---|
| Profil | depth | 0,62 [0,50 ; 0,74] | 0 → −0,43 (hanche à 0,43 cuisse **au-dessus** du genou) |
| Profil | heel_rise | 0,44 [0,36 ; 0,52] | 0,15 → 0,35 (instable : 0,15 à 0,37) |
| Profil | trunk_vs_tibia | 0,54 [0,44 ; 0,64] | 15° → 18,9° |
| Profil | head_position | 0,48 [0,36 ; 0,60] | 30° → 14,4° |
| Profil | ≥ 1 règle | 0,56 [0,50 ; 0,62] | – |
| Face | ≥ 1 règle | 0,64 [0,50 ; 0,78] | valgus 1 → 0,98 ; bassin 10° → 5,2° |
| Diagonale | ≥ 1 règle | 0,69 [0,58 ; 0,80] | valgus 1 → 1,02 ; bassin 10° → 11,8° |
| Toutes | ≥ 1 règle | 0,63 [0,56 ; 0,71] | – |

Même recalibrées, les règles de profil restent au niveau du hasard, sauf la profondeur. Le seuil de profondeur que
choisissent les données (hanche nettement au-dessus du genou) confirme le désaccord de référentiel (limite 2) : le
dataset tolère des squats bien moins profonds que Myer. Le meilleur chiffre (diagonale, 0,69) est obtenu parmi
12 combinaisons testées, sur des indicateurs conçus après avoir vu les données : c'est une piste, pas un résultat.

### 6. Table de fiabilité des règles pour l'app (`docs/results/rule_reliability.csv`)

| Règle | Vue | Signalée sur les vidéos good | … sur les bad | Fiabilité affichée dans l'app |
|---|---|---|---|---|
| depth | profil | 92 % | 100 % | détection 0,95 / fausses alertes 0,97 contre l'annotation |
| heel_rise | profil | 80 % | 76 % | détection 0,75 / fausses alertes 0,79 contre l'annotation |
| trunk_vs_tibia | profil | 60 % | 60 % | détection 0,62 / fausses alertes 0,57 contre l'annotation |
| head_position | profil | 24 % | 16 % | détection 0,24 / fausses alertes 0/8 contre l'annotation |
| knee_valgus | face | 16 % | 28 % | non évaluée contre annotation |
| knee_valgus | diagonale | 25 % | 48 % | non évaluée contre annotation (fiabilité réduite) |
| hip_shift | face | 0 % | 16 % | non évaluée contre annotation |
| hip_shift | diagonale | 75 % | 84 % | non évaluée contre annotation (fiabilité réduite) |

### Limites ajoutées par la phase B

6. **Pas de vérité de référence.** Le label du dataset et l'annotation de l'auteur sont en faible accord
   (kappa 0,19). Toute « performance » mesure l'accord avec un juge non expert, pas la qualité réelle du squat.
7. **Le signal du classifieur vient beaucoup du tempo**, un indicateur sans règle, mesuré à 15 fps (±0,07 s). Les
   vidéos bad seraient-elles simplement faites plus vite ? Le tempo fait partie de la grille de Myer, mais ce n'est
   pas ce que l'utilisateur attend d'un « verdict technique ».
8. **Probabilités non fiables** (Brier = référence constante). Le verdict de l'app doit être présenté comme une
   tendance faible, avec la performance LOSO mesurée à côté.
9. **Petits effectifs.** 25 sujets ; IC larges (souvent ±0,10 à ±0,20) ; 8 vidéos seulement sans défaut de tête
   selon l'annotateur. Les comparaisons par vue ne sont pas concluantes.
10. **Un seul annotateur, non expert**, sans mesure de reproductibilité (pas de deuxième annotation).

## Validation externe (en cours, étapes 1-2 faites le 13/10/2026)

Deux jeux ouverts (CC BY 4.0), jamais commités, extraits hors du dépôt (`config.yaml` → `external`).
Attribution et écarts vérifiés : `docs/DATA_LICENSES.md`. Description : `python scripts/inspect_external.py`
(sorties dans `outputs/external/`, planches d'images dans `Datasets\extracted\_inspection\`, hors du dépôt).

### Étape 2 : compatibilité avec le pipeline (`src/angles.py`, `src/rules.py`)

**A. Teng 2025 (3806 images de profil, Good / Bad Back / Bad Heel)**

Ce qui change dans le pipeline :
- `src/pose.py` : extraction image par image (`extract_pose_images`, mode IMAGE, sans suivi). Même choix de
  personne et mêmes coordonnées en pixels. Le nettoyage temporel (interpolation, lissage) ne s'applique pas.
- `src/angles.py` : utilisable tel quel. La résolution de chaque image est connue, donc les angles sont corrects.
- `src/reps.py` : inapplicable (pas de répétition). Toutes les règles sont définies sur une répétition
  (`window`, `reference: start`, défaut retenu sur ≥ 2 répétitions). Les appliquer à **une image** est une
  adaptation à déclarer, pas la règle d'origine.

| Règle | Évaluable sur A ? | Pourquoi |
|---|---|---|
| `trunk_vs_tibia` | **oui, avec adaptation** (une image au lieu de la moitié basse d'une répétition) | angles à la verticale, mesurables sur une image ; seule règle liée à une classe (Bad Back). Mais Bad Back = « spinal misalignment » selon Zenodo : notre règle mesure l'**inclinaison** du tronc par rapport au tibia, pas la courbure du dos (MediaPipe n'a aucun point sur la colonne). C'est un indicateur approché |
| `heel_rise` | **non** | la règle mesure la montée du talon par rapport à la posture debout du début de répétition ; aucune image debout n'est associée. Une hauteur absolue ou l'angle du pied serait un **nouvel** indicateur, non sourcé (l'angle du pied a déjà été écarté en phase A, p = 0,17). La classe Bad Heel ne peut donc pas être évaluée par nos règles |
| `depth` | calculable, mais **sans label correspondant** | aucune classe de A n'est définie par la profondeur ; on ne sait pas si l'image est le point bas. Taux de signalement par classe seulement (descriptif) |
| `head_position` | **non** | référence = posture debout du début de répétition |
| `knee_valgus`, `hip_shift` | **non** | règles de face ; A est de profil |
| tempo, classifieur | **non** | pas de répétition ni de durée ; le classifieur utilise les médianes par répétition et le tempo |

**Phase du mouvement** (angle du genou, descriptif) : presque toutes les images sont prises près du point bas.
Genou < 90° pour 95 % des Good, 98 % des Bad Heel, mais seulement 66 % des Bad Back (31 % entre 90° et 120°).
Médiane : Good 70°, Bad Heel 59°, Bad Back 84°. **Les Bad Back sont moins profonds** : comme l'inclinaison du tronc
dépend de la profondeur, c'est un facteur de confusion pour `trunk_vs_tibia`.

**Détection MediaPipe** (points nécessaires visibles à ≥ 0,5) : les échecs dépendent de la classe plus que de
la résolution. Personne non détectée : Bad Back 2,6 %, Bad Heel 0,1 %, Good 0 %. Points du profil incomplets : Bad
Back 5,6 %. Aucun échec en 720×720 ni en 3456×3456 ; les échecs (1,1 à 1,6 %) se concentrent en 1072×1080 et
1080×1080, presque tous chez les Bad Back. Les postures très penchées sont plus difficiles à détecter. Ces images
seront comptées comme « non mesurables », jamais retirées en silence.

**Lien résolution / classe (limite) :** 1080×1080 = 257 Bad Back, 102 Bad Heel, 17 Good ; 1072×1080 = 828, 859,
1038. Nos règles sont figées, donc elles ne peuvent pas « apprendre » la résolution. Mais la résolution suit la
prise de vue (cadrage, distance), qui peut changer la mesure.

**Corrélation entre images voisines : proposition.**
- Constat : les fichiers sont numérotés dans l'ordre du temps. Deux images voisines diffèrent de 0,004 en
  moyenne (vignettes 32×32), contre 0,187 pour deux images tirées au hasard. Ce sont des images de vidéos : 3806
  images ne sont pas 3806 observations.
- **Blocs consécutifs** (nouvelle image quand la différence dépasse un seuil) : 248, 119, 70 ou 53 blocs pour un
  seuil de 0,02, 0,03, 0,05 ou 0,08. Ce découpage est fragile : dans `test/Bad Back`, la numérotation alterne entre
  deux prises de vue (blocs d'une image), et une même prise de vue est coupée en plusieurs blocs.
- **Scènes** (regroupement par ressemblance globale, vignette couleur 16×16, lien moyen) : avec 10 scènes, 8
  contiennent les 3 classes. À l'œil, une scène ≈ une personne dans un lieu avec une tenue (planche
  `A_teng_scenes_10.jpg`), avec des erreurs : 2 scènes mélangent deux personnes.
- **Proposition (à figer à l'étape 3) :** unité statistique = la scène (10 scènes). IC par bootstrap **sur les
  scènes**, avec une analyse de sensibilité à 6, 8, 12, 15 scènes et aux blocs consécutifs (seuil 0,05). Les
  effectifs en images sont donnés pour information, sans IC au niveau de l'image.
- **Ce qui reste impossible :** un IC par sujet (aucun identifiant) ; vérifier qu'une scène correspond à un sujet ;
  savoir combien de personnes différentes il y a (à l'œil : 8 à 10 tenues ou lieux, pas un comptage fiable) ;
  vérifier l'indépendance entre train et test (les mêmes scènes sont dans les deux, donc on les fusionne).

**B. Dai 2026 v2 (344 fichiers CSV uniques, 28 sujets)**

Repère et unités :
- coordonnées MediaPipe **normalisées** : x divisé par la largeur, y par la hauteur de la vidéo ; y vers le bas
  (nez au-dessus des chevilles sur 100 % des images), comme dans le pipeline. `v` = visibilité (même seuil 0,5) ;
  `z` n'est pas utilisé ;
- **le pipeline travaille en pixels** (x et y à la même échelle). Il faut donc multiplier x par le rapport
  largeur/hauteur, qui n'est pas documenté (`src/external.py`, `to_pose_series(..., aspect=...)`, jamais choisi
  en silence) ;
- **un fichier = une répétition** (343 fichiers sur 344 ont un seul point bas ; 99 % commencent et 97 % finissent
  debout). La règle « défaut retenu sur ≥ 2 répétitions » ne peut pas se déclencher sur un fichier ;
- fréquence d'images inconnue : seuls les **durées** (tempo) en dépendent. Le lissage (7 images) et le
  comblement des trous (3 images) sont définis en images, donc applicables, mais leur durée réelle est inconnue.
  5 fichiers ont des trous dans la numérotation (images sautées) ;
- vue : probablement de profil (écart des hanches en x faible devant la hauteur du tronc, indicatif). Le côté
  visible dépend de la classe : jambe droite pour les 10 GoodMorning et 32 Shallow sur 38, partagé pour Standard
  (161 gauche, 135 droite). Pour un même sujet, les classes sont parfois filmées de côtés opposés : Shallow à droite
  et Standard à gauche pour les sujets 03, 05 et 06 ; GoodMorning à droite et le reste à gauche pour le sujet 01.
  **Les prises de vue diffèrent selon la classe** : c'est un facteur de confusion possible.

**Rapport largeur/hauteur : hypothèses testées.**
- Test : la longueur réelle de la cuisse et du tibia ne change pas pendant le squat ; on cherche le rapport qui
  la rend la plus stable (`src/external.py`).
- **Validation sur Mendeley** (résolution connue, 50 vidéos de profil, vrai rapport 0,5625) : erreur médiane
  +0,05 (de −0,08 à +0,17). La méthode **échoue de face et en diagonale** (+0,7 à +0,9), car le mouvement sort du
  plan de l'image.
- Résultat sur B : rapport estimé médian **2,19** (10-90 % : 1,91 à 2,69) ; GoodMorning 1,91, Standard 2,17,
  Shallow 2,46 (dispersion forte, jusqu'à 3,9). Instabilité des longueurs : 9:16 → 0,53 ; 3:4 → 0,45 ; 1:1 → 0,36 ;
  4:3 → 0,25 ; 16:9 → **0,14** (le plus plausible des formats courants).
- Lecture : les formats **portrait (9:16, 3:4) sont exclus**. **16:9 (paysage) est l'hypothèse la plus plausible,
  mais pas confirmée** : aucun format courant ne dépasse 1,78, donc les valeurs au-delà indiquent que l'hypothèse
  de la méthode (mouvement dans le plan, vue de profil pure) n'est pas respectée, comme en vue diagonale sur
  Mendeley. Le biais varie avec la classe (Shallow plus élevé).
- Proposition (à figer à l'étape 3) : hypothèse principale 16:9, et analyse de sensibilité à 4:3 et 1:1 pour
  toute règle dépendant du rapport. Une conclusion n'est retenue que si elle tient sous les trois hypothèses.

| Règle | Évaluable sur B ? | Dépend du rapport largeur/hauteur ? |
|---|---|---|
| `depth` | **oui** ; correspond à la classe Shallow | **non** : la règle ne regarde que le **signe** de (y hanche − y genou), seuil 0. Diviser par la longueur de cuisse ne change pas le signe. Évaluable sans hypothèse |
| `trunk_vs_tibia` | oui ; correspond le mieux à GoodMorning (tronc très penché) | **oui** : angles à la verticale. Sous hypothèses seulement. GoodMorning = 1 sujet : pas de conclusion possible |
| `heel_rise` | mesurable (référence = début de la répétition, présent dans chaque fichier), **sans classe correspondante** | **oui** (longueur du pied surtout horizontale). Seulement le taux de signalement sur Standard (fausses alertes probables), sous hypothèses |
| `head_position` | mesurable, sans classe correspondante | **oui** (angle). Taux de signalement seulement |
| `knee_valgus`, `hip_shift` | **non** | règles de face ; B est de profil |
| tempo, classifieur | **non** | fréquence d'images inconnue ; le classifieur a besoin du tempo |

**Éléments facultatifs, en fin de planning (après la phase C, si le temps le permet) :** un second annotateur en
aveugle sur les 50 vidéos de profil (kappa inter-juges) et `scripts/sensitivity_synthetic.py` (sensibilité des
règles sur des squats synthétiques).

Niveau de décision : une répétition par fichier. Deux options à figer à l'étape 3 : (a) verdict **par
répétition** (la règle telle qu'appliquée à chaque répétition, sans le critère « ≥ 2 répétitions ») ; (b)
regrouper les fichiers d'un même sujet et d'une même classe et appliquer « ≥ 2 répétitions ». Effectifs v2 :
Standard 296 fichiers / 28 sujets, Shallow 38 / 6 sujets (01 à 06, qui ont aussi du Standard), GoodMorning 10 /
1 sujet.

### Étape 3 : protocole de la validation externe (figé et committé AVANT toute exécution, 13/10/2026)

Ce protocole est committé avant d'écrire et d'exécuter le script d'évaluation (`scripts/evaluate_external.py`).
Toute déviation sera signalée comme telle dans les résultats. Décisions validées par l'auteur le 13/10/2026.

**1. Ce qui est figé**
- Règles, seuils, fenêtres, sens et paramètres techniques : `config.yaml` tel qu'au commit de ce protocole.
  **Aucune recalibration sur A ni sur B, aucun classifieur** (le classifieur a besoin du tempo, indisponible).
- Clusters de A (scènes et blocs) : calculés par `scripts/inspect_external.py` (commit `5545dc2`), sans regarder
  aucune règle ; enregistrés dans `outputs/external/teng_images.csv`.
- Mêmes définitions qu'en phase B : seuil de visibilité 0,5 ; un point sous ce seuil est absent.

**2. Métriques, pour chaque couple (règle, classe cible)**
- **Détection** = part des unités de la classe cible signalées par la règle.
- **Fausses alertes** = part des unités de la classe de référence signalées (A : Good ; B : Standard).
- **Exactitude équilibrée** = (détection + 1 − fausses alertes) / 2. Hasard = 0,5.
- Secondaire : kappa de Cohen entre le signalement et la classe (cible contre référence) ; taux de signalement sur
  les autres classes (descriptif).
- **IC 95 %** : bootstrap percentile, 2000 tirages, graine 42 (`src/metrics.py`). On tire des **clusters** avec
  remise, en gardant toutes leurs unités : scènes pour A, sujets pour B. Jamais d'IC au niveau de l'image ou du
  fichier.

**3. Critère décidé d'avance pour ne pas conclure (effectifs trop petits)**

Le nombre de clusters compte, pas le nombre d'images ou de fichiers. Une métrique est classée selon le nombre de
clusters qui contiennent la classe concernée :

| Clusters | Statut | Ce qui est rapporté |
|---|---|---|
| ≥ 10 | **concluant** | estimation + IC ; « sépare mieux que le hasard » seulement si l'IC de l'exactitude équilibrée est entièrement au-dessus de 0,5 ; « sens inverse » s'il est entièrement en dessous ; sinon « aucune séparation démontrée » |
| 5 à 9 | **indicatif** | estimation + IC, marqués « indicatif, pas de conclusion » |
| < 5 | **descriptif** | comptes seulement, sans IC ni conclusion |

L'exactitude équilibrée prend le statut le plus faible de ses deux composantes. Pourquoi 10 ? C'est un choix
d'ingénierie, conservateur et non sourcé. Avec peu de clusters, le bootstrap percentile donne des IC trop étroits
(problème décrit par Cameron, Gelbach et Miller, 2008, *Review of Economics and Statistics* 90(3):414-427,
doi:10.1162/rest.90.3.414). Conséquences connues **avant** l'exécution (comptes de clusters, aucune règle
calculée) :
- A, 10 scènes : Bad Back présent dans 10 scènes (détection concluante), Good dans 9 (fausses alertes et
  exactitude équilibrée indicatives) ;
- B : Standard 28 sujets (fausses alertes concluantes) ; **Shallow 6 sujets (indicatif)** ; **GoodMorning 1 sujet
  (descriptif seulement)**.

**4. Unités non mesurables : comptées, jamais retirées en silence**
- Une unité est « non mesurable » pour une règle si sa valeur ne peut pas être calculée : personne non détectée,
  point nécessaire sous le seuil de visibilité, ou (B) aucune répétition détectée.
- Pour chaque couple (règle, classe), on publie le nombre d'unités mesurables et non mesurables.
- Analyse principale : métriques sur les unités mesurables. Sensibilité : non mesurable = « non signalé »
  (c'est ce que verrait l'utilisateur de l'app : aucun défaut affiché).

**5. Jeu A (Teng 2025, images)**
- Unité : l'image. Les dossiers train et test sont fusionnés : on n'entraîne rien, et les mêmes scènes sont dans
  les deux. La classe vient du nom du dossier, sans tenir compte de la casse.
- Pipeline : points du cache `outputs/external/teng_landmarks.npz` (MediaPipe, mode IMAGE), visibilité < 0,5 →
  absent, puis `compute_indicators` sur une série d'une image. Pas d'interpolation ni de lissage (image unique).
- **Adaptation déclarée :** la règle est appliquée à la valeur de l'indicateur sur **une image**, et non à
  l'agrégat (`max`) sur une fenêtre de répétition, sans le critère « ≥ 2 répétitions ». Ce n'est pas la règle
  d'origine.
- Couples évalués :
  - `trunk_vs_tibia` → cible Bad Back, référence Good (analyse principale de A) ;
  - Bad Heel : **non évaluable** (aucune règle sourcée ne le mesure, voir étape 2). Le taux de signalement de
    `trunk_vs_tibia` et de `depth` sur Bad Heel est donné à titre descriptif ;
  - `depth` : taux de signalement par classe, descriptif seulement (aucune classe ne correspond).
- Clusters : **10 scènes (principal)**. Sensibilité : 6, 8, 12 et 15 scènes, et blocs consécutifs (seuil 0,05).
  Dans ce dernier découpage, aucun bloc ne contient à la fois Bad Back et Good. La règle des 10 clusters
  s'applique à chaque découpage.
- **Facteur de confusion : la profondeur.** Les Bad Back sont moins profonds (angle du genou médian 84°, contre
  70° pour Good), et l'inclinaison du tronc dépend de la profondeur. Analyse secondaire prévue : métriques par
  tranche d'angle du genou (< 90°, 90-120°), en gardant le même critère d'effectif. Si l'écart entre Bad Back et
  Good disparaît dans les tranches, il est attribué à la profondeur et non au dos.
- Limite rappelée : Bad Back = « spinal misalignment » selon Zenodo. `trunk_vs_tibia` mesure l'inclinaison du
  tronc, pas la courbure du dos.

**6. Jeu B (Dai 2026 v2, CSV)**
- Unité : le fichier (= une répétition). Copies identiques retirées par MD5 : 344 fichiers.
- Pipeline : `to_pose_series(aspect=…)`, puis `analyze_pose` exactement comme dans l'app (nettoyage, lissage sur
  7 images, segmentation des répétitions, règles). La fréquence d'images, inconnue, est fixée à 30 images/s par
  convention : elle n'entre dans aucune règle, seulement dans les durées, qui ne sont pas rapportées.
- **Hypothèses de format** : 16:9 (principale), 4:3 et 1:1 (sensibilité). **Une conclusion n'est retenue que si
  elle est la même sous les trois.**
- **(a) Analyse principale : verdict par répétition.** Le fichier est signalé si sa répétition dépasse le seuil.
  Sans critère « ≥ 2 répétitions », puisqu'il n'y a qu'une répétition. Si la segmentation trouve plusieurs
  répétitions (1 fichier sur 344 attendu), le fichier est signalé si au moins une l'est, et on le compte. Si elle
  n'en trouve aucune, le fichier est non mesurable (point 4).
- **(b) Analyse secondaire : regroupement par (sujet, classe).** Le groupe est signalé si au moins 2 répétitions
  le sont (`min_reps_for_video_error`), comme une vidéo du dataset principal. Un groupe avec moins de 2
  répétitions mesurables est non mesurable.
- **Sensibilité « fichier = répétition entière » :** début = première image, point bas = hanche la plus basse,
  fin = dernière image, au lieu de la segmentation automatique. La segmentation dépend en effet du format, via la
  longueur de cuisse qui sert d'unité.
- Bootstrap : **par sujet (28 sujets)** dans (a) et (b).
- Couples évalués :
  - **`depth` → cible Shallow, référence Standard : résultat mis en avant.** La règle ne dépend que du signe de
    (y hanche − y genou), donc pas du format. On le vérifie en comparant les trois hypothèses : toute différence
    viendrait de la segmentation et serait rapportée. Statut attendu : indicatif (6 sujets Shallow) ;
  - `trunk_vs_tibia` → cible GoodMorning, référence Standard : **descriptif seulement** (1 sujet) ;
  - `heel_rise`, `head_position` : fausses alertes sur Standard seulement (aucune classe cible), sous les trois
    formats ;
  - `knee_valgus`, `hip_shift`, tempo, classifieur : non évalués (étape 2).
- **Confusion entre côté filmé et classe.**
  - Le côté visible est publié par classe.
  - Analyse secondaire « mêmes sujets » : référence = les répétitions Standard des 6 sujets qui ont aussi du
    Shallow (01 à 06), pour comparer chaque personne à elle-même.
  - Description séparée des sujets filmés du même côté dans les deux classes (01, 02, 04) et du côté opposé (03,
    05, 06). Effectifs trop petits pour conclure : descriptif.
  - Pour GoodMorning, sujet unique filmé de l'autre côté : aucune conclusion possible.

**7. Sorties prévues**
- `docs/results/external_*.csv` (métriques, IC, statut, effectifs, non mesurables) et `docs/figures/external_*.png`.
- Une table de fiabilité étendue lue par l'app : `docs/results/rule_reliability.csv`, avec une colonne de
  provenance (Mendeley/annotation, Teng, Dai). Seules les métriques au statut « concluant » sont présentées comme
  telles dans l'app ; les autres sont étiquetées « indicatif » ou « descriptif ».
- Mise à jour de cette page (résultats, limites, question 15 du jury). Aucune image ni coordonnée brute publiée.

## Questions probables du jury

**1. Votre modèle fait-il mieux que le hasard ?**
Oui, mais de peu. AUC 0,60 [0,54 ; 0,67] et exactitude équilibrée 0,61 en leave-one-subject-out, p = 0,002 au
test de permutation. Ses probabilités ne sont pas mieux calibrées qu'un 50 % constant (Brier 0,251 contre 0,250).
C'est un signal faible, pas un outil de décision.

**2. Pourquoi le F1 de « toujours bad » (0,67) est-il supérieur à celui de votre modèle (0,60) ?**
Le F1 ne regarde que la classe positive et ignore les vrais négatifs. Prédire « bad » partout donne un rappel de
1 et une précision de 0,5, d'où F1 = 0,67. Avec 50 % de bad, c'est l'exactitude équilibrée et l'AUC qu'il faut
comparer au hasard (0,5). J'ai gardé le F1 parce qu'il était demandé, mais il ne sert pas à conclure.

**3. Pourquoi leave-one-subject-out et pas une validation croisée classique ?**
Chaque sujet a 6 vidéos. Avec un découpage aléatoire, une vidéo du sujet serait à l'entraînement et une autre au
test : le modèle reconnaîtrait la personne (morphologie, caméra) plutôt que la technique. Le LOSO met les 6
vidéos d'un sujet dans le même pli. Un test automatique vérifie que le sujet testé n'est jamais vu à
l'entraînement.

**4. Comment avez-vous évité la fuite d'information ?**
Quatre précautions : LOSO par sujet ; imputation et normalisation dans un Pipeline réappris à chaque pli ;
hyperparamètres fixés d'avance sans recherche ; protocole committé avant la première exécution (commit
`6e89270`). Il reste une fuite que je ne peux pas éliminer : les indicateurs ont été conçus après avoir exploré les
25 sujets (limite 1).

**5. Pourquoi les intervalles de confiance par bootstrap sur les sujets ?**
Les 6 vidéos d'un même sujet ne sont pas indépendantes. Rééchantillonner les vidéos ferait comme si on avait 150
observations indépendantes et donnerait des IC trop étroits. On tire 25 sujets avec remise et on garde toutes
leurs vidéos.

**6. Vos règles viennent de la littérature. Pourquoi ne fonctionnent-elles pas ?**
Trois raisons. (1) Le dataset applique un autre standard : 92 % des squats good n'atteignent pas la profondeur
de Myer. (2) Certaines mesures ne sont pas fiables à ces résolutions (talons). (3) La règle tronc−tibia va dans le
sens inverse des données. Une règle de la littérature n'est pas automatiquement valide pour un autre dataset,
une autre caméra et un autre juge.

**7. Votre propre annotation contredit le dataset (kappa 0,19). Qui a raison ?**
Personne ne peut le dire. Ni le protocole du dataset ni mon expertise ne sont documentés comme référence. Je juge
35 vidéos sur 50 « mauvaises », donc mon standard est plus strict. Mais je vois plus de défauts dans les vidéos bad
(par exemple tronc 19 contre 10) : nous ne sommes pas totalement indépendants. Il faudrait un expert certifié, et
deux annotateurs pour mesurer la reproductibilité.

**8. Pourquoi ne pas avoir recalibré les seuils pour améliorer les résultats ?**
Je l'ai fait, mais en analyse secondaire (option B), en LOSO pour ne pas tricher. Même recalibrées, les règles de
profil restent au hasard sauf la profondeur, et le seuil choisi par les données (hanche à 0,43 cuisse au-dessus du
genou) n'est pas un critère technique défendable. Ces seuils sont étiquetés « calibrés sur le dataset », jamais
« sourcés ».

**9. Que signifie la probabilité affichée par l'app ?**
La probabilité que le label du dataset soit « bad », pas la probabilité que le squat soit dangereux ou mauvais.
Elle est mal calibrée : l'app affiche donc un verdict en trois niveaux (good / incertain / bad), la performance
mesurée (exactitude équilibrée 0,61), et « incertain » pour environ 2 vidéos sur 3.

**10. Pourquoi la zone « incertain » est-elle [0,35 ; 0,65] ?**
C'est un choix d'ingénierie, fixé avant les résultats et non optimisé. Optimiser la largeur sur les mêmes données
aurait été une fuite. Hors de la zone, l'exactitude est de 0,64 (55 vidéos), à peine meilleure.

**11. Pourquoi une régression logistique plutôt qu'un modèle plus puissant ?**
Choix fixé d'avance pour l'explicabilité : chaque coefficient se lit (« moins de profondeur → plus de chances
d'être bad »). Avec 48 vidéos d'entraînement par vue, un modèle flexible surapprend : la forêt aléatoire fait
moins bien (AUC 0,55, compatible avec le hasard).

**12. Qu'est-ce que le modèle a vraiment appris ?**
Surtout le tempo (descente relativement plus courte → bad), puis la profondeur et le rapport tronc/tibia. Le sens
du tronc/tibia contredit Myer. Le valgus change de signe selon la vue. Le modèle a donc appris des corrélations du
dataset, pas une connaissance biomécanique.

**13. Pourquoi une vidéo est-elle exclue ?**
Le sujet 007 good diagonale : aucune répétition détectée. Elle est comptée comme « non analysable » et exclue des
métriques, comme le prévoyait le protocole. L'app fera de même : pas de verdict sans répétition.

**14. Votre outil est-il utile ?**
Je ne l'ai pas démontré. Les résultats montrent qu'il ne reproduit ni le label du dataset ni mon annotation de
façon fiable. Il reste conçu comme aide pédagogique : il montre des mesures, leur source, et leur fiabilité réelle.
Son utilité devrait être évaluée par une étude avec des utilisateurs et un expert.

**15. Que feriez-vous avec plus de temps ?**
(1) Faire annoter les vidéos par un coach certifié et par un deuxième annotateur (accord inter-juges).
(2) Filmer à 30 ou 60 fps et en plus haute résolution, pour le tempo et les talons. (3) Tester sur des sujets
jamais vus pendant la conception, pour supprimer la contamination. (4) Obtenir des labels par répétition et par
type d'erreur. (5) Évaluer l'utilité pédagogique avec de vrais utilisateurs.
