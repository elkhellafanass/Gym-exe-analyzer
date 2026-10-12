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

Autres limites, détaillées dans `docs/INSPECTION.md` et `docs/SOURCES.md` :
- 15 fps : tempo à ±0,07 s ;
- plusieurs personnes à l'image, cas hors du périmètre prévu par la fiche du modèle ;
- mesures 2D (projections) ;
- 25 sujets seulement ;
- un label par vidéo, sans type d'erreur ;
- inspection visuelle faite par un non-expert.

## Questions probables du jury

*(Les 15 questions et leurs réponses seront rédigées à partir des résultats réels de la phase B.)*
