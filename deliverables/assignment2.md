# Assignment 2 - Feature engineering et dataset preprocessé

## Dataset preprocessé

Le dataset preprocessé principal est :

`data/processed/student_data_features.csv`

Il contient 1 044 lignes et 43 colonnes. Il est construit a partir des deux fichiers UCI bruts `student-mat.csv` et `student-por.csv`, puis enrichi avec des variables derivees.

Un second niveau de feature engineering est applique dans `src/data.py` au moment du chargement du modele. Cette etape recree les variables necessaires au pipeline final et garantit que l'application Streamlit utilise les memes transformations que l'entrainement.

## Nettoyage et preparation

Les principales etapes de preparation sont :

- fusion des donnees sources disponibles ;
- conservation de la variable cible `higher` ;
- creation d'un encodage de risque `target_risk` pour l'entrainement ;
- transformation de variables binaires comme `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `internet`, `romantic` ;
- plafonnement de `absences` au quantile 99 dans le split modele pour limiter l'effet des valeurs extremes ;
- separation train/test stratifiee avec `random_state = 42`.

## Features créées dans le dataset preprocessé

Les features suivantes sont deja presentes dans `data/processed/student_data_features.csv` :

| Feature | Description |
| --- | --- |
| `parents_edu_sum` | Somme du niveau d'education de la mere et du pere. |
| `study_travel_balance` | Difference entre temps d'etude et temps de trajet. |
| `study_effort_index` | Ratio simple entre effort d'etude et contrainte de trajet. |
| `school_support_count` | Nombre d'aides scolaires/familiales identifiees. |
| `no_support` | Indicateur d'absence de soutien. |
| `alcohol_total` | Somme des consommations d'alcool semaine et week-end. |
| `risk_behavior_score` | Synthese de signaux sociaux/comportementaux. |
| `academic_avg_g1_g2` | Moyenne des notes G1 et G2. |
| `early_academic_risk` | Indicateur de risque academique precoce. |
| `older_than_cohort` | Indicateur d'age superieur a la cohorte attendue. |

## Features recréées côté code modèle

Dans `src/data.py`, la fonction `add_model_features()` ajoute ou recree des variables utilisees par le modele final :

| Feature | Description |
| --- | --- |
| `grade_avg_12` | Moyenne des notes `G1` et `G2`. |
| `grade_trend_12` | Evolution entre `G1` et `G2`. |
| `grade_momentum` | Evolution relative entre `G1` et `G2`. |
| `high_absence_flag` | Indicateur d'absenteisme eleve. |
| `alc_index` | Moyenne de la consommation d'alcool semaine/week-end. |
| `social_index` | Moyenne de `goout` et `freetime`. |
| `risk_behavior_index` | Synthese entre comportement social et alcool. |
| `support_index` | Somme des soutiens et activites disponibles. |
| `parent_edu_avg` | Moyenne du niveau d'education des parents. |
| `parent_edu_gap` | Ecart entre education de la mere et du pere. |
| `failure_flag` | Indique si l'eleve a deja eu au moins un echec scolaire. |
| `high_failure_flag` | Indique si l'eleve a eu au moins deux echecs scolaires. |

## Nouveau dataset exploitable

Le modele final n'utilise pas `G3`, car cette note finale serait trop proche du resultat scolaire final et risquerait d'introduire une fuite d'information. La cible `higher` et la cible encodee `target_risk` sont egalement retirees des variables explicatives.

Le dataset utilise pour le modele contient donc des variables brutes utiles et des features derivees, afin de mieux representer :

- le niveau academique de l'eleve ;
- la dynamique de progression ;
- le contexte familial ;
- le soutien disponible ;
- l'absenteisme ;
- certains comportements sociaux.
