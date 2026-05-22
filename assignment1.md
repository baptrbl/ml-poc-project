# Assignment 1 - Projet et donnees utilisees

## Projet

Ce projet est un proof of concept de machine learning applique a l'education. L'objectif est de predire si un eleve est susceptible de poursuivre des etudes superieures, afin d'aider une equipe pedagogique a prioriser les profils qui pourraient avoir besoin d'un accompagnement.

Le modele ne remplace pas une decision humaine. Il sert uniquement de signal d'aide a l'analyse pour mieux cibler les actions d'orientation, de soutien scolaire ou de suivi individualise.

## Business case

Le cas d'usage vise a reduire les inegalites d'opportunite dans l'orientation. Un etablissement peut utiliser ce type d'outil pour reperer plus tot des eleves fragiles, puis declencher un echange humain avec eux.

Dans ce projet, la classe suivie est l'absence de poursuite vers les etudes superieures. Elle est minoritaire dans le dataset, ce qui rend les metriques comme le recall et le F1-score importantes en plus de l'accuracy.

## Donnees utilisees

Le dataset source est le Student Performance Dataset de l'UCI Machine Learning Repository :

https://archive.ics.uci.edu/dataset/320/student+performance

Les donnees de depart decrivent des eleves portugais avec des variables scolaires, familiales, sociales et comportementales. Les fichiers sources utilises dans le repo sont :

- `data/raw/student-mat.csv`
- `data/raw/student-por.csv`

Les donnees preparees sont disponibles ici :

- `data/processed/student_data.csv`
- `data/processed/student_data_features.csv`

## Cible

La variable cible est `higher`, qui indique si l'eleve souhaite poursuivre des etudes superieures.

Pour l'entrainement, elle est encodee en cible de risque :

- `higher = yes` devient `0`
- `higher = no` devient `1`

Ainsi, la classe `1` correspond aux eleves a risque de ne pas poursuivre vers les etudes superieures.
