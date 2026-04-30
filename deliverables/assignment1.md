Mon projet :
J’ai développé un modèle permettant de prédire la probabilité qu’un élève souhaite poursuivre des études supérieures, afin d’aider un lycée à mieux cibler les élèves à accompagner.

Le business case :
Face aux inégalités des chances dans l’éducation, l’objectif est d’identifier les élèves les plus à risque de ne pas poursuivre leurs études, pour leur proposer un accompagnement prioritaire et améliorer l’égalité des opportunités.

Les sources de données :
Je me suis appuyé sur un dataset existant: 
Student Performance Dataset (Portugal)
https://archive.ics.uci.edu/dataset/320/student+performance
que j’ai enrichi en créant 10 nouvelles features. Celles-ci permettent de mieux représenter le contexte familial (parents_edu_sum), l’organisation de l’élève (study_travel_balance, study_effort_index), le niveau de soutien et les comportements à risque (school_support_count, no_support, risk_behavior_score), ainsi que la performance académique (academic_avg_g1_g2, early_academic_risk, older_than_cohort).
L’objectif est de rendre les données plus pertinentes et exploitables par le modèle.
