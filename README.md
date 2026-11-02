# Lecture automatique des chèques

- Étude de cas 1 du module Advanced Artificial Intelligence (M2 SDIA, ENSET Mohammedia).
- Quatre jalons sur la même tâche : lire le montant d'un chèque, détecter les altérations
et décider quand la lecture est assez sûre pour être automatisée.
- Auteurs : Hafssa Mifah Idrissi - Hanane Ait Lhaj - Fatima Zahra Lasfar
- Lien app : https://lecture-automatique-des-cheques.streamlit.app/

## Contenu

| Élément | Rôle |
|---|---|
| `notebooks/` | Un notebook exécuté par jalon (code, expériences, résultats, page de décisions) |
| `models/` | Modèles retenus, au format Keras (voir le tableau ci-dessous) |
| `data/exemples.npz` | 60 chèques de test par scanner, pour la démonstration |
| `app.py` | Application Streamlit : démonstration des quatre jalons |
| `journal/journal.jsonl` | Journal de toutes les expériences, échecs compris |

## Modèles

| Jalon | Fichiers | Usage |
|---|---|---|
| 1 | `mlp_base_80ep.keras` | MLP à cinq têtes de position |
| 2 | `cnn_final_80ep.keras`, `detecteur_blocs4.keras`, `debruiteur_v4.keras` | Lecteur CNN, détecteur d'altération, auto-encodeur débruiteur |
| 3 | `aug_corr50`, `aug_corr100`, `multi_lam0.5`, `multi_chaud_lam2_120ep`, `detecteur_aug` | Ensemble de 4 lecteurs et détecteur dédié (scanner B) |
| 4 | `seq_gru128.keras`, `seq_gru128_s1.keras` | Paire de GRU encodeur-décodeur |

Le modèle du jalon 1 chargé par l'application est le MLP de référence du jalon 2,
qui reprend l'architecture et le budget d'entraînement du modèle final du jalon 1.

## Installation et lancement

    python -m venv .venv
    source .venv/bin/activate        # Windows : .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

Versions utilisées pour l'entraînement : Python 3.13, TensorFlow 2.20 à 2.21,
scikit-learn 1.6.1, NumPy 2.1.3.

## Reproduction

1. Placer `uc1_domaineA.npz` et `uc1_domaineB.npz` (fournis par l'enseignant, non redistribués ici)
   dans le dossier de travail de Colab.
2. Exécuter les notebooks dans l'ordre 1, 2, 3, 4. Les entraînements longs sont sauvegardés
   sur Drive et repris automatiquement.
3. Les graines sont fixées (`SEED = 0`) ; les résultats GPU peuvent varier légèrement d'une exécution à l'autre.


## Protocole

Tous les réglages sont faits sur la validation. Le jeu de test est évalué une seule fois par jalon,
après déclaration du système retenu dans `journal.jsonl`.

## Limites

Voir la page de décisions de chaque notebook : cibles manquées (jalons 1, 2 et 4 sur le montant exact),
modèles issus d'une seule graine pour les jalons 2 et 3, décodage glouton sans recherche en faisceau.
