import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

# 1. PARAMETRES ET CHARGEMENT DES DONNEES

CHEMIN_DONNEES = "data/train_data_QUICK_START.csv"
N_LIGNES = 20_000
TYPE_EXPERIENCE = "DMS_MaP"
CIBLE = "reactivity_0028"
ALPHA_RIDGE = 1.0

# Bornes : les réactivités (cibles ET prédictions) sont ramenées dans [0, 1]
BORNE_MIN = 0.0
BORNE_MAX = 1.0

# Filtre de bruit : le fichier QUICK_START n'a pas de colonne
# SN_filter / signal_to_noise. On utilise donc l'erreur de mesure
# reactivity_error_XXXX : une mesure d'entraînement est écartée si son
# erreur dépasse ce seuil (~95e percentile )
# Le filtre s'applique uniquement à l'entraînement : 
# validation et test gardent toutes les mesures, pour une évaluation correcte
SEUIL_ERREUR = 0.416

df = pd.read_csv(CHEMIN_DONNEES, nrows=N_LIGNES)

# Garder le type d'expérience étudié
df = df[df["experiment_type"] == TYPE_EXPERIENCE].copy()

print("Nombre de lignes après filtrage :", len(df))

# 2. SEPARATION ENTRAINEMENT / VALIDATION / TEST

# Première séparation : 70 % entraînement, 30 % temporaire
separateur_1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

indices_train, indices_temp = next(
    separateur_1.split(df, groups=df["sequence"])
)

entrainement = df.iloc[indices_train].copy()
temporaire = df.iloc[indices_temp].copy()

# Deuxième séparation : 15 % validation, 15 % test
separateur_2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=43
)

indices_val, indices_test = next(
    separateur_2.split(
        temporaire,
        groups=temporaire["sequence"]
    )
)

validation = temporaire.iloc[indices_val].copy()
test = temporaire.iloc[indices_test].copy()

print("\n Séparation finale des données ")
print("Entraînement :", len(entrainement))
print("Validation :", len(validation))
print("Test :", len(test))

print(
    "Séquences communes entraînement/validation :",
    len(set(entrainement["sequence"]) & set(validation["sequence"]))
)
print(
    "Séquences communes entraînement/test :",
    len(set(entrainement["sequence"]) & set(test["sequence"]))
)
print(
    "Séquences communes validation/test :",
    len(set(validation["sequence"]) & set(test["sequence"]))
)

# 3. FONCTIONS COMMUNES

resultats = []


def enregistrer_resultat(nom, mae, nombre_mesures, groupe):
    """Enregistre un score dans le tableau comparatif."""
    resultats.append({
        "Groupe": groupe,
        "Methode": nom,
        "MAE": mae,
        "Nombre de mesures": nombre_mesures
    })

    print(f"{nom} : MAE = {mae:.6f}")


def borner(valeurs):
    """Ramène des réactivités (ou des prédictions) dans [0, 1]."""
    return np.clip(valeurs, BORNE_MIN, BORNE_MAX)


def evaluer(nom, vraies_valeurs, predictions, groupe):
    """Calcule la MAE (prédictions bornées) et enregistre le résultat."""
    score = mean_absolute_error(vraies_valeurs, borner(predictions))

    enregistrer_resultat(
        nom,
        score,
        len(vraies_valeurs),
        groupe
    )

    return score


def colonnes_de_reactivite(dataframe):
    """Retourne les colonnes de réactivité, sans les erreurs de mesure."""
    return [
        c for c in dataframe.columns
        if c.startswith("reactivity_")
        and not c.startswith("reactivity_error_")
    ]


def extraire_fenetre(sequence, position, taille=5):
    """
    Extrait une fenêtre centrée sur la position cible.
    """
    i = position - 1
    rayon = taille // 2

    debut = i - rayon
    fin = i + rayon + 1

    if debut < 0 or fin > len(sequence):
        return None

    fenetre = sequence[debut:fin]

    if len(fenetre) != taille:
        return None

    if any(n not in "ACGU" for n in fenetre):
        return None

    return list(fenetre)


_avertissement_bruit_affiche = False


def selectionner_mesures(dataframe, colonne_cible, filtrer_bruit=False):
    """
    Garde les lignes où la cible est connue, applique 
    le filtre de bruit, puis borne la cible dans [0, 1].
    """
    global _avertissement_bruit_affiche

    sous_ensemble = dataframe.dropna(
        subset=["sequence", colonne_cible]
    ).copy()

    if filtrer_bruit:
        colonne_erreur = colonne_cible.replace(
            "reactivity_", "reactivity_error_"
        )

        if colonne_erreur in sous_ensemble.columns:
            erreurs = sous_ensemble[colonne_erreur]
            sous_ensemble = sous_ensemble[
                erreurs.notna() & (erreurs <= SEUIL_ERREUR)
            ]
        elif not _avertissement_bruit_affiche:
            print(
                f"Attention : colonne {colonne_erreur} absente, "
                "le filtre de bruit n'est pas appliqué."
            )
            _avertissement_bruit_affiche = True

    sous_ensemble[colonne_cible] = borner(sous_ensemble[colonne_cible])

    return sous_ensemble


def preparer_donnees(dataframe, colonne_cible, position, taille=5,
                     filtrer_bruit=False, avec_sequences=False):
    """
    Prépare X (fenêtres de nucléotides) et y (réactivités bornées)
    pour une position.
    filtrer_bruit=True : à utiliser pour l'entraînement uniquement.
    """
    sous_ensemble = selectionner_mesures(
        dataframe, colonne_cible, filtrer_bruit
    )

    X = []
    y = []
    sequences = []

    for sequence, valeur in zip(
        sous_ensemble["sequence"],
        sous_ensemble[colonne_cible]
    ):
        fenetre = extraire_fenetre(sequence, position, taille)

        if fenetre is not None:
            X.append(fenetre)
            y.append(float(valeur))
            sequences.append(sequence)

    X = np.asarray(X)
    y = np.asarray(y, dtype=float)

    if avec_sequences:
        return X, y, sequences

    return X, y


def creer_ridge(avec_interactions=False, alpha=None):
    if alpha is None:
        alpha = ALPHA_RIDGE

    etapes = [OneHotEncoder(handle_unknown="ignore")]

    if avec_interactions:
        etapes.append(
            PolynomialFeatures(
                degree=2,
                interaction_only=True,
                include_bias=False
            )
        )

    etapes.append(Ridge(alpha=alpha))

    return make_pipeline(*etapes)


def predire_ridge(modele, X):
    """Prédictions Ridge bornées dans [0, 1]."""
    return borner(modele.predict(X))


print("\n TEST DE LA FONCTION ")

sequence_test = validation.iloc[0]["sequence"]

print("Séquence :", sequence_test)
print("Longueur :", len(sequence_test))

print(
    "Fenêtre de 9 nucléotides à la position 28 :",
    extraire_fenetre(sequence_test, 28, 9)
)

print(
    "Fenêtre de 9 nucléotides à la position 28, longueur :",
    len(extraire_fenetre(sequence_test, 28, 9) or [])
)

# 4. MODELES POUR LA POSITION 28

print("\nMODELES POUR LA POSITION 28")

# Entraînement : filtre de bruit + bornes.
# Validation : bornes seulement (toutes les mesures sont gardées).
train_28 = selectionner_mesures(entrainement, CIBLE, filtrer_bruit=True)
validation_28 = selectionner_mesures(validation, CIBLE, filtrer_bruit=False)

y_train = train_28[CIBLE]
y_validation = validation_28[CIBLE]

print("Mesures d'entraînement gardées après filtre :", len(train_28))

# Modèle 1 : moyenne générale

moyenne_generale = y_train.mean()

predictions = np.full(
    len(validation_28),
    moyenne_generale
)

evaluer(
    "Position 28 : Moyenne générale",
    y_validation,
    predictions,
    "Position 28"
)

# Modèle 2 : moyenne par nucléotide à la position 28

nucleotide_train = train_28["sequence"].str[27]
nucleotide_validation = validation_28["sequence"].str[27]

moyennes_nucleotides = y_train.groupby(nucleotide_train).mean()

predictions = (
    nucleotide_validation
    .map(moyennes_nucleotides)
    .fillna(moyenne_generale)
)

evaluer(
    "Position 28 : Moyenne par nucléotide",
    y_validation,
    predictions,
    "Position 28"
)

# Modèle 3 : moyenne par motif de 3 nucléotides

motif_train = train_28["sequence"].str[26:29]
motif_validation = validation_28["sequence"].str[26:29]

moyennes_motifs_28 = y_train.groupby(motif_train).mean()

predictions = (
    motif_validation
    .map(moyennes_motifs_28)
    .fillna(moyenne_generale)
)

evaluer(
    "Position 28 : Moyenne par motif",
    y_validation,
    predictions,
    "Position 28"
)

# Modèles 4 et 5 : Ridge sur une fenêtre de 5 nucléotides

X_train_28, y_train_ridge = preparer_donnees(
    entrainement, CIBLE, 28, taille=5, filtrer_bruit=True
)

X_validation_28, y_validation_ridge = preparer_donnees(
    validation, CIBLE, 28, taille=5, filtrer_bruit=False
)

# Ridge sans interactions
modele_ridge = creer_ridge(avec_interactions=False)
modele_ridge.fit(X_train_28, y_train_ridge)

evaluer(
    "Position 28 : Ridge",
    y_validation_ridge,
    predire_ridge(modele_ridge, X_validation_28),
    "Position 28"
)

# Ridge avec interactions
modele_interactions = creer_ridge(avec_interactions=True)
modele_interactions.fit(X_train_28, y_train_ridge)

evaluer(
    "Position 28 : Ridge avec interactions",
    y_validation_ridge,
    predire_ridge(modele_interactions, X_validation_28),
    "Position 28"
)

# 5. PREPARATION DE TOUTES LES POSITIONS MESUREES

print("\nMODELES SUR TOUTES LES POSITIONS")

colonnes_react = colonnes_de_reactivite(df)

positions = sorted(
    int(c.split("_")[1])
    for c in colonnes_react
    if entrainement[c].notna().any()
    and validation[c].notna().any()
)

print("Nombre de positions disponibles :", len(positions))

# Listes communes pour calculer les MAE globales
vraies_valeurs_globales = []
predictions_baseline_globales = []
predictions_motifs_globales = []
predictions_ridge_globales = []

resultats_positions = []

# 6. ENTRAINEMENT ET EVALUATION POSITION PAR POSITION

for position in positions:

    cible_position = f"reactivity_{position:04d}"

    X_train, y_train_pos = preparer_donnees(
        entrainement, cible_position, position,
        taille=5, filtrer_bruit=True
    )

    X_validation, y_validation_pos = preparer_donnees(
        validation, cible_position, position,
        taille=5, filtrer_bruit=False
    )

    if len(y_train_pos) == 0 or len(y_validation_pos) == 0:
        continue

    # Modèle A : moyenne par position (sur les mesures filtrées et bornées)
    moyenne_position = float(np.mean(y_train_pos))

    # Modèle B : moyenne par motif de 3 nucléotides.
    # Le motif de 3 est le centre de la fenêtre de 5 : les lignes
    # sont donc exactement les mêmes que pour Ridge.
    motifs_train = ["".join(f[1:4]) for f in X_train]
    motifs_validation = ["".join(f[1:4]) for f in X_validation]

    moyennes_motifs = (
        pd.Series(y_train_pos)
        .groupby(motifs_train)
        .mean()
        .to_dict()
    )

    # Modèle C : Ridge avec interactions
    modele = creer_ridge(avec_interactions=True)
    modele.fit(X_train, y_train_pos)
    predictions_ridge = predire_ridge(modele, X_validation)

    predictions_baseline = np.full(
        len(y_validation_pos),
        moyenne_position
    )

    predictions_motifs = np.array([
        moyennes_motifs.get(motif, moyenne_position)
        for motif in motifs_validation
    ])

    # Accumuler les mesures pour la comparaison globale
    vraies_valeurs_globales.extend(y_validation_pos)
    predictions_baseline_globales.extend(predictions_baseline)
    predictions_motifs_globales.extend(predictions_motifs)
    predictions_ridge_globales.extend(predictions_ridge)

    resultats_positions.append({
        "Position": position,
        "Nombre de mesures": len(y_validation_pos),
        "MAE baseline": mean_absolute_error(
            y_validation_pos, predictions_baseline
        ),
        "MAE motif": mean_absolute_error(
            y_validation_pos, predictions_motifs
        ),
        "MAE Ridge interactions": mean_absolute_error(
            y_validation_pos, predictions_ridge
        )
    })

# 7. COMPARAISON GLOBALE SUR TOUTES LES POSITIONS

y_global = np.asarray(vraies_valeurs_globales)

enregistrer_resultat(
    "Toutes positions : Moyenne par position",
    mean_absolute_error(y_global, predictions_baseline_globales),
    len(y_global),
    "Toutes positions"
)

enregistrer_resultat(
    "Toutes positions : Moyenne par motif",
    mean_absolute_error(y_global, predictions_motifs_globales),
    len(y_global),
    "Toutes positions"
)

enregistrer_resultat(
    "Toutes positions : Ridge avec interactions",
    mean_absolute_error(y_global, predictions_ridge_globales),
    len(y_global),
    "Toutes positions"
)

# 8. TABLEAUX DE VALIDATION

tableau_resultats = (
    pd.DataFrame(resultats)
    .sort_values(["Groupe", "MAE"])
    .reset_index(drop=True)
)

print("\nCOMPARAISON FINALE (VALIDATION)")
print(tableau_resultats.to_string(index=False))

tableau_positions = pd.DataFrame(resultats_positions)

print("\n10 POSITIONS LES MIEUX PREDITES PAR RIDGE")
print(
    tableau_positions
    .sort_values("MAE Ridge interactions")
    .head(10)
    .to_string(index=False)
)

print("\n10 POSITIONS LES PLUS DIFFICILES POUR RIDGE")
print(
    tableau_positions
    .sort_values("MAE Ridge interactions", ascending=False)
    .head(10)
    .to_string(index=False)
)

# 9. TEST DE PLUSIEURS VALEURS DE ALPHA (FENETRE DE 5)

alphas = [1, 10, 100.0, 200.0, 500.0, 1000.0]

resultats_alpha = []

# Stocker les données préparées pour ne pas les recalculer
donnees_par_position = {}

for position in positions:
    cible_position = f"reactivity_{position:04d}"

    X_train, y_train = preparer_donnees(
        entrainement, cible_position, position,
        taille=5, filtrer_bruit=True
    )

    X_val, y_val = preparer_donnees(
        validation, cible_position, position,
        taille=5, filtrer_bruit=False
    )

    if len(y_train) == 0 or len(y_val) == 0:
        continue

    donnees_par_position[position] = (X_train, y_train, X_val, y_val)

for alpha in alphas:

    toutes_valeurs_reelles = []
    toutes_predictions = []

    for position, (X_train, y_train, X_val, y_val) in \
            donnees_par_position.items():

        modele = creer_ridge(avec_interactions=True, alpha=alpha)
        modele.fit(X_train, y_train)

        toutes_valeurs_reelles.extend(y_val)
        toutes_predictions.extend(predire_ridge(modele, X_val))

    mae_globale = mean_absolute_error(
        toutes_valeurs_reelles,
        toutes_predictions
    )

    resultats_alpha.append({
        "Alpha": alpha,
        "MAE globale": mae_globale,
        "Nombre de positions": len(donnees_par_position),
        "Nombre de mesures": len(toutes_valeurs_reelles)
    })

    print(
        f"Alpha = {alpha:6.1f} | "
        f"MAE globale = {mae_globale:.6f} | "
        f"Positions = {len(donnees_par_position)} | "
        f"Mesures = {len(toutes_valeurs_reelles)}"
    )

tableau_alpha = (
    pd.DataFrame(resultats_alpha)
    .sort_values("MAE globale")
    .reset_index(drop=True)
)

print("\n COMPARAISON DES ALPHAS ")
print(tableau_alpha.to_string(index=False))

meilleur = tableau_alpha.iloc[0]

print(
    f"\nMeilleur alpha : {meilleur['Alpha']}"
    f"\nMAE globale : {meilleur['MAE globale']:.6f}"
)

# 10. COMPARAISON DES TAILLES DE FENETRE

print("\n")
print("COMPARAISON DES TAILLES DE FENETRE")


tailles_fenetre = [3, 5, 7, 9, 13, 15, 17, 19]
alpha_choisi = 100

resultats_fenetres = []

colonnes_cibles = colonnes_de_reactivite(df)

# A. Observations de validation communes.

validation_commune = {}

for colonne_cible in colonnes_cibles:
    pos = int(colonne_cible.split("_")[-1])

    _, y_val_9, sequences_val_9 = preparer_donnees(
        validation, colonne_cible, pos,
        taille=max(tailles_fenetre),
        filtrer_bruit=False,
        avec_sequences=True
    )

    if len(y_val_9) > 0:
        validation_commune[pos] = (sequences_val_9, y_val_9)

print("Nombre de positions avec validation :", len(validation_commune))


def evaluer_fenetre_alpha(taille, alpha):
    """MAE globale de validation pour une taille de fenêtre et un alpha."""
    y_reels_total = []
    y_predits_total = []

    for colonne_cible in colonnes_cibles:
        pos = int(colonne_cible.split("_")[-1])

        if pos not in validation_commune:
            continue

        X_train, y_train = preparer_donnees(
            entrainement, colonne_cible, pos,
            taille=taille, filtrer_bruit=True
        )

        if len(y_train) == 0:
            continue

        sequences_val, y_val = validation_commune[pos]

        X_val = [
            extraire_fenetre(sequence, pos, taille)
            for sequence in sequences_val
        ]

        if any(fenetre is None for fenetre in X_val):
            continue

        # Correction : alpha passé par mot-clé (avant, alpha_choisi
        # était pris pour avec_interactions et alpha restait à 1.0).
        modele = creer_ridge(avec_interactions=True, alpha=alpha)
        modele.fit(X_train, y_train)

        y_reels_total.extend(y_val)
        y_predits_total.extend(predire_ridge(modele, np.asarray(X_val)))

    if not y_reels_total:
        return None, 0

    return (
        mean_absolute_error(y_reels_total, y_predits_total),
        len(y_reels_total)
    )


# B. Tester chaque taille de fenêtre

for taille in tailles_fenetre:
    print(f"\nFenêtre de {taille} nucléotides")

    mae, nb = evaluer_fenetre_alpha(taille, alpha_choisi)

    if mae is None:
        print("Aucune observation disponible.")
        continue

    resultats_fenetres.append({
        "taille_fenetre": taille,
        "alpha": alpha_choisi,
        "MAE": mae,
        "nb_observations": nb
    })

    print(f"MAE globale : {mae:.6f}")
    print(f"Observations évaluées : {nb}")

# C. Classement final et choix automatique de la meilleure fenêtre

if not resultats_fenetres:
    raise RuntimeError(
        "Aucun résultat pour les tailles de fenêtre : "
        "il faut vérifier la préparation des données."
    )

resultats_fenetres_df = (
    pd.DataFrame(resultats_fenetres)
    .sort_values("MAE")
    .reset_index(drop=True)
)

print("\nClassement des tailles de fenêtre :")
print(resultats_fenetres_df.to_string(index=False))

MEILLEURE_FENETRE = int(resultats_fenetres_df.iloc[0]["taille_fenetre"])
print(f"\nMeilleure fenêtre retenue : {MEILLEURE_FENETRE} nucléotides")

if MEILLEURE_FENETRE == max(tailles_fenetre):
    print(
        "Attention : la meilleure fenêtre est la plus grande testée. "
        "Une fenêtre encore plus large pourrait être meilleure."
    )

# 11. OPTIMISATION DE ALPHA POUR LA MEILLEURE FENETRE


print("\n" + "=" * 60)
print(f"OPTIMISATION DE ALPHA POUR UNE FENETRE DE {MEILLEURE_FENETRE}")
print("=" * 60)

# Grille large : avec bornes et filtre de bruit, l'alpha optimal
# peut être très différent des valeurs trouvées avant.
alphas_a_tester = [1, 10, 50, 100, 150, 200, 225, 250, 300, 500, 1000]

resultats_alpha_fenetre = []

for alpha_test in alphas_a_tester:
    print(f"\nTest de alpha = {alpha_test}")

    mae, nb = evaluer_fenetre_alpha(MEILLEURE_FENETRE, alpha_test)

    if mae is None:
        print("Aucune observation disponible.")
        continue

    resultats_alpha_fenetre.append({
        "Fenetre": MEILLEURE_FENETRE,
        "Alpha": alpha_test,
        "MAE": mae,
        "Nombre de mesures": nb
    })

    print(f"MAE globale = {mae:.6f}")
    print(f"Nombre de mesures = {nb}")

if not resultats_alpha_fenetre:
    raise RuntimeError("Aucun résultat pour la recherche d'alpha.")

tableau_alpha_fenetre = (
    pd.DataFrame(resultats_alpha_fenetre)
    .sort_values("MAE")
    .reset_index(drop=True)
)

print(f"\nCLASSEMENT DES ALPHAS : FENETRE {MEILLEURE_FENETRE}")
print(tableau_alpha_fenetre.to_string(index=False))

MEILLEUR_ALPHA = float(tableau_alpha_fenetre.iloc[0]["Alpha"])
print(f"\nMeilleur alpha : {MEILLEUR_ALPHA}")
print(f"Meilleure MAE : {tableau_alpha_fenetre.iloc[0]['MAE']:.6f}")

if MEILLEUR_ALPHA in (min(alphas_a_tester), max(alphas_a_tester)):
    print(
        "Attention : le meilleur alpha est au bord de la grille. "
        "Il faudrait élargir la grille dans cette direction."
    )

# 12. EVALUATION FINALE SUR LE JEU DE TEST + GRAPHIQUES

# Fenêtre et alpha choisis automatiquement sur la validation
# (sections 10 et 11) : le jeu de test n'a servi à aucun choix.
TAILLE_FENETRE_FINALE = MEILLEURE_FENETRE
ALPHA_FINAL = MEILLEUR_ALPHA

print("\n" + "=" * 65)
print("EVALUATION FINALE SUR LE JEU DE TEST")
print("=" * 65)
print(f"Bornes : [{BORNE_MIN}, {BORNE_MAX}]")
print(f"Filtre de bruit (entraînement) : reactivity_error <= {SEUIL_ERREUR}")

colonnes_cibles = colonnes_de_reactivite(entrainement)

# Toutes les méthodes sont évaluées sur exactement les mêmes mesures.
y_reels_test = []
y_predits_ridge_interactions = []
y_predits_ridge_sans_interactions = []
y_predits_moyenne_position = []
y_predits_motif3 = []
y_predits_motif_fenetre = []

resultats_par_position = []

centre = TAILLE_FENETRE_FINALE // 2

for colonne_cible in colonnes_cibles:
    position = int(colonne_cible.split("_")[-1])

    X_train, y_train = preparer_donnees(
        entrainement, colonne_cible, position,
        taille=TAILLE_FENETRE_FINALE, filtrer_bruit=True
    )

    if len(y_train) == 0:
        continue

    X_test, y_test_position = preparer_donnees(
        test, colonne_cible, position,
        taille=TAILLE_FENETRE_FINALE, filtrer_bruit=False
    )

    if len(y_test_position) == 0:
        continue

    # Modèle 1 : moyenne par position
    moyenne_position = float(np.mean(y_train))
    pred_position = np.full(len(y_test_position), moyenne_position)

    # Modèle 2 : moyenne par motif central de 3 nucléotides
    motifs3_train = ["".join(f[centre - 1:centre + 2]) for f in X_train]
    motifs3_test = ["".join(f[centre - 1:centre + 2]) for f in X_test]

    moyennes3 = (
        pd.Series(y_train).groupby(motifs3_train).mean().to_dict()
    )

    pred_motif3 = np.asarray([
        moyennes3.get(motif, moyenne_position)
        for motif in motifs3_test
    ])

    # Modèle 3 : moyenne par motif de toute la fenêtre
    # (motif inconnu -> moyenne de la position)
    motifs_fenetre_train = ["".join(f) for f in X_train]
    motifs_fenetre_test = ["".join(f) for f in X_test]

    moyennes_fenetre = (
        pd.Series(y_train).groupby(motifs_fenetre_train).mean().to_dict()
    )

    pred_motif_fenetre = np.asarray([
        moyennes_fenetre.get(motif, moyenne_position)
        for motif in motifs_fenetre_test
    ])

    # Modèle 4 : Ridge sans interactions
    modele_ridge_simple = creer_ridge(
        avec_interactions=False, alpha=ALPHA_FINAL
    )
    modele_ridge_simple.fit(X_train, y_train)
    pred_ridge_simple = predire_ridge(modele_ridge_simple, X_test)

    # Modèle 5 : Ridge avec interactions
    modele_ridge_interactions = creer_ridge(
        avec_interactions=True, alpha=ALPHA_FINAL
    )
    modele_ridge_interactions.fit(X_train, y_train)
    pred_ridge_interactions = predire_ridge(
        modele_ridge_interactions, X_test
    )

    y_reels_test.extend(y_test_position)
    y_predits_moyenne_position.extend(pred_position)
    y_predits_motif3.extend(pred_motif3)
    y_predits_motif_fenetre.extend(pred_motif_fenetre)
    y_predits_ridge_sans_interactions.extend(pred_ridge_simple)
    y_predits_ridge_interactions.extend(pred_ridge_interactions)

    resultats_par_position.append({
        "Position": position,
        "Nombre de mesures": len(y_test_position),
        "MAE test": mean_absolute_error(
            y_test_position, pred_ridge_interactions
        )
    })

# 13. SCORES GLOBAUX ET GRAPHIQUES

if len(y_reels_test) == 0:
    print("Aucune mesure valide trouvée dans le jeu de test.")
else:
    y_reel = np.asarray(y_reels_test)
    predictions_modeles = {
        "Moyenne par position": np.asarray(y_predits_moyenne_position),
        "Moyenne par motif central 3 nt": np.asarray(y_predits_motif3),
        f"Moyenne par motif fenêtre {TAILLE_FENETRE_FINALE} nt": np.asarray(y_predits_motif_fenetre),
        f"Ridge sans interactions ({TAILLE_FENETRE_FINALE} nt)": np.asarray(
            y_predits_ridge_sans_interactions
        ),
        f"Ridge avec interactions ({TAILLE_FENETRE_FINALE} nt)": np.asarray(
            y_predits_ridge_interactions
        )
    }

    for nom, predictions in predictions_modeles.items():
        if len(predictions) != len(y_reel):
            raise ValueError(
                f"Nombre de prédictions différent pour {nom}."
            )

    scores = [
        {"Modele": nom, "MAE test": mean_absolute_error(y_reel, pred)}
        for nom, pred in predictions_modeles.items()
    ]

    tableau_scores = (
        pd.DataFrame(scores)
        .sort_values("MAE test")
        .reset_index(drop=True)
    )

    print(f"\nFenêtre Ridge : {TAILLE_FENETRE_FINALE} nucléotides")
    print(f"Alpha Ridge : {ALPHA_FINAL}")
    print(f"Positions évaluées : {len(resultats_par_position)}")
    print(f"Mesures communes évaluées : {len(y_reel)}")
    print("\nComparaison des modèles (MAE sur le test) :")
    print(tableau_scores.to_string(
        index=False, float_format=lambda x: f"{x:.6f}"
    ))

    tableau_test = pd.DataFrame(resultats_par_position)

    print("\n10 positions les mieux prédites par Ridge avec interactions :")
    print(
        tableau_test.sort_values("MAE test")
        .head(10).to_string(index=False)
    )

    print("\n10 positions les plus difficiles pour Ridge avec interactions :")
    print(
        tableau_test.sort_values("MAE test", ascending=False)
        .head(10).to_string(index=False)
    )

    dossier_graphiques = "graphiques"
    os.makedirs(dossier_graphiques, exist_ok=True)

    # Graphique 1 : comparaison des 5 modèles
    noms = tableau_scores["Modele"].tolist()
    maes = tableau_scores["MAE test"].tolist()

    plt.figure(figsize=(12, 6))
    barres = plt.bar(noms, maes)
    plt.ylabel("MAE sur le jeu de test")
    plt.title("Comparaison des modèles sur le jeu de test")
    plt.xticks(rotation=18, ha="right")

    for barre, valeur in zip(barres, maes):
        plt.text(
            barre.get_x() + barre.get_width() / 2,
            barre.get_height(),
            f"{valeur:.4f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(dossier_graphiques, "01_comparaison_modeles.png"),
        dpi=300
    )
    plt.show()
    plt.close()

    # Graphique 2 : valeurs réelles contre valeurs prédites
    y_ridge = predictions_modeles[f"Ridge avec interactions ({TAILLE_FENETRE_FINALE} nt)"]

    plt.figure(figsize=(8, 8))
    plt.scatter(y_reel, y_ridge, alpha=0.25, s=12)
    plt.plot(
        [BORNE_MIN, BORNE_MAX],
        [BORNE_MIN, BORNE_MAX],
        linestyle="--",
        label="Prédiction parfaite"
    )
    plt.xlim(BORNE_MIN, BORNE_MAX)
    plt.ylim(BORNE_MIN, BORNE_MAX)
    plt.xlabel("Réactivité réelle (bornée)")
    plt.ylabel("Réactivité prédite (bornée)")
    plt.title("Valeurs réelles contre valeurs prédites - Ridge")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(dossier_graphiques, "02_reel_vs_predit.png"),
        dpi=300
    )
    plt.show()
    plt.close()

    # Graphique 3 : distribution des erreurs Ridge
    erreurs = y_ridge - y_reel

    plt.figure(figsize=(10, 6))
    plt.hist(erreurs, bins=50, edgecolor="black")
    plt.axvline(0, linestyle="--", label="Erreur nulle")
    plt.axvline(
        erreurs.mean(),
        linestyle=":",
        label=f"Erreur moyenne = {erreurs.mean():.4f}"
    )
    plt.xlabel("Erreur (prédiction - valeur réelle)")
    plt.ylabel("Nombre de mesures")
    plt.title("Distribution des erreurs du modèle Ridge")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(dossier_graphiques, "03_distribution_erreurs.png"),
        dpi=300
    )
    plt.show()
    plt.close()

    print(
        "\nGraphiques enregistrés dans :",
        os.path.abspath(dossier_graphiques)
    )