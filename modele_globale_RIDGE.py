"""
Comparaison de trois stratégies pour prédire la réactivité de l'ARN :
1. Modèle entraîné uniquement sur 2A3_MaP
2. Modèle entraîné uniquement sur DMS_MaP
3. Modèle global entraîné sur les deux expériences, avec le type d'expérience
   comme variable explicative.

Le script est autonome : il contient les imports, le chargement des données,
les fonctions, la séparation train/validation/test, l'entraînement et les graphiques.

Modifications par rapport à la version précédente :
- filtre des mesures bruyantes (SN_filter, ou signal_to_noise à défaut) et
  filtre optionnel sur l'erreur de mesure (ERREUR_MAX) ;
- bornage à [0, 1] des cibles et des prédictions, comme dans la métrique
  officielle de la compétition Stanford Ribonanza RNA Folding ;
- taille de fenêtre et alpha choisis sur la VALIDATION pour chaque stratégie
  (grille croisée), puis un seul passage sur le TEST ;
- le réglage de référence (7 nt, alpha 225) est conservé pour comparaison ;
- ÉVALUATION ÉQUITABLE : chaque stratégie est évaluée sur les mêmes mesures de
  test, expérience par expérience. Le modèle global est évalué séparément sur
  le test 2A3_MaP et sur le test DMS_MaP, pour être comparé au modèle
  spécialisé correspondant (avant, il était évalué sur les deux mélangés) ;
- la MAE « globale » (toutes les mesures regroupées) est ajoutée au résumé, en
  plus de la moyenne des MAE par position ;
- extraction des fenêtres plus rapide (plus de iterrows), et prétraitement
  one-hot + interactions calculé une seule fois par taille de fenêtre.

À lancer depuis le dossier du projet :
    python comparaison_2A3_DMS_modele_global.py

Adapte DATA_PATH si le fichier CSV se trouve ailleurs.
Conseil : fais un premier essai avec N_LIGNES = 20000 avant de lancer tout le fichier.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings(
    "ignore", message=".*encountered in matmul", category=RuntimeWarning
)
# 1. PARAMÈTRES

DATA_PATH = Path("data/train_data_QUICK_START.csv")
DOSSIER_SORTIE = Path("graphiques_comparaison")

RANDOM_STATE = 42

N_LIGNES = None

EXPERIENCES = ["2A3_MaP", "DMS_MaP"]

# Grille de réglage, explorée sur la validation uniquement.
# Les tailles doivent être impaires (1 = nucléotide seul, sans voisins).
TAILLES_A_TESTER = [1, 3, 7, 11, 15]
ALPHAS_A_TESTER = [1, 10, 30, 100, 300, 1000]

# Pour aller plus vite, la grille peut n'utiliser qu'une position sur N
# (par exemple 3). Le réglage final et le test utilisent toujours toutes les positions.
PAS_POSITIONS_GRILLE = 1

# Réglage de référence (de base)
TAILLE_REFERENCE = 7
ALPHA_REFERENCE = 225.0

# Filtre des mesures bruyantes.
# - Si la colonne SN_filter existe, on garde les lignes où SN_filter == 1.
# - Sinon, si la colonne signal_to_noise existe, on garde celles dont la valeur
#   est >= SEUIL_SIGNAL_BRUIT.
# Mettre FILTRER_MESURES_BRUYANTES = False pour désactiver le filtre.
FILTRER_MESURES_BRUYANTES = True
SEUIL_SIGNAL_BRUIT = 1.0

# Filtre optionnel sur l'erreur de mesure de chaque position (colonnes
# reactivity_error_XXXX). None = pas de filtre. Le script affiche les quantiles
# de l'erreur pour t'aider à choisir un seuil (par exemple le 90e centile).
ERREUR_MAX = None

# Bornes appliquées aux réactivités et aux prédictions (métrique officielle).
BORNE_MIN = 0.0
BORNE_MAX = 1.0

# Plus grande fenêtre utilisée : les observations retenues sont celles pour
# lesquelles cette fenêtre existe, ce qui rend toutes les tailles comparables.
TAILLE_MAX = max(TAILLES_A_TESTER + [TAILLE_REFERENCE])

# Stratégies : "experiences" = expériences utilisées pour l'entraînement, et
# aussi pour le test (chacune évaluée séparément).
SCENARIOS = {
    "2A3_MaP uniquement": {"experiences": ["2A3_MaP"], "avec_type": False},
    "DMS_MaP uniquement": {"experiences": ["DMS_MaP"], "avec_type": False},
    "Modèle global": {"experiences": EXPERIENCES, "avec_type": True},
}

ETAT_FILTRES = {"bruit": "non appliqué"}

# 2. FONCTIONS UTILITAIRES

def colonnes_de_reactivite(dataframe):
    """Retourne les colonnes reactivity_ sans les colonnes d'erreur."""
    return [
        colonne
        for colonne in dataframe.columns
        if colonne.startswith("reactivity_")
        and not colonne.startswith("reactivity_error_")
    ]


def liste_positions(colonnes_cibles):
    """Retourne [(position, colonne)] triée par position."""
    positions = []
    for cible in colonnes_cibles:
        try:
            positions.append((int(cible.replace("reactivity_", "")), cible))
        except ValueError:
            continue
    return sorted(positions)


def filtrer_mesures_bruyantes(dataframe):
    """
    Retire les lignes dont la mesure est jugée trop bruyante.
    Utilise SN_filter (== 1) si disponible, sinon signal_to_noise (>= seuil).
    Si aucune des deux colonnes n'existe, le dataframe est renvoyé comme ca.
    """
    if not FILTRER_MESURES_BRUYANTES:
        ETAT_FILTRES["bruit"] = "désactivé"
        print("Filtre des mesures bruyantes : désactivé.")
        return dataframe

    if "SN_filter" in dataframe.columns:
        sn_filter = pd.to_numeric(dataframe["SN_filter"], errors="coerce")
        masque = sn_filter == 1
        critere = "SN_filter == 1"
    elif "signal_to_noise" in dataframe.columns:
        signal = pd.to_numeric(dataframe["signal_to_noise"], errors="coerce")
        masque = signal >= SEUIL_SIGNAL_BRUIT
        critere = f"signal_to_noise >= {SEUIL_SIGNAL_BRUIT}"
    else:
        ETAT_FILTRES["bruit"] = "NON appliqué (colonne absente)"
        autres = [c for c in dataframe.columns if not c.startswith("reactivity")]
        print(
            "ATTENTION : ni SN_filter ni signal_to_noise dans le CSV, "
            "aucun filtre de bruit appliqué."
        )
        print("Colonnes hors réactivité disponibles :", autres)
        return dataframe

    avant = len(dataframe)
    dataframe_filtre = dataframe[masque].copy()
    apres = len(dataframe_filtre)

    ETAT_FILTRES["bruit"] = critere
    print(f"Filtre des mesures bruyantes ({critere}) :")
    print(f"  Lignes avant : {avant:,} | après : {apres:,} "
          f"| retirées : {avant - apres:,}")
    return dataframe_filtre


def decrire_filtres():
    """Résume les filtres réellement appliqués (pas seulement les paramètres)."""
    erreur = "aucun" if ERREUR_MAX is None else f"erreur <= {ERREUR_MAX}"
    return f"bruit : {ETAT_FILTRES['bruit']} | filtre d'erreur : {erreur}"


def afficher_quantiles_erreur(dataframe):
    """Aide au choix d'ERREUR_MAX : quantiles de l'erreur de mesure (échantillon)."""
    colonnes_erreur = [c for c in dataframe.columns
                       if c.startswith("reactivity_error_")]
    if not colonnes_erreur:
        return
    echantillon = dataframe.sample(
        n=min(len(dataframe), 5000), random_state=RANDOM_STATE
    )
    erreurs = pd.Series(echantillon[colonnes_erreur].to_numpy().ravel())
    erreurs = pd.to_numeric(erreurs, errors="coerce").dropna()
    quantiles = erreurs.quantile([0.5, 0.75, 0.9, 0.95]).round(3).to_dict()
    print(f"Quantiles de l'erreur de mesure (50/75/90/95 %) : {quantiles}")


def borner(valeurs):
    """Borne des valeurs (scalaire ou tableau) entre BORNE_MIN et BORNE_MAX."""
    return np.clip(valeurs, BORNE_MIN, BORNE_MAX)


def extraire_fenetre(sequence, position, taille=5):
    """
    Extrait une fenêtre centrée sur position (positions numérotées à partir de 1).
    Renvoie une liste de nucléotides, ou None si la fenêtre est hors limites
    ou contient des caractères autres que A, C, G et U.
    """
    if not isinstance(sequence, str):
        return None

    i = position - 1
    rayon = taille // 2
    debut = i - rayon
    fin = i + rayon + 1

    if debut < 0 or fin > len(sequence):
        return None

    fenetre = sequence[debut:fin]
    if len(fenetre) != taille:
        return None

    if any(nucleotide not in "ACGU" for nucleotide in fenetre):
        return None

    return list(fenetre)


def creer_pretraitement():
    """One-hot puis interactions de degré 2 (sans les carrés)."""
    return make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
    )


def creer_ridge(alpha):
    """Pipeline complet : one-hot, interactions, Ridge."""
    return make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
        Ridge(alpha=alpha),
    )


def predire(modele, X):
    """Prédit puis borne les prédictions dans [BORNE_MIN, BORNE_MAX]."""
    predictions = modele.predict(X)
    if not np.all(np.isfinite(predictions)):
        raise ValueError(
            "Prédictions non finies : vrai problème numérique dans le Ridge "
            "(augmente alpha ou réduis la taille de fenêtre)."
        )
    return borner(predictions)


def donnees_position(dataframe, cible, position):
    """
    Pour une position : fenêtres de TAILLE_MAX nucléotides centrées sur elle,
    cibles bornées à [0, 1] et types d'expérience, pour les lignes valides.
    Les fenêtres plus petites s'obtiennent ensuite en gardant les colonnes
    centrales (voir construire_X).
    """
    vide = (
        np.empty((0, TAILLE_MAX), dtype="U1"),
        np.array([], dtype=float),
        np.array([], dtype=object),
    )

    if cible not in dataframe.columns:
        return vide

    colonnes = ["sequence", "experiment_type", cible]
    colonne_erreur = cible.replace("reactivity_", "reactivity_error_")
    utiliser_erreur = ERREUR_MAX is not None and colonne_erreur in dataframe.columns
    if utiliser_erreur:
        colonnes.append(colonne_erreur)

    sous = dataframe[colonnes].dropna(
        subset=["sequence", "experiment_type", cible]
    )
    if utiliser_erreur:
        erreurs = pd.to_numeric(sous[colonne_erreur], errors="coerce")
        sous = sous[erreurs <= ERREUR_MAX]

    valeurs = pd.to_numeric(sous[cible], errors="coerce").to_numpy(dtype=float)
    sequences = sous["sequence"].to_numpy()
    types = sous["experiment_type"].astype(str).to_numpy()

    fenetres, ys, ts = [], [], []
    for sequence, valeur, type_exp in zip(sequences, valeurs, types):
        if not np.isfinite(valeur):
            continue
        fenetre = extraire_fenetre(sequence, position, TAILLE_MAX)
        if fenetre is None:
            continue
        fenetres.append(fenetre)
        ys.append(valeur)
        ts.append(type_exp)

    if not ys:
        return vide

    return (
        np.asarray(fenetres, dtype="U1"),
        borner(np.asarray(ys, dtype=float)),
        np.asarray(ts, dtype=object),
    )


def selectionner(donnees, experiences):
    """Garde les lignes dont le type d'expérience figure dans `experiences`."""
    X, y, types = donnees
    masque = np.isin(types, experiences)
    return X[masque], y[masque], types[masque]


def construire_X(X_max, types, taille, avec_type):
    """
    Réduit les fenêtres de TAILLE_MAX à `taille` nucléotides (colonnes centrales)
    et ajoute, si demandé, le type d'expérience comme variable explicative.
    """
    centre = TAILLE_MAX // 2
    rayon = taille // 2
    X = X_max[:, centre - rayon: centre + rayon + 1]
    if avec_type:
        X = np.column_stack([X, types.astype(str)])
    return X


def separer_train_validation_test(dataframe):
    """
    Sépare les données par séquences, afin qu'une même séquence ne soit pas
    présente à la fois dans train, validation et test.
    """
    if "sequence" not in dataframe.columns:
        raise ValueError("Le CSV doit contenir une colonne 'sequence'.")

    split_test = GroupShuffleSplit(
        n_splits=1, test_size=0.15, random_state=RANDOM_STATE
    )
    indices_train_validation, indices_test = next(
        split_test.split(dataframe, groups=dataframe["sequence"])
    )

    train_validation = dataframe.iloc[indices_train_validation].copy()
    test = dataframe.iloc[indices_test].copy()

    split_validation = GroupShuffleSplit(
        n_splits=1, test_size=0.1765, random_state=RANDOM_STATE + 1
    )
    indices_train, indices_validation = next(
        split_validation.split(
            train_validation, groups=train_validation["sequence"]
        )
    )

    entrainement = train_validation.iloc[indices_train].copy()
    validation = train_validation.iloc[indices_validation].copy()

    return entrainement, validation, test

# 3. RÉGLAGE DE LA FENÊTRE ET D'ALPHA (VALIDATION UNIQUEMENT)

def rechercher_reglages(entrainement, validation, colonnes_cibles):
    """
    Pour chaque stratégie, teste toutes les combinaisons (taille de fenêtre,
    alpha) et garde celle qui a la plus petite MAE sur la validation.
    La MAE est calculée sur toutes les mesures regroupées (cibles et
    prédictions bornées à [0, 1]).
    Renvoie {stratégie: {"réglé": (taille, alpha), "référence": (taille, alpha)}}
    et le tableau complet de la grille.
    """
    print("\n")
    print("RÉGLAGE : TAILLE DE FENÊTRE x ALPHA (validation uniquement)")

    positions = liste_positions(colonnes_cibles)[::PAS_POSITIONS_GRILLE]
    somme_erreurs = {nom: {} for nom in SCENARIOS}
    compte = {nom: 0 for nom in SCENARIOS}

    for numero, (position, cible) in enumerate(positions, start=1):
        d_train = donnees_position(entrainement, cible, position)
        d_val = donnees_position(validation, cible, position)

        for nom, scenario in SCENARIOS.items():
            X_tr, y_tr, t_tr = selectionner(d_train, scenario["experiences"])
            X_va, y_va, t_va = selectionner(d_val, scenario["experiences"])
            if len(y_tr) == 0 or len(y_va) == 0:
                continue

            compte[nom] += len(y_va)

            for taille in TAILLES_A_TESTER:
                A = construire_X(X_tr, t_tr, taille, scenario["avec_type"])
                B = construire_X(X_va, t_va, taille, scenario["avec_type"])

                # Le prétraitement ne dépend pas d'alpha : on le fait une fois.
                pretraitement = creer_pretraitement()
                A_t = pretraitement.fit_transform(A)
                B_t = pretraitement.transform(B)

                for alpha in ALPHAS_A_TESTER:
                    ridge = Ridge(alpha=alpha)
                    ridge.fit(A_t, y_tr)
                    pred = predire(ridge, B_t)
                    cle = (taille, alpha)
                    somme_erreurs[nom][cle] = (
                        somme_erreurs[nom].get(cle, 0.0)
                        + float(np.abs(y_va - pred).sum())
                    )

        if numero % 20 == 0:
            print(f"  Grille : {numero}/{len(positions)} positions traitées")

    lignes = []
    for nom in SCENARIOS:
        if compte[nom] == 0:
            continue
        for (taille, alpha), somme in somme_erreurs[nom].items():
            lignes.append({
                "strategie": nom,
                "taille_fenetre": taille,
                "alpha": alpha,
                "MAE_validation": somme / compte[nom],
                "nombre_observations": compte[nom],
            })

    grille = pd.DataFrame(lignes)
    if grille.empty:
        raise ValueError(
            "La grille est vide : aucune observation valide pour la fenêtre "
            f"de {TAILLE_MAX} nucléotides. Vérifie les données."
        )

    grille.to_csv(DOSSIER_SORTIE / "grille_reglages_validation.csv", index=False)

    reglages = {}
    for nom in SCENARIOS:
        sous = grille[grille["strategie"] == nom]
        if sous.empty:
            continue

        print(f"\n{nom} : MAE de validation (lignes : fenêtre, colonnes : alpha)")
        print(
            sous.pivot(
                index="taille_fenetre", columns="alpha", values="MAE_validation"
            ).to_string(float_format=lambda x: f"{x:.5f}")
        )

        meilleur = sous.loc[sous["MAE_validation"].idxmin()]
        taille = int(meilleur["taille_fenetre"])
        alpha = float(meilleur["alpha"])
        print(f"-> retenu : fenêtre = {taille}, alpha = {alpha} "
              f"(MAE validation = {meilleur['MAE_validation']:.5f})")

        if taille == max(TAILLES_A_TESTER):
            print("   NB : c'est la plus grande fenêtre testée, "
                  "essaie d'ajouter des tailles plus grandes.")
        if alpha in (min(ALPHAS_A_TESTER), max(ALPHAS_A_TESTER)):
            print("   NB : alpha est au bord de la grille, "
                  "élargis ALPHAS_A_TESTER de ce côté.")

        reglages[nom] = {
            "réglé": (taille, alpha),
            "référence": (TAILLE_REFERENCE, ALPHA_REFERENCE),
        }

    return reglages, grille

# 4. ÉVALUATION FINALE SUR LE TEST

def evaluer_sur_test(entrainement, test, colonnes_cibles, reglages):
    """
    Pour chaque stratégie et chaque réglage (réglé / référence) : entraîne un
    modèle par position sur l'entraînement, puis calcule la MAE (cibles et
    prédictions bornées) sur le test, expérience par expérience.
    Toutes les stratégies sont évaluées sur les mêmes mesures de test.
    """
    print("\n")
    print("ÉVALUATION FINALE SUR LE JEU DE TEST")


    positions = liste_positions(colonnes_cibles)
    resultats = []

    for numero, (position, cible) in enumerate(positions, start=1):
        d_train = donnees_position(entrainement, cible, position)
        d_test = donnees_position(test, cible, position)

        for nom, scenario in SCENARIOS.items():
            if nom not in reglages:
                continue

            X_tr, y_tr, t_tr = selectionner(d_train, scenario["experiences"])
            if len(y_tr) == 0:
                continue

            for etiquette, (taille, alpha) in reglages[nom].items():
                A = construire_X(X_tr, t_tr, taille, scenario["avec_type"])
                modele = creer_ridge(alpha)
                modele.fit(A, y_tr)

                # Évaluation séparée sur chaque expérience du test.
                for experience in scenario["experiences"]:
                    X_te, y_te, t_te = selectionner(d_test, [experience])
                    if len(y_te) == 0:
                        continue

                    B = construire_X(X_te, t_te, taille, scenario["avec_type"])
                    pred = predire(modele, B)

                    resultats.append({
                        "experience_test": experience,
                        "strategie": nom,
                        "reglage": etiquette,
                        "taille_fenetre": taille,
                        "alpha": alpha,
                        "position": position,
                        "nombre_observations": len(y_te),
                        "MAE": mean_absolute_error(y_te, pred),
                    })

        if numero % 20 == 0:
            print(f"  Test : {numero}/{len(positions)} positions traitées")

    return pd.DataFrame(resultats)


def resumer(resultats):
    """
    Résume les MAE par (expérience de test, stratégie, réglage).
    MAE_globale = MAE sur toutes les mesures regroupées (pondérée par le nombre
    d'observations de chaque position) ; MAE_moyenne_positions = moyenne simple
    des MAE par position.
    """
    resultats = resultats.copy()
    resultats["somme_erreurs"] = resultats["MAE"] * resultats["nombre_observations"]

    resume = (
        resultats.groupby(
            ["experience_test", "strategie", "reglage"], as_index=False
        )
        .agg(
            somme_erreurs=("somme_erreurs", "sum"),
            nombre_observations=("nombre_observations", "sum"),
            MAE_moyenne_positions=("MAE", "mean"),
            MAE_mediane_positions=("MAE", "median"),
            nombre_positions=("position", "nunique"),
            taille_fenetre=("taille_fenetre", "first"),
            alpha=("alpha", "first"),
        )
    )
    resume["MAE_globale"] = resume["somme_erreurs"] / resume["nombre_observations"]
    resume = resume.drop(columns="somme_erreurs")
    return resume.sort_values(["experience_test", "MAE_globale"]).reset_index(drop=True)

# 5. GRAPHIQUES

def tracer_graphiques(resultats, resume, grille):
    # Graphique 1 : MAE globale par stratégie, une figure par expérience de test
    fig, axes = plt.subplots(1, len(EXPERIENCES), figsize=(13, 6), sharey=True)
    for ax, experience in zip(np.atleast_1d(axes), EXPERIENCES):
        sous = resume[resume["experience_test"] == experience]
        etiquettes = [
            f"{ligne.strategie}\n({ligne.reglage})" for ligne in sous.itertuples()
        ]
        ax.bar(etiquettes, sous["MAE_globale"])
        ax.set_title(f"Test sur {experience}")
        ax.tick_params(axis="x", labelrotation=20)
        for etiquette in ax.get_xticklabels():
            etiquette.set_horizontalalignment("right")
    np.atleast_1d(axes)[0].set_ylabel("MAE globale sur le test (plus petit = meilleur)")
    fig.suptitle("Comparaison des stratégies (mêmes mesures de test pour toutes)")
    fig.tight_layout()
    fig.savefig(DOSSIER_SORTIE / "mae_globale_par_strategie.png", dpi=150)
    plt.show()

    # Graphique 2 : distribution des MAE par position (réglage retenu)
    fig, axes = plt.subplots(1, len(EXPERIENCES), figsize=(13, 6), sharey=True)
    for ax, experience in zip(np.atleast_1d(axes), EXPERIENCES):
        donnees, etiquettes = [], []
        for nom, scenario in SCENARIOS.items():
            if experience not in scenario["experiences"]:
                continue
            valeurs = resultats.loc[
                (resultats["experience_test"] == experience)
                & (resultats["strategie"] == nom)
                & (resultats["reglage"] == "réglé"),
                "MAE",
            ].dropna().values
            if len(valeurs) > 0:
                donnees.append(valeurs)
                etiquettes.append(nom)
        if donnees:
            ax.boxplot(donnees)
            ax.set_xticks(range(1, len(etiquettes) + 1))
            ax.set_xticklabels(etiquettes, rotation=15, ha="right")
        ax.set_title(f"Test sur {experience}")
    np.atleast_1d(axes)[0].set_ylabel("MAE par position")
    fig.suptitle("Distribution des erreurs selon la stratégie (réglage retenu)")
    fig.tight_layout()
    fig.savefig(DOSSIER_SORTIE / "distribution_mae_par_strategie.png", dpi=150)
    plt.show()

    # Graphique 3 : MAE par position (réglage retenu)
    fig, axes = plt.subplots(len(EXPERIENCES), 1, figsize=(14, 9), sharex=True)
    for ax, experience in zip(np.atleast_1d(axes), EXPERIENCES):
        sous = resultats[
            (resultats["experience_test"] == experience)
            & (resultats["reglage"] == "réglé")
        ]
        pivot = sous.pivot_table(
            index="position", columns="strategie", values="MAE", aggfunc="mean"
        )
        for nom in pivot.columns:
            ax.plot(pivot.index, pivot[nom], label=nom, alpha=0.8)
        ax.set_title(f"MAE par position - test sur {experience}")
        ax.set_ylabel("MAE")
        ax.legend()
    np.atleast_1d(axes)[-1].set_xlabel("Position dans la séquence")
    fig.tight_layout()
    fig.savefig(DOSSIER_SORTIE / "mae_par_position.png", dpi=150)
    plt.show()

    # Graphique 4 : effet des nucléotides voisins (MAE de validation selon la fenêtre)
    plt.figure(figsize=(9, 5.5))
    for nom in SCENARIOS:
        sous = grille[grille["strategie"] == nom]
        if sous.empty:
            continue
        meilleur = sous.groupby("taille_fenetre")["MAE_validation"].min()
        plt.plot(meilleur.index, meilleur.values, marker="o", label=nom)
    plt.xlabel("Taille de la fenêtre (nucléotides, position cible au centre)")
    plt.ylabel("MAE de validation (meilleur alpha pour chaque taille)")
    plt.title("Effet des nucléotides voisins")
    plt.xticks(sorted(grille["taille_fenetre"].unique()))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(DOSSIER_SORTIE / "mae_selon_fenetre.png", dpi=150)
    plt.show()

# 6. PROGRAMME PRINCIPAL

def main():
    if any(t % 2 == 0 for t in TAILLES_A_TESTER + [TAILLE_REFERENCE]):
        raise ValueError("Les tailles de fenêtre doivent être impaires.")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATA_PATH.resolve()}\n"
            "Modifie DATA_PATH au début du script pour indiquer le bon chemin."
        )

    print("Chargement du fichier CSV...")
    if N_LIGNES is None:
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH, nrows=N_LIGNES)

    colonnes_obligatoires = {"sequence", "experiment_type"}
    manquantes = colonnes_obligatoires - set(df.columns)
    if manquantes:
        raise ValueError(f"Colonnes obligatoires absentes du CSV : {manquantes}")

    df = df[df["experiment_type"].isin(EXPERIENCES)].copy()
    df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()

    # Retrait des mesures bruyantes avant tout découpage.
    df = filtrer_mesures_bruyantes(df)
    if df.empty:
        raise ValueError(
            "Plus aucune ligne après le filtre de bruit. "
            "Vérifie SN_filter / signal_to_noise ou désactive le filtre."
        )

    if ERREUR_MAX is None:
        afficher_quantiles_erreur(df)

    colonnes_cibles = colonnes_de_reactivite(df)
    if not colonnes_cibles:
        raise ValueError(
            "Aucune colonne reactivity_ trouvée. Vérifie le nom des colonnes du CSV."
        )

    print(f"Nombre de lignes chargées : {len(df):,}")
    print("Répartition des expériences :")
    print(df["experiment_type"].value_counts().to_string())
    print(f"Nombre de colonnes de réactivité : {len(colonnes_cibles)}")

    entrainement, validation, test = separer_train_validation_test(df)

    print("\nSéparation par séquences :")
    print(f"  Entraînement : {len(entrainement):,} lignes")
    print(f"  Validation   : {len(validation):,} lignes")
    print(f"  Test         : {len(test):,} lignes")

    for nom_ensemble, ensemble in [("entraînement", entrainement),
                                   ("validation", validation), ("test", test)]:
        for experience in EXPERIENCES:
            if (ensemble["experiment_type"] == experience).sum() == 0:
                raise ValueError(
                    f"Aucune ligne {experience} dans l'ensemble {nom_ensemble}. "
                    "Vérifie experiment_type, N_LIGNES et le filtre de bruit."
                )

    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

    # Réglage sur la validation, puis un seul passage sur le test.
    reglages, grille = rechercher_reglages(entrainement, validation, colonnes_cibles)

    print("\nRéglages retenus :")
    for nom, valeurs in reglages.items():
        taille, alpha = valeurs["réglé"]
        print(f"  {nom} : fenêtre = {taille}, alpha = {alpha}")

    resultats = evaluer_sur_test(entrainement, test, colonnes_cibles, reglages)
    if resultats.empty:
        raise ValueError("Aucun résultat de test : vérifie les données.")

    chemin_csv = DOSSIER_SORTIE / "resultats_comparaison.csv"
    resultats.to_csv(chemin_csv, index=False)

    resume = resumer(resultats)
    chemin_resume = DOSSIER_SORTIE / "resume_comparaison.csv"
    resume.to_csv(chemin_resume, index=False)

    print("\n RÉSULTATS ")
    print(f"Bornes : [{BORNE_MIN}, {BORNE_MAX}] | Filtres : {decrire_filtres()}")
    print("MAE_globale = toutes les mesures regroupées ; chaque stratégie est "
          "évaluée sur les mêmes mesures de test.\n")
    print(resume.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # Apport du réglage par rapport à la référence
    pivot = resume.pivot_table(
        index=["experience_test", "strategie"],
        columns="reglage",
        values="MAE_globale",
    )
    if {"référence", "réglé"}.issubset(pivot.columns):
        pivot["gain_%"] = (pivot["référence"] - pivot["réglé"]) / pivot["référence"] * 100
        print("\nApport du réglage (MAE globale de test) :")
        print(pivot.to_string(float_format=lambda x: f"{x:.5f}"))

    print(f"\nRésultats détaillés enregistrés dans : {chemin_csv.resolve()}")
    print(f"Résumé enregistré dans : {chemin_resume.resolve()}")

    tracer_graphiques(resultats, resume, grille)

    print(f"\nGraphiques enregistrés dans : {DOSSIER_SORTIE.resolve()}")


if __name__ == "__main__":
    main()