import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

# 1. Chargement et preparation des donnees
df = pd.read_csv("OLD/train_data.csv", nrows=20000)

# Garder uniquement les experiences 2A3_MaP
df = df[df["experiment_type"] == "2A3_MaP"].copy()

#reactivite a la position 28
cible = "reactivity_0028"

# Garde les lignes avec une mesure disponible
df = df.dropna(subset=[cible]).copy()

# 2. Separation entrainement / validation par sequence

separateur = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

indices_train, indices_validation = next(
    separateur.split(df, groups=df["sequence"])
)

entrainement = df.iloc[indices_train].copy()
validation = df.iloc[indices_validation].copy()

# Verification de l'absence de sequences identiques entre les ensembles
sequences_train = set(entrainement["sequence"])
sequences_validation = set(validation["sequence"])

print("Sequences communes :",
      len(sequences_train & sequences_validation))
print("Lignes d'entrainement :", len(entrainement))
print("Lignes de validation :", len(validation))
print("Nombre total de lignes conservees :", len(df))

# 3. Fonction d'evaluation commune
y_train = entrainement[cible]
y_validation = validation[cible]

resultats = []

def evaluer(nom, predictions):
    """Calcule la MAE et ajoute le resultat au tableau comparatif."""
    score = mean_absolute_error(y_validation, predictions)
    resultats.append({
        "Methode": nom,
        "MAE": score
    })
    print(f"MAE - {nom} : {score:.6f}")
    return score

# 4. Baseline 1 : moyenne generale

moyenne_generale = y_train.mean()

predictions_moyenne = [moyenne_generale] * len(validation)

mae_moyenne = evaluer(
    "Moyenne generale",
    predictions_moyenne
)

print("Moyenne reelle en entrainement :", y_train.mean())
print("Moyenne reelle en validation :", y_validation.mean())
print("Minimum en entrainement :", y_train.min())
print("Maximum en entrainement :", y_train.max())


# 5. Baseline 2 : nucleotide a la position 28

#ca commence a compter les indices a zero 
# l'indice 27 correspond donc a la position 28.
nucleotide_train = entrainement["sequence"].str[27]
nucleotide_validation = validation["sequence"].str[27]

moyennes_nucleotide = y_train.groupby(nucleotide_train).mean()

predictions_nucleotide = (
    nucleotide_validation
    .map(moyennes_nucleotide)
    .fillna(moyenne_generale)
)

mae_nucleotide = evaluer(
    "Nucleotide position 28",
    predictions_nucleotide
)

print("\nMoyennes par nucleotide :")
print(moyennes_nucleotide)

# 6. Baseline 3 : motif des positions 27 a 29 (autour du 28)

motif_train = entrainement["sequence"].str[26:29]
motif_validation = validation["sequence"].str[26:29]

moyennes_motif = y_train.groupby(motif_train).mean()

predictions_motif = (
    motif_validation
    .map(moyennes_motif)
    .fillna(moyenne_generale)
)

mae_motif = evaluer(
    "Motif positions 27-29",
    predictions_motif
)

print("\nQuelques moyennes par motif :")
print(moyennes_motif.head(10))

amelioration = (mae_moyenne - mae_motif) / mae_moyenne * 100
print(f"\nAmelioration du motif vs moyenne : {amelioration:.2f} %")

# 7. Preparation des entrees pour Ridge

# Utilise les positions 26 a 30 autour de la cible ici 28.
positions = range(26, 31)

X_train = pd.DataFrame({
    f"pos_{pos}": entrainement["sequence"].str[pos - 1]
    for pos in positions
})

X_validation = pd.DataFrame({
    f"pos_{pos}": validation["sequence"].str[pos - 1]
    for pos in positions
})

# 8. Modele 4 : Ridge sans interactions

modele_ridge = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    Ridge(alpha=1.0)
)

modele_ridge.fit(X_train, y_train)
predictions_ridge = modele_ridge.predict(X_validation)

mae_ridge = evaluer(
    "Regression Ridge",
    predictions_ridge
)

# 9. Modele 5 : Ridge avec interactions

modele_interactions = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    PolynomialFeatures(
        degree=2,
        interaction_only=True,
        include_bias=False
    ),
    Ridge(alpha=1.0)
)

modele_interactions.fit(X_train, y_train)
predictions_interactions = modele_interactions.predict(X_validation)

mae_interactions = evaluer(
    "Ridge avec interactions",
    predictions_interactions
)

# 10. Comparaison finale

tableau_resultats = (
    pd.DataFrame(resultats)
    .sort_values("MAE")
    .reset_index(drop=True)
)

print("\nComparaison finale des modeles")
print(tableau_resultats.to_string(index=False))