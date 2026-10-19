
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# 1. Charger un échantillon réel du concours
df = pd.read_csv("OLD/train_data.csv", nrows=2000)

# 2. Garder uniquement les expériences 2A3_MaP
df = df[df["experiment_type"] == "2A3_MaP"].copy()

# 3. Choisir la réactivité à prédire
colonne = "reactivity_0028"

# 4. Garder les lignes pour lesquelles cette mesure existe
df = df.dropna(subset=[colonne])

# 5. Séparer les données en entraînement et validation
entrainement, validation = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

# 6. Calculer la moyenne sur les données d'entraînement
prediction = entrainement[colonne].mean()

# 7. Évaluer les prédictions sur les données de validation
predictions = [prediction] * len(validation)

mae = mean_absolute_error(
    validation[colonne],
    predictions
)

# 8. Afficher les résultats
print("Nombre de lignes d'entraînement :", len(entrainement))
print("Nombre de lignes de validation :", len(validation))
print("Réactivité moyenne prédite :", prediction)
print("MAE sur la validation :", mae)

# 9. Vérifier les données utilisées
print("Moyenne réelle en entraînement :", entrainement[colonne].mean())
print("Moyenne réelle en validation :", validation[colonne].mean())
print("Minimum en entraînement :", entrainement[colonne].min())
print("Maximum en entraînement :", entrainement[colonne].max())
print("Nombre de valeurs conservées :", len(df))



# 10. Tester une prédiction basée sur le nucléotide en position 28

# Récupérer le nucléotide à la position 28 de chaque séquence
entrainement = entrainement.copy()
validation = validation.copy()

entrainement["nucleotide_28"] = entrainement["sequence"].str[27]
validation["nucleotide_28"] = validation["sequence"].str[27]

# Calculer la réactivité moyenne pour chaque nucléotide
moyennes_par_nucleotide = entrainement.groupby(
    "nucleotide_28"
)[colonne].mean()

print("\nMoyennes par nucléotide en position 28 :")
print(moyennes_par_nucleotide)

# Prédire selon le nucléotide observé en validation
predictions_nucleotide = validation["nucleotide_28"].map(
    moyennes_par_nucleotide
)

# Utiliser la moyenne générale si un nucléotide n'a pas été vu
predictions_nucleotide = predictions_nucleotide.fillna(prediction)

# Calculer la nouvelle MAE
mae_nucleotide = mean_absolute_error(
    validation[colonne],
    predictions_nucleotide
)

print("MAE avec le nucléotide :", mae_nucleotide)

