
import pandas as pd

# Charger 2 000 lignes
df = pd.read_csv("OLD/train_data.csv", nrows=2000)

# Garder uniquement les mesures 2A3_MaP
df = df[df["experiment_type"] == "2A3_MaP"].copy()

# Garder uniquement les lignes où la mesure existe
colonne = "reactivity_0027"
df = df.dropna(subset=[colonne])


# Séparer les données en deux parties
milieu = len(df) // 2
entrainement = df.iloc[:milieu]
validation = df.iloc[milieu:]

# Calculer la moyenne sur les données d'entraînement
prediction = entrainement[colonne].mean()

# Calculer une deuxième référence :
# la moyenne réelle de la validation
moyenne_validation = validation[colonne].mean()

# Calculer les deux MAE
mae_modele = (validation[colonne] - prediction).abs().mean()

mae_reference = (
    validation[colonne] - moyenne_validation
).abs().mean()

print("Prédiction du modèle :", prediction)
print("Moyenne de validation :", moyenne_validation)
print("MAE du modèle :", mae_modele)
print("MAE de référence :", mae_reference)

# Calculer la médiane sur les données d'entraînement
prediction_mediane = entrainement[colonne].median()

# Calculer son erreur sur la validation
mae_mediane = (
    validation[colonne] - prediction_mediane
).abs().mean()

print("Prédiction médiane :", prediction_mediane)
print("MAE de la médiane :", mae_mediane)