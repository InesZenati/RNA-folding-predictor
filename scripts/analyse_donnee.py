import pandas as pd
from pathlib import Path

fichier = Path("OLD/train_data.csv")

# Colonnes contenant les mesures de réactivité,
# sans inclure les colonnes d'erreur de mesure.
colonnes = pd.read_csv(fichier, nrows=0).columns.tolist()

reactivites = [
    c for c in colonnes
    if c.startswith("reactivity_")
    and c.removeprefix("reactivity_").isdigit()
]

print("Nombre de colonnes de réactivité :", len(reactivites))
print("Colonnes de réactivité :", reactivites[:5])

# Statistiques calculées progressivement
nb_lignes = 0
experiences = {}
valeurs_manquantes = pd.Series(0, index=reactivites, dtype="int64")
total_cellules = 0

for bloc in pd.read_csv(fichier, chunksize=10_000):
    nb_lignes += len(bloc)

    # Compter les lignes par type d'expérience
    comptes = bloc["experiment_type"].value_counts()
    for nom, nombre in comptes.items():
        experiences[nom] = experiences.get(nom, 0) + int(nombre)

    # Compter les mesures manquantes
    valeurs_manquantes += bloc[reactivites].isna().sum()
    total_cellules += len(bloc) * len(reactivites)

print("\n--- Résultats ---")
print("Nombre total de lignes :", nb_lignes)
print("\nNombre de lignes par expérience :")
print(experiences)

nb_nan = int(valeurs_manquantes.sum())
print("\nNombre total de mesures manquantes :", nb_nan)
print("Pourcentage manquant :",
      round(100 * nb_nan / total_cellules, 2), "%")

print("\nColonnes les plus souvent manquantes :")
print(valeurs_manquantes.sort_values(ascending=False).head(10))


import pandas as pd
import numpy as np

fichier = "OLD/train_data.csv"

# Lire seulement 1 000 lignes
df = pd.read_csv(fichier, nrows=1000)

# Colonnes de réactivité numérique
reactivites = [
    c for c in df.columns
    if c.startswith("reactivity_")
    and c.removeprefix("reactivity_").isdigit()
]

print("Nombre de lignes étudiées :", len(df))
print("Nombre de colonnes de réactivité :", len(reactivites))

# Longueur des séquences
longueurs = df["sequence"].str.len()

print("\nLongueurs des séquences :")
print(longueurs.describe())

# Nombre de mesures disponibles par ligne
mesures_disponibles = df[reactivites].notna().sum(axis=1)

print("\nNombre de mesures disponibles par ligne :")
print(mesures_disponibles.describe())

# Afficher quelques exemples
print("\nExemples :")

for i in range(min(5, len(df))):
    print("\nSéquence :", df.loc[i, "sequence"][:50], "...")
    print("Longueur :", longueurs.iloc[i])
    print("Expérience :", df.loc[i, "experiment_type"])
    print("Mesures disponibles :", mesures_disponibles.iloc[i])

# Vérifier la présence des identifiants
print("\nColonnes d'identification :")
print([
    c for c in df.columns
    if c in ["sequence_id", "sequence", "experiment_type", "dataset_name"]
])



import pandas as pd

fichier = "OLD/train_data.csv"

# Charger seulement 1 000 lignes
df = pd.read_csv(fichier, nrows=1000)

# Identifier les colonnes de réactivité
reactivites = [
    c for c in df.columns
    if c.startswith("reactivity_")
    and c.removeprefix("reactivity_").isdigit()
]

# Étudier la première ligne
ligne = df.iloc[0]
sequence = ligne["sequence"]

positions_disponibles = [
    int(c.removeprefix("reactivity_"))
    for c in reactivites
    if pd.notna(ligne[c])
]

print("Type d'expérience :", ligne["experiment_type"])
print("Longueur de la séquence :", len(sequence))
print("Nombre de réactivités disponibles :", len(positions_disponibles))
print("Premières positions mesurées :", positions_disponibles[:20])
print("Dernières positions mesurées :", positions_disponibles[-20:])

# Comparaison sur les deux types d'expérience
print("\nMesures disponibles selon l'expérience :")

for nom, groupe in df.groupby("experiment_type"):
    mesures = groupe[reactivites].notna().sum(axis=1)
    print(nom)
    print("  Nombre de lignes :", len(groupe))
    print("  Moyenne de mesures disponibles :", mesures.mean())
    print("  Minimum :", mesures.min())
    print("  Maximum :", mesures.max())






