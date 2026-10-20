
from optparse import OptionParser
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Chemin du dataset et répertoire de résultats
DATA_DIR = Path("/home/caop/RNA_ribonanza/small_data")
#file_path = DATA_DIR/"/home/caop/RNA_ribonanza/small_data/train_data_QUICK_START.csv"
FIGURES_DIR = Path("/home/caop/RNA_ribonanza/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_data(file_path):
    """Charger le dataset."""
    df = pd.read_csv(file_path)
    print("Dimensions du dataset :", df.shape) # (335616 lignes x 416 colonnes)
    print("Nombre de lignes :", df.shape[0])
    print("Nombre de colonnes :", df.shape[1])
    print("Nom de colonnes :", df.columns.tolist())
    print("Premières échantillons :", df.head())
    return df

def explore_metadata(df):
    """Explorer les métadonnées des séquences."""
    # Metadata (str) : 'sequence_id', 'sequence', 'experiment_type', 'dataset_name'
    
    print("Nombre de séquences ID uniques :", df["sequence_id"].nunique())
    # 167808 séquence unique -> 2 lignes / séquence
    # chaque ligne contient des valeurs de réactivité pour une molécule 
    print("Nombre de séquences uniques :", df["sequence"].nunique())
    print("Valeur de experiment_type\n", df["experiment_type"].value_counts())
    print("Valeur de dataset_name;", df["dataset_name"].value_counts())
    df["sequence_length"] = df["sequence"].str.len()
    print(df["sequence_length"].describe())

# Réactivité (float) : reactivity_i, i va de 1 à 206 
# Valeurs de réactivité chimique mesurées pour chaque nucléotide de l’ARN. 
# Ce sont les variables à prédire (Y). 
# Les valeurs sont normalisées

# Réactivitité erreur (float) : reactivity_error_i, i va de 1 à 206
# Estimations des incertitudes expérimentales associées aux valeurs de réactivité
# calculées à partir des statistiques de séquençage. 
# Elles renseignent sur la fiabilité des mesures.

def get_reactivity_columns(df):
    """Récupérer les colonnes de réactivité dans l'ordre numérique."""

    reactivity_cols = []
    for col in df.columns:
        if col.startswith("reactivity_"):
            if not col.startswith("reactivity_error_"):
                reactivity_cols.append(col)

    reactivity_cols = sorted(reactivity_cols, 
                             key=lambda col: int(col.split("_")[-1]))
    return reactivity_cols


def analyze_sequence_lengths(df):
    """Analyser la longueur des séquences ARN."""
    sequences = df.drop_duplicates("sequence_id").copy()
    sequences["sequence_length"] = sequences["sequence"].str.len()
    print("\nStatistiques des longueurs :")
    print(sequences["sequence_length"].describe())
    length_counts = (sequences["sequence_length"].value_counts().sort_index())

    print("\nNombre de séquences par longueur :")
    print(length_counts)
    total = length_counts.sum()
    count_177 = length_counts.get(177, 0)

    print("Séquences de 177 bases :", count_177)
    print(f"Pourcentage : {count_177 / total * 100:.2f}%")

    return length_counts


def plot_sequence_lengths(length_counts):
    """Tracer la distribution des longueurs des séquences."""
    plt.figure(figsize=(12, 5))
    length_counts.plot(kind="bar")
    plt.xlabel("Longueur de la séquence")
    plt.ylabel("Nombre de séquences")
    plt.title("Distribution des longueurs des séquences ARN")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/"distribution_longueurs.png", dpi=300)
    


def plot_reactivity(df, reactivity_cols):
    """Tracer les profils de réactivité DMS et 2A3."""

    # Sélectionner la première séquence
    seq_id = df["sequence_id"].iloc[0]
    # Récupérer les lignes associées à cette séquence
    deux_lignes = df[df["sequence_id"] == seq_id]
    sequence = deux_lignes["sequence"].iloc[0]
    length = len(sequence)

    print("Sequence ID :", seq_id)
    print("Longueur :", length)
    print("Séquence :", sequence)

    print("Lignes de cette séquence :")
    print(deux_lignes)

    # Tracer les profils de réactivité
    plt.figure(figsize=(12, 5))

    for _, row in deux_lignes.iterrows():
        y = row[reactivity_cols[:length]].to_numpy(dtype=float)
        plt.plot(range(1, length + 1), y, label=row["experiment_type"])
    plt.xlabel("Position du nucléotide")
    plt.ylabel("Réactivité chimique")
    plt.title("Réactivité DMS et 2A3 d'une séquence ARN")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "reactivite_premiere_sequence.png", dpi=300)
    

def analyze_missing_values(df):
    """Analyser les valeurs manquantes dans les réactivités."""

    # Récupérer les colonnes de réactivité
    reactivity_cols = get_reactivity_columns(df)

    # Nombre de valeurs manquantes par colonne
    missing_counts = df[reactivity_cols].isna().sum()

    # Pourcentage de valeurs manquantes par colonne
    missing_percent = df[reactivity_cols].isna().mean() * 100

    print("Nombre total de valeurs manquantes :")
    print(missing_counts.sum())

    # Sélectionner les colonnes contenant 100 % de NaN
    all_nan_columns = missing_percent[missing_percent == 100]

    print("\nColonnes contenant 100 % de NaN :")
    print(all_nan_columns.to_string())

    print("\nNombre de colonnes contenant 100 % de NaN :")
    print(len(all_nan_columns))

    return missing_percent

def analyze_internal_missing_values(df):

    # Récupérer les colonnes de réactivité
    reactivity_cols = get_reactivity_columns(df)

    # Sélectionner les positions internes (27 à 165)
    internal_cols = []

    for col in reactivity_cols:
        position = int(col.split("_")[-1])

        if 27 <= position <= 165:
            internal_cols.append(col)

    # Calculer le pourcentage de NaN par position
    missing_percent = df[internal_cols].isna().mean() * 100

    # Afficher les statistiques
    print("Nombre de positions internes :", len(internal_cols))
    print(f"Pourcentage moyen de NaN : {missing_percent.mean():.2f}%")
    print(f"Pourcentage minimum de NaN : {missing_percent.min():.2f}%")
    print(f"Pourcentage maximum de NaN : {missing_percent.max():.2f}%")

    # Récupérer les numéros des positions
    positions = []

    for col in missing_percent.index:
        position = int(col.split("_")[-1])
        positions.append(position)

    # Créer les graphiques
    #fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Graphique 1 : pourcentage de NaN par position
    plt.figure(figsize=(12, 5))

    plt.plot(positions, missing_percent.values)
    plt.xlabel("Position dans la séquence ARN")
    plt.ylabel("Pourcentage de NaN (%)")
    plt.title("Valeurs manquantes par position")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pourcentage_nan_par_positions.png", dpi=300)
    

    # Graphique 2 : distribution des pourcentages de NaN
    plt.figure(figsize=(8, 5))

    plt.hist(
        missing_percent.values,
        bins=np.arange(0, 111, 10),
        edgecolor="black"
    )

    plt.xlabel("Pourcentage de NaN (%)")
    plt.ylabel("Nombre de positions")
    plt.title("Distribution des valeurs manquantes")
    plt.xticks(np.arange(0, 101, 10))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "distribution_pourcentages_NaN.png", dpi=300)
    

    return missing_percent

def analyze_missing_by_experiment(df):

    reactivity_cols = get_reactivity_columns(df)

    internal_cols = []

    for col in reactivity_cols:
        position = int(col.split("_")[-1])

        if 27 <= position <= 165:
            internal_cols.append(col)

    for experiment, group in df.groupby("experiment_type"):

        missing_percent = group[internal_cols].isna().mean() * 100

        positions = []

        for col in missing_percent.index:
            positions.append(int(col.split("_")[-1]))

        if experiment == "DMS_MaP":
            plt.plot(
                positions,
                missing_percent.values,
                label=experiment,
                color="green",
                linestyle="-",
                linewidth=2
            )
        else:
            plt.plot(
                positions,
                missing_percent.values,
                label=experiment,
                color="orange",
                linestyle="--",
                linewidth=2
            )

    plt.xlabel("Position dans la séquence ARN")
    plt.ylabel("Pourcentage de NaN (%)")
    plt.title("Valeurs manquantes : DMS vs 2A3")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "valeurs_manquantes_DMS_2A3_NaN.png", dpi=300)

def compare_missing_experiments(df):

    reactivity_cols = get_reactivity_columns(df)

    internal_cols = []

    for col in reactivity_cols:
        position = int(col.split("_")[-1])

        if 27 <= position <= 165:
            internal_cols.append(col)

    dms = df[df["experiment_type"] == "DMS_MaP"]
    a3 = df[df["experiment_type"] == "2A3_MaP"]

    dms_missing = dms[internal_cols].isna().mean() * 100
    a3_missing = a3[internal_cols].isna().mean() * 100

    differences = (dms_missing - a3_missing).abs()

    print("Différence moyenne :", differences.mean())
    print("Différence maximale :", differences.max())

    print("\nPositions avec des différences :")
    print(differences[differences > 0].to_string())

def compare_nan_masks(df):

    reactivity_cols = get_reactivity_columns(df)

    # Séparer les expériences
    dms = df[df["experiment_type"] == "DMS_MaP"]
    a3 = df[df["experiment_type"] == "2A3_MaP"]

    # Aligner les lignes par sequence_id
    dms = dms.set_index("sequence_id")
    a3 = a3.set_index("sequence_id")

    common_ids = dms.index.intersection(a3.index)

    dms = dms.loc[common_ids, reactivity_cols]
    a3 = a3.loc[common_ids, reactivity_cols]

    # Comparer les positions NaN
    dms_mask = dms.isna()
    a3_mask = a3.isna()

    differences = dms_mask != a3_mask

    print("Nombre de séquences comparées :", len(common_ids))
    print("Nombre de positions avec des masques différents :")
    print(differences.sum().sum())

    print("Nombre de séquences avec au moins une différence :")
    print(differences.any(axis=1).sum())

def create_reactivity_mask(df):

    # Récupérer les colonnes de réactivité
    reactivity_cols = get_reactivity_columns(df)

    # Extraire les réactivités
    reactivity = df[reactivity_cols].to_numpy(dtype=np.float32)

    # Créer le masque
    # True = valeur disponible
    # False = valeur NaN
    mask = ~np.isnan(reactivity)

    # Remplacer les NaN par 0 pour les calculs futurs
    reactivity_clean = np.nan_to_num(reactivity, nan=0.0)

    print("Dimensions des réactivités :", reactivity.shape)
    print("Dimensions du masque :", mask.shape)

    print("Nombre de mesures disponibles :", mask.sum())
    print("Nombre de mesures manquantes :", (~mask).sum())

    return reactivity_clean, mask

def plot_all_missing_values(df):
    """Tracer le pourcentage de NaN pour les 206 positions."""

    # Récupérer toutes les colonnes de réactivité
    reactivity_cols = get_reactivity_columns(df)

    # Calculer le pourcentage de NaN par position
    missing_percent = df[reactivity_cols].isna().mean() * 100

    # Récupérer les numéros des positions
    positions = []

    for col in reactivity_cols:
        position = int(col.split("_")[-1])
        positions.append(position)

    # Créer le graphique
    plt.figure(figsize=(12, 5))
    plt.bar(positions, missing_percent.values)
    plt.xlabel("Position dans la séquence ARN")
    plt.ylabel("Pourcentage de NaN (%)")
    plt.title("Valeurs manquantes par position (1 à 206)")
    plt.xticks(range(0, 207, 20))
    plt.ylim(0, 105)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Sauvegarder le graphique
    plt.savefig(FIGURES_DIR / "pourcentage_nan_toutes_positions.png",dpi=300)
    plt.close()

def main():
    """Analyse exploratoire."""
    # Argument
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="file_path", default=str(DATA_DIR/"train_data_QUICK_START.csv"), help="Chemin vers le fichier CSV à analyser")
    options, args = parser.parse_args()

    file_path = Path(options.file_path)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Charger le dataset
    df = load_data(file_path)

    # Explorer les métadonnées
    explore_metadata(df)

    # Analyser les longueurs des séquences
    length_counts = analyze_sequence_lengths(df)

    # Tracer la distribution des longueurs
    plot_sequence_lengths(length_counts)

    # Récupérer les colonnes de réactivité
    reactivity_cols = get_reactivity_columns(df)

    # Tracer les profils de réactivité
    plot_reactivity(df, reactivity_cols)

    # Missing values
    missing_percent = analyze_missing_values(df)
    internal_missing_percent = analyze_internal_missing_values(df)
    analyze_missing_by_experiment(df)
    compare_missing_experiments(df)
    compare_nan_masks(df)
    reactivity_clean, reactivity_mask = create_reactivity_mask(df)
    plot_all_missing_values(df)

if __name__ == "__main__":
    main()