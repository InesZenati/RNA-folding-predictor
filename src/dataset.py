import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from embedding import pad_sequences

# Types d'expérience
experiment_vocab = {"DMS_MaP": 0, "2A3_MaP": 1}

# Récupérer les colonnes de réactivité dans le bon ordre
def get_reactivity_columns(df):
    columns = []

    for i in range(1, 207):
        col = f"reactivity_{i:04d}"
        if col in df.columns:
            columns.append(col)

    return columns


# Dataset ARN
class RNADataset(Dataset):

    def __init__(self, df, reactivity_cols):
        self.df = df.reset_index(drop=True)
        self.reactivity_cols = reactivity_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        sequence = row["sequence"]
        experiment_type = experiment_vocab[row["experiment_type"]]

        # Valeurs de réactivité
        reactivity = row[self.reactivity_cols].to_numpy(dtype=np.float32)

        # True pour les mesures disponibles
        reactivity_mask = ~np.isnan(reactivity)

        # Remplacer les NaN par zéro pour le stockage
        reactivity = np.nan_to_num(reactivity, nan=0.0)

        return {"sequence": sequence, "experiment_type": experiment_type, "reactivity": torch.tensor(reactivity, dtype=torch.float32), "reactivity_mask": torch.tensor(reactivity_mask, dtype=torch.bool)}


# Regrouper les séquences en batch
def collate_fn(batch):
    sequences = []
    experiments = []
    reactivities = []
    reactivity_masks = []

    for sample in batch:
        sequences.append(sample["sequence"])
        experiments.append(sample["experiment_type"])
        reactivities.append(sample["reactivity"])
        reactivity_masks.append(sample["reactivity_mask"])

    # Tokenisation et padding
    padded, padding_mask = pad_sequences(sequences)

    # Types d'expérience
    experiment_type = torch.tensor(experiments, dtype=torch.long)

    # Réactivités et masques
    reactivity = torch.stack(reactivities)
    reactivity_mask = torch.stack(reactivity_masks)

    # Adapter les cibles à la longueur des séquences du batch
    seq_len = padded.shape[1]
    reactivity = reactivity[:, :seq_len]
    reactivity_mask = reactivity_mask[:, :seq_len]

    # Garantir la même longueur pour les prédictions et les cibles
    if reactivity.shape[1] < seq_len:
        missing = seq_len - reactivity.shape[1]
        reactivity = torch.nn.functional.pad(reactivity, (0, missing), value=0.0)
        reactivity_mask = torch.nn.functional.pad(reactivity_mask, (0, missing), value=False)

    # Ne jamais utiliser les positions PAD dans la loss
    reactivity_mask = reactivity_mask & ~padding_mask

    return {"padded": padded, "padding_mask": padding_mask, "experiment_type": experiment_type, "reactivity": reactivity, "reactivity_mask": reactivity_mask}


# Tests
if __name__ == "__main__":

    # Charger le dataset
    data_path = "/home/caop/RNA_ribonanza/small_data/train_data_QUICK_START.csv"
    df = pd.read_csv(data_path)

    # Identifier les colonnes de réactivité
    reactivity_cols = get_reactivity_columns(df)

    print("\nNombre de colonnes de réactivité :", len(reactivity_cols))
    print("Nombre de lignes :", len(df))

    # Séparer les séquences pour éviter les fuites de données
    sequence_ids = df["sequence_id"].unique()
    train_ids, val_ids = train_test_split(sequence_ids, test_size=0.2, random_state=42)

    train_df = df[df["sequence_id"].isin(train_ids)]
    val_df = df[df["sequence_id"].isin(val_ids)]

    print("Nombre de lignes train :", len(train_df))
    print("Nombre de lignes validation :", len(val_df))

    # Créer les datasets PyTorch
    train_dataset = RNADataset(train_df, reactivity_cols)
    val_dataset = RNADataset(val_df, reactivity_cols)

    # Créer les DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Tester un batch
    batch = next(iter(train_loader))

    print("\nDimensions des séquences :", batch["padded"].shape)
    print("Dimensions du padding mask :", batch["padding_mask"].shape)
    print("Dimensions des expériences :", batch["experiment_type"].shape)
    print("Dimensions des réactivités :", batch["reactivity"].shape)
    print("Dimensions du reactivity mask :", batch["reactivity_mask"].shape)

    print("\nExemple de séquence tokenisée :")
    print(batch["padded"][0])

    print("\nExemple de type d'expérience :")
    print(batch["experiment_type"][0])

    print("\nNombre de réactivités disponibles dans le batch :")
    print(batch["reactivity_mask"].sum().item())

    # Vérifications automatiques
    assert batch["padded"].shape == batch["reactivity"].shape
    assert batch["padding_mask"].shape == batch["reactivity_mask"].shape
    assert not torch.any(batch["reactivity_mask"] & batch["padding_mask"])
    assert not torch.isnan(batch["reactivity"]).any()
    assert len(set(train_ids) & set(val_ids)) == 0
        # Vérifier l'alignement des réactivités par petits blocs
    print("\nVérification de l'alignement des réactivités...")

    batch_size_check = 1000

    for start in range(0, len(df), batch_size_check):
        chunk = df.iloc[start:start + batch_size_check]

        lengths = chunk["sequence"].str.len().to_numpy()
        values = chunk[reactivity_cols].to_numpy(dtype=np.float32)

        positions = np.arange(len(reactivity_cols))
        outside_sequence = positions[None, :] >= lengths[:, None]

        invalid = outside_sequence & ~np.isnan(values)

        assert not np.any(invalid), f"Mesures hors séquence dans les lignes {start} à {start + len(chunk) - 1}"

    print("Aucune mesure trouvée au-delà des séquences")
 
    print("Tous les tests du Dataset sont réussis")