from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import RNADataset, collate_fn, get_reactivity_columns
from model import RNATransformer
from optparse import OptionParser


# Paramètres d'entraînement
data_path = "/home/caop/RNA_ribonanza/small_data/train_data_QUICK_START.csv"
batch_size = 32
epochs = 30
learning_rate = 0.001
max_train_batches = 10
max_val_batches = 5
patience = 5
min_delta = 0.0001


# Choisir GPU si disponible, sinon CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Appareil utilisé :", device)


# Calculer la MSE uniquement sur les valeurs disponibles
def masked_mse_loss(predictions, targets, mask):
    squared_errors = (predictions - targets) ** 2
    valid_errors = squared_errors[mask]

    if valid_errors.numel() == 0:
        return predictions.sum() * 0.0

    return valid_errors.mean()

def masked_mae_loss(predictions, targets, mask):
    targets_clipped = torch.clamp(targets, min=0.0, max=1.0)
    absolute_errors = torch.abs(predictions - targets_clipped)
    valid_errors = absolute_errors[mask]

    if valid_errors.numel() == 0:
        return predictions.sum() * 0.0

    return valid_errors.mean()

def mse_by_experiment(predictions, targets, mask, experiment_type):
    results = {}

    for exp_id, exp_name in [(0, "DMS_MaP"), (1, "2A3_MaP")]:
        exp_mask = mask & (experiment_type == exp_id).unsqueeze(1)

        if exp_mask.any().item():
            mse = masked_mse_loss(predictions, targets, exp_mask)
            results[exp_name] = mse.item()

    return results

# Entraîner le modèle pendant une epoch
def train_one_epoch(model, loader, optimizer, device, max_batches):
    model.train()

    total_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
        #if batch_idx >= max_batches:
            break

        padded = batch["padded"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        experiment_type = batch["experiment_type"].to(device)
        targets = batch["reactivity"].to(device)
        reactivity_mask = batch["reactivity_mask"].to(device)

        # Ignorer les batches sans mesure valide
        if not reactivity_mask.any().item():
            continue

        # Remettre les gradients à zéro
        optimizer.zero_grad()

        # Prédictions
        predictions = model(padded, padding_mask, experiment_type)

        # Calcul de la loss
        #loss = masked_mse_loss(predictions, targets, reactivity_mask)
        loss = masked_mae_loss(predictions, targets, reactivity_mask)

        # Backpropagation
        loss.backward()

        # Mise à jour des poids
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


# Évaluer le modèle sur la validation
def validate(model, loader, device, max_batches):
    model.eval()

    total_absolute_error = 0.0
    total_valid_positions = 0

    experiment_errors = {0: 0.0, 1: 0.0}
    experiment_counts = {0: 0, 1: 0}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
            #if batch_idx >= max_batches:
                break

            padded = batch["padded"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            experiment_type = batch["experiment_type"].to(device)
            targets = batch["reactivity"].to(device)
            reactivity_mask = batch["reactivity_mask"].to(device)

            predictions = model(padded, padding_mask, experiment_type)

            # Clipping des valeurs réelles selon le concours
            targets_clipped = torch.clamp(targets, min=0.0, max=1.0)

            # Erreurs absolues
            absolute_errors = torch.abs(predictions - targets_clipped)

            # MAE globale
            total_absolute_error += absolute_errors[reactivity_mask].sum().item()
            total_valid_positions += reactivity_mask.sum().item()

            # MAE par type d'expérience
            for exp_id in [0, 1]:
                exp_mask = reactivity_mask & (experiment_type == exp_id).unsqueeze(1)

                experiment_errors[exp_id] += absolute_errors[exp_mask].sum().item()
                experiment_counts[exp_id] += exp_mask.sum().item()

    val_mae = total_absolute_error / max(total_valid_positions, 1)
    dms_mae = experiment_errors[0] / max(experiment_counts[0], 1)
    twoa3_mae = experiment_errors[1] / max(experiment_counts[1], 1)

    print(f"Validation MAE DMS_MaP : {dms_mae:.4f}")
    print(f"Validation MAE 2A3_MaP : {twoa3_mae:.4f}")

    return val_mae

def save_validation_predictions(model, val_df, reactivity_cols, device, output_path):

    model.eval()

    # Matrice pour stocker toutes les prédictions
    predictions_array = np.full((len(val_df), len(reactivity_cols)), np.nan, dtype=np.float32)

    from embedding import pad_sequences

    with torch.no_grad():

        for i, (_, row) in enumerate(tqdm(
            val_df.iterrows(),
            total=len(val_df),
            desc="Prédictions validation")):

            sequence = row["sequence"]
            padded, padding_mask = pad_sequences([sequence])
            padded = padded.to(device)
            padding_mask = padding_mask.to(device)

            exp_id = 0 if row["experiment_type"] == "DMS_MaP" else 1

            experiment_type = torch.tensor([exp_id], device=device)

            predictions = model(padded, padding_mask, experiment_type)

            predictions = predictions[0].cpu().numpy()

            n = min(len(sequence), len(reactivity_cols))

            predictions_array[i, :n] = predictions[:n]

    # Construire les résultats en une seule fois
    metadata = val_df[["sequence_id", "sequence", "experiment_type"]].reset_index(drop=True)
    reactivity_df = pd.DataFrame(predictions_array, columns=reactivity_cols)
    result_df = pd.concat([metadata, reactivity_df], axis=1)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result_df.to_csv(output_path, index=False, na_rep="NaN")
    print(f"Prédictions sauvegardées dans : {output_path}")

# Programme principal
if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--output_dir", dest="output_dir", default="results", help="Dossier de sauvegarde des résultats")
    (options, args) = parser.parse_args()

    # Créer le dossier de sortie
    os.makedirs(options.output_dir, exist_ok=True)

    # Charger les données nécessaires
    df = pd.read_csv(data_path)

    # Identifier les colonnes de réactivité
    reactivity_cols = get_reactivity_columns(df)

    # Séparer les séquences entre train et validation
    sequence_ids = df["sequence_id"].unique()
    train_ids, val_ids = train_test_split(sequence_ids, test_size=0.2, random_state=42)

    train_df = df[df["sequence_id"].isin(train_ids)].copy()
    val_df = df[df["sequence_id"].isin(val_ids)].copy()

    # Créer les datasets
    train_dataset = RNADataset(train_df, reactivity_cols)
    val_dataset = RNADataset(val_df, reactivity_cols)

    # Créer les DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print("\nNombre d'exemples train :", len(train_dataset))
    print("Nombre d'exemples validation :", len(val_dataset))

    # Créer le Transformer
    model = RNATransformer(vocab_size=5, embedding_dim=16, num_heads=4, num_layers=2)
    model = model.to(device)

    # Optimiseur Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Entraînement
        # Entraînement avec Early Stopping
    best_val_mae = float("inf")
    epochs_without_improvement = 0

    best_model_path = os.path.join(options.output_dir, "rna_transformer_best.pth")

    for epoch in range(epochs):

        # Entraînement
        train_loss = train_one_epoch(model, train_loader, optimizer, device, max_train_batches)

        # Validation
        val_loss = validate(model, val_loader, device, max_val_batches)

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train MAE : {train_loss:.4f} - "
              f"Validation MAE : {val_loss:.4f}")

        # Vérifier si la Validation MAE s'améliore
        if val_loss < best_val_mae - min_delta:

            best_val_mae = val_loss
            epochs_without_improvement = 0

            # Sauvegarder le meilleur modèle
            torch.save(model.state_dict(), best_model_path)

            print(f"Meilleur modèle sauvegardé ! "f"Validation MAE : {best_val_mae:.4f}")

        else:
            epochs_without_improvement += 1
            print(f"Pas d'amélioration : "
                  f"{epochs_without_improvement}/{patience}")

        # Early Stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping à l'epoch {epoch + 1}. "
                  f"Meilleure Validation MAE : {best_val_mae:.4f}")
            break
    # Recharger le meilleur modèle
    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True))

    print(f"Meilleur modèle rechargé : MAE = {best_val_mae:.4f}")

    # Sauvegarder le modèle final (meilleurs poids)
    model_path = os.path.join(options.output_dir, "rna_transformer.pth")

    torch.save(model.state_dict(), model_path)

    print("Entraînement terminé !")
    print(f"Modèle sauvegardé dans : {model_path}")

    # Sauvegarder les prédictions de validation
    predictions_path = os.path.join(options.output_dir, "validation_predictions.csv")

    save_validation_predictions(
        model=model,
        val_df=val_df.head(20),
        reactivity_cols=reactivity_cols,
        device=device,
        output_path=predictions_path)