
import torch
import torch.nn as nn

from embedding import RNAEmbedding, pad_sequences


# Modèle Transformer pour prédire la réactivité ARN
class RNATransformer(nn.Module):

    def __init__(self, vocab_size=5, embedding_dim=16, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()

        # Embedding des nucléotides et encodage positionnel
        self.embedding = RNAEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim)

        # Une couche Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=embedding_dim * 4, dropout=dropout, batch_first=True)

        # Empiler plusieurs couches Transformer
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Embedding du type d'expérience : DMS ou 2A3
        self.experiment_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)

        # Couche finale : prédire une réactivité par nucléotide
        self.output_layer = nn.Linear(embedding_dim, 1)

    def forward(self, padded, padding_mask, experiment_type):

        # Transformer les tokens en embeddings
        x = self.embedding(padded)

        # Ajouter l'information sur le type d'expérience
        experiment_emb = self.experiment_embedding(experiment_type)
        x = x + experiment_emb.unsqueeze(1)

        # Appliquer le Transformer en ignorant les positions PAD
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Prédire une valeur par position
        predictions = self.output_layer(x).squeeze(-1)

        # Mettre les prédictions des positions PAD à zéro
        predictions = predictions.masked_fill(padding_mask, 0.0)

        return predictions


# Tests
if __name__ == "__main__":

    # Séquences ARN
    sequences = ["AUGCA", "ACU", "GGCA"]

    # Tokenisation et padding
    padded, padding_mask = pad_sequences(sequences)

    # Types d'expérience : 0 = DMS_MaP, 1 = 2A3_MaP
    experiment_type = torch.tensor([0, 1, 0], dtype=torch.long)

    # Création du modèle
    model = RNATransformer(vocab_size=5, embedding_dim=16, num_heads=4, num_layers=2)

    # Passage des séquences dans le modèle
    model.eval()

    with torch.no_grad():
        predictions = model(padded, padding_mask, experiment_type)

    print("\nSéquences tokenisées :")
    print(padded)

    print("\nTypes d'expérience :")
    print(experiment_type)

    print("\nDimensions des prédictions :")
    print(predictions.shape)

    print("\nPrédictions :")
    print(predictions)

    # Vérifications automatiques
    assert predictions.shape == padded.shape
    assert torch.all(predictions[padding_mask] == 0)

    print("\nTous les tests du Transformer sont réussis !")
