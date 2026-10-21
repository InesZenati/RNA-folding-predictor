
import torch
import torch.nn as nn

from embedding import RNAEmbedding, pad_sequences


# Modèle Transformer pour prédire la réactivité ARN
class RNATransformer(nn.Module):

    def __init__(self, vocab_size=5, embedding_dim=16, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()

        # Embedding des nucléotides, encodage positionnel et type d'expérience
        self.embedding = RNAEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim)

        # Une couche Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=embedding_dim * 4, dropout=dropout, batch_first=True)

        # Empiler plusieurs couches Transformer
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Couche finale : prédire une réactivité par nucléotide
        self.output_layer = nn.Linear(embedding_dim, 1)

    def forward(self, padded, padding_mask, experiment_type):
        
        #  Embedding : nucléotides + positions + expérience
        x = self.embedding(padded, experiment_type)

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
    assert torch.isfinite(predictions).all()
    assert len(model.transformer.layers) == 2

    print("Architecture du Transformer :")

    for i, layer in enumerate(model.transformer.layers):
        print(f"Transformer Encoder Layer {i + 1}")

        # Multi-Head Attention
        print("Multi-Head Attention :", layer.self_attn)
        print("Nombre de têtes :", layer.self_attn.num_heads)

        # Feed-Forward Network
        print("FFN couche 1 :", layer.linear1)
        print("FFN couche 2 :", layer.linear2)

        # LayerNorm
        print("LayerNorm 1 :", layer.norm1)
        print("LayerNorm 2 :", layer.norm2)

    print("Tests passés")
