
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


# Vocabulaire ARN
vocab = {"PAD": 0, "A": 1, "C": 2, "G": 3, "U": 4}


# Tokenisation des séquences
def tokenize_sequence(sequence):
    tokens = []
    for nucleotide in sequence:
        tokens.append(vocab[nucleotide])
    return tokens


# Vérification des nucléotides présents dans le dataset
def check_nucleotides(df):
    sequences = df["sequence"].drop_duplicates()
    nucleotides = set()

    for sequence in sequences:
        for nucleotide in sequence:
            nucleotides.add(nucleotide)

    print("Nucléotides présents :", sorted(nucleotides))

    unknown = nucleotides - {"A", "C", "G", "U"}
    print("Nucléotides inconnus :", unknown)


# Padding des séquences et création du masque
def pad_sequences(sequences):
    tokenized = []

    for sequence in sequences:
        tokens = tokenize_sequence(sequence)
        tensor = torch.tensor(tokens, dtype=torch.long)
        tokenized.append(tensor)

    padded = pad_sequence(tokenized, batch_first=True, padding_value=vocab["PAD"])
    padding_mask = (padded == vocab["PAD"])

    return padded, padding_mask


# Encodage positionnel sinusoïdal
def positional_encoding(seq_len, embedding_dim=16):
    pe = torch.zeros(seq_len, embedding_dim)

    for pos in range(seq_len):
        for i in range(0, embedding_dim, 2):
            angle = pos / (10000 ** (i / embedding_dim))
            pe[pos, i] = math.sin(angle)

            if i + 1 < embedding_dim:
                pe[pos, i + 1] = math.cos(angle)

    return pe


# Classe d'embedding ARN
class RNAEmbedding(nn.Module):

    def __init__(self, vocab_size=5, embedding_dim=16):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=vocab["PAD"])

    def forward(self, padded):
        # Embedding des nucléotides
        embeddings = self.embedding(padded)

        # Calcul de l'encodage positionnel
        seq_len = padded.shape[1]
        pe = positional_encoding(seq_len, self.embedding_dim)
        pe = pe.to(device=embeddings.device, dtype=embeddings.dtype)

        # Addition des embeddings et des positions
        embeddings = embeddings + pe.unsqueeze(0)

        # Remettre les positions PAD à zéro
        padding_mask = (padded == vocab["PAD"])
        embeddings = embeddings.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return embeddings


# Tests
if __name__ == "__main__":

    # Test de tokenisation
    sequence = "AUGCA"
    tokens = tokenize_sequence(sequence)

    print("\nSéquence :", sequence)
    print("Tokens :", tokens)

    # Vérification des nucléotides dans le dataset
    data_path = "/home/caop/RNA_ribonanza/small_data/train_data_QUICK_START.csv"
    df = pd.read_csv(data_path, usecols=["sequence"])
    check_nucleotides(df)

    # Test du padding
    sequences = ["AUGCA", "ACU", "GGCA"]
    padded, padding_mask = pad_sequences(sequences)

    print("\nSéquences après padding :")
    print(padded)

    print("\nMasque de padding :")
    print(padding_mask)

    print("\nDimensions :", padded.shape)

    # Test de l'encodage positionnel
    pe = positional_encoding(seq_len=padded.shape[1], embedding_dim=16)

    print("\nEncodage positionnel :")
    print(pe)

    # Test de la classe RNAEmbedding
    model = RNAEmbedding(vocab_size=len(vocab), embedding_dim=16)
    embeddings = model(padded)

    print("\nDimensions des embeddings :")
    print(embeddings.shape)

    print("\nEmbedding du premier A :")
    print(embeddings[0, 0])

    print("\nEmbedding du dernier A :")
    print(embeddings[0, 4])

    print("\nEmbedding du premier PAD :")
    print(embeddings[1, 3])

    # Vérification des paramètres entraînables
    print("\nParamètres entraînables :")

    for name, param in model.named_parameters():
        print(name, param.shape, param.requires_grad)

    # Tests automatiques
    assert padded.shape == (3, 5)
    assert embeddings.shape == (3, 5, 16)
    assert torch.all(embeddings[padding_mask] == 0)
    assert not torch.allclose(embeddings[0, 0], embeddings[0, 4])

    print("\nTous les tests sont réussis !")
