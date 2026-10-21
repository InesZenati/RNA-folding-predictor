
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from optparse import OptionParser


# Vocabulaire ARN
vocab = {"PAD": 0, "A": 1, "C": 2, "G": 3, "U": 4}

def tokenize_sequence(sequence):
    """Tokenisation des séquences."""
    tokens = []
    for nucleotide in sequence:
        tokens.append(vocab[nucleotide])
    return tokens

def check_nucleotides(df):
    """Vérification des nucléotides présents dans le dataset."""
    sequences = df["sequence"].drop_duplicates()
    nucleotides = set()

    for sequence in sequences:
        for nucleotide in sequence:
            nucleotides.add(nucleotide)

    print("Nucléotides présents :", sorted(nucleotides))

    unknown = nucleotides - {"A", "C", "G", "U"}
    print("Nucléotides inconnus :", unknown)

def pad_sequences(sequences):
    """Padding des séquences et création du masque."""
    tokenized = []

    for sequence in sequences:
        tokens = tokenize_sequence(sequence)
        tensor = torch.tensor(tokens, dtype=torch.long)
        tokenized.append(tensor)

    padded = pad_sequence(tokenized, batch_first=True, padding_value=vocab["PAD"])
    padding_mask = (padded == vocab["PAD"])

    return padded, padding_mask


def positional_encoding(seq_len, embedding_dim=16):
    """Créer un encodage positionnel à l'aide de sinus et cosinus.
       Référence : Article Attention is all you need.
    """
    # Matrice de O
    pe = torch.zeros(seq_len, embedding_dim)

    for pos in range(seq_len):
        # calculer un vecteur différent pour chaque position
        for i in range(0, embedding_dim, 2):
            angle = pos / (10000 ** (i / embedding_dim))
            pe[pos, i] = math.sin(angle)

            if i + 1 < embedding_dim:
                pe[pos, i + 1] = math.cos(angle)

    return pe

class RNAEmbedding(nn.Module):
    # Classe d'embedding ARN, héritée de nn.Module
    # nn.Module nn.Module : classe de base utilisée par PyTorch pour construire des réseaux de neurones

    def __init__(self, vocab_size=5, embedding_dim=16, num_experiments=2):
        super().__init__()

        self.embedding_dim = embedding_dim # 16
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=vocab["PAD"])
        # Chaque token est représenté par un vecteur de 16 valeurs

        # Embedding du type d'expérience (DMS ou 2A3)
        self.experiment_embedding = nn.Embedding(num_embeddings=num_experiments, embedding_dim=embedding_dim)

    def forward(self, padded, experiment_type):
        # Embedding des nucléotides
        embeddings = self.embedding(padded)

        # Calcul de l'encodage positionnel
        seq_len = padded.shape[1]
        pe = positional_encoding(seq_len, self.embedding_dim)
        pe = pe.to(device=embeddings.device, dtype=embeddings.dtype)
        experiment_emb = self.experiment_embedding(experiment_type)

        # Addition des embeddings, des positions embeddings et embeddings experiences
        #embeddings = embeddings + pe.unsqueeze(0)
        embeddings = (embeddings + pe.unsqueeze(0) + experiment_emb.unsqueeze(1))

        # Remettre les positions PAD à zéro
        padding_mask = (padded == vocab["PAD"])
        embeddings = embeddings.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return embeddings


# Tests
if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("-i", "--input", dest="input_file",
                      default="/home/caop/RNA_ribonanza/small_data/train_data_QUICK_START.csv",
                      help="Chemin du fichier CSV contenant les séquences ARN")

    options, args = parser.parse_args()

    # Test de tokenisation
    sequence = "AUGCA"
    tokens = tokenize_sequence(sequence)

    print("Séquence :", sequence)
    print("Tokens :", tokens)

    data_path = options.input_file

    print("Fichier d'entrée :", data_path)

    df = pd.read_csv(data_path, usecols=["sequence"])
    check_nucleotides(df)

    # Test du padding
    sequences = ["AUGCA", "ACU", "GGCA"]
    padded, padding_mask = pad_sequences(sequences)

    print("Séquences après padding :")
    print(padded)

    print("Masque de padding :")
    print(padding_mask)

    print("Dimensions :", padded.shape)

    # Test de l'encodage positionnel
    pe = positional_encoding(seq_len=padded.shape[1], embedding_dim=16)

    print("Encodage positionnel :")
    print(pe)

    # Test de la classe RNAEmbedding
    #model = RNAEmbedding(vocab_size=len(vocab), embedding_dim=16)
    #embeddings = model(padded)

    model = RNAEmbedding(vocab_size=len(vocab), embedding_dim=16)
    # 0 = DMS_MaP, 1 = 2A3_MaP
    experiment_type = torch.tensor([0, 1, 0], dtype=torch.long)
    embeddings = model(padded, experiment_type)

    print("Dimensions des embeddings :")
    print(embeddings.shape)

    print("Embedding du premier A :")
    print(embeddings[0, 0])

    print("Embedding du dernier A :")
    print(embeddings[0, 4])

    print("Embedding du premier PAD :")
    print(embeddings[1, 3])

    # Vérification des paramètres entraînables
    print("Paramètres entraînables :")

    for name, param in model.named_parameters():
        print(name, param.shape, param.requires_grad)

    # Tests 
    assert padded.shape == (3, 5)
    assert embeddings.shape == (3, 5, 16)
    assert torch.all(embeddings[padding_mask] == 0)
    assert not torch.allclose(embeddings[0, 0], embeddings[0, 4])

    # Test de l'embedding du type d'expérience

    # Même séquence ARN, deux expériences différentes
    dms = torch.tensor([0], dtype=torch.long)
    a3 = torch.tensor([1], dtype=torch.long)

    emb_dms = model(padded[:1], dms)
    emb_a3 = model(padded[:1], a3)

    print("Embedding DMS du premier A :")
    print(emb_dms[0, 0])

    print("Embedding 2A3 du premier A :")
    print(emb_a3[0, 0])

    # Les représentations doivent être différentes
    assert not torch.allclose(emb_dms, emb_a3)

    print("Tests passés")
