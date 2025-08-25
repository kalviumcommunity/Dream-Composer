import torch
from embedding.embedding import EmbeddingLayer

def test_embedding_shape():
    vocab_size = 100
    embedding_dim = 16
    model = EmbeddingLayer(vocab_size, embedding_dim)

    tokens = torch.tensor([1, 5, 20])
    out = model(tokens)

    assert out.shape == (3, embedding_dim)
