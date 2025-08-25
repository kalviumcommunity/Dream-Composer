"""
Integration tests for Dream Composer components.

Tests the interaction between tokenization and embedding layers,
demonstrating proper security practices and error handling.
"""

import torch
import pytest
from tokenization.tokenizer import SimpleTokenizer
from embedding.embedding import EmbeddingLayer


def test_tokenizer_embedding_integration():
    """Test integration between tokenizer and embedding layer."""
    # Create tokenizer
    tokenizer = SimpleTokenizer()

    # Create some sample text
    text = "hello world test"

    # Tokenize (returns list of strings)
    tokens = tokenizer.tokenize(text)

    # Create a simple vocabulary mapping
    vocab = {token: idx for idx, token in enumerate(set(tokens))}
    vocab_size = len(vocab) + 10  # Add buffer for unknown tokens
    embedding_dim = 16

    embedding_layer = EmbeddingLayer(vocab_size, embedding_dim)

    # Convert string tokens to integer indices
    token_indices = [vocab[token] for token in tokens]
    token_tensor = torch.tensor(token_indices, dtype=torch.long)

    # Get embeddings
    embeddings = embedding_layer(token_tensor)

    # Verify output shape
    expected_shape = (len(tokens), embedding_dim)
    assert embeddings.shape == expected_shape

    # Verify we can detokenize
    detokenized = tokenizer.detokenize(tokens)
    assert detokenized == text


def test_tokenizer_embedding_with_padding():
    """Test integration with padding tokens for batch processing."""
    tokenizer = SimpleTokenizer()

    # Create texts of different lengths
    texts = ["hello", "hello world", "hello world test"]

    # Tokenize all texts (returns lists of strings)
    all_tokens = [tokenizer.tokenize(text) for text in texts]

    # Create vocabulary from all unique tokens
    all_unique_tokens = set()
    for tokens in all_tokens:
        all_unique_tokens.update(tokens)

    # Create vocab mapping (reserve 0 for padding)
    vocab = {token: idx + 1 for idx, token in enumerate(all_unique_tokens)}
    vocab_size = len(vocab) + 10  # Add buffer
    max_length = max(len(tokens) for tokens in all_tokens)

    # Use 0 as padding token
    padding_idx = 0

    # Create embedding layer with padding
    embedding_layer = EmbeddingLayer(
        vocab_size=vocab_size,
        embedding_dim=16,
        padding_idx=padding_idx
    )

    # Convert string tokens to indices and pad sequences
    padded_indices = []
    for tokens in all_tokens:
        indices = [vocab[token] for token in tokens]
        padded = indices + [padding_idx] * (max_length - len(indices))
        padded_indices.append(padded)

    # Convert to tensor
    batch_tensor = torch.tensor(padded_indices, dtype=torch.long)

    # Get embeddings
    embeddings = embedding_layer(batch_tensor)

    # Verify shape
    assert embeddings.shape == (len(texts), max_length, 16)

    # Verify padding embeddings are zero
    for i, tokens in enumerate(all_tokens):
        if len(tokens) < max_length:
            # Check that padding positions have zero embeddings
            padding_positions = range(len(tokens), max_length)
            for pos in padding_positions:
                assert torch.allclose(
                    embeddings[i, pos],
                    torch.zeros(16)
                ), f"Padding at position {pos} should be zero"


def test_security_validation_integration():
    """Test that security validations work in integrated scenario."""
    tokenizer = SimpleTokenizer()
    embedding_layer = EmbeddingLayer(vocab_size=100, embedding_dim=16)
    
    # Test with invalid token indices
    with pytest.raises(ValueError, match="Token indices must be less than vocab_size"):
        invalid_tokens = torch.tensor([1, 2, 150], dtype=torch.long)  # 150 > vocab_size
        embedding_layer(invalid_tokens)
    
    # Test with wrong dtype
    with pytest.raises(ValueError, match="Expected Long or Int tensor"):
        float_tokens = torch.tensor([1.0, 2.0, 3.0])
        embedding_layer(float_tokens)
    
    # Test with negative indices
    with pytest.raises(ValueError, match="Token indices must be non-negative"):
        negative_tokens = torch.tensor([-1, 1, 2], dtype=torch.long)
        embedding_layer(negative_tokens)


def test_device_consistency():
    """Test device handling in integrated scenario."""
    tokenizer = SimpleTokenizer()

    # Create embedding on CPU
    embedding_layer = EmbeddingLayer(
        vocab_size=100,
        embedding_dim=16,
        device='cpu'
    )

    # Tokenize and convert to indices
    tokens = tokenizer.tokenize("hello world")
    # Create simple vocab mapping
    vocab = {token: idx for idx, token in enumerate(set(tokens))}
    token_indices = [vocab[token] for token in tokens]
    token_tensor = torch.tensor(token_indices, dtype=torch.long, device='cpu')

    # Should work fine
    embeddings = embedding_layer(token_tensor)
    assert embeddings.device.type == 'cpu'

    # Test device mismatch (if CUDA available)
    if torch.cuda.is_available():
        cuda_embedding = EmbeddingLayer(
            vocab_size=100,
            embedding_dim=16,
            device='cuda'
        )

        with pytest.raises(RuntimeError, match="Input tensor is on device"):
            cuda_embedding(token_tensor)  # CPU tensor to CUDA model


def test_empty_and_edge_cases():
    """Test edge cases in integration."""
    tokenizer = SimpleTokenizer()
    embedding_layer = EmbeddingLayer(vocab_size=100, embedding_dim=16)

    # Test empty string
    empty_tokens = tokenizer.tokenize("")
    if empty_tokens:  # If tokenizer returns tokens for empty string
        vocab = {token: idx for idx, token in enumerate(set(empty_tokens))}
        token_indices = [vocab[token] for token in empty_tokens]
        token_tensor = torch.tensor(token_indices, dtype=torch.long)
        embeddings = embedding_layer(token_tensor)
        assert embeddings.shape[0] == len(empty_tokens)

    # Test single character
    single_tokens = tokenizer.tokenize("a")
    if single_tokens:
        vocab = {token: idx for idx, token in enumerate(set(single_tokens))}
        token_indices = [vocab[token] for token in single_tokens]
        token_tensor = torch.tensor(token_indices, dtype=torch.long)
        embeddings = embedding_layer(token_tensor)
        assert embeddings.shape == (len(single_tokens), 16)


def test_batch_processing_security():
    """Test security validations work correctly in batch scenarios."""
    embedding_layer = EmbeddingLayer(vocab_size=50, embedding_dim=8)
    
    # Create a batch with mixed valid and invalid indices
    # This should fail because one sequence has invalid indices
    batch_with_invalid = torch.tensor([
        [1, 2, 3],      # Valid
        [4, 5, 60],     # Invalid: 60 >= vocab_size (50)
        [7, 8, 9]       # Valid
    ], dtype=torch.long)
    
    with pytest.raises(ValueError, match="Token indices must be less than vocab_size"):
        embedding_layer(batch_with_invalid)
    
    # Test valid batch
    valid_batch = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=torch.long)
    
    embeddings = embedding_layer(valid_batch)
    assert embeddings.shape == (3, 3, 8)  # batch_size, seq_len, embedding_dim
