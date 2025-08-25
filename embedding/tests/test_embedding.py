import torch
import pytest
from embedding.embedding import EmbeddingLayer


def test_embedding_shape():
    """Test basic embedding functionality and output shape."""
    vocab_size = 100
    embedding_dim = 16
    model = EmbeddingLayer(vocab_size, embedding_dim)

    tokens = torch.tensor([1, 5, 20])
    out = model(tokens)

    assert out.shape == (3, embedding_dim)


def test_embedding_with_padding():
    """Test embedding layer with padding token."""
    vocab_size = 100
    embedding_dim = 16
    padding_idx = 0

    model = EmbeddingLayer(vocab_size, embedding_dim, padding_idx=padding_idx)

    # Test that padding embeddings are zeros
    padding_tokens = torch.tensor([0, 0, 0])
    out = model(padding_tokens)

    # Padding embeddings should be zero
    assert torch.allclose(out, torch.zeros_like(out))

    # Test mixed tokens with padding
    mixed_tokens = torch.tensor([0, 1, 2, 0])
    out = model(mixed_tokens)

    assert out.shape == (4, embedding_dim)
    # First and last should be zero (padding)
    assert torch.allclose(out[0], torch.zeros(embedding_dim))
    assert torch.allclose(out[3], torch.zeros(embedding_dim))


def test_input_dtype_validation():
    """Test input dtype validation - should accept integer tensors."""
    vocab_size = 100
    embedding_dim = 16
    model = EmbeddingLayer(vocab_size, embedding_dim)

    # Test valid integer types (PyTorch embedding only accepts int32 and int64)
    valid_tokens = [
        torch.tensor([1, 2, 3], dtype=torch.long),  # int64
        torch.tensor([1, 2, 3], dtype=torch.int),   # int32
    ]

    for tokens in valid_tokens:
        out = model(tokens)
        assert out.shape == (3, embedding_dim)

    # Test invalid dtypes
    with pytest.raises(ValueError, match="Expected Long or Int tensor"):
        float_tokens = torch.tensor([1.0, 2.0, 3.0])
        model(float_tokens)

    with pytest.raises(ValueError, match="Expected Long or Int tensor"):
        short_tokens = torch.tensor([1, 2, 3], dtype=torch.short)
        model(short_tokens)


def test_device_handling():
    """Test device placement and validation."""
    vocab_size = 100
    embedding_dim = 16

    # Test CPU device
    model_cpu = EmbeddingLayer(vocab_size, embedding_dim, device='cpu')
    tokens_cpu = torch.tensor([1, 2, 3], device='cpu')
    out = model_cpu(tokens_cpu)
    assert out.device.type == 'cpu'

    # Test device mismatch error
    if torch.cuda.is_available():
        model_cuda = EmbeddingLayer(vocab_size, embedding_dim, device='cuda')
        tokens_cpu = torch.tensor([1, 2, 3], device='cpu')

        with pytest.raises(RuntimeError, match="Input tensor is on device"):
            model_cuda(tokens_cpu)


def test_token_index_validation():
    """Test validation of token indices."""
    vocab_size = 100
    embedding_dim = 16
    model = EmbeddingLayer(vocab_size, embedding_dim)

    # Test valid indices
    valid_tokens = torch.tensor([0, 50, 99])  # Valid range [0, 99]
    out = model(valid_tokens)
    assert out.shape == (3, embedding_dim)

    # Test negative indices
    with pytest.raises(ValueError, match="Token indices must be non-negative"):
        negative_tokens = torch.tensor([-1, 1, 2])
        model(negative_tokens)

    # Test indices >= vocab_size
    with pytest.raises(ValueError, match="Token indices must be less than vocab_size"):
        large_tokens = torch.tensor([1, 2, 100])  # 100 >= vocab_size
        model(large_tokens)


def test_input_type_validation():
    """Test that non-tensor inputs are rejected."""
    vocab_size = 100
    embedding_dim = 16
    model = EmbeddingLayer(vocab_size, embedding_dim)

    # Test list input
    with pytest.raises(TypeError, match="Expected torch.Tensor"):
        model([1, 2, 3])

    # Test numpy array input (if numpy is available)
    try:
        import numpy as np
        with pytest.raises(TypeError, match="Expected torch.Tensor"):
            model(np.array([1, 2, 3]))
    except ImportError:
        # Skip numpy test if not available
        pass


def test_constructor_validation():
    """Test constructor parameter validation."""
    # Test invalid vocab_size
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        EmbeddingLayer(0, 16)

    with pytest.raises(ValueError, match="vocab_size must be positive"):
        EmbeddingLayer(-10, 16)

    # Test invalid embedding_dim
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        EmbeddingLayer(100, 0)

    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        EmbeddingLayer(100, -5)

    # Test invalid padding_idx
    with pytest.raises(ValueError, match="padding_idx must be in range"):
        EmbeddingLayer(100, 16, padding_idx=100)

    with pytest.raises(ValueError, match="padding_idx must be in range"):
        EmbeddingLayer(100, 16, padding_idx=-1)


def test_utility_methods():
    """Test utility methods for getting layer properties."""
    vocab_size = 100
    embedding_dim = 16
    padding_idx = 0

    model = EmbeddingLayer(vocab_size, embedding_dim, padding_idx=padding_idx)

    assert model.get_vocab_size() == vocab_size
    assert model.get_embedding_dim() == embedding_dim
    assert model.get_padding_idx() == padding_idx

    # Test without padding
    model_no_pad = EmbeddingLayer(vocab_size, embedding_dim)
    assert model_no_pad.get_padding_idx() is None


def test_empty_tensor():
    """Test handling of empty tensors."""
    vocab_size = 100
    embedding_dim = 16
    model = EmbeddingLayer(vocab_size, embedding_dim)

    # Empty tensor should work
    empty_tokens = torch.tensor([], dtype=torch.long)
    out = model(empty_tokens)
    assert out.shape == (0, embedding_dim)


def test_multidimensional_input():
    """Test with multidimensional input tensors."""
    vocab_size = 100
    embedding_dim = 16
    model = EmbeddingLayer(vocab_size, embedding_dim)

    # 2D input (batch_size, seq_len)
    tokens_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = model(tokens_2d)
    assert out.shape == (2, 3, embedding_dim)

    # 3D input
    tokens_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    out = model(tokens_3d)
    assert out.shape == (2, 2, 2, embedding_dim)
