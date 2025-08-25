import torch
import torch.nn as nn
from typing import Optional, Union


class EmbeddingLayer(nn.Module):
    """
    Enhanced embedding layer with proper input validation, device handling,
    and support for special tokens.

    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of the embedding vectors
        padding_idx (Optional[int]): Index for padding token. If specified,
            the embedding at this index will be initialized to zeros and
            will not be updated during training.
        device (Optional[Union[str, torch.device]]): Device to place the model on
        dtype (Optional[torch.dtype]): Data type for the embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        # Validate inputs
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if padding_idx is not None and (padding_idx < 0 or padding_idx >= vocab_size):
            raise ValueError(f"padding_idx must be in range [0, {vocab_size}), got {padding_idx}")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Create embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
            device=device,
            dtype=dtype
        )

        # Store device and dtype for validation
        self.expected_device = device
        self.expected_dtype = dtype or torch.long

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the embedding layer.

        Args:
            token_ids (torch.Tensor): Input tensor containing token indices.
                Expected to be LongTensor with values in range [0, vocab_size).

        Returns:
            torch.Tensor: Embedded representations with shape (*input_shape, embedding_dim)

        Raises:
            TypeError: If input is not a torch.Tensor
            ValueError: If input contains invalid token indices
            RuntimeError: If there are device mismatches
        """
        # Input validation
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(token_ids)}")

        # Check dtype - PyTorch embedding only accepts Long and Int tensors
        valid_dtypes = {torch.int32, torch.int64}  # int and long
        if token_ids.dtype not in valid_dtypes:
            raise ValueError(
                f"Expected Long or Int tensor for token indices, got {token_ids.dtype}. "
                f"PyTorch embedding requires int32 or int64 tensors. "
                f"Consider using token_ids.long() to convert to LongTensor."
            )

        # Check for valid token indices
        if token_ids.numel() > 0:  # Only check if tensor is not empty
            min_val = token_ids.min().item()
            max_val = token_ids.max().item()

            if min_val < 0:
                raise ValueError(f"Token indices must be non-negative, got minimum value {min_val}")
            if max_val >= self.vocab_size:
                raise ValueError(
                    f"Token indices must be less than vocab_size ({self.vocab_size}), "
                    f"got maximum value {max_val}"
                )

        # Check device compatibility
        if self.expected_device is not None:
            expected_device = torch.device(self.expected_device)
            if token_ids.device != expected_device:
                raise RuntimeError(
                    f"Input tensor is on device {token_ids.device}, "
                    f"but model expects device {expected_device}. "
                    f"Use token_ids.to('{expected_device}') to move the tensor."
                )

        return self.embedding(token_ids)

    def get_padding_idx(self) -> Optional[int]:
        """Return the padding index if set."""
        return self.padding_idx

    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim
