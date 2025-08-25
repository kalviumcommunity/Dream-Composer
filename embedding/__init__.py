"""
Embedding module for Dream Composer.

This module provides embedding layers with proper input validation,
device handling, and support for special tokens like padding and unknown tokens.
"""

from .embedding import EmbeddingLayer

__all__ = ['EmbeddingLayer']
