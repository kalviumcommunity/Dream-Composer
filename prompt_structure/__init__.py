"""
Structured Prompt module for Dream Composer.

This module provides structured prompting capabilities for analyzing dream descriptions
and extracting emotions, moods, and musical parameters using AI/NLP APIs.
"""

from .prompt_builder import PromptBuilder, PromptType
from .dream_analyzer import DreamAnalyzer
from .music_mapper import MusicMapper
from .emotion_extractor import EmotionExtractor

__all__ = [
    'PromptBuilder',
    'PromptType',
    'DreamAnalyzer',
    'MusicMapper',
    'EmotionExtractor'
]
