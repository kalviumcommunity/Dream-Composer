"""
Structured Prompt module for Dream Composer.

This module provides structured prompting capabilities for analyzing dream descriptions
and extracting emotions, moods, and musical parameters using AI/NLP APIs.
"""

from .prompt_builder import PromptBuilder, PromptType
from .dream_analyzer import DreamAnalyzer
from .music_mapper import MusicMapper
from .emotion_extractor import EmotionExtractor
from .zero_shot_prompts import ZeroShotPromptBuilder, ZeroShotTask
from .zero_shot_analyzer import ZeroShotDreamAnalyzer, ZeroShotAnalysisResult, ComprehensiveZeroShotAnalysis
from .dynamic_shot_prompts import DynamicShotPromptBuilder, DreamComplexity, ExampleType, DreamExample, DynamicShotConfig
from .dynamic_shot_analyzer import DynamicShotDreamAnalyzer, DynamicShotAnalysisResult, ComprehensiveDynamicShotAnalysis
from .one_shot_prompts import OneShotPromptBuilder, OneShotStrategy, OneShotConfig
from .one_shot_analyzer import OneShotDreamAnalyzer, OneShotAnalysisResult, ComprehensiveOneShotAnalysis
from .multi_shot_prompts import MultiShotPromptBuilder, MultiShotStrategy, MultiShotConfig
from .multi_shot_analyzer import MultiShotDreamAnalyzer, MultiShotAnalysisResult, ComprehensiveMultiShotAnalysis

__all__ = [
    'PromptBuilder',
    'PromptType',
    'DreamAnalyzer',
    'MusicMapper',
    'EmotionExtractor',
    'ZeroShotPromptBuilder',
    'ZeroShotTask',
    'ZeroShotDreamAnalyzer',
    'ZeroShotAnalysisResult',
    'ComprehensiveZeroShotAnalysis',
    'DynamicShotPromptBuilder',
    'DreamComplexity',
    'ExampleType',
    'DreamExample',
    'DynamicShotConfig',
    'DynamicShotDreamAnalyzer',
    'DynamicShotAnalysisResult',
    'ComprehensiveDynamicShotAnalysis',
    'OneShotPromptBuilder',
    'OneShotStrategy',
    'OneShotConfig',
    'OneShotDreamAnalyzer',
    'OneShotAnalysisResult',
    'ComprehensiveOneShotAnalysis',
    'MultiShotPromptBuilder',
    'MultiShotStrategy',
    'MultiShotConfig',
    'MultiShotDreamAnalyzer',
    'MultiShotAnalysisResult',
    'ComprehensiveMultiShotAnalysis'
]
