"""
One-Shot Prompting for Dream Composer.

This module implements one-shot prompting techniques that provide exactly one
carefully selected example to guide AI analysis. One-shot prompting strikes
a balance between zero-shot (no examples) and few-shot (multiple examples)
approaches, offering focused guidance without overwhelming the model.
"""

import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .zero_shot_prompts import ZeroShotTask
from .dynamic_shot_prompts import DreamComplexity, ExampleType, DreamExample

# Configure logging
logger = logging.getLogger(__name__)


class OneShotStrategy(Enum):
    """Strategies for selecting the single example in one-shot prompting."""
    BEST_MATCH = "best_match"           # Select the most relevant example
    REPRESENTATIVE = "representative"    # Select the most typical example
    COMPLEXITY_MATCHED = "complexity_matched"  # Select based on complexity match
    BALANCED = "balanced"               # Balance relevance and representativeness
    RANDOM_QUALITY = "random_quality"   # Random selection from high-quality examples


@dataclass
class OneShotConfig:
    """Configuration for one-shot prompting."""
    strategy: OneShotStrategy = OneShotStrategy.BEST_MATCH
    quality_threshold: float = 0.8      # Minimum quality score for examples
    relevance_weight: float = 0.6       # Weight for relevance in balanced strategy
    representativeness_weight: float = 0.4  # Weight for representativeness
    use_complexity_boost: bool = True   # Whether to boost complexity-matched examples
    fallback_to_representative: bool = True  # Fallback if no relevant examples found


class OneShotPromptBuilder:
    """
    Builder for one-shot prompts that provide exactly one example.
    
    One-shot prompting provides focused guidance through a single, carefully
    selected example that best represents the desired analysis approach.
    """
    
    def __init__(self, config: Optional[OneShotConfig] = None):
        self.config = config or OneShotConfig()
        self.example_database = self._initialize_example_database()
        self.selection_history = {}  # Track selection patterns
    
    def _initialize_example_database(self) -> Dict[ZeroShotTask, List[DreamExample]]:
        """Initialize database of high-quality examples for one-shot prompting."""
        return {
            ZeroShotTask.DREAM_EMOTION_ANALYSIS: [
                DreamExample(
                    dream_text="I was flying over a beautiful city at sunset, feeling incredibly free and joyful.",
                    analysis={
                        "primary_emotions": ["joy", "freedom"],
                        "emotion_intensities": {"joy": 9, "freedom": 8},
                        "emotional_progression": "consistent positive emotions throughout",
                        "dominant_mood": "euphoric",
                        "emotional_triggers": ["flying sensation", "beautiful scenery"],
                        "confidence": 0.92
                    },
                    example_type=ExampleType.BASIC_EMOTION,
                    complexity=DreamComplexity.SIMPLE,
                    keywords=["flying", "city", "sunset", "free", "joyful", "beautiful"]
                ),
                DreamExample(
                    dream_text="I started happy in a garden, but then storm clouds gathered and I felt anxious and sad.",
                    analysis={
                        "primary_emotions": ["joy", "anxiety", "sadness"],
                        "emotion_intensities": {"joy": 7, "anxiety": 8, "sadness": 6},
                        "emotional_progression": "positive to negative emotional shift",
                        "dominant_mood": "transitional",
                        "emotional_triggers": ["weather change", "environmental shift"],
                        "confidence": 0.87
                    },
                    example_type=ExampleType.MIXED_EMOTIONS,
                    complexity=DreamComplexity.MODERATE,
                    keywords=["garden", "storm", "clouds", "happy", "anxious", "sad", "transition"]
                ),
                DreamExample(
                    dream_text="I was swimming deep underwater with glowing fish, feeling peaceful yet mysterious.",
                    analysis={
                        "primary_emotions": ["peace", "wonder", "mystery"],
                        "emotion_intensities": {"peace": 8, "wonder": 7, "mystery": 6},
                        "emotional_progression": "consistent contemplative emotions",
                        "dominant_mood": "mystical",
                        "emotional_triggers": ["underwater environment", "bioluminescence"],
                        "confidence": 0.89
                    },
                    example_type=ExampleType.SYMBOLIC_CONTENT,
                    complexity=DreamComplexity.MODERATE,
                    keywords=["swimming", "underwater", "fish", "glowing", "peaceful", "mysterious"]
                )
            ],
            
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION: [
                DreamExample(
                    dream_text="I was dancing in a moonlit ballroom with elegant music playing.",
                    analysis={
                        "recommended_style": "classical_waltz",
                        "tempo": {"bpm": 120, "description": "graceful and flowing"},
                        "key_signature": {"key": "F", "mode": "major", "reasoning": "elegant and romantic"},
                        "instruments": ["piano", "strings", "harp"],
                        "dynamics": "mp to mf with gentle swells",
                        "musical_structure": "ternary form with recurring themes",
                        "special_techniques": ["legato phrasing", "rubato"],
                        "confidence": 0.91
                    },
                    example_type=ExampleType.MUSICAL_SPECIFIC,
                    complexity=DreamComplexity.SIMPLE,
                    keywords=["dancing", "ballroom", "music", "elegant", "moonlit"]
                ),
                DreamExample(
                    dream_text="I was in a thunderstorm, feeling both terrified and exhilarated by the power.",
                    analysis={
                        "recommended_style": "dramatic_orchestral",
                        "tempo": {"bpm": 140, "description": "intense and building"},
                        "key_signature": {"key": "D", "mode": "minor", "reasoning": "dramatic tension"},
                        "instruments": ["full_orchestra", "timpani", "brass"],
                        "dynamics": "f to ff with dramatic contrasts",
                        "musical_structure": "through-composed with climactic development",
                        "special_techniques": ["tremolo", "crescendo", "sforzando"],
                        "confidence": 0.88
                    },
                    example_type=ExampleType.MIXED_EMOTIONS,
                    complexity=DreamComplexity.MODERATE,
                    keywords=["thunderstorm", "terrified", "exhilarated", "power", "intense"]
                )
            ],
            
            ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION: [
                DreamExample(
                    dream_text="I found a golden key that opened doors to rooms filled with memories.",
                    analysis={
                        "symbols": [
                            {
                                "element": "golden key",
                                "interpretation": "access to hidden knowledge or memories",
                                "psychological_meaning": "unlocking subconscious insights",
                                "emotional_significance": "discovery and revelation",
                                "musical_relevance": "bright, revelatory musical phrases"
                            },
                            {
                                "element": "doors",
                                "interpretation": "transitions and opportunities",
                                "psychological_meaning": "choices and new possibilities",
                                "emotional_significance": "anticipation and potential",
                                "musical_relevance": "modulation and key changes"
                            }
                        ],
                        "overall_symbolic_theme": "discovery and revelation",
                        "archetypal_patterns": ["the quest", "transformation"],
                        "confidence": 0.90
                    },
                    example_type=ExampleType.SYMBOLIC_CONTENT,
                    complexity=DreamComplexity.COMPLEX,
                    keywords=["key", "golden", "doors", "rooms", "memories", "unlock"]
                )
            ],
            
            ZeroShotTask.MOOD_TO_MUSIC_MAPPING: [
                DreamExample(
                    dream_text="I felt overwhelmed by sadness as I walked through an empty house.",
                    analysis={
                        "mood_analysis": {
                            "primary_mood": "melancholy",
                            "intensity": 8,
                            "emotional_texture": "heavy and introspective"
                        },
                        "musical_mapping": {
                            "tempo": {"bpm": 60, "description": "slow and contemplative"},
                            "key": {"key": "F", "mode": "minor", "reasoning": "deep sadness"},
                            "rhythm": "simple, sparse patterns",
                            "harmony": "minor chords with added tensions",
                            "melody": "descending phrases with long notes"
                        },
                        "instrumentation": ["solo piano", "cello", "soft strings"],
                        "confidence": 0.86
                    },
                    example_type=ExampleType.BASIC_EMOTION,
                    complexity=DreamComplexity.SIMPLE,
                    keywords=["sadness", "overwhelmed", "empty", "house", "melancholy"]
                )
            ],
            
            ZeroShotTask.DREAM_NARRATIVE_ANALYSIS: [
                DreamExample(
                    dream_text="First I was in my childhood home, then suddenly in a forest, finally flying above clouds.",
                    analysis={
                        "narrative_structure": {
                            "acts": [
                                {"setting": "childhood home", "mood": "nostalgic", "significance": "past/memory"},
                                {"setting": "forest", "mood": "uncertain", "significance": "transition/unknown"},
                                {"setting": "above clouds", "mood": "liberated", "significance": "transcendence"}
                            ],
                            "progression": "past → present → future/transcendence",
                            "pacing": "rapid transitions with emotional shifts"
                        },
                        "thematic_elements": ["journey", "transformation", "elevation"],
                        "musical_narrative": {
                            "structure": "three-movement composition",
                            "transitions": "smooth modulations between sections",
                            "climax": "flying sequence with soaring melodies"
                        },
                        "confidence": 0.84
                    },
                    example_type=ExampleType.NARRATIVE_STRUCTURE,
                    complexity=DreamComplexity.COMPLEX,
                    keywords=["childhood", "home", "forest", "flying", "clouds", "transition"]
                )
            ]
        }
    
    def calculate_example_quality(self, example: DreamExample) -> float:
        """
        Calculate the overall quality score of an example.
        
        Quality is based on:
        - Analysis completeness
        - Confidence score
        - Keyword richness
        - Example type appropriateness
        """
        score = 0.0
        
        # Base confidence score (40% weight)
        if "confidence" in example.analysis:
            score += example.analysis["confidence"] * 0.4
        
        # Analysis completeness (30% weight)
        analysis_fields = len(example.analysis.keys())
        completeness = min(analysis_fields / 5, 1.0)  # Normalize to 5 expected fields
        score += completeness * 0.3
        
        # Keyword richness (20% weight)
        keyword_richness = min(len(example.keywords) / 6, 1.0)  # Normalize to 6 keywords
        score += keyword_richness * 0.2
        
        # Dream text quality (10% weight)
        text_quality = min(len(example.dream_text.split()) / 15, 1.0)  # Normalize to 15 words
        score += text_quality * 0.1
        
        return min(score, 1.0)
    
    def calculate_representativeness(self, example: DreamExample, task: ZeroShotTask) -> float:
        """
        Calculate how representative an example is for the given task.
        
        Representativeness considers:
        - Typical complexity for the task
        - Common example type
        - Balanced emotional content
        """
        score = 0.0
        
        # Complexity representativeness (40% weight)
        if example.complexity == DreamComplexity.MODERATE:
            score += 0.4  # Moderate complexity is most representative
        elif example.complexity in [DreamComplexity.SIMPLE, DreamComplexity.COMPLEX]:
            score += 0.3
        else:
            score += 0.2
        
        # Example type representativeness (30% weight)
        representative_types = {
            ZeroShotTask.DREAM_EMOTION_ANALYSIS: ExampleType.BASIC_EMOTION,
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION: ExampleType.MUSICAL_SPECIFIC,
            ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION: ExampleType.SYMBOLIC_CONTENT,
            ZeroShotTask.MOOD_TO_MUSIC_MAPPING: ExampleType.BASIC_EMOTION,
            ZeroShotTask.DREAM_NARRATIVE_ANALYSIS: ExampleType.NARRATIVE_STRUCTURE
        }
        
        if example.example_type == representative_types.get(task):
            score += 0.3
        else:
            score += 0.1
        
        # Emotional balance (30% weight)
        # Check if example has balanced emotional content (not too extreme)
        if "confidence" in example.analysis:
            confidence = example.analysis["confidence"]
            if 0.8 <= confidence <= 0.95:  # Sweet spot for representativeness
                score += 0.3
            elif 0.7 <= confidence < 0.8 or 0.95 < confidence <= 1.0:
                score += 0.2
            else:
                score += 0.1
        
        return min(score, 1.0)
    
    def select_one_shot_example(
        self,
        task: ZeroShotTask,
        dream_text: str,
        complexity: DreamComplexity,
        keywords: List[str]
    ) -> Optional[DreamExample]:
        """
        Select the single best example for one-shot prompting.
        
        Args:
            task: The analysis task
            dream_text: The dream description
            complexity: Analyzed complexity level
            keywords: Extracted keywords
            
        Returns:
            The selected example or None if no suitable example found
        """
        if task not in self.example_database:
            logger.warning(f"No examples available for task {task.value}")
            return None
        
        available_examples = self.example_database[task]
        
        # Filter by quality threshold
        quality_examples = [
            ex for ex in available_examples 
            if self.calculate_example_quality(ex) >= self.config.quality_threshold
        ]
        
        if not quality_examples:
            if self.config.fallback_to_representative:
                # Fallback to best available example
                quality_examples = available_examples
            else:
                logger.warning(f"No examples meet quality threshold for task {task.value}")
                return None
        
        # Apply selection strategy
        if self.config.strategy == OneShotStrategy.BEST_MATCH:
            return self._select_best_match(quality_examples, dream_text, keywords, complexity)
        elif self.config.strategy == OneShotStrategy.REPRESENTATIVE:
            return self._select_representative(quality_examples, task)
        elif self.config.strategy == OneShotStrategy.COMPLEXITY_MATCHED:
            return self._select_complexity_matched(quality_examples, complexity)
        elif self.config.strategy == OneShotStrategy.BALANCED:
            return self._select_balanced(quality_examples, task, dream_text, keywords, complexity)
        elif self.config.strategy == OneShotStrategy.RANDOM_QUALITY:
            return self._select_random_quality(quality_examples)
        else:
            # Default to best match
            return self._select_best_match(quality_examples, dream_text, keywords, complexity)
    
    def _select_best_match(
        self, 
        examples: List[DreamExample], 
        dream_text: str, 
        keywords: List[str],
        complexity: DreamComplexity
    ) -> DreamExample:
        """Select the example with highest relevance to the dream."""
        def relevance_score(example):
            base_score = example._calculate_relevance_score(dream_text, keywords)
            
            # Complexity boost
            if self.config.use_complexity_boost and example.complexity == complexity:
                base_score += 0.2
            
            return base_score
        
        return max(examples, key=relevance_score)
    
    def _select_representative(self, examples: List[DreamExample], task: ZeroShotTask) -> DreamExample:
        """Select the most representative example for the task."""
        return max(examples, key=lambda ex: self.calculate_representativeness(ex, task))
    
    def _select_complexity_matched(self, examples: List[DreamExample], complexity: DreamComplexity) -> DreamExample:
        """Select example that best matches the dream complexity."""
        # First try exact complexity match
        exact_matches = [ex for ex in examples if ex.complexity == complexity]
        if exact_matches:
            return max(exact_matches, key=self.calculate_example_quality)
        
        # Then try adjacent complexity levels
        complexity_order = [DreamComplexity.SIMPLE, DreamComplexity.MODERATE, 
                          DreamComplexity.COMPLEX, DreamComplexity.HIGHLY_COMPLEX]
        target_idx = complexity_order.index(complexity)
        
        for offset in [1, -1, 2, -2]:
            adj_idx = target_idx + offset
            if 0 <= adj_idx < len(complexity_order):
                adj_matches = [ex for ex in examples if ex.complexity == complexity_order[adj_idx]]
                if adj_matches:
                    return max(adj_matches, key=self.calculate_example_quality)
        
        # Fallback to highest quality
        return max(examples, key=self.calculate_example_quality)
    
    def _select_balanced(
        self, 
        examples: List[DreamExample], 
        task: ZeroShotTask,
        dream_text: str, 
        keywords: List[str],
        complexity: DreamComplexity
    ) -> DreamExample:
        """Select example balancing relevance and representativeness."""
        def balanced_score(example):
            relevance = example._calculate_relevance_score(dream_text, keywords)
            representativeness = self.calculate_representativeness(example, task)
            
            score = (relevance * self.config.relevance_weight + 
                    representativeness * self.config.representativeness_weight)
            
            # Complexity boost
            if self.config.use_complexity_boost and example.complexity == complexity:
                score += 0.1
            
            return score
        
        return max(examples, key=balanced_score)
    
    def _select_random_quality(self, examples: List[DreamExample]) -> DreamExample:
        """Select randomly from high-quality examples."""
        import random
        
        # Sort by quality and take top 50%
        sorted_examples = sorted(examples, key=self.calculate_example_quality, reverse=True)
        top_half = sorted_examples[:max(1, len(sorted_examples) // 2)]
        
        return random.choice(top_half)

    def build_one_shot_prompt(
        self,
        task: ZeroShotTask,
        dream_text: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build a one-shot prompt with exactly one example.

        Args:
            task: The analysis task
            dream_text: The dream description
            additional_context: Optional additional context

        Returns:
            Dictionary containing the complete one-shot prompt
        """
        # Analyze dream for example selection
        from .dynamic_shot_prompts import DynamicShotPromptBuilder
        temp_builder = DynamicShotPromptBuilder()
        complexity = temp_builder.analyze_dream_complexity(dream_text)
        keywords = temp_builder.extract_keywords(dream_text)

        # Select the single best example
        selected_example = self.select_one_shot_example(task, dream_text, complexity, keywords)

        # Track selection for analytics
        selection_key = f"{task.value}:{self.config.strategy.value}"
        self.selection_history[selection_key] = self.selection_history.get(selection_key, 0) + 1

        # Build the prompt
        system_message = self._build_one_shot_system_message(task, complexity, selected_example)
        user_message = self._build_one_shot_user_message(
            task, dream_text, selected_example, additional_context
        )

        # Safe example text truncation
        example_text = None
        if selected_example and selected_example.dream_text:
            try:
                # Ensure safe UTF-8 handling and length checking
                text = str(selected_example.dream_text)
                if len(text) > 100:
                    # Find a safe truncation point to avoid breaking UTF-8
                    truncate_point = 100
                    while truncate_point > 80 and not text[truncate_point-1].isspace():
                        truncate_point -= 1
                    example_text = text[:truncate_point].rstrip() + "..."
                else:
                    example_text = text
            except (AttributeError, TypeError, UnicodeError):
                example_text = "Example text unavailable"

        # Safe example quality calculation
        example_quality = 0.0
        if selected_example:
            try:
                example_quality = self.calculate_example_quality(selected_example)
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning(f"Failed to calculate example quality: {e}")
                example_quality = 0.0

        return {
            "system_message": system_message,
            "user_message": user_message,
            "task": task.value,
            "complexity": complexity.value,
            "strategy": self.config.strategy.value,
            "selected_example": example_text,
            "example_quality": example_quality,
            "keywords": keywords[:10]
        }

    def _build_one_shot_system_message(
        self,
        task: ZeroShotTask,
        complexity: DreamComplexity,
        example: Optional[DreamExample]
    ) -> str:
        """Build system message for one-shot prompting."""
        task_descriptions = {
            ZeroShotTask.DREAM_EMOTION_ANALYSIS: "dream emotion analysis with expertise in psychology and emotional intelligence",
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION: "musical style recommendation with deep knowledge of music theory and composition",
            ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION: "dream symbolism interpretation with understanding of psychology and archetypal patterns",
            ZeroShotTask.MOOD_TO_MUSIC_MAPPING: "mood-to-music mapping with expertise in emotional expression through music",
            ZeroShotTask.DREAM_NARRATIVE_ANALYSIS: "dream narrative analysis with knowledge of storytelling and psychological themes"
        }

        base_instruction = f"You are an expert in {task_descriptions.get(task, task.value.replace('_', ' '))}."

        complexity_guidance = {
            DreamComplexity.SIMPLE: "Focus on clear, direct analysis of the main elements.",
            DreamComplexity.MODERATE: "Provide balanced analysis considering multiple elements and their interactions.",
            DreamComplexity.COMPLEX: "Deliver nuanced analysis addressing multiple layers and interconnections.",
            DreamComplexity.HIGHLY_COMPLEX: "Provide comprehensive analysis covering all narrative elements and their deep meanings."
        }

        example_guidance = (
            "You will be provided with one high-quality example to guide your analysis approach. "
            "Use this example to understand the expected format, depth, and style of analysis, "
            "but adapt your response to the specific content of the new dream."
        ) if example else (
            "No example is available for this analysis. "
            "Provide thorough analysis based on your expertise."
        )

        system_parts = [
            base_instruction,
            "",
            f"COMPLEXITY LEVEL: {complexity.value.upper()}",
            complexity_guidance[complexity],
            "",
            example_guidance,
            "",
            "Provide thorough, insightful analysis in the specified JSON format.",
            "Ensure your confidence score reflects the clarity and completeness of the dream content.",
            "Be precise, analytical, and maintain consistency with the example's approach."
        ]

        return "\n".join(system_parts)

    def _build_one_shot_user_message(
        self,
        task: ZeroShotTask,
        dream_text: str,
        example: Optional[DreamExample],
        additional_context: Optional[str] = None
    ) -> str:
        """Build user message with one example."""
        user_parts = []

        # Add the single example if available
        if example:
            user_parts.extend([
                "Here is a high-quality example to guide your analysis:",
                "",
                "EXAMPLE:",
                f"Dream: \"{example.dream_text}\"",
                f"Analysis: {json.dumps(example.analysis, indent=2)}",
                ""
            ])

        # Add the actual dream to analyze
        user_parts.extend([
            "Now analyze this new dream using the same approach:",
            "",
            "DREAM TO ANALYZE:",
            f"\"{dream_text}\"",
            ""
        ])

        # Add additional context if provided
        if additional_context:
            user_parts.extend([
                "ADDITIONAL CONTEXT:",
                additional_context,
                ""
            ])

        user_parts.extend([
            "Please provide your analysis in the same JSON format as the example above." if example else "Please provide your analysis in JSON format.",
            "Analysis:"
        ])

        return "\n".join(user_parts)

    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get statistics about example selection patterns."""
        if not self.selection_history:
            return {
                "total_selections": 0,
                "strategy_usage": {},
                "most_used_strategies": []
            }

        total_selections = sum(self.selection_history.values())

        # Group by strategy
        strategy_usage = {}
        for key, count in self.selection_history.items():
            _, strategy = key.split(':', 1)
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + count

        # Most used strategies
        sorted_strategies = sorted(strategy_usage.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_selections": total_selections,
            "strategy_usage": strategy_usage,
            "most_used_strategies": sorted_strategies[:3],
            "selection_history": dict(self.selection_history)
        }

    def get_example_statistics(self) -> Dict[str, Any]:
        """Get statistics about the example database."""
        stats = {
            "total_examples": sum(len(examples) for examples in self.example_database.values()),
            "examples_by_task": {task.value: len(examples) for task, examples in self.example_database.items()},
            "quality_distribution": {},
            "complexity_distribution": {},
            "example_types": {}
        }

        # Calculate quality and complexity distributions
        all_examples = []
        for examples in self.example_database.values():
            all_examples.extend(examples)

        for example in all_examples:
            # Quality distribution with non-overlapping buckets
            quality = self.calculate_example_quality(example)

            # Create non-overlapping quality buckets using half-open intervals [a, b)
            if quality < 0.5:
                quality_bucket = "[0.0, 0.5)"
            elif quality < 0.6:
                quality_bucket = "[0.5, 0.6)"
            elif quality < 0.7:
                quality_bucket = "[0.6, 0.7)"
            elif quality < 0.8:
                quality_bucket = "[0.7, 0.8)"
            elif quality < 0.9:
                quality_bucket = "[0.8, 0.9)"
            else:
                quality_bucket = "[0.9, 1.0]"

            stats["quality_distribution"][quality_bucket] = stats["quality_distribution"].get(quality_bucket, 0) + 1

            # Complexity distribution
            complexity = example.complexity.value
            stats["complexity_distribution"][complexity] = stats["complexity_distribution"].get(complexity, 0) + 1

            # Example types
            example_type = example.example_type.value
            stats["example_types"][example_type] = stats["example_types"].get(example_type, 0) + 1

        return stats

    def add_example(self, task: ZeroShotTask, example: DreamExample) -> None:
        """Add a new example to the database."""
        if task not in self.example_database:
            self.example_database[task] = []

        # Validate example quality
        quality = self.calculate_example_quality(example)
        if quality < 0.7:
            logger.warning(f"Adding example with low quality score: {quality:.2f}")

        self.example_database[task].append(example)
        logger.info(f"Added new example for task {task.value} (quality: {quality:.2f})")

    def clear_selection_history(self) -> None:
        """Clear the selection history."""
        self.selection_history.clear()
        logger.info("Cleared one-shot selection history")
