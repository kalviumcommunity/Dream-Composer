"""
Multi-Shot Prompting for Dream Composer.

This module implements multi-shot prompting techniques that provide multiple
carefully selected examples to guide AI analysis. Multi-shot prompting offers
comprehensive guidance through diverse examples that cover different aspects,
scenarios, and complexity levels.
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


class MultiShotStrategy(Enum):
    """Strategies for selecting multiple examples in multi-shot prompting."""
    DIVERSE_COVERAGE = "diverse_coverage"         # Cover different aspects and scenarios
    COMPLEXITY_PROGRESSION = "complexity_progression"  # Examples from simple to complex
    THEMATIC_CLUSTERING = "thematic_clustering"   # Group examples by themes
    BALANCED_REPRESENTATION = "balanced_representation"  # Balance across all dimensions
    QUALITY_RANKED = "quality_ranked"            # Select highest quality examples
    CONTEXTUAL_SIMILARITY = "contextual_similarity"  # Examples similar to input context


@dataclass
class MultiShotConfig:
    """Configuration for multi-shot prompting."""
    strategy: MultiShotStrategy = MultiShotStrategy.DIVERSE_COVERAGE
    min_examples: int = 2                        # Minimum number of examples
    max_examples: int = 5                        # Maximum number of examples
    quality_threshold: float = 0.7               # Minimum quality score for examples
    diversity_weight: float = 0.4                # Weight for diversity in selection
    relevance_weight: float = 0.3                # Weight for relevance to input
    quality_weight: float = 0.3                  # Weight for example quality
    complexity_spread: bool = True               # Ensure examples span complexity levels
    type_diversity: bool = True                  # Ensure examples cover different types
    fallback_to_available: bool = True           # Use available examples if criteria not met


class MultiShotPromptBuilder:
    """
    Builder for multi-shot prompts that provide multiple examples.
    
    Multi-shot prompting provides comprehensive guidance through multiple,
    carefully selected examples that cover different aspects and scenarios.
    """
    
    def __init__(self, config: Optional[MultiShotConfig] = None):
        self.config = config or MultiShotConfig()
        self.example_database = self._initialize_example_database()
        self.selection_history = {}  # Track selection patterns
        self.diversity_metrics = {}  # Track diversity statistics
    
    def _initialize_example_database(self) -> Dict[ZeroShotTask, List[DreamExample]]:
        """Initialize comprehensive database of examples for multi-shot prompting."""
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
                ),
                DreamExample(
                    dream_text="I was being chased through dark corridors by an unseen presence, feeling terrified and helpless.",
                    analysis={
                        "primary_emotions": ["fear", "anxiety", "helplessness"],
                        "emotion_intensities": {"fear": 9, "anxiety": 8, "helplessness": 7},
                        "emotional_progression": "escalating negative emotions",
                        "dominant_mood": "nightmare",
                        "emotional_triggers": ["pursuit", "darkness", "unknown threat"],
                        "confidence": 0.91
                    },
                    example_type=ExampleType.BASIC_EMOTION,
                    complexity=DreamComplexity.SIMPLE,
                    keywords=["chased", "dark", "corridors", "terrified", "helpless", "fear"]
                ),
                DreamExample(
                    dream_text="I was in a vast library where books floated and rearranged themselves. I felt curious, then overwhelmed, then enlightened as I understood their pattern.",
                    analysis={
                        "primary_emotions": ["curiosity", "overwhelm", "enlightenment"],
                        "emotion_intensities": {"curiosity": 8, "overwhelm": 6, "enlightenment": 9},
                        "emotional_progression": "complex emotional journey with resolution",
                        "dominant_mood": "transformative",
                        "emotional_triggers": ["magical environment", "knowledge acquisition", "pattern recognition"],
                        "confidence": 0.85
                    },
                    example_type=ExampleType.SYMBOLIC_CONTENT,
                    complexity=DreamComplexity.COMPLEX,
                    keywords=["library", "books", "floating", "curious", "overwhelmed", "enlightened", "pattern"]
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
                ),
                DreamExample(
                    dream_text="I was floating in space, surrounded by singing stars and cosmic melodies.",
                    analysis={
                        "recommended_style": "ambient_electronic",
                        "tempo": {"bpm": 60, "description": "ethereal and spacious"},
                        "key_signature": {"key": "C", "mode": "lydian", "reasoning": "otherworldly and expansive"},
                        "instruments": ["synthesizers", "ethereal_pads", "cosmic_effects"],
                        "dynamics": "pp to mp with gradual evolution",
                        "musical_structure": "ambient form with layered textures",
                        "special_techniques": ["reverb", "delay", "pitch_shifting"],
                        "confidence": 0.86
                    },
                    example_type=ExampleType.SYMBOLIC_CONTENT,
                    complexity=DreamComplexity.COMPLEX,
                    keywords=["floating", "space", "stars", "singing", "cosmic", "melodies"]
                ),
                DreamExample(
                    dream_text="I was walking through a jazz club where the music changed with my emotions.",
                    analysis={
                        "recommended_style": "adaptive_jazz",
                        "tempo": {"bpm": 110, "description": "syncopated and responsive"},
                        "key_signature": {"key": "Bb", "mode": "major", "reasoning": "warm and expressive"},
                        "instruments": ["piano_trio", "saxophone", "double_bass"],
                        "dynamics": "mf with expressive variations",
                        "musical_structure": "jazz standard form with improvisation",
                        "special_techniques": ["swing_rhythm", "blue_notes", "call_and_response"],
                        "confidence": 0.89
                    },
                    example_type=ExampleType.MIXED_EMOTIONS,
                    complexity=DreamComplexity.MODERATE,
                    keywords=["jazz", "club", "music", "emotions", "walking", "changing"]
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
                ),
                DreamExample(
                    dream_text="I was climbing a spiral staircase that seemed to go on forever, with mirrors on every level.",
                    analysis={
                        "symbols": [
                            {
                                "element": "spiral staircase",
                                "interpretation": "spiritual ascension and personal growth",
                                "psychological_meaning": "journey of self-improvement",
                                "emotional_significance": "aspiration and determination",
                                "musical_relevance": "ascending melodic sequences"
                            },
                            {
                                "element": "mirrors",
                                "interpretation": "self-reflection and multiple perspectives",
                                "psychological_meaning": "examining different aspects of self",
                                "emotional_significance": "introspection and self-awareness",
                                "musical_relevance": "canon and echo effects"
                            }
                        ],
                        "overall_symbolic_theme": "self-discovery and growth",
                        "archetypal_patterns": ["the journey", "the mirror"],
                        "confidence": 0.87
                    },
                    example_type=ExampleType.SYMBOLIC_CONTENT,
                    complexity=DreamComplexity.COMPLEX,
                    keywords=["climbing", "spiral", "staircase", "forever", "mirrors", "level"]
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
                ),
                DreamExample(
                    dream_text="I was energized and excited, running through a field of sunflowers.",
                    analysis={
                        "mood_analysis": {
                            "primary_mood": "exuberant",
                            "intensity": 9,
                            "emotional_texture": "bright and energetic"
                        },
                        "musical_mapping": {
                            "tempo": {"bpm": 140, "description": "fast and lively"},
                            "key": {"key": "D", "mode": "major", "reasoning": "bright and joyful"},
                            "rhythm": "syncopated, driving patterns",
                            "harmony": "major chords with added sixths",
                            "melody": "ascending phrases with quick runs"
                        },
                        "instrumentation": ["acoustic guitar", "violin", "percussion"],
                        "confidence": 0.92
                    },
                    example_type=ExampleType.BASIC_EMOTION,
                    complexity=DreamComplexity.SIMPLE,
                    keywords=["energized", "excited", "running", "sunflowers", "field"]
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
    
    def calculate_example_diversity(self, examples: List[DreamExample]) -> float:
        """
        Calculate the diversity score of a set of examples.
        
        Diversity considers:
        - Complexity spread
        - Example type variety
        - Keyword uniqueness
        - Emotional range
        """
        if not examples:
            return 0.0
        
        diversity_score = 0.0
        
        # Complexity diversity (25% weight)
        complexities = [ex.complexity for ex in examples]
        unique_complexities = len(set(complexities))
        max_complexities = min(len(examples), len(DreamComplexity))
        complexity_diversity = unique_complexities / max_complexities
        diversity_score += complexity_diversity * 0.25
        
        # Example type diversity (25% weight)
        types = [ex.example_type for ex in examples]
        unique_types = len(set(types))
        max_types = min(len(examples), len(ExampleType))
        type_diversity = unique_types / max_types
        diversity_score += type_diversity * 0.25
        
        # Keyword diversity (25% weight)
        all_keywords = set()
        total_keywords = 0
        for ex in examples:
            all_keywords.update(ex.keywords)
            total_keywords += len(ex.keywords)
        
        if total_keywords > 0:
            keyword_diversity = len(all_keywords) / total_keywords
            diversity_score += keyword_diversity * 0.25
        
        # Emotional range diversity (25% weight)
        # Check for variety in emotional content
        emotional_variety = 0.0
        if len(examples) > 1:
            # Simple heuristic: check for different emotional patterns
            emotional_patterns = set()
            for ex in examples:
                if "primary_emotions" in ex.analysis:
                    emotions = ex.analysis["primary_emotions"]
                    if emotions:
                        # Create pattern from first emotion
                        emotional_patterns.add(emotions[0] if isinstance(emotions, list) else str(emotions))
            
            emotional_variety = len(emotional_patterns) / len(examples)
        
        diversity_score += emotional_variety * 0.25
        
        return min(diversity_score, 1.0)

    def calculate_example_relevance(self, example: DreamExample, dream_text: str, keywords: List[str]) -> float:
        """Calculate relevance score between an example and dream content."""
        return example._calculate_relevance_score(dream_text, keywords)

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

    def select_multi_shot_examples(
        self,
        task: ZeroShotTask,
        dream_text: str,
        complexity: DreamComplexity,
        keywords: List[str]
    ) -> List[DreamExample]:
        """
        Select multiple examples for multi-shot prompting.

        Args:
            task: The analysis task
            dream_text: The dream description
            complexity: Analyzed complexity level
            keywords: Extracted keywords

        Returns:
            List of selected examples
        """
        if task not in self.example_database:
            logger.warning(f"No examples available for task {task.value}")
            return []

        available_examples = self.example_database[task]

        # Filter by quality threshold
        quality_examples = [
            ex for ex in available_examples
            if self.calculate_example_quality(ex) >= self.config.quality_threshold
        ]

        if not quality_examples:
            if self.config.fallback_to_available:
                quality_examples = available_examples
            else:
                logger.warning(f"No examples meet quality threshold for task {task.value}")
                return []

        # Apply selection strategy
        if self.config.strategy == MultiShotStrategy.DIVERSE_COVERAGE:
            return self._select_diverse_coverage(quality_examples, dream_text, keywords, complexity)
        elif self.config.strategy == MultiShotStrategy.COMPLEXITY_PROGRESSION:
            return self._select_complexity_progression(quality_examples, complexity)
        elif self.config.strategy == MultiShotStrategy.THEMATIC_CLUSTERING:
            return self._select_thematic_clustering(quality_examples, dream_text, keywords)
        elif self.config.strategy == MultiShotStrategy.BALANCED_REPRESENTATION:
            return self._select_balanced_representation(quality_examples, dream_text, keywords, complexity)
        elif self.config.strategy == MultiShotStrategy.QUALITY_RANKED:
            return self._select_quality_ranked(quality_examples)
        elif self.config.strategy == MultiShotStrategy.CONTEXTUAL_SIMILARITY:
            return self._select_contextual_similarity(quality_examples, dream_text, keywords, complexity)
        else:
            # Default to diverse coverage
            return self._select_diverse_coverage(quality_examples, dream_text, keywords, complexity)

    def _select_diverse_coverage(
        self,
        examples: List[DreamExample],
        dream_text: str,
        keywords: List[str],
        complexity: DreamComplexity
    ) -> List[DreamExample]:
        """Select examples that provide diverse coverage of different aspects."""
        if len(examples) <= self.config.max_examples:
            return examples[:self.config.max_examples]

        selected = []
        remaining = examples.copy()

        # Start with the most relevant example
        best_relevance = max(remaining, key=lambda ex: self.calculate_example_relevance(ex, dream_text, keywords))
        selected.append(best_relevance)
        remaining.remove(best_relevance)

        # Add examples that maximize diversity
        while len(selected) < self.config.max_examples and remaining:
            best_candidate = None
            best_score = -1

            for candidate in remaining:
                # Calculate diversity if we add this candidate
                test_set = selected + [candidate]
                diversity = self.calculate_example_diversity(test_set)
                relevance = self.calculate_example_relevance(candidate, dream_text, keywords)
                quality = self.calculate_example_quality(candidate)

                # Combined score
                score = (diversity * self.config.diversity_weight +
                        relevance * self.config.relevance_weight +
                        quality * self.config.quality_weight)

                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break

        # Ensure minimum examples
        while len(selected) < self.config.min_examples and remaining:
            selected.append(remaining.pop(0))

        return selected

    def _select_complexity_progression(
        self,
        examples: List[DreamExample],
        target_complexity: DreamComplexity
    ) -> List[DreamExample]:
        """Select examples that show progression from simple to complex."""
        # Sort by complexity
        complexity_order = [DreamComplexity.SIMPLE, DreamComplexity.MODERATE,
                          DreamComplexity.COMPLEX, DreamComplexity.HIGHLY_COMPLEX]

        sorted_examples = sorted(examples, key=lambda ex: complexity_order.index(ex.complexity))

        # Select examples to show progression
        selected = []
        target_idx = complexity_order.index(target_complexity)

        # Include examples from different complexity levels
        for i, complexity_level in enumerate(complexity_order):
            level_examples = [ex for ex in sorted_examples if ex.complexity == complexity_level]
            if level_examples:
                # Prefer examples closer to target complexity
                weight = 1.0 / (abs(i - target_idx) + 1)
                best_example = max(level_examples, key=lambda ex: self.calculate_example_quality(ex) * weight)
                selected.append(best_example)

                if len(selected) >= self.config.max_examples:
                    break

        # Ensure we have minimum examples
        if len(selected) < self.config.min_examples:
            remaining = [ex for ex in sorted_examples if ex not in selected]
            while len(selected) < self.config.min_examples and remaining:
                selected.append(remaining.pop(0))

        return selected[:self.config.max_examples]

    def _select_thematic_clustering(
        self,
        examples: List[DreamExample],
        dream_text: str,
        keywords: List[str]
    ) -> List[DreamExample]:
        """Select examples that cluster around similar themes."""
        # Score examples by thematic similarity
        scored_examples = []
        for example in examples:
            relevance = self.calculate_example_relevance(example, dream_text, keywords)
            quality = self.calculate_example_quality(example)

            # Thematic similarity based on keywords and example type
            thematic_score = relevance * 0.7 + quality * 0.3
            scored_examples.append((example, thematic_score))

        # Sort by thematic score and select top examples
        scored_examples.sort(key=lambda x: x[1], reverse=True)
        selected = [ex for ex, score in scored_examples[:self.config.max_examples]]

        # Ensure minimum examples
        while len(selected) < self.config.min_examples and len(selected) < len(examples):
            remaining = [ex for ex in examples if ex not in selected]
            if remaining:
                selected.append(remaining[0])

        return selected

    def _select_balanced_representation(
        self,
        examples: List[DreamExample],
        dream_text: str,
        keywords: List[str],
        complexity: DreamComplexity
    ) -> List[DreamExample]:
        """Select examples that provide balanced representation across all dimensions."""
        selected = []

        # Ensure representation across example types
        type_groups = {}
        for example in examples:
            if example.example_type not in type_groups:
                type_groups[example.example_type] = []
            type_groups[example.example_type].append(example)

        # Select best example from each type group
        for example_type, type_examples in type_groups.items():
            if len(selected) >= self.config.max_examples:
                break

            best_example = max(type_examples, key=lambda ex: (
                self.calculate_example_relevance(ex, dream_text, keywords) * 0.5 +
                self.calculate_example_quality(ex) * 0.5
            ))
            selected.append(best_example)

        # Fill remaining slots with highest scoring examples
        remaining = [ex for ex in examples if ex not in selected]
        while len(selected) < self.config.max_examples and remaining:
            best_remaining = max(remaining, key=lambda ex: (
                self.calculate_example_relevance(ex, dream_text, keywords) * self.config.relevance_weight +
                self.calculate_example_quality(ex) * self.config.quality_weight
            ))
            selected.append(best_remaining)
            remaining.remove(best_remaining)

        # Ensure minimum examples
        while len(selected) < self.config.min_examples and remaining:
            selected.append(remaining.pop(0))

        return selected

    def _select_quality_ranked(self, examples: List[DreamExample]) -> List[DreamExample]:
        """Select examples ranked by quality score."""
        sorted_examples = sorted(examples, key=self.calculate_example_quality, reverse=True)

        num_examples = max(self.config.min_examples,
                          min(self.config.max_examples, len(sorted_examples)))

        return sorted_examples[:num_examples]

    def _select_contextual_similarity(
        self,
        examples: List[DreamExample],
        dream_text: str,
        keywords: List[str],
        complexity: DreamComplexity
    ) -> List[DreamExample]:
        """Select examples most similar to the input context."""
        scored_examples = []

        for example in examples:
            relevance = self.calculate_example_relevance(example, dream_text, keywords)
            quality = self.calculate_example_quality(example)

            # Complexity similarity bonus
            complexity_bonus = 0.2 if example.complexity == complexity else 0.0

            total_score = relevance * 0.6 + quality * 0.3 + complexity_bonus
            scored_examples.append((example, total_score))

        # Sort by similarity score and select top examples
        scored_examples.sort(key=lambda x: x[1], reverse=True)

        num_examples = max(self.config.min_examples,
                          min(self.config.max_examples, len(scored_examples)))

        return [ex for ex, score in scored_examples[:num_examples]]

    def build_multi_shot_prompt(
        self,
        task: ZeroShotTask,
        dream_text: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build a multi-shot prompt with multiple examples.

        Args:
            task: The analysis task
            dream_text: The dream description
            additional_context: Optional additional context

        Returns:
            Dictionary containing the complete multi-shot prompt
        """
        # Analyze dream for example selection
        from .dynamic_shot_prompts import DynamicShotPromptBuilder
        temp_builder = DynamicShotPromptBuilder()
        complexity = temp_builder.analyze_dream_complexity(dream_text)
        keywords = temp_builder.extract_keywords(dream_text)

        # Select multiple examples
        selected_examples = self.select_multi_shot_examples(task, dream_text, complexity, keywords)

        # Track selection for analytics
        selection_key = f"{task.value}:{self.config.strategy.value}"
        self.selection_history[selection_key] = self.selection_history.get(selection_key, 0) + 1

        # Calculate diversity metrics
        diversity_score = self.calculate_example_diversity(selected_examples)
        avg_quality = sum(self.calculate_example_quality(ex) for ex in selected_examples) / len(selected_examples) if selected_examples else 0.0

        # Build the prompt
        system_message = self._build_multi_shot_system_message(task, complexity, selected_examples)
        user_message = self._build_multi_shot_user_message(
            task, dream_text, selected_examples, additional_context
        )

        return {
            "system_message": system_message,
            "user_message": user_message,
            "task": task.value,
            "complexity": complexity.value,
            "strategy": self.config.strategy.value,
            "num_examples": len(selected_examples),
            "selected_examples": [ex.dream_text[:80] + "..." if len(ex.dream_text) > 80 else ex.dream_text
                                 for ex in selected_examples],
            "diversity_score": diversity_score,
            "average_quality": avg_quality,
            "keywords": keywords[:10]
        }

    def _build_multi_shot_system_message(
        self,
        task: ZeroShotTask,
        complexity: DreamComplexity,
        examples: List[DreamExample]
    ) -> str:
        """Build system message for multi-shot prompting."""
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
            f"You will be provided with {len(examples)} high-quality examples to guide your analysis approach. "
            f"These examples demonstrate different aspects and approaches to {task.value.replace('_', ' ')}. "
            "Study the patterns, formats, and analytical depth shown in these examples, "
            "then apply similar rigor and insight to the new dream while adapting to its specific content."
        ) if examples else (
            "No examples are available for this analysis. "
            "Provide thorough analysis based on your expertise."
        )

        strategy_note = ""
        if self.config.strategy == MultiShotStrategy.DIVERSE_COVERAGE:
            strategy_note = "The examples cover diverse aspects and scenarios to provide comprehensive guidance."
        elif self.config.strategy == MultiShotStrategy.COMPLEXITY_PROGRESSION:
            strategy_note = "The examples show progression from simple to complex analysis approaches."
        elif self.config.strategy == MultiShotStrategy.THEMATIC_CLUSTERING:
            strategy_note = "The examples are thematically related to provide focused guidance."
        elif self.config.strategy == MultiShotStrategy.BALANCED_REPRESENTATION:
            strategy_note = "The examples provide balanced representation across different analytical dimensions."
        elif self.config.strategy == MultiShotStrategy.QUALITY_RANKED:
            strategy_note = "The examples are selected for their exceptional analytical quality."
        elif self.config.strategy == MultiShotStrategy.CONTEXTUAL_SIMILARITY:
            strategy_note = "The examples are chosen for their similarity to the current dream context."

        system_parts = [
            base_instruction,
            "",
            f"COMPLEXITY LEVEL: {complexity.value.upper()}",
            complexity_guidance[complexity],
            "",
            example_guidance,
            "",
            strategy_note,
            "",
            "Provide thorough, insightful analysis in the specified JSON format.",
            "Ensure your confidence score reflects the clarity and completeness of the dream content.",
            "Maintain consistency with the examples' analytical approach while adapting to the new content."
        ]

        return "\n".join(system_parts)

    def _build_multi_shot_user_message(
        self,
        task: ZeroShotTask,
        dream_text: str,
        examples: List[DreamExample],
        additional_context: Optional[str] = None
    ) -> str:
        """Build user message with multiple examples."""
        user_parts = []

        # Add multiple examples if available
        if examples:
            user_parts.extend([
                f"Here are {len(examples)} high-quality examples to guide your analysis:",
                ""
            ])

            for i, example in enumerate(examples, 1):
                user_parts.extend([
                    f"EXAMPLE {i}:",
                    f"Dream: \"{example.dream_text}\"",
                    f"Analysis: {json.dumps(example.analysis, indent=2)}",
                    ""
                ])

        # Add the actual dream to analyze
        user_parts.extend([
            "Now analyze this new dream using the same approach demonstrated in the examples above:",
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
            "Please provide your analysis in the same JSON format as the examples above." if examples else "Please provide your analysis in JSON format.",
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
            "example_types": {},
            "diversity_metrics": {}
        }

        # Calculate quality and complexity distributions
        all_examples = []
        for examples in self.example_database.values():
            all_examples.extend(examples)

        for example in all_examples:
            # Quality distribution with non-overlapping buckets
            quality = self.calculate_example_quality(example)

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

        # Calculate diversity metrics for each task
        for task, examples in self.example_database.items():
            if examples:
                diversity = self.calculate_example_diversity(examples)
                stats["diversity_metrics"][task.value] = diversity

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
        logger.info("Cleared multi-shot selection history")

    def get_strategy_comparison(self, task: ZeroShotTask, dream_text: str) -> Dict[str, Any]:
        """
        Compare how different strategies would select examples for a given dream.

        Args:
            task: The task to perform
            dream_text: The dream to analyze

        Returns:
            Dictionary comparing strategy selections
        """
        from .dynamic_shot_prompts import DynamicShotPromptBuilder
        temp_builder = DynamicShotPromptBuilder()
        complexity = temp_builder.analyze_dream_complexity(dream_text)
        keywords = temp_builder.extract_keywords(dream_text)

        comparison = {
            "dream_text": dream_text[:100] + "..." if len(dream_text) > 100 else dream_text,
            "complexity": complexity.value,
            "keywords": keywords[:5],
            "strategy_selections": {}
        }

        # Test each strategy
        original_strategy = self.config.strategy

        for strategy in MultiShotStrategy:
            try:
                self.config.strategy = strategy
                selected_examples = self.select_multi_shot_examples(task, dream_text, complexity, keywords)

                if selected_examples:
                    diversity = self.calculate_example_diversity(selected_examples)
                    avg_quality = sum(self.calculate_example_quality(ex) for ex in selected_examples) / len(selected_examples)

                    comparison["strategy_selections"][strategy.value] = {
                        "num_examples": len(selected_examples),
                        "example_texts": [ex.dream_text[:60] + "..." for ex in selected_examples],
                        "diversity_score": diversity,
                        "average_quality": avg_quality,
                        "complexities": [ex.complexity.value for ex in selected_examples],
                        "example_types": [ex.example_type.value for ex in selected_examples]
                    }
                else:
                    comparison["strategy_selections"][strategy.value] = {
                        "num_examples": 0,
                        "example_texts": [],
                        "diversity_score": 0.0,
                        "average_quality": 0.0,
                        "complexities": [],
                        "example_types": []
                    }
            except Exception as e:
                comparison["strategy_selections"][strategy.value] = {
                    "error": str(e),
                    "num_examples": 0
                }

        # Restore original strategy
        self.config.strategy = original_strategy

        return comparison
