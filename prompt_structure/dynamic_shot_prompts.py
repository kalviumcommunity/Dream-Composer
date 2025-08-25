"""
Dynamic Shot Prompting for Dream Composer.

This module implements dynamic shot prompting techniques that intelligently
select and adapt the number and type of examples based on dream content,
complexity, and context. Unlike zero-shot or fixed few-shot prompting,
dynamic shot prompting adapts to provide the most relevant examples.
"""

import re
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .zero_shot_prompts import ZeroShotTask

# Configure logging
logger = logging.getLogger(__name__)


class DreamComplexity(Enum):
    """Complexity levels for dream content."""
    SIMPLE = "simple"           # Single scene, clear emotions
    MODERATE = "moderate"       # Multiple scenes or mixed emotions
    COMPLEX = "complex"         # Multiple scenes, complex emotions, symbols
    HIGHLY_COMPLEX = "highly_complex"  # Narrative dreams with multiple elements


class ExampleType(Enum):
    """Types of examples for dynamic shot prompting."""
    BASIC_EMOTION = "basic_emotion"
    MIXED_EMOTIONS = "mixed_emotions"
    SYMBOLIC_CONTENT = "symbolic_content"
    NARRATIVE_STRUCTURE = "narrative_structure"
    SENSORY_RICH = "sensory_rich"
    CULTURAL_CONTEXT = "cultural_context"
    MUSICAL_SPECIFIC = "musical_specific"


@dataclass
class DreamExample:
    """Example for dynamic shot prompting."""
    dream_text: str
    analysis: Dict[str, Any]
    example_type: ExampleType
    complexity: DreamComplexity
    keywords: List[str]
    cultural_context: Optional[str] = None
    
    def _calculate_relevance_score(self, dream_text: str, keywords: List[str]) -> float:
        """
        Calculate relevance score for this example (internal method).

        This is an internal method that should not be called directly.
        Use DynamicShotPromptBuilder.calculate_example_relevance() instead.
        """
        score = 0.0

        # Keyword matching
        dream_lower = dream_text.lower()
        matching_keywords = sum(1 for kw in self.keywords if kw.lower() in dream_lower)
        if self.keywords:
            score += (matching_keywords / len(self.keywords)) * 0.4

        # Content similarity (simple word overlap)
        example_words = set(self.dream_text.lower().split())
        dream_words = set(dream_text.lower().split())
        if example_words and dream_words:
            overlap = len(example_words.intersection(dream_words))
            score += (overlap / len(example_words.union(dream_words))) * 0.3

        # Length similarity
        length_ratio = min(len(dream_text), len(self.dream_text)) / max(len(dream_text), len(self.dream_text))
        score += length_ratio * 0.3

        return min(score, 1.0)


@dataclass
class DynamicShotConfig:
    """Configuration for dynamic shot prompting."""
    max_examples: int = 5
    min_examples: int = 1
    relevance_threshold: float = 0.3
    complexity_weight: float = 0.4
    diversity_weight: float = 0.3
    recency_weight: float = 0.3


class DynamicShotPromptBuilder:
    """
    Builder for dynamic shot prompts that adapt examples based on content.
    
    Dynamic shot prompting intelligently selects the most relevant examples
    for each dream analysis task, adapting to content complexity and context.
    """
    
    def __init__(self, config: Optional[DynamicShotConfig] = None):
        self.config = config or DynamicShotConfig()
        self.example_database = self._initialize_example_database()
        self.usage_history = {}  # Track example usage for diversity
    
    def _initialize_example_database(self) -> Dict[ZeroShotTask, List[DreamExample]]:
        """Initialize database of examples for different tasks."""
        return {
            ZeroShotTask.DREAM_EMOTION_ANALYSIS: [
                DreamExample(
                    dream_text="I was flying over a beautiful city at sunset, feeling incredibly free and joyful.",
                    analysis={
                        "primary_emotions": ["joy", "freedom"],
                        "emotional_intensity": {"joy": 9, "freedom": 8},
                        "emotional_progression": "consistent positive emotions",
                        "dominant_mood": "euphoric",
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
                        "emotional_intensity": {"joy": 7, "anxiety": 8, "sadness": 6},
                        "emotional_progression": "positive to negative emotional shift",
                        "dominant_mood": "transitional",
                        "confidence": 0.85
                    },
                    example_type=ExampleType.MIXED_EMOTIONS,
                    complexity=DreamComplexity.MODERATE,
                    keywords=["garden", "storm", "clouds", "happy", "anxious", "sad", "transition"]
                ),
                DreamExample(
                    dream_text="I was swimming deep underwater with glowing fish, feeling peaceful yet mysterious.",
                    analysis={
                        "primary_emotions": ["peace", "wonder", "mystery"],
                        "emotional_intensity": {"peace": 8, "wonder": 7, "mystery": 6},
                        "emotional_progression": "consistent contemplative emotions",
                        "dominant_mood": "mystical",
                        "confidence": 0.88
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
                        "confidence": 0.90
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
                        "confidence": 0.87
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
                                "musical_relevance": "bright, revelatory musical phrases"
                            },
                            {
                                "element": "doors",
                                "interpretation": "transitions and opportunities",
                                "psychological_meaning": "choices and new possibilities",
                                "musical_relevance": "modulation and key changes"
                            }
                        ],
                        "overall_symbolic_theme": "discovery and revelation",
                        "confidence": 0.89
                    },
                    example_type=ExampleType.SYMBOLIC_CONTENT,
                    complexity=DreamComplexity.COMPLEX,
                    keywords=["key", "golden", "doors", "rooms", "memories", "unlock"]
                )
            ]
        }
    
    def analyze_dream_complexity(self, dream_text: str) -> DreamComplexity:
        """
        Analyze the complexity of a dream description.
        
        Args:
            dream_text: The dream description to analyze
            
        Returns:
            DreamComplexity level
        """
        # Count sentences and clauses
        sentences = len(re.split(r'[.!?]+', dream_text.strip()))
        clauses = len(re.split(r'[,;]+', dream_text))
        
        # Count emotional words
        emotion_words = [
            'happy', 'sad', 'angry', 'fearful', 'joyful', 'anxious', 'peaceful',
            'excited', 'worried', 'calm', 'frustrated', 'elated', 'depressed',
            'nervous', 'content', 'overwhelmed', 'serene', 'agitated'
        ]
        emotion_count = sum(1 for word in emotion_words if word in dream_text.lower())
        
        # Count symbolic elements
        symbolic_words = [
            'flying', 'falling', 'water', 'fire', 'animals', 'doors', 'keys',
            'mirrors', 'bridges', 'mountains', 'ocean', 'forest', 'light', 'darkness'
        ]
        symbol_count = sum(1 for word in symbolic_words if word in dream_text.lower())
        
        # Count transition words (indicating narrative complexity)
        transition_words = [
            'then', 'suddenly', 'after', 'before', 'meanwhile', 'later',
            'first', 'next', 'finally', 'however', 'although', 'because'
        ]
        transition_count = sum(1 for word in transition_words if word in dream_text.lower())
        
        # Calculate complexity score with better weighting
        word_count = len(dream_text.split())

        complexity_score = (
            sentences * 0.4 +
            emotion_count * 0.3 +
            symbol_count * 0.3 +
            transition_count * 0.4 +
            (word_count / 20) * 0.2  # Length factor
        )

        # Determine complexity level with adjusted thresholds
        if complexity_score <= 1.5:
            return DreamComplexity.SIMPLE
        elif complexity_score <= 3.5:
            return DreamComplexity.MODERATE
        elif complexity_score <= 6.0:
            return DreamComplexity.COMPLEX
        else:
            return DreamComplexity.HIGHLY_COMPLEX
    
    def extract_keywords(self, dream_text: str) -> List[str]:
        """Extract relevant keywords from dream text."""
        # Simple keyword extraction (in production, could use NLP libraries)
        words = re.findall(r'\b\w+\b', dream_text.lower())
        
        # Filter out common stop words
        stop_words = {
            'i', 'was', 'were', 'am', 'is', 'are', 'the', 'a', 'an', 'and', 'or',
            'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'that',
            'this', 'it', 'he', 'she', 'they', 'we', 'you', 'me', 'him', 'her'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Return unique keywords, limited to most relevant
        return list(set(keywords))[:20]
    
    def select_dynamic_examples(
        self,
        task: ZeroShotTask,
        dream_text: str,
        complexity: DreamComplexity,
        keywords: List[str],
        additional_context: Optional[str] = None
    ) -> List[DreamExample]:
        """
        Dynamically select the most relevant examples for the given dream.
        
        Args:
            task: The analysis task
            dream_text: The dream description
            complexity: Analyzed complexity level
            keywords: Extracted keywords
            additional_context: Optional additional context
            
        Returns:
            List of selected examples
        """
        if task not in self.example_database:
            return []
        
        available_examples = self.example_database[task]
        
        # Calculate relevance scores for each example
        scored_examples = []
        for example in available_examples:
            relevance_score = self.calculate_example_relevance(example, dream_text, keywords)

            # Adjust score based on complexity matching
            complexity_bonus = self._calculate_complexity_bonus(example.complexity, complexity)

            # Diversity penalty for recently used examples
            usage_penalty = self._calculate_usage_penalty(task, example)

            final_score = relevance_score + complexity_bonus - usage_penalty

            # Use a more lenient threshold for selection
            effective_threshold = max(0.1, self.config.relevance_threshold - 0.2)
            if final_score >= effective_threshold:
                scored_examples.append((example, final_score))
        
        # Sort by score and select top examples
        scored_examples.sort(key=lambda x: x[1], reverse=True)

        # Determine number of examples based on complexity
        num_examples = self._determine_example_count(complexity)

        # If no examples meet threshold but we have examples available, take the best one
        if not scored_examples and available_examples:
            # Take the best available example regardless of threshold
            best_example = max(available_examples,
                             key=lambda ex: self.calculate_example_relevance(ex, dream_text, keywords))
            selected_examples = [best_example]
        else:
            selected_examples = [ex for ex, score in scored_examples[:num_examples]]

        # Update usage history
        self._update_usage_history(task, selected_examples)
        
        logger.debug(f"Selected {len(selected_examples)} examples for {task.value} with complexity {complexity.value}")
        
        return selected_examples
    
    def _determine_example_count(self, complexity: DreamComplexity) -> int:
        """Determine the number of examples based on complexity."""
        complexity_to_count = {
            DreamComplexity.SIMPLE: 1,
            DreamComplexity.MODERATE: 2,
            DreamComplexity.COMPLEX: 3,
            DreamComplexity.HIGHLY_COMPLEX: 4
        }
        
        base_count = complexity_to_count.get(complexity, 2)
        return min(max(base_count, self.config.min_examples), self.config.max_examples)
    
    def build_dynamic_shot_prompt(
        self,
        task: ZeroShotTask,
        dream_text: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build a dynamic shot prompt with intelligently selected examples.
        
        Args:
            task: The analysis task
            dream_text: The dream description
            additional_context: Optional additional context
            
        Returns:
            Dictionary containing the complete dynamic shot prompt
        """
        # Analyze dream complexity and extract keywords
        complexity = self.analyze_dream_complexity(dream_text)
        keywords = self.extract_keywords(dream_text)
        
        # Select relevant examples
        selected_examples = self.select_dynamic_examples(
            task, dream_text, complexity, keywords, additional_context
        )
        
        # Build the prompt
        system_message = self._build_dynamic_system_message(task, complexity, selected_examples)
        user_message = self._build_dynamic_user_message(
            task, dream_text, selected_examples, additional_context
        )
        
        return {
            "system_message": system_message,
            "user_message": user_message,
            "task": task.value,
            "complexity": complexity.value,
            "num_examples": len(selected_examples),
            "selected_examples": [ex.dream_text[:100] + "..." for ex in selected_examples],
            "keywords": keywords[:10]  # Top 10 keywords for reference
        }
    
    def _build_dynamic_system_message(
        self,
        task: ZeroShotTask,
        complexity: DreamComplexity,
        examples: List[DreamExample]
    ) -> str:
        """Build system message for dynamic shot prompting."""
        base_instruction = f"You are an expert in {task.value.replace('_', ' ')}."
        
        complexity_guidance = {
            DreamComplexity.SIMPLE: "Focus on clear, direct analysis of the main emotional or symbolic content.",
            DreamComplexity.MODERATE: "Consider multiple elements and their interactions in your analysis.",
            DreamComplexity.COMPLEX: "Provide nuanced analysis considering multiple layers of meaning and emotion.",
            DreamComplexity.HIGHLY_COMPLEX: "Deliver comprehensive analysis addressing all narrative elements and their interconnections."
        }
        
        system_parts = [
            base_instruction,
            "",
            f"COMPLEXITY LEVEL: {complexity.value.upper()}",
            complexity_guidance[complexity],
            "",
            f"You have been provided with {len(examples)} relevant example(s) to guide your analysis.",
            "Use these examples to understand the expected analysis format and depth,",
            "but adapt your response to the specific content of the new dream.",
            "",
            "Provide thorough analysis in the specified JSON format.",
            "Ensure your confidence score reflects the clarity and completeness of the dream content."
        ]
        
        return "\n".join(system_parts)
    
    def _build_dynamic_user_message(
        self,
        task: ZeroShotTask,
        dream_text: str,
        examples: List[DreamExample],
        additional_context: Optional[str] = None
    ) -> str:
        """Build user message with dynamic examples."""
        user_parts = []
        
        # Add examples if available
        if examples:
            user_parts.extend([
                "Here are relevant examples to guide your analysis:",
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
            "Now analyze this new dream:",
            "",
            f"DREAM TO ANALYZE:",
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
            "Please provide your analysis in the same JSON format as the examples above.",
            "Analysis:"
        ])
        
        return "\n".join(user_parts)
    
    def add_example(self, task: ZeroShotTask, example: DreamExample) -> None:
        """Add a new example to the database."""
        if task not in self.example_database:
            self.example_database[task] = []
        
        self.example_database[task].append(example)
        logger.info(f"Added new example for task {task.value}")
    
    def get_example_statistics(self) -> Dict[str, Any]:
        """Get statistics about the example database."""
        stats = {
            "total_examples": sum(len(examples) for examples in self.example_database.values()),
            "examples_by_task": {task.value: len(examples) for task, examples in self.example_database.items()},
            "complexity_distribution": {},
            "example_types": {}
        }
        
        # Count complexity and type distributions
        for examples in self.example_database.values():
            for example in examples:
                complexity = example.complexity.value
                example_type = example.example_type.value
                
                stats["complexity_distribution"][complexity] = stats["complexity_distribution"].get(complexity, 0) + 1
                stats["example_types"][example_type] = stats["example_types"].get(example_type, 0) + 1
        
        return stats
    
    def calculate_example_relevance(self, example: DreamExample, dream_text: str, keywords: List[str]) -> float:
        """
        Calculate relevance score between an example and dream content.

        Public API method for calculating how relevant an example is to given dream content.

        Args:
            example: The dream example to score
            dream_text: The dream description to match against
            keywords: Extracted keywords from the dream

        Returns:
            Relevance score between 0.0 and 1.0
        """
        return example._calculate_relevance_score(dream_text, keywords)

    def _calculate_complexity_bonus(self, example_complexity: DreamComplexity, dream_complexity: DreamComplexity) -> float:
        """Calculate bonus score for complexity matching (private method)."""
        if example_complexity == dream_complexity:
            return 0.2
        elif abs(list(DreamComplexity).index(example_complexity) -
                list(DreamComplexity).index(dream_complexity)) == 1:
            return 0.1
        return 0.0

    def _calculate_usage_penalty(self, task: ZeroShotTask, example: DreamExample) -> float:
        """Calculate penalty for frequently used examples (private method)."""
        usage_key = f"{task.value}:{example.dream_text[:50]}"
        return self.usage_history.get(usage_key, 0) * 0.1

    def _update_usage_history(self, task: ZeroShotTask, selected_examples: List[DreamExample]) -> None:
        """Update usage history for selected examples (private method)."""
        for example in selected_examples:
            usage_key = f"{task.value}:{example.dream_text[:50]}"
            self.usage_history[usage_key] = self.usage_history.get(usage_key, 0) + 1

    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about example usage patterns.

        Returns:
            Dictionary containing usage statistics
        """
        if not self.usage_history:
            return {
                "total_usages": 0,
                "unique_examples": 0,
                "most_used_examples": [],
                "usage_distribution": {}
            }

        total_usages = sum(self.usage_history.values())
        unique_examples = len(self.usage_history)

        # Get most used examples
        sorted_usage = sorted(self.usage_history.items(), key=lambda x: x[1], reverse=True)
        most_used = [(key.split(':', 1)[1], count) for key, count in sorted_usage[:5]]

        # Usage distribution
        usage_counts = list(self.usage_history.values())
        usage_distribution = {
            "min_usage": min(usage_counts),
            "max_usage": max(usage_counts),
            "avg_usage": total_usages / unique_examples
        }

        return {
            "total_usages": total_usages,
            "unique_examples": unique_examples,
            "most_used_examples": most_used,
            "usage_distribution": usage_distribution
        }

    def clear_usage_history(self) -> None:
        """Clear the usage history for fresh example selection."""
        self.usage_history.clear()
        logger.info("Cleared example usage history")
