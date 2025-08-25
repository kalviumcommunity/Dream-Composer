"""
Zero-Shot Prompting for Dream Composer.

This module implements zero-shot prompting techniques that allow AI models
to analyze dreams without requiring specific examples or training data.
Zero-shot prompts rely on the model's pre-trained knowledge and clear instructions.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

from .prompt_builder import PromptType


class ZeroShotTask(Enum):
    """Types of zero-shot tasks for dream analysis."""
    DREAM_EMOTION_ANALYSIS = "dream_emotion_analysis"
    MUSICAL_STYLE_RECOMMENDATION = "musical_style_recommendation"
    DREAM_SYMBOLISM_INTERPRETATION = "dream_symbolism_interpretation"
    MOOD_TO_MUSIC_MAPPING = "mood_to_music_mapping"
    DREAM_NARRATIVE_ANALYSIS = "dream_narrative_analysis"
    SENSORY_EXPERIENCE_EXTRACTION = "sensory_experience_extraction"
    PSYCHOLOGICAL_THEME_IDENTIFICATION = "psychological_theme_identification"


@dataclass
class ZeroShotPrompt:
    """Zero-shot prompt structure."""
    task: ZeroShotTask
    instruction: str
    context: str
    output_format: str
    constraints: List[str]
    reasoning_steps: List[str]


class ZeroShotPromptBuilder:
    """
    Builder for zero-shot prompts that require no training examples.
    
    Zero-shot prompting leverages the AI model's pre-trained knowledge
    to perform tasks by providing clear instructions and context.
    """
    
    def __init__(self):
        self.prompts = self._initialize_zero_shot_prompts()
    
    def _initialize_zero_shot_prompts(self) -> Dict[ZeroShotTask, ZeroShotPrompt]:
        """Initialize zero-shot prompt templates."""
        return {
            ZeroShotTask.DREAM_EMOTION_ANALYSIS: ZeroShotPrompt(
                task=ZeroShotTask.DREAM_EMOTION_ANALYSIS,
                instruction=(
                    "Analyze the emotional content of the following dream description. "
                    "Identify the primary emotions, their intensity, and how they evolve "
                    "throughout the dream narrative."
                ),
                context=(
                    "Dreams often contain complex emotional landscapes that reflect "
                    "the dreamer's subconscious feelings, fears, desires, and experiences. "
                    "Emotions in dreams can be explicit (directly stated) or implicit "
                    "(conveyed through imagery, actions, or atmosphere)."
                ),
                output_format="""{
    "primary_emotions": ["emotion1", "emotion2", "emotion3"],
    "emotion_intensities": {"emotion1": 8, "emotion2": 6, "emotion3": 7},
    "emotional_progression": "description of how emotions change",
    "dominant_mood": "overall emotional tone",
    "emotional_triggers": ["trigger1", "trigger2"],
    "confidence": 0.85
}""",
                constraints=[
                    "Limit to maximum 5 primary emotions",
                    "Intensity scale: 1-10 (1=barely present, 10=overwhelming)",
                    "Focus on emotions actually present in the dream text",
                    "Consider both explicit and implicit emotional content"
                ],
                reasoning_steps=[
                    "Read the dream description carefully",
                    "Identify explicit emotional words and phrases",
                    "Analyze implicit emotions from imagery and actions",
                    "Assess the intensity of each emotion",
                    "Determine the overall emotional progression"
                ]
            ),
            
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION: ZeroShotPrompt(
                task=ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
                instruction=(
                    "Based on the dream's emotional content and imagery, recommend "
                    "appropriate musical styles, instruments, tempo, and key signatures "
                    "that would best represent this dream as music."
                ),
                context=(
                    "Music has the power to evoke and represent emotions, atmospheres, "
                    "and narratives. Different musical elements (tempo, key, instruments, "
                    "dynamics) can effectively convey the emotional and atmospheric "
                    "qualities found in dreams."
                ),
                output_format="""{
    "recommended_style": "musical_genre_or_style",
    "tempo": {"bpm": 120, "description": "moderate and flowing"},
    "key_signature": {"key": "C", "mode": "major", "reasoning": "uplifting mood"},
    "instruments": ["piano", "strings", "flute"],
    "dynamics": "mp to mf with gentle crescendos",
    "musical_structure": "ABA form with development",
    "special_techniques": ["legato phrasing", "rubato"],
    "confidence": 0.88
}""",
                constraints=[
                    "Choose instruments that match the dream's atmosphere",
                    "Tempo should reflect the dream's pacing and energy",
                    "Key signature should align with emotional content",
                    "Consider both Western and non-Western musical traditions"
                ],
                reasoning_steps=[
                    "Analyze the dream's emotional landscape",
                    "Consider the dream's pacing and energy level",
                    "Match atmospheric qualities to musical timbres",
                    "Select appropriate harmonic language",
                    "Consider cultural and stylistic associations"
                ]
            ),
            
            ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION: ZeroShotPrompt(
                task=ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION,
                instruction=(
                    "Identify and interpret symbolic elements in the dream. "
                    "Explain their potential psychological meanings and how they "
                    "might relate to the dreamer's emotional or psychological state."
                ),
                context=(
                    "Dream symbols often represent deeper psychological meanings, "
                    "archetypal patterns, or personal associations. Common symbols "
                    "include water (emotions/subconscious), flying (freedom/transcendence), "
                    "animals (instincts), and various objects or scenarios that carry "
                    "metaphorical significance."
                ),
                output_format="""{
    "symbols": [
        {
            "element": "flying",
            "interpretation": "desire for freedom and transcendence",
            "psychological_meaning": "escape from limitations",
            "emotional_significance": "liberation and empowerment",
            "musical_relevance": "ascending melodies, light textures"
        }
    ],
    "overall_symbolic_theme": "transformation and growth",
    "archetypal_patterns": ["hero's journey", "transformation"],
    "confidence": 0.82
}""",
                constraints=[
                    "Focus on symbols actually present in the dream",
                    "Provide balanced, non-prescriptive interpretations",
                    "Consider universal and personal symbolic meanings",
                    "Connect symbols to potential musical representations"
                ],
                reasoning_steps=[
                    "Identify concrete objects, actions, and scenarios",
                    "Research common symbolic meanings",
                    "Consider personal and cultural contexts",
                    "Analyze relationships between symbols",
                    "Connect to musical representation possibilities"
                ]
            ),
            
            ZeroShotTask.MOOD_TO_MUSIC_MAPPING: ZeroShotPrompt(
                task=ZeroShotTask.MOOD_TO_MUSIC_MAPPING,
                instruction=(
                    "Create a detailed mapping from the dream's mood and atmosphere "
                    "to specific musical parameters that could effectively represent "
                    "these qualities in a musical composition."
                ),
                context=(
                    "Mood-to-music mapping is based on established relationships "
                    "between emotional states and musical elements. For example, "
                    "major keys often convey happiness, minor keys suggest melancholy, "
                    "fast tempos create excitement, and certain instruments evoke "
                    "specific atmospheres."
                ),
                output_format="""{
    "mood_analysis": {
        "primary_mood": "peaceful",
        "secondary_moods": ["nostalgic", "contemplative"],
        "atmosphere": "serene and introspective"
    },
    "musical_mapping": {
        "tempo_range": [60, 80],
        "key_preferences": ["F major", "C major", "A minor"],
        "instrument_palette": ["piano", "strings", "harp"],
        "rhythmic_character": "flowing and gentle",
        "harmonic_language": "consonant with occasional color tones",
        "dynamic_range": "soft to medium-soft",
        "articulation": "legato and expressive"
    },
    "composition_suggestions": {
        "form": "through-composed or ABA",
        "melodic_contour": "gentle rises and falls",
        "texture": "homophonic with occasional polyphony"
    },
    "confidence": 0.91
}""",
                constraints=[
                    "Base mappings on established music theory principles",
                    "Consider cross-cultural musical associations",
                    "Provide specific, actionable musical parameters",
                    "Ensure coherence between all musical elements"
                ],
                reasoning_steps=[
                    "Identify the dream's primary and secondary moods",
                    "Map moods to established musical associations",
                    "Consider tempo, key, and instrumental timbres",
                    "Ensure all elements work together cohesively",
                    "Provide specific composition guidance"
                ]
            ),
            
            ZeroShotTask.DREAM_NARRATIVE_ANALYSIS: ZeroShotPrompt(
                task=ZeroShotTask.DREAM_NARRATIVE_ANALYSIS,
                instruction=(
                    "Analyze the narrative structure of the dream, including its "
                    "pacing, dramatic arc, key events, and how these elements "
                    "could be represented in musical form."
                ),
                context=(
                    "Dreams often follow narrative patterns similar to stories, "
                    "with beginnings, developments, climaxes, and resolutions. "
                    "Understanding this structure helps in creating musical "
                    "compositions that mirror the dream's dramatic flow."
                ),
                output_format="""{
    "narrative_structure": {
        "opening": "description of dream beginning",
        "development": "how the dream unfolds",
        "climax": "peak moment or turning point",
        "resolution": "how the dream concludes"
    },
    "pacing": "gradual build with sudden shifts",
    "dramatic_arc": "rising action to climax",
    "key_events": ["event1", "event2", "event3"],
    "musical_form_suggestion": "sonata form with development",
    "timing_structure": {
        "introduction": "0-20%",
        "development": "20-70%",
        "climax": "70-85%",
        "resolution": "85-100%"
    },
    "confidence": 0.87
}""",
                constraints=[
                    "Identify clear narrative elements in the dream",
                    "Map narrative structure to musical forms",
                    "Consider pacing and dramatic timing",
                    "Suggest appropriate musical development techniques"
                ],
                reasoning_steps=[
                    "Identify the dream's beginning, middle, and end",
                    "Analyze the dramatic progression",
                    "Map narrative elements to musical forms",
                    "Consider timing and pacing for musical structure",
                    "Suggest specific compositional approaches"
                ]
            )
        }
    
    def build_zero_shot_prompt(
        self, 
        task: ZeroShotTask, 
        dream_text: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Build a zero-shot prompt for the specified task.
        
        Args:
            task: The zero-shot task to perform
            dream_text: The dream description to analyze
            additional_context: Optional additional context
            
        Returns:
            Dictionary containing the complete prompt structure
        """
        if task not in self.prompts:
            raise ValueError(f"Unsupported zero-shot task: {task}")
        
        prompt_template = self.prompts[task]
        
        # Build the complete prompt
        system_message = self._build_system_message(prompt_template)
        user_message = self._build_user_message(prompt_template, dream_text, additional_context)
        
        return {
            "system_message": system_message,
            "user_message": user_message,
            "expected_format": prompt_template.output_format,
            "task": task.value,
            "reasoning_steps": prompt_template.reasoning_steps
        }
    
    def _build_system_message(self, prompt_template: ZeroShotPrompt) -> str:
        """Build the system message for zero-shot prompting."""
        system_parts = [
            f"You are an expert in {prompt_template.task.value.replace('_', ' ')}.",
            "",
            "CONTEXT:",
            prompt_template.context,
            "",
            "CONSTRAINTS:",
        ]
        
        for constraint in prompt_template.constraints:
            system_parts.append(f"- {constraint}")
        
        system_parts.extend([
            "",
            "REASONING APPROACH:",
        ])
        
        for i, step in enumerate(prompt_template.reasoning_steps, 1):
            system_parts.append(f"{i}. {step}")
        
        system_parts.extend([
            "",
            "Provide your analysis in the specified JSON format.",
            "Be thorough but concise in your analysis.",
            "Base your conclusions on the actual content of the dream."
        ])
        
        return "\n".join(system_parts)
    
    def _build_user_message(
        self, 
        prompt_template: ZeroShotPrompt, 
        dream_text: str,
        additional_context: Optional[str] = None
    ) -> str:
        """Build the user message for zero-shot prompting."""
        user_parts = [
            prompt_template.instruction,
            "",
            f"DREAM DESCRIPTION:",
            f'"{dream_text}"',
            ""
        ]
        
        if additional_context:
            user_parts.extend([
                "ADDITIONAL CONTEXT:",
                additional_context,
                ""
            ])
        
        user_parts.extend([
            "Please provide your analysis in the following JSON format:",
            "",
            prompt_template.output_format,
            "",
            "Analysis:"
        ])
        
        return "\n".join(user_parts)
    
    def get_available_tasks(self) -> List[ZeroShotTask]:
        """Get list of available zero-shot tasks."""
        return list(self.prompts.keys())
    
    def get_task_description(self, task: ZeroShotTask) -> str:
        """Get description of a specific zero-shot task."""
        if task not in self.prompts:
            raise ValueError(f"Unknown task: {task}")
        return self.prompts[task].instruction
    
    def validate_response_format(self, task: ZeroShotTask, response: str) -> bool:
        """
        Validate if a response matches the expected format for a task.
        
        Args:
            task: The zero-shot task
            response: The AI response to validate
            
        Returns:
            True if response format is valid, False otherwise
        """
        try:
            parsed_response = json.loads(response)
            
            if not isinstance(parsed_response, dict):
                return False
            
            # Basic validation - check for confidence score
            if "confidence" not in parsed_response:
                return False
            
            confidence = parsed_response["confidence"]
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                return False
            
            return True
            
        except (json.JSONDecodeError, KeyError, TypeError):
            return False
