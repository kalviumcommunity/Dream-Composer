"""
Structured Prompt Builder for Dream Composer.

This module provides a framework for building structured prompts for AI/NLP APIs
to analyze dream descriptions and extract meaningful information.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class PromptType(Enum):
    """Types of prompts for different analysis tasks."""
    EMOTION_EXTRACTION = "emotion_extraction"
    MOOD_ANALYSIS = "mood_analysis"
    SYMBOL_INTERPRETATION = "symbol_interpretation"
    MUSIC_MAPPING = "music_mapping"
    COMPREHENSIVE_ANALYSIS = "comprehensive_analysis"


@dataclass
class PromptTemplate:
    """Template for structured prompts."""
    name: str
    prompt_type: PromptType
    system_message: str
    user_template: str
    expected_format: str
    examples: List[Dict[str, str]]
    parameters: Dict[str, Any]


class PromptBuilder:
    """
    Builder class for creating structured prompts for dream analysis.
    
    This class provides methods to build consistent, structured prompts
    for various AI/NLP tasks in the Dream Composer application.
    """
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[PromptType, PromptTemplate]:
        """Initialize predefined prompt templates."""
        return {
            PromptType.EMOTION_EXTRACTION: PromptTemplate(
                name="Emotion Extraction",
                prompt_type=PromptType.EMOTION_EXTRACTION,
                system_message=(
                    "You are an expert dream analyst and emotion recognition specialist. "
                    "Your task is to analyze dream descriptions and extract the primary "
                    "emotions and feelings expressed. Focus on identifying both explicit "
                    "and implicit emotional content."
                ),
                user_template=(
                    "Analyze the following dream description and extract the primary emotions:\n\n"
                    "Dream: \"{dream_text}\"\n\n"
                    "Please identify:\n"
                    "1. Primary emotions (2-3 main emotions)\n"
                    "2. Emotional intensity (scale 1-10)\n"
                    "3. Emotional progression (if emotions change during the dream)\n"
                    "4. Overall mood classification\n\n"
                    "Format your response as JSON with the specified structure."
                ),
                expected_format="""{
    "primary_emotions": ["emotion1", "emotion2", "emotion3"],
    "emotional_intensity": {"emotion1": 8, "emotion2": 6, "emotion3": 7},
    "emotional_progression": "description of how emotions change",
    "overall_mood": "mood_classification",
    "confidence_score": 0.85
}""",
                examples=[
                    {
                        "dream": "I was flying over a beautiful city at sunset, feeling free and peaceful.",
                        "response": '{"primary_emotions": ["joy", "peace", "freedom"], "emotional_intensity": {"joy": 8, "peace": 9, "freedom": 8}, "emotional_progression": "consistent positive emotions throughout", "overall_mood": "euphoric", "confidence_score": 0.9}'
                    }
                ],
                parameters={"max_emotions": 3, "intensity_scale": 10}
            ),
            
            PromptType.MUSIC_MAPPING: PromptTemplate(
                name="Music Parameter Mapping",
                prompt_type=PromptType.MUSIC_MAPPING,
                system_message=(
                    "You are a music theory expert and composer specializing in emotional "
                    "music composition. Your task is to map emotions and dream content to "
                    "specific musical parameters that will create appropriate musical compositions."
                ),
                user_template=(
                    "Based on the following dream analysis, suggest musical parameters:\n\n"
                    "Dream: \"{dream_text}\"\n"
                    "Emotions: {emotions}\n"
                    "Mood: {mood}\n"
                    "Intensity: {intensity}\n\n"
                    "Please suggest:\n"
                    "1. Tempo (BPM)\n"
                    "2. Key signature\n"
                    "3. Time signature\n"
                    "4. Instruments/timbres\n"
                    "5. Musical style/genre\n"
                    "6. Dynamics and expression\n"
                    "7. Harmonic progression suggestions\n\n"
                    "Format your response as JSON."
                ),
                expected_format="""{
    "tempo": {"bpm": 95, "description": "moderately slow, flowing"},
    "key": {"signature": "C major", "mode": "major", "reasoning": "uplifting and peaceful"},
    "time_signature": "4/4",
    "instruments": ["piano", "strings", "soft_pad"],
    "style": "ambient_classical",
    "dynamics": {"overall": "mp", "variations": ["p", "mf"]},
    "harmonic_progression": ["I", "vi", "IV", "V"],
    "expression_marks": ["legato", "dolce"],
    "confidence_score": 0.88
}""",
                examples=[
                    {
                        "input": "Flying dream with joy and freedom",
                        "response": '{"tempo": {"bpm": 120, "description": "uplifting and energetic"}, "key": {"signature": "D major", "mode": "major"}, "instruments": ["piano", "strings", "flute"]}'
                    }
                ],
                parameters={"tempo_range": [60, 180], "preferred_keys": ["C", "G", "D", "A", "F"]}
            ),
            
            PromptType.COMPREHENSIVE_ANALYSIS: PromptTemplate(
                name="Comprehensive Dream Analysis",
                prompt_type=PromptType.COMPREHENSIVE_ANALYSIS,
                system_message=(
                    "You are a comprehensive dream analyst with expertise in psychology, "
                    "emotion recognition, symbolism, and music therapy. Provide a complete "
                    "analysis of dream content for musical composition purposes."
                ),
                user_template=(
                    "Provide a comprehensive analysis of this dream for music composition:\n\n"
                    "Dream: \"{dream_text}\"\n\n"
                    "Please analyze:\n"
                    "1. Emotional content and progression\n"
                    "2. Symbolic elements and their meanings\n"
                    "3. Narrative structure and pacing\n"
                    "4. Sensory details (colors, sounds, textures)\n"
                    "5. Overall atmosphere and mood\n"
                    "6. Musical composition recommendations\n\n"
                    "Format as structured JSON."
                ),
                expected_format="""{
    "emotions": {"primary": [], "secondary": [], "progression": ""},
    "symbols": [{"element": "", "meaning": "", "musical_relevance": ""}],
    "narrative": {"structure": "", "pacing": "", "climax": ""},
    "sensory": {"colors": [], "sounds": [], "textures": [], "atmosphere": ""},
    "mood": {"overall": "", "intensity": 0, "stability": ""},
    "music_recommendations": {
        "tempo": 0, "key": "", "instruments": [], "style": "",
        "dynamics": "", "structure": ""
    },
    "confidence_score": 0.0
}""",
                examples=[],
                parameters={"analysis_depth": "comprehensive", "focus": "musical_composition"}
            )
        }
    
    def build_prompt(
        self, 
        prompt_type: PromptType, 
        dream_text: str, 
        **kwargs
    ) -> Dict[str, str]:
        """
        Build a structured prompt for the specified analysis type.
        
        Args:
            prompt_type: Type of analysis prompt to build
            dream_text: The dream description to analyze
            **kwargs: Additional parameters for prompt customization
            
        Returns:
            Dictionary containing system_message and user_message
            
        Raises:
            ValueError: If prompt_type is not supported
        """
        if prompt_type not in self.templates:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")
        
        template = self.templates[prompt_type]
        
        # Format the user message with provided parameters
        format_params = {"dream_text": dream_text, **kwargs}
        user_message = template.user_template.format(**format_params)
        
        return {
            "system_message": template.system_message,
            "user_message": user_message,
            "expected_format": template.expected_format,
            "examples": template.examples
        }
    
    def get_template(self, prompt_type: PromptType) -> PromptTemplate:
        """Get the template for a specific prompt type."""
        if prompt_type not in self.templates:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")
        return self.templates[prompt_type]
    
    def add_custom_template(self, template: PromptTemplate) -> None:
        """Add a custom prompt template."""
        self.templates[template.prompt_type] = template
    
    def list_available_types(self) -> List[PromptType]:
        """List all available prompt types."""
        return list(self.templates.keys())
    
    def validate_prompt_structure(self, prompt_type: PromptType, response: str) -> bool:
        """
        Validate if a response matches the expected format for a prompt type.
        
        Args:
            prompt_type: The type of prompt
            response: The response to validate
            
        Returns:
            True if response structure is valid, False otherwise
        """
        try:
            import json
            template = self.templates[prompt_type]
            
            # Try to parse as JSON
            parsed_response = json.loads(response)
            
            # Basic validation - check if it's a dictionary
            if not isinstance(parsed_response, dict):
                return False
            
            # Additional validation could be added here based on expected_format
            return True
            
        except (json.JSONDecodeError, KeyError):
            return False

    def get_prompt_statistics(self) -> Dict[str, Any]:
        """Get statistics about available prompt templates."""
        return {
            "total_templates": len(self.templates),
            "template_types": [pt.value for pt in self.templates.keys()],
            "templates_with_examples": sum(1 for t in self.templates.values() if t.examples),
            "average_system_message_length": sum(len(t.system_message) for t in self.templates.values()) // len(self.templates)
        }
