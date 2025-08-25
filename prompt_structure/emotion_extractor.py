"""
Emotion Extractor for Dream Composer.

This module provides functionality to extract emotions and emotional patterns
from dream descriptions using structured prompts and NLP analysis.
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .prompt_builder import PromptBuilder, PromptType


class EmotionCategory(Enum):
    """Categories of emotions for classification."""
    JOY = "joy"
    SADNESS = "sadness"
    FEAR = "fear"
    ANGER = "anger"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    LOVE = "love"
    PEACE = "peace"
    EXCITEMENT = "excitement"
    NOSTALGIA = "nostalgia"
    ANXIETY = "anxiety"
    WONDER = "wonder"
    FREEDOM = "freedom"
    MELANCHOLY = "melancholy"
    EUPHORIA = "euphoria"


@dataclass
class EmotionResult:
    """Result of emotion extraction analysis."""
    primary_emotions: List[str]
    emotional_intensity: Dict[str, float]
    emotional_progression: str
    overall_mood: str
    confidence_score: float
    raw_response: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "primary_emotions": self.primary_emotions,
            "emotional_intensity": self.emotional_intensity,
            "emotional_progression": self.emotional_progression,
            "overall_mood": self.overall_mood,
            "confidence_score": self.confidence_score
        }


class EmotionExtractor:
    """
    Extracts emotions and emotional patterns from dream descriptions.
    
    This class uses structured prompts to analyze dream text and extract
    emotional content that can be mapped to musical parameters.
    """
    
    def __init__(self):
        self.prompt_builder = PromptBuilder()
        self.emotion_keywords = self._initialize_emotion_keywords()
        self.mood_mappings = self._initialize_mood_mappings()
    
    def _initialize_emotion_keywords(self) -> Dict[str, List[str]]:
        """Initialize emotion keyword mappings for fallback analysis."""
        return {
            "joy": ["happy", "joyful", "elated", "cheerful", "delighted", "blissful", "ecstatic"],
            "sadness": ["sad", "melancholy", "melancholic", "sorrowful", "gloomy", "depressed", "mournful"],
            "fear": ["afraid", "scared", "terrified", "anxious", "worried", "nervous", "frightened"],
            "anger": ["angry", "furious", "mad", "irritated", "enraged", "hostile", "frustrated"],
            "peace": ["peaceful", "calm", "serene", "tranquil", "relaxed", "content", "harmonious"],
            "excitement": ["excited", "thrilled", "energetic", "enthusiastic", "exhilarated"],
            "love": ["loving", "affectionate", "tender", "caring", "warm", "compassionate"],
            "wonder": ["amazed", "awed", "curious", "fascinated", "intrigued", "mystified"],
            "freedom": ["free", "liberated", "unbound", "independent", "unrestricted"],
            "nostalgia": ["nostalgic", "wistful", "reminiscent", "sentimental", "longing"]
        }
    
    def _initialize_mood_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mood to musical parameter mappings."""
        return {
            "euphoric": {"tempo_range": [120, 140], "key_preference": "major", "energy": "high"},
            "peaceful": {"tempo_range": [60, 90], "key_preference": "major", "energy": "low"},
            "melancholic": {"tempo_range": [70, 100], "key_preference": "minor", "energy": "medium"},
            "anxious": {"tempo_range": [100, 130], "key_preference": "minor", "energy": "high"},
            "nostalgic": {"tempo_range": [80, 110], "key_preference": "major", "energy": "medium"},
            "mysterious": {"tempo_range": [90, 120], "key_preference": "minor", "energy": "medium"},
            "energetic": {"tempo_range": [130, 160], "key_preference": "major", "energy": "high"},
            "contemplative": {"tempo_range": [60, 80], "key_preference": "minor", "energy": "low"}
        }
    
    def extract_emotions(self, dream_text: str) -> EmotionResult:
        """
        Extract emotions from dream description.
        
        Args:
            dream_text: The dream description to analyze
            
        Returns:
            EmotionResult containing extracted emotional information
        """
        # Build structured prompt
        prompt_data = self.prompt_builder.build_prompt(
            PromptType.EMOTION_EXTRACTION,
            dream_text
        )
        
        # For now, simulate AI response (in real implementation, this would call an AI API)
        simulated_response = self._simulate_ai_response(dream_text)
        
        # Parse the response
        try:
            parsed_result = self._parse_emotion_response(simulated_response)
            return EmotionResult(
                primary_emotions=parsed_result.get("primary_emotions", []),
                emotional_intensity=parsed_result.get("emotional_intensity", {}),
                emotional_progression=parsed_result.get("emotional_progression", ""),
                overall_mood=parsed_result.get("overall_mood", "neutral"),
                confidence_score=parsed_result.get("confidence_score", 0.5),
                raw_response=simulated_response
            )
        except Exception as e:
            # Fallback to keyword-based analysis
            return self._fallback_emotion_analysis(dream_text)
    
    def _simulate_ai_response(self, dream_text: str) -> str:
        """
        Simulate AI response for testing purposes.
        In production, this would be replaced with actual AI API calls.
        """
        # Simple keyword-based simulation
        detected_emotions = []
        intensities = {}
        
        text_lower = dream_text.lower()
        
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_emotions.append(emotion)
                    # Simulate intensity based on context
                    intensities[emotion] = min(8.0, len([k for k in keywords if k in text_lower]) * 2.5)
                    break
        
        # Limit to top 3 emotions
        detected_emotions = detected_emotions[:3]
        
        # Determine overall mood
        if "joy" in detected_emotions or "peace" in detected_emotions:
            overall_mood = "euphoric" if "joy" in detected_emotions else "peaceful"
        elif "sadness" in detected_emotions or "fear" in detected_emotions:
            overall_mood = "melancholic" if "sadness" in detected_emotions else "anxious"
        else:
            overall_mood = "contemplative"
        
        # Create simulated JSON response
        response = {
            "primary_emotions": detected_emotions or ["neutral"],
            "emotional_intensity": intensities or {"neutral": 5.0},
            "emotional_progression": "consistent emotional tone throughout the dream",
            "overall_mood": overall_mood,
            "confidence_score": 0.75 if detected_emotions else 0.4
        }
        
        return json.dumps(response, indent=2)
    
    def _parse_emotion_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into structured format."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response if it's embedded in text
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError("Could not parse emotion response")
    
    def _fallback_emotion_analysis(self, dream_text: str) -> EmotionResult:
        """Fallback emotion analysis using keyword matching."""
        text_lower = dream_text.lower()
        detected_emotions = []
        intensities = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                detected_emotions.append(emotion)
                intensities[emotion] = min(10.0, matches * 3.0)
        
        # Default values if no emotions detected
        if not detected_emotions:
            detected_emotions = ["neutral"]
            intensities = {"neutral": 5.0}
        
        overall_mood = self._determine_overall_mood(detected_emotions)
        
        return EmotionResult(
            primary_emotions=detected_emotions[:3],
            emotional_intensity=intensities,
            emotional_progression="analyzed using keyword matching",
            overall_mood=overall_mood,
            confidence_score=0.3,  # Lower confidence for fallback method
            raw_response="fallback_analysis"
        )
    
    def _determine_overall_mood(self, emotions: List[str]) -> str:
        """Determine overall mood from list of emotions."""
        if not emotions:
            return "neutral"
        
        # Mood priority mapping
        mood_priorities = {
            "euphoric": ["joy", "excitement", "love"],
            "peaceful": ["peace", "calm"],
            "melancholic": ["sadness", "nostalgia"],
            "anxious": ["fear", "anxiety"],
            "energetic": ["excitement", "freedom"],
            "contemplative": ["wonder", "curiosity"]
        }
        
        for mood, mood_emotions in mood_priorities.items():
            if any(emotion in emotions for emotion in mood_emotions):
                return mood
        
        return "contemplative"
    
    def get_emotion_statistics(self, emotion_results: List[EmotionResult]) -> Dict[str, Any]:
        """Get statistics from multiple emotion analysis results."""
        if not emotion_results:
            return {}
        
        all_emotions = []
        all_moods = []
        confidence_scores = []
        
        for result in emotion_results:
            all_emotions.extend(result.primary_emotions)
            all_moods.append(result.overall_mood)
            confidence_scores.append(result.confidence_score)
        
        # Count frequencies
        emotion_counts = {}
        for emotion in all_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        mood_counts = {}
        for mood in all_moods:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        
        return {
            "total_analyses": len(emotion_results),
            "most_common_emotions": sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "most_common_moods": sorted(mood_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            "average_confidence": sum(confidence_scores) / len(confidence_scores),
            "unique_emotions_detected": len(emotion_counts),
            "unique_moods_detected": len(mood_counts)
        }
    
    def validate_emotion_result(self, result: EmotionResult) -> bool:
        """Validate an emotion extraction result."""
        if not result.primary_emotions:
            return False
        
        if result.confidence_score < 0 or result.confidence_score > 1:
            return False
        
        # Check if intensities are in valid range
        for intensity in result.emotional_intensity.values():
            if intensity < 0 or intensity > 10:
                return False
        
        return True
