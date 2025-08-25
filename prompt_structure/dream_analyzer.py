"""
Dream Analyzer for Dream Composer.

This module provides comprehensive analysis of dream descriptions,
combining emotion extraction, symbolism interpretation, and musical mapping.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .prompt_builder import PromptBuilder, PromptType
from .emotion_extractor import EmotionExtractor, EmotionResult
from .music_mapper import MusicMapper, MusicalParameters


@dataclass
class SymbolInterpretation:
    """Interpretation of symbolic elements in dreams."""
    element: str
    meaning: str
    musical_relevance: str
    emotional_weight: float


@dataclass
class NarrativeStructure:
    """Structure and pacing of dream narrative."""
    structure: str
    pacing: str
    climax: str
    transitions: List[str]


@dataclass
class SensoryDetails:
    """Sensory information from dream description."""
    colors: List[str]
    sounds: List[str]
    textures: List[str]
    atmosphere: str
    visual_intensity: float


@dataclass
class ComprehensiveAnalysis:
    """Complete analysis of a dream description."""
    dream_text: str
    emotion_result: EmotionResult
    musical_parameters: MusicalParameters
    symbols: List[SymbolInterpretation]
    narrative: NarrativeStructure
    sensory: SensoryDetails
    overall_confidence: float
    analysis_timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "dream_text": self.dream_text,
            "emotions": self.emotion_result.to_dict(),
            "musical_parameters": self.musical_parameters.to_dict(),
            "symbols": [
                {
                    "element": s.element,
                    "meaning": s.meaning,
                    "musical_relevance": s.musical_relevance,
                    "emotional_weight": s.emotional_weight
                } for s in self.symbols
            ],
            "narrative": {
                "structure": self.narrative.structure,
                "pacing": self.narrative.pacing,
                "climax": self.narrative.climax,
                "transitions": self.narrative.transitions
            },
            "sensory": {
                "colors": self.sensory.colors,
                "sounds": self.sensory.sounds,
                "textures": self.sensory.textures,
                "atmosphere": self.sensory.atmosphere,
                "visual_intensity": self.sensory.visual_intensity
            },
            "overall_confidence": self.overall_confidence,
            "analysis_timestamp": self.analysis_timestamp
        }


class DreamAnalyzer:
    """
    Comprehensive dream analysis system.
    
    This class orchestrates the complete analysis of dream descriptions,
    combining emotion extraction, symbolism interpretation, and musical mapping
    to provide comprehensive insights for music composition.
    """
    
    def __init__(self):
        self.prompt_builder = PromptBuilder()
        self.emotion_extractor = EmotionExtractor()
        self.music_mapper = MusicMapper()
        self.symbol_database = self._initialize_symbol_database()
        self.color_mappings = self._initialize_color_mappings()
    
    def _initialize_symbol_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize database of dream symbols and their interpretations."""
        return {
            "flying": {
                "meaning": "freedom, liberation, transcendence",
                "musical_relevance": "ascending melodies, light instrumentation",
                "emotional_associations": ["freedom", "joy", "peace"],
                "tempo_influence": 1.1
            },
            "water": {
                "meaning": "emotions, subconscious, flow of life",
                "musical_relevance": "flowing rhythms, fluid dynamics",
                "emotional_associations": ["peace", "mystery", "cleansing"],
                "tempo_influence": 0.9
            },
            "falling": {
                "meaning": "loss of control, anxiety, transition",
                "musical_relevance": "descending passages, tension",
                "emotional_associations": ["fear", "anxiety", "uncertainty"],
                "tempo_influence": 1.2
            },
            "animals": {
                "meaning": "instincts, natural wisdom, companionship",
                "musical_relevance": "organic rhythms, natural sounds",
                "emotional_associations": ["wonder", "connection", "wildness"],
                "tempo_influence": 1.0
            },
            "darkness": {
                "meaning": "unknown, mystery, hidden aspects",
                "musical_relevance": "minor keys, low registers",
                "emotional_associations": ["mystery", "fear", "introspection"],
                "tempo_influence": 0.8
            },
            "light": {
                "meaning": "clarity, hope, divine presence",
                "musical_relevance": "bright timbres, major keys",
                "emotional_associations": ["hope", "joy", "enlightenment"],
                "tempo_influence": 1.1
            },
            "city": {
                "meaning": "civilization, complexity, social connections",
                "musical_relevance": "complex harmonies, urban rhythms",
                "emotional_associations": ["excitement", "overwhelm", "connection"],
                "tempo_influence": 1.15
            },
            "nature": {
                "meaning": "natural state, growth, harmony",
                "musical_relevance": "organic forms, natural rhythms",
                "emotional_associations": ["peace", "growth", "harmony"],
                "tempo_influence": 0.95
            }
        }
    
    def _initialize_color_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize color to musical parameter mappings."""
        return {
            "blue": {"mood": "calm", "key_preference": "major", "instruments": ["flute", "strings"]},
            "red": {"mood": "passionate", "key_preference": "major", "instruments": ["brass", "strings"]},
            "green": {"mood": "natural", "key_preference": "major", "instruments": ["woodwinds", "harp"]},
            "yellow": {"mood": "bright", "key_preference": "major", "instruments": ["brass", "piano"]},
            "purple": {"mood": "mysterious", "key_preference": "minor", "instruments": ["strings", "pad"]},
            "black": {"mood": "dark", "key_preference": "minor", "instruments": ["low_strings", "organ"]},
            "white": {"mood": "pure", "key_preference": "major", "instruments": ["piano", "choir"]},
            "gold": {"mood": "majestic", "key_preference": "major", "instruments": ["brass", "strings"]},
            "silver": {"mood": "ethereal", "key_preference": "major", "instruments": ["bells", "harp"]},
            "orange": {"mood": "energetic", "key_preference": "major", "instruments": ["brass", "percussion"]}
        }
    
    def analyze_dream(self, dream_text: str) -> ComprehensiveAnalysis:
        """
        Perform comprehensive analysis of a dream description.
        
        Args:
            dream_text: The dream description to analyze
            
        Returns:
            ComprehensiveAnalysis containing all extracted information
        """
        import datetime
        
        # Extract emotions
        emotion_result = self.emotion_extractor.extract_emotions(dream_text)
        
        # Map to musical parameters
        musical_parameters = self.music_mapper.map_emotions_to_music(emotion_result, dream_text)
        
        # Analyze symbols
        symbols = self._analyze_symbols(dream_text)
        
        # Analyze narrative structure
        narrative = self._analyze_narrative(dream_text)
        
        # Extract sensory details
        sensory = self._extract_sensory_details(dream_text)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            emotion_result, musical_parameters, symbols
        )
        
        return ComprehensiveAnalysis(
            dream_text=dream_text,
            emotion_result=emotion_result,
            musical_parameters=musical_parameters,
            symbols=symbols,
            narrative=narrative,
            sensory=sensory,
            overall_confidence=overall_confidence,
            analysis_timestamp=datetime.datetime.now().isoformat()
        )
    
    def _analyze_symbols(self, dream_text: str) -> List[SymbolInterpretation]:
        """Analyze symbolic elements in the dream."""
        symbols = []
        text_lower = dream_text.lower()
        
        for symbol, data in self.symbol_database.items():
            if symbol in text_lower:
                # Calculate emotional weight based on context
                emotional_weight = self._calculate_symbol_weight(symbol, text_lower)
                
                symbols.append(SymbolInterpretation(
                    element=symbol,
                    meaning=data["meaning"],
                    musical_relevance=data["musical_relevance"],
                    emotional_weight=emotional_weight
                ))
        
        return symbols
    
    def _calculate_symbol_weight(self, symbol: str, text: str) -> float:
        """Calculate the emotional weight of a symbol in context."""
        # Count occurrences
        occurrences = text.count(symbol)
        
        # Check for intensifying words nearby
        intensifiers = ["very", "extremely", "incredibly", "massive", "huge", "beautiful", "terrifying"]
        context_boost = 0
        
        for intensifier in intensifiers:
            if intensifier in text:
                context_boost += 0.2
        
        # Base weight + occurrence multiplier + context boost
        weight = min(1.0, 0.5 + (occurrences * 0.2) + context_boost)
        return weight
    
    def _analyze_narrative(self, dream_text: str) -> NarrativeStructure:
        """Analyze the narrative structure of the dream."""
        # Simple heuristic analysis
        sentences = dream_text.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Determine structure
        if sentence_count <= 2:
            structure = "simple"
        elif sentence_count <= 5:
            structure = "moderate"
        else:
            structure = "complex"
        
        # Analyze pacing based on sentence length and conjunctions
        avg_sentence_length = len(dream_text) / max(sentence_count, 1)
        
        if avg_sentence_length < 50:
            pacing = "fast"
        elif avg_sentence_length < 100:
            pacing = "moderate"
        else:
            pacing = "slow"
        
        # Find potential climax (often in middle or end)
        climax_indicators = ["suddenly", "then", "finally", "climax", "peak", "moment"]
        climax = "gradual build" if any(indicator in dream_text.lower() for indicator in climax_indicators) else "steady progression"
        
        # Identify transitions
        transition_words = ["then", "next", "suddenly", "after", "before", "while"]
        transitions = [word for word in transition_words if word in dream_text.lower()]
        
        return NarrativeStructure(
            structure=structure,
            pacing=pacing,
            climax=climax,
            transitions=transitions
        )
    
    def _extract_sensory_details(self, dream_text: str) -> SensoryDetails:
        """Extract sensory information from dream description."""
        text_lower = dream_text.lower()
        
        # Extract colors
        color_words = list(self.color_mappings.keys())
        colors = [color for color in color_words if color in text_lower]
        
        # Extract sounds
        sound_words = ["music", "singing", "noise", "quiet", "loud", "whisper", "scream", "echo"]
        sounds = [sound for sound in sound_words if sound in text_lower]
        
        # Extract textures
        texture_words = ["soft", "rough", "smooth", "hard", "warm", "cold", "wet", "dry"]
        textures = [texture for texture in texture_words if texture in text_lower]
        
        # Determine atmosphere
        atmosphere_indicators = {
            "peaceful": ["calm", "peaceful", "serene", "quiet"],
            "tense": ["tense", "anxious", "worried", "scary"],
            "mysterious": ["mysterious", "strange", "unknown", "dark"],
            "joyful": ["happy", "joyful", "bright", "cheerful"],
            "melancholic": ["sad", "melancholy", "gloomy", "somber"]
        }
        
        atmosphere = "neutral"
        for mood, indicators in atmosphere_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                atmosphere = mood
                break
        
        # Calculate visual intensity
        visual_words = ["bright", "dark", "colorful", "vivid", "clear", "blurry", "shimmering"]
        visual_intensity = min(1.0, sum(0.2 for word in visual_words if word in text_lower))
        
        return SensoryDetails(
            colors=colors,
            sounds=sounds,
            textures=textures,
            atmosphere=atmosphere,
            visual_intensity=visual_intensity
        )
    
    def _calculate_overall_confidence(
        self, 
        emotion_result: EmotionResult, 
        musical_parameters: MusicalParameters,
        symbols: List[SymbolInterpretation]
    ) -> float:
        """Calculate overall confidence score for the analysis."""
        # Weight different components
        emotion_weight = 0.4
        music_weight = 0.3
        symbol_weight = 0.3
        
        # Get individual confidence scores
        emotion_confidence = emotion_result.confidence_score
        music_confidence = musical_parameters.confidence_score
        symbol_confidence = min(1.0, len(symbols) * 0.2) if symbols else 0.1
        
        # Calculate weighted average
        overall = (
            emotion_confidence * emotion_weight +
            music_confidence * music_weight +
            symbol_confidence * symbol_weight
        )
        
        return round(overall, 3)
    
    def get_analysis_summary(self, analysis: ComprehensiveAnalysis) -> Dict[str, Any]:
        """Get a summary of the analysis results."""
        return {
            "primary_emotions": analysis.emotion_result.primary_emotions,
            "overall_mood": analysis.emotion_result.overall_mood,
            "suggested_tempo": analysis.musical_parameters.tempo["bpm"],
            "suggested_key": analysis.musical_parameters.key["signature"],
            "main_instruments": analysis.musical_parameters.instruments,
            "key_symbols": [s.element for s in analysis.symbols],
            "atmosphere": analysis.sensory.atmosphere,
            "confidence": analysis.overall_confidence
        }
