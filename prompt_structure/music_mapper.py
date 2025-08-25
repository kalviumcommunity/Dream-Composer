"""
Music Mapper for Dream Composer.

This module maps emotions and dream content to specific musical parameters
for composition and generation purposes.
"""

import json
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .prompt_builder import PromptBuilder, PromptType
from .emotion_extractor import EmotionResult


class MusicalKey(Enum):
    """Musical keys with their characteristics."""
    C_MAJOR = {"key": "C", "mode": "major", "mood": "pure, simple"}
    G_MAJOR = {"key": "G", "mode": "major", "mood": "bright, cheerful"}
    D_MAJOR = {"key": "D", "mode": "major", "mood": "triumphant, joyful"}
    A_MAJOR = {"key": "A", "mode": "major", "mood": "confident, bright"}
    E_MAJOR = {"key": "E", "mode": "major", "mood": "passionate, brilliant"}
    F_MAJOR = {"key": "F", "mode": "major", "mood": "pastoral, peaceful"}
    Bb_MAJOR = {"key": "Bb", "mode": "major", "mood": "noble, heroic"}
    
    A_MINOR = {"key": "A", "mode": "minor", "mood": "natural, melancholic"}
    E_MINOR = {"key": "E", "mode": "minor", "mood": "deep, contemplative"}
    B_MINOR = {"key": "B", "mode": "minor", "mood": "tragic, passionate"}
    FS_MINOR = {"key": "F#", "mode": "minor", "mood": "mysterious, dark"}
    CS_MINOR = {"key": "C#", "mode": "minor", "mood": "intense, dramatic"}
    D_MINOR = {"key": "D", "mode": "minor", "mood": "serious, solemn"}


class InstrumentFamily(Enum):
    """Instrument families and their emotional associations."""
    STRINGS = {"instruments": ["violin", "viola", "cello", "double_bass"], "emotions": ["love", "sadness", "peace"]}
    WOODWINDS = {"instruments": ["flute", "clarinet", "oboe", "bassoon"], "emotions": ["joy", "wonder", "nature"]}
    BRASS = {"instruments": ["trumpet", "horn", "trombone", "tuba"], "emotions": ["triumph", "power", "celebration"]}
    PERCUSSION = {"instruments": ["timpani", "snare", "cymbals", "bells"], "emotions": ["excitement", "tension", "rhythm"]}
    PIANO = {"instruments": ["piano", "harpsichord"], "emotions": ["versatile", "intimate", "expressive"]}
    ELECTRONIC = {"instruments": ["synthesizer", "pad", "ambient"], "emotions": ["mystery", "modern", "ethereal"]}


@dataclass
class MusicalParameters:
    """Musical parameters for composition."""
    tempo: Dict[str, Any]
    key: Dict[str, str]
    time_signature: str
    instruments: List[str]
    style: str
    dynamics: Dict[str, Any]
    harmonic_progression: List[str]
    expression_marks: List[str]
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "tempo": self.tempo,
            "key": self.key,
            "time_signature": self.time_signature,
            "instruments": self.instruments,
            "style": self.style,
            "dynamics": self.dynamics,
            "harmonic_progression": self.harmonic_progression,
            "expression_marks": self.expression_marks,
            "confidence_score": self.confidence_score
        }


class MusicMapper:
    """
    Maps emotions and dream content to musical parameters.
    
    This class takes emotion analysis results and converts them into
    specific musical parameters that can be used for composition.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        self.prompt_builder = PromptBuilder()
        self.emotion_to_tempo = self._initialize_emotion_tempo_mapping()
        self.emotion_to_key = self._initialize_emotion_key_mapping()
        self.emotion_to_instruments = self._initialize_emotion_instrument_mapping()
        self.style_mappings = self._initialize_style_mappings()
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
    
    def _initialize_emotion_tempo_mapping(self) -> Dict[str, Tuple[int, int]]:
        """Initialize emotion to tempo range mappings."""
        return {
            "joy": (110, 140),
            "excitement": (130, 160),
            "peace": (60, 90),
            "sadness": (70, 100),
            "fear": (100, 130),
            "anger": (120, 150),
            "love": (80, 110),
            "wonder": (90, 120),
            "freedom": (115, 145),
            "nostalgia": (75, 105),
            "anxiety": (105, 135),
            "melancholy": (65, 95),
            "euphoria": (125, 155),
            "contemplative": (55, 85)
        }
    
    def _initialize_emotion_key_mapping(self) -> Dict[str, List[MusicalKey]]:
        """Initialize emotion to musical key mappings."""
        return {
            "joy": [MusicalKey.C_MAJOR, MusicalKey.G_MAJOR, MusicalKey.D_MAJOR],
            "peace": [MusicalKey.F_MAJOR, MusicalKey.C_MAJOR, MusicalKey.Bb_MAJOR],
            "sadness": [MusicalKey.A_MINOR, MusicalKey.D_MINOR, MusicalKey.E_MINOR],
            "fear": [MusicalKey.B_MINOR, MusicalKey.FS_MINOR, MusicalKey.CS_MINOR],
            "love": [MusicalKey.A_MAJOR, MusicalKey.E_MAJOR, MusicalKey.F_MAJOR],
            "excitement": [MusicalKey.E_MAJOR, MusicalKey.A_MAJOR, MusicalKey.D_MAJOR],
            "melancholy": [MusicalKey.E_MINOR, MusicalKey.A_MINOR, MusicalKey.B_MINOR],
            "wonder": [MusicalKey.G_MAJOR, MusicalKey.C_MAJOR, MusicalKey.A_MINOR],
            "freedom": [MusicalKey.D_MAJOR, MusicalKey.A_MAJOR, MusicalKey.G_MAJOR],
            "nostalgia": [MusicalKey.F_MAJOR, MusicalKey.A_MINOR, MusicalKey.D_MINOR]
        }
    
    def _initialize_emotion_instrument_mapping(self) -> Dict[str, List[str]]:
        """Initialize emotion to instrument mappings."""
        return {
            "joy": ["piano", "violin", "flute", "guitar"],
            "peace": ["piano", "strings", "harp", "soft_pad"],
            "sadness": ["cello", "violin", "piano", "oboe"],
            "fear": ["low_strings", "timpani", "dissonant_pad", "brass"],
            "love": ["strings", "piano", "warm_pad", "horn"],
            "excitement": ["full_orchestra", "brass", "percussion", "electric_guitar"],
            "wonder": ["harp", "flute", "strings", "bells"],
            "freedom": ["brass", "strings", "piano", "choir"],
            "nostalgia": ["piano", "strings", "clarinet", "soft_pad"],
            "melancholy": ["cello", "piano", "viola", "bassoon"]
        }
    
    def _initialize_style_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize musical style mappings."""
        return {
            "classical": {"tempo_modifier": 1.0, "complexity": "high", "instruments": "orchestral"},
            "ambient": {"tempo_modifier": 0.8, "complexity": "low", "instruments": "electronic"},
            "cinematic": {"tempo_modifier": 1.1, "complexity": "high", "instruments": "orchestral"},
            "minimalist": {"tempo_modifier": 0.9, "complexity": "low", "instruments": "piano_strings"},
            "romantic": {"tempo_modifier": 0.95, "complexity": "medium", "instruments": "strings_piano"},
            "modern": {"tempo_modifier": 1.05, "complexity": "medium", "instruments": "mixed"},
            "folk": {"tempo_modifier": 1.0, "complexity": "low", "instruments": "acoustic"},
            "electronic": {"tempo_modifier": 1.2, "complexity": "medium", "instruments": "synthesized"}
        }
    
    def map_emotions_to_music(
        self,
        emotion_result: EmotionResult,
        dream_text: str = ""
    ) -> MusicalParameters:
        """
        Map emotion analysis results to musical parameters.

        Args:
            emotion_result: Results from emotion extraction
            dream_text: Original dream text for additional context

        Returns:
            MusicalParameters object with composition parameters
        """
        # Reset random seed if one was provided for deterministic results
        if self.random_seed is not None:
            random.seed(self.random_seed)
        # Determine primary emotion for mapping
        primary_emotion = emotion_result.primary_emotions[0] if emotion_result.primary_emotions else "peace"
        
        # Calculate tempo based on emotions and intensities
        tempo = self._calculate_tempo(emotion_result)
        
        # Select musical key
        key = self._select_key(emotion_result)
        
        # Choose instruments
        instruments = self._select_instruments(emotion_result)
        
        # Determine style
        style = self._determine_style(emotion_result, dream_text)
        
        # Set dynamics
        dynamics = self._determine_dynamics(emotion_result)
        
        # Create harmonic progression
        harmonic_progression = self._create_harmonic_progression(key, emotion_result)
        
        # Add expression marks
        expression_marks = self._select_expression_marks(emotion_result)
        
        return MusicalParameters(
            tempo=tempo,
            key=key,
            time_signature="4/4",  # Default, could be made more sophisticated
            instruments=instruments,
            style=style,
            dynamics=dynamics,
            harmonic_progression=harmonic_progression,
            expression_marks=expression_marks,
            confidence_score=emotion_result.confidence_score * 0.9  # Slightly lower for mapping
        )
    
    def _calculate_tempo(self, emotion_result: EmotionResult) -> Dict[str, Any]:
        """Calculate tempo based on emotions and intensities."""
        base_tempo = 90  # Default moderate tempo
        tempo_adjustments = []
        
        for emotion in emotion_result.primary_emotions:
            if emotion in self.emotion_to_tempo:
                min_tempo, max_tempo = self.emotion_to_tempo[emotion]
                intensity = emotion_result.emotional_intensity.get(emotion, 5.0)
                
                # Scale tempo based on intensity (1-10 scale)
                tempo_in_range = min_tempo + (max_tempo - min_tempo) * (intensity / 10.0)
                tempo_adjustments.append(tempo_in_range)
        
        if tempo_adjustments:
            final_tempo = int(sum(tempo_adjustments) / len(tempo_adjustments))
        else:
            final_tempo = base_tempo
        
        # Determine tempo description
        if final_tempo < 70:
            description = "very slow, contemplative"
        elif final_tempo < 90:
            description = "slow, peaceful"
        elif final_tempo < 110:
            description = "moderate, flowing"
        elif final_tempo < 130:
            description = "moderately fast, energetic"
        else:
            description = "fast, exciting"
        
        return {
            "bpm": final_tempo,
            "description": description
        }
    
    def _select_key(self, emotion_result: EmotionResult) -> Dict[str, str]:
        """Select musical key based on emotions."""
        primary_emotion = emotion_result.primary_emotions[0] if emotion_result.primary_emotions else "peace"

        if primary_emotion in self.emotion_to_key:
            possible_keys = self.emotion_to_key[primary_emotion]
            # Add variety by randomly selecting from possible keys
            selected_key = random.choice(possible_keys)
        else:
            # Default options with some variety
            default_keys = [MusicalKey.C_MAJOR, MusicalKey.G_MAJOR, MusicalKey.F_MAJOR]
            selected_key = random.choice(default_keys)

        key_info = selected_key.value

        return {
            "signature": f"{key_info['key']} {key_info['mode']}",
            "mode": key_info["mode"],
            "reasoning": f"Selected for {primary_emotion}: {key_info['mood']}"
        }
    
    def _select_instruments(self, emotion_result: EmotionResult, max_instruments: int = 4) -> List[str]:
        """Select instruments based on emotions."""
        selected_instruments = set()

        for emotion in emotion_result.primary_emotions:
            if emotion in self.emotion_to_instruments:
                instruments = self.emotion_to_instruments[emotion]
                # Add variety by randomly selecting from available instruments
                num_to_select = min(2, len(instruments))
                selected = random.sample(instruments, num_to_select)
                selected_instruments.update(selected)

        if not selected_instruments:
            # Default options with variety
            default_instruments = ["piano", "strings", "flute", "guitar", "harp"]
            selected_instruments.update(random.sample(default_instruments, 2))

        # Convert to list and limit, with some randomization in order
        instrument_list = list(selected_instruments)
        if len(instrument_list) > max_instruments:
            instrument_list = random.sample(instrument_list, max_instruments)

        return instrument_list
    
    def _determine_style(self, emotion_result: EmotionResult, dream_text: str) -> str:
        """Determine musical style based on emotions and dream content."""
        # Simple heuristic based on primary emotion
        primary_emotion = emotion_result.primary_emotions[0] if emotion_result.primary_emotions else "peace"
        
        style_mapping = {
            "joy": "classical",
            "peace": "ambient",
            "sadness": "romantic",
            "fear": "cinematic",
            "excitement": "modern",
            "love": "romantic",
            "wonder": "cinematic",
            "freedom": "classical",
            "nostalgia": "romantic"
        }
        
        return style_mapping.get(primary_emotion, "ambient")
    
    def _determine_dynamics(self, emotion_result: EmotionResult) -> Dict[str, Any]:
        """Determine dynamics based on emotional intensity."""
        avg_intensity = sum(emotion_result.emotional_intensity.values()) / len(emotion_result.emotional_intensity) if emotion_result.emotional_intensity else 5.0
        
        if avg_intensity < 3:
            overall = "pp"  # Very soft
            variations = ["ppp", "pp", "p"]
        elif avg_intensity < 5:
            overall = "p"   # Soft
            variations = ["pp", "p", "mp"]
        elif avg_intensity < 7:
            overall = "mp"  # Medium soft
            variations = ["p", "mp", "mf"]
        elif avg_intensity < 9:
            overall = "mf"  # Medium loud
            variations = ["mp", "mf", "f"]
        else:
            overall = "f"   # Loud
            variations = ["mf", "f", "ff"]
        
        return {
            "overall": overall,
            "variations": variations
        }
    
    def _create_harmonic_progression(self, key: Dict[str, str], emotion_result: EmotionResult) -> List[str]:
        """Create harmonic progression based on key and emotions."""
        mode = key["mode"]
        primary_emotion = emotion_result.primary_emotions[0] if emotion_result.primary_emotions else "peace"
        
        if mode == "major":
            if primary_emotion in ["joy", "excitement", "freedom"]:
                return ["I", "V", "vi", "IV"]  # Popular progression
            elif primary_emotion in ["peace", "love"]:
                return ["I", "vi", "IV", "V"]  # Gentle progression
            else:
                return ["I", "IV", "V", "I"]   # Simple progression
        else:  # minor
            if primary_emotion in ["sadness", "melancholy"]:
                return ["i", "VI", "III", "VII"]  # Melancholic
            elif primary_emotion in ["fear", "anxiety"]:
                return ["i", "ii°", "V", "i"]     # Tense
            else:
                return ["i", "iv", "V", "i"]      # Standard minor
    
    def _select_expression_marks(self, emotion_result: EmotionResult) -> List[str]:
        """Select expression marks based on emotions."""
        marks = []
        
        for emotion in emotion_result.primary_emotions:
            if emotion in ["peace", "love", "nostalgia"]:
                marks.extend(["legato", "dolce"])
            elif emotion in ["joy", "excitement"]:
                marks.extend(["allegro", "vivace"])
            elif emotion in ["sadness", "melancholy"]:
                marks.extend(["espressivo", "cantabile"])
            elif emotion in ["fear", "anxiety"]:
                marks.extend(["marcato", "agitato"])
        
        return list(set(marks))[:3]  # Remove duplicates and limit to 3
