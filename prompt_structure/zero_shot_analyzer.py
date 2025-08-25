"""
Zero-Shot Dream Analyzer for Dream Composer.

This module provides zero-shot analysis capabilities that can analyze dreams
without requiring training examples, leveraging AI models' pre-trained knowledge.
"""

import json
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

from .zero_shot_prompts import ZeroShotPromptBuilder, ZeroShotTask
from .emotion_extractor import EmotionResult
from .music_mapper import MusicalParameters


@dataclass
class ZeroShotAnalysisResult:
    """Result from zero-shot analysis."""
    task: ZeroShotTask
    analysis: Dict[str, Any]
    confidence: float
    raw_response: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "task": self.task.value,
            "analysis": self.analysis,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }


@dataclass
class ComprehensiveZeroShotAnalysis:
    """Complete zero-shot analysis of a dream."""
    dream_text: str
    emotion_analysis: Optional[ZeroShotAnalysisResult]
    musical_recommendation: Optional[ZeroShotAnalysisResult]
    symbolism_interpretation: Optional[ZeroShotAnalysisResult]
    mood_mapping: Optional[ZeroShotAnalysisResult]
    narrative_analysis: Optional[ZeroShotAnalysisResult]
    overall_confidence: float
    analysis_timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "dream_text": self.dream_text,
            "emotion_analysis": self.emotion_analysis.to_dict() if self.emotion_analysis else None,
            "musical_recommendation": self.musical_recommendation.to_dict() if self.musical_recommendation else None,
            "symbolism_interpretation": self.symbolism_interpretation.to_dict() if self.symbolism_interpretation else None,
            "mood_mapping": self.mood_mapping.to_dict() if self.mood_mapping else None,
            "narrative_analysis": self.narrative_analysis.to_dict() if self.narrative_analysis else None,
            "overall_confidence": self.overall_confidence,
            "analysis_timestamp": self.analysis_timestamp
        }


class ZeroShotDreamAnalyzer:
    """
    Zero-shot dream analyzer that performs analysis without training examples.
    
    This analyzer uses zero-shot prompting techniques to leverage AI models'
    pre-trained knowledge for dream analysis tasks.
    """
    
    def __init__(self):
        self.prompt_builder = ZeroShotPromptBuilder()
        self.analysis_cache = {}  # Simple cache for repeated analyses
    
    def analyze_single_task(
        self, 
        task: ZeroShotTask, 
        dream_text: str,
        additional_context: Optional[str] = None,
        use_cache: bool = True
    ) -> ZeroShotAnalysisResult:
        """
        Perform zero-shot analysis for a single task.
        
        Args:
            task: The zero-shot task to perform
            dream_text: The dream description to analyze
            additional_context: Optional additional context
            use_cache: Whether to use cached results
            
        Returns:
            ZeroShotAnalysisResult containing the analysis
        """
        # Check cache first
        cache_key = f"{task.value}:{hash(dream_text)}"
        if use_cache and cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Build the zero-shot prompt
        prompt_data = self.prompt_builder.build_zero_shot_prompt(
            task, dream_text, additional_context
        )
        
        # Simulate AI response (in production, this would call an AI API)
        simulated_response = self._simulate_zero_shot_response(task, dream_text)
        
        # Parse the response
        try:
            analysis = self._parse_zero_shot_response(task, simulated_response)
            confidence = analysis.get("confidence", 0.5)
            
            result = ZeroShotAnalysisResult(
                task=task,
                analysis=analysis,
                confidence=confidence,
                raw_response=simulated_response,
                timestamp=datetime.now().isoformat()
            )
            
            # Cache the result
            if use_cache:
                self.analysis_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            # Fallback analysis
            return self._fallback_analysis(task, dream_text)
    
    def analyze_comprehensive(
        self, 
        dream_text: str,
        tasks: Optional[List[ZeroShotTask]] = None,
        additional_context: Optional[str] = None
    ) -> ComprehensiveZeroShotAnalysis:
        """
        Perform comprehensive zero-shot analysis across multiple tasks.
        
        Args:
            dream_text: The dream description to analyze
            tasks: List of tasks to perform (default: all tasks)
            additional_context: Optional additional context
            
        Returns:
            ComprehensiveZeroShotAnalysis containing all results
        """
        if tasks is None:
            tasks = [
                ZeroShotTask.DREAM_EMOTION_ANALYSIS,
                ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
                ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION,
                ZeroShotTask.MOOD_TO_MUSIC_MAPPING,
                ZeroShotTask.DREAM_NARRATIVE_ANALYSIS
            ]
        
        # Perform individual analyses
        results = {}
        confidences = []
        
        for task in tasks:
            try:
                result = self.analyze_single_task(task, dream_text, additional_context)
                results[task] = result
                confidences.append(result.confidence)
            except Exception as e:
                # Continue with other tasks if one fails
                print(f"Warning: Failed to analyze task {task}: {e}")
                continue
        
        # Calculate overall confidence
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return ComprehensiveZeroShotAnalysis(
            dream_text=dream_text,
            emotion_analysis=results.get(ZeroShotTask.DREAM_EMOTION_ANALYSIS),
            musical_recommendation=results.get(ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION),
            symbolism_interpretation=results.get(ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION),
            mood_mapping=results.get(ZeroShotTask.MOOD_TO_MUSIC_MAPPING),
            narrative_analysis=results.get(ZeroShotTask.DREAM_NARRATIVE_ANALYSIS),
            overall_confidence=overall_confidence,
            analysis_timestamp=datetime.now().isoformat()
        )
    
    def _simulate_zero_shot_response(self, task: ZeroShotTask, dream_text: str) -> str:
        """
        Simulate AI response for zero-shot analysis.
        In production, this would be replaced with actual AI API calls.
        """
        text_lower = dream_text.lower()
        
        if task == ZeroShotTask.DREAM_EMOTION_ANALYSIS:
            # Analyze emotions based on keywords and context
            emotions = []
            intensities = {}
            
            emotion_keywords = {
                "joy": ["happy", "joyful", "excited", "elated", "cheerful"],
                "fear": ["scared", "afraid", "terrified", "anxious", "worried"],
                "peace": ["peaceful", "calm", "serene", "tranquil", "relaxed"],
                "sadness": ["sad", "melancholy", "gloomy", "depressed"],
                "wonder": ["amazed", "awed", "curious", "fascinated"],
                "freedom": ["free", "liberated", "flying", "soaring"]
            }
            
            for emotion, keywords in emotion_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    emotions.append(emotion)
                    # Simulate intensity based on context
                    intensities[emotion] = min(9, len([k for k in keywords if k in text_lower]) * 3)
            
            if not emotions:
                emotions = ["neutral"]
                intensities = {"neutral": 5}
            
            response = {
                "primary_emotions": emotions[:3],
                "emotion_intensities": intensities,
                "emotional_progression": "consistent emotional tone throughout",
                "dominant_mood": emotions[0] if emotions else "neutral",
                "emotional_triggers": ["dream imagery", "symbolic content"],
                "confidence": 0.8 if emotions else 0.4
            }
            
        elif task == ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION:
            # Recommend musical style based on dream content
            if any(word in text_lower for word in ["flying", "soaring", "floating"]):
                style = "ambient_classical"
                tempo = {"bpm": 95, "description": "flowing and ethereal"}
                key = {"key": "C", "mode": "major", "reasoning": "uplifting and transcendent"}
                instruments = ["strings", "harp", "flute"]
            elif any(word in text_lower for word in ["dark", "scary", "nightmare"]):
                style = "cinematic_dramatic"
                tempo = {"bpm": 110, "description": "tense and building"}
                key = {"key": "D", "mode": "minor", "reasoning": "mysterious and dramatic"}
                instruments = ["low_strings", "brass", "timpani"]
            else:
                style = "contemporary_classical"
                tempo = {"bpm": 85, "description": "moderate and expressive"}
                key = {"key": "F", "mode": "major", "reasoning": "warm and contemplative"}
                instruments = ["piano", "strings", "woodwinds"]
            
            response = {
                "recommended_style": style,
                "tempo": tempo,
                "key_signature": key,
                "instruments": instruments,
                "dynamics": "mp to mf with expressive variations",
                "musical_structure": "through-composed with thematic development",
                "special_techniques": ["legato phrasing", "dynamic contrast"],
                "confidence": 0.85
            }
            
        elif task == ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION:
            # Interpret symbols in the dream
            symbols = []
            
            symbol_meanings = {
                "flying": {
                    "interpretation": "desire for freedom and transcendence",
                    "psychological_meaning": "escape from limitations or constraints",
                    "emotional_significance": "liberation and empowerment",
                    "musical_relevance": "ascending melodies and light textures"
                },
                "water": {
                    "interpretation": "emotional depth and subconscious exploration",
                    "psychological_meaning": "connection to emotions and intuition",
                    "emotional_significance": "cleansing and renewal",
                    "musical_relevance": "flowing rhythms and fluid dynamics"
                },
                "city": {
                    "interpretation": "social connections and complexity",
                    "psychological_meaning": "navigation of social structures",
                    "emotional_significance": "community and belonging",
                    "musical_relevance": "complex harmonies and urban rhythms"
                }
            }
            
            for symbol, meaning in symbol_meanings.items():
                if symbol in text_lower:
                    symbols.append({
                        "element": symbol,
                        **meaning
                    })
            
            response = {
                "symbols": symbols,
                "overall_symbolic_theme": "personal growth and exploration",
                "archetypal_patterns": ["transformation", "journey"],
                "confidence": 0.75 if symbols else 0.3
            }
            
        else:
            # Default response for other tasks
            response = {
                "analysis": "Basic analysis completed",
                "confidence": 0.5
            }
        
        return json.dumps(response, indent=2)
    
    def _parse_zero_shot_response(self, task: ZeroShotTask, response: str) -> Dict[str, Any]:
        """Parse zero-shot AI response into structured format."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',
                r'\{[^{}]*"confidence"[^{}]*\}',
                r'\{.*?\}'
            ]
            
            for pattern in json_patterns:
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    try:
                        json_text = match.group(1) if match.lastindex else match.group()
                        return json.loads(json_text)
                    except json.JSONDecodeError:
                        continue
            
            raise ValueError("Could not parse zero-shot response")
    
    def _fallback_analysis(self, task: ZeroShotTask, dream_text: str) -> ZeroShotAnalysisResult:
        """Provide fallback analysis when parsing fails."""
        fallback_analysis = {
            "analysis": f"Fallback analysis for {task.value}",
            "confidence": 0.2,
            "note": "Analysis performed using fallback method"
        }
        
        return ZeroShotAnalysisResult(
            task=task,
            analysis=fallback_analysis,
            confidence=0.2,
            raw_response="fallback_analysis",
            timestamp=datetime.now().isoformat()
        )
    
    def get_analysis_summary(self, analysis: ComprehensiveZeroShotAnalysis) -> Dict[str, Any]:
        """Get a summary of the zero-shot analysis results."""
        summary = {
            "dream_text": analysis.dream_text,
            "overall_confidence": analysis.overall_confidence,
            "analysis_timestamp": analysis.analysis_timestamp
        }
        
        if analysis.emotion_analysis:
            emotions = analysis.emotion_analysis.analysis.get("primary_emotions", [])
            summary["primary_emotions"] = emotions
            summary["dominant_mood"] = analysis.emotion_analysis.analysis.get("dominant_mood", "unknown")
        
        if analysis.musical_recommendation:
            musical = analysis.musical_recommendation.analysis
            summary["recommended_style"] = musical.get("recommended_style", "unknown")
            summary["suggested_tempo"] = musical.get("tempo", {}).get("bpm", 0)
            summary["suggested_key"] = musical.get("key_signature", {}).get("key", "unknown")
        
        if analysis.symbolism_interpretation:
            symbols = analysis.symbolism_interpretation.analysis.get("symbols", [])
            summary["key_symbols"] = [s.get("element", "") for s in symbols]
        
        return summary
    
    def clear_cache(self):
        """Clear the analysis cache."""
        self.analysis_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the analysis cache."""
        return {
            "cache_size": len(self.analysis_cache),
            "cached_tasks": list(set(key.split(":")[0] for key in self.analysis_cache.keys()))
        }
