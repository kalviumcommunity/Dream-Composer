"""
Dynamic Shot Dream Analyzer for Dream Composer.

This module implements a dream analyzer that uses dynamic shot prompting
to provide context-aware analysis with intelligently selected examples.
"""

import json
import hashlib
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from .dynamic_shot_prompts import (
    DynamicShotPromptBuilder, 
    DreamComplexity, 
    ExampleType,
    DynamicShotConfig
)
from .zero_shot_prompts import ZeroShotTask
from .zero_shot_analyzer import ZeroShotAnalysisResult

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DynamicShotAnalysisResult:
    """Result from dynamic shot analysis."""
    task: ZeroShotTask
    analysis: Dict[str, Any]
    confidence: float
    complexity: DreamComplexity
    num_examples_used: int
    selected_examples: List[str]
    keywords: List[str]
    raw_response: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task": self.task.value,
            "analysis": self.analysis,
            "confidence": self.confidence,
            "complexity": self.complexity.value,
            "num_examples_used": self.num_examples_used,
            "selected_examples": self.selected_examples,
            "keywords": self.keywords,
            "raw_response": self.raw_response,
            "timestamp": self.timestamp
        }


@dataclass
class ComprehensiveDynamicShotAnalysis:
    """Comprehensive analysis using dynamic shot prompting."""
    dream_text: str
    complexity: DreamComplexity
    keywords: List[str]
    emotion_analysis: Optional[DynamicShotAnalysisResult]
    musical_recommendation: Optional[DynamicShotAnalysisResult]
    symbolism_interpretation: Optional[DynamicShotAnalysisResult]
    mood_mapping: Optional[DynamicShotAnalysisResult]
    narrative_analysis: Optional[DynamicShotAnalysisResult]
    overall_confidence: float
    total_examples_used: int
    analysis_timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dream_text": self.dream_text,
            "complexity": self.complexity.value,
            "keywords": self.keywords,
            "emotion_analysis": self.emotion_analysis.to_dict() if self.emotion_analysis else None,
            "musical_recommendation": self.musical_recommendation.to_dict() if self.musical_recommendation else None,
            "symbolism_interpretation": self.symbolism_interpretation.to_dict() if self.symbolism_interpretation else None,
            "mood_mapping": self.mood_mapping.to_dict() if self.mood_mapping else None,
            "narrative_analysis": self.narrative_analysis.to_dict() if self.narrative_analysis else None,
            "overall_confidence": self.overall_confidence,
            "total_examples_used": self.total_examples_used,
            "analysis_timestamp": self.analysis_timestamp
        }


class DynamicShotDreamAnalyzer:
    """
    Dream analyzer using dynamic shot prompting techniques.
    
    This analyzer intelligently selects relevant examples based on dream content,
    complexity, and context to provide more accurate and contextual analysis.
    """
    
    def __init__(self, config: Optional[DynamicShotConfig] = None):
        self.prompt_builder = DynamicShotPromptBuilder(config)
        self.analysis_cache = {}
        self.performance_metrics = {
            "total_analyses": 0,
            "cache_hits": 0,
            "complexity_distribution": {},
            "total_examples_used": 0,
            "analysis_count": 0  # Separate counter for average calculation
        }
    
    def _generate_cache_key(
        self,
        task: ZeroShotTask,
        dream_text: str,
        additional_context: Optional[str] = None
    ) -> str:
        """Generate cache key for dynamic shot analysis."""
        key_components = [
            "dynamic_shot",
            task.value,
            dream_text,
            additional_context or ""
        ]
        
        combined_key = "|".join(key_components)
        hash_object = hashlib.sha256(combined_key.encode('utf-8'))
        return f"dynamic:{task.value}:{hash_object.hexdigest()[:16]}"
    
    def analyze_single_task(
        self,
        task: ZeroShotTask,
        dream_text: str,
        additional_context: Optional[str] = None,
        use_cache: bool = True
    ) -> DynamicShotAnalysisResult:
        """
        Perform dynamic shot analysis for a single task.
        
        Args:
            task: The analysis task to perform
            dream_text: The dream description to analyze
            additional_context: Optional additional context
            use_cache: Whether to use cached results
            
        Returns:
            DynamicShotAnalysisResult containing the analysis
        """
        # Always increment total analyses counter
        self.performance_metrics["total_analyses"] += 1

        # Check cache first
        cache_key = self._generate_cache_key(task, dream_text, additional_context)
        if use_cache and cache_key in self.analysis_cache:
            self.performance_metrics["cache_hits"] += 1
            logger.debug(f"Cache hit for dynamic shot task {task.value}")
            return self.analysis_cache[cache_key]
        
        try:
            # Build dynamic shot prompt
            prompt_data = self.prompt_builder.build_dynamic_shot_prompt(
                task, dream_text, additional_context
            )
            
            # Simulate AI response (in production, this would call actual AI API)
            simulated_response = self._simulate_dynamic_shot_response(
                task, dream_text, prompt_data
            )
            
            # Parse the response
            analysis = self._parse_dynamic_shot_response(task, simulated_response)
            confidence = analysis.get("confidence", 0.5)
            
            # Create result
            result = DynamicShotAnalysisResult(
                task=task,
                analysis=analysis,
                confidence=confidence,
                complexity=DreamComplexity(prompt_data["complexity"]),
                num_examples_used=prompt_data["num_examples"],
                selected_examples=prompt_data["selected_examples"],
                keywords=prompt_data["keywords"],
                raw_response=simulated_response,
                timestamp=datetime.now().isoformat()
            )
            
            # Update metrics for successful analysis
            self.performance_metrics["analysis_count"] += 1
            self.performance_metrics["total_examples_used"] += result.num_examples_used

            complexity = result.complexity.value
            self.performance_metrics["complexity_distribution"][complexity] = \
                self.performance_metrics["complexity_distribution"].get(complexity, 0) + 1
            
            # Cache the result
            if use_cache:
                self.analysis_cache[cache_key] = result
                logger.debug(f"Cached dynamic shot result for task {task.value}")
            
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parsing failed for dynamic shot task {task.value}: {e}")
            return self._fallback_dynamic_analysis(task, dream_text, f"JSON parsing error: {e}")
        except KeyError as e:
            logger.warning(f"Missing required key in dynamic shot response for task {task.value}: {e}")
            return self._fallback_dynamic_analysis(task, dream_text, f"Missing key error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in dynamic shot task {task.value}: {type(e).__name__}: {e}")
            return self._fallback_dynamic_analysis(task, dream_text, f"Unexpected error: {e}")
    
    def analyze_comprehensive(
        self,
        dream_text: str,
        tasks: Optional[List[ZeroShotTask]] = None,
        additional_context: Optional[str] = None
    ) -> ComprehensiveDynamicShotAnalysis:
        """
        Perform comprehensive dynamic shot analysis across multiple tasks.
        
        Args:
            dream_text: The dream description to analyze
            tasks: List of tasks to perform (default: all tasks)
            additional_context: Optional additional context
            
        Returns:
            ComprehensiveDynamicShotAnalysis containing all results
        """
        if tasks is None:
            tasks = [
                ZeroShotTask.DREAM_EMOTION_ANALYSIS,
                ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
                ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION,
                ZeroShotTask.MOOD_TO_MUSIC_MAPPING,
                ZeroShotTask.DREAM_NARRATIVE_ANALYSIS
            ]
        
        # Analyze complexity once for all tasks
        complexity = self.prompt_builder.analyze_dream_complexity(dream_text)
        keywords = self.prompt_builder.extract_keywords(dream_text)
        
        # Perform individual analyses
        results = {}
        confidences = []
        total_examples_used = 0
        
        for task in tasks:
            try:
                result = self.analyze_single_task(task, dream_text, additional_context)
                results[task] = result
                confidences.append(result.confidence)
                total_examples_used += result.num_examples_used
                logger.debug(f"Successfully completed dynamic shot task {task.value}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"JSON parsing failed for dynamic shot task {task.value}: {e}")
                continue
            except KeyError as e:
                logger.warning(f"Missing required key for dynamic shot task {task.value}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error in dynamic shot task {task.value}: {type(e).__name__}: {e}")
                continue
        
        # Calculate overall confidence
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return ComprehensiveDynamicShotAnalysis(
            dream_text=dream_text,
            complexity=complexity,
            keywords=keywords,
            emotion_analysis=results.get(ZeroShotTask.DREAM_EMOTION_ANALYSIS),
            musical_recommendation=results.get(ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION),
            symbolism_interpretation=results.get(ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION),
            mood_mapping=results.get(ZeroShotTask.MOOD_TO_MUSIC_MAPPING),
            narrative_analysis=results.get(ZeroShotTask.DREAM_NARRATIVE_ANALYSIS),
            overall_confidence=overall_confidence,
            total_examples_used=total_examples_used,
            analysis_timestamp=datetime.now().isoformat()
        )
    
    def _simulate_dynamic_shot_response(
        self,
        task: ZeroShotTask,
        dream_text: str,
        prompt_data: Dict[str, Any]
    ) -> str:
        """Simulate AI response for dynamic shot prompting."""
        complexity = DreamComplexity(prompt_data["complexity"])
        num_examples = prompt_data["num_examples"]
        
        # Adjust response quality based on number of examples and complexity
        base_confidence = 0.7
        example_bonus = min(num_examples * 0.05, 0.2)  # Up to 0.2 bonus
        complexity_adjustment = {
            DreamComplexity.SIMPLE: 0.1,
            DreamComplexity.MODERATE: 0.0,
            DreamComplexity.COMPLEX: -0.05,
            DreamComplexity.HIGHLY_COMPLEX: -0.1
        }
        
        final_confidence = base_confidence + example_bonus + complexity_adjustment[complexity]
        final_confidence = max(0.5, min(0.95, final_confidence))
        
        # Generate task-specific responses
        if task == ZeroShotTask.DREAM_EMOTION_ANALYSIS:
            return json.dumps({
                "primary_emotions": ["wonder", "curiosity"],
                "emotion_intensities": {"wonder": 8, "curiosity": 7},
                "emotional_progression": "consistent exploratory emotions",
                "dominant_mood": "inquisitive",
                "emotional_triggers": ["dream imagery", "symbolic content"],
                "confidence": final_confidence
            })
        
        elif task == ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION:
            return json.dumps({
                "recommended_style": "ambient_orchestral",
                "tempo": {"bpm": 90, "description": "contemplative and flowing"},
                "key_signature": {"key": "A", "mode": "minor", "reasoning": "introspective mood"},
                "instruments": ["strings", "piano", "woodwinds"],
                "dynamics": "mp to mf with gentle expression",
                "musical_structure": "through-composed with thematic development",
                "special_techniques": ["legato phrasing", "subtle dynamics"],
                "confidence": final_confidence
            })
        
        elif task == ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION:
            return json.dumps({
                "symbols": [
                    {
                        "element": "exploration",
                        "interpretation": "seeking new understanding",
                        "psychological_meaning": "personal growth and discovery",
                        "musical_relevance": "ascending melodic lines"
                    }
                ],
                "overall_symbolic_theme": "journey of discovery",
                "archetypal_patterns": ["the quest"],
                "confidence": final_confidence
            })
        
        # Default response
        return json.dumps({
            "analysis": f"Dynamic shot analysis for {task.value}",
            "confidence": final_confidence,
            "note": f"Analysis used {num_examples} examples with {complexity.value} complexity"
        })
    
    def _parse_dynamic_shot_response(self, task: ZeroShotTask, response: str) -> Dict[str, Any]:
        """Parse dynamic shot AI response."""
        logger.debug(f"Parsing dynamic shot response for task {task.value}")
        
        try:
            parsed = json.loads(response)
            logger.debug(f"Successfully parsed direct JSON for dynamic shot task {task.value}")
            return parsed
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parsing failed for dynamic shot task {task.value}: {e}")
            
            # Try to extract JSON using patterns
            import re
            json_patterns = [
                (r'```json\s*(\{.*?\})\s*```', "JSON code block"),
                (r'\{[^{}]*"confidence"[^{}]*\}', "confidence-containing JSON"),
                (r'\{.*?\}', "any JSON-like structure")
            ]
            
            for pattern, description in json_patterns:
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    try:
                        json_text = match.group(1) if match.lastindex else match.group()
                        parsed = json.loads(json_text)
                        logger.debug(f"Successfully extracted JSON using {description} for dynamic shot task {task.value}")
                        return parsed
                    except json.JSONDecodeError as parse_error:
                        logger.debug(f"Failed to parse {description} for dynamic shot task {task.value}: {parse_error}")
                        continue
            
            error_msg = f"Could not parse dynamic shot response for task {task.value}"
            logger.warning(error_msg)
            raise ValueError(error_msg)
    
    def _fallback_dynamic_analysis(
        self,
        task: ZeroShotTask,
        dream_text: str,
        error_message: Optional[str] = None
    ) -> DynamicShotAnalysisResult:
        """Provide fallback analysis for dynamic shot prompting."""
        logger.info(f"Using fallback analysis for dynamic shot task {task.value}")
        if error_message:
            logger.debug(f"Fallback reason: {error_message}")
        
        # Analyze complexity for fallback
        complexity = self.prompt_builder.analyze_dream_complexity(dream_text)
        keywords = self.prompt_builder.extract_keywords(dream_text)
        
        fallback_analysis = {
            "analysis": f"Fallback dynamic shot analysis for {task.value}",
            "confidence": 0.3,
            "note": "Analysis performed using fallback method",
            "error_info": error_message if error_message else "Unknown error"
        }
        
        return DynamicShotAnalysisResult(
            task=task,
            analysis=fallback_analysis,
            confidence=0.3,
            complexity=complexity,
            num_examples_used=0,
            selected_examples=[],
            keywords=keywords,
            raw_response="fallback_analysis",
            timestamp=datetime.now().isoformat()
        )
    
    def get_analysis_summary(self, analysis: ComprehensiveDynamicShotAnalysis) -> Dict[str, Any]:
        """Get a summary of comprehensive dynamic shot analysis."""
        summary = {
            "dream_text": analysis.dream_text[:100] + "..." if len(analysis.dream_text) > 100 else analysis.dream_text,
            "complexity": analysis.complexity.value,
            "total_keywords": len(analysis.keywords),
            "top_keywords": analysis.keywords[:5],
            "overall_confidence": analysis.overall_confidence,
            "total_examples_used": analysis.total_examples_used,
            "completed_tasks": []
        }
        
        # Add completed task summaries
        task_results = [
            ("emotion_analysis", analysis.emotion_analysis),
            ("musical_recommendation", analysis.musical_recommendation),
            ("symbolism_interpretation", analysis.symbolism_interpretation),
            ("mood_mapping", analysis.mood_mapping),
            ("narrative_analysis", analysis.narrative_analysis)
        ]
        
        for task_name, result in task_results:
            if result:
                summary["completed_tasks"].append({
                    "task": task_name,
                    "confidence": result.confidence,
                    "examples_used": result.num_examples_used,
                    "key_findings": list(result.analysis.keys())[:3]
                })
        
        return summary
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the dynamic shot analyzer."""
        cache_hit_rate = (
            self.performance_metrics["cache_hits"] / self.performance_metrics["total_analyses"]
            if self.performance_metrics["total_analyses"] > 0 else 0.0
        )

        average_examples_used = (
            self.performance_metrics["total_examples_used"] / self.performance_metrics["analysis_count"]
            if self.performance_metrics["analysis_count"] > 0 else 0.0
        )

        return {
            "total_analyses": self.performance_metrics["total_analyses"],
            "cache_hits": self.performance_metrics["cache_hits"],
            "cache_hit_rate": cache_hit_rate,
            "complexity_distribution": self.performance_metrics["complexity_distribution"],
            "total_examples_used": self.performance_metrics["total_examples_used"],
            "analysis_count": self.performance_metrics["analysis_count"],
            "average_examples_used": average_examples_used,
            "cache_size": len(self.analysis_cache),
            "example_database_stats": self.prompt_builder.get_example_statistics(),
            "usage_statistics": self.prompt_builder.get_usage_statistics()
        }
    
    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self.analysis_cache.clear()
        logger.info("Cleared dynamic shot analysis cache")
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics = {
            "total_analyses": 0,
            "cache_hits": 0,
            "complexity_distribution": {},
            "total_examples_used": 0,
            "analysis_count": 0
        }
        logger.info("Reset dynamic shot performance metrics")
