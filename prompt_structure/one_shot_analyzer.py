"""
One-Shot Dream Analyzer for Dream Composer.

This module implements a dream analyzer that uses one-shot prompting
to provide focused, example-guided analysis with exactly one example.
"""

import json
import hashlib
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from .one_shot_prompts import (
    OneShotPromptBuilder,
    OneShotStrategy,
    OneShotConfig
)
from .dynamic_shot_prompts import DreamComplexity
from .zero_shot_prompts import ZeroShotTask

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class OneShotAnalysisResult:
    """Result from one-shot analysis."""
    task: ZeroShotTask
    analysis: Dict[str, Any]
    confidence: float
    complexity: DreamComplexity
    strategy_used: OneShotStrategy
    example_used: Optional[str]
    example_quality: float
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
            "strategy_used": self.strategy_used.value,
            "example_used": self.example_used,
            "example_quality": self.example_quality,
            "keywords": self.keywords,
            "raw_response": self.raw_response,
            "timestamp": self.timestamp
        }


@dataclass
class ComprehensiveOneShotAnalysis:
    """Comprehensive analysis using one-shot prompting."""
    dream_text: str
    complexity: DreamComplexity
    keywords: List[str]
    emotion_analysis: Optional[OneShotAnalysisResult]
    musical_recommendation: Optional[OneShotAnalysisResult]
    symbolism_interpretation: Optional[OneShotAnalysisResult]
    mood_mapping: Optional[OneShotAnalysisResult]
    narrative_analysis: Optional[OneShotAnalysisResult]
    overall_confidence: float
    total_example_quality: float
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
            "total_example_quality": self.total_example_quality,
            "analysis_timestamp": self.analysis_timestamp
        }


class OneShotDreamAnalyzer:
    """
    Dream analyzer using one-shot prompting techniques.
    
    This analyzer provides exactly one high-quality example to guide AI analysis,
    striking a balance between zero-shot and few-shot approaches.
    """
    
    def __init__(self, config: Optional[OneShotConfig] = None):
        self.prompt_builder = OneShotPromptBuilder(config)
        self.analysis_cache = {}
        self.performance_metrics = {
            "total_analyses": 0,
            "cache_hits": 0,
            "strategy_usage": {},
            "complexity_distribution": {},
            "average_example_quality": 0.0,
            "total_example_quality": 0.0,
            "analysis_count": 0
        }
    
    def _generate_cache_key(
        self,
        task: ZeroShotTask,
        dream_text: str,
        additional_context: Optional[str] = None
    ) -> str:
        """Generate cache key for one-shot analysis."""
        key_components = [
            "one_shot",
            task.value,
            self.prompt_builder.config.strategy.value,
            dream_text,
            additional_context or ""
        ]
        
        combined_key = "|".join(key_components)
        hash_object = hashlib.sha256(combined_key.encode('utf-8'))
        return f"oneshot:{task.value}:{hash_object.hexdigest()[:16]}"
    
    def analyze_single_task(
        self,
        task: ZeroShotTask,
        dream_text: str,
        additional_context: Optional[str] = None,
        use_cache: bool = True
    ) -> OneShotAnalysisResult:
        """
        Perform one-shot analysis for a single task.
        
        Args:
            task: The analysis task to perform
            dream_text: The dream description to analyze
            additional_context: Optional additional context
            use_cache: Whether to use cached results
            
        Returns:
            OneShotAnalysisResult containing the analysis
        """
        # Always increment total analyses counter
        self.performance_metrics["total_analyses"] += 1
        
        # Check cache first
        cache_key = self._generate_cache_key(task, dream_text, additional_context)
        if use_cache and cache_key in self.analysis_cache:
            self.performance_metrics["cache_hits"] += 1
            logger.debug(f"Cache hit for one-shot task {task.value}")
            return self.analysis_cache[cache_key]
        
        try:
            # Build one-shot prompt
            prompt_data = self.prompt_builder.build_one_shot_prompt(
                task, dream_text, additional_context
            )
            
            # Simulate AI response (in production, this would call actual AI API)
            simulated_response = self._simulate_one_shot_response(
                task, dream_text, prompt_data
            )
            
            # Parse the response
            analysis = self._parse_one_shot_response(task, simulated_response)
            confidence = analysis.get("confidence", 0.5)
            
            # Create result
            result = OneShotAnalysisResult(
                task=task,
                analysis=analysis,
                confidence=confidence,
                complexity=DreamComplexity(prompt_data["complexity"]),
                strategy_used=OneShotStrategy(prompt_data["strategy"]),
                example_used=prompt_data["selected_example"],
                example_quality=prompt_data["example_quality"],
                keywords=prompt_data["keywords"],
                raw_response=simulated_response,
                timestamp=datetime.now().isoformat()
            )
            
            # Update metrics for successful analysis
            self.performance_metrics["analysis_count"] += 1
            self.performance_metrics["total_example_quality"] += result.example_quality
            
            # Update strategy usage
            strategy = result.strategy_used.value
            self.performance_metrics["strategy_usage"][strategy] = \
                self.performance_metrics["strategy_usage"].get(strategy, 0) + 1
            
            # Update complexity distribution
            complexity = result.complexity.value
            self.performance_metrics["complexity_distribution"][complexity] = \
                self.performance_metrics["complexity_distribution"].get(complexity, 0) + 1
            
            # Cache the result
            if use_cache:
                self.analysis_cache[cache_key] = result
                logger.debug(f"Cached one-shot result for task {task.value}")
            
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parsing failed for one-shot task {task.value}: {e}")
            return self._fallback_one_shot_analysis(task, dream_text, f"JSON parsing error: {e}")
        except KeyError as e:
            logger.warning(f"Missing required key in one-shot response for task {task.value}: {e}")
            return self._fallback_one_shot_analysis(task, dream_text, f"Missing key error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in one-shot task {task.value}: {type(e).__name__}: {e}")
            return self._fallback_one_shot_analysis(task, dream_text, f"Unexpected error: {e}")
    
    def analyze_comprehensive(
        self,
        dream_text: str,
        tasks: Optional[List[ZeroShotTask]] = None,
        additional_context: Optional[str] = None
    ) -> ComprehensiveOneShotAnalysis:
        """
        Perform comprehensive one-shot analysis across multiple tasks.
        
        Args:
            dream_text: The dream description to analyze
            tasks: List of tasks to perform (default: all tasks)
            additional_context: Optional additional context
            
        Returns:
            ComprehensiveOneShotAnalysis containing all results
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
        from .dynamic_shot_prompts import DynamicShotPromptBuilder
        temp_builder = DynamicShotPromptBuilder()
        complexity = temp_builder.analyze_dream_complexity(dream_text)
        keywords = temp_builder.extract_keywords(dream_text)
        
        # Perform individual analyses
        results = {}
        confidences = []
        example_qualities = []
        
        for task in tasks:
            try:
                result = self.analyze_single_task(task, dream_text, additional_context)
                results[task] = result
                confidences.append(result.confidence)
                example_qualities.append(result.example_quality)
                logger.debug(f"Successfully completed one-shot task {task.value}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"JSON parsing failed for one-shot task {task.value}: {e}")
                continue
            except KeyError as e:
                logger.warning(f"Missing required key for one-shot task {task.value}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error in one-shot task {task.value}: {type(e).__name__}: {e}")
                continue
        
        # Calculate overall metrics
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        total_example_quality = sum(example_qualities) if example_qualities else 0.0
        
        return ComprehensiveOneShotAnalysis(
            dream_text=dream_text,
            complexity=complexity,
            keywords=keywords,
            emotion_analysis=results.get(ZeroShotTask.DREAM_EMOTION_ANALYSIS),
            musical_recommendation=results.get(ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION),
            symbolism_interpretation=results.get(ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION),
            mood_mapping=results.get(ZeroShotTask.MOOD_TO_MUSIC_MAPPING),
            narrative_analysis=results.get(ZeroShotTask.DREAM_NARRATIVE_ANALYSIS),
            overall_confidence=overall_confidence,
            total_example_quality=total_example_quality,
            analysis_timestamp=datetime.now().isoformat()
        )
    
    def _simulate_one_shot_response(
        self,
        task: ZeroShotTask,
        dream_text: str,
        prompt_data: Dict[str, Any]
    ) -> str:
        """Simulate AI response for one-shot prompting."""
        complexity = DreamComplexity(prompt_data["complexity"])
        example_quality = prompt_data["example_quality"]
        
        # Adjust response quality based on example quality and complexity
        base_confidence = 0.75
        quality_bonus = example_quality * 0.15  # Up to 0.15 bonus for high-quality examples
        complexity_adjustment = {
            DreamComplexity.SIMPLE: 0.05,
            DreamComplexity.MODERATE: 0.0,
            DreamComplexity.COMPLEX: -0.05,
            DreamComplexity.HIGHLY_COMPLEX: -0.1
        }
        
        final_confidence = base_confidence + quality_bonus + complexity_adjustment[complexity]
        final_confidence = max(0.6, min(0.95, final_confidence))
        
        # Generate task-specific responses
        if task == ZeroShotTask.DREAM_EMOTION_ANALYSIS:
            return json.dumps({
                "primary_emotions": ["curiosity", "wonder"],
                "emotion_intensities": {"curiosity": 7, "wonder": 8},
                "emotional_progression": "consistent exploratory emotions",
                "dominant_mood": "inquisitive",
                "emotional_triggers": ["dream imagery", "symbolic elements"],
                "confidence": final_confidence
            })
        
        elif task == ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION:
            return json.dumps({
                "recommended_style": "contemplative_classical",
                "tempo": {"bpm": 85, "description": "thoughtful and flowing"},
                "key_signature": {"key": "G", "mode": "major", "reasoning": "hopeful curiosity"},
                "instruments": ["piano", "strings", "woodwinds"],
                "dynamics": "mp to mf with expressive phrasing",
                "musical_structure": "sonata form with development",
                "special_techniques": ["legato", "dynamic contrast"],
                "confidence": final_confidence
            })
        
        elif task == ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION:
            return json.dumps({
                "symbols": [
                    {
                        "element": "exploration",
                        "interpretation": "seeking understanding and knowledge",
                        "psychological_meaning": "intellectual curiosity and growth",
                        "emotional_significance": "wonder and discovery",
                        "musical_relevance": "ascending melodic lines and modulation"
                    }
                ],
                "overall_symbolic_theme": "quest for knowledge",
                "archetypal_patterns": ["the seeker", "the journey"],
                "confidence": final_confidence
            })
        
        # Default response
        return json.dumps({
            "analysis": f"One-shot analysis for {task.value}",
            "confidence": final_confidence,
            "note": f"Analysis guided by example with quality {example_quality:.2f}"
        })

    def _parse_one_shot_response(self, task: ZeroShotTask, response: str) -> Dict[str, Any]:
        """Parse one-shot AI response."""
        logger.debug(f"Parsing one-shot response for task {task.value}")

        try:
            parsed = json.loads(response)
            logger.debug(f"Successfully parsed direct JSON for one-shot task {task.value}")
            return parsed
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parsing failed for one-shot task {task.value}: {e}")

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
                        logger.debug(f"Successfully extracted JSON using {description} for one-shot task {task.value}")
                        return parsed
                    except json.JSONDecodeError as parse_error:
                        logger.debug(f"Failed to parse {description} for one-shot task {task.value}: {parse_error}")
                        continue

            error_msg = f"Could not parse one-shot response for task {task.value}"
            logger.warning(error_msg)
            raise ValueError(error_msg)

    def _fallback_one_shot_analysis(
        self,
        task: ZeroShotTask,
        dream_text: str,
        error_message: Optional[str] = None
    ) -> OneShotAnalysisResult:
        """Provide fallback analysis for one-shot prompting."""
        logger.info(f"Using fallback analysis for one-shot task {task.value}")
        if error_message:
            logger.debug(f"Fallback reason: {error_message}")

        # Analyze complexity for fallback
        from .dynamic_shot_prompts import DynamicShotPromptBuilder
        temp_builder = DynamicShotPromptBuilder()
        complexity = temp_builder.analyze_dream_complexity(dream_text)
        keywords = temp_builder.extract_keywords(dream_text)

        fallback_analysis = {
            "analysis": f"Fallback one-shot analysis for {task.value}",
            "confidence": 0.4,
            "note": "Analysis performed using fallback method",
            "error_info": error_message if error_message else "Unknown error"
        }

        return OneShotAnalysisResult(
            task=task,
            analysis=fallback_analysis,
            confidence=0.4,
            complexity=complexity,
            strategy_used=self.prompt_builder.config.strategy,
            example_used=None,
            example_quality=0.0,
            keywords=keywords,
            raw_response="fallback_analysis",
            timestamp=datetime.now().isoformat()
        )

    def get_analysis_summary(self, analysis: ComprehensiveOneShotAnalysis) -> Dict[str, Any]:
        """Get a summary of comprehensive one-shot analysis."""
        summary = {
            "dream_text": analysis.dream_text[:100] + "..." if len(analysis.dream_text) > 100 else analysis.dream_text,
            "complexity": analysis.complexity.value,
            "total_keywords": len(analysis.keywords),
            "top_keywords": analysis.keywords[:5],
            "overall_confidence": analysis.overall_confidence,
            "total_example_quality": analysis.total_example_quality,
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
                    "strategy_used": result.strategy_used.value,
                    "example_quality": result.example_quality,
                    "key_findings": list(result.analysis.keys())[:3]
                })

        return summary

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the one-shot analyzer."""
        cache_hit_rate = (
            self.performance_metrics["cache_hits"] / self.performance_metrics["total_analyses"]
            if self.performance_metrics["total_analyses"] > 0 else 0.0
        )

        average_example_quality = (
            self.performance_metrics["total_example_quality"] / self.performance_metrics["analysis_count"]
            if self.performance_metrics["analysis_count"] > 0 else 0.0
        )

        return {
            "total_analyses": self.performance_metrics["total_analyses"],
            "cache_hits": self.performance_metrics["cache_hits"],
            "cache_hit_rate": cache_hit_rate,
            "strategy_usage": self.performance_metrics["strategy_usage"],
            "complexity_distribution": self.performance_metrics["complexity_distribution"],
            "total_example_quality": self.performance_metrics["total_example_quality"],
            "analysis_count": self.performance_metrics["analysis_count"],
            "average_example_quality": average_example_quality,
            "cache_size": len(self.analysis_cache),
            "example_database_stats": self.prompt_builder.get_example_statistics(),
            "selection_statistics": self.prompt_builder.get_selection_statistics()
        }

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self.analysis_cache.clear()
        logger.info("Cleared one-shot analysis cache")

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics = {
            "total_analyses": 0,
            "cache_hits": 0,
            "strategy_usage": {},
            "complexity_distribution": {},
            "average_example_quality": 0.0,
            "total_example_quality": 0.0,
            "analysis_count": 0
        }
        logger.info("Reset one-shot performance metrics")

    def change_strategy(self, new_strategy: OneShotStrategy) -> None:
        """
        Change the selection strategy for future analyses.

        Args:
            new_strategy: The new strategy to use
        """
        old_strategy = self.prompt_builder.config.strategy
        self.prompt_builder.config.strategy = new_strategy
        logger.info(f"Changed one-shot strategy from {old_strategy.value} to {new_strategy.value}")

        # Clear cache since strategy affects example selection
        self.clear_cache()

    def get_strategy_comparison(self, dream_text: str, task: ZeroShotTask) -> Dict[str, Any]:
        """
        Compare how different strategies would select examples for a given dream.

        Args:
            dream_text: The dream to analyze
            task: The task to perform

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
        original_strategy = self.prompt_builder.config.strategy

        for strategy in OneShotStrategy:
            try:
                self.prompt_builder.config.strategy = strategy
                selected_example = self.prompt_builder.select_one_shot_example(
                    task, dream_text, complexity, keywords
                )

                if selected_example:
                    comparison["strategy_selections"][strategy.value] = {
                        "example_text": selected_example.dream_text[:80] + "...",
                        "example_quality": self.prompt_builder.calculate_example_quality(selected_example),
                        "example_complexity": selected_example.complexity.value,
                        "example_type": selected_example.example_type.value
                    }
                else:
                    comparison["strategy_selections"][strategy.value] = {
                        "example_text": "No suitable example found",
                        "example_quality": 0.0,
                        "example_complexity": "N/A",
                        "example_type": "N/A"
                    }
            except Exception as e:
                comparison["strategy_selections"][strategy.value] = {
                    "error": str(e),
                    "example_quality": 0.0
                }

        # Restore original strategy
        self.prompt_builder.config.strategy = original_strategy

        return comparison
