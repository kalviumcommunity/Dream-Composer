"""
Multi-Shot Dream Analyzer for Dream Composer.

This module implements a dream analyzer that uses multi-shot prompting
to provide comprehensive, example-guided analysis with multiple examples.
"""

import json
import hashlib
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from .multi_shot_prompts import (
    MultiShotPromptBuilder,
    MultiShotStrategy,
    MultiShotConfig
)
from .dynamic_shot_prompts import DreamComplexity
from .zero_shot_prompts import ZeroShotTask

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class MultiShotAnalysisResult:
    """Result from multi-shot analysis."""
    task: ZeroShotTask
    analysis: Dict[str, Any]
    confidence: float
    complexity: DreamComplexity
    strategy_used: MultiShotStrategy
    num_examples_used: int
    examples_used: List[str]
    diversity_score: float
    average_example_quality: float
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
            "num_examples_used": self.num_examples_used,
            "examples_used": self.examples_used,
            "diversity_score": self.diversity_score,
            "average_example_quality": self.average_example_quality,
            "keywords": self.keywords,
            "raw_response": self.raw_response,
            "timestamp": self.timestamp
        }


@dataclass
class ComprehensiveMultiShotAnalysis:
    """Comprehensive analysis using multi-shot prompting."""
    dream_text: str
    complexity: DreamComplexity
    keywords: List[str]
    emotion_analysis: Optional[MultiShotAnalysisResult]
    musical_recommendation: Optional[MultiShotAnalysisResult]
    symbolism_interpretation: Optional[MultiShotAnalysisResult]
    mood_mapping: Optional[MultiShotAnalysisResult]
    narrative_analysis: Optional[MultiShotAnalysisResult]
    overall_confidence: float
    total_examples_used: int
    average_diversity_score: float
    overall_example_quality: float
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
            "average_diversity_score": self.average_diversity_score,
            "overall_example_quality": self.overall_example_quality,
            "analysis_timestamp": self.analysis_timestamp
        }


class MultiShotDreamAnalyzer:
    """
    Dream analyzer using multi-shot prompting techniques.
    
    This analyzer provides multiple high-quality examples to guide AI analysis,
    offering comprehensive guidance through diverse examples.
    """
    
    def __init__(self, config: Optional[MultiShotConfig] = None):
        self.prompt_builder = MultiShotPromptBuilder(config)
        self.analysis_cache = {}
        self.performance_metrics = {
            "total_analyses": 0,
            "cache_hits": 0,
            "strategy_usage": {},
            "complexity_distribution": {},
            "total_examples_used": 0,
            "total_diversity_score": 0.0,
            "total_example_quality": 0.0,
            "analysis_count": 0
        }
    
    def _generate_cache_key(
        self,
        task: ZeroShotTask,
        dream_text: str,
        additional_context: Optional[str] = None
    ) -> str:
        """Generate cache key for multi-shot analysis including config parameters."""
        # Include all relevant config parameters that affect example selection
        config = self.prompt_builder.config
        config_components = [
            config.strategy.value,
            str(config.min_examples),
            str(config.max_examples),
            str(config.quality_threshold),
            str(config.diversity_weight),
            str(config.relevance_weight),
            str(config.quality_weight),
            str(config.complexity_spread),
            str(config.type_diversity),
            str(config.fallback_to_available)
        ]
        
        key_components = [
            "multi_shot",
            task.value,
            "|".join(config_components),
            dream_text,
            additional_context or ""
        ]
        
        combined_key = "|".join(key_components)
        hash_object = hashlib.sha256(combined_key.encode('utf-8'))
        return f"multishot:{task.value}:{hash_object.hexdigest()[:16]}"
    
    def analyze_single_task(
        self,
        task: ZeroShotTask,
        dream_text: str,
        additional_context: Optional[str] = None,
        use_cache: bool = True
    ) -> MultiShotAnalysisResult:
        """
        Perform multi-shot analysis for a single task.
        
        Args:
            task: The analysis task to perform
            dream_text: The dream description to analyze
            additional_context: Optional additional context
            use_cache: Whether to use cached results
            
        Returns:
            MultiShotAnalysisResult containing the analysis
        """
        # Always increment total analyses counter
        self.performance_metrics["total_analyses"] += 1
        
        # Check cache first
        cache_key = self._generate_cache_key(task, dream_text, additional_context)
        if use_cache and cache_key in self.analysis_cache:
            self.performance_metrics["cache_hits"] += 1
            logger.debug(f"Cache hit for multi-shot task {task.value}")
            return self.analysis_cache[cache_key]
        
        try:
            # Build multi-shot prompt
            prompt_data = self.prompt_builder.build_multi_shot_prompt(
                task, dream_text, additional_context
            )
            
            # Simulate AI response (in production, this would call actual AI API)
            simulated_response = self._simulate_multi_shot_response(
                task, dream_text, prompt_data
            )
            
            # Parse the response
            analysis = self._parse_multi_shot_response(task, simulated_response)
            confidence = analysis.get("confidence", 0.5)
            
            # Create result
            result = MultiShotAnalysisResult(
                task=task,
                analysis=analysis,
                confidence=confidence,
                complexity=DreamComplexity(prompt_data["complexity"]),
                strategy_used=MultiShotStrategy(prompt_data["strategy"]),
                num_examples_used=prompt_data["num_examples"],
                examples_used=prompt_data["selected_examples"],
                diversity_score=prompt_data["diversity_score"],
                average_example_quality=prompt_data["average_quality"],
                keywords=prompt_data["keywords"],
                raw_response=simulated_response,
                timestamp=datetime.now().isoformat()
            )
            
            # Update metrics for successful analysis
            self.performance_metrics["analysis_count"] += 1
            self.performance_metrics["total_examples_used"] += result.num_examples_used
            self.performance_metrics["total_diversity_score"] += result.diversity_score
            self.performance_metrics["total_example_quality"] += result.average_example_quality
            
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
                logger.debug(f"Cached multi-shot result for task {task.value}")
            
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parsing failed for multi-shot task {task.value}: {e}")
            return self._fallback_multi_shot_analysis(task, dream_text, f"JSON parsing error: {e}")
        except KeyError as e:
            logger.warning(f"Missing required key in multi-shot response for task {task.value}: {e}")
            return self._fallback_multi_shot_analysis(task, dream_text, f"Missing key error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in multi-shot task {task.value}: {type(e).__name__}: {e}")
            return self._fallback_multi_shot_analysis(task, dream_text, f"Unexpected error: {e}")
    
    def analyze_comprehensive(
        self,
        dream_text: str,
        tasks: Optional[List[ZeroShotTask]] = None,
        additional_context: Optional[str] = None
    ) -> ComprehensiveMultiShotAnalysis:
        """
        Perform comprehensive multi-shot analysis across multiple tasks.
        
        Args:
            dream_text: The dream description to analyze
            tasks: List of tasks to perform (default: all tasks)
            additional_context: Optional additional context
            
        Returns:
            ComprehensiveMultiShotAnalysis containing all results
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
        total_examples = 0
        diversity_scores = []
        example_qualities = []
        
        for task in tasks:
            try:
                result = self.analyze_single_task(task, dream_text, additional_context)
                results[task] = result
                confidences.append(result.confidence)
                total_examples += result.num_examples_used
                diversity_scores.append(result.diversity_score)
                example_qualities.append(result.average_example_quality)
                logger.debug(f"Successfully completed multi-shot task {task.value}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"JSON parsing failed for multi-shot task {task.value}: {e}")
                continue
            except KeyError as e:
                logger.warning(f"Missing required key for multi-shot task {task.value}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error in multi-shot task {task.value}: {type(e).__name__}: {e}")
                continue
        
        # Calculate overall metrics
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        average_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0
        overall_quality = sum(example_qualities) / len(example_qualities) if example_qualities else 0.0
        
        return ComprehensiveMultiShotAnalysis(
            dream_text=dream_text,
            complexity=complexity,
            keywords=keywords,
            emotion_analysis=results.get(ZeroShotTask.DREAM_EMOTION_ANALYSIS),
            musical_recommendation=results.get(ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION),
            symbolism_interpretation=results.get(ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION),
            mood_mapping=results.get(ZeroShotTask.MOOD_TO_MUSIC_MAPPING),
            narrative_analysis=results.get(ZeroShotTask.DREAM_NARRATIVE_ANALYSIS),
            overall_confidence=overall_confidence,
            total_examples_used=total_examples,
            average_diversity_score=average_diversity,
            overall_example_quality=overall_quality,
            analysis_timestamp=datetime.now().isoformat()
        )

    def _simulate_multi_shot_response(
        self,
        task: ZeroShotTask,
        dream_text: str,
        prompt_data: Dict[str, Any]
    ) -> str:
        """Simulate AI response for multi-shot prompting."""
        complexity = DreamComplexity(prompt_data["complexity"])
        num_examples = prompt_data["num_examples"]
        diversity_score = prompt_data["diversity_score"]
        avg_quality = prompt_data["average_quality"]

        # Adjust response quality based on examples and diversity
        base_confidence = 0.75
        example_bonus = min(num_examples * 0.05, 0.2)  # Up to 0.2 bonus for multiple examples
        diversity_bonus = diversity_score * 0.1  # Up to 0.1 bonus for high diversity
        quality_bonus = avg_quality * 0.1  # Up to 0.1 bonus for high-quality examples

        complexity_adjustment = {
            DreamComplexity.SIMPLE: 0.05,
            DreamComplexity.MODERATE: 0.0,
            DreamComplexity.COMPLEX: -0.05,
            DreamComplexity.HIGHLY_COMPLEX: -0.1
        }

        final_confidence = base_confidence + example_bonus + diversity_bonus + quality_bonus + complexity_adjustment[complexity]
        final_confidence = max(0.6, min(0.95, final_confidence))

        # Generate task-specific responses
        if task == ZeroShotTask.DREAM_EMOTION_ANALYSIS:
            return json.dumps({
                "primary_emotions": ["wonder", "curiosity", "anticipation"],
                "emotion_intensities": {"wonder": 8, "curiosity": 7, "anticipation": 6},
                "emotional_progression": "building exploratory emotions with growing excitement",
                "dominant_mood": "exploratory",
                "emotional_triggers": ["dream imagery", "symbolic elements", "narrative progression"],
                "confidence": final_confidence,
                "analysis_depth": f"Enhanced by {num_examples} examples with {diversity_score:.2f} diversity"
            })

        elif task == ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION:
            return json.dumps({
                "recommended_style": "cinematic_orchestral",
                "tempo": {"bpm": 100, "description": "building and expansive"},
                "key_signature": {"key": "A", "mode": "major", "reasoning": "hopeful exploration"},
                "instruments": ["full_orchestra", "piano", "strings", "woodwinds"],
                "dynamics": "mp to f with dramatic builds",
                "musical_structure": "programmatic form with thematic development",
                "special_techniques": ["leitmotifs", "orchestral colors", "dynamic contrast"],
                "confidence": final_confidence,
                "guidance_quality": f"Informed by {num_examples} diverse examples"
            })

        elif task == ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION:
            return json.dumps({
                "symbols": [
                    {
                        "element": "exploration",
                        "interpretation": "seeking understanding and new experiences",
                        "psychological_meaning": "intellectual and emotional growth",
                        "emotional_significance": "wonder and discovery",
                        "musical_relevance": "ascending themes and expanding harmonies"
                    },
                    {
                        "element": "journey",
                        "interpretation": "personal transformation and progress",
                        "psychological_meaning": "life path and development",
                        "emotional_significance": "anticipation and purpose",
                        "musical_relevance": "motivic development and progression"
                    }
                ],
                "overall_symbolic_theme": "quest for knowledge and growth",
                "archetypal_patterns": ["the explorer", "the seeker"],
                "confidence": final_confidence,
                "example_guidance": f"Analysis enriched by {num_examples} symbolic examples"
            })

        # Default response
        return json.dumps({
            "analysis": f"Multi-shot analysis for {task.value}",
            "confidence": final_confidence,
            "examples_used": num_examples,
            "diversity_score": diversity_score,
            "note": f"Analysis guided by {num_examples} examples with diversity {diversity_score:.2f}"
        })

    def _parse_multi_shot_response(self, task: ZeroShotTask, response: str) -> Dict[str, Any]:
        """Parse multi-shot AI response."""
        logger.debug(f"Parsing multi-shot response for task {task.value}")

        try:
            parsed = json.loads(response)
            logger.debug(f"Successfully parsed direct JSON for multi-shot task {task.value}")
            return parsed
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parsing failed for multi-shot task {task.value}: {e}")

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
                        logger.debug(f"Successfully extracted JSON using {description} for multi-shot task {task.value}")
                        return parsed
                    except json.JSONDecodeError as parse_error:
                        logger.debug(f"Failed to parse {description} for multi-shot task {task.value}: {parse_error}")
                        continue

            error_msg = f"Could not parse multi-shot response for task {task.value}"
            logger.warning(error_msg)
            raise ValueError(error_msg)

    def _fallback_multi_shot_analysis(
        self,
        task: ZeroShotTask,
        dream_text: str,
        error_message: Optional[str] = None
    ) -> MultiShotAnalysisResult:
        """Provide fallback analysis for multi-shot prompting."""
        logger.info(f"Using fallback analysis for multi-shot task {task.value}")
        if error_message:
            logger.debug(f"Fallback reason: {error_message}")

        # Analyze complexity for fallback
        from .dynamic_shot_prompts import DynamicShotPromptBuilder
        temp_builder = DynamicShotPromptBuilder()
        complexity = temp_builder.analyze_dream_complexity(dream_text)
        keywords = temp_builder.extract_keywords(dream_text)

        fallback_analysis = {
            "analysis": f"Fallback multi-shot analysis for {task.value}",
            "confidence": 0.4,
            "note": "Analysis performed using fallback method",
            "error_info": error_message if error_message else "Unknown error"
        }

        return MultiShotAnalysisResult(
            task=task,
            analysis=fallback_analysis,
            confidence=0.4,
            complexity=complexity,
            strategy_used=self.prompt_builder.config.strategy,
            num_examples_used=0,
            examples_used=[],
            diversity_score=0.0,
            average_example_quality=0.0,
            keywords=keywords,
            raw_response="fallback_analysis",
            timestamp=datetime.now().isoformat()
        )

    def get_analysis_summary(self, analysis: ComprehensiveMultiShotAnalysis) -> Dict[str, Any]:
        """Get a summary of comprehensive multi-shot analysis."""
        summary = {
            "dream_text": analysis.dream_text[:100] + "..." if len(analysis.dream_text) > 100 else analysis.dream_text,
            "complexity": analysis.complexity.value,
            "total_keywords": len(analysis.keywords),
            "top_keywords": analysis.keywords[:5],
            "overall_confidence": analysis.overall_confidence,
            "total_examples_used": analysis.total_examples_used,
            "average_diversity_score": analysis.average_diversity_score,
            "overall_example_quality": analysis.overall_example_quality,
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
                    "num_examples_used": result.num_examples_used,
                    "diversity_score": result.diversity_score,
                    "average_example_quality": result.average_example_quality,
                    "key_findings": list(result.analysis.keys())[:3]
                })

        return summary

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the multi-shot analyzer."""
        cache_hit_rate = (
            self.performance_metrics["cache_hits"] / self.performance_metrics["total_analyses"]
            if self.performance_metrics["total_analyses"] > 0 else 0.0
        )

        average_examples_used = (
            self.performance_metrics["total_examples_used"] / self.performance_metrics["analysis_count"]
            if self.performance_metrics["analysis_count"] > 0 else 0.0
        )

        average_diversity_score = (
            self.performance_metrics["total_diversity_score"] / self.performance_metrics["analysis_count"]
            if self.performance_metrics["analysis_count"] > 0 else 0.0
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
            "total_examples_used": self.performance_metrics["total_examples_used"],
            "analysis_count": self.performance_metrics["analysis_count"],
            "average_examples_used": average_examples_used,
            "total_diversity_score": self.performance_metrics["total_diversity_score"],
            "average_diversity_score": average_diversity_score,
            "total_example_quality": self.performance_metrics["total_example_quality"],
            "average_example_quality": average_example_quality,
            "cache_size": len(self.analysis_cache),
            "example_database_stats": self.prompt_builder.get_example_statistics(),
            "selection_statistics": self.prompt_builder.get_selection_statistics()
        }

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self.analysis_cache.clear()
        logger.info("Cleared multi-shot analysis cache")

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics = {
            "total_analyses": 0,
            "cache_hits": 0,
            "strategy_usage": {},
            "complexity_distribution": {},
            "total_examples_used": 0,
            "total_diversity_score": 0.0,
            "total_example_quality": 0.0,
            "analysis_count": 0
        }
        logger.info("Reset multi-shot performance metrics")

    def change_strategy(self, new_strategy: MultiShotStrategy) -> None:
        """
        Change the selection strategy for future analyses.

        Args:
            new_strategy: The new strategy to use
        """
        old_strategy = self.prompt_builder.config.strategy
        self.prompt_builder.config.strategy = new_strategy
        logger.info(f"Changed multi-shot strategy from {old_strategy.value} to {new_strategy.value}")

        # Clear cache since strategy affects example selection
        self.clear_cache()

    def update_config(self, new_config: MultiShotConfig) -> None:
        """
        Update the configuration for future analyses.

        Args:
            new_config: The new configuration to use
        """
        old_config = self.prompt_builder.config
        self.prompt_builder.config = new_config
        logger.info(f"Updated multi-shot configuration")
        logger.debug(f"Strategy: {old_config.strategy.value} -> {new_config.strategy.value}")
        logger.debug(f"Max examples: {old_config.max_examples} -> {new_config.max_examples}")

        # Clear cache since config changes affect example selection and caching
        self.clear_cache()

    def get_config_hash(self) -> str:
        """
        Get a hash of the current configuration for cache validation.

        Returns:
            SHA-256 hash of the current configuration
        """
        config = self.prompt_builder.config
        config_string = f"{config.strategy.value}|{config.min_examples}|{config.max_examples}|{config.quality_threshold}|{config.diversity_weight}|{config.relevance_weight}|{config.quality_weight}|{config.complexity_spread}|{config.type_diversity}|{config.fallback_to_available}"
        return hashlib.sha256(config_string.encode('utf-8')).hexdigest()[:16]

    def get_strategy_comparison(self, dream_text: str, task: ZeroShotTask) -> Dict[str, Any]:
        """
        Compare how different strategies would select examples for a given dream.

        Args:
            dream_text: The dream to analyze
            task: The task to perform

        Returns:
            Dictionary comparing strategy selections
        """
        comparison = self.prompt_builder.get_strategy_comparison(task, dream_text)

        # Add analyzer-specific information
        comparison["analyzer_config"] = {
            "current_strategy": self.prompt_builder.config.strategy.value,
            "min_examples": self.prompt_builder.config.min_examples,
            "max_examples": self.prompt_builder.config.max_examples,
            "quality_threshold": self.prompt_builder.config.quality_threshold
        }

        return comparison
