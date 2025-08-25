"""
Tests for One-Shot Dream Analyzer functionality.
"""

import pytest
import json
from prompt_structure.one_shot_analyzer import (
    OneShotDreamAnalyzer,
    OneShotAnalysisResult,
    ComprehensiveOneShotAnalysis
)
from prompt_structure.one_shot_prompts import (
    OneShotStrategy,
    OneShotConfig
)
from prompt_structure.dynamic_shot_prompts import DreamComplexity
from prompt_structure.zero_shot_prompts import ZeroShotTask


class TestOneShotDreamAnalyzer:
    """Test cases for OneShotDreamAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = OneShotDreamAnalyzer()
        
        # Sample dreams for testing
        self.sample_dreams = {
            "simple": "I was flying over a city, feeling happy.",
            "moderate": "I started in a garden feeling peaceful, but then storm clouds gathered and I became anxious.",
            "complex": "I was in a vast library with floating books. I could read them by thinking, feeling amazed and curious.",
            "nightmare": "I was being chased through dark corridors, feeling terrified and helpless."
        }
    
    def test_initialization(self):
        """Test analyzer initialization."""
        assert isinstance(self.analyzer, OneShotDreamAnalyzer)
        assert hasattr(self.analyzer, 'prompt_builder')
        assert hasattr(self.analyzer, 'analysis_cache')
        assert hasattr(self.analyzer, 'performance_metrics')
        
        # Check initial metrics
        metrics = self.analyzer.performance_metrics
        assert metrics["total_analyses"] == 0
        assert metrics["cache_hits"] == 0
        assert isinstance(metrics["strategy_usage"], dict)
        assert isinstance(metrics["complexity_distribution"], dict)
        assert metrics["average_example_quality"] == 0.0
        assert metrics["total_example_quality"] == 0.0
        assert metrics["analysis_count"] == 0
    
    def test_initialization_with_custom_config(self):
        """Test analyzer initialization with custom configuration."""
        custom_config = OneShotConfig(
            strategy=OneShotStrategy.REPRESENTATIVE,
            quality_threshold=0.9
        )
        
        custom_analyzer = OneShotDreamAnalyzer(custom_config)
        assert custom_analyzer.prompt_builder.config.strategy == OneShotStrategy.REPRESENTATIVE
        assert custom_analyzer.prompt_builder.config.quality_threshold == 0.9
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        key1 = self.analyzer._generate_cache_key(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            "I was flying",
            "context"
        )
        
        key2 = self.analyzer._generate_cache_key(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            "I was flying",
            "context"
        )
        
        key3 = self.analyzer._generate_cache_key(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            "I was swimming",
            "context"
        )
        
        # Same inputs should generate same key
        assert key1 == key2
        
        # Different inputs should generate different keys
        assert key1 != key3
        
        # Keys should be strings and contain task info
        assert isinstance(key1, str)
        assert "oneshot" in key1
        assert "dream_emotion_analysis" in key1
    
    def test_analyze_single_task_emotion_analysis(self):
        """Test single task analysis for emotion analysis."""
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"]
        )
        
        assert isinstance(result, OneShotAnalysisResult)
        assert result.task == ZeroShotTask.DREAM_EMOTION_ANALYSIS
        assert isinstance(result.analysis, dict)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.complexity, DreamComplexity)
        assert isinstance(result.strategy_used, OneShotStrategy)
        assert isinstance(result.example_quality, float)
        assert 0.0 <= result.example_quality <= 1.0
        assert isinstance(result.keywords, list)
        assert isinstance(result.raw_response, str)
        assert isinstance(result.timestamp, str)
    
    def test_analyze_single_task_musical_recommendation(self):
        """Test single task analysis for musical recommendation."""
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
            self.sample_dreams["moderate"]
        )
        
        assert isinstance(result, OneShotAnalysisResult)
        assert result.task == ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION
        assert "recommended_style" in result.analysis or "analysis" in result.analysis
        assert isinstance(result.confidence, float)
        assert isinstance(result.complexity, DreamComplexity)
        assert isinstance(result.strategy_used, OneShotStrategy)
    
    def test_analyze_single_task_symbolism_interpretation(self):
        """Test single task analysis for symbolism interpretation."""
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION,
            self.sample_dreams["complex"]
        )
        
        assert isinstance(result, OneShotAnalysisResult)
        assert result.task == ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION
        assert isinstance(result.analysis, dict)
        assert isinstance(result.complexity, DreamComplexity)
        assert isinstance(result.strategy_used, OneShotStrategy)
    
    def test_analyze_single_task_with_additional_context(self):
        """Test single task analysis with additional context."""
        additional_context = "The dreamer is a professional pilot."
        
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            additional_context
        )
        
        assert isinstance(result, OneShotAnalysisResult)
        assert result.task == ZeroShotTask.DREAM_EMOTION_ANALYSIS
    
    def test_analyze_single_task_caching(self):
        """Test that caching works correctly."""
        # First analysis
        result1 = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            use_cache=True
        )
        
        # Second analysis should use cache
        result2 = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            use_cache=True
        )
        
        # Results should be identical (from cache)
        assert result1.timestamp == result2.timestamp
        assert result1.raw_response == result2.raw_response
        
        # Cache hit should be recorded
        assert self.analyzer.performance_metrics["cache_hits"] > 0
    
    def test_analyze_single_task_no_caching(self):
        """Test analysis without caching."""
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            use_cache=False
        )
        
        assert isinstance(result, OneShotAnalysisResult)
        # Cache should remain empty
        assert len(self.analyzer.analysis_cache) == 0
    
    def test_analyze_comprehensive_default_tasks(self):
        """Test comprehensive analysis with default tasks."""
        analysis = self.analyzer.analyze_comprehensive(
            self.sample_dreams["moderate"]
        )
        
        assert isinstance(analysis, ComprehensiveOneShotAnalysis)
        assert analysis.dream_text == self.sample_dreams["moderate"]
        assert isinstance(analysis.complexity, DreamComplexity)
        assert isinstance(analysis.keywords, list)
        assert isinstance(analysis.overall_confidence, float)
        assert 0.0 <= analysis.overall_confidence <= 1.0
        assert isinstance(analysis.total_example_quality, float)
        assert analysis.total_example_quality >= 0.0
        assert isinstance(analysis.analysis_timestamp, str)
        
        # Should have at least some analyses completed
        completed_analyses = [
            analysis.emotion_analysis,
            analysis.musical_recommendation,
            analysis.symbolism_interpretation,
            analysis.mood_mapping,
            analysis.narrative_analysis
        ]
        assert any(result is not None for result in completed_analyses)
    
    def test_analyze_comprehensive_specific_tasks(self):
        """Test comprehensive analysis with specific tasks."""
        tasks = [
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION
        ]
        
        analysis = self.analyzer.analyze_comprehensive(
            self.sample_dreams["simple"],
            tasks=tasks
        )
        
        assert isinstance(analysis, ComprehensiveOneShotAnalysis)
        assert analysis.emotion_analysis is not None
        assert analysis.musical_recommendation is not None
        # These should be None since not requested
        assert analysis.symbolism_interpretation is None
        assert analysis.mood_mapping is None
        assert analysis.narrative_analysis is None
    
    def test_analyze_comprehensive_with_context(self):
        """Test comprehensive analysis with additional context."""
        additional_context = "The dreamer is going through a major life transition."
        
        analysis = self.analyzer.analyze_comprehensive(
            self.sample_dreams["complex"],
            additional_context=additional_context
        )
        
        assert isinstance(analysis, ComprehensiveOneShotAnalysis)
        assert analysis.dream_text == self.sample_dreams["complex"]
    
    def test_one_shot_analysis_result_to_dict(self):
        """Test OneShotAnalysisResult to_dict method."""
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"]
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "task" in result_dict
        assert "analysis" in result_dict
        assert "confidence" in result_dict
        assert "complexity" in result_dict
        assert "strategy_used" in result_dict
        assert "example_used" in result_dict
        assert "example_quality" in result_dict
        assert "keywords" in result_dict
        assert "raw_response" in result_dict
        assert "timestamp" in result_dict
        
        assert result_dict["task"] == result.task.value
        assert result_dict["complexity"] == result.complexity.value
        assert result_dict["strategy_used"] == result.strategy_used.value
    
    def test_comprehensive_one_shot_analysis_to_dict(self):
        """Test ComprehensiveOneShotAnalysis to_dict method."""
        analysis = self.analyzer.analyze_comprehensive(
            self.sample_dreams["moderate"]
        )
        
        analysis_dict = analysis.to_dict()
        
        assert isinstance(analysis_dict, dict)
        assert "dream_text" in analysis_dict
        assert "complexity" in analysis_dict
        assert "keywords" in analysis_dict
        assert "overall_confidence" in analysis_dict
        assert "total_example_quality" in analysis_dict
        assert "analysis_timestamp" in analysis_dict
        
        assert analysis_dict["complexity"] == analysis.complexity.value
    
    def test_get_analysis_summary(self):
        """Test getting analysis summary."""
        analysis = self.analyzer.analyze_comprehensive(
            self.sample_dreams["complex"]
        )
        
        summary = self.analyzer.get_analysis_summary(analysis)
        
        assert isinstance(summary, dict)
        assert "dream_text" in summary
        assert "complexity" in summary
        assert "total_keywords" in summary
        assert "top_keywords" in summary
        assert "overall_confidence" in summary
        assert "total_example_quality" in summary
        assert "completed_tasks" in summary
        
        assert isinstance(summary["completed_tasks"], list)
        assert summary["complexity"] == analysis.complexity.value
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        # Perform some analyses to generate metrics
        self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"]
        )
        self.analyzer.analyze_single_task(
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
            self.sample_dreams["moderate"]
        )
        
        metrics = self.analyzer.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_analyses" in metrics
        assert "cache_hits" in metrics
        assert "cache_hit_rate" in metrics
        assert "strategy_usage" in metrics
        assert "complexity_distribution" in metrics
        assert "total_example_quality" in metrics
        assert "analysis_count" in metrics
        assert "average_example_quality" in metrics
        assert "cache_size" in metrics
        assert "example_database_stats" in metrics
        assert "selection_statistics" in metrics
        
        assert metrics["total_analyses"] >= 2
        assert isinstance(metrics["cache_hit_rate"], float)
        assert 0.0 <= metrics["cache_hit_rate"] <= 1.0
        assert isinstance(metrics["average_example_quality"], float)
        assert metrics["average_example_quality"] >= 0.0
    
    def test_clear_cache(self):
        """Test clearing the analysis cache."""
        # Perform analysis to populate cache
        self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"]
        )
        
        assert len(self.analyzer.analysis_cache) > 0
        
        # Clear cache
        self.analyzer.clear_cache()
        assert len(self.analyzer.analysis_cache) == 0
    
    def test_reset_metrics(self):
        """Test resetting performance metrics."""
        # Perform analysis to generate metrics
        self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"]
        )
        
        assert self.analyzer.performance_metrics["total_analyses"] > 0
        
        # Reset metrics
        self.analyzer.reset_metrics()
        
        metrics = self.analyzer.performance_metrics
        assert metrics["total_analyses"] == 0
        assert metrics["cache_hits"] == 0
        assert metrics["strategy_usage"] == {}
        assert metrics["complexity_distribution"] == {}
        assert metrics["average_example_quality"] == 0.0
        assert metrics["total_example_quality"] == 0.0
        assert metrics["analysis_count"] == 0
    
    def test_change_strategy(self):
        """Test changing the selection strategy."""
        original_strategy = self.analyzer.prompt_builder.config.strategy
        new_strategy = OneShotStrategy.REPRESENTATIVE
        
        # Change strategy
        self.analyzer.change_strategy(new_strategy)
        
        assert self.analyzer.prompt_builder.config.strategy == new_strategy
        assert self.analyzer.prompt_builder.config.strategy != original_strategy
        
        # Cache should be cleared
        assert len(self.analyzer.analysis_cache) == 0
    
    def test_get_strategy_comparison(self):
        """Test strategy comparison functionality."""
        comparison = self.analyzer.get_strategy_comparison(
            self.sample_dreams["simple"],
            ZeroShotTask.DREAM_EMOTION_ANALYSIS
        )
        
        assert isinstance(comparison, dict)
        assert "dream_text" in comparison
        assert "complexity" in comparison
        assert "keywords" in comparison
        assert "strategy_selections" in comparison
        
        # Should have entries for all strategies
        strategy_selections = comparison["strategy_selections"]
        assert len(strategy_selections) == len(OneShotStrategy)
        
        for strategy in OneShotStrategy:
            assert strategy.value in strategy_selections
            selection = strategy_selections[strategy.value]
            assert "example_quality" in selection
    
    def test_simulate_one_shot_response_emotion_analysis(self):
        """Test simulated response for emotion analysis."""
        prompt_data = {
            "complexity": DreamComplexity.MODERATE.value,
            "example_quality": 0.85
        }
        
        response = self.analyzer._simulate_one_shot_response(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["moderate"],
            prompt_data
        )
        
        assert isinstance(response, str)
        
        # Should be valid JSON
        parsed = json.loads(response)
        assert isinstance(parsed, dict)
        assert "confidence" in parsed
        assert isinstance(parsed["confidence"], float)
    
    def test_simulate_one_shot_response_musical_recommendation(self):
        """Test simulated response for musical recommendation."""
        prompt_data = {
            "complexity": DreamComplexity.SIMPLE.value,
            "example_quality": 0.9
        }
        
        response = self.analyzer._simulate_one_shot_response(
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
            self.sample_dreams["simple"],
            prompt_data
        )
        
        assert isinstance(response, str)
        
        # Should be valid JSON
        parsed = json.loads(response)
        assert isinstance(parsed, dict)
        assert "confidence" in parsed
    
    def test_parse_one_shot_response_valid_json(self):
        """Test parsing valid JSON response."""
        valid_response = json.dumps({
            "primary_emotions": ["joy", "freedom"],
            "confidence": 0.85
        })
        
        result = self.analyzer._parse_one_shot_response(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            valid_response
        )
        
        assert isinstance(result, dict)
        assert "primary_emotions" in result
        assert result["confidence"] == 0.85
    
    def test_parse_one_shot_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        invalid_response = "This is not valid JSON"
        
        with pytest.raises(ValueError):
            self.analyzer._parse_one_shot_response(
                ZeroShotTask.DREAM_EMOTION_ANALYSIS,
                invalid_response
            )
    
    def test_fallback_one_shot_analysis(self):
        """Test fallback analysis."""
        result = self.analyzer._fallback_one_shot_analysis(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            "Test error message"
        )
        
        assert isinstance(result, OneShotAnalysisResult)
        assert result.task == ZeroShotTask.DREAM_EMOTION_ANALYSIS
        assert result.confidence == 0.4  # Fallback confidence
        assert "fallback" in result.analysis["analysis"].lower()
        assert "Test error message" in result.analysis["error_info"]
        assert isinstance(result.complexity, DreamComplexity)
        assert isinstance(result.strategy_used, OneShotStrategy)
        assert result.example_used is None
        assert result.example_quality == 0.0
    
    def test_empty_dream_analysis(self):
        """Test analysis of empty dream text."""
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            ""
        )
        
        assert isinstance(result, OneShotAnalysisResult)
        assert result.complexity == DreamComplexity.SIMPLE
        assert result.keywords == []
    
    def test_very_long_dream_analysis(self):
        """Test analysis of very long dream text."""
        long_dream = " ".join([self.sample_dreams["complex"]] * 10)
        
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            long_dream
        )
        
        assert isinstance(result, OneShotAnalysisResult)
        assert result.complexity == DreamComplexity.HIGHLY_COMPLEX
        assert len(result.keywords) <= 20  # Should be limited
    
    def test_all_strategies_produce_results(self):
        """Test that all strategies produce valid results."""
        for strategy in OneShotStrategy:
            # Create analyzer with specific strategy
            config = OneShotConfig(strategy=strategy)
            analyzer = OneShotDreamAnalyzer(config)
            
            result = analyzer.analyze_single_task(
                ZeroShotTask.DREAM_EMOTION_ANALYSIS,
                self.sample_dreams["simple"]
            )
            
            assert isinstance(result, OneShotAnalysisResult)
            assert result.strategy_used == strategy
            assert result.confidence > 0.0
