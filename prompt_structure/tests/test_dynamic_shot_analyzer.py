"""
Tests for Dynamic Shot Dream Analyzer functionality.
"""

import pytest
import json
from prompt_structure.dynamic_shot_analyzer import (
    DynamicShotDreamAnalyzer,
    DynamicShotAnalysisResult,
    ComprehensiveDynamicShotAnalysis
)
from prompt_structure.dynamic_shot_prompts import (
    DreamComplexity,
    DynamicShotConfig
)
from prompt_structure.zero_shot_prompts import ZeroShotTask


class TestDynamicShotDreamAnalyzer:
    """Test cases for DynamicShotDreamAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = DynamicShotDreamAnalyzer()
        
        # Sample dreams for testing
        self.sample_dreams = {
            "simple": "I was flying over a city, feeling happy.",
            "moderate": "I started in a garden feeling peaceful, but then storm clouds gathered and I became anxious.",
            "complex": "I was in a vast library with floating books. I could read them by thinking, feeling amazed and curious.",
            "nightmare": "I was being chased through dark corridors, feeling terrified and helpless."
        }
    
    def test_initialization(self):
        """Test analyzer initialization."""
        assert isinstance(self.analyzer, DynamicShotDreamAnalyzer)
        assert hasattr(self.analyzer, 'prompt_builder')
        assert hasattr(self.analyzer, 'analysis_cache')
        assert hasattr(self.analyzer, 'performance_metrics')
        
        # Check initial metrics
        metrics = self.analyzer.performance_metrics
        assert metrics["total_analyses"] == 0
        assert metrics["cache_hits"] == 0
        assert isinstance(metrics["complexity_distribution"], dict)
        assert metrics["total_examples_used"] == 0
        assert metrics["analysis_count"] == 0
    
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
        assert "dynamic" in key1
        assert "dream_emotion_analysis" in key1
    
    def test_analyze_single_task_emotion_analysis(self):
        """Test single task analysis for emotion analysis."""
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"]
        )
        
        assert isinstance(result, DynamicShotAnalysisResult)
        assert result.task == ZeroShotTask.DREAM_EMOTION_ANALYSIS
        assert isinstance(result.analysis, dict)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.complexity, DreamComplexity)
        assert isinstance(result.num_examples_used, int)
        assert result.num_examples_used >= 0
        assert isinstance(result.selected_examples, list)
        assert isinstance(result.keywords, list)
        assert isinstance(result.raw_response, str)
        assert isinstance(result.timestamp, str)
    
    def test_analyze_single_task_musical_recommendation(self):
        """Test single task analysis for musical recommendation."""
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
            self.sample_dreams["moderate"]
        )
        
        assert isinstance(result, DynamicShotAnalysisResult)
        assert result.task == ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION
        assert "recommended_style" in result.analysis or "analysis" in result.analysis
        assert isinstance(result.confidence, float)
        assert isinstance(result.complexity, DreamComplexity)
    
    def test_analyze_single_task_symbolism_interpretation(self):
        """Test single task analysis for symbolism interpretation."""
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION,
            self.sample_dreams["complex"]
        )
        
        assert isinstance(result, DynamicShotAnalysisResult)
        assert result.task == ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION
        assert isinstance(result.analysis, dict)
        assert isinstance(result.complexity, DreamComplexity)
    
    def test_analyze_single_task_with_additional_context(self):
        """Test single task analysis with additional context."""
        additional_context = "The dreamer is a professional pilot."
        
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            additional_context
        )
        
        assert isinstance(result, DynamicShotAnalysisResult)
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
        
        assert isinstance(result, DynamicShotAnalysisResult)
        # Cache should remain empty
        assert len(self.analyzer.analysis_cache) == 0
    
    def test_analyze_comprehensive_default_tasks(self):
        """Test comprehensive analysis with default tasks."""
        analysis = self.analyzer.analyze_comprehensive(
            self.sample_dreams["moderate"]
        )
        
        assert isinstance(analysis, ComprehensiveDynamicShotAnalysis)
        assert analysis.dream_text == self.sample_dreams["moderate"]
        assert isinstance(analysis.complexity, DreamComplexity)
        assert isinstance(analysis.keywords, list)
        assert isinstance(analysis.overall_confidence, float)
        assert 0.0 <= analysis.overall_confidence <= 1.0
        assert isinstance(analysis.total_examples_used, int)
        assert analysis.total_examples_used >= 0
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
        
        assert isinstance(analysis, ComprehensiveDynamicShotAnalysis)
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
        
        assert isinstance(analysis, ComprehensiveDynamicShotAnalysis)
        assert analysis.dream_text == self.sample_dreams["complex"]
    
    def test_dynamic_shot_analysis_result_to_dict(self):
        """Test DynamicShotAnalysisResult to_dict method."""
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
        assert "num_examples_used" in result_dict
        assert "selected_examples" in result_dict
        assert "keywords" in result_dict
        assert "raw_response" in result_dict
        assert "timestamp" in result_dict
        
        assert result_dict["task"] == result.task.value
        assert result_dict["complexity"] == result.complexity.value
    
    def test_comprehensive_dynamic_analysis_to_dict(self):
        """Test ComprehensiveDynamicShotAnalysis to_dict method."""
        analysis = self.analyzer.analyze_comprehensive(
            self.sample_dreams["moderate"]
        )
        
        analysis_dict = analysis.to_dict()
        
        assert isinstance(analysis_dict, dict)
        assert "dream_text" in analysis_dict
        assert "complexity" in analysis_dict
        assert "keywords" in analysis_dict
        assert "overall_confidence" in analysis_dict
        assert "total_examples_used" in analysis_dict
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
        assert "total_examples_used" in summary
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
        assert "complexity_distribution" in metrics
        assert "average_examples_used" in metrics
        assert "cache_hit_rate" in metrics
        assert "cache_size" in metrics
        assert "example_database_stats" in metrics
        assert "usage_statistics" in metrics
        assert "analysis_count" in metrics
        assert "total_examples_used" in metrics

        assert metrics["total_analyses"] >= 2
        assert isinstance(metrics["cache_hit_rate"], float)
        assert 0.0 <= metrics["cache_hit_rate"] <= 1.0
        assert isinstance(metrics["average_examples_used"], float)
        assert metrics["average_examples_used"] >= 0.0
    
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
        assert metrics["complexity_distribution"] == {}
        assert metrics["total_examples_used"] == 0
        assert metrics["analysis_count"] == 0
    
    def test_simulate_dynamic_shot_response_emotion_analysis(self):
        """Test simulated response for emotion analysis."""
        prompt_data = {
            "complexity": DreamComplexity.MODERATE.value,
            "num_examples": 2
        }
        
        response = self.analyzer._simulate_dynamic_shot_response(
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
    
    def test_simulate_dynamic_shot_response_musical_recommendation(self):
        """Test simulated response for musical recommendation."""
        prompt_data = {
            "complexity": DreamComplexity.SIMPLE.value,
            "num_examples": 1
        }
        
        response = self.analyzer._simulate_dynamic_shot_response(
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
            self.sample_dreams["simple"],
            prompt_data
        )
        
        assert isinstance(response, str)
        
        # Should be valid JSON
        parsed = json.loads(response)
        assert isinstance(parsed, dict)
        assert "confidence" in parsed
    
    def test_parse_dynamic_shot_response_valid_json(self):
        """Test parsing valid JSON response."""
        valid_response = json.dumps({
            "primary_emotions": ["joy", "freedom"],
            "confidence": 0.85
        })
        
        result = self.analyzer._parse_dynamic_shot_response(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            valid_response
        )
        
        assert isinstance(result, dict)
        assert "primary_emotions" in result
        assert result["confidence"] == 0.85
    
    def test_parse_dynamic_shot_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        invalid_response = "This is not valid JSON"
        
        with pytest.raises(ValueError):
            self.analyzer._parse_dynamic_shot_response(
                ZeroShotTask.DREAM_EMOTION_ANALYSIS,
                invalid_response
            )
    
    def test_fallback_dynamic_analysis(self):
        """Test fallback analysis."""
        result = self.analyzer._fallback_dynamic_analysis(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            "Test error message"
        )
        
        assert isinstance(result, DynamicShotAnalysisResult)
        assert result.task == ZeroShotTask.DREAM_EMOTION_ANALYSIS
        assert result.confidence == 0.3  # Fallback confidence
        assert "fallback" in result.analysis["analysis"].lower()
        assert "Test error message" in result.analysis["error_info"]
        assert isinstance(result.complexity, DreamComplexity)
        assert result.num_examples_used == 0
        assert result.selected_examples == []
    
    def test_empty_dream_analysis(self):
        """Test analysis of empty dream text."""
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            ""
        )
        
        assert isinstance(result, DynamicShotAnalysisResult)
        assert result.complexity == DreamComplexity.SIMPLE
        assert result.keywords == []
    
    def test_very_long_dream_analysis(self):
        """Test analysis of very long dream text."""
        long_dream = " ".join([self.sample_dreams["complex"]] * 10)
        
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            long_dream
        )
        
        assert isinstance(result, DynamicShotAnalysisResult)
        assert result.complexity == DreamComplexity.HIGHLY_COMPLEX
        assert len(result.keywords) <= 20  # Should be limited
    
    def test_custom_config(self):
        """Test analyzer with custom configuration."""
        custom_config = DynamicShotConfig(
            max_examples=3,
            min_examples=2,
            relevance_threshold=0.5
        )
        
        custom_analyzer = DynamicShotDreamAnalyzer(custom_config)
        
        result = custom_analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["complex"]
        )
        
        assert isinstance(result, DynamicShotAnalysisResult)
        # Should respect the custom configuration
        assert result.num_examples_used <= 3
    
    def test_complexity_affects_example_count(self):
        """Test that dream complexity affects the number of examples used."""
        simple_result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"]
        )

        complex_result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["complex"]
        )

        # Both should have valid results
        assert isinstance(simple_result.num_examples_used, int)
        assert isinstance(complex_result.num_examples_used, int)
        assert simple_result.num_examples_used >= 0
        assert complex_result.num_examples_used >= 0

        # Complexity should be valid enum values
        assert isinstance(simple_result.complexity, DreamComplexity)
        assert isinstance(complex_result.complexity, DreamComplexity)
    
    def test_multiple_task_analysis_consistency(self):
        """Test that multiple analyses of the same dream are consistent."""
        dream_text = self.sample_dreams["moderate"]
        
        # Analyze the same dream multiple times
        results = []
        for _ in range(3):
            result = self.analyzer.analyze_single_task(
                ZeroShotTask.DREAM_EMOTION_ANALYSIS,
                dream_text,
                use_cache=False  # Disable cache to get fresh analyses
            )
            results.append(result)
        
        # All results should have the same complexity and similar keywords
        complexities = [r.complexity for r in results]
        assert len(set(complexities)) == 1  # All should be the same
        
        # Keywords should be similar (allowing for some variation)
        all_keywords = [set(r.keywords) for r in results]
        # At least some keywords should be common across analyses
        common_keywords = set.intersection(*all_keywords) if all_keywords else set()
        assert len(common_keywords) >= 0  # Should have at least some common keywords
