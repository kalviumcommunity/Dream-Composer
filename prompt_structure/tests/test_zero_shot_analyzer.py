"""
Tests for the Zero-Shot Analyzer module.
"""

import pytest
import json
from prompt_structure.zero_shot_analyzer import (
    ZeroShotDreamAnalyzer, 
    ZeroShotAnalysisResult, 
    ComprehensiveZeroShotAnalysis
)
from prompt_structure.zero_shot_prompts import ZeroShotTask


class TestZeroShotDreamAnalyzer:
    """Test cases for ZeroShotDreamAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ZeroShotDreamAnalyzer()
        self.sample_dreams = {
            "flying": "I was flying over a beautiful city at sunset, feeling free and peaceful.",
            "water": "I found myself swimming in crystal clear water, feeling calm and refreshed.",
            "nightmare": "I was being chased through dark corridors, feeling terrified and trapped.",
            "nature": "I walked through a golden forest with sunlight filtering through leaves.",
            "complex": "I started in a dark cave, then flew out into bright sunlight over an ocean."
        }
    
    def test_initialization(self):
        """Test ZeroShotDreamAnalyzer initialization."""
        assert isinstance(self.analyzer, ZeroShotDreamAnalyzer)
        assert hasattr(self.analyzer, 'prompt_builder')
        assert hasattr(self.analyzer, 'analysis_cache')
        assert len(self.analyzer.analysis_cache) == 0
    
    def test_analyze_single_task_emotion_analysis(self):
        """Test single task analysis for emotion analysis."""
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["flying"]
        )
        
        assert isinstance(result, ZeroShotAnalysisResult)
        assert result.task == ZeroShotTask.DREAM_EMOTION_ANALYSIS
        assert isinstance(result.analysis, dict)
        assert 0 <= result.confidence <= 1
        assert result.timestamp
        assert result.raw_response
        
        # Check analysis content
        analysis = result.analysis
        assert "primary_emotions" in analysis
        assert "confidence" in analysis
        assert isinstance(analysis["primary_emotions"], list)
    
    def test_analyze_single_task_musical_recommendation(self):
        """Test single task analysis for musical recommendation."""
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
            self.sample_dreams["flying"]
        )
        
        assert isinstance(result, ZeroShotAnalysisResult)
        assert result.task == ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION
        
        analysis = result.analysis
        assert "recommended_style" in analysis
        assert "tempo" in analysis
        assert "key_signature" in analysis
        assert "instruments" in analysis
        assert "confidence" in analysis
    
    def test_analyze_single_task_symbolism_interpretation(self):
        """Test single task analysis for symbolism interpretation."""
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION,
            self.sample_dreams["flying"]
        )
        
        assert isinstance(result, ZeroShotAnalysisResult)
        assert result.task == ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION
        
        analysis = result.analysis
        assert "symbols" in analysis
        assert "confidence" in analysis
        assert isinstance(analysis["symbols"], list)
    
    def test_analyze_single_task_with_additional_context(self):
        """Test single task analysis with additional context."""
        additional_context = "The dreamer is a professional musician."
        
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["flying"],
            additional_context=additional_context
        )
        
        assert isinstance(result, ZeroShotAnalysisResult)
        assert result.confidence > 0
    
    def test_analyze_single_task_caching(self):
        """Test that caching works for single task analysis."""
        # First analysis
        result1 = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["flying"],
            use_cache=True
        )
        
        # Second analysis should use cache
        result2 = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["flying"],
            use_cache=True
        )
        
        # Should be the same object (from cache)
        assert result1.timestamp == result2.timestamp
        assert len(self.analyzer.analysis_cache) > 0
    
    def test_analyze_single_task_no_caching(self):
        """Test single task analysis without caching."""
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["flying"],
            use_cache=False
        )
        
        assert isinstance(result, ZeroShotAnalysisResult)
        # Cache should remain empty
        assert len(self.analyzer.analysis_cache) == 0
    
    def test_analyze_comprehensive_default_tasks(self):
        """Test comprehensive analysis with default tasks."""
        analysis = self.analyzer.analyze_comprehensive(self.sample_dreams["flying"])
        
        assert isinstance(analysis, ComprehensiveZeroShotAnalysis)
        assert analysis.dream_text == self.sample_dreams["flying"]
        assert 0 <= analysis.overall_confidence <= 1
        assert analysis.analysis_timestamp
        
        # Should have multiple analysis results
        assert analysis.emotion_analysis is not None
        assert analysis.musical_recommendation is not None
        assert analysis.symbolism_interpretation is not None
        assert analysis.mood_mapping is not None
        assert analysis.narrative_analysis is not None
    
    def test_analyze_comprehensive_specific_tasks(self):
        """Test comprehensive analysis with specific tasks."""
        tasks = [
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION
        ]
        
        analysis = self.analyzer.analyze_comprehensive(
            self.sample_dreams["flying"],
            tasks=tasks
        )
        
        assert isinstance(analysis, ComprehensiveZeroShotAnalysis)
        assert analysis.emotion_analysis is not None
        assert analysis.musical_recommendation is not None
        # These should be None since not requested
        assert analysis.symbolism_interpretation is None
        assert analysis.mood_mapping is None
        assert analysis.narrative_analysis is None
    
    def test_analyze_comprehensive_with_context(self):
        """Test comprehensive analysis with additional context."""
        additional_context = "The dreamer is experiencing a major life transition."
        
        analysis = self.analyzer.analyze_comprehensive(
            self.sample_dreams["complex"],
            additional_context=additional_context
        )
        
        assert isinstance(analysis, ComprehensiveZeroShotAnalysis)
        assert analysis.overall_confidence > 0
    
    def test_comprehensive_analysis_to_dict(self):
        """Test ComprehensiveZeroShotAnalysis to_dict conversion."""
        analysis = self.analyzer.analyze_comprehensive(self.sample_dreams["flying"])
        analysis_dict = analysis.to_dict()
        
        assert isinstance(analysis_dict, dict)
        assert "dream_text" in analysis_dict
        assert "emotion_analysis" in analysis_dict
        assert "musical_recommendation" in analysis_dict
        assert "symbolism_interpretation" in analysis_dict
        assert "overall_confidence" in analysis_dict
        assert "analysis_timestamp" in analysis_dict
        
        # Check nested structure
        if analysis_dict["emotion_analysis"]:
            assert "task" in analysis_dict["emotion_analysis"]
            assert "analysis" in analysis_dict["emotion_analysis"]
            assert "confidence" in analysis_dict["emotion_analysis"]
    
    def test_zero_shot_analysis_result_to_dict(self):
        """Test ZeroShotAnalysisResult to_dict conversion."""
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["flying"]
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "task" in result_dict
        assert "analysis" in result_dict
        assert "confidence" in result_dict
        assert "timestamp" in result_dict
        
        assert result_dict["task"] == "dream_emotion_analysis"
        assert isinstance(result_dict["analysis"], dict)
        assert 0 <= result_dict["confidence"] <= 1
    
    def test_get_analysis_summary(self):
        """Test analysis summary generation."""
        analysis = self.analyzer.analyze_comprehensive(self.sample_dreams["flying"])
        summary = self.analyzer.get_analysis_summary(analysis)
        
        assert isinstance(summary, dict)
        assert "dream_text" in summary
        assert "overall_confidence" in summary
        assert "analysis_timestamp" in summary
        
        # Should include extracted information
        if "primary_emotions" in summary:
            assert isinstance(summary["primary_emotions"], list)
        if "recommended_style" in summary:
            assert isinstance(summary["recommended_style"], str)
        if "key_symbols" in summary:
            assert isinstance(summary["key_symbols"], list)
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        # Add something to cache
        self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["flying"],
            use_cache=True
        )
        
        assert len(self.analyzer.analysis_cache) > 0
        
        # Clear cache
        self.analyzer.clear_cache()
        
        assert len(self.analyzer.analysis_cache) == 0
    
    def test_get_cache_stats(self):
        """Test cache statistics functionality."""
        # Add multiple analyses to cache
        self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["flying"],
            use_cache=True
        )
        self.analyzer.analyze_single_task(
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
            self.sample_dreams["water"],
            use_cache=True
        )
        
        stats = self.analyzer.get_cache_stats()
        
        assert isinstance(stats, dict)
        assert "cache_size" in stats
        assert "cached_tasks" in stats
        
        assert stats["cache_size"] >= 2
        assert isinstance(stats["cached_tasks"], list)
    
    def test_simulate_zero_shot_response_flying_dream(self):
        """Test simulated response for flying dream."""
        response = self.analyzer._simulate_zero_shot_response(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["flying"]
        )
        
        assert isinstance(response, str)
        
        # Should be valid JSON
        parsed = json.loads(response)
        assert isinstance(parsed, dict)
        assert "confidence" in parsed
        
        # Should detect appropriate emotions for flying dream
        if "primary_emotions" in parsed:
            emotions = parsed["primary_emotions"]
            assert any(emotion in ["freedom", "joy", "peace"] for emotion in emotions)
    
    def test_simulate_zero_shot_response_nightmare(self):
        """Test simulated response for nightmare."""
        response = self.analyzer._simulate_zero_shot_response(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["nightmare"]
        )
        
        parsed = json.loads(response)
        
        # Should detect fear-related emotions
        if "primary_emotions" in parsed:
            emotions = parsed["primary_emotions"]
            assert any(emotion in ["fear", "anxiety"] for emotion in emotions)
    
    def test_simulate_musical_recommendation_response(self):
        """Test simulated musical recommendation response."""
        response = self.analyzer._simulate_zero_shot_response(
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
            self.sample_dreams["flying"]
        )
        
        parsed = json.loads(response)
        
        assert "recommended_style" in parsed
        assert "tempo" in parsed
        assert "key_signature" in parsed
        assert "instruments" in parsed
        assert "confidence" in parsed
        
        # Check structure of nested objects
        assert isinstance(parsed["tempo"], dict)
        assert "bpm" in parsed["tempo"]
        assert isinstance(parsed["key_signature"], dict)
        assert "key" in parsed["key_signature"]
    
    def test_parse_zero_shot_response_valid_json(self):
        """Test parsing valid JSON response."""
        valid_response = json.dumps({
            "primary_emotions": ["joy", "freedom"],
            "confidence": 0.85
        })
        
        result = self.analyzer._parse_zero_shot_response(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            valid_response
        )
        
        assert isinstance(result, dict)
        assert "primary_emotions" in result
        assert result["primary_emotions"] == ["joy", "freedom"]
    
    def test_parse_zero_shot_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        invalid_response = "This is not valid JSON"
        
        with pytest.raises(ValueError, match="Could not parse zero-shot response"):
            self.analyzer._parse_zero_shot_response(
                ZeroShotTask.DREAM_EMOTION_ANALYSIS,
                invalid_response
            )
    
    def test_fallback_analysis(self):
        """Test fallback analysis functionality."""
        result = self.analyzer._fallback_analysis(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["flying"]
        )
        
        assert isinstance(result, ZeroShotAnalysisResult)
        assert result.task == ZeroShotTask.DREAM_EMOTION_ANALYSIS
        assert result.confidence == 0.2  # Low confidence for fallback
        assert result.raw_response == "fallback_analysis"
        assert "fallback" in result.analysis["analysis"].lower()
    
    def test_empty_dream_analysis(self):
        """Test analysis of empty dream text."""
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            ""
        )
        
        assert isinstance(result, ZeroShotAnalysisResult)
        assert result.confidence >= 0
    
    def test_very_long_dream_analysis(self):
        """Test analysis of very long dream text."""
        long_dream = "I was flying through the sky. " * 200
        
        result = self.analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            long_dream
        )
        
        assert isinstance(result, ZeroShotAnalysisResult)
        assert result.confidence >= 0
    
    def test_multiple_task_analysis_consistency(self):
        """Test that multiple analyses of the same dream are consistent."""
        dream_text = self.sample_dreams["flying"]
        
        # Analyze the same dream multiple times
        results = []
        for _ in range(3):
            result = self.analyzer.analyze_single_task(
                ZeroShotTask.DREAM_EMOTION_ANALYSIS,
                dream_text,
                use_cache=False  # Disable cache to get fresh analyses
            )
            results.append(result)
        
        # All should be valid results
        for result in results:
            assert isinstance(result, ZeroShotAnalysisResult)
            assert result.confidence > 0
        
        # Should have some consistency in detected emotions
        all_emotions = []
        for result in results:
            if "primary_emotions" in result.analysis:
                all_emotions.extend(result.analysis["primary_emotions"])
        
        # Should have some overlap in emotions detected
        unique_emotions = set(all_emotions)
        assert len(unique_emotions) <= len(all_emotions)  # Some repetition expected
