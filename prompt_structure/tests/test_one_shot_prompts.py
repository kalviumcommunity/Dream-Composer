"""
Tests for One-Shot Prompting functionality.
"""

import pytest
import json
from prompt_structure.one_shot_prompts import (
    OneShotPromptBuilder,
    OneShotStrategy,
    OneShotConfig
)
from prompt_structure.dynamic_shot_prompts import (
    DreamComplexity,
    ExampleType,
    DreamExample
)
from prompt_structure.zero_shot_prompts import ZeroShotTask


class TestOneShotPromptBuilder:
    """Test cases for OneShotPromptBuilder."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = OneShotPromptBuilder()
        
        # Sample dreams for testing
        self.sample_dreams = {
            "simple": "I was flying over a city, feeling happy.",
            "moderate": "I started in a garden feeling peaceful, but then storm clouds gathered and I became anxious.",
            "complex": "I was in a vast library with floating books. I could read them by thinking, feeling amazed and curious.",
            "symbolic": "I found a golden key that opened doors to rooms filled with memories."
        }
    
    def test_initialization(self):
        """Test builder initialization."""
        assert isinstance(self.builder, OneShotPromptBuilder)
        assert isinstance(self.builder.config, OneShotConfig)
        assert len(self.builder.example_database) > 0
        assert ZeroShotTask.DREAM_EMOTION_ANALYSIS in self.builder.example_database
        assert isinstance(self.builder.selection_history, dict)
    
    def test_initialization_with_custom_config(self):
        """Test builder initialization with custom configuration."""
        custom_config = OneShotConfig(
            strategy=OneShotStrategy.REPRESENTATIVE,
            quality_threshold=0.9,
            relevance_weight=0.7
        )
        
        custom_builder = OneShotPromptBuilder(custom_config)
        assert custom_builder.config.strategy == OneShotStrategy.REPRESENTATIVE
        assert custom_builder.config.quality_threshold == 0.9
        assert custom_builder.config.relevance_weight == 0.7
    
    def test_calculate_example_quality(self):
        """Test example quality calculation."""
        high_quality_example = DreamExample(
            dream_text="I was flying over a beautiful city at sunset, feeling incredibly free and joyful.",
            analysis={
                "primary_emotions": ["joy", "freedom"],
                "emotion_intensities": {"joy": 9, "freedom": 8},
                "dominant_mood": "euphoric",
                "confidence": 0.92
            },
            example_type=ExampleType.BASIC_EMOTION,
            complexity=DreamComplexity.SIMPLE,
            keywords=["flying", "city", "sunset", "free", "joyful", "beautiful"]
        )
        
        quality = self.builder.calculate_example_quality(high_quality_example)
        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0
        assert quality > 0.8  # Should be high quality
    
    def test_calculate_example_quality_low_quality(self):
        """Test example quality calculation for low quality example."""
        low_quality_example = DreamExample(
            dream_text="Dream.",
            analysis={"confidence": 0.3},
            example_type=ExampleType.BASIC_EMOTION,
            complexity=DreamComplexity.SIMPLE,
            keywords=["dream"]
        )
        
        quality = self.builder.calculate_example_quality(low_quality_example)
        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0
        assert quality < 0.5  # Should be low quality
    
    def test_calculate_representativeness(self):
        """Test representativeness calculation."""
        example = DreamExample(
            dream_text="I was flying over a city, feeling happy.",
            analysis={"confidence": 0.85},
            example_type=ExampleType.BASIC_EMOTION,
            complexity=DreamComplexity.MODERATE,
            keywords=["flying", "city", "happy"]
        )
        
        representativeness = self.builder.calculate_representativeness(
            example, ZeroShotTask.DREAM_EMOTION_ANALYSIS
        )
        
        assert isinstance(representativeness, float)
        assert 0.0 <= representativeness <= 1.0
    
    def test_select_one_shot_example_best_match(self):
        """Test one-shot example selection with best match strategy."""
        self.builder.config.strategy = OneShotStrategy.BEST_MATCH
        
        example = self.builder.select_one_shot_example(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            DreamComplexity.SIMPLE,
            ["flying", "city", "happy"]
        )
        
        assert example is not None
        assert isinstance(example, DreamExample)
    
    def test_select_one_shot_example_representative(self):
        """Test one-shot example selection with representative strategy."""
        self.builder.config.strategy = OneShotStrategy.REPRESENTATIVE
        
        example = self.builder.select_one_shot_example(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["moderate"],
            DreamComplexity.MODERATE,
            ["garden", "storm", "peaceful", "anxious"]
        )
        
        assert example is not None
        assert isinstance(example, DreamExample)
    
    def test_select_one_shot_example_complexity_matched(self):
        """Test one-shot example selection with complexity matched strategy."""
        self.builder.config.strategy = OneShotStrategy.COMPLEXITY_MATCHED
        
        example = self.builder.select_one_shot_example(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["complex"],
            DreamComplexity.COMPLEX,
            ["library", "books", "floating", "thinking"]
        )
        
        assert example is not None
        assert isinstance(example, DreamExample)
    
    def test_select_one_shot_example_balanced(self):
        """Test one-shot example selection with balanced strategy."""
        self.builder.config.strategy = OneShotStrategy.BALANCED
        
        example = self.builder.select_one_shot_example(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            DreamComplexity.SIMPLE,
            ["flying", "city", "happy"]
        )
        
        assert example is not None
        assert isinstance(example, DreamExample)
    
    def test_select_one_shot_example_random_quality(self):
        """Test one-shot example selection with random quality strategy."""
        self.builder.config.strategy = OneShotStrategy.RANDOM_QUALITY
        
        example = self.builder.select_one_shot_example(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            DreamComplexity.SIMPLE,
            ["flying", "city", "happy"]
        )
        
        assert example is not None
        assert isinstance(example, DreamExample)
    
    def test_select_one_shot_example_no_examples(self):
        """Test one-shot example selection when no examples available."""
        # Create a mock task that doesn't exist in the database
        class MockTask:
            value = "nonexistent_task"
        
        mock_task = MockTask()
        
        example = self.builder.select_one_shot_example(
            mock_task,
            self.sample_dreams["simple"],
            DreamComplexity.SIMPLE,
            ["test"]
        )
        
        assert example is None
    
    def test_build_one_shot_prompt_emotion_analysis(self):
        """Test building one-shot prompt for emotion analysis."""
        prompt = self.builder.build_one_shot_prompt(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"]
        )
        
        assert isinstance(prompt, dict)
        assert "system_message" in prompt
        assert "user_message" in prompt
        assert "task" in prompt
        assert "complexity" in prompt
        assert "strategy" in prompt
        assert "selected_example" in prompt
        assert "example_quality" in prompt
        assert "keywords" in prompt
        
        assert prompt["task"] == ZeroShotTask.DREAM_EMOTION_ANALYSIS.value
        assert prompt["strategy"] == self.builder.config.strategy.value
        assert isinstance(prompt["example_quality"], float)
    
    def test_build_one_shot_prompt_musical_recommendation(self):
        """Test building one-shot prompt for musical recommendation."""
        prompt = self.builder.build_one_shot_prompt(
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
            self.sample_dreams["moderate"]
        )
        
        assert isinstance(prompt, dict)
        assert prompt["task"] == ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION.value
        assert "system_message" in prompt
        assert "user_message" in prompt
    
    def test_build_one_shot_prompt_with_context(self):
        """Test building one-shot prompt with additional context."""
        additional_context = "The dreamer is a professional musician."
        
        prompt = self.builder.build_one_shot_prompt(
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
            self.sample_dreams["simple"],
            additional_context
        )
        
        assert isinstance(prompt, dict)
        assert additional_context in prompt["user_message"]
    
    def test_build_one_shot_system_message(self):
        """Test system message building."""
        example = self.builder.example_database[ZeroShotTask.DREAM_EMOTION_ANALYSIS][0]
        
        system_message = self.builder._build_one_shot_system_message(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            DreamComplexity.SIMPLE,
            example
        )
        
        assert isinstance(system_message, str)
        assert "expert" in system_message.lower()
        assert "COMPLEXITY LEVEL: SIMPLE" in system_message
        assert "one high-quality example" in system_message.lower()
    
    def test_build_one_shot_system_message_no_example(self):
        """Test system message building with no example."""
        system_message = self.builder._build_one_shot_system_message(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            DreamComplexity.SIMPLE,
            None
        )
        
        assert isinstance(system_message, str)
        assert "No example is available" in system_message
    
    def test_build_one_shot_user_message(self):
        """Test user message building."""
        example = self.builder.example_database[ZeroShotTask.DREAM_EMOTION_ANALYSIS][0]
        
        user_message = self.builder._build_one_shot_user_message(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            example
        )
        
        assert isinstance(user_message, str)
        assert "EXAMPLE:" in user_message
        assert "DREAM TO ANALYZE:" in user_message
        assert self.sample_dreams["simple"] in user_message
        assert example.dream_text in user_message
    
    def test_build_one_shot_user_message_no_example(self):
        """Test user message building with no example."""
        user_message = self.builder._build_one_shot_user_message(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            None
        )
        
        assert isinstance(user_message, str)
        assert "EXAMPLE:" not in user_message
        assert "DREAM TO ANALYZE:" in user_message
        assert self.sample_dreams["simple"] in user_message
    
    def test_get_selection_statistics(self):
        """Test getting selection statistics."""
        # Initially should have empty statistics
        stats = self.builder.get_selection_statistics()
        assert stats["total_selections"] == 0
        assert stats["strategy_usage"] == {}
        assert stats["most_used_strategies"] == []
        
        # Perform some selections to generate statistics
        for _ in range(3):
            self.builder.build_one_shot_prompt(
                ZeroShotTask.DREAM_EMOTION_ANALYSIS,
                self.sample_dreams["simple"]
            )
        
        # Get updated statistics
        stats = self.builder.get_selection_statistics()
        assert stats["total_selections"] > 0
        assert len(stats["strategy_usage"]) > 0
    
    def test_get_example_statistics(self):
        """Test getting example database statistics."""
        stats = self.builder.get_example_statistics()
        
        assert isinstance(stats, dict)
        assert "total_examples" in stats
        assert "examples_by_task" in stats
        assert "quality_distribution" in stats
        assert "complexity_distribution" in stats
        assert "example_types" in stats
        
        assert stats["total_examples"] > 0
        assert isinstance(stats["examples_by_task"], dict)
        assert isinstance(stats["quality_distribution"], dict)
        assert isinstance(stats["complexity_distribution"], dict)
        assert isinstance(stats["example_types"], dict)
    
    def test_add_example(self):
        """Test adding new examples to the database."""
        initial_count = len(self.builder.example_database[ZeroShotTask.DREAM_EMOTION_ANALYSIS])
        
        new_example = DreamExample(
            dream_text="I was dancing in the rain, feeling completely free.",
            analysis={
                "primary_emotions": ["joy", "freedom"],
                "confidence": 0.88
            },
            example_type=ExampleType.BASIC_EMOTION,
            complexity=DreamComplexity.SIMPLE,
            keywords=["dancing", "rain", "free"]
        )
        
        self.builder.add_example(ZeroShotTask.DREAM_EMOTION_ANALYSIS, new_example)
        
        final_count = len(self.builder.example_database[ZeroShotTask.DREAM_EMOTION_ANALYSIS])
        assert final_count == initial_count + 1
    
    def test_add_low_quality_example(self):
        """Test adding low quality example (should warn but still add)."""
        low_quality_example = DreamExample(
            dream_text="Bad dream.",
            analysis={"confidence": 0.2},
            example_type=ExampleType.BASIC_EMOTION,
            complexity=DreamComplexity.SIMPLE,
            keywords=["bad"]
        )
        
        initial_count = len(self.builder.example_database[ZeroShotTask.DREAM_EMOTION_ANALYSIS])
        self.builder.add_example(ZeroShotTask.DREAM_EMOTION_ANALYSIS, low_quality_example)
        final_count = len(self.builder.example_database[ZeroShotTask.DREAM_EMOTION_ANALYSIS])
        
        assert final_count == initial_count + 1
    
    def test_clear_selection_history(self):
        """Test clearing selection history."""
        # Generate some selection history
        self.builder.build_one_shot_prompt(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"]
        )
        
        assert len(self.builder.selection_history) > 0
        
        # Clear history
        self.builder.clear_selection_history()
        assert len(self.builder.selection_history) == 0
    
    def test_all_strategies_work(self):
        """Test that all selection strategies work without errors."""
        for strategy in OneShotStrategy:
            self.builder.config.strategy = strategy
            
            example = self.builder.select_one_shot_example(
                ZeroShotTask.DREAM_EMOTION_ANALYSIS,
                self.sample_dreams["simple"],
                DreamComplexity.SIMPLE,
                ["flying", "city", "happy"]
            )
            
            assert example is not None
            assert isinstance(example, DreamExample)
    
    def test_quality_threshold_filtering(self):
        """Test that quality threshold filtering works."""
        # Set very high quality threshold
        self.builder.config.quality_threshold = 0.99
        self.builder.config.fallback_to_representative = False
        
        example = self.builder.select_one_shot_example(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            DreamComplexity.SIMPLE,
            ["flying", "city", "happy"]
        )
        
        # Might be None if no examples meet the high threshold
        if example is not None:
            quality = self.builder.calculate_example_quality(example)
            assert quality >= 0.99
    
    def test_fallback_to_representative(self):
        """Test fallback to representative when quality threshold not met."""
        # Set very high quality threshold but enable fallback
        self.builder.config.quality_threshold = 0.99
        self.builder.config.fallback_to_representative = True
        
        example = self.builder.select_one_shot_example(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            DreamComplexity.SIMPLE,
            ["flying", "city", "happy"]
        )
        
        # Should still get an example due to fallback
        assert example is not None
        assert isinstance(example, DreamExample)
