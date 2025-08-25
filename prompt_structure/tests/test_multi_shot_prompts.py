"""
Tests for Multi-Shot Prompting functionality.
"""

import pytest
import json
from prompt_structure.multi_shot_prompts import (
    MultiShotPromptBuilder,
    MultiShotStrategy,
    MultiShotConfig
)
from prompt_structure.dynamic_shot_prompts import (
    DreamComplexity,
    ExampleType,
    DreamExample
)
from prompt_structure.zero_shot_prompts import ZeroShotTask


class TestMultiShotPromptBuilder:
    """Test cases for MultiShotPromptBuilder."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = MultiShotPromptBuilder()
        
        # Sample dreams for testing
        self.sample_dreams = {
            "simple": "I was flying over a city, feeling happy.",
            "moderate": "I started in a garden feeling peaceful, but then storm clouds gathered and I became anxious.",
            "complex": "I was in a vast library with floating books. I could read them by thinking, feeling amazed and curious.",
            "symbolic": "I found a golden key that opened doors to rooms filled with memories."
        }
    
    def test_initialization(self):
        """Test builder initialization."""
        assert isinstance(self.builder, MultiShotPromptBuilder)
        assert isinstance(self.builder.config, MultiShotConfig)
        assert len(self.builder.example_database) > 0
        assert ZeroShotTask.DREAM_EMOTION_ANALYSIS in self.builder.example_database
        assert isinstance(self.builder.selection_history, dict)
        assert isinstance(self.builder.diversity_metrics, dict)
    
    def test_initialization_with_custom_config(self):
        """Test builder initialization with custom configuration."""
        custom_config = MultiShotConfig(
            strategy=MultiShotStrategy.QUALITY_RANKED,
            min_examples=3,
            max_examples=6,
            quality_threshold=0.9
        )
        
        custom_builder = MultiShotPromptBuilder(custom_config)
        assert custom_builder.config.strategy == MultiShotStrategy.QUALITY_RANKED
        assert custom_builder.config.min_examples == 3
        assert custom_builder.config.max_examples == 6
        assert custom_builder.config.quality_threshold == 0.9
    
    def test_calculate_example_diversity(self):
        """Test example diversity calculation."""
        examples = self.builder.example_database[ZeroShotTask.DREAM_EMOTION_ANALYSIS][:3]
        
        diversity = self.builder.calculate_example_diversity(examples)
        assert isinstance(diversity, float)
        assert 0.0 <= diversity <= 1.0
        
        # Single example should have lower diversity
        single_diversity = self.builder.calculate_example_diversity([examples[0]])
        assert single_diversity < diversity
        
        # Empty list should have zero diversity
        empty_diversity = self.builder.calculate_example_diversity([])
        assert empty_diversity == 0.0
    
    def test_calculate_example_relevance(self):
        """Test example relevance calculation."""
        example = self.builder.example_database[ZeroShotTask.DREAM_EMOTION_ANALYSIS][0]
        
        relevance = self.builder.calculate_example_relevance(
            example, 
            self.sample_dreams["simple"], 
            ["flying", "city", "happy"]
        )
        
        assert isinstance(relevance, float)
        assert 0.0 <= relevance <= 1.0
    
    def test_calculate_example_quality(self):
        """Test example quality calculation."""
        example = self.builder.example_database[ZeroShotTask.DREAM_EMOTION_ANALYSIS][0]
        
        quality = self.builder.calculate_example_quality(example)
        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0
        assert quality > 0.5  # Should be reasonably high quality
    
    def test_select_multi_shot_examples_diverse_coverage(self):
        """Test multi-shot example selection with diverse coverage strategy."""
        self.builder.config.strategy = MultiShotStrategy.DIVERSE_COVERAGE
        
        examples = self.builder.select_multi_shot_examples(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["moderate"],
            DreamComplexity.MODERATE,
            ["garden", "storm", "peaceful", "anxious"]
        )
        
        assert isinstance(examples, list)
        assert len(examples) >= self.builder.config.min_examples
        assert len(examples) <= self.builder.config.max_examples
        
        # Check diversity
        if len(examples) > 1:
            diversity = self.builder.calculate_example_diversity(examples)
            assert diversity > 0.0
    
    def test_select_multi_shot_examples_complexity_progression(self):
        """Test multi-shot example selection with complexity progression strategy."""
        self.builder.config.strategy = MultiShotStrategy.COMPLEXITY_PROGRESSION
        
        examples = self.builder.select_multi_shot_examples(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["complex"],
            DreamComplexity.COMPLEX,
            ["library", "books", "floating", "thinking"]
        )
        
        assert isinstance(examples, list)
        assert len(examples) >= self.builder.config.min_examples
        assert len(examples) <= self.builder.config.max_examples
        
        # Should include examples of different complexities
        if len(examples) > 1:
            complexities = [ex.complexity for ex in examples]
            assert len(set(complexities)) > 1  # Should have variety
    
    def test_select_multi_shot_examples_thematic_clustering(self):
        """Test multi-shot example selection with thematic clustering strategy."""
        self.builder.config.strategy = MultiShotStrategy.THEMATIC_CLUSTERING
        
        examples = self.builder.select_multi_shot_examples(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            DreamComplexity.SIMPLE,
            ["flying", "city", "happy"]
        )
        
        assert isinstance(examples, list)
        assert len(examples) >= self.builder.config.min_examples
        assert len(examples) <= self.builder.config.max_examples
    
    def test_select_multi_shot_examples_balanced_representation(self):
        """Test multi-shot example selection with balanced representation strategy."""
        self.builder.config.strategy = MultiShotStrategy.BALANCED_REPRESENTATION
        
        examples = self.builder.select_multi_shot_examples(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["moderate"],
            DreamComplexity.MODERATE,
            ["garden", "storm", "peaceful", "anxious"]
        )
        
        assert isinstance(examples, list)
        assert len(examples) >= self.builder.config.min_examples
        assert len(examples) <= self.builder.config.max_examples
        
        # Should have balanced representation across types if possible
        if len(examples) > 1:
            types = [ex.example_type for ex in examples]
            # At least some variety in types (if available)
            assert len(set(types)) >= 1
    
    def test_select_multi_shot_examples_quality_ranked(self):
        """Test multi-shot example selection with quality ranked strategy."""
        self.builder.config.strategy = MultiShotStrategy.QUALITY_RANKED
        
        examples = self.builder.select_multi_shot_examples(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            DreamComplexity.SIMPLE,
            ["flying", "city", "happy"]
        )
        
        assert isinstance(examples, list)
        assert len(examples) >= self.builder.config.min_examples
        assert len(examples) <= self.builder.config.max_examples
        
        # Should be sorted by quality (highest first)
        if len(examples) > 1:
            qualities = [self.builder.calculate_example_quality(ex) for ex in examples]
            assert qualities == sorted(qualities, reverse=True)
    
    def test_select_multi_shot_examples_contextual_similarity(self):
        """Test multi-shot example selection with contextual similarity strategy."""
        self.builder.config.strategy = MultiShotStrategy.CONTEXTUAL_SIMILARITY
        
        examples = self.builder.select_multi_shot_examples(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            DreamComplexity.SIMPLE,
            ["flying", "city", "happy"]
        )
        
        assert isinstance(examples, list)
        assert len(examples) >= self.builder.config.min_examples
        assert len(examples) <= self.builder.config.max_examples
    
    def test_select_multi_shot_examples_no_examples(self):
        """Test multi-shot example selection when no examples available."""
        # Create a mock task that doesn't exist in the database
        class MockTask:
            value = "nonexistent_task"
        
        mock_task = MockTask()
        
        examples = self.builder.select_multi_shot_examples(
            mock_task,
            self.sample_dreams["simple"],
            DreamComplexity.SIMPLE,
            ["test"]
        )
        
        assert examples == []
    
    def test_build_multi_shot_prompt_emotion_analysis(self):
        """Test building multi-shot prompt for emotion analysis."""
        prompt = self.builder.build_multi_shot_prompt(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["moderate"]
        )
        
        assert isinstance(prompt, dict)
        assert "system_message" in prompt
        assert "user_message" in prompt
        assert "task" in prompt
        assert "complexity" in prompt
        assert "strategy" in prompt
        assert "num_examples" in prompt
        assert "selected_examples" in prompt
        assert "diversity_score" in prompt
        assert "average_quality" in prompt
        assert "keywords" in prompt
        
        assert prompt["task"] == ZeroShotTask.DREAM_EMOTION_ANALYSIS.value
        assert prompt["strategy"] == self.builder.config.strategy.value
        assert isinstance(prompt["num_examples"], int)
        assert prompt["num_examples"] >= 0
        assert isinstance(prompt["diversity_score"], float)
        assert isinstance(prompt["average_quality"], float)
    
    def test_build_multi_shot_prompt_musical_recommendation(self):
        """Test building multi-shot prompt for musical recommendation."""
        prompt = self.builder.build_multi_shot_prompt(
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
            self.sample_dreams["complex"]
        )
        
        assert isinstance(prompt, dict)
        assert prompt["task"] == ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION.value
        assert "system_message" in prompt
        assert "user_message" in prompt
    
    def test_build_multi_shot_prompt_with_context(self):
        """Test building multi-shot prompt with additional context."""
        additional_context = "The dreamer is a professional musician with classical training."
        
        prompt = self.builder.build_multi_shot_prompt(
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
            self.sample_dreams["simple"],
            additional_context
        )
        
        assert isinstance(prompt, dict)
        assert additional_context in prompt["user_message"]
    
    def test_build_multi_shot_system_message(self):
        """Test system message building."""
        examples = self.builder.example_database[ZeroShotTask.DREAM_EMOTION_ANALYSIS][:2]
        
        system_message = self.builder._build_multi_shot_system_message(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            DreamComplexity.MODERATE,
            examples
        )
        
        assert isinstance(system_message, str)
        assert "expert" in system_message.lower()
        assert "COMPLEXITY LEVEL: MODERATE" in system_message
        assert f"{len(examples)} high-quality examples" in system_message
    
    def test_build_multi_shot_system_message_no_examples(self):
        """Test system message building with no examples."""
        system_message = self.builder._build_multi_shot_system_message(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            DreamComplexity.SIMPLE,
            []
        )
        
        assert isinstance(system_message, str)
        assert "No examples are available" in system_message
    
    def test_build_multi_shot_user_message(self):
        """Test user message building."""
        examples = self.builder.example_database[ZeroShotTask.DREAM_EMOTION_ANALYSIS][:2]
        
        user_message = self.builder._build_multi_shot_user_message(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["moderate"],
            examples
        )
        
        assert isinstance(user_message, str)
        assert f"{len(examples)} high-quality examples" in user_message
        assert "EXAMPLE 1:" in user_message
        assert "EXAMPLE 2:" in user_message
        assert "DREAM TO ANALYZE:" in user_message
        assert self.sample_dreams["moderate"] in user_message
    
    def test_build_multi_shot_user_message_no_examples(self):
        """Test user message building with no examples."""
        user_message = self.builder._build_multi_shot_user_message(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            []
        )
        
        assert isinstance(user_message, str)
        assert "EXAMPLE" not in user_message
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
            self.builder.build_multi_shot_prompt(
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
        assert "diversity_metrics" in stats
        
        assert stats["total_examples"] > 0
        assert isinstance(stats["examples_by_task"], dict)
        assert isinstance(stats["quality_distribution"], dict)
        assert isinstance(stats["complexity_distribution"], dict)
        assert isinstance(stats["example_types"], dict)
        assert isinstance(stats["diversity_metrics"], dict)
    
    def test_add_example(self):
        """Test adding new examples to the database."""
        initial_count = len(self.builder.example_database[ZeroShotTask.DREAM_EMOTION_ANALYSIS])
        
        new_example = DreamExample(
            dream_text="I was dancing in the rain, feeling completely free and alive.",
            analysis={
                "primary_emotions": ["joy", "freedom", "vitality"],
                "confidence": 0.88
            },
            example_type=ExampleType.BASIC_EMOTION,
            complexity=DreamComplexity.SIMPLE,
            keywords=["dancing", "rain", "free", "alive"]
        )
        
        self.builder.add_example(ZeroShotTask.DREAM_EMOTION_ANALYSIS, new_example)
        
        final_count = len(self.builder.example_database[ZeroShotTask.DREAM_EMOTION_ANALYSIS])
        assert final_count == initial_count + 1
    
    def test_clear_selection_history(self):
        """Test clearing selection history."""
        # Generate some selection history
        self.builder.build_multi_shot_prompt(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"]
        )
        
        assert len(self.builder.selection_history) > 0
        
        # Clear history
        self.builder.clear_selection_history()
        assert len(self.builder.selection_history) == 0
    
    def test_get_strategy_comparison(self):
        """Test strategy comparison functionality."""
        comparison = self.builder.get_strategy_comparison(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["moderate"]
        )
        
        assert isinstance(comparison, dict)
        assert "dream_text" in comparison
        assert "complexity" in comparison
        assert "keywords" in comparison
        assert "strategy_selections" in comparison
        
        # Should have entries for all strategies
        strategy_selections = comparison["strategy_selections"]
        assert len(strategy_selections) == len(MultiShotStrategy)
        
        for strategy in MultiShotStrategy:
            assert strategy.value in strategy_selections
            selection = strategy_selections[strategy.value]
            assert "num_examples" in selection
    
    def test_all_strategies_work(self):
        """Test that all selection strategies work without errors."""
        for strategy in MultiShotStrategy:
            self.builder.config.strategy = strategy
            
            examples = self.builder.select_multi_shot_examples(
                ZeroShotTask.DREAM_EMOTION_ANALYSIS,
                self.sample_dreams["moderate"],
                DreamComplexity.MODERATE,
                ["garden", "storm", "peaceful", "anxious"]
            )
            
            assert isinstance(examples, list)
            assert len(examples) >= 0  # Could be empty for some strategies
    
    def test_quality_threshold_filtering(self):
        """Test that quality threshold filtering works."""
        # Set very high quality threshold
        self.builder.config.quality_threshold = 0.99
        self.builder.config.fallback_to_available = False
        
        examples = self.builder.select_multi_shot_examples(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            DreamComplexity.SIMPLE,
            ["flying", "city", "happy"]
        )
        
        # Might be empty if no examples meet the high threshold
        for example in examples:
            quality = self.builder.calculate_example_quality(example)
            assert quality >= 0.99
    
    def test_fallback_to_available(self):
        """Test fallback to available when quality threshold not met."""
        # Set very high quality threshold but enable fallback
        self.builder.config.quality_threshold = 0.99
        self.builder.config.fallback_to_available = True
        
        examples = self.builder.select_multi_shot_examples(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            DreamComplexity.SIMPLE,
            ["flying", "city", "happy"]
        )
        
        # Should still get examples due to fallback
        assert len(examples) >= self.builder.config.min_examples
    
    def test_min_max_examples_constraints(self):
        """Test that min/max examples constraints are respected."""
        # Test with custom constraints
        self.builder.config.min_examples = 2
        self.builder.config.max_examples = 4
        
        examples = self.builder.select_multi_shot_examples(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["moderate"],
            DreamComplexity.MODERATE,
            ["garden", "storm", "peaceful", "anxious"]
        )
        
        assert len(examples) >= 2
        assert len(examples) <= 4
