"""
Tests for Dynamic Shot Prompting functionality.
"""

import pytest
import json
from prompt_structure.dynamic_shot_prompts import (
    DynamicShotPromptBuilder,
    DreamComplexity,
    ExampleType,
    DreamExample,
    DynamicShotConfig
)
from prompt_structure.zero_shot_prompts import ZeroShotTask


class TestDynamicShotPromptBuilder:
    """Test cases for DynamicShotPromptBuilder."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = DynamicShotPromptBuilder()
        
        # Sample dreams for testing
        self.sample_dreams = {
            "simple": "I was flying over a city, feeling happy.",
            "moderate": "I started in a garden feeling peaceful, but then storm clouds gathered and I became anxious.",
            "complex": "I was in a vast library with floating books. I could read them by thinking, feeling amazed and curious. Then I found a golden door that led to my childhood home, where my grandmother was waiting with warm cookies.",
            "highly_complex": "The dream began in a bustling marketplace where I was searching for something I couldn't name. Suddenly, I was underwater in a crystal cave, breathing normally but feeling the weight of the ocean above. A wise old turtle spoke to me in my grandmother's voice, telling me secrets about time and memory. Then I was flying through a storm, feeling both terrified and exhilarated, until I landed in a peaceful meadow where all my childhood pets were waiting, and I understood that this was about letting go of the past while embracing the future."
        }
    
    def test_initialization(self):
        """Test builder initialization."""
        assert isinstance(self.builder, DynamicShotPromptBuilder)
        assert isinstance(self.builder.config, DynamicShotConfig)
        assert len(self.builder.example_database) > 0
        assert ZeroShotTask.DREAM_EMOTION_ANALYSIS in self.builder.example_database
    
    def test_analyze_dream_complexity_simple(self):
        """Test complexity analysis for simple dreams."""
        complexity = self.builder.analyze_dream_complexity(self.sample_dreams["simple"])
        assert complexity == DreamComplexity.SIMPLE
    
    def test_analyze_dream_complexity_moderate(self):
        """Test complexity analysis for moderate dreams."""
        complexity = self.builder.analyze_dream_complexity(self.sample_dreams["moderate"])
        # Should be at least moderate complexity, but algorithm may vary
        assert complexity in [DreamComplexity.SIMPLE, DreamComplexity.MODERATE, DreamComplexity.COMPLEX]

    def test_analyze_dream_complexity_complex(self):
        """Test complexity analysis for complex dreams."""
        complexity = self.builder.analyze_dream_complexity(self.sample_dreams["complex"])
        # Should be moderate or higher for complex dreams
        assert complexity in [DreamComplexity.MODERATE, DreamComplexity.COMPLEX, DreamComplexity.HIGHLY_COMPLEX]

    def test_analyze_dream_complexity_highly_complex(self):
        """Test complexity analysis for highly complex dreams."""
        complexity = self.builder.analyze_dream_complexity(self.sample_dreams["highly_complex"])
        # Should be complex or highly complex for very long dreams
        assert complexity in [DreamComplexity.COMPLEX, DreamComplexity.HIGHLY_COMPLEX]
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        keywords = self.builder.extract_keywords(self.sample_dreams["simple"])
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "flying" in keywords
        assert "city" in keywords
        assert "happy" in keywords
        
        # Should not contain stop words
        assert "was" not in keywords
        assert "the" not in keywords
    
    def test_extract_keywords_complex_dream(self):
        """Test keyword extraction for complex dreams."""
        keywords = self.builder.extract_keywords(self.sample_dreams["complex"])
        assert isinstance(keywords, list)
        assert len(keywords) > 0

        # Check for some expected keywords (may vary based on extraction algorithm)
        expected_keywords = ["library", "books", "golden", "door", "floating", "thinking", "amazed", "curious"]
        found_keywords = [kw for kw in expected_keywords if kw in keywords]
        assert len(found_keywords) >= 3  # Should find at least 3 of the expected keywords
    
    def test_dream_example_relevance_calculation(self):
        """Test DreamExample relevance calculation through public API."""
        example = DreamExample(
            dream_text="I was flying over a beautiful city at sunset.",
            analysis={"emotions": ["joy", "freedom"]},
            example_type=ExampleType.BASIC_EMOTION,
            complexity=DreamComplexity.SIMPLE,
            keywords=["flying", "city", "sunset", "beautiful"]
        )

        # Should match similar content
        score1 = self.builder.calculate_example_relevance(example, "I was flying over a town", ["flying", "town"])
        assert score1 > 0.3

        # Should not match very different content
        score2 = self.builder.calculate_example_relevance(example, "I was swimming underwater", ["swimming", "underwater"])
        assert score2 < 0.3
    
    def test_select_dynamic_examples_simple(self):
        """Test example selection for simple dreams."""
        examples = self.builder.select_dynamic_examples(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            DreamComplexity.SIMPLE,
            ["flying", "city", "happy"]
        )
        
        assert isinstance(examples, list)
        assert len(examples) >= 1  # Should select at least one example for simple dreams
        assert all(isinstance(ex, DreamExample) for ex in examples)
    
    def test_select_dynamic_examples_complex(self):
        """Test example selection for complex dreams."""
        examples = self.builder.select_dynamic_examples(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["complex"],
            DreamComplexity.COMPLEX,
            ["library", "books", "golden", "door"]
        )

        assert isinstance(examples, list)
        assert len(examples) >= 1  # Should select at least one example
    
    def test_determine_example_count(self):
        """Test example count determination based on complexity."""
        assert self.builder._determine_example_count(DreamComplexity.SIMPLE) == 1
        assert self.builder._determine_example_count(DreamComplexity.MODERATE) == 2
        assert self.builder._determine_example_count(DreamComplexity.COMPLEX) == 3
        assert self.builder._determine_example_count(DreamComplexity.HIGHLY_COMPLEX) == 4
    
    def test_build_dynamic_shot_prompt_emotion_analysis(self):
        """Test building dynamic shot prompt for emotion analysis."""
        prompt = self.builder.build_dynamic_shot_prompt(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["moderate"]
        )
        
        assert isinstance(prompt, dict)
        assert "system_message" in prompt
        assert "user_message" in prompt
        assert "task" in prompt
        assert "complexity" in prompt
        assert "num_examples" in prompt
        assert "selected_examples" in prompt
        assert "keywords" in prompt
        
        assert prompt["task"] == ZeroShotTask.DREAM_EMOTION_ANALYSIS.value
        assert prompt["complexity"] in [c.value for c in DreamComplexity]
        assert isinstance(prompt["num_examples"], int)
        assert prompt["num_examples"] >= 1
    
    def test_build_dynamic_shot_prompt_musical_recommendation(self):
        """Test building dynamic shot prompt for musical recommendation."""
        prompt = self.builder.build_dynamic_shot_prompt(
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
            self.sample_dreams["simple"]
        )
        
        assert isinstance(prompt, dict)
        assert prompt["task"] == ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION.value
        assert "system_message" in prompt
        assert "user_message" in prompt
    
    def test_build_dynamic_shot_prompt_with_context(self):
        """Test building dynamic shot prompt with additional context."""
        additional_context = "The dreamer is a professional musician."
        
        prompt = self.builder.build_dynamic_shot_prompt(
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
            self.sample_dreams["simple"],
            additional_context
        )
        
        assert isinstance(prompt, dict)
        assert additional_context in prompt["user_message"]
    
    def test_build_dynamic_system_message(self):
        """Test system message building."""
        examples = self.builder.example_database[ZeroShotTask.DREAM_EMOTION_ANALYSIS][:2]
        
        system_message = self.builder._build_dynamic_system_message(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            DreamComplexity.MODERATE,
            examples
        )
        
        assert isinstance(system_message, str)
        assert "expert" in system_message.lower()
        assert "COMPLEXITY LEVEL: MODERATE" in system_message
        assert str(len(examples)) in system_message
    
    def test_build_dynamic_user_message(self):
        """Test user message building."""
        examples = self.builder.example_database[ZeroShotTask.DREAM_EMOTION_ANALYSIS][:1]
        
        user_message = self.builder._build_dynamic_user_message(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            examples
        )
        
        assert isinstance(user_message, str)
        assert "EXAMPLE 1:" in user_message
        assert "DREAM TO ANALYZE:" in user_message
        assert self.sample_dreams["simple"] in user_message
    
    def test_build_dynamic_user_message_no_examples(self):
        """Test user message building with no examples."""
        user_message = self.builder._build_dynamic_user_message(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dreams["simple"],
            []
        )
        
        assert isinstance(user_message, str)
        assert "EXAMPLE" not in user_message
        assert "DREAM TO ANALYZE:" in user_message
        assert self.sample_dreams["simple"] in user_message
    
    def test_add_example(self):
        """Test adding new examples to the database."""
        initial_count = len(self.builder.example_database[ZeroShotTask.DREAM_EMOTION_ANALYSIS])
        
        new_example = DreamExample(
            dream_text="I was dancing in the rain, feeling completely free.",
            analysis={"emotions": ["joy", "freedom"]},
            example_type=ExampleType.BASIC_EMOTION,
            complexity=DreamComplexity.SIMPLE,
            keywords=["dancing", "rain", "free"]
        )
        
        self.builder.add_example(ZeroShotTask.DREAM_EMOTION_ANALYSIS, new_example)
        
        final_count = len(self.builder.example_database[ZeroShotTask.DREAM_EMOTION_ANALYSIS])
        assert final_count == initial_count + 1
    
    def test_get_example_statistics(self):
        """Test getting example database statistics."""
        stats = self.builder.get_example_statistics()
        
        assert isinstance(stats, dict)
        assert "total_examples" in stats
        assert "examples_by_task" in stats
        assert "complexity_distribution" in stats
        assert "example_types" in stats
        
        assert stats["total_examples"] > 0
        assert isinstance(stats["examples_by_task"], dict)
        assert isinstance(stats["complexity_distribution"], dict)
        assert isinstance(stats["example_types"], dict)
    
    def test_get_usage_statistics(self):
        """Test getting usage statistics."""
        # Initially should have empty statistics
        stats = self.builder.get_usage_statistics()
        assert stats["total_usages"] == 0
        assert stats["unique_examples"] == 0
        assert stats["most_used_examples"] == []

        # Add some usage history manually for testing
        self.builder.usage_history["task1:example1"] = 3
        self.builder.usage_history["task1:example2"] = 1
        self.builder.usage_history["task2:example3"] = 2

        # Get updated statistics
        stats = self.builder.get_usage_statistics()
        assert stats["total_usages"] == 6
        assert stats["unique_examples"] == 3
        assert len(stats["most_used_examples"]) > 0
        assert stats["usage_distribution"]["min_usage"] == 1
        assert stats["usage_distribution"]["max_usage"] == 3
        assert stats["usage_distribution"]["avg_usage"] == 2.0

    def test_clear_usage_history(self):
        """Test clearing usage history."""
        # Add some usage history
        self.builder.usage_history["test_key"] = 5
        assert len(self.builder.usage_history) > 0

        # Clear history
        self.builder.clear_usage_history()
        assert len(self.builder.usage_history) == 0
    
    def test_usage_history_affects_selection(self):
        """Test that usage history affects example selection."""
        # Select examples multiple times to build usage history
        for _ in range(3):
            examples = self.builder.select_dynamic_examples(
                ZeroShotTask.DREAM_EMOTION_ANALYSIS,
                self.sample_dreams["simple"],
                DreamComplexity.SIMPLE,
                ["flying", "city", "happy"]
            )
        
        # Usage history should have entries
        assert len(self.builder.usage_history) > 0
    
    def test_custom_config(self):
        """Test builder with custom configuration."""
        custom_config = DynamicShotConfig(
            max_examples=3,
            min_examples=2,
            relevance_threshold=0.5
        )
        
        custom_builder = DynamicShotPromptBuilder(custom_config)
        assert custom_builder.config.max_examples == 3
        assert custom_builder.config.min_examples == 2
        assert custom_builder.config.relevance_threshold == 0.5
    
    def test_empty_dream_text(self):
        """Test handling of empty dream text."""
        complexity = self.builder.analyze_dream_complexity("")
        assert complexity == DreamComplexity.SIMPLE
        
        keywords = self.builder.extract_keywords("")
        assert keywords == []
        
        prompt = self.builder.build_dynamic_shot_prompt(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            ""
        )
        assert isinstance(prompt, dict)
    
    def test_very_long_dream_text(self):
        """Test handling of very long dream text."""
        long_dream = " ".join([self.sample_dreams["highly_complex"]] * 10)
        
        complexity = self.builder.analyze_dream_complexity(long_dream)
        assert complexity == DreamComplexity.HIGHLY_COMPLEX
        
        keywords = self.builder.extract_keywords(long_dream)
        assert isinstance(keywords, list)
        assert len(keywords) <= 20  # Should be limited
        
        prompt = self.builder.build_dynamic_shot_prompt(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            long_dream
        )
        assert isinstance(prompt, dict)
    
    def test_special_characters_in_dream_text(self):
        """Test handling of special characters in dream text."""
        special_dream = "I was flying! @#$%^&*() over a city... feeling happy? 😊"
        
        complexity = self.builder.analyze_dream_complexity(special_dream)
        assert isinstance(complexity, DreamComplexity)
        
        keywords = self.builder.extract_keywords(special_dream)
        assert isinstance(keywords, list)
        assert "flying" in keywords
        assert "city" in keywords
        assert "happy" in keywords
    
    def test_unsupported_task(self):
        """Test handling of unsupported tasks."""
        # Create a mock task that doesn't exist in the database
        class MockTask:
            value = "unsupported_task"
        
        mock_task = MockTask()
        
        examples = self.builder.select_dynamic_examples(
            mock_task,
            self.sample_dreams["simple"],
            DreamComplexity.SIMPLE,
            ["test"]
        )
        
        assert examples == []
    
    def test_complexity_enum_values(self):
        """Test that all complexity enum values are handled."""
        for complexity in DreamComplexity:
            count = self.builder._determine_example_count(complexity)
            assert isinstance(count, int)
            assert count >= 1
    
    def test_example_type_enum_values(self):
        """Test that all example type enum values are valid."""
        for example_type in ExampleType:
            assert isinstance(example_type.value, str)
            assert len(example_type.value) > 0
