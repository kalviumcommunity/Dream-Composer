"""
Tests for the Zero-Shot Prompts module.
"""

import pytest
import json
from prompt_structure.zero_shot_prompts import ZeroShotPromptBuilder, ZeroShotTask, ZeroShotPrompt


class TestZeroShotPromptBuilder:
    """Test cases for ZeroShotPromptBuilder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = ZeroShotPromptBuilder()
        self.sample_dream = "I was flying over a beautiful city at sunset, feeling free and peaceful."
    
    def test_initialization(self):
        """Test ZeroShotPromptBuilder initialization."""
        assert isinstance(self.builder, ZeroShotPromptBuilder)
        assert len(self.builder.prompts) > 0
        assert ZeroShotTask.DREAM_EMOTION_ANALYSIS in self.builder.prompts
        assert ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION in self.builder.prompts
    
    def test_build_emotion_analysis_prompt(self):
        """Test building emotion analysis zero-shot prompt."""
        prompt = self.builder.build_zero_shot_prompt(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dream
        )
        
        assert "system_message" in prompt
        assert "user_message" in prompt
        assert "expected_format" in prompt
        assert "task" in prompt
        assert "reasoning_steps" in prompt
        
        assert self.sample_dream in prompt["user_message"]
        assert "emotion" in prompt["system_message"].lower()
        assert prompt["task"] == "dream_emotion_analysis"
    
    def test_build_musical_recommendation_prompt(self):
        """Test building musical recommendation zero-shot prompt."""
        prompt = self.builder.build_zero_shot_prompt(
            ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
            self.sample_dream
        )
        
        assert "system_message" in prompt
        assert "user_message" in prompt
        assert "musical" in prompt["system_message"].lower()
        assert self.sample_dream in prompt["user_message"]
        assert prompt["task"] == "musical_style_recommendation"
    
    def test_build_symbolism_interpretation_prompt(self):
        """Test building symbolism interpretation zero-shot prompt."""
        prompt = self.builder.build_zero_shot_prompt(
            ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION,
            self.sample_dream
        )
        
        assert "system_message" in prompt
        assert "user_message" in prompt
        assert "symbol" in prompt["system_message"].lower()
        assert self.sample_dream in prompt["user_message"]
    
    def test_build_mood_mapping_prompt(self):
        """Test building mood to music mapping zero-shot prompt."""
        prompt = self.builder.build_zero_shot_prompt(
            ZeroShotTask.MOOD_TO_MUSIC_MAPPING,
            self.sample_dream
        )
        
        assert "system_message" in prompt
        assert "user_message" in prompt
        assert "mood" in prompt["system_message"].lower()
        assert "music" in prompt["system_message"].lower()
    
    def test_build_narrative_analysis_prompt(self):
        """Test building narrative analysis zero-shot prompt."""
        prompt = self.builder.build_zero_shot_prompt(
            ZeroShotTask.DREAM_NARRATIVE_ANALYSIS,
            self.sample_dream
        )
        
        assert "system_message" in prompt
        assert "user_message" in prompt
        assert "narrative" in prompt["system_message"].lower()
    
    def test_build_prompt_with_additional_context(self):
        """Test building prompt with additional context."""
        additional_context = "The dreamer is a musician who often dreams about flying."
        
        prompt = self.builder.build_zero_shot_prompt(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dream,
            additional_context
        )
        
        assert additional_context in prompt["user_message"]
        assert "ADDITIONAL CONTEXT:" in prompt["user_message"]
    
    def test_unsupported_task(self):
        """Test handling of unsupported zero-shot tasks."""
        with pytest.raises(ValueError, match="Unsupported zero-shot task"):
            # Create a fake task
            fake_task = "FAKE_TASK"
            self.builder.build_zero_shot_prompt(fake_task, self.sample_dream)
    
    def test_get_available_tasks(self):
        """Test getting list of available zero-shot tasks."""
        tasks = self.builder.get_available_tasks()
        
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        assert ZeroShotTask.DREAM_EMOTION_ANALYSIS in tasks
        assert ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION in tasks
        assert ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION in tasks
    
    def test_get_task_description(self):
        """Test getting task descriptions."""
        description = self.builder.get_task_description(ZeroShotTask.DREAM_EMOTION_ANALYSIS)
        
        assert isinstance(description, str)
        assert len(description) > 0
        assert "emotion" in description.lower()
    
    def test_get_task_description_invalid_task(self):
        """Test getting description for invalid task."""
        with pytest.raises(ValueError, match="Unknown task"):
            # Create a fake enum value
            fake_task = type('FakeTask', (), {'value': 'fake'})()
            self.builder.get_task_description(fake_task)
    
    def test_validate_response_format_valid(self):
        """Test validation with valid response format."""
        valid_response = json.dumps({
            "primary_emotions": ["joy", "peace"],
            "confidence": 0.85,
            "analysis": "detailed analysis"
        })
        
        is_valid = self.builder.validate_response_format(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            valid_response
        )
        
        assert is_valid is True
    
    def test_validate_response_format_invalid_json(self):
        """Test validation with invalid JSON."""
        invalid_response = "This is not valid JSON"
        
        is_valid = self.builder.validate_response_format(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            invalid_response
        )
        
        assert is_valid is False
    
    def test_validate_response_format_missing_confidence(self):
        """Test validation with missing confidence score."""
        response_without_confidence = json.dumps({
            "primary_emotions": ["joy", "peace"],
            "analysis": "detailed analysis"
        })
        
        is_valid = self.builder.validate_response_format(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            response_without_confidence
        )
        
        assert is_valid is False
    
    def test_validate_response_format_invalid_confidence(self):
        """Test validation with invalid confidence score."""
        response_invalid_confidence = json.dumps({
            "primary_emotions": ["joy", "peace"],
            "confidence": 1.5,  # Invalid: > 1
            "analysis": "detailed analysis"
        })
        
        is_valid = self.builder.validate_response_format(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            response_invalid_confidence
        )
        
        assert is_valid is False
    
    def test_validate_response_format_non_dict(self):
        """Test validation with non-dictionary JSON."""
        non_dict_response = json.dumps(["this", "is", "a", "list"])
        
        is_valid = self.builder.validate_response_format(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            non_dict_response
        )
        
        assert is_valid is False
    
    def test_prompt_structure_completeness(self):
        """Test that all prompts have complete structure."""
        for task, prompt in self.builder.prompts.items():
            assert isinstance(prompt, ZeroShotPrompt)
            assert prompt.task == task
            assert prompt.instruction
            assert prompt.context
            assert prompt.output_format
            assert isinstance(prompt.constraints, list)
            assert isinstance(prompt.reasoning_steps, list)
            assert len(prompt.constraints) > 0
            assert len(prompt.reasoning_steps) > 0
    
    def test_system_message_structure(self):
        """Test that system messages have proper structure."""
        prompt = self.builder.build_zero_shot_prompt(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dream
        )
        
        system_message = prompt["system_message"]
        
        assert "CONTEXT:" in system_message
        assert "CONSTRAINTS:" in system_message
        assert "REASONING APPROACH:" in system_message
        assert "JSON format" in system_message
    
    def test_user_message_structure(self):
        """Test that user messages have proper structure."""
        prompt = self.builder.build_zero_shot_prompt(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            self.sample_dream
        )
        
        user_message = prompt["user_message"]
        
        assert "DREAM DESCRIPTION:" in user_message
        assert self.sample_dream in user_message
        assert "JSON format:" in user_message
        assert "Analysis:" in user_message
    
    def test_expected_format_is_valid_json(self):
        """Test that expected formats are valid JSON."""
        for task in self.builder.get_available_tasks():
            prompt = self.builder.build_zero_shot_prompt(task, self.sample_dream)
            expected_format = prompt["expected_format"]
            
            # Should be parseable as JSON
            try:
                json.loads(expected_format)
            except json.JSONDecodeError:
                pytest.fail(f"Expected format for {task} is not valid JSON")
    
    def test_reasoning_steps_are_actionable(self):
        """Test that reasoning steps are actionable and clear."""
        for task, prompt in self.builder.prompts.items():
            reasoning_steps = prompt.reasoning_steps
            
            assert len(reasoning_steps) >= 3  # Should have at least 3 steps
            
            for step in reasoning_steps:
                assert isinstance(step, str)
                assert len(step) > 10  # Should be descriptive
                # Should contain action words
                action_words = ["identify", "analyze", "consider", "determine", "assess", "examine",
                               "read", "map", "match", "select", "provide", "ensure", "suggest", "research", "connect"]
                assert any(word in step.lower() for word in action_words)
    
    def test_constraints_are_specific(self):
        """Test that constraints are specific and actionable."""
        for task, prompt in self.builder.prompts.items():
            constraints = prompt.constraints
            
            assert len(constraints) >= 2  # Should have at least 2 constraints
            
            for constraint in constraints:
                assert isinstance(constraint, str)
                assert len(constraint) > 15  # Should be descriptive
    
    def test_empty_dream_text(self):
        """Test handling of empty dream text."""
        prompt = self.builder.build_zero_shot_prompt(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            ""
        )
        
        assert "system_message" in prompt
        assert "user_message" in prompt
        assert "DREAM DESCRIPTION:" in prompt["user_message"]
    
    def test_very_long_dream_text(self):
        """Test handling of very long dream text."""
        long_dream = "I was flying through the sky. " * 500  # Very long text
        
        prompt = self.builder.build_zero_shot_prompt(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            long_dream
        )
        
        assert long_dream in prompt["user_message"]
        assert len(prompt["user_message"]) > len(long_dream)  # Should include template text
    
    def test_special_characters_in_dream_text(self):
        """Test handling of special characters in dream text."""
        special_dream = "I dreamt of symbols: @#$%^&*(){}[]|\\:;\"'<>,.?/~`"
        
        prompt = self.builder.build_zero_shot_prompt(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            special_dream
        )
        
        assert special_dream in prompt["user_message"]
        assert "system_message" in prompt
