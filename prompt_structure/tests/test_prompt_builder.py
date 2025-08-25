"""
Tests for the PromptBuilder module.
"""

import pytest
import json
from prompt_structure.prompt_builder import PromptBuilder, PromptType, PromptTemplate


class TestPromptBuilder:
    """Test cases for PromptBuilder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = PromptBuilder()
        self.sample_dream = "I was flying over a beautiful city at sunset, feeling free and peaceful."
    
    def test_initialization(self):
        """Test PromptBuilder initialization."""
        assert isinstance(self.builder, PromptBuilder)
        assert len(self.builder.templates) > 0
        assert PromptType.EMOTION_EXTRACTION in self.builder.templates
        assert PromptType.MUSIC_MAPPING in self.builder.templates
    
    def test_build_emotion_extraction_prompt(self):
        """Test building emotion extraction prompt."""
        prompt = self.builder.build_prompt(
            PromptType.EMOTION_EXTRACTION,
            self.sample_dream
        )
        
        assert "system_message" in prompt
        assert "user_message" in prompt
        assert "expected_format" in prompt
        assert self.sample_dream in prompt["user_message"]
        assert "emotion" in prompt["system_message"].lower()
    
    def test_build_music_mapping_prompt(self):
        """Test building music mapping prompt."""
        prompt = self.builder.build_prompt(
            PromptType.MUSIC_MAPPING,
            self.sample_dream,
            emotions=["joy", "peace"],
            mood="euphoric",
            intensity=8
        )
        
        assert "system_message" in prompt
        assert "user_message" in prompt
        assert "music" in prompt["system_message"].lower()
        assert "joy" in prompt["user_message"]
        assert "peace" in prompt["user_message"]
    
    def test_build_comprehensive_analysis_prompt(self):
        """Test building comprehensive analysis prompt."""
        prompt = self.builder.build_prompt(
            PromptType.COMPREHENSIVE_ANALYSIS,
            self.sample_dream
        )
        
        assert "system_message" in prompt
        assert "user_message" in prompt
        assert "comprehensive" in prompt["system_message"].lower()
        assert self.sample_dream in prompt["user_message"]
    
    def test_unsupported_prompt_type(self):
        """Test handling of unsupported prompt types."""
        with pytest.raises(ValueError, match="Unsupported prompt type"):
            # Create a fake enum value
            fake_type = "FAKE_TYPE"
            self.builder.build_prompt(fake_type, self.sample_dream)
    
    def test_get_template(self):
        """Test getting specific templates."""
        template = self.builder.get_template(PromptType.EMOTION_EXTRACTION)
        
        assert isinstance(template, PromptTemplate)
        assert template.prompt_type == PromptType.EMOTION_EXTRACTION
        assert template.system_message
        assert template.user_template
    
    def test_list_available_types(self):
        """Test listing available prompt types."""
        types = self.builder.list_available_types()
        
        assert isinstance(types, list)
        assert len(types) > 0
        assert PromptType.EMOTION_EXTRACTION in types
        assert PromptType.MUSIC_MAPPING in types
    
    def test_validate_prompt_structure_valid_json(self):
        """Test validation with valid JSON response."""
        valid_response = json.dumps({
            "primary_emotions": ["joy", "peace"],
            "emotional_intensity": {"joy": 8, "peace": 7},
            "overall_mood": "euphoric",
            "confidence_score": 0.85
        })
        
        is_valid = self.builder.validate_prompt_structure(
            PromptType.EMOTION_EXTRACTION,
            valid_response
        )
        
        assert is_valid is True
    
    def test_validate_prompt_structure_invalid_json(self):
        """Test validation with invalid JSON response."""
        invalid_response = "This is not valid JSON"
        
        is_valid = self.builder.validate_prompt_structure(
            PromptType.EMOTION_EXTRACTION,
            invalid_response
        )
        
        assert is_valid is False
    
    def test_validate_prompt_structure_non_dict_json(self):
        """Test validation with JSON that's not a dictionary."""
        non_dict_response = json.dumps(["this", "is", "a", "list"])
        
        is_valid = self.builder.validate_prompt_structure(
            PromptType.EMOTION_EXTRACTION,
            non_dict_response
        )
        
        assert is_valid is False
    
    def test_add_custom_template(self):
        """Test adding custom prompt template."""
        custom_template = PromptTemplate(
            name="Custom Test",
            prompt_type=PromptType.EMOTION_EXTRACTION,  # Reuse existing type for test
            system_message="Custom system message",
            user_template="Custom user template: {dream_text}",
            expected_format="Custom format",
            examples=[],
            parameters={}
        )
        
        original_count = len(self.builder.templates)
        self.builder.add_custom_template(custom_template)
        
        # Should replace existing template for same type
        assert len(self.builder.templates) == original_count
        
        # Verify the template was updated
        retrieved = self.builder.get_template(PromptType.EMOTION_EXTRACTION)
        assert retrieved.system_message == "Custom system message"
    
    def test_get_prompt_statistics(self):
        """Test getting prompt statistics."""
        stats = self.builder.get_prompt_statistics()
        
        assert isinstance(stats, dict)
        assert "total_templates" in stats
        assert "template_types" in stats
        assert "templates_with_examples" in stats
        assert "average_system_message_length" in stats
        
        assert stats["total_templates"] > 0
        assert isinstance(stats["template_types"], list)
        assert isinstance(stats["average_system_message_length"], int)
    
    def test_prompt_template_structure(self):
        """Test that all templates have required structure."""
        for prompt_type, template in self.builder.templates.items():
            assert isinstance(template, PromptTemplate)
            assert template.name
            assert template.system_message
            assert template.user_template
            assert template.expected_format
            assert isinstance(template.examples, list)
            assert isinstance(template.parameters, dict)
            assert template.prompt_type == prompt_type
    
    def test_prompt_formatting_with_kwargs(self):
        """Test prompt formatting with additional keyword arguments."""
        prompt = self.builder.build_prompt(
            PromptType.MUSIC_MAPPING,
            self.sample_dream,
            emotions=["joy", "freedom"],
            mood="euphoric",
            intensity=9,
            custom_param="test_value"
        )
        
        assert "joy" in prompt["user_message"]
        assert "freedom" in prompt["user_message"]
        assert "euphoric" in prompt["user_message"]
        assert "9" in prompt["user_message"]
    
    def test_empty_dream_text(self):
        """Test handling of empty dream text."""
        prompt = self.builder.build_prompt(
            PromptType.EMOTION_EXTRACTION,
            ""
        )
        
        assert "system_message" in prompt
        assert "user_message" in prompt
        # Should still contain the template structure even with empty text
        assert "Dream:" in prompt["user_message"]
    
    def test_special_characters_in_dream_text(self):
        """Test handling of special characters in dream text."""
        special_dream = "I dreamt of symbols: @#$%^&*(){}[]|\\:;\"'<>,.?/~`"
        
        prompt = self.builder.build_prompt(
            PromptType.EMOTION_EXTRACTION,
            special_dream
        )
        
        assert special_dream in prompt["user_message"]
        assert "system_message" in prompt
    
    def test_very_long_dream_text(self):
        """Test handling of very long dream text."""
        long_dream = "I was flying " * 1000  # Very long text
        
        prompt = self.builder.build_prompt(
            PromptType.EMOTION_EXTRACTION,
            long_dream
        )
        
        assert long_dream in prompt["user_message"]
        assert len(prompt["user_message"]) > len(long_dream)  # Should include template text
