"""
Tests for the EmotionExtractor module.
"""

import pytest
import json
from prompt_structure.emotion_extractor import EmotionExtractor, EmotionResult, EmotionCategory


class TestEmotionExtractor:
    """Test cases for EmotionExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = EmotionExtractor()
        self.sample_dreams = {
            "happy": "I was flying over a beautiful city, feeling joyful and free.",
            "sad": "I found myself in a dark, empty room, feeling lonely and melancholic.",
            "fearful": "I was being chased by shadows, feeling terrified and anxious.",
            "peaceful": "I was sitting by a calm lake, feeling peaceful and serene.",
            "mixed": "I started happy flying, but then became scared when I started falling."
        }
    
    def test_initialization(self):
        """Test EmotionExtractor initialization."""
        assert isinstance(self.extractor, EmotionExtractor)
        assert len(self.extractor.emotion_keywords) > 0
        assert len(self.extractor.mood_mappings) > 0
        assert "joy" in self.extractor.emotion_keywords
        assert "euphoric" in self.extractor.mood_mappings
    
    def test_extract_emotions_happy_dream(self):
        """Test emotion extraction from happy dream."""
        result = self.extractor.extract_emotions(self.sample_dreams["happy"])
        
        assert isinstance(result, EmotionResult)
        assert len(result.primary_emotions) > 0
        assert any(emotion in ["joy", "freedom", "peace"] for emotion in result.primary_emotions)
        assert result.confidence_score > 0
        assert result.overall_mood in ["euphoric", "peaceful", "energetic"]
    
    def test_extract_emotions_sad_dream(self):
        """Test emotion extraction from sad dream."""
        result = self.extractor.extract_emotions(self.sample_dreams["sad"])
        
        assert isinstance(result, EmotionResult)
        assert len(result.primary_emotions) > 0
        assert any(emotion in ["sadness", "melancholy"] for emotion in result.primary_emotions)
        assert result.overall_mood in ["melancholic", "contemplative"]
    
    def test_extract_emotions_fearful_dream(self):
        """Test emotion extraction from fearful dream."""
        result = self.extractor.extract_emotions(self.sample_dreams["fearful"])
        
        assert isinstance(result, EmotionResult)
        assert len(result.primary_emotions) > 0
        assert any(emotion in ["fear", "anxiety"] for emotion in result.primary_emotions)
        assert result.overall_mood in ["anxious", "melancholic"]
    
    def test_extract_emotions_peaceful_dream(self):
        """Test emotion extraction from peaceful dream."""
        result = self.extractor.extract_emotions(self.sample_dreams["peaceful"])
        
        assert isinstance(result, EmotionResult)
        assert len(result.primary_emotions) > 0
        assert any(emotion in ["peace", "calm"] for emotion in result.primary_emotions)
        assert result.overall_mood in ["peaceful", "contemplative"]
    
    def test_extract_emotions_empty_text(self):
        """Test emotion extraction from empty text."""
        result = self.extractor.extract_emotions("")
        
        assert isinstance(result, EmotionResult)
        assert len(result.primary_emotions) > 0  # Should have default emotions
        assert result.confidence_score >= 0
    
    def test_extract_emotions_neutral_text(self):
        """Test emotion extraction from neutral text."""
        neutral_text = "I walked down the street and saw a building."
        result = self.extractor.extract_emotions(neutral_text)
        
        assert isinstance(result, EmotionResult)
        assert len(result.primary_emotions) > 0
        # Should have low confidence for neutral text
        assert result.confidence_score < 0.8
    
    def test_emotion_result_to_dict(self):
        """Test EmotionResult to_dict conversion."""
        result = self.extractor.extract_emotions(self.sample_dreams["happy"])
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "primary_emotions" in result_dict
        assert "emotional_intensity" in result_dict
        assert "emotional_progression" in result_dict
        assert "overall_mood" in result_dict
        assert "confidence_score" in result_dict
    
    def test_fallback_emotion_analysis(self):
        """Test fallback emotion analysis method."""
        result = self.extractor._fallback_emotion_analysis(self.sample_dreams["happy"])
        
        assert isinstance(result, EmotionResult)
        assert result.raw_response == "fallback_analysis"
        assert result.confidence_score <= 0.5  # Fallback should have lower confidence
    
    def test_determine_overall_mood(self):
        """Test overall mood determination."""
        # Test various emotion combinations
        test_cases = [
            (["joy", "excitement"], "euphoric"),
            (["peace", "calm"], "peaceful"),
            (["sadness", "melancholy"], "melancholic"),
            (["fear", "anxiety"], "anxious"),
            (["wonder", "curiosity"], "contemplative"),
            ([], "neutral")
        ]
        
        for emotions, expected_mood_type in test_cases:
            mood = self.extractor._determine_overall_mood(emotions)
            assert isinstance(mood, str)
            assert len(mood) > 0
    
    def test_parse_emotion_response_valid_json(self):
        """Test parsing valid JSON emotion response."""
        valid_json = json.dumps({
            "primary_emotions": ["joy", "peace"],
            "emotional_intensity": {"joy": 8, "peace": 7},
            "emotional_progression": "consistent",
            "overall_mood": "euphoric",
            "confidence_score": 0.85
        })
        
        result = self.extractor._parse_emotion_response(valid_json)
        
        assert isinstance(result, dict)
        assert "primary_emotions" in result
        assert result["primary_emotions"] == ["joy", "peace"]
    
    def test_parse_emotion_response_invalid_json(self):
        """Test parsing invalid JSON emotion response."""
        invalid_json = "This is not valid JSON"
        
        with pytest.raises(ValueError, match="Could not parse emotion response"):
            self.extractor._parse_emotion_response(invalid_json)
    
    def test_parse_emotion_response_embedded_json(self):
        """Test parsing JSON embedded in text response."""
        embedded_json = 'Here is the analysis: {"primary_emotions": ["joy"], "confidence_score": 0.8} and some more text.'
        
        result = self.extractor._parse_emotion_response(embedded_json)
        
        assert isinstance(result, dict)
        assert "primary_emotions" in result
        assert result["primary_emotions"] == ["joy"]
    
    def test_get_emotion_statistics(self):
        """Test emotion statistics calculation."""
        # Create multiple emotion results
        results = []
        for dream in self.sample_dreams.values():
            result = self.extractor.extract_emotions(dream)
            results.append(result)
        
        stats = self.extractor.get_emotion_statistics(results)
        
        assert isinstance(stats, dict)
        assert "total_analyses" in stats
        assert "most_common_emotions" in stats
        assert "most_common_moods" in stats
        assert "average_confidence" in stats
        assert "unique_emotions_detected" in stats
        
        assert stats["total_analyses"] == len(results)
        assert isinstance(stats["most_common_emotions"], list)
        assert isinstance(stats["average_confidence"], float)
    
    def test_get_emotion_statistics_empty_list(self):
        """Test emotion statistics with empty results list."""
        stats = self.extractor.get_emotion_statistics([])
        
        assert stats == {}
    
    def test_validate_emotion_result_valid(self):
        """Test validation of valid emotion result."""
        result = self.extractor.extract_emotions(self.sample_dreams["happy"])
        
        is_valid = self.extractor.validate_emotion_result(result)
        
        assert is_valid is True
    
    def test_validate_emotion_result_invalid_confidence(self):
        """Test validation with invalid confidence score."""
        result = self.extractor.extract_emotions(self.sample_dreams["happy"])
        result.confidence_score = 1.5  # Invalid: > 1
        
        is_valid = self.extractor.validate_emotion_result(result)
        
        assert is_valid is False
    
    def test_validate_emotion_result_invalid_intensity(self):
        """Test validation with invalid emotional intensity."""
        result = self.extractor.extract_emotions(self.sample_dreams["happy"])
        result.emotional_intensity = {"joy": 15}  # Invalid: > 10
        
        is_valid = self.extractor.validate_emotion_result(result)
        
        assert is_valid is False
    
    def test_validate_emotion_result_no_emotions(self):
        """Test validation with no primary emotions."""
        result = self.extractor.extract_emotions(self.sample_dreams["happy"])
        result.primary_emotions = []  # Invalid: empty list
        
        is_valid = self.extractor.validate_emotion_result(result)
        
        assert is_valid is False
    
    def test_emotion_keywords_coverage(self):
        """Test that emotion keywords cover major emotion categories."""
        expected_emotions = ["joy", "sadness", "fear", "anger", "peace", "excitement"]
        
        for emotion in expected_emotions:
            assert emotion in self.extractor.emotion_keywords
            assert len(self.extractor.emotion_keywords[emotion]) > 0
    
    def test_mood_mappings_structure(self):
        """Test that mood mappings have proper structure."""
        for mood, mapping in self.extractor.mood_mappings.items():
            assert isinstance(mapping, dict)
            assert "tempo_range" in mapping
            assert "key_preference" in mapping
            assert "energy" in mapping
            
            # Validate tempo range
            tempo_range = mapping["tempo_range"]
            assert isinstance(tempo_range, list)
            assert len(tempo_range) == 2
            assert tempo_range[0] < tempo_range[1]
            assert tempo_range[0] > 0
    
    def test_simulate_ai_response_consistency(self):
        """Test that simulated AI response is consistent."""
        dream_text = self.sample_dreams["happy"]
        
        # Run multiple times to check consistency
        responses = []
        for _ in range(3):
            response = self.extractor._simulate_ai_response(dream_text)
            responses.append(response)
        
        # All responses should be valid JSON
        for response in responses:
            parsed = json.loads(response)
            assert isinstance(parsed, dict)
            assert "primary_emotions" in parsed
            assert "confidence_score" in parsed
        
        # Responses should be similar (same input should give similar output)
        parsed_responses = [json.loads(r) for r in responses]
        first_emotions = set(parsed_responses[0]["primary_emotions"])
        
        for parsed in parsed_responses[1:]:
            current_emotions = set(parsed["primary_emotions"])
            # Should have some overlap in detected emotions
            assert len(first_emotions.intersection(current_emotions)) > 0
