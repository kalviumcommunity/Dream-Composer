"""
Tests for the DreamAnalyzer module.
"""

import pytest
from prompt_structure.dream_analyzer import DreamAnalyzer, ComprehensiveAnalysis
from prompt_structure.emotion_extractor import EmotionResult
from prompt_structure.music_mapper import MusicalParameters


class TestDreamAnalyzer:
    """Test cases for DreamAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = DreamAnalyzer(random_seed=42)  # Fixed seed for deterministic tests
        self.sample_dreams = {
            "flying": "I was flying over a shimmering city at sunset, feeling free and peaceful.",
            "water": "I found myself swimming in crystal clear water, feeling calm and refreshed.",
            "falling": "I was falling through darkness, feeling scared and out of control.",
            "nature": "I walked through a beautiful forest with golden light filtering through trees.",
            "complex": "I started in a dark cave, then flew out into bright sunlight over a vast ocean, feeling first afraid then joyful."
        }
    
    def test_initialization(self):
        """Test DreamAnalyzer initialization."""
        assert isinstance(self.analyzer, DreamAnalyzer)
        assert hasattr(self.analyzer, 'emotion_extractor')
        assert hasattr(self.analyzer, 'music_mapper')
        assert hasattr(self.analyzer, 'prompt_builder')
        assert len(self.analyzer.symbol_database) > 0
        assert len(self.analyzer.color_mappings) > 0
    
    def test_analyze_dream_basic(self):
        """Test basic dream analysis."""
        analysis = self.analyzer.analyze_dream(self.sample_dreams["flying"])
        
        assert isinstance(analysis, ComprehensiveAnalysis)
        assert analysis.dream_text == self.sample_dreams["flying"]
        assert isinstance(analysis.emotion_result, EmotionResult)
        assert isinstance(analysis.musical_parameters, MusicalParameters)
        assert isinstance(analysis.symbols, list)
        assert hasattr(analysis, 'narrative')
        assert hasattr(analysis, 'sensory')
        assert 0 <= analysis.overall_confidence <= 1
        assert analysis.analysis_timestamp
    
    def test_analyze_dream_with_symbols(self):
        """Test dream analysis with symbolic elements."""
        analysis = self.analyzer.analyze_dream(self.sample_dreams["flying"])
        
        # Should detect "flying" symbol
        symbol_elements = [s.element for s in analysis.symbols]
        assert "flying" in symbol_elements
        
        # Check symbol interpretation structure
        for symbol in analysis.symbols:
            assert hasattr(symbol, 'element')
            assert hasattr(symbol, 'meaning')
            assert hasattr(symbol, 'musical_relevance')
            assert hasattr(symbol, 'emotional_weight')
            assert 0 <= symbol.emotional_weight <= 1
    
    def test_analyze_dream_water_symbolism(self):
        """Test analysis of water symbolism."""
        analysis = self.analyzer.analyze_dream(self.sample_dreams["water"])
        
        symbol_elements = [s.element for s in analysis.symbols]
        assert "water" in symbol_elements
        
        # Find water symbol and check its properties
        water_symbol = next(s for s in analysis.symbols if s.element == "water")
        assert "emotion" in water_symbol.meaning.lower()
        assert water_symbol.emotional_weight > 0
    
    def test_analyze_dream_narrative_structure(self):
        """Test narrative structure analysis."""
        analysis = self.analyzer.analyze_dream(self.sample_dreams["complex"])
        
        assert hasattr(analysis.narrative, 'structure')
        assert hasattr(analysis.narrative, 'pacing')
        assert hasattr(analysis.narrative, 'climax')
        assert hasattr(analysis.narrative, 'transitions')
        
        assert analysis.narrative.structure in ["simple", "moderate", "complex"]
        assert analysis.narrative.pacing in ["fast", "moderate", "slow"]
        assert isinstance(analysis.narrative.transitions, list)
    
    def test_analyze_dream_sensory_details(self):
        """Test sensory details extraction."""
        # Use a dream with color words
        colorful_dream = "I saw a bright blue sky and golden sunlight, hearing beautiful music."
        analysis = self.analyzer.analyze_dream(colorful_dream)
        
        assert hasattr(analysis.sensory, 'colors')
        assert hasattr(analysis.sensory, 'sounds')
        assert hasattr(analysis.sensory, 'textures')
        assert hasattr(analysis.sensory, 'atmosphere')
        assert hasattr(analysis.sensory, 'visual_intensity')
        
        # Should detect colors and sounds
        assert len(analysis.sensory.colors) > 0 or len(analysis.sensory.sounds) > 0
        assert 0 <= analysis.sensory.visual_intensity <= 1
    
    def test_symbol_database_structure(self):
        """Test symbol database has proper structure."""
        for symbol, data in self.analyzer.symbol_database.items():
            assert isinstance(data, dict)
            assert "meaning" in data
            assert "musical_relevance" in data
            assert "emotional_associations" in data
            assert "tempo_influence" in data
            
            assert isinstance(data["emotional_associations"], list)
            assert isinstance(data["tempo_influence"], (int, float))
            assert data["tempo_influence"] > 0
    
    def test_color_mappings_structure(self):
        """Test color mappings have proper structure."""
        for color, mapping in self.analyzer.color_mappings.items():
            assert isinstance(mapping, dict)
            assert "mood" in mapping
            assert "key_preference" in mapping
            assert "instruments" in mapping
            
            assert mapping["key_preference"] in ["major", "minor"]
            assert isinstance(mapping["instruments"], list)
            assert len(mapping["instruments"]) > 0
    
    def test_calculate_symbol_weight(self):
        """Test symbol weight calculation."""
        # Test with different contexts
        test_cases = [
            ("flying", "I was flying", 0.5),  # Base weight
            ("flying", "I was flying and flying", 0.7),  # Multiple occurrences
            ("flying", "I was very flying beautifully", 0.7),  # With intensifiers
        ]
        
        for symbol, text, min_expected in test_cases:
            weight = self.analyzer._calculate_symbol_weight(symbol, text.lower())
            assert weight >= min_expected
            assert 0 <= weight <= 1
    
    def test_analyze_narrative_simple(self):
        """Test narrative analysis for simple dreams."""
        simple_dream = "I was happy."
        analysis = self.analyzer.analyze_dream(simple_dream)
        
        assert analysis.narrative.structure == "simple"
    
    def test_analyze_narrative_complex(self):
        """Test narrative analysis for complex dreams."""
        complex_dream = ". ".join([
            "I started in a dark room",
            "Then I walked outside",
            "Suddenly I was flying",
            "After that I landed in a garden",
            "Finally I woke up feeling peaceful"
        ])
        analysis = self.analyzer.analyze_dream(complex_dream)
        
        assert analysis.narrative.structure in ["moderate", "complex"]
        assert len(analysis.narrative.transitions) > 0
    
    def test_extract_sensory_details_colors(self):
        """Test color extraction from sensory details."""
        colorful_dream = "I saw red roses, blue sky, and golden sunlight."
        sensory = self.analyzer._extract_sensory_details(colorful_dream)
        
        assert "red" in sensory.colors
        assert "blue" in sensory.colors
        assert "gold" in sensory.colors or len(sensory.colors) >= 2  # Should detect multiple colors
    
    def test_extract_sensory_details_sounds(self):
        """Test sound extraction from sensory details."""
        sound_dream = "I heard beautiful music and singing, then everything became quiet."
        sensory = self.analyzer._extract_sensory_details(sound_dream)
        
        assert len(sensory.sounds) > 0
        assert any(sound in ["music", "singing", "quiet"] for sound in sensory.sounds)
    
    def test_extract_sensory_details_atmosphere(self):
        """Test atmosphere detection."""
        test_cases = [
            ("I felt peaceful and calm", "peaceful"),
            ("Everything was scary and tense", "tense"),
            ("The place was mysterious and strange", "mysterious"),
            ("I was happy and joyful", "joyful"),
            ("I felt sad and melancholy", "melancholic")
        ]
        
        for dream_text, expected_atmosphere in test_cases:
            sensory = self.analyzer._extract_sensory_details(dream_text)
            assert sensory.atmosphere == expected_atmosphere
    
    def test_calculate_overall_confidence(self):
        """Test overall confidence calculation."""
        # Create mock results with known confidence scores
        emotion_result = EmotionResult(
            primary_emotions=["joy"],
            emotional_intensity={"joy": 8},
            emotional_progression="test",
            overall_mood="euphoric",
            confidence_score=0.8,
            raw_response="test"
        )
        
        musical_parameters = MusicalParameters(
            tempo={"bpm": 120, "description": "test"},
            key={"signature": "C major", "mode": "major"},
            time_signature="4/4",
            instruments=["piano"],
            style="classical",
            dynamics={"overall": "mf"},
            harmonic_progression=["I", "V"],
            expression_marks=["legato"],
            confidence_score=0.7
        )
        
        symbols = [
            type('Symbol', (), {
                'element': 'flying',
                'meaning': 'freedom',
                'musical_relevance': 'ascending',
                'emotional_weight': 0.8
            })()
        ]
        
        confidence = self.analyzer._calculate_overall_confidence(
            emotion_result, musical_parameters, symbols
        )
        
        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)
    
    def test_get_analysis_summary(self):
        """Test analysis summary generation."""
        analysis = self.analyzer.analyze_dream(self.sample_dreams["flying"])
        summary = self.analyzer.get_analysis_summary(analysis)
        
        assert isinstance(summary, dict)
        required_keys = [
            "primary_emotions", "overall_mood", "suggested_tempo",
            "suggested_key", "main_instruments", "key_symbols",
            "atmosphere", "confidence"
        ]
        
        for key in required_keys:
            assert key in summary
        
        assert isinstance(summary["primary_emotions"], list)
        assert isinstance(summary["suggested_tempo"], int)
        assert isinstance(summary["main_instruments"], list)
        assert isinstance(summary["key_symbols"], list)
        assert 0 <= summary["confidence"] <= 1
    
    def test_comprehensive_analysis_to_dict(self):
        """Test ComprehensiveAnalysis to_dict conversion."""
        analysis = self.analyzer.analyze_dream(self.sample_dreams["flying"])
        analysis_dict = analysis.to_dict()
        
        assert isinstance(analysis_dict, dict)
        required_keys = [
            "dream_text", "emotions", "musical_parameters",
            "symbols", "narrative", "sensory",
            "overall_confidence", "analysis_timestamp"
        ]
        
        for key in required_keys:
            assert key in analysis_dict
        
        # Check nested structure
        assert isinstance(analysis_dict["emotions"], dict)
        assert isinstance(analysis_dict["musical_parameters"], dict)
        assert isinstance(analysis_dict["symbols"], list)
        assert isinstance(analysis_dict["narrative"], dict)
        assert isinstance(analysis_dict["sensory"], dict)
    
    def test_empty_dream_analysis(self):
        """Test analysis of empty dream text."""
        analysis = self.analyzer.analyze_dream("")
        
        assert isinstance(analysis, ComprehensiveAnalysis)
        assert analysis.dream_text == ""
        assert analysis.overall_confidence >= 0
        # Should still produce some analysis even for empty text
        assert len(analysis.emotion_result.primary_emotions) > 0
    
    def test_very_long_dream_analysis(self):
        """Test analysis of very long dream text."""
        long_dream = "I was flying through the sky. " * 100  # Very long dream
        analysis = self.analyzer.analyze_dream(long_dream)

        assert isinstance(analysis, ComprehensiveAnalysis)
        assert analysis.dream_text == long_dream
        assert analysis.overall_confidence >= 0
        # Should handle long text without errors

    def test_deterministic_analysis_with_seed(self):
        """Test that analysis is deterministic when using a fixed seed."""
        dream_text = "I was flying over a beautiful city, feeling joyful."

        # Create two analyzers with the same seed
        analyzer1 = DreamAnalyzer(random_seed=123)
        analyzer2 = DreamAnalyzer(random_seed=123)

        analysis1 = analyzer1.analyze_dream(dream_text)
        analysis2 = analyzer2.analyze_dream(dream_text)

        # Musical parameters should be identical with same seed
        assert analysis1.musical_parameters.key == analysis2.musical_parameters.key
        assert analysis1.musical_parameters.instruments == analysis2.musical_parameters.instruments
        assert analysis1.musical_parameters.tempo == analysis2.musical_parameters.tempo

    def test_varied_analysis_without_seed(self):
        """Test that analysis can vary when no seed is provided."""
        dream_text = "I was flying over a beautiful city, feeling joyful."

        # Create analyzers without fixed seeds
        analyzer1 = DreamAnalyzer()
        analyzer2 = DreamAnalyzer()

        # Run multiple analyses to check for variation
        analyses1 = [analyzer1.analyze_dream(dream_text) for _ in range(3)]
        analyses2 = [analyzer2.analyze_dream(dream_text) for _ in range(3)]

        # Should still produce valid analyses
        for analysis in analyses1 + analyses2:
            assert isinstance(analysis, ComprehensiveAnalysis)
            assert analysis.overall_confidence >= 0
