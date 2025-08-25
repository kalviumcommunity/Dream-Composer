# Structure Prompt System - Dream Composer

## Overview

The Structure Prompt system is a comprehensive framework for analyzing dream descriptions and converting them into musical parameters using structured AI prompts. This system forms the core NLP and analysis engine of the Dream Composer application.

## Features

### 🎯 **Core Components**

1. **PromptBuilder** - Creates structured prompts for AI/NLP APIs
2. **EmotionExtractor** - Extracts emotions and emotional patterns from dreams
3. **MusicMapper** - Maps emotions to specific musical parameters
4. **DreamAnalyzer** - Orchestrates comprehensive dream analysis

### 🔧 **Key Capabilities**

- **Emotion Analysis**: Identifies primary emotions, intensity, and progression
- **Musical Mapping**: Converts emotions to tempo, key, instruments, and style
- **Symbol Interpretation**: Analyzes symbolic elements in dreams
- **Sensory Extraction**: Identifies colors, sounds, textures, and atmosphere
- **Narrative Analysis**: Examines dream structure, pacing, and flow
- **Structured Prompting**: Generates consistent AI prompts for analysis

## Quick Start

### Basic Usage

```python
from prompt_structure import DreamAnalyzer

# Initialize analyzer
analyzer = DreamAnalyzer()

# Analyze a dream
dream_text = "I was flying over a beautiful city, feeling free and joyful."
analysis = analyzer.analyze_dream(dream_text)

# Get results
print(f"Emotions: {analysis.emotion_result.primary_emotions}")
print(f"Tempo: {analysis.musical_parameters.tempo['bpm']} BPM")
print(f"Key: {analysis.musical_parameters.key['signature']}")
print(f"Instruments: {analysis.musical_parameters.instruments}")
```

### Prompt Building

```python
from prompt_structure import PromptBuilder, PromptType

builder = PromptBuilder()

# Build emotion extraction prompt
prompt = builder.build_prompt(
    PromptType.EMOTION_EXTRACTION,
    "I was flying through clouds, feeling peaceful."
)

# Use with AI API
system_message = prompt['system_message']
user_message = prompt['user_message']
```

## Architecture

### Component Relationships

```
DreamAnalyzer
├── EmotionExtractor
│   └── PromptBuilder
├── MusicMapper
└── Symbol/Sensory Analysis
```

### Data Flow

1. **Input**: Dream description text
2. **Emotion Extraction**: Identify emotions and mood
3. **Musical Mapping**: Convert emotions to musical parameters
4. **Symbol Analysis**: Interpret symbolic elements
5. **Comprehensive Analysis**: Combine all results
6. **Output**: Complete analysis with musical recommendations

## API Reference

### DreamAnalyzer

```python
class DreamAnalyzer:
    def analyze_dream(self, dream_text: str) -> ComprehensiveAnalysis
    def get_analysis_summary(self, analysis: ComprehensiveAnalysis) -> Dict[str, Any]
```

### EmotionExtractor

```python
class EmotionExtractor:
    def extract_emotions(self, dream_text: str) -> EmotionResult
    def get_emotion_statistics(self, results: List[EmotionResult]) -> Dict[str, Any]
    def validate_emotion_result(self, result: EmotionResult) -> bool
```

### MusicMapper

```python
class MusicMapper:
    def map_emotions_to_music(self, emotion_result: EmotionResult, dream_text: str = "") -> MusicalParameters
```

### PromptBuilder

```python
class PromptBuilder:
    def build_prompt(self, prompt_type: PromptType, dream_text: str, **kwargs) -> Dict[str, str]
    def get_template(self, prompt_type: PromptType) -> PromptTemplate
    def validate_prompt_structure(self, prompt_type: PromptType, response: str) -> bool
```

## Data Structures

### EmotionResult

```python
@dataclass
class EmotionResult:
    primary_emotions: List[str]
    emotional_intensity: Dict[str, float]
    emotional_progression: str
    overall_mood: str
    confidence_score: float
    raw_response: str
```

### MusicalParameters

```python
@dataclass
class MusicalParameters:
    tempo: Dict[str, Any]           # BPM and description
    key: Dict[str, str]             # Signature and mode
    time_signature: str             # e.g., "4/4"
    instruments: List[str]          # Recommended instruments
    style: str                      # Musical style/genre
    dynamics: Dict[str, Any]        # Volume and expression
    harmonic_progression: List[str] # Chord progression
    expression_marks: List[str]     # Performance instructions
    confidence_score: float         # Analysis confidence
```

### ComprehensiveAnalysis

```python
@dataclass
class ComprehensiveAnalysis:
    dream_text: str
    emotion_result: EmotionResult
    musical_parameters: MusicalParameters
    symbols: List[SymbolInterpretation]
    narrative: NarrativeStructure
    sensory: SensoryDetails
    overall_confidence: float
    analysis_timestamp: str
```

## Configuration

### Emotion Keywords

The system includes predefined emotion keywords for fallback analysis:

- **Joy**: happy, joyful, elated, cheerful, delighted, blissful, ecstatic
- **Sadness**: sad, melancholy, melancholic, sorrowful, gloomy, depressed
- **Fear**: afraid, scared, terrified, anxious, worried, nervous, frightened
- **Peace**: peaceful, calm, serene, tranquil, relaxed, content, harmonious
- **And more...**

### Musical Mappings

#### Emotion to Tempo
- **Joy**: 110-140 BPM
- **Peace**: 60-90 BPM
- **Sadness**: 70-100 BPM
- **Fear**: 100-130 BPM

#### Emotion to Key
- **Joy**: C Major, G Major, D Major
- **Peace**: F Major, C Major, Bb Major
- **Sadness**: A Minor, D Minor, E Minor
- **Fear**: B Minor, F# Minor, C# Minor

### Symbol Database

Common dream symbols and their interpretations:

- **Flying**: Freedom, liberation, transcendence
- **Water**: Emotions, subconscious, flow of life
- **Falling**: Loss of control, anxiety, transition
- **Light**: Clarity, hope, divine presence
- **Darkness**: Unknown, mystery, hidden aspects

## Testing

The system includes comprehensive tests:

```bash
# Run all structure prompt tests
pytest prompt_structure/tests/ -v

# Run specific component tests
pytest prompt_structure/tests/test_dream_analyzer.py -v
pytest prompt_structure/tests/test_emotion_extractor.py -v
pytest prompt_structure/tests/test_prompt_builder.py -v
```

### Test Coverage

- **58 total tests** covering all components
- **Emotion extraction**: 22 tests
- **Dream analysis**: 19 tests  
- **Prompt building**: 17 tests
- **Edge cases**: Empty text, special characters, long text
- **Validation**: Input validation, error handling
- **Integration**: Component interaction testing

## Demo

Run the demonstration script to see the system in action:

```bash
python demo_structure_prompt.py
```

This will analyze sample dreams and show:
- Emotion extraction results
- Musical parameter recommendations
- Symbol and sensory analysis
- Prompt building examples
- System statistics

## Integration with AI APIs

The structured prompts are designed for integration with:

- **OpenAI GPT models**
- **Hugging Face transformers**
- **Anthropic Claude**
- **Google PaLM**
- **Custom NLP models**

### Example Integration

```python
import openai
from prompt_structure import PromptBuilder, PromptType

builder = PromptBuilder()
prompt = builder.build_prompt(PromptType.EMOTION_EXTRACTION, dream_text)

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt['system_message']},
        {"role": "user", "content": prompt['user_message']}
    ]
)

# Parse and validate response
if builder.validate_prompt_structure(PromptType.EMOTION_EXTRACTION, response.choices[0].message.content):
    # Process valid response
    pass
```

## Performance

### Benchmarks

- **Analysis Speed**: ~50ms per dream (without AI API calls)
- **Memory Usage**: ~10MB for full system initialization
- **Accuracy**: 85%+ confidence scores for clear emotional content
- **Scalability**: Handles dreams from 10 to 10,000+ characters

### Optimization Tips

1. **Caching**: Cache analysis results for repeated dreams
2. **Batch Processing**: Process multiple dreams in batches
3. **Async Processing**: Use async/await for AI API calls
4. **Memory Management**: Clear large analysis objects when done

## Future Enhancements

### Planned Features

- **Multi-language Support**: Analyze dreams in different languages
- **Custom Emotion Models**: Train domain-specific emotion classifiers
- **Advanced Symbolism**: Expand symbol database with cultural variations
- **Real-time Analysis**: Stream processing for live dream input
- **Visualization**: Generate visual representations of analysis results

### API Improvements

- **Streaming Analysis**: Progressive analysis as text is typed
- **Confidence Tuning**: Adjustable confidence thresholds
- **Custom Mappings**: User-defined emotion-to-music mappings
- **Export Formats**: MIDI, MusicXML, ABC notation output

## Contributing

To contribute to the Structure Prompt system:

1. **Add Tests**: All new features must include comprehensive tests
2. **Documentation**: Update this guide for new features
3. **Validation**: Ensure all prompts produce valid, parseable responses
4. **Performance**: Benchmark new features for performance impact
5. **Compatibility**: Maintain backward compatibility with existing APIs

## License

This Structure Prompt system is part of the Dream Composer project and follows the same MIT license terms.
