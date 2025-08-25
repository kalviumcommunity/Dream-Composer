# Zero-Shot Prompting System - Dream Composer

## Overview

The Zero-Shot Prompting system enables AI models to analyze dreams without requiring training examples or few-shot prompts. This approach leverages the pre-trained knowledge of large language models to perform sophisticated dream analysis tasks through carefully crafted instructions and context.

## What is Zero-Shot Prompting?

Zero-shot prompting is a technique where AI models perform tasks they haven't been explicitly trained for by relying on:

- **Pre-trained Knowledge**: Leveraging the model's existing understanding
- **Clear Instructions**: Detailed task descriptions and expectations
- **Structured Context**: Relevant background information and constraints
- **Reasoning Guidance**: Step-by-step approaches to problem-solving

## Features

### 🎯 **Core Zero-Shot Tasks**

1. **Dream Emotion Analysis** - Extract emotions without emotion training data
2. **Musical Style Recommendation** - Suggest music based on dream content
3. **Dream Symbolism Interpretation** - Analyze symbolic elements
4. **Mood-to-Music Mapping** - Convert emotional states to musical parameters
5. **Dream Narrative Analysis** - Examine story structure and pacing

### 🔧 **Key Capabilities**

- **No Training Required**: Works immediately without examples
- **Flexible Analysis**: Adapts to any dream content
- **Structured Output**: Consistent JSON response format
- **Confidence Scoring**: Reliability assessment for each analysis
- **Context Integration**: Incorporates additional dreamer information
- **Validation Framework**: Ensures response quality and format

## Quick Start

### Basic Zero-Shot Analysis

```python
from prompt_structure import ZeroShotDreamAnalyzer, ZeroShotTask

# Initialize analyzer
analyzer = ZeroShotDreamAnalyzer()

# Analyze emotions without training examples
result = analyzer.analyze_single_task(
    ZeroShotTask.DREAM_EMOTION_ANALYSIS,
    "I was flying over a beautiful city, feeling incredibly free and joyful."
)

print(f"Detected emotions: {result.analysis['primary_emotions']}")
print(f"Confidence: {result.confidence}")
```

### Comprehensive Zero-Shot Analysis

```python
# Analyze multiple aspects of a dream
analysis = analyzer.analyze_comprehensive(
    "I found myself in a vast library with books floating in the air. "
    "I could read them just by thinking about them, feeling amazed and curious.",
    additional_context="The dreamer is a researcher and lifelong learner"
)

# Get summary of all analyses
summary = analyzer.get_analysis_summary(analysis)
print(summary)
```

### Building Custom Zero-Shot Prompts

```python
from prompt_structure import ZeroShotPromptBuilder

builder = ZeroShotPromptBuilder()

# Build a zero-shot prompt for musical recommendation
prompt = builder.build_zero_shot_prompt(
    ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
    "I was dancing in a moonlit garden with ethereal beings.",
    additional_context="The dreamer is a professional dancer"
)

# Use with any AI API
system_message = prompt['system_message']
user_message = prompt['user_message']
```

## Architecture

### Zero-Shot Prompt Structure

```
System Message:
├── Expert Role Definition
├── Task Context & Background
├── Analysis Constraints
├── Reasoning Steps
└── Output Format Instructions

User Message:
├── Task Instruction
├── Dream Description
├── Additional Context (optional)
├── Expected JSON Format
└── Analysis Request
```

### Component Relationships

```
ZeroShotDreamAnalyzer
├── ZeroShotPromptBuilder
│   ├── Task-Specific Prompts
│   ├── Reasoning Frameworks
│   └── Validation Logic
├── Response Parser
├── Cache Management
└── Fallback Analysis
```

## Zero-Shot Tasks

### 1. Dream Emotion Analysis

**Purpose**: Extract emotional content without emotion training data

**Output Format**:
```json
{
    "primary_emotions": ["joy", "freedom", "wonder"],
    "emotion_intensities": {"joy": 8, "freedom": 9, "wonder": 7},
    "emotional_progression": "consistent positive emotions throughout",
    "dominant_mood": "euphoric",
    "emotional_triggers": ["flying sensation", "beautiful scenery"],
    "confidence": 0.87
}
```

### 2. Musical Style Recommendation

**Purpose**: Suggest musical parameters based on dream content

**Output Format**:
```json
{
    "recommended_style": "ambient_classical",
    "tempo": {"bpm": 95, "description": "flowing and ethereal"},
    "key_signature": {"key": "C", "mode": "major", "reasoning": "uplifting mood"},
    "instruments": ["strings", "harp", "flute"],
    "dynamics": "mp to mf with gentle crescendos",
    "musical_structure": "through-composed with thematic development",
    "special_techniques": ["legato phrasing", "rubato"],
    "confidence": 0.85
}
```

### 3. Dream Symbolism Interpretation

**Purpose**: Analyze symbolic elements without symbol training

**Output Format**:
```json
{
    "symbols": [
        {
            "element": "flying",
            "interpretation": "desire for freedom and transcendence",
            "psychological_meaning": "escape from limitations",
            "emotional_significance": "liberation and empowerment",
            "musical_relevance": "ascending melodies, light textures"
        }
    ],
    "overall_symbolic_theme": "transformation and growth",
    "archetypal_patterns": ["hero's journey", "transformation"],
    "confidence": 0.82
}
```

## API Reference

### ZeroShotDreamAnalyzer

```python
class ZeroShotDreamAnalyzer:
    def analyze_single_task(
        self, 
        task: ZeroShotTask, 
        dream_text: str,
        additional_context: Optional[str] = None,
        use_cache: bool = True
    ) -> ZeroShotAnalysisResult
    
    def analyze_comprehensive(
        self, 
        dream_text: str,
        tasks: Optional[List[ZeroShotTask]] = None,
        additional_context: Optional[str] = None
    ) -> ComprehensiveZeroShotAnalysis
    
    def get_analysis_summary(self, analysis: ComprehensiveZeroShotAnalysis) -> Dict[str, Any]
    def clear_cache(self) -> None
    def get_cache_stats(self) -> Dict[str, Any]
```

### ZeroShotPromptBuilder

```python
class ZeroShotPromptBuilder:
    def build_zero_shot_prompt(
        self, 
        task: ZeroShotTask, 
        dream_text: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, str]
    
    def get_available_tasks(self) -> List[ZeroShotTask]
    def get_task_description(self, task: ZeroShotTask) -> str
    def validate_response_format(self, task: ZeroShotTask, response: str) -> bool
```

## Data Structures

### ZeroShotAnalysisResult

```python
@dataclass
class ZeroShotAnalysisResult:
    task: ZeroShotTask
    analysis: Dict[str, Any]
    confidence: float
    raw_response: str
    timestamp: str
```

### ComprehensiveZeroShotAnalysis

```python
@dataclass
class ComprehensiveZeroShotAnalysis:
    dream_text: str
    emotion_analysis: Optional[ZeroShotAnalysisResult]
    musical_recommendation: Optional[ZeroShotAnalysisResult]
    symbolism_interpretation: Optional[ZeroShotAnalysisResult]
    mood_mapping: Optional[ZeroShotAnalysisResult]
    narrative_analysis: Optional[ZeroShotAnalysisResult]
    overall_confidence: float
    analysis_timestamp: str
```

## Zero-Shot Prompt Design Principles

### 1. Clear Role Definition
```
"You are an expert in dream emotion analysis with deep understanding 
of psychology, neuroscience, and emotional intelligence."
```

### 2. Rich Context Provision
```
"Dreams often contain complex emotional landscapes that reflect the 
dreamer's subconscious feelings, fears, desires, and experiences."
```

### 3. Specific Constraints
```
- Limit to maximum 5 primary emotions
- Intensity scale: 1-10 (1=barely present, 10=overwhelming)
- Focus on emotions actually present in the dream text
```

### 4. Reasoning Guidance
```
1. Read the dream description carefully
2. Identify explicit emotional words and phrases
3. Analyze implicit emotions from imagery and actions
4. Assess the intensity of each emotion
5. Determine the overall emotional progression
```

### 5. Structured Output Format
```json
{
    "primary_emotions": ["emotion1", "emotion2"],
    "confidence": 0.85
}
```

## Testing

### Comprehensive Test Suite

```bash
# Run zero-shot prompting tests
pytest prompt_structure/tests/test_zero_shot_prompts.py -v
pytest prompt_structure/tests/test_zero_shot_analyzer.py -v

# Run all tests including zero-shot
pytest prompt_structure/tests/ -v
```

### Test Coverage

- **Zero-Shot Prompt Building**: 25 tests
- **Zero-Shot Analysis**: 23 tests
- **Edge Cases**: Empty text, long text, special characters
- **Validation**: Response format validation, error handling
- **Caching**: Cache functionality and statistics
- **Integration**: Component interaction testing

## Demo

Run the demonstration script to see zero-shot prompting in action:

```bash
python demo_zero_shot_prompting.py
```

This demonstrates:
- Individual zero-shot task analysis
- Comprehensive multi-task analysis
- Prompt building and structure
- Response validation
- Cache management
- Real-world dream examples

## AI API Integration

### OpenAI Integration

```python
import openai
from prompt_structure import ZeroShotPromptBuilder, ZeroShotTask

builder = ZeroShotPromptBuilder()
prompt = builder.build_zero_shot_prompt(
    ZeroShotTask.DREAM_EMOTION_ANALYSIS, 
    dream_text
)

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": prompt['system_message']},
        {"role": "user", "content": prompt['user_message']}
    ],
    temperature=0.3  # Lower temperature for more consistent analysis
)

# Validate and parse response
if builder.validate_response_format(ZeroShotTask.DREAM_EMOTION_ANALYSIS, response.choices[0].message.content):
    analysis = json.loads(response.choices[0].message.content)
```

### Anthropic Claude Integration

```python
import anthropic
from prompt_structure import ZeroShotPromptBuilder, ZeroShotTask

client = anthropic.Anthropic(api_key="your-api-key")
builder = ZeroShotPromptBuilder()

prompt = builder.build_zero_shot_prompt(
    ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
    dream_text
)

response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1000,
    system=prompt['system_message'],
    messages=[{"role": "user", "content": prompt['user_message']}]
)
```

## Performance

### Benchmarks

- **Prompt Generation**: <5ms per prompt
- **Response Parsing**: <10ms per response
- **Cache Lookup**: <1ms per cached result
- **Memory Usage**: ~15MB for full system
- **Accuracy**: 80-90% confidence scores for clear dream content

### Optimization Tips

1. **Use Caching**: Enable caching for repeated dream analyses
2. **Batch Processing**: Process multiple dreams in parallel
3. **Context Optimization**: Provide relevant additional context
4. **Temperature Tuning**: Use lower temperatures (0.2-0.4) for consistency
5. **Response Validation**: Always validate AI responses before use

## Advantages of Zero-Shot Prompting

### 🚀 **Immediate Deployment**
- No training data collection required
- No model fine-tuning needed
- Works with any capable language model

### 🎯 **Flexibility**
- Adapts to any dream content
- Handles novel scenarios automatically
- Easy to modify and extend

### 📊 **Consistency**
- Structured output format
- Repeatable analysis framework
- Confidence scoring for reliability

### 💡 **Scalability**
- Works with various AI providers
- Handles high-volume processing
- Easy horizontal scaling

## Future Enhancements

### Planned Features

- **Multi-language Zero-Shot**: Support for dreams in different languages
- **Custom Task Creation**: User-defined zero-shot tasks
- **Prompt Optimization**: Automatic prompt improvement based on results
- **Ensemble Analysis**: Combine multiple zero-shot approaches
- **Real-time Streaming**: Progressive analysis as dreams are typed

### Advanced Techniques

- **Chain-of-Thought**: Enhanced reasoning for complex dreams
- **Self-Consistency**: Multiple analysis passes for improved accuracy
- **Prompt Chaining**: Sequential zero-shot tasks building on each other
- **Dynamic Context**: Adaptive context based on dream content

## Contributing

To contribute to the Zero-Shot Prompting system:

1. **Add New Tasks**: Create new zero-shot analysis tasks
2. **Improve Prompts**: Enhance existing prompt templates
3. **Add Tests**: Comprehensive testing for new features
4. **Documentation**: Update guides and examples
5. **Validation**: Ensure robust response validation

## License

This Zero-Shot Prompting system is part of the Dream Composer project and follows the same MIT license terms.
