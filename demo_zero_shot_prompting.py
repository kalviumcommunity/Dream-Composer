#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for the Dream Composer Zero-Shot Prompting system.

This script demonstrates how to use zero-shot prompting techniques
to analyze dreams without requiring training examples or few-shot prompts.
"""

import json
import sys
from prompt_structure import ZeroShotDreamAnalyzer, ZeroShotPromptBuilder, ZeroShotTask

# Ensure UTF-8 encoding for console output with safe detection
def configure_utf8_output():
    """Safely configure UTF-8 output with proper error handling."""
    try:
        # Check if stdout has encoding attribute and reconfigure method
        if (hasattr(sys.stdout, 'encoding') and
            hasattr(sys.stdout, 'reconfigure') and
            sys.stdout.encoding is not None and
            sys.stdout.encoding.lower() != 'utf-8'):
            sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError, ValueError):
        # Fallback: continue without reconfiguration
        # This handles cases where reconfigure is not available or fails
        pass

configure_utf8_output()

def safe_print(text: str, fallback_text: str = None) -> None:
    """Print text with emoji fallback for encoding issues."""
    try:
        print(text)
    except UnicodeEncodeError:
        if fallback_text:
            print(fallback_text)
        else:
            safe_text = ''.join(char for char in text if ord(char) < 128)
            print(safe_text)


def main():
    """Main demonstration function."""
    safe_print("🧠 Dream Composer - Zero-Shot Prompting Demo 🧠", 
               "Dream Composer - Zero-Shot Prompting Demo")
    print("=" * 55)
    
    # Sample dream descriptions for zero-shot analysis
    sample_dreams = [
        {
            "title": "Lucid Flying Dream",
            "description": "I realized I was dreaming and decided to fly. I soared above my hometown, feeling incredibly free and powerful. The sensation of flight was so realistic - I could feel the wind and see everything in perfect detail below.",
            "context": "The dreamer practices lucid dreaming techniques"
        },
        {
            "title": "Ocean Depths",
            "description": "I was swimming deep underwater in a vast ocean. Strange, beautiful creatures swam around me, and I could breathe underwater somehow. The water was crystal clear and filled with golden light.",
            "context": "The dreamer is going through a period of emotional exploration"
        },
        {
            "title": "Childhood Home",
            "description": "I found myself back in my childhood bedroom, but everything was slightly different. The walls were painted a color I'd never seen before, and there were doors leading to rooms that never existed. I felt both nostalgic and confused.",
            "context": "The dreamer recently moved to a new city"
        },
        {
            "title": "Musical Performance",
            "description": "I was performing on stage with an orchestra, playing a piece I'd never heard before but somehow knew perfectly. The music was hauntingly beautiful, and the audience was completely silent, mesmerized.",
            "context": "The dreamer is a professional musician"
        }
    ]
    
    # Initialize the zero-shot analyzer
    print("Initializing Zero-Shot Dream Analyzer...")
    analyzer = ZeroShotDreamAnalyzer()
    prompt_builder = ZeroShotPromptBuilder()
    safe_print("✅ Zero-Shot Analyzer ready!\n", "Zero-Shot Analyzer ready!\n")
    
    # Demonstrate individual zero-shot tasks
    print("🎯 Individual Zero-Shot Task Demonstrations")
    print("-" * 45)
    
    dream = sample_dreams[0]  # Use the flying dream
    
    # 1. Emotion Analysis
    safe_print("1. 🎭 Zero-Shot Emotion Analysis", "1. Zero-Shot Emotion Analysis")
    emotion_result = analyzer.analyze_single_task(
        ZeroShotTask.DREAM_EMOTION_ANALYSIS,
        dream["description"],
        additional_context=dream["context"]
    )
    
    print(f"   Dream: {dream['title']}")
    print(f"   Detected Emotions: {emotion_result.analysis.get('primary_emotions', [])}")
    print(f"   Dominant Mood: {emotion_result.analysis.get('dominant_mood', 'unknown')}")
    print(f"   Confidence: {emotion_result.confidence:.2f}")
    print()
    
    # 2. Musical Style Recommendation
    safe_print("2. 🎵 Zero-Shot Musical Recommendation", "2. Zero-Shot Musical Recommendation")
    musical_result = analyzer.analyze_single_task(
        ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
        dream["description"]
    )
    
    musical_analysis = musical_result.analysis
    print(f"   Recommended Style: {musical_analysis.get('recommended_style', 'unknown')}")
    print(f"   Tempo: {musical_analysis.get('tempo', {}).get('bpm', 0)} BPM")
    print(f"   Key: {musical_analysis.get('key_signature', {}).get('key', 'unknown')} {musical_analysis.get('key_signature', {}).get('mode', '')}")
    print(f"   Instruments: {', '.join(musical_analysis.get('instruments', []))}")
    print(f"   Confidence: {musical_result.confidence:.2f}")
    print()
    
    # 3. Symbolism Interpretation
    safe_print("3. 🔮 Zero-Shot Symbolism Interpretation", "3. Zero-Shot Symbolism Interpretation")
    symbol_result = analyzer.analyze_single_task(
        ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION,
        dream["description"]
    )
    
    symbols = symbol_result.analysis.get('symbols', [])
    if symbols:
        for symbol in symbols[:2]:  # Show first 2 symbols
            print(f"   Symbol: {symbol.get('element', 'unknown')}")
            print(f"   Meaning: {symbol.get('interpretation', 'unknown')}")
            print(f"   Musical Relevance: {symbol.get('musical_relevance', 'unknown')}")
            print()
    else:
        print("   No specific symbols detected in this dream.")
        print()
    
    print("=" * 55)
    
    # Demonstrate comprehensive zero-shot analysis
    safe_print("🔬 Comprehensive Zero-Shot Analysis", "Comprehensive Zero-Shot Analysis")
    print("-" * 35)
    
    for i, dream in enumerate(sample_dreams[1:3], 2):  # Analyze 2 more dreams
        safe_print(f"🌙 Dream {i}: {dream['title']}", f"Dream {i}: {dream['title']}")
        print(f"Description: {dream['description'][:100]}...")
        print(f"Context: {dream['context']}")
        print("-" * 40)
        
        # Perform comprehensive zero-shot analysis
        analysis = analyzer.analyze_comprehensive(
            dream['description'],
            additional_context=dream['context']
        )
        
        # Display summary
        summary = analyzer.get_analysis_summary(analysis)
        
        safe_print("📊 Zero-Shot Analysis Results:", "Zero-Shot Analysis Results:")
        
        if 'primary_emotions' in summary:
            print(f"  Emotions: {', '.join(summary['primary_emotions'])}")
        if 'dominant_mood' in summary:
            print(f"  Dominant Mood: {summary['dominant_mood']}")
        if 'recommended_style' in summary:
            print(f"  Musical Style: {summary['recommended_style']}")
        if 'suggested_tempo' in summary:
            print(f"  Suggested Tempo: {summary['suggested_tempo']} BPM")
        if 'key_symbols' in summary:
            symbols = [s for s in summary['key_symbols'] if s]
            if symbols:
                print(f"  Key Symbols: {', '.join(symbols)}")
        
        print(f"  Overall Confidence: {summary['overall_confidence']:.2f}")
        print("\n" + "=" * 55 + "\n")
    
    # Demonstrate zero-shot prompt building
    safe_print("🔧 Zero-Shot Prompt Building Demo", "Zero-Shot Prompt Building Demo")
    print("-" * 35)
    
    print("Available Zero-Shot Tasks:")
    tasks = prompt_builder.get_available_tasks()
    for i, task in enumerate(tasks, 1):
        description = prompt_builder.get_task_description(task)
        print(f"  {i}. {task.value.replace('_', ' ').title()}")
        print(f"     {description[:80]}...")
    print()
    
    # Show example zero-shot prompt structure
    print("Example Zero-Shot Prompt Structure:")
    example_prompt = prompt_builder.build_zero_shot_prompt(
        ZeroShotTask.MOOD_TO_MUSIC_MAPPING,
        sample_dreams[0]["description"]
    )
    
    print("System Message (excerpt):")
    print(f"  {example_prompt['system_message'][:150]}...")
    print()
    print("User Message (excerpt):")
    print(f"  {example_prompt['user_message'][:150]}...")
    print()
    print("Expected Format (excerpt):")
    print(f"  {example_prompt['expected_format'][:100]}...")
    print()
    
    # Show reasoning steps
    print("Reasoning Steps:")
    for i, step in enumerate(example_prompt['reasoning_steps'], 1):
        print(f"  {i}. {step}")
    print()
    
    # Demonstrate cache functionality
    safe_print("💾 Cache Functionality Demo", "Cache Functionality Demo")
    print("-" * 25)
    
    # Show cache stats
    cache_stats = analyzer.get_cache_stats()
    print(f"Cache Size: {cache_stats['cache_size']} analyses")
    print(f"Cached Tasks: {', '.join(cache_stats['cached_tasks'])}")
    
    # Clear cache
    analyzer.clear_cache()
    print("Cache cleared.")
    
    new_stats = analyzer.get_cache_stats()
    print(f"New Cache Size: {new_stats['cache_size']}")
    print()
    
    # Show validation example
    safe_print("✅ Response Validation Demo", "Response Validation Demo")
    print("-" * 25)
    
    # Valid response
    valid_response = json.dumps({
        "primary_emotions": ["wonder", "curiosity"],
        "confidence": 0.87,
        "analysis": "Detailed analysis here"
    })
    
    is_valid = prompt_builder.validate_response_format(
        ZeroShotTask.DREAM_EMOTION_ANALYSIS,
        valid_response
    )
    print(f"Valid response validation: {is_valid}")
    
    # Invalid response
    invalid_response = "This is not JSON"
    is_invalid = prompt_builder.validate_response_format(
        ZeroShotTask.DREAM_EMOTION_ANALYSIS,
        invalid_response
    )
    print(f"Invalid response validation: {is_invalid}")
    print()
    
    safe_print("🎉 Zero-Shot Prompting Demo Complete!", "Zero-Shot Prompting Demo Complete!")
    print("\nKey Benefits of Zero-Shot Prompting:")
    print("• No training examples required")
    print("• Leverages AI model's pre-trained knowledge")
    print("• Flexible and adaptable to new dream types")
    print("• Consistent analysis framework")
    print("• Easy integration with various AI APIs")
    print("\nThe Zero-Shot Prompting system is ready for production use!")


if __name__ == "__main__":
    main()
