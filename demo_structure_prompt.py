#!/usr/bin/env python3
"""
Demo script for the Dream Composer Structure Prompt system.

This script demonstrates how to use the structured prompting system
to analyze dream descriptions and generate musical parameters.
"""

import json
from prompt_structure import DreamAnalyzer, PromptBuilder, PromptType


def main():
    """Main demonstration function."""
    print("🎶 Dream Composer - Structure Prompt Demo 🎶")
    print("=" * 50)
    
    # Sample dream descriptions
    sample_dreams = [
        {
            "title": "Flying Dream",
            "description": "I was flying over a shimmering city at sunset, feeling free and peaceful. The golden light made everything look magical."
        },
        {
            "title": "Ocean Adventure", 
            "description": "I found myself swimming in crystal clear blue water with colorful fish. I felt calm and refreshed, like I was one with the ocean."
        },
        {
            "title": "Dark Forest",
            "description": "I was walking through a dark, mysterious forest. Strange sounds echoed around me, and I felt anxious and uncertain about what lay ahead."
        },
        {
            "title": "Childhood Memory",
            "description": "I was back in my childhood home, playing piano while golden sunlight streamed through the windows. I felt nostalgic and warm."
        }
    ]
    
    # Initialize the dream analyzer
    print("Initializing Dream Analyzer...")
    analyzer = DreamAnalyzer()
    print("✅ Dream Analyzer ready!\n")
    
    # Analyze each dream
    for i, dream in enumerate(sample_dreams, 1):
        print(f"🌙 Dream {i}: {dream['title']}")
        print(f"Description: {dream['description']}")
        print("-" * 40)
        
        # Perform comprehensive analysis
        analysis = analyzer.analyze_dream(dream['description'])
        
        # Display results
        print("📊 Analysis Results:")
        
        # Emotions
        emotions = analysis.emotion_result.primary_emotions
        mood = analysis.emotion_result.overall_mood
        print(f"  Emotions: {', '.join(emotions)}")
        print(f"  Overall Mood: {mood}")
        
        # Musical Parameters
        musical = analysis.musical_parameters
        print(f"  Suggested Tempo: {musical.tempo['bpm']} BPM ({musical.tempo['description']})")
        print(f"  Key Signature: {musical.key['signature']}")
        print(f"  Instruments: {', '.join(musical.instruments)}")
        print(f"  Style: {musical.style}")
        
        # Symbols
        if analysis.symbols:
            symbols = [s.element for s in analysis.symbols]
            print(f"  Symbolic Elements: {', '.join(symbols)}")
        
        # Sensory Details
        if analysis.sensory.colors:
            print(f"  Colors: {', '.join(analysis.sensory.colors)}")
        if analysis.sensory.sounds:
            print(f"  Sounds: {', '.join(analysis.sensory.sounds)}")
        print(f"  Atmosphere: {analysis.sensory.atmosphere}")
        
        # Confidence
        print(f"  Confidence Score: {analysis.overall_confidence:.2f}")
        
        print("\n" + "=" * 50 + "\n")
    
    # Demonstrate prompt building
    print("🔧 Prompt Building Demo")
    print("-" * 30)
    
    prompt_builder = PromptBuilder()
    
    # Build different types of prompts
    dream_text = sample_dreams[0]['description']
    
    print("1. Emotion Extraction Prompt:")
    emotion_prompt = prompt_builder.build_prompt(PromptType.EMOTION_EXTRACTION, dream_text)
    print(f"System Message: {emotion_prompt['system_message'][:100]}...")
    print(f"User Message: {emotion_prompt['user_message'][:150]}...")
    print()
    
    print("2. Music Mapping Prompt:")
    music_prompt = prompt_builder.build_prompt(
        PromptType.MUSIC_MAPPING,
        dream_text,
        emotions=["joy", "peace"],
        mood="euphoric",
        intensity=8
    )
    print(f"System Message: {music_prompt['system_message'][:100]}...")
    print(f"User Message: {music_prompt['user_message'][:150]}...")
    print()
    
    # Show prompt statistics
    stats = prompt_builder.get_prompt_statistics()
    print("📈 Prompt System Statistics:")
    print(f"  Total Templates: {stats['total_templates']}")
    print(f"  Available Types: {', '.join(stats['template_types'])}")
    print(f"  Templates with Examples: {stats['templates_with_examples']}")
    print()
    
    # Demonstrate analysis summary
    print("📋 Analysis Summary Example:")
    analysis = analyzer.analyze_dream(sample_dreams[0]['description'])
    summary = analyzer.get_analysis_summary(analysis)
    
    print(json.dumps(summary, indent=2))
    print()
    
    print("🎉 Demo Complete!")
    print("The Structure Prompt system is ready for integration with AI/NLP APIs!")


if __name__ == "__main__":
    main()
