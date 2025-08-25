#!/usr/bin/env python3
"""
Dynamic Shot Prompting Demo for Dream Composer.

This demo showcases the dynamic shot prompting capabilities that intelligently
select relevant examples based on dream content, complexity, and context.
"""

import sys
import json
from datetime import datetime

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

from prompt_structure import (
    DynamicShotDreamAnalyzer,
    DynamicShotPromptBuilder,
    DreamComplexity,
    ExampleType,
    DreamExample,
    DynamicShotConfig,
    ZeroShotTask
)


def print_header(title: str, emoji: str = "🎯"):
    """Print a formatted header."""
    print(f"\n{emoji} {title}")
    print("=" * (len(title) + 4))


def print_subheader(title: str, emoji: str = "📊"):
    """Print a formatted subheader."""
    print(f"\n{emoji} {title}")
    print("-" * (len(title) + 4))


def demo_complexity_analysis():
    """Demonstrate dream complexity analysis."""
    print_header("Dream Complexity Analysis", "🧠")
    
    builder = DynamicShotPromptBuilder()
    
    # Sample dreams of varying complexity
    dreams = {
        "Simple": "I was flying over a city, feeling happy and free.",
        "Moderate": "I started in a peaceful garden, but then storm clouds gathered and I felt anxious as the weather changed.",
        "Complex": "I was in a vast library with floating books. I could read them by thinking, feeling amazed and curious. Then I found a golden door that led to my childhood home.",
        "Highly Complex": "The dream began in a bustling marketplace where I was searching for something I couldn't name. Suddenly, I was underwater in a crystal cave, breathing normally but feeling the weight of the ocean above. A wise old turtle spoke to me in my grandmother's voice, telling me secrets about time and memory. Then I was flying through a storm, feeling both terrified and exhilarated, until I landed in a peaceful meadow where all my childhood pets were waiting."
    }
    
    for label, dream in dreams.items():
        complexity = builder.analyze_dream_complexity(dream)
        keywords = builder.extract_keywords(dream)
        
        print(f"\n🌙 {label} Dream:")
        print(f"   Text: {dream[:80]}{'...' if len(dream) > 80 else ''}")
        print(f"   Complexity: {complexity.value.upper()}")
        print(f"   Keywords: {', '.join(keywords[:8])}")


def demo_example_selection():
    """Demonstrate intelligent example selection."""
    print_header("Intelligent Example Selection", "🎯")
    
    builder = DynamicShotPromptBuilder()
    
    # Test different types of dreams
    test_dreams = [
        ("Flying Dream", "I was soaring through clouds, feeling incredibly free and joyful."),
        ("Nightmare", "I was being chased through dark corridors, feeling terrified and helpless."),
        ("Symbolic Dream", "I found a golden key that opened doors to rooms filled with memories."),
        ("Musical Dream", "I was conducting an orchestra in a grand concert hall, feeling powerful and inspired.")
    ]
    
    for dream_type, dream_text in test_dreams:
        print_subheader(f"Example Selection for {dream_type}")
        
        complexity = builder.analyze_dream_complexity(dream_text)
        keywords = builder.extract_keywords(dream_text)
        
        selected_examples = builder.select_dynamic_examples(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            dream_text,
            complexity,
            keywords
        )
        
        print(f"Dream: \"{dream_text}\"")
        print(f"Complexity: {complexity.value}")
        print(f"Keywords: {', '.join(keywords[:6])}")
        print(f"Selected Examples: {len(selected_examples)}")
        
        for i, example in enumerate(selected_examples, 1):
            relevance = example.matches_content(dream_text, keywords)
            print(f"  {i}. {example.dream_text[:60]}... (Relevance: {relevance:.2f})")


def demo_dynamic_shot_analysis():
    """Demonstrate dynamic shot analysis."""
    print_header("Dynamic Shot Analysis", "🔬")
    
    analyzer = DynamicShotDreamAnalyzer()
    
    # Sample dreams for analysis
    dreams = [
        {
            "title": "Lucid Flying Dream",
            "text": "I realized I was dreaming and decided to fly. I soared over a beautiful landscape, feeling completely in control and euphoric.",
            "context": "The dreamer practices lucid dreaming techniques"
        },
        {
            "title": "Emotional Transition",
            "text": "I was at my childhood home, but it kept changing. Sometimes it was bright and welcoming, other times dark and scary. I felt confused but also nostalgic.",
            "context": "The dreamer recently moved to a new city"
        },
        {
            "title": "Symbolic Journey",
            "text": "I was walking through a forest where each tree had doors. Behind each door was a different version of my life. I felt overwhelmed by the choices but also curious about the possibilities.",
            "context": "The dreamer is considering a major career change"
        }
    ]
    
    for dream in dreams:
        print_subheader(f"Analyzing: {dream['title']}")
        
        # Perform comprehensive analysis
        analysis = analyzer.analyze_comprehensive(
            dream["text"],
            additional_context=dream["context"]
        )
        
        print(f"Dream: {dream['text']}")
        print(f"Context: {dream['context']}")
        print(f"Complexity: {analysis.complexity.value.upper()}")
        print(f"Keywords: {', '.join(analysis.keywords[:8])}")
        print(f"Overall Confidence: {analysis.overall_confidence:.2f}")
        print(f"Total Examples Used: {analysis.total_examples_used}")
        
        # Show key findings from each analysis
        if analysis.emotion_analysis:
            emotions = analysis.emotion_analysis.analysis.get("primary_emotions", ["N/A"])
            print(f"Primary Emotions: {', '.join(emotions[:3])}")
        
        if analysis.musical_recommendation:
            style = analysis.musical_recommendation.analysis.get("recommended_style", "N/A")
            print(f"Musical Style: {style}")
        
        if analysis.symbolism_interpretation:
            symbols = analysis.symbolism_interpretation.analysis.get("symbols", [])
            if symbols and isinstance(symbols, list) and len(symbols) > 0:
                symbol_names = [s.get("element", "unknown") for s in symbols[:2]]
                print(f"Key Symbols: {', '.join(symbol_names)}")


def demo_prompt_building():
    """Demonstrate dynamic shot prompt building."""
    print_header("Dynamic Shot Prompt Building", "🔧")
    
    builder = DynamicShotPromptBuilder()
    
    # Build a prompt for a complex dream
    complex_dream = "I was in a vast underwater city with bioluminescent creatures. I could breathe underwater and communicate telepathically with the sea life. I felt a deep sense of belonging and ancient wisdom."
    
    prompt = builder.build_dynamic_shot_prompt(
        ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION,
        complex_dream,
        "The dreamer is interested in marine biology and meditation"
    )
    
    print("Sample Dynamic Shot Prompt Structure:")
    print(f"Task: {prompt['task']}")
    print(f"Complexity: {prompt['complexity']}")
    print(f"Examples Used: {prompt['num_examples']}")
    print(f"Keywords: {', '.join(prompt['keywords'])}")
    
    print("\nSystem Message (excerpt):")
    system_excerpt = prompt['system_message'][:200] + "..."
    print(f"  {system_excerpt}")
    
    print("\nUser Message (excerpt):")
    user_excerpt = prompt['user_message'][:300] + "..."
    print(f"  {user_excerpt}")
    
    print(f"\nSelected Examples:")
    for i, example in enumerate(prompt['selected_examples'], 1):
        print(f"  {i}. {example}")


def demo_performance_metrics():
    """Demonstrate performance metrics and caching."""
    print_header("Performance Metrics & Caching", "📈")
    
    analyzer = DynamicShotDreamAnalyzer()
    
    # Perform multiple analyses
    test_dreams = [
        "I was flying through clouds, feeling free and happy.",
        "I was swimming in a dark ocean, feeling scared but curious.",
        "I was in a library with infinite books, feeling overwhelmed but excited.",
        "I was flying through clouds, feeling free and happy.",  # Duplicate for cache test
    ]
    
    print("Performing analyses...")
    for i, dream in enumerate(test_dreams, 1):
        result = analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            dream
        )
        print(f"  {i}. Analyzed dream (Complexity: {result.complexity.value}, Examples: {result.num_examples_used})")
    
    # Show performance metrics
    metrics = analyzer.get_performance_metrics()
    
    print(f"\nPerformance Metrics:")
    print(f"  Total Analyses: {metrics['total_analyses']}")
    print(f"  Cache Hits: {metrics['cache_hits']}")
    print(f"  Cache Hit Rate: {metrics['cache_hit_rate']:.2%}")
    print(f"  Average Examples Used: {metrics['average_examples_used']:.1f}")
    print(f"  Cache Size: {metrics['cache_size']}")
    
    print(f"\nComplexity Distribution:")
    for complexity, count in metrics['complexity_distribution'].items():
        print(f"  {complexity.title()}: {count}")
    
    # Example database statistics
    db_stats = metrics['example_database_stats']
    print(f"\nExample Database:")
    print(f"  Total Examples: {db_stats['total_examples']}")
    print(f"  Examples by Task:")
    for task, count in db_stats['examples_by_task'].items():
        print(f"    {task.replace('_', ' ').title()}: {count}")


def demo_custom_configuration():
    """Demonstrate custom configuration options."""
    print_header("Custom Configuration", "⚙️")
    
    # Create custom configuration
    custom_config = DynamicShotConfig(
        max_examples=3,
        min_examples=2,
        relevance_threshold=0.4,
        complexity_weight=0.5,
        diversity_weight=0.3,
        recency_weight=0.2
    )
    
    # Create analyzer with custom config
    custom_analyzer = DynamicShotDreamAnalyzer(custom_config)
    
    print("Custom Configuration:")
    print(f"  Max Examples: {custom_config.max_examples}")
    print(f"  Min Examples: {custom_config.min_examples}")
    print(f"  Relevance Threshold: {custom_config.relevance_threshold}")
    print(f"  Complexity Weight: {custom_config.complexity_weight}")
    print(f"  Diversity Weight: {custom_config.diversity_weight}")
    print(f"  Recency Weight: {custom_config.recency_weight}")
    
    # Test with custom configuration
    result = custom_analyzer.analyze_single_task(
        ZeroShotTask.DREAM_EMOTION_ANALYSIS,
        "I was in a complex maze with shifting walls, feeling both frustrated and determined to find the exit."
    )
    
    print(f"\nAnalysis with Custom Config:")
    print(f"  Examples Used: {result.num_examples_used}")
    print(f"  Complexity: {result.complexity.value}")
    print(f"  Confidence: {result.confidence:.2f}")


def demo_example_management():
    """Demonstrate example database management."""
    print_header("Example Database Management", "📚")
    
    builder = DynamicShotPromptBuilder()
    
    # Show initial statistics
    initial_stats = builder.get_example_statistics()
    print("Initial Example Database:")
    print(f"  Total Examples: {initial_stats['total_examples']}")
    
    # Add a new example
    new_example = DreamExample(
        dream_text="I was painting with colors that don't exist in reality, creating a masterpiece that expressed my deepest emotions.",
        analysis={
            "primary_emotions": ["creativity", "fulfillment", "wonder"],
            "emotional_intensity": {"creativity": 9, "fulfillment": 8, "wonder": 7},
            "dominant_mood": "inspired",
            "confidence": 0.91
        },
        example_type=ExampleType.BASIC_EMOTION,
        complexity=DreamComplexity.MODERATE,
        keywords=["painting", "colors", "masterpiece", "emotions", "creativity"]
    )
    
    builder.add_example(ZeroShotTask.DREAM_EMOTION_ANALYSIS, new_example)
    
    # Show updated statistics
    updated_stats = builder.get_example_statistics()
    print(f"\nAfter Adding New Example:")
    print(f"  Total Examples: {updated_stats['total_examples']}")
    print(f"  Examples Added: {updated_stats['total_examples'] - initial_stats['total_examples']}")
    
    print(f"\nExample Types Distribution:")
    for example_type, count in updated_stats['example_types'].items():
        print(f"  {example_type.replace('_', ' ').title()}: {count}")
    
    print(f"\nComplexity Distribution:")
    for complexity, count in updated_stats['complexity_distribution'].items():
        print(f"  {complexity.title()}: {count}")


def main():
    """Run the dynamic shot prompting demo."""
    print_header("Dynamic Shot Prompting Demo for Dream Composer", "🧠")
    print("Demonstrating intelligent example selection and context-aware analysis")
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run all demo sections
        demo_complexity_analysis()
        demo_example_selection()
        demo_dynamic_shot_analysis()
        demo_prompt_building()
        demo_performance_metrics()
        demo_custom_configuration()
        demo_example_management()
        
        print_header("Demo Complete! 🎉", "✅")
        print("\nKey Benefits of Dynamic Shot Prompting:")
        print("• 🎯 Intelligent example selection based on content relevance")
        print("• 🧠 Complexity-aware analysis with appropriate example counts")
        print("• 📊 Performance optimization through smart caching")
        print("• ⚙️ Configurable parameters for different use cases")
        print("• 📚 Expandable example database for improved accuracy")
        print("• 🔄 Adaptive prompting that learns from usage patterns")
        print("\nThe Dynamic Shot Prompting system provides context-aware,")
        print("intelligent analysis that adapts to dream content and complexity!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Please check the implementation and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
