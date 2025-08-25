#!/usr/bin/env python3
"""
One-Shot Prompting Demo for Dream Composer.

This demo showcases the one-shot prompting system that provides exactly one
carefully selected example to guide AI analysis, striking a balance between
zero-shot and few-shot approaches.
"""

import json
from datetime import datetime
from typing import Dict, Any

from prompt_structure.one_shot_analyzer import OneShotDreamAnalyzer
from prompt_structure.one_shot_prompts import OneShotStrategy, OneShotConfig
from prompt_structure.zero_shot_prompts import ZeroShotTask


def print_header(title: str, emoji: str = "🎯"):
    """Print a formatted header."""
    print(f"\n{emoji} {title}")
    print("=" * (len(title) + 4))


def print_subheader(title: str, emoji: str = "📊"):
    """Print a formatted subheader."""
    print(f"\n{emoji} {title}")
    print("-" * (len(title) + 4))


def safe_get_analysis_field(analysis_result, field_name: str, default="Not available"):
    """
    Safely extract a field from analysis results with proper error handling.
    
    Args:
        analysis_result: The analysis result object
        field_name: The field name to extract
        default: Default value if field is not available
        
    Returns:
        The field value or default if not available
    """
    if not analysis_result or not hasattr(analysis_result, 'analysis'):
        return default
    
    try:
        value = analysis_result.analysis.get(field_name)
        if value is None:
            return default
        return value
    except (AttributeError, TypeError):
        return default


def format_list_field(value, max_items: int = 3, separator: str = ", ") -> str:
    """
    Safely format a list field for display.
    
    Args:
        value: The value to format (should be a list)
        max_items: Maximum number of items to display
        separator: Separator between items
        
    Returns:
        Formatted string representation
    """
    if not value:
        return "Not available"
    
    if not isinstance(value, list):
        return str(value)
    
    try:
        formatted_items = [str(item) for item in value[:max_items]]
        return separator.join(formatted_items)
    except (TypeError, ValueError):
        return "Not available"


def demo_strategy_selection():
    """Demonstrate different one-shot selection strategies."""
    print_header("One-Shot Strategy Selection", "🎯")
    
    sample_dreams = [
        "I was flying over a beautiful city at sunset, feeling incredibly free and joyful.",
        "I started in a peaceful garden, but then storm clouds gathered and I felt anxious.",
        "I was swimming deep underwater with glowing fish, feeling peaceful yet mysterious.",
        "I found a golden key that opened doors to rooms filled with memories."
    ]
    
    for i, dream in enumerate(sample_dreams, 1):
        print_subheader(f"Dream {i}: Strategy Comparison", "🌙")
        print(f"Dream: \"{dream[:60]}...\"")
        
        # Create analyzer for strategy comparison
        analyzer = OneShotDreamAnalyzer()
        
        # Compare strategies
        comparison = analyzer.get_strategy_comparison(dream, ZeroShotTask.DREAM_EMOTION_ANALYSIS)
        
        print(f"Complexity: {comparison['complexity']}")
        print(f"Keywords: {', '.join(comparison['keywords'])}")
        print("\nStrategy Selections:")
        
        for strategy, selection in comparison["strategy_selections"].items():
            if "error" not in selection:
                print(f"  {strategy.upper()}: {selection['example_text'][:50]}... (Quality: {selection['example_quality']:.2f})")
            else:
                print(f"  {strategy.upper()}: Error - {selection['error']}")


def demo_one_shot_analysis():
    """Demonstrate one-shot analysis with different strategies."""
    print_header("One-Shot Analysis Demonstration", "🔬")
    
    sample_dreams = [
        ("Lucid Flying Dream", "I realized I was dreaming and decided to fly. I soared over a beautiful landscape, feeling completely in control and euphoric.", "The dreamer practices lucid dreaming techniques"),
        ("Emotional Transition", "I was at my childhood home, but it kept changing. Sometimes it was bright and welcoming, other times dark and scary. I felt confused but also nostalgic.", "The dreamer recently moved to a new city"),
        ("Symbolic Journey", "I was walking through a forest where each tree had doors. Behind each door was a different version of my life. I felt overwhelmed by the choices but also curious about the possibilities.", "The dreamer is considering a major career change")
    ]
    
    strategies = [OneShotStrategy.BEST_MATCH, OneShotStrategy.REPRESENTATIVE, OneShotStrategy.BALANCED]
    
    for strategy in strategies:
        print_subheader(f"Analysis with {strategy.value.upper()} Strategy", "📊")
        
        # Create analyzer with specific strategy
        config = OneShotConfig(strategy=strategy)
        analyzer = OneShotDreamAnalyzer(config)
        
        for dream_name, dream_text, context in sample_dreams:
            print(f"\n🌙 {dream_name}:")
            print(f"   Dream: {dream_text[:80]}...")
            print(f"   Context: {context}")
            
            try:
                # Perform comprehensive analysis
                analysis = analyzer.analyze_comprehensive(dream_text, additional_context=context)
                
                print(f"   Complexity: {analysis.complexity.value}")
                print(f"   Keywords: {', '.join(analysis.keywords[:5])}")
                print(f"   Overall Confidence: {analysis.overall_confidence:.2f}")
                print(f"   Total Example Quality: {analysis.total_example_quality:.2f}")
                
                # Show key findings from each analysis with safe access
                emotions = safe_get_analysis_field(analysis.emotion_analysis, "primary_emotions", [])
                emotions_str = format_list_field(emotions, max_items=3)
                print(f"   Primary Emotions: {emotions_str}")
                
                style = safe_get_analysis_field(analysis.musical_recommendation, "recommended_style")
                print(f"   Musical Style: {style}")
                
                # Handle symbols with special formatting
                symbols = safe_get_analysis_field(analysis.symbolism_interpretation, "symbols", [])
                if symbols and isinstance(symbols, list):
                    try:
                        symbol_names = []
                        for s in symbols[:2]:
                            if isinstance(s, dict) and "element" in s:
                                symbol_names.append(str(s["element"]))
                        symbols_str = format_list_field(symbol_names, max_items=2)
                        print(f"   Key Symbols: {symbols_str}")
                    except (TypeError, KeyError):
                        print(f"   Key Symbols: Not available")
                else:
                    print(f"   Key Symbols: Not available")
                
            except Exception as e:
                print(f"   ❌ Error during analysis: {e}")


def demo_one_shot_prompt_building():
    """Demonstrate one-shot prompt building."""
    print_header("One-Shot Prompt Building", "🔧")
    
    analyzer = OneShotDreamAnalyzer()
    
    sample_dream = "I was conducting an orchestra in a grand concert hall, feeling powerful and inspired."
    
    print_subheader("Sample One-Shot Prompt Structure", "📝")
    print(f"Dream: \"{sample_dream}\"")
    print(f"Task: {ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION.value}")
    
    # Build prompt
    prompt_data = analyzer.prompt_builder.build_one_shot_prompt(
        ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
        sample_dream
    )
    
    print(f"Strategy: {prompt_data['strategy']}")
    print(f"Complexity: {prompt_data['complexity']}")
    print(f"Example Quality: {prompt_data['example_quality']:.2f}")
    print(f"Keywords: {', '.join(prompt_data['keywords'])}")
    
    print("\nSystem Message (excerpt):")
    system_lines = prompt_data['system_message'].split('\n')
    for line in system_lines[:8]:
        print(f"  {line}")
    if len(system_lines) > 8:
        print("  ...")
    
    print("\nUser Message (excerpt):")
    user_lines = prompt_data['user_message'].split('\n')
    for line in user_lines[:12]:
        print(f"  {line}")
    if len(user_lines) > 12:
        print("  ...")
    
    if prompt_data['selected_example']:
        print(f"\nSelected Example:")
        print(f"  {prompt_data['selected_example']}")


def demo_performance_metrics():
    """Demonstrate performance metrics and caching."""
    print_header("Performance Metrics & Caching", "📈")
    
    analyzer = OneShotDreamAnalyzer()
    
    test_dreams = [
        "I was flying through clouds, feeling free and happy.",
        "I was lost in a dark maze, feeling confused and scared.",
        "I was dancing in a field of flowers, feeling joyful and alive.",
        "I was swimming in an ocean of stars, feeling peaceful and wonder."
    ]
    
    print("Performing analyses...")
    for i, dream in enumerate(test_dreams, 1):
        result = analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            dream
        )
        print(f"  {i}. Analyzed dream (Complexity: {result.complexity.value}, Strategy: {result.strategy_used.value}, Quality: {result.example_quality:.2f})")
    
    # Repeat first dream to test caching
    print("  5. Re-analyzing first dream (should use cache)...")
    analyzer.analyze_single_task(
        ZeroShotTask.DREAM_EMOTION_ANALYSIS,
        test_dreams[0]
    )
    
    # Get performance metrics
    metrics = analyzer.get_performance_metrics()
    
    print("\nPerformance Metrics:")
    print(f"  Total Analyses: {metrics['total_analyses']}")
    print(f"  Cache Hits: {metrics['cache_hits']}")
    print(f"  Cache Hit Rate: {metrics['cache_hit_rate']:.2%}")
    print(f"  Average Example Quality: {metrics['average_example_quality']:.2f}")
    print(f"  Cache Size: {metrics['cache_size']}")
    
    print("\nStrategy Usage:")
    for strategy, count in metrics['strategy_usage'].items():
        print(f"  {strategy}: {count}")
    
    print("\nComplexity Distribution:")
    for complexity, count in metrics['complexity_distribution'].items():
        print(f"  {complexity}: {count}")
    
    print("\nExample Database:")
    db_stats = metrics['example_database_stats']
    print(f"  Total Examples: {db_stats['total_examples']}")
    print("  Examples by Task:")
    for task, count in db_stats['examples_by_task'].items():
        print(f"    {task}: {count}")


def demo_custom_configuration():
    """Demonstrate custom configuration options."""
    print_header("Custom Configuration", "⚙️")
    
    # Create custom configuration
    custom_config = OneShotConfig(
        strategy=OneShotStrategy.COMPLEXITY_MATCHED,
        quality_threshold=0.8,
        relevance_weight=0.7,
        representativeness_weight=0.3,
        use_complexity_boost=True,
        fallback_to_representative=True
    )
    
    print("Custom Configuration:")
    print(f"  Strategy: {custom_config.strategy.value}")
    print(f"  Quality Threshold: {custom_config.quality_threshold}")
    print(f"  Relevance Weight: {custom_config.relevance_weight}")
    print(f"  Representativeness Weight: {custom_config.representativeness_weight}")
    print(f"  Use Complexity Boost: {custom_config.use_complexity_boost}")
    print(f"  Fallback to Representative: {custom_config.fallback_to_representative}")
    
    # Create analyzer with custom config
    analyzer = OneShotDreamAnalyzer(custom_config)
    
    sample_dream = "I was exploring a mysterious castle with hidden passages and ancient secrets."
    
    print(f"\nAnalysis with Custom Config:")
    print(f"Dream: \"{sample_dream}\"")
    
    result = analyzer.analyze_single_task(
        ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION,
        sample_dream
    )
    
    print(f"  Strategy Used: {result.strategy_used.value}")
    print(f"  Example Quality: {result.example_quality:.2f}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Complexity: {result.complexity.value}")


def demo_example_database_management():
    """Demonstrate example database management."""
    print_header("Example Database Management", "📚")
    
    analyzer = OneShotDreamAnalyzer()
    
    print("Initial Example Database:")
    stats = analyzer.prompt_builder.get_example_statistics()
    print(f"  Total Examples: {stats['total_examples']}")
    
    print("\nExample Types Distribution:")
    for example_type, count in stats['example_types'].items():
        print(f"  {example_type}: {count}")
    
    print("\nComplexity Distribution:")
    for complexity, count in stats['complexity_distribution'].items():
        print(f"  {complexity}: {count}")
    
    print("\nQuality Distribution:")
    for quality_range, count in stats['quality_distribution'].items():
        print(f"  {quality_range}: {count}")


def main():
    """Run the complete one-shot prompting demo."""
    print_header("One-Shot Prompting Demo for Dream Composer", "🧠")
    print("Demonstrating balanced example-guided analysis with exactly one example")
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        demo_strategy_selection()
        demo_one_shot_analysis()
        demo_one_shot_prompt_building()
        demo_performance_metrics()
        demo_custom_configuration()
        demo_example_database_management()
        
        print_header("Demo Complete! 🎉", "✅")
        print("\nKey Benefits of One-Shot Prompting:")
        print("• 🎯 Balanced guidance with exactly one carefully selected example")
        print("• 🧠 Multiple selection strategies for different use cases")
        print("• 📊 Performance optimization through smart caching")
        print("• ⚙️ Configurable parameters for different scenarios")
        print("• 📚 High-quality example database for consistent guidance")
        print("• 🔄 Strategy comparison for optimal example selection")
        print("\nThe One-Shot Prompting system provides focused, example-guided")
        print("analysis that balances simplicity with intelligent guidance!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Please check the implementation and try again.")


if __name__ == "__main__":
    main()
