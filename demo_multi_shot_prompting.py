#!/usr/bin/env python3
"""
Multi-Shot Prompting Demo for Dream Composer.

This demo showcases the multi-shot prompting system that provides multiple
carefully selected examples to guide AI analysis, offering comprehensive
guidance through diverse examples that cover different aspects and scenarios.
"""

import json
from datetime import datetime
from typing import Dict, Any

from prompt_structure.multi_shot_analyzer import MultiShotDreamAnalyzer
from prompt_structure.multi_shot_prompts import MultiShotStrategy, MultiShotConfig
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
    """Demonstrate different multi-shot selection strategies."""
    print_header("Multi-Shot Strategy Selection", "🎯")
    
    sample_dreams = [
        "I was flying over a beautiful city at sunset, feeling incredibly free and joyful.",
        "I started in a peaceful garden, but then storm clouds gathered and I felt anxious.",
        "I was in a vast library where books floated and rearranged themselves, feeling curious then overwhelmed.",
        "I found a golden key that opened doors to rooms filled with memories and emotions."
    ]
    
    for i, dream in enumerate(sample_dreams, 1):
        print_subheader(f"Dream {i}: Strategy Comparison", "🌙")
        print(f"Dream: \"{dream[:70]}...\"")
        
        # Create analyzer for strategy comparison
        analyzer = MultiShotDreamAnalyzer()
        
        # Compare strategies
        comparison = analyzer.get_strategy_comparison(dream, ZeroShotTask.DREAM_EMOTION_ANALYSIS)
        
        print(f"Complexity: {comparison['complexity']}")
        print(f"Keywords: {', '.join(comparison['keywords'])}")
        print("\nStrategy Selections:")
        
        for strategy, selection in comparison["strategy_selections"].items():
            if "error" not in selection:
                print(f"  {strategy.upper()}:")
                print(f"    Examples: {selection['num_examples']}")
                print(f"    Diversity: {selection['diversity_score']:.2f}")
                print(f"    Avg Quality: {selection['average_quality']:.2f}")
                if selection['example_texts']:
                    print(f"    First Example: {selection['example_texts'][0][:50]}...")
            else:
                print(f"  {strategy.upper()}: Error - {selection['error']}")


def demo_multi_shot_analysis():
    """Demonstrate multi-shot analysis with different strategies."""
    print_header("Multi-Shot Analysis Demonstration", "🔬")
    
    sample_dreams = [
        ("Complex Emotional Journey", "I was in my childhood home, but it kept changing. Sometimes bright and welcoming, other times dark and scary. I felt confused, nostalgic, then finally at peace.", "The dreamer recently lost a parent"),
        ("Symbolic Transformation", "I was a caterpillar crawling through a maze of mirrors. Each reflection showed a different stage of becoming a butterfly. I felt anticipation mixed with fear of change.", "The dreamer is considering a major career change"),
        ("Musical Dream Sequence", "I was conducting an orchestra where each instrument represented a different emotion. The music started chaotic but gradually became harmonious as I learned to balance them.", "The dreamer is a music therapist")
    ]
    
    strategies = [MultiShotStrategy.DIVERSE_COVERAGE, MultiShotStrategy.COMPLEXITY_PROGRESSION, MultiShotStrategy.BALANCED_REPRESENTATION]
    
    for strategy in strategies:
        print_subheader(f"Analysis with {strategy.value.upper()} Strategy", "📊")
        
        # Create analyzer with specific strategy
        config = MultiShotConfig(strategy=strategy, min_examples=2, max_examples=4)
        analyzer = MultiShotDreamAnalyzer(config)
        
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
                print(f"   Total Examples Used: {analysis.total_examples_used}")
                print(f"   Average Diversity: {analysis.average_diversity_score:.2f}")
                print(f"   Overall Quality: {analysis.overall_example_quality:.2f}")
                
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


def demo_multi_shot_prompt_building():
    """Demonstrate multi-shot prompt building."""
    print_header("Multi-Shot Prompt Building", "🔧")
    
    analyzer = MultiShotDreamAnalyzer()
    
    sample_dream = "I was conducting an orchestra in a grand concert hall, but the musicians were all different versions of myself from different ages."
    
    print_subheader("Sample Multi-Shot Prompt Structure", "📝")
    print(f"Dream: \"{sample_dream}\"")
    print(f"Task: {ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION.value}")
    
    # Build prompt
    prompt_data = analyzer.prompt_builder.build_multi_shot_prompt(
        ZeroShotTask.MUSICAL_STYLE_RECOMMENDATION,
        sample_dream
    )
    
    print(f"Strategy: {prompt_data['strategy']}")
    print(f"Complexity: {prompt_data['complexity']}")
    print(f"Number of Examples: {prompt_data['num_examples']}")
    print(f"Diversity Score: {prompt_data['diversity_score']:.2f}")
    print(f"Average Quality: {prompt_data['average_quality']:.2f}")
    print(f"Keywords: {', '.join(prompt_data['keywords'])}")
    
    print("\nSelected Examples:")
    for i, example in enumerate(prompt_data['selected_examples'], 1):
        print(f"  {i}. {example}")
    
    print("\nSystem Message (excerpt):")
    system_lines = prompt_data['system_message'].split('\n')
    for line in system_lines[:10]:
        print(f"  {line}")
    if len(system_lines) > 10:
        print("  ...")
    
    print("\nUser Message (excerpt):")
    user_lines = prompt_data['user_message'].split('\n')
    for line in user_lines[:15]:
        print(f"  {line}")
    if len(user_lines) > 15:
        print("  ...")


def demo_diversity_and_quality_metrics():
    """Demonstrate diversity and quality metrics."""
    print_header("Diversity & Quality Metrics", "📈")
    
    analyzer = MultiShotDreamAnalyzer()
    
    test_dreams = [
        "I was flying through clouds of different colors, each representing a different emotion.",
        "I was lost in a dark forest, but found a path lit by glowing stones.",
        "I was dancing in a ballroom where the music changed with my movements.",
        "I was swimming in an ocean of stars, feeling both small and infinite."
    ]
    
    print("Performing analyses with different strategies...")
    
    strategies = [MultiShotStrategy.DIVERSE_COVERAGE, MultiShotStrategy.QUALITY_RANKED, MultiShotStrategy.BALANCED_REPRESENTATION]
    
    for strategy in strategies:
        print_subheader(f"Strategy: {strategy.value.upper()}", "🎯")
        
        config = MultiShotConfig(strategy=strategy, min_examples=2, max_examples=4)
        strategy_analyzer = MultiShotDreamAnalyzer(config)
        
        total_examples = 0
        total_diversity = 0.0
        total_quality = 0.0
        analyses_count = 0
        
        for i, dream in enumerate(test_dreams, 1):
            result = strategy_analyzer.analyze_single_task(
                ZeroShotTask.DREAM_EMOTION_ANALYSIS,
                dream
            )
            
            total_examples += result.num_examples_used
            total_diversity += result.diversity_score
            total_quality += result.average_example_quality
            analyses_count += 1
            
            print(f"  Dream {i}: {result.num_examples_used} examples, "
                  f"diversity {result.diversity_score:.2f}, "
                  f"quality {result.average_example_quality:.2f}")
        
        avg_examples = total_examples / analyses_count
        avg_diversity = total_diversity / analyses_count
        avg_quality = total_quality / analyses_count
        
        print(f"  Strategy Averages:")
        print(f"    Examples per analysis: {avg_examples:.1f}")
        print(f"    Diversity score: {avg_diversity:.2f}")
        print(f"    Example quality: {avg_quality:.2f}")


def demo_performance_metrics():
    """Demonstrate performance metrics and caching."""
    print_header("Performance Metrics & Caching", "📊")
    
    analyzer = MultiShotDreamAnalyzer()
    
    test_dreams = [
        "I was exploring a castle with rooms that defied physics.",
        "I was painting with colors that had sounds and emotions.",
        "I was reading a book where the words came alive and acted out the story.",
        "I was gardening with plants that grew into musical instruments."
    ]
    
    print("Performing analyses...")
    for i, dream in enumerate(test_dreams, 1):
        result = analyzer.analyze_single_task(
            ZeroShotTask.DREAM_EMOTION_ANALYSIS,
            dream
        )
        print(f"  {i}. Analyzed dream (Examples: {result.num_examples_used}, "
              f"Diversity: {result.diversity_score:.2f}, "
              f"Quality: {result.average_example_quality:.2f})")
    
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
    print(f"  Average Examples Used: {metrics['average_examples_used']:.1f}")
    print(f"  Average Diversity Score: {metrics['average_diversity_score']:.2f}")
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
    
    print("  Diversity Metrics by Task:")
    for task, diversity in db_stats['diversity_metrics'].items():
        print(f"    {task}: {diversity:.2f}")


def demo_custom_configuration():
    """Demonstrate custom configuration options."""
    print_header("Custom Configuration", "⚙️")
    
    # Create custom configuration
    custom_config = MultiShotConfig(
        strategy=MultiShotStrategy.COMPLEXITY_PROGRESSION,
        min_examples=3,
        max_examples=5,
        quality_threshold=0.8,
        diversity_weight=0.5,
        relevance_weight=0.3,
        quality_weight=0.2,
        complexity_spread=True,
        type_diversity=True
    )
    
    print("Custom Configuration:")
    print(f"  Strategy: {custom_config.strategy.value}")
    print(f"  Min Examples: {custom_config.min_examples}")
    print(f"  Max Examples: {custom_config.max_examples}")
    print(f"  Quality Threshold: {custom_config.quality_threshold}")
    print(f"  Diversity Weight: {custom_config.diversity_weight}")
    print(f"  Relevance Weight: {custom_config.relevance_weight}")
    print(f"  Quality Weight: {custom_config.quality_weight}")
    print(f"  Complexity Spread: {custom_config.complexity_spread}")
    print(f"  Type Diversity: {custom_config.type_diversity}")
    
    # Create analyzer with custom config
    analyzer = MultiShotDreamAnalyzer(custom_config)
    
    sample_dream = "I was climbing a spiral staircase that seemed to go on forever, with each level showing a different period of my life."
    
    print(f"\nAnalysis with Custom Config:")
    print(f"Dream: \"{sample_dream}\"")
    
    result = analyzer.analyze_single_task(
        ZeroShotTask.DREAM_SYMBOLISM_INTERPRETATION,
        sample_dream
    )
    
    print(f"  Strategy Used: {result.strategy_used.value}")
    print(f"  Examples Used: {result.num_examples_used}")
    print(f"  Diversity Score: {result.diversity_score:.2f}")
    print(f"  Average Quality: {result.average_example_quality:.2f}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Complexity: {result.complexity.value}")


def main():
    """Run the complete multi-shot prompting demo."""
    print_header("Multi-Shot Prompting Demo for Dream Composer", "🧠")
    print("Demonstrating comprehensive example-guided analysis with multiple examples")
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        demo_strategy_selection()
        demo_multi_shot_analysis()
        demo_multi_shot_prompt_building()
        demo_diversity_and_quality_metrics()
        demo_performance_metrics()
        demo_custom_configuration()
        
        print_header("Demo Complete! 🎉", "✅")
        print("\nKey Benefits of Multi-Shot Prompting:")
        print("• 🎯 Comprehensive guidance through multiple carefully selected examples")
        print("• 🧠 Six strategic selection methods for different analysis requirements")
        print("• 📊 Advanced diversity and quality metrics for optimal example selection")
        print("• ⚙️ Highly configurable parameters for different deployment scenarios")
        print("• 📚 Rich example database with sophisticated selection algorithms")
        print("• 🔄 Strategy comparison for optimal multi-example selection")
        print("• 📈 Performance optimization through intelligent caching and metrics")
        print("\nThe Multi-Shot Prompting system provides comprehensive, diverse")
        print("example-guided analysis for superior AI performance!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Please check the implementation and try again.")


if __name__ == "__main__":
    main()
