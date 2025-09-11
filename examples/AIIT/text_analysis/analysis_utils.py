# SPDX-License-Identifier: Apache-2.0
"""Analysis utilities for structured text insights demonstration.

This module provides helper functions for analyzing and visualizing results
from the structured insights flow. It includes utilities for data processing,
visualization, and comparison of different analysis runs.
"""

# Standard
import json
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Third Party
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import Dataset


def parse_insights_to_dataframe(
    results: Dataset, 
    original_dataset: Optional[Dataset] = None
) -> pd.DataFrame:
    """Parse structured insights results into a pandas DataFrame.
    
    Parameters
    ----------
    results : Dataset
        Results from running the structured insights flow.
    original_dataset : Dataset, optional
        Original dataset with additional metadata (headlines, dates, etc.).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with parsed insights and metadata.
    """
    insights_list = []
    
    for i, result in enumerate(results):
        try:
            # Parse the JSON insights
            if "structured_insights" in result:
                insights = json.loads(result["structured_insights"])
            elif "enhanced_structured_insights" in result:
                insights = json.loads(result["enhanced_structured_insights"])
            else:
                raise ValueError("No insights column found in results")
            
            # Add metadata
            insights["article_id"] = i
            insights["article_length"] = len(result["text"])
            insights["summary_length"] = len(insights.get("summary", ""))
            
            # Add original dataset metadata if available
            if original_dataset and i < len(original_dataset):
                original = original_dataset[i]
                insights["headline"] = original.get("Headline", "")
                insights["date"] = original.get("Date", "")
                insights["journalists"] = original.get("Journalists", [])
                insights["link"] = original.get("Link", "")
            
            insights_list.append(insights)
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse insights for article {i}: {e}")
            continue
    
    return pd.DataFrame(insights_list)


def analyze_sentiment_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze sentiment distribution across articles.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with parsed insights.
        
    Returns
    -------
    Dict[str, Any]
        Sentiment analysis results.
    """
    sentiment_counts = df["sentiment"].value_counts()
    total_articles = len(df)
    
    return {
        "distribution": sentiment_counts.to_dict(),
        "percentages": (sentiment_counts / total_articles * 100).round(1).to_dict(),
        "most_common": sentiment_counts.index[0] if len(sentiment_counts) > 0 else None,
        "total_articles": total_articles
    }


def extract_top_keywords(
    df: pd.DataFrame, 
    top_n: int = 20
) -> List[Tuple[str, int]]:
    """Extract most common keywords across all articles.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with parsed insights.
    top_n : int
        Number of top keywords to return.
        
    Returns
    -------
    List[Tuple[str, int]]
        List of (keyword, count) tuples.
    """
    all_keywords = []
    
    for keywords_str in df["keywords"]:
        if isinstance(keywords_str, str) and keywords_str.strip():
            keywords = [k.strip() for k in keywords_str.split(",")]
            all_keywords.extend(keywords)
    
    keyword_counts = Counter(all_keywords)
    return keyword_counts.most_common(top_n)


def extract_top_entities(
    df: pd.DataFrame, 
    top_n: int = 15
) -> List[Tuple[str, int]]:
    """Extract most common entities across all articles.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with parsed insights.
    top_n : int
        Number of top entities to return.
        
    Returns
    -------
    List[Tuple[str, int]]
        List of (entity, count) tuples.
    """
    all_entities = []
    
    for entities_str in df["entities"]:
        if (isinstance(entities_str, str) 
            and entities_str.strip() 
            and entities_str.lower() != "none"):
            entities = [e.strip() for e in entities_str.split(",")]
            all_entities.extend(entities)
    
    entity_counts = Counter(all_entities)
    return entity_counts.most_common(top_n)


def plot_sentiment_distribution(df: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)):
    """Plot sentiment distribution across articles.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with parsed insights.
    figsize : Tuple[int, int]
        Figure size for the plot.
    """
    sentiment_analysis = analyze_sentiment_distribution(df)
    sentiment_counts = pd.Series(sentiment_analysis["distribution"])
    
    plt.figure(figsize=figsize)
    colors = {"positive": "green", "negative": "red", "neutral": "gray"}
    bar_colors = [colors.get(sentiment, "blue") for sentiment in sentiment_counts.index]
    
    bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors, alpha=0.7)
    
    # Add percentage labels on bars
    for bar, sentiment in zip(bars, sentiment_counts.index):
        height = bar.get_height()
        percentage = sentiment_analysis["percentages"][sentiment]
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f"{percentage}%", ha="center", va="bottom", fontweight="bold")
    
    plt.title("Sentiment Distribution in Financial News", fontsize=14, fontweight="bold")
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Number of Articles", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()


def plot_top_keywords(df: pd.DataFrame, top_n: int = 15, figsize: Tuple[int, int] = (12, 8)):
    """Plot most common keywords as a horizontal bar chart.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with parsed insights.
    top_n : int
        Number of top keywords to show.
    figsize : Tuple[int, int]
        Figure size for the plot.
    """
    top_keywords = extract_top_keywords(df, top_n)
    
    if not top_keywords:
        print("No keywords found to plot.")
        return
    
    keywords, counts = zip(*top_keywords)
    
    plt.figure(figsize=figsize)
    y_pos = range(len(keywords))
    
    bars = plt.barh(y_pos, counts, alpha=0.7)
    plt.yticks(y_pos, keywords)
    plt.xlabel("Frequency", fontsize=12)
    plt.title(f"Top {len(keywords)} Keywords in Financial News", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(count), ha="left", va="center", fontweight="bold")
    
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()


def plot_article_length_vs_summary_length(df: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)):
    """Plot relationship between article length and summary length.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with parsed insights.
    figsize : Tuple[int, int]
        Figure size for the plot.
    """
    plt.figure(figsize=figsize)
    
    plt.scatter(df["article_length"], df["summary_length"], alpha=0.6, s=50)
    
    # Add trend line
    z = np.polyfit(df["article_length"], df["summary_length"], 1)
    p = np.poly1d(z)
    plt.plot(df["article_length"], p(df["article_length"]), "r--", alpha=0.8)
    
    plt.xlabel("Article Length (characters)", fontsize=12)
    plt.ylabel("Summary Length (characters)", fontsize=12)
    plt.title("Summary Length vs Article Length", fontsize=14, fontweight="bold")
    plt.grid(alpha=0.3)
    plt.tight_layout()


def plot_topic_distribution(df: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)):
    """Plot topic distribution (if topic classification was performed).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with parsed insights including topic classification.
    figsize : Tuple[int, int]
        Figure size for the plot.
    """
    if "topic" not in df.columns:
        print("No topic classification found in the data.")
        return
    
    topic_counts = df["topic"].value_counts()
    
    plt.figure(figsize=figsize)
    colors = plt.cm.Set3(range(len(topic_counts)))
    
    bars = plt.bar(topic_counts.index, topic_counts.values, color=colors, alpha=0.7)
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                int(height), ha="center", va="bottom", fontweight="bold")
    
    plt.title("Financial Topic Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Topic Category", fontsize=12)
    plt.ylabel("Number of Articles", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()


def generate_summary_report(df: pd.DataFrame, original_dataset: Optional[Dataset] = None) -> str:
    """Generate a comprehensive text summary report of the analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with parsed insights.
    original_dataset : Dataset, optional
        Original dataset with metadata.
        
    Returns
    -------
    str
        Formatted summary report.
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("STRUCTURED INSIGHTS ANALYSIS REPORT")
    report_lines.append("=" * 60)
    
    # Basic statistics
    report_lines.append(f"\n📊 DATASET OVERVIEW:")
    report_lines.append(f"   • Articles analyzed: {len(df):,}")
    report_lines.append(f"   • Average article length: {df['article_length'].mean():.0f} characters")
    report_lines.append(f"   • Average summary length: {df['summary_length'].mean():.0f} characters")
    
    # Sentiment analysis
    sentiment_analysis = analyze_sentiment_distribution(df)
    report_lines.append(f"\n😊 SENTIMENT ANALYSIS:")
    for sentiment, count in sentiment_analysis["distribution"].items():
        percentage = sentiment_analysis["percentages"][sentiment]
        report_lines.append(f"   • {sentiment.capitalize()}: {count} articles ({percentage}%)")
    
    # Top keywords
    top_keywords = extract_top_keywords(df, 10)
    if top_keywords:
        report_lines.append(f"\n🔑 TOP KEYWORDS:")
        for i, (keyword, count) in enumerate(top_keywords[:5], 1):
            report_lines.append(f"   {i}. {keyword}: {count} occurrences")
    
    # Top entities
    top_entities = extract_top_entities(df, 10)
    if top_entities:
        report_lines.append(f"\n🏷️ TOP ENTITIES:")
        for i, (entity, count) in enumerate(top_entities[:5], 1):
            report_lines.append(f"   {i}. {entity}: {count} occurrences")
    
    # Topic distribution (if available)
    if "topic" in df.columns:
        topic_counts = df["topic"].value_counts()
        report_lines.append(f"\n📈 TOPIC DISTRIBUTION:")
        for topic, count in topic_counts.items():
            percentage = (count / len(df)) * 100
            report_lines.append(f"   • {topic}: {count} articles ({percentage:.1f}%)")
    
    # Date analysis (if available)
    if "date" in df.columns and not df["date"].isna().all():
        report_lines.append(f"\n📅 TEMPORAL ANALYSIS:")
        unique_dates = df["date"].nunique()
        report_lines.append(f"   • Unique dates: {unique_dates}")
        if unique_dates > 1:
            date_range = f"{df['date'].min()} to {df['date'].max()}"
            report_lines.append(f"   • Date range: {date_range}")
    
    report_lines.append("\n" + "=" * 60)
    
    return "\n".join(report_lines)


def compare_basic_vs_enhanced_insights(
    basic_df: pd.DataFrame, 
    enhanced_df: pd.DataFrame
) -> Dict[str, Any]:
    """Compare results between basic and enhanced structured insights flows.
    
    Parameters
    ----------
    basic_df : pd.DataFrame
        Results from basic structured insights flow.
    enhanced_df : pd.DataFrame
        Results from enhanced flow with topic classification.
        
    Returns
    -------
    Dict[str, Any]
        Comparison results.
    """
    comparison = {
        "basic_articles": len(basic_df),
        "enhanced_articles": len(enhanced_df),
        "new_features": [],
        "sentiment_comparison": {},
        "keyword_overlap": 0
    }
    
    # Check for new features in enhanced version
    basic_columns = set(basic_df.columns)
    enhanced_columns = set(enhanced_df.columns)
    new_features = enhanced_columns - basic_columns
    comparison["new_features"] = list(new_features)
    
    # Compare sentiment distributions
    if "sentiment" in basic_df.columns and "sentiment" in enhanced_df.columns:
        basic_sentiment = basic_df["sentiment"].value_counts(normalize=True)
        enhanced_sentiment = enhanced_df["sentiment"].value_counts(normalize=True)
        comparison["sentiment_comparison"] = {
            "basic": basic_sentiment.to_dict(),
            "enhanced": enhanced_sentiment.to_dict()
        }
    
    # Compare keyword overlap (if same articles)
    if len(basic_df) == len(enhanced_df):
        basic_keywords = set()
        enhanced_keywords = set()
        
        for keywords_str in basic_df["keywords"]:
            if isinstance(keywords_str, str):
                basic_keywords.update(k.strip() for k in keywords_str.split(","))
        
        for keywords_str in enhanced_df["keywords"]:
            if isinstance(keywords_str, str):
                enhanced_keywords.update(k.strip() for k in keywords_str.split(","))
        
        if basic_keywords and enhanced_keywords:
            overlap = len(basic_keywords & enhanced_keywords)
            total_unique = len(basic_keywords | enhanced_keywords)
            comparison["keyword_overlap"] = overlap / total_unique if total_unique > 0 else 0
    
    return comparison


def demonstrate_flow_extension(
    original_results: Dataset, 
    enhanced_insights: List[Dict[str, Any]], 
    original_dataset: Optional[Dataset] = None
) -> Dict[str, Any]:
    """Demonstrate the benefits of flow extension with custom blocks.
    
    Parameters
    ----------
    original_results : Dataset
        Results from the original structured insights flow.
    enhanced_insights : List[Dict[str, Any]]
        Enhanced insights with additional custom block outputs.
    original_dataset : Dataset, optional
        Original dataset with metadata.
        
    Returns
    -------
    Dict[str, Any]
        Demonstration results showing extension benefits.
    """
    demo_results = {
        "extension_success": True,
        "original_features": [],
        "new_features": [],
        "example_enhancements": [],
        "processing_summary": {}
    }
    
    # Analyze original results structure
    if len(original_results) > 0:
        original_insight = json.loads(original_results[0]["structured_insights"])
        demo_results["original_features"] = list(original_insight.keys())
    
    # Analyze enhanced results structure
    if enhanced_insights:
        demo_results["new_features"] = list(enhanced_insights[0].keys())
        new_only = set(demo_results["new_features"]) - set(demo_results["original_features"])
        demo_results["added_features"] = list(new_only)
    
    # Create example enhancements
    for i, enhanced in enumerate(enhanced_insights[:3]):  # Show first 3
        example = {
            "article_id": i,
            "original_fields": len(demo_results["original_features"]),
            "enhanced_fields": len(demo_results["new_features"]),
            "new_capabilities": []
        }
        
        # Highlight new capabilities
        if "topic" in enhanced:
            example["new_capabilities"].append(f"Topic Classification: {enhanced['topic']}")
        if "topic_confidence" in enhanced:
            example["new_capabilities"].append(f"Confidence Scoring: {enhanced['topic_confidence']:.3f}")
        
        demo_results["example_enhancements"].append(example)
    
    # Processing summary
    demo_results["processing_summary"] = {
        "articles_processed": len(enhanced_insights),
        "extension_method": "Dynamic custom block integration",
        "maintained_compatibility": True,
        "added_domain_knowledge": "topic" in demo_results["new_features"]
    }
    
    return demo_results


def validate_custom_block_integration(results: Dataset) -> Dict[str, Any]:
    """Validate that custom block integration was successful.
    
    Parameters
    ----------
    results : Dataset
        Results from processing with custom blocks.
        
    Returns
    -------
    Dict[str, Any]
        Validation results.
    """
    validation = {
        "integration_successful": False,
        "custom_fields_present": [],
        "data_quality": {},
        "recommendations": []
    }
    
    # Check if custom fields are present
    if len(results) > 0:
        sample_data = results[0]
        
        # Look for topic-related fields (from FinancialTopicBlock)
        custom_fields = []
        if "topic" in sample_data:
            custom_fields.append("topic")
        if "topic_confidence" in sample_data:
            custom_fields.append("topic_confidence")
        
        validation["custom_fields_present"] = custom_fields
        validation["integration_successful"] = len(custom_fields) > 0
    
    # Data quality checks
    if validation["integration_successful"]:
        # Check topic distribution
        topics = [result.get("topic", "Unknown") for result in results]
        topic_counts = Counter(topics)
        
        validation["data_quality"] = {
            "unique_topics": len(set(topics)),
            "most_common_topic": topic_counts.most_common(1)[0] if topic_counts else None,
            "classification_coverage": sum(1 for t in topics if t != "Unknown") / len(topics)
        }
        
        # Generate recommendations
        if validation["data_quality"]["classification_coverage"] < 0.8:
            validation["recommendations"].append(
                "Consider lowering confidence threshold for better coverage"
            )
        if validation["data_quality"]["unique_topics"] == 1:
            validation["recommendations"].append(
                "Review keyword patterns - all articles classified as same topic"
            )
        if not validation["recommendations"]:
            validation["recommendations"].append("Custom block integration looks good!")
    
    return validation


def plot_enhancement_comparison(basic_df: pd.DataFrame, enhanced_df: pd.DataFrame):
    """Create side-by-side comparison plots of basic vs enhanced results.
    
    Parameters
    ----------
    basic_df : pd.DataFrame
        Results from basic structured insights flow.
    enhanced_df : pd.DataFrame
        Results from enhanced flow with custom blocks.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Sentiment comparison
    axes[0, 0].set_title("Sentiment Distribution Comparison")
    basic_sentiment = basic_df["sentiment"].value_counts()
    enhanced_sentiment = enhanced_df["sentiment"].value_counts()
    
    x = range(len(basic_sentiment))
    width = 0.35
    axes[0, 0].bar([i - width/2 for i in x], basic_sentiment.values, 
                   width, label="Basic", alpha=0.7)
    axes[0, 0].bar([i + width/2 for i in x], enhanced_sentiment.values, 
                   width, label="Enhanced", alpha=0.7)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(basic_sentiment.index)
    axes[0, 0].legend()
    axes[0, 0].set_ylabel("Number of Articles")
    
    # Topic distribution (if available)
    if "topic" in enhanced_df.columns:
        axes[0, 1].set_title("Topic Distribution (Enhanced Only)")
        topic_counts = enhanced_df["topic"].value_counts()
        axes[0, 1].bar(topic_counts.index, topic_counts.values, color="skyblue", alpha=0.7)
        axes[0, 1].set_ylabel("Number of Articles")
        axes[0, 1].tick_params(axis='x', rotation=45)
    else:
        axes[0, 1].text(0.5, 0.5, "No topic data available", 
                        ha="center", va="center", transform=axes[0, 1].transAxes)
        axes[0, 1].set_title("Topic Distribution (Not Available)")
    
    # Summary length comparison
    axes[1, 0].set_title("Summary Length Comparison")
    basic_summary_lengths = [len(s) for s in basic_df["summary"]]
    enhanced_summary_lengths = [len(s) for s in enhanced_df["summary"]]
    
    axes[1, 0].hist(basic_summary_lengths, alpha=0.7, label="Basic", bins=10)
    axes[1, 0].hist(enhanced_summary_lengths, alpha=0.7, label="Enhanced", bins=10)
    axes[1, 0].set_xlabel("Summary Length (characters)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()
    
    # Feature count comparison
    axes[1, 1].set_title("Feature Count Comparison")
    basic_features = len(basic_df.columns)
    enhanced_features = len(enhanced_df.columns)
    
    categories = ["Basic Flow", "Enhanced Flow"]
    feature_counts = [basic_features, enhanced_features]
    bars = axes[1, 1].bar(categories, feature_counts, color=["lightcoral", "lightblue"])
    
    # Add value labels on bars
    for bar, count in zip(bars, feature_counts):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        str(count), ha="center", va="bottom", fontweight="bold")
    
    axes[1, 1].set_ylabel("Number of Features")
    
    plt.tight_layout()
    plt.show()


def create_extension_tutorial_summary() -> str:
    """Create a summary of the flow extension tutorial.
    
    Returns
    -------
    str
        Formatted tutorial summary.
    """
    summary = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                    FLOW EXTENSION TUTORIAL SUMMARY                   ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║ 🎯 WHAT WE DEMONSTRATED:                                             ║
    ║                                                                      ║
    ║   ✅ Custom Block Creation                                           ║
    ║      • Built FinancialTopicBlock from scratch                       ║
    ║      • Implemented rule-based classification logic                  ║
    ║      • Used SDG Hub BaseBlock patterns                              ║
    ║                                                                      ║
    ║   ✅ Dynamic Flow Extension                                          ║
    ║      • Loaded existing structured insights flow                     ║
    ║      • Added custom block without modifying original flow           ║
    ║      • Combined results from multiple processing stages             ║
    ║                                                                      ║
    ║   ✅ Enhanced Output Generation                                      ║
    ║      • Extended JSON structure with new fields                      ║
    ║      • Maintained backward compatibility                            ║
    ║      • Added domain-specific financial topic classification         ║
    ║                                                                      ║
    ║   ✅ Integration Validation                                          ║
    ║      • Compared basic vs enhanced results                           ║
    ║      • Verified data quality and coverage                           ║
    ║      • Demonstrated extensibility benefits                          ║
    ║                                                                      ║
    ║ 🚀 KEY BENEFITS:                                                     ║
    ║                                                                      ║
    ║   • Modular Architecture: Easy to add new capabilities              ║
    ║   • Runtime Flexibility: No need to modify core flows               ║
    ║   • Domain Adaptation: Customize for specific use cases             ║
    ║   • Quality Assurance: Built-in validation and testing              ║
    ║                                                                      ║
    ║ 📚 WHAT YOU LEARNED:                                                 ║
    ║                                                                      ║
    ║   1. How to create custom transform blocks                           ║
    ║   2. How to integrate blocks with existing flows                     ║
    ║   3. How to extend JSON output structures                            ║
    ║   4. How to validate and test extensions                             ║
    ║   5. How to compare and analyze enhanced results                     ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    return summary


# Additional imports for numpy functions
try:
    import numpy as np
except ImportError:
    # Fallback implementation for polyfit if numpy is not available
    def np_polyfit_fallback(x, y, deg):
        """Simple linear regression fallback."""
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        return [slope, intercept]
    
    class np:
        @staticmethod
        def polyfit(x, y, deg):
            return np_polyfit_fallback(x, y, deg)
        
        @staticmethod
        def poly1d(coeffs):
            def poly_func(x_vals):
                return [coeffs[0] * x + coeffs[1] for x in x_vals]
            return poly_func