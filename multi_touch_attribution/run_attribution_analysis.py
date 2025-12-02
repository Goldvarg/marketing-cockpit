"""
Run Shapley Attribution Analysis

This script runs the complete Shapley value attribution analysis and generates
visualizations and reports.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shapley_attribution import ShapleyAttribution
import os

# Configure matplotlib for PNG output (compatible with Git)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['savefig.bbox'] = 'tight'

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


def create_visualizations(summary_df: pd.DataFrame, output_dir: str = '.'):
    """
    Create visualizations for attribution results.

    Args:
        summary_df: Attribution summary DataFrame
        output_dir: Directory to save visualizations
    """
    print("\nGenerating visualizations...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Shapley Values Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(summary_df['channel'], summary_df['shapley_value'])

    # Color bars
    colors = plt.cm.viridis(summary_df['shapley_value'] / summary_df['shapley_value'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel('Shapley Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Channel', fontsize=12, fontweight='bold')
    ax.set_title('Channel Attribution - Shapley Values', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (channel, value) in enumerate(zip(summary_df['channel'], summary_df['shapley_value'])):
        ax.text(value, i, f'  {value:.2f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/shapley_values.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/shapley_values.png")

    # 2. Removal Effects Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(summary_df['channel'], summary_df['removal_effect'])

    # Color bars
    colors = plt.cm.plasma(summary_df['removal_effect'] / summary_df['removal_effect'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel('Removal Effect', fontsize=12, fontweight='bold')
    ax.set_ylabel('Channel', fontsize=12, fontweight='bold')
    ax.set_title('Channel Attribution - Removal Effects', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (channel, value) in enumerate(zip(summary_df['channel'], summary_df['removal_effect'])):
        ax.text(value, i, f'  {value:.2f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/removal_effects.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/removal_effects.png")

    # 3. Comparison: Shapley vs Removal Effect
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Shapley percentage
    ax1.pie(summary_df['shapley_pct'], labels=summary_df['channel'],
            autopct='%1.1f%%', startangle=90)
    ax1.set_title('Shapley Value Distribution', fontsize=12, fontweight='bold')

    # Removal effect percentage
    ax2.pie(summary_df['removal_pct'], labels=summary_df['channel'],
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Removal Effect Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/attribution_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/attribution_comparison.png")

    # 4. Side-by-side comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(summary_df))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], summary_df['shapley_value'],
                   width, label='Shapley Value', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], summary_df['removal_effect'],
                   width, label='Removal Effect', alpha=0.8)

    ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attribution Value', fontsize=12, fontweight='bold')
    ax.set_title('Shapley Value vs Removal Effect by Channel', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['channel'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/shapley_vs_removal.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/shapley_vs_removal.png")


def generate_report(model: ShapleyAttribution, summary_df: pd.DataFrame,
                   interactions_df: pd.DataFrame, output_dir: str = '.'):
    """
    Generate a markdown report with analysis findings.

    Args:
        model: ShapleyAttribution model instance
        summary_df: Attribution summary DataFrame
        interactions_df: Channel interactions DataFrame
        output_dir: Directory to save report
    """
    print("\nGenerating report...")

    report_lines = [
        "# Shapley Value Attribution Analysis Report",
        "",
        "## Executive Summary",
        "",
        f"This report presents the results of Shapley value attribution analysis for marketing channels.",
        f"The analysis is based on {len(model.journeys):,} customer journeys.",
        "",
        "## Key Metrics",
        "",
        f"- **Total Journeys Analyzed**: {len(model.journeys):,}",
        f"- **Converting Journeys**: {model.journeys['converted'].sum():,}",
        f"- **Conversion Rate**: {model.journeys['converted'].mean()*100:.2f}%",
        f"- **Average Touchpoints per Journey**: {model.journeys['num_touchpoints'].mean():.2f}",
        f"- **Total Revenue**: ${model.journeys['revenue'].sum():,.2f}",
        "",
        "## Attribution Results",
        "",
        "### Shapley Values by Channel",
        "",
        "Shapley values represent the fair marginal contribution of each channel:",
        "",
    ]

    # Add summary table
    report_lines.append("| Channel | Shapley Value | % of Total | Removal Effect | % of Total |")
    report_lines.append("|---------|---------------|------------|----------------|------------|")

    for _, row in summary_df.iterrows():
        report_lines.append(
            f"| {row['channel']} | {row['shapley_value']:.2f} | "
            f"{row['shapley_pct']:.1f}% | {row['removal_effect']:.2f} | "
            f"{row['removal_pct']:.1f}% |"
        )

    report_lines.extend([
        "",
        "### Interpretation",
        "",
        "**Shapley Value**: Represents the average marginal contribution of each channel "
        "across all possible combinations and orderings. This is the fairest way to "
        "distribute credit among channels.",
        "",
        "**Removal Effect**: Shows the impact of removing a channel from the full set. "
        "This indicates how much value would be lost if we stopped using that channel.",
        "",
        "## Channel Interactions",
        "",
        "Top performing channel combinations:",
        "",
    ])

    # Add interactions table
    report_lines.append("| Channels | Conversion Value |")
    report_lines.append("|----------|------------------|")

    for _, row in interactions_df.head(10).iterrows():
        report_lines.append(
            f"| {row['channels']} | {row['conversion_value']:.2f} |"
        )

    report_lines.extend([
        "",
        "## Visualizations",
        "",
        "![Shapley Values](shapley_values.png)",
        "",
        "![Removal Effects](removal_effects.png)",
        "",
        "![Attribution Comparison](attribution_comparison.png)",
        "",
        "![Shapley vs Removal](shapley_vs_removal.png)",
        "",
        "## Recommendations",
        "",
        "Based on the Shapley value analysis:",
        "",
    ])

    # Add recommendations based on top channels
    top_channel = summary_df.iloc[0]
    bottom_channel = summary_df.iloc[-1]

    report_lines.extend([
        f"1. **{top_channel['channel']}** has the highest Shapley value ({top_channel['shapley_value']:.2f}), "
        f"indicating it contributes the most to conversions. Consider maintaining or increasing investment.",
        "",
        f"2. **{bottom_channel['channel']}** has the lowest Shapley value ({bottom_channel['shapley_value']:.2f}). "
        f"Evaluate whether this channel is cost-effective or if budget should be reallocated.",
        "",
        "3. Review channel interactions to identify synergies - some channels may work better together.",
        "",
        "4. The removal effects show what would be lost by eliminating each channel. "
        "Use this to make informed decisions about channel mix.",
        "",
        "---",
        "",
        f"*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    # Write report
    report_path = f'{output_dir}/ATTRIBUTION_REPORT.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✓ Saved: {report_path}")


def main():
    """
    Main execution function.
    """
    print("="*80)
    print("SHAPLEY VALUE ATTRIBUTION ANALYSIS")
    print("="*80)

    # Configuration
    OUTPUT_DIR = 'multi_touch_attribution'
    MARKETING_CLICKS_PATH = 'raw_data/marketing_clicks.csv'
    ORDERS_PATH = 'raw_data/orders.csv'
    ATTRIBUTION_WINDOW = 30  # days
    SAMPLE_SIZE = 50000  # Sample size for faster computation

    # Initialize model
    print("\n[1/5] Initializing model...")
    model = ShapleyAttribution()

    # Load data
    print("\n[2/5] Loading data...")
    model.load_data(
        marketing_clicks_path=MARKETING_CLICKS_PATH,
        orders_path=ORDERS_PATH,
        attribution_window_days=ATTRIBUTION_WINDOW
    )

    # Calculate Shapley values
    print("\n[3/5] Calculating Shapley values...")
    shapley_values = model.calculate_shapley_values(
        use_revenue=True,
        sample_size=SAMPLE_SIZE
    )

    # Calculate removal effects
    print("\n[4/5] Calculating removal effects...")
    removal_effects = model.calculate_removal_effects()

    # Get summary and interactions
    summary = model.get_attribution_summary()
    interactions = model.analyze_channel_interactions()

    # Display results
    print("\n" + "="*80)
    print("ATTRIBUTION SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))

    # Save results
    print("\n[5/5] Generating outputs...")
    summary.to_csv(f'{OUTPUT_DIR}/shapley_results.csv', index=False)
    interactions.to_csv(f'{OUTPUT_DIR}/channel_interactions.csv', index=False)
    print(f"✓ Saved: {OUTPUT_DIR}/shapley_results.csv")
    print(f"✓ Saved: {OUTPUT_DIR}/channel_interactions.csv")

    # Create visualizations
    create_visualizations(summary, output_dir=OUTPUT_DIR)

    # Generate report
    generate_report(model, summary, interactions, output_dir=OUTPUT_DIR)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - shapley_results.csv (summary statistics)")
    print("  - channel_interactions.csv (channel combination analysis)")
    print("  - shapley_values.png (Shapley values chart)")
    print("  - removal_effects.png (removal effects chart)")
    print("  - attribution_comparison.png (pie chart comparison)")
    print("  - shapley_vs_removal.png (side-by-side comparison)")
    print("  - ATTRIBUTION_REPORT.md (complete analysis report)")
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
