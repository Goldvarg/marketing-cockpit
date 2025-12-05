"""
Multi-Touch Attribution Analysis Runner

This script runs the complete attribution analysis:
1. Generates customer journey data
2. Runs all attribution models
3. Creates visualizations
4. Saves results

Just run: python run_analysis.py
"""

import os
import sys

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("\n" + "="*70)
    print("MULTI-TOUCH ATTRIBUTION ANALYSIS")
    print("="*70)

    # Step 1: Generate data
    print("\n" + "-"*70)
    print("STEP 1: Generating Customer Journey Data")
    print("-"*70)

    from generate_data import generate_dataset, save_data, print_summary
    touchpoints_df, conversions_df = generate_dataset(n_users=5000)
    save_data(touchpoints_df, conversions_df)
    print_summary(touchpoints_df, conversions_df)

    # Step 2: Run attribution models
    print("\n" + "-"*70)
    print("STEP 2: Running Attribution Models")
    print("-"*70)

    from attribution_models import AttributionAnalysis
    analysis = AttributionAnalysis()
    analysis.load_data()
    analysis.run_all_models()

    # Get and print summary
    summary = analysis.get_summary_table()
    print("\n" + "="*70)
    print("ATTRIBUTION MODEL COMPARISON (% of Credit)")
    print("="*70)
    print(summary.to_string(index=False))

    # Save detailed results
    os.makedirs('results', exist_ok=True)
    summary.to_csv('results/attribution_comparison.csv', index=False)

    # Print key insight
    print("\n" + "="*70)
    print("KEY INSIGHT: The Hidden Value Problem")
    print("="*70)

    lt = analysis.results['Last Touch']
    sv = analysis.results['Shapley Value']
    lt_total = sum(lt.values())
    sv_total = sum(sv.values())

    print("\nChannels UNDERVALUED by Last Touch Attribution:")
    for channel in ['Display', 'Content', 'Social']:
        if channel in lt and channel in sv:
            lt_pct = lt[channel] / lt_total * 100
            sv_pct = sv[channel] / sv_total * 100
            diff = sv_pct - lt_pct
            if diff > 2:
                print(f"  - {channel}: Gets {lt_pct:.1f}% credit but actually deserves {sv_pct:.1f}%")
                print(f"    (Undervalued by {diff:.1f} percentage points!)")

    print("\nChannels OVERVALUED by Last Touch Attribution:")
    for channel in ['Paid Search', 'Direct', 'Email']:
        if channel in lt and channel in sv:
            lt_pct = lt[channel] / lt_total * 100
            sv_pct = sv[channel] / sv_total * 100
            diff = lt_pct - sv_pct
            if diff > 2:
                print(f"  - {channel}: Gets {lt_pct:.1f}% credit but actually deserves {sv_pct:.1f}%")
                print(f"    (Overvalued by {diff:.1f} percentage points!)")

    # Step 3: Create visualizations
    print("\n" + "-"*70)
    print("STEP 3: Creating Visualizations")
    print("-"*70)

    from visualize_results import create_all_visualizations
    create_all_visualizations(analysis.results)

    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nFiles generated:")
    print("  - touchpoints.csv: Raw touchpoint data")
    print("  - conversions.csv: Conversion data")
    print("  - results/attribution_comparison.csv: Model comparison table")
    print("  - results/attribution_comparison.png: All models bar chart")
    print("  - results/last_vs_shapley.png: Key comparison chart")
    print("  - results/channel_roles.png: Introducer vs Closer analysis")
    print("  - results/model_summary.png: Executive summary chart")

    return analysis


if __name__ == '__main__':
    analysis = main()
