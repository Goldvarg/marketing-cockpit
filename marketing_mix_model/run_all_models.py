"""
Marketing Mix Model Comparison

This script runs all three MMM approaches on the same dataset and
compares their results to help you understand the tradeoffs.

Models compared:
1. Simple Regression MMM - Easy to understand, fast, manual tuning
2. Google Meridian MMM - Bayesian approach with uncertainty estimates
3. Robyn-Style MMM - Automated hyperparameter optimization

Run this script to generate all results and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

# Add models directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_all_models():
    """Run all three MMM models and generate comparison."""

    print("\n" + "="*70)
    print("MARKETING MIX MODEL COMPARISON")
    print("="*70)

    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'mmm_data.csv')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path, parse_dates=['date'])
    channels = ['Google', 'Meta', 'LinkedIn', 'TV', 'Email']

    # Load ground truth
    truth_path = os.path.join(base_dir, 'data', 'ground_truth_parameters.csv')
    truth_df = pd.read_csv(truth_path)
    ground_truth = dict(zip(truth_df['channel'], truth_df['true_roi']))

    results = {}
    timings = {}

    # =========================================================================
    # Model 1: Simple Regression
    # =========================================================================
    print("\n" + "-"*70)
    print("MODEL 1: SIMPLE REGRESSION MMM")
    print("-"*70)

    from models.simple_regression_mmm import SimpleRegressionMMM

    output_dir = os.path.join(results_dir, 'simple_regression')
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    model1 = SimpleRegressionMMM()
    model1.fit(df, channels)
    timings['Simple Regression'] = time.time() - start_time

    summary1 = model1.get_results_summary(df)
    summary1.to_csv(os.path.join(output_dir, 'results_summary.csv'), index=False)
    model1.plot_contribution(df, save_path=os.path.join(output_dir, 'channel_contribution.png'))
    model1.plot_roi_comparison(df, ground_truth, save_path=os.path.join(output_dir, 'roi_comparison.png'))

    results['Simple Regression'] = {
        'roi': model1.calculate_roi(df),
        'contribution': model1.get_channel_contribution(df),
        'summary': summary1
    }

    # =========================================================================
    # Model 2: Robyn-Style (run before Meridian as it's faster)
    # =========================================================================
    print("\n" + "-"*70)
    print("MODEL 2: ROBYN-STYLE MMM")
    print("-"*70)

    from models.robyn_style_mmm import RobynStyleMMM

    output_dir = os.path.join(results_dir, 'robyn')
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    model2 = RobynStyleMMM()
    model2.fit(df, channels, n_iterations=30)
    timings['Robyn-Style'] = time.time() - start_time

    summary2 = model2.get_results_summary(df)
    summary2.to_csv(os.path.join(output_dir, 'results_summary.csv'), index=False)
    model2.plot_contribution(df, save_path=os.path.join(output_dir, 'channel_contribution.png'))
    model2.plot_roi_comparison(df, ground_truth, save_path=os.path.join(output_dir, 'roi_comparison.png'))
    model2.plot_response_curves(df, save_path=os.path.join(output_dir, 'response_curves.png'))

    results['Robyn-Style'] = {
        'roi': model2.get_roi_estimates(df),
        'contribution': model2.decomposition['channels'],
        'summary': summary2
    }

    # =========================================================================
    # Model 3: Google Meridian
    # =========================================================================
    print("\n" + "-"*70)
    print("MODEL 3: GOOGLE MERIDIAN MMM")
    print("-"*70)

    try:
        from models.meridian_mmm import MeridianMMM

        output_dir = os.path.join(results_dir, 'meridian')
        os.makedirs(output_dir, exist_ok=True)

        start_time = time.time()
        model3 = MeridianMMM()
        model3.fit(df, channels, n_chains=4, n_adapt=500, n_burnin=250, n_keep=500)
        timings['Meridian'] = time.time() - start_time

        model3.plot_model_fit(save_path=os.path.join(output_dir, 'model_fit.png'))
        model3.plot_channel_contribution(save_path=os.path.join(output_dir, 'channel_contribution.png'))
        model3.plot_roi_comparison(save_path=os.path.join(output_dir, 'roi_comparison.png'))
        model3.plot_response_curves(save_path=os.path.join(output_dir, 'response_curves.png'))
        model3.plot_rhat_diagnostics(save_path=os.path.join(output_dir, 'convergence_diagnostics.png'))

        results['Meridian'] = {
            'model': model3,
            'note': 'See visualizations for detailed results'
        }

    except Exception as e:
        print(f"\nMeridian model encountered an issue: {e}")
        print("Continuing with comparison of available models...")
        timings['Meridian'] = None

    # =========================================================================
    # Generate Comparison
    # =========================================================================
    print("\n" + "="*70)
    print("GENERATING MODEL COMPARISON")
    print("="*70)

    generate_comparison(results, ground_truth, channels, timings, results_dir)

    print("\n" + "="*70)
    print("ALL MODELS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {results_dir}/")
    print("\nFiles generated:")
    print("  - simple_regression/ - Simple regression model results")
    print("  - robyn/ - Robyn-style model results")
    print("  - meridian/ - Meridian model results")
    print("  - comparison/ - Cross-model comparison")


def generate_comparison(results, ground_truth, channels, timings, results_dir):
    """Generate comparison visualizations and summary."""

    comp_dir = os.path.join(results_dir, 'comparison')
    os.makedirs(comp_dir, exist_ok=True)

    # =========================================================================
    # ROI Comparison Chart
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(channels))
    width = 0.2
    offset = 0

    # Ground truth
    truth_values = [ground_truth.get(ch, 0) for ch in channels]
    ax.bar(x - 1.5*width, truth_values, width, label='Ground Truth', color='#2ecc71', alpha=0.9)

    # Model results
    colors = ['#3498db', '#e74c3c', '#9b59b6']
    model_names = ['Simple Regression', 'Robyn-Style', 'Meridian']

    for i, model_name in enumerate(model_names):
        if model_name in results and 'roi' in results[model_name]:
            roi = results[model_name]['roi']
            values = [roi.get(ch, 0) for ch in channels]
            ax.bar(x + (i - 0.5) * width, values, width, label=model_name, color=colors[i], alpha=0.8)

    ax.set_ylabel('ROI (Revenue per $1 Spend)', fontsize=12)
    ax.set_xlabel('Channel', fontsize=12)
    ax.set_title('ROI Comparison Across Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(truth_values) * 1.3)

    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, 'roi_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: comparison/roi_comparison.png")

    # =========================================================================
    # Contribution Comparison
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    model_data = [
        ('Simple Regression', results.get('Simple Regression', {}).get('contribution', {})),
        ('Robyn-Style', results.get('Robyn-Style', {}).get('contribution', {})),
    ]

    colors = plt.cm.Set3(np.linspace(0, 1, len(channels)))

    for idx, (name, contribution) in enumerate(model_data):
        if contribution:
            values = [contribution.get(ch, 0) for ch in channels]
            axes[idx].pie(values, labels=channels, autopct='%1.1f%%', colors=colors)
            axes[idx].set_title(name, fontsize=12, fontweight='bold')
        else:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[idx].set_title(name, fontsize=12)

    # Third pie for ground truth spend allocation
    spend_truth = [5000, 4000, 2000, 8000, 500]  # From data generation
    axes[2].pie(spend_truth, labels=channels, autopct='%1.1f%%', colors=colors)
    axes[2].set_title('Actual Spend Allocation', fontsize=12, fontweight='bold')

    plt.suptitle('Channel Contribution Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, 'contribution_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: comparison/contribution_comparison.png")

    # =========================================================================
    # ROI Error Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    errors = {}
    for model_name in ['Simple Regression', 'Robyn-Style']:
        if model_name in results and 'roi' in results[model_name]:
            roi = results[model_name]['roi']
            error = []
            for ch in channels:
                estimated = roi.get(ch, 0)
                actual = ground_truth.get(ch, 0)
                pct_error = abs(estimated - actual) / actual * 100 if actual > 0 else 0
                error.append(pct_error)
            errors[model_name] = np.mean(error)

    if errors:
        models = list(errors.keys())
        error_values = list(errors.values())

        bars = ax.bar(models, error_values, color=['#3498db', '#e74c3c'])
        ax.set_ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
        ax.set_title('Model Accuracy: ROI Estimation Error', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, error_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, 'error_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: comparison/error_comparison.png")

    # =========================================================================
    # Generate Summary Table
    # =========================================================================
    summary_data = []

    for model_name in ['Simple Regression', 'Robyn-Style']:
        if model_name in results and 'roi' in results[model_name]:
            roi = results[model_name]['roi']
            for ch in channels:
                estimated = roi.get(ch, 0)
                actual = ground_truth.get(ch, 0)
                error = abs(estimated - actual) / actual * 100 if actual > 0 else 0

                summary_data.append({
                    'Model': model_name,
                    'Channel': ch,
                    'Estimated ROI': round(estimated, 3),
                    'True ROI': actual,
                    'Error %': round(error, 1)
                })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(comp_dir, 'detailed_comparison.csv'), index=False)
    print("Saved: comparison/detailed_comparison.csv")

    # =========================================================================
    # Timing Comparison
    # =========================================================================
    timing_data = []
    for model, t in timings.items():
        if t is not None:
            timing_data.append({'Model': model, 'Runtime (seconds)': round(t, 1)})

    if timing_data:
        timing_df = pd.DataFrame(timing_data)
        timing_df.to_csv(os.path.join(comp_dir, 'timing_comparison.csv'), index=False)
        print("Saved: comparison/timing_comparison.csv")

        print("\nModel Runtimes:")
        for row in timing_data:
            print(f"  {row['Model']}: {row['Runtime (seconds)']}s")


if __name__ == '__main__':
    run_all_models()
