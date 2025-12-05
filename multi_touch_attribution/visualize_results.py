"""
Visualization for Multi-Touch Attribution Analysis

Creates charts comparing different attribution models
and highlighting the key insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def plot_attribution_comparison(results, save_path='results/attribution_comparison.png'):
    """Create a grouped bar chart comparing all attribution models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    channels = list(results['Last Touch'].keys())
    models = list(results.keys())

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(channels))
    width = 0.12
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    for i, (model, color) in enumerate(zip(models, colors)):
        values = [results[model].get(ch, 0) for ch in channels]
        total = sum(values)
        percentages = [v / total * 100 for v in values]
        ax.bar(x + i * width, percentages, width, label=model, color=color, alpha=0.85)

    ax.set_ylabel('Share of Credit (%)', fontsize=12)
    ax.set_xlabel('Marketing Channel', fontsize=12)
    ax.set_title('Attribution Model Comparison\nHow Each Model Allocates Credit', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(channels, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 50)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_last_vs_shapley(results, save_path='results/last_vs_shapley.png'):
    """Highlight the difference between Last Touch and Shapley."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lt = results['Last Touch']
    sv = results['Shapley Value']

    channels = list(lt.keys())
    lt_total = sum(lt.values())
    sv_total = sum(sv.values())

    lt_pct = [lt.get(ch, 0) / lt_total * 100 for ch in channels]
    sv_pct = [sv.get(ch, 0) / sv_total * 100 for ch in channels]
    diff = [s - l for s, l in zip(sv_pct, lt_pct)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Bar comparison
    x = np.arange(len(channels))
    width = 0.35

    axes[0].bar(x - width/2, lt_pct, width, label='Last Touch', color='#e74c3c', alpha=0.8)
    axes[0].bar(x + width/2, sv_pct, width, label='Shapley Value', color='#1abc9c', alpha=0.8)
    axes[0].set_ylabel('Share of Credit (%)', fontsize=11)
    axes[0].set_xlabel('Channel', fontsize=11)
    axes[0].set_title('Last Touch vs Shapley Value', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(channels, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Right: Difference chart
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in diff]
    bars = axes[1].barh(channels, diff, color=colors, alpha=0.8)
    axes[1].axvline(x=0, color='black', linewidth=0.8)
    axes[1].set_xlabel('Difference (Shapley - Last Touch) in percentage points', fontsize=10)
    axes[1].set_title('Who Gets More Credit with Multi-Touch?', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, d in zip(bars, diff):
        x_pos = bar.get_width() + 0.5 if d > 0 else bar.get_width() - 0.5
        ha = 'left' if d > 0 else 'right'
        axes[1].text(x_pos, bar.get_y() + bar.get_height()/2,
                    f'{d:+.1f}pp', va='center', ha=ha, fontsize=9)

    # Add legend
    green_patch = mpatches.Patch(color='#2ecc71', label='Undervalued by Last Touch')
    red_patch = mpatches.Patch(color='#e74c3c', label='Overvalued by Last Touch')
    axes[1].legend(handles=[green_patch, red_patch], loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_channel_roles(results, save_path='results/channel_roles.png'):
    """Visualize channels by their role (introducer vs closer)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ft = results['First Touch']
    lt = results['Last Touch']

    ft_total = sum(ft.values())
    lt_total = sum(lt.values())

    channels = list(ft.keys())

    fig, ax = plt.subplots(figsize=(10, 8))

    for ch in channels:
        ft_pct = ft.get(ch, 0) / ft_total * 100
        lt_pct = lt.get(ch, 0) / lt_total * 100

        # Color based on role
        if ft_pct > lt_pct * 1.5:
            color = '#3498db'  # Introducer (blue)
            role = 'Introducer'
        elif lt_pct > ft_pct * 1.5:
            color = '#e74c3c'  # Closer (red)
            role = 'Closer'
        else:
            color = '#95a5a6'  # Balanced (gray)
            role = 'Balanced'

        ax.scatter(ft_pct, lt_pct, s=300, c=color, alpha=0.7, edgecolors='white', linewidth=2)
        ax.annotate(ch, (ft_pct, lt_pct), fontsize=10, ha='center', va='bottom', xytext=(0, 8),
                   textcoords='offset points')

    # Add diagonal line (balanced channels would be on this line)
    max_val = max(max(ft.values()) / ft_total * 100, max(lt.values()) / lt_total * 100) * 1.2
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Balanced')

    ax.set_xlabel('First Touch Credit (%)', fontsize=12)
    ax.set_ylabel('Last Touch Credit (%)', fontsize=12)
    ax.set_title('Channel Roles: Introducers vs Closers\n(Above line = Closers, Below line = Introducers)',
                fontsize=13, fontweight='bold')

    # Add legend
    intro_patch = mpatches.Patch(color='#3498db', label='Introducers (start journeys)')
    closer_patch = mpatches.Patch(color='#e74c3c', label='Closers (end journeys)')
    balanced_patch = mpatches.Patch(color='#95a5a6', label='Balanced')
    ax.legend(handles=[intro_patch, closer_patch, balanced_patch], loc='upper left')

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_model_summary(results, save_path='results/model_summary.png'):
    """Create a summary visualization with all key insights."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(16, 10))

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Plot 1: Pie charts for Last Touch vs Shapley
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    lt = results['Last Touch']
    sv = results['Shapley Value']

    colors = plt.cm.Set3(np.linspace(0, 1, len(lt)))

    # Last Touch Pie
    lt_values = list(lt.values())
    lt_labels = list(lt.keys())
    ax1.pie(lt_values, labels=lt_labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Last Touch Attribution', fontsize=12, fontweight='bold')

    # Shapley Pie
    sv_values = [sv.get(ch, 0) for ch in lt_labels]
    ax2.pie(sv_values, labels=lt_labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Shapley Value Attribution', fontsize=12, fontweight='bold')

    # Plot 2: All models comparison
    ax3 = fig.add_subplot(gs[1, :])

    channels = list(lt.keys())
    models = ['Last Touch', 'First Touch', 'Linear', 'Shapley Value']
    x = np.arange(len(channels))
    width = 0.2
    model_colors = ['#e74c3c', '#3498db', '#f39c12', '#1abc9c']

    for i, (model, color) in enumerate(zip(models, model_colors)):
        values = results[model]
        total = sum(values.values())
        pcts = [values.get(ch, 0) / total * 100 for ch in channels]
        ax3.bar(x + i * width, pcts, width, label=model, color=color, alpha=0.85)

    ax3.set_ylabel('Share of Credit (%)', fontsize=11)
    ax3.set_xlabel('Marketing Channel', fontsize=11)
    ax3.set_title('Model Comparison: Who Gets the Credit?', fontsize=12, fontweight='bold')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(channels, rotation=45, ha='right')
    ax3.legend(loc='upper right')
    ax3.grid(axis='y', alpha=0.3)

    plt.suptitle('Multi-Touch Attribution Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_all_visualizations(results):
    """Generate all visualization charts."""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60 + "\n")

    plot_attribution_comparison(results)
    plot_last_vs_shapley(results)
    plot_channel_roles(results)
    plot_model_summary(results)

    print("\nAll visualizations saved to results/ folder")


if __name__ == '__main__':
    from attribution_models import AttributionAnalysis

    # Run analysis
    analysis = AttributionAnalysis()
    analysis.load_data()
    analysis.run_all_models()

    # Create visualizations
    create_all_visualizations(analysis.results)
