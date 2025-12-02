"""
Generate Dummy Data and Run Shapley Attribution Analysis

This script generates synthetic marketing data and runs the complete
Shapley value attribution analysis for demonstration purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shapley_attribution import ShapleyAttribution
from run_attribution_analysis import create_visualizations, generate_report


def generate_dummy_data(n_users=10000, n_days=180):
    """
    Generate realistic dummy marketing data.

    Args:
        n_users: Number of unique users
        n_days: Number of days of data

    Returns:
        Tuple of (clicks_df, orders_df)
    """
    print("Generating dummy marketing data...")

    np.random.seed(42)

    channels = ['Google', 'Meta', 'LinkedIn', 'Direct', 'Email']
    channel_weights = [0.35, 0.30, 0.15, 0.12, 0.08]  # Different channel volumes

    # Generate clicks
    clicks_data = []
    start_date = datetime(2024, 1, 1)

    for user_id in range(n_users):
        # Number of clicks per user (more realistic distribution)
        n_clicks = np.random.choice([1, 2, 3, 4, 5, 6, 7],
                                     p=[0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02])

        # Generate click timestamps
        user_clicks = []
        for _ in range(n_clicks):
            days_offset = np.random.randint(0, n_days)
            hours_offset = np.random.randint(0, 24)
            click_time = start_date + timedelta(days=days_offset, hours=hours_offset)

            # Select channel (weighted random)
            channel = np.random.choice(channels, p=channel_weights)

            user_clicks.append({
                'user_id': f'user_{user_id}',
                'click_ts': click_time,
                'channel': channel
            })

        # Sort clicks chronologically
        user_clicks.sort(key=lambda x: x['click_ts'])
        clicks_data.extend(user_clicks)

    clicks_df = pd.DataFrame(clicks_data)

    # Generate orders (conversions)
    # Conversion rate depends on channel mix
    conversion_multipliers = {
        'Google': 1.2,
        'Meta': 1.1,
        'LinkedIn': 1.3,
        'Direct': 0.9,
        'Email': 1.0
    }

    orders_data = []
    user_clicks_grouped = clicks_df.groupby('user_id')

    for user_id, user_clicks_df in user_clicks_grouped:
        # Calculate conversion probability based on channels used
        unique_channels = user_clicks_df['channel'].unique()
        base_conversion_rate = 0.15  # 15% base conversion

        # Multi-touch bonus
        multi_touch_bonus = len(unique_channels) * 0.05

        # Channel quality bonus
        channel_bonus = sum([conversion_multipliers.get(ch, 1.0) for ch in unique_channels]) * 0.02

        conversion_prob = min(base_conversion_rate + multi_touch_bonus + channel_bonus, 0.8)

        # Decide if user converts
        if np.random.random() < conversion_prob:
            # Order occurs after last click
            last_click = user_clicks_df['click_ts'].max()
            order_time = last_click + timedelta(hours=np.random.randint(1, 72))

            # Revenue varies by channel mix
            base_revenue = 100
            revenue_multiplier = 1 + (len(unique_channels) * 0.2) + np.random.normal(0, 0.3)
            revenue = max(base_revenue * revenue_multiplier, 20)

            orders_data.append({
                'user_id': user_id,
                'order_ts': order_time,
                'revenue': revenue
            })

    orders_df = pd.DataFrame(orders_data)

    print(f"Generated {len(clicks_df):,} clicks and {len(orders_df):,} orders")
    print(f"Conversion rate: {len(orders_df) / len(clicks_df.groupby('user_id')) * 100:.2f}%")

    return clicks_df, orders_df


def main():
    """
    Generate data and run analysis.
    """
    print("="*80)
    print("SHAPLEY ATTRIBUTION ANALYSIS WITH DUMMY DATA")
    print("="*80)

    OUTPUT_DIR = 'multi_touch_attribution'

    # Generate dummy data
    print("\n[1/6] Generating dummy data...")
    clicks_df, orders_df = generate_dummy_data(n_users=10000, n_days=180)

    # Save to temporary files
    clicks_df.to_csv(f'{OUTPUT_DIR}/temp_clicks.csv', index=False)
    orders_df.to_csv(f'{OUTPUT_DIR}/temp_orders.csv', index=False)

    # Initialize model
    print("\n[2/6] Initializing model...")
    model = ShapleyAttribution()

    # Load data
    print("\n[3/6] Building customer journeys...")
    model.load_data(
        marketing_clicks_path=f'{OUTPUT_DIR}/temp_clicks.csv',
        orders_path=f'{OUTPUT_DIR}/temp_orders.csv',
        attribution_window_days=30
    )

    # Calculate Shapley values
    print("\n[4/6] Calculating Shapley values...")
    shapley_values = model.calculate_shapley_values(
        use_revenue=True,
        sample_size=5000  # Sample for faster computation
    )

    # Calculate removal effects
    print("\n[5/6] Calculating removal effects...")
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
    print("\n[6/6] Generating outputs...")
    summary.to_csv(f'{OUTPUT_DIR}/shapley_results.csv', index=False)
    interactions.to_csv(f'{OUTPUT_DIR}/channel_interactions.csv', index=False)
    print(f"✓ Saved: {OUTPUT_DIR}/shapley_results.csv")
    print(f"✓ Saved: {OUTPUT_DIR}/channel_interactions.csv")

    # Create visualizations
    create_visualizations(summary, output_dir=OUTPUT_DIR)

    # Generate report
    generate_report(model, summary, interactions, output_dir=OUTPUT_DIR)

    # Clean up temporary files
    os.remove(f'{OUTPUT_DIR}/temp_clicks.csv')
    os.remove(f'{OUTPUT_DIR}/temp_orders.csv')

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("\n✓ Done!")

    return model, summary, interactions


if __name__ == '__main__':
    main()
