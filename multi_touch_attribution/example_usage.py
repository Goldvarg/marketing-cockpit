"""
Example Usage of Shapley Attribution Model

This script demonstrates how to use the Shapley attribution model with
simple examples and explanations.
"""

import pandas as pd
import numpy as np
from shapley_attribution import ShapleyAttribution


def create_sample_data():
    """
    Create a small sample dataset for demonstration purposes.

    Returns:
        Tuple of (clicks_df, orders_df)
    """
    # Sample marketing clicks
    clicks_data = [
        # User 1: Google -> Meta -> Conversion
        {'user_id': 'user_1', 'click_ts': '2024-01-01 10:00:00', 'channel': 'Google'},
        {'user_id': 'user_1', 'click_ts': '2024-01-02 14:00:00', 'channel': 'Meta'},

        # User 2: LinkedIn -> Google -> Conversion
        {'user_id': 'user_2', 'click_ts': '2024-01-01 11:00:00', 'channel': 'LinkedIn'},
        {'user_id': 'user_2', 'click_ts': '2024-01-03 15:00:00', 'channel': 'Google'},

        # User 3: Meta only -> Conversion
        {'user_id': 'user_3', 'click_ts': '2024-01-02 12:00:00', 'channel': 'Meta'},

        # User 4: Google -> Meta -> LinkedIn -> Conversion
        {'user_id': 'user_4', 'click_ts': '2024-01-01 09:00:00', 'channel': 'Google'},
        {'user_id': 'user_4', 'click_ts': '2024-01-02 13:00:00', 'channel': 'Meta'},
        {'user_id': 'user_4', 'click_ts': '2024-01-03 16:00:00', 'channel': 'LinkedIn'},

        # User 5: No conversion
        {'user_id': 'user_5', 'click_ts': '2024-01-01 10:00:00', 'channel': 'Google'},
    ]

    # Sample orders
    orders_data = [
        {'user_id': 'user_1', 'order_ts': '2024-01-03 10:00:00', 'revenue': 100},
        {'user_id': 'user_2', 'order_ts': '2024-01-04 10:00:00', 'revenue': 150},
        {'user_id': 'user_3', 'order_ts': '2024-01-03 10:00:00', 'revenue': 80},
        {'user_id': 'user_4', 'order_ts': '2024-01-04 10:00:00', 'revenue': 200},
    ]

    clicks_df = pd.DataFrame(clicks_data)
    orders_df = pd.DataFrame(orders_data)

    return clicks_df, orders_df


def example_1_basic_usage():
    """
    Example 1: Basic usage with sample data
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage with Sample Data")
    print("="*80)

    # Create sample data
    clicks_df, orders_df = create_sample_data()

    # Save to temporary files
    clicks_df.to_csv('/tmp/sample_clicks.csv', index=False)
    orders_df.to_csv('/tmp/sample_orders.csv', index=False)

    print("\nSample Customer Journeys:")
    print("-" * 40)
    print("User 1: Google → Meta → Conversion ($100)")
    print("User 2: LinkedIn → Google → Conversion ($150)")
    print("User 3: Meta → Conversion ($80)")
    print("User 4: Google → Meta → LinkedIn → Conversion ($200)")
    print("User 5: Google → No Conversion")

    # Initialize model
    model = ShapleyAttribution()

    # Load data
    print("\nLoading data...")
    model.load_data(
        marketing_clicks_path='/tmp/sample_clicks.csv',
        orders_path='/tmp/sample_orders.csv',
        attribution_window_days=30
    )

    # Calculate Shapley values
    print("\nCalculating Shapley values...")
    shapley_values = model.calculate_shapley_values(use_revenue=True)

    # Calculate removal effects
    print("\nCalculating removal effects...")
    removal_effects = model.calculate_removal_effects()

    # Get summary
    summary = model.get_attribution_summary()

    print("\n" + "-"*80)
    print("RESULTS:")
    print("-"*80)
    print(summary.to_string(index=False))

    print("\n" + "-"*80)
    print("INTERPRETATION:")
    print("-"*80)
    print("• Shapley Value: Average marginal contribution of each channel")
    print("• Removal Effect: Value lost if channel is removed from the mix")
    print("• Higher values indicate greater importance to conversions")


def example_2_real_data():
    """
    Example 2: Using real data from raw_data folder
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Analysis with Real Data")
    print("="*80)

    # Initialize model
    model = ShapleyAttribution()

    # Load real data
    print("\nLoading real marketing data...")
    try:
        model.load_data(
            marketing_clicks_path='raw_data/marketing_clicks.csv',
            orders_path='raw_data/orders.csv',
            attribution_window_days=30
        )

        print("\nDataset Statistics:")
        print(f"  • Total Journeys: {len(model.journeys):,}")
        print(f"  • Conversions: {model.journeys['converted'].sum():,}")
        print(f"  • Conversion Rate: {model.journeys['converted'].mean()*100:.2f}%")
        print(f"  • Total Revenue: ${model.journeys['revenue'].sum():,.2f}")
        print(f"  • Avg Touchpoints: {model.journeys['num_touchpoints'].mean():.2f}")

        # Calculate with sampling for performance
        print("\nCalculating Shapley values (using sample of 10,000 journeys)...")
        shapley_values = model.calculate_shapley_values(
            use_revenue=True,
            sample_size=10000
        )

        print("\nCalculating removal effects...")
        removal_effects = model.calculate_removal_effects()

        # Get summary
        summary = model.get_attribution_summary()

        print("\n" + "-"*80)
        print("ATTRIBUTION RESULTS:")
        print("-"*80)
        print(summary.to_string(index=False))

        # Analyze top channel combinations
        print("\n" + "-"*80)
        print("TOP CHANNEL COMBINATIONS:")
        print("-"*80)
        interactions = model.analyze_channel_interactions()
        print(interactions.head(10).to_string(index=False))

    except FileNotFoundError as e:
        print(f"\n✗ Error: Could not find data files")
        print(f"  Make sure marketing_clicks.csv and orders.csv exist in raw_data/")
        print(f"  Error details: {e}")


def example_3_conversion_count():
    """
    Example 3: Using conversion count instead of revenue
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Attribution by Conversion Count (not Revenue)")
    print("="*80)

    # Create sample data
    clicks_df, orders_df = create_sample_data()
    clicks_df.to_csv('/tmp/sample_clicks.csv', index=False)
    orders_df.to_csv('/tmp/sample_orders.csv', index=False)

    # Initialize model
    model = ShapleyAttribution()
    model.load_data(
        marketing_clicks_path='/tmp/sample_clicks.csv',
        orders_path='/tmp/sample_orders.csv',
        attribution_window_days=30
    )

    # Calculate using conversion count (not revenue)
    print("\nCalculating Shapley values based on CONVERSION COUNT...")
    shapley_values = model.calculate_shapley_values(use_revenue=False)

    print("\nCalculating removal effects...")
    removal_effects = model.calculate_removal_effects()

    # Get summary
    summary = model.get_attribution_summary()

    print("\n" + "-"*80)
    print("RESULTS (Conversion Count):")
    print("-"*80)
    print(summary.to_string(index=False))

    print("\n" + "-"*80)
    print("NOTE:")
    print("-"*80)
    print("• use_revenue=True  → Attributes based on total revenue")
    print("• use_revenue=False → Attributes based on number of conversions")
    print("• Choose based on your business objective!")


def main():
    """
    Run all examples
    """
    print("\n" + "="*80)
    print("SHAPLEY ATTRIBUTION MODEL - EXAMPLE USAGE")
    print("="*80)

    # Run examples
    example_1_basic_usage()

    print("\n" + "="*80)
    input("Press Enter to continue to Example 2 (real data)...")
    example_2_real_data()

    print("\n" + "="*80)
    input("Press Enter to continue to Example 3 (conversion count)...")
    example_3_conversion_count()

    print("\n" + "="*80)
    print("EXAMPLES COMPLETE!")
    print("="*80)
    print("\nFor full analysis with visualizations, run:")
    print("  python multi_touch_attribution/run_attribution_analysis.py")
    print("\n")


if __name__ == '__main__':
    main()
