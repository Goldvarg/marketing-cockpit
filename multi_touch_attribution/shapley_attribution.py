"""
Shapley Value Attribution Model for Multi-Touch Marketing Attribution

This module implements Shapley value calculation for determining the marginal
contribution of each marketing channel in a customer's conversion journey.

The Shapley value provides a fair allocation of credit to each touchpoint by
considering all possible orderings and combinations of channels.
"""

import pandas as pd
import numpy as np
from itertools import combinations, permutations
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class ShapleyAttribution:
    """
    Implements Shapley Value attribution for multi-touch marketing attribution.

    The Shapley value calculates the marginal contribution of each channel by:
    1. Considering all possible subsets of channels
    2. Calculating the marginal contribution when adding each channel
    3. Averaging across all possible orderings

    Attributes:
        journeys (pd.DataFrame): Customer journeys with touchpoints and conversions
        channels (List[str]): List of unique marketing channels
        conversion_rate (Dict): Conversion rates for each subset of channels
    """

    def __init__(self):
        self.journeys = None
        self.channels = []
        self.conversion_rate = {}
        self.shapley_values = {}
        self.removal_effects = {}

    def load_data(self, marketing_clicks_path: str, orders_path: str,
                  attribution_window_days: int = 30):
        """
        Load and prepare data for Shapley value calculation.

        Args:
            marketing_clicks_path: Path to marketing clicks CSV
            orders_path: Path to orders/conversions CSV
            attribution_window_days: Window for attributing clicks to conversions
        """
        print("Loading data...")

        # Load marketing clicks
        clicks_df = pd.read_csv(marketing_clicks_path)
        clicks_df['click_ts'] = pd.to_datetime(clicks_df['click_ts'])

        # Load orders/conversions
        orders_df = pd.read_csv(orders_path)
        orders_df['order_ts'] = pd.to_datetime(orders_df['order_ts'])

        print(f"Loaded {len(clicks_df):,} clicks and {len(orders_df):,} orders")

        # Build customer journeys
        self.journeys = self._build_journeys(
            clicks_df, orders_df, attribution_window_days
        )

        # Extract unique channels
        self.channels = sorted(self.journeys['channels'].explode().unique().tolist())

        print(f"Built {len(self.journeys):,} customer journeys")
        print(f"Unique channels: {self.channels}")

        return self.journeys

    def _build_journeys(self, clicks_df: pd.DataFrame, orders_df: pd.DataFrame,
                        window_days: int) -> pd.DataFrame:
        """
        Build customer journeys by linking clicks to conversions.

        Args:
            clicks_df: DataFrame with marketing clicks
            orders_df: DataFrame with orders/conversions
            window_days: Attribution window in days

        Returns:
            DataFrame with customer journeys
        """
        journeys = []

        # Group clicks by user
        clicks_grouped = clicks_df.sort_values('click_ts').groupby('user_id')

        for user_id, user_clicks in clicks_grouped:
            # Get user's orders
            user_orders = orders_df[orders_df['user_id'] == user_id].sort_values('order_ts')

            if len(user_orders) == 0:
                # Non-converting journey
                journey_channels = user_clicks['channel'].tolist()
                journeys.append({
                    'user_id': user_id,
                    'channels': journey_channels,
                    'converted': False,
                    'revenue': 0,
                    'num_touchpoints': len(journey_channels)
                })
            else:
                # For each conversion, attribute clicks within window
                for _, order in user_orders.iterrows():
                    order_time = order['order_ts']

                    # Get clicks within attribution window
                    window_clicks = user_clicks[
                        (user_clicks['click_ts'] <= order_time) &
                        (user_clicks['click_ts'] >= order_time - pd.Timedelta(days=window_days))
                    ]

                    if len(window_clicks) > 0:
                        journey_channels = window_clicks['channel'].tolist()
                        journeys.append({
                            'user_id': user_id,
                            'channels': journey_channels,
                            'converted': True,
                            'revenue': order['revenue'],
                            'num_touchpoints': len(journey_channels),
                            'order_ts': order_time
                        })

        return pd.DataFrame(journeys)

    def calculate_shapley_values(self, use_revenue: bool = True,
                                 sample_size: int = None) -> Dict[str, float]:
        """
        Calculate Shapley values for each marketing channel.

        Args:
            use_revenue: If True, use revenue; if False, use conversion count
            sample_size: If provided, sample journeys for faster computation

        Returns:
            Dictionary mapping channel to Shapley value
        """
        print("\nCalculating Shapley values...")

        # Sample journeys if requested
        if sample_size and len(self.journeys) > sample_size:
            print(f"Sampling {sample_size:,} journeys for computation")
            journeys_sample = self.journeys.sample(n=sample_size, random_state=42)
        else:
            journeys_sample = self.journeys

        # Calculate conversion value for each subset of channels
        print("Computing conversion values for all channel subsets...")
        self.conversion_rate = self._calculate_conversion_values(
            journeys_sample, use_revenue
        )

        # Calculate Shapley value for each channel
        print("Computing Shapley values...")
        shapley_values = {}

        for channel in self.channels:
            shapley_value = self._calculate_channel_shapley(channel)
            shapley_values[channel] = shapley_value
            print(f"  {channel}: {shapley_value:.2f}")

        self.shapley_values = shapley_values
        return shapley_values

    def _calculate_conversion_values(self, journeys: pd.DataFrame,
                                     use_revenue: bool) -> Dict[frozenset, float]:
        """
        Calculate conversion value (revenue or count) for each subset of channels.

        Args:
            journeys: DataFrame with customer journeys
            use_revenue: Whether to use revenue or conversion count

        Returns:
            Dictionary mapping channel subset to conversion value
        """
        conversion_values = defaultdict(float)
        subset_counts = defaultdict(int)

        # For each journey, determine which channel subsets it matches
        for _, journey in journeys.iterrows():
            journey_channels = set(journey['channels'])
            value = journey['revenue'] if use_revenue else (1 if journey['converted'] else 0)

            # Generate all subsets of channels
            for r in range(len(journey_channels) + 1):
                for subset in combinations(journey_channels, r):
                    subset_frozen = frozenset(subset)
                    conversion_values[subset_frozen] += value
                    subset_counts[subset_frozen] += 1

        # Calculate average conversion value per journey for each subset
        conversion_rates = {}
        for subset, total_value in conversion_values.items():
            count = subset_counts[subset]
            conversion_rates[subset] = total_value / count if count > 0 else 0

        return conversion_rates

    def _calculate_channel_shapley(self, channel: str) -> float:
        """
        Calculate Shapley value for a specific channel.

        The Shapley value is calculated as:
        φ_i = Σ [|S|! * (n - |S| - 1)! / n!] * [v(S ∪ {i}) - v(S)]

        Where:
        - S is a subset not containing channel i
        - v(S) is the value function for subset S
        - n is the total number of channels

        Args:
            channel: Channel name

        Returns:
            Shapley value for the channel
        """
        other_channels = [c for c in self.channels if c != channel]
        n = len(self.channels)
        shapley_value = 0

        # Iterate over all subsets of other channels
        for r in range(len(other_channels) + 1):
            for subset in combinations(other_channels, r):
                subset_set = set(subset)
                subset_with_channel = subset_set | {channel}

                # Calculate marginal contribution
                v_with = self.conversion_rate.get(frozenset(subset_with_channel), 0)
                v_without = self.conversion_rate.get(frozenset(subset_set), 0)
                marginal_contribution = v_with - v_without

                # Calculate Shapley weight
                s_size = len(subset_set)
                weight = (
                    np.math.factorial(s_size) *
                    np.math.factorial(n - s_size - 1) /
                    np.math.factorial(n)
                )

                shapley_value += weight * marginal_contribution

        return shapley_value

    def calculate_removal_effects(self) -> Dict[str, float]:
        """
        Calculate the removal effect for each channel.

        Removal effect = Value with all channels - Value without this channel

        Returns:
            Dictionary mapping channel to removal effect
        """
        print("\nCalculating removal effects...")

        all_channels = frozenset(self.channels)
        value_all = self.conversion_rate.get(all_channels, 0)

        removal_effects = {}
        for channel in self.channels:
            channels_without = frozenset(c for c in self.channels if c != channel)
            value_without = self.conversion_rate.get(channels_without, 0)
            removal_effect = value_all - value_without
            removal_effects[channel] = removal_effect
            print(f"  {channel}: {removal_effect:.2f}")

        self.removal_effects = removal_effects
        return removal_effects

    def get_attribution_summary(self) -> pd.DataFrame:
        """
        Generate a summary DataFrame with Shapley values and removal effects.

        Returns:
            DataFrame with attribution metrics for each channel
        """
        if not self.shapley_values:
            raise ValueError("Must calculate Shapley values first")

        summary_data = []
        for channel in self.channels:
            summary_data.append({
                'channel': channel,
                'shapley_value': self.shapley_values.get(channel, 0),
                'removal_effect': self.removal_effects.get(channel, 0),
                'shapley_pct': 0,  # Will calculate after
                'removal_pct': 0   # Will calculate after
            })

        summary_df = pd.DataFrame(summary_data)

        # Calculate percentages
        total_shapley = summary_df['shapley_value'].sum()
        total_removal = summary_df['removal_effect'].sum()

        if total_shapley > 0:
            summary_df['shapley_pct'] = (
                summary_df['shapley_value'] / total_shapley * 100
            )

        if total_removal > 0:
            summary_df['removal_pct'] = (
                summary_df['removal_effect'] / total_removal * 100
            )

        # Sort by Shapley value
        summary_df = summary_df.sort_values('shapley_value', ascending=False)

        return summary_df

    def analyze_channel_interactions(self) -> pd.DataFrame:
        """
        Analyze how channels work together (synergy effects).

        Returns:
            DataFrame showing conversion values for different channel combinations
        """
        interaction_data = []

        # Look at all 2-channel and 3-channel combinations
        for r in [1, 2, 3]:
            if r <= len(self.channels):
                for combo in combinations(self.channels, r):
                    combo_set = frozenset(combo)
                    value = self.conversion_rate.get(combo_set, 0)

                    interaction_data.append({
                        'channels': ', '.join(sorted(combo)),
                        'num_channels': len(combo),
                        'conversion_value': value
                    })

        df = pd.DataFrame(interaction_data)
        df = df.sort_values('conversion_value', ascending=False)

        return df


def main():
    """
    Example usage of the Shapley attribution model.
    """
    # Initialize model
    model = ShapleyAttribution()

    # Load data
    model.load_data(
        marketing_clicks_path='../raw_data/marketing_clicks.csv',
        orders_path='../raw_data/orders.csv',
        attribution_window_days=30
    )

    # Calculate Shapley values (using revenue)
    shapley_values = model.calculate_shapley_values(
        use_revenue=True,
        sample_size=10000  # Sample for faster computation
    )

    # Calculate removal effects
    removal_effects = model.calculate_removal_effects()

    # Get summary
    summary = model.get_attribution_summary()
    print("\n" + "="*80)
    print("ATTRIBUTION SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))

    # Analyze channel interactions
    print("\n" + "="*80)
    print("CHANNEL INTERACTIONS (Top 15)")
    print("="*80)
    interactions = model.analyze_channel_interactions()
    print(interactions.head(15).to_string(index=False))

    # Save results
    summary.to_csv('multi_touch_attribution/shapley_results.csv', index=False)
    interactions.to_csv('multi_touch_attribution/channel_interactions.csv', index=False)
    print("\n✓ Results saved to multi_touch_attribution/")


if __name__ == '__main__':
    main()
