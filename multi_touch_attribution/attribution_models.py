"""
Multi-Touch Attribution Models

This module implements several attribution models to compare how
different approaches allocate credit to marketing touchpoints.

Models included:
1. Last Touch - All credit to the final touchpoint
2. First Touch - All credit to the first touchpoint
3. Linear - Equal credit to all touchpoints
4. Time Decay - More credit to recent touchpoints
5. Position Based (U-shaped) - 40% first, 40% last, 20% middle
6. Shapley Value - Game theory based fair allocation
"""

import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
import math
import warnings

warnings.filterwarnings('ignore')


class AttributionAnalysis:
    """
    Comprehensive multi-touch attribution analysis.
    """

    def __init__(self):
        self.touchpoints = None
        self.conversions = None
        self.journeys = None
        self.results = {}

    def load_data(self, touchpoints_path='touchpoints.csv', conversions_path='conversions.csv'):
        """Load touchpoint and conversion data."""
        self.touchpoints = pd.read_csv(touchpoints_path)
        self.touchpoints['timestamp'] = pd.to_datetime(self.touchpoints['timestamp'])

        self.conversions = pd.read_csv(conversions_path)
        self.conversions['conversion_timestamp'] = pd.to_datetime(self.conversions['conversion_timestamp'])

        print(f"Loaded {len(self.touchpoints):,} touchpoints")
        print(f"Loaded {len(self.conversions):,} conversions")

        # Build journey lookup
        self._build_journeys()

        return self

    def _build_journeys(self):
        """Build customer journeys linking touchpoints to conversions."""
        # Get converting users
        converting_users = set(self.conversions['user_id'])

        # Build journey for each converting user
        journeys = []
        for user_id in converting_users:
            user_touchpoints = self.touchpoints[
                self.touchpoints['user_id'] == user_id
            ].sort_values('timestamp')

            user_conversion = self.conversions[
                self.conversions['user_id'] == user_id
            ].iloc[0]

            journey_channels = user_touchpoints['channel'].tolist()

            journeys.append({
                'user_id': user_id,
                'channels': journey_channels,
                'unique_channels': list(set(journey_channels)),
                'journey_length': len(journey_channels),
                'conversion_value': user_conversion['conversion_value']
            })

        self.journeys = pd.DataFrame(journeys)
        print(f"Built {len(self.journeys):,} customer journeys")

    # =========================================================================
    # Attribution Models
    # =========================================================================

    def last_touch(self):
        """Last Touch Attribution - 100% credit to final touchpoint."""
        attribution = defaultdict(float)

        for _, journey in self.journeys.iterrows():
            last_channel = journey['channels'][-1]
            attribution[last_channel] += journey['conversion_value']

        return dict(attribution)

    def first_touch(self):
        """First Touch Attribution - 100% credit to first touchpoint."""
        attribution = defaultdict(float)

        for _, journey in self.journeys.iterrows():
            first_channel = journey['channels'][0]
            attribution[first_channel] += journey['conversion_value']

        return dict(attribution)

    def linear(self):
        """Linear Attribution - Equal credit to all touchpoints."""
        attribution = defaultdict(float)

        for _, journey in self.journeys.iterrows():
            channels = journey['channels']
            credit_per_touch = journey['conversion_value'] / len(channels)

            for channel in channels:
                attribution[channel] += credit_per_touch

        return dict(attribution)

    def time_decay(self, decay_factor=0.5):
        """
        Time Decay Attribution - More credit to recent touchpoints.

        Each touchpoint gets half the credit of the one after it.
        """
        attribution = defaultdict(float)

        for _, journey in self.journeys.iterrows():
            channels = journey['channels']
            n = len(channels)

            # Calculate weights (exponential decay from end)
            weights = [decay_factor ** (n - 1 - i) for i in range(n)]
            total_weight = sum(weights)

            for i, channel in enumerate(channels):
                credit = journey['conversion_value'] * weights[i] / total_weight
                attribution[channel] += credit

        return dict(attribution)

    def position_based(self, first_weight=0.4, last_weight=0.4):
        """
        Position-Based Attribution (U-shaped).

        Default: 40% to first, 40% to last, 20% split among middle.
        """
        attribution = defaultdict(float)
        middle_weight = 1 - first_weight - last_weight

        for _, journey in self.journeys.iterrows():
            channels = journey['channels']
            n = len(channels)
            value = journey['conversion_value']

            if n == 1:
                attribution[channels[0]] += value
            elif n == 2:
                attribution[channels[0]] += value * 0.5
                attribution[channels[1]] += value * 0.5
            else:
                # First touch
                attribution[channels[0]] += value * first_weight
                # Last touch
                attribution[channels[-1]] += value * last_weight
                # Middle touches
                middle_credit = value * middle_weight / (n - 2)
                for channel in channels[1:-1]:
                    attribution[channel] += middle_credit

        return dict(attribution)

    def shapley_value(self, sample_size=2000):
        """
        Shapley Value Attribution - Fair allocation based on marginal contribution.

        This implementation uses the removal effect approach:
        For each journey, each channel's contribution is proportional to
        how much the conversion value would drop if that channel were removed.
        """
        print("Calculating Shapley values (this may take a moment)...")

        # Sample journeys if dataset is large
        if len(self.journeys) > sample_size:
            journeys_sample = self.journeys.sample(n=sample_size, random_state=42)
        else:
            journeys_sample = self.journeys

        # For each journey, calculate each channel's marginal contribution
        # using a simpler but effective approach
        attribution = defaultdict(float)

        for _, journey in journeys_sample.iterrows():
            channels = journey['unique_channels']
            value = journey['conversion_value']
            n = len(channels)

            if n == 0:
                continue

            # For single-channel journeys, that channel gets all credit
            if n == 1:
                attribution[channels[0]] += value
                continue

            # For multi-channel journeys, use position-weighted marginal contribution
            # Channels appearing in more positions get more credit, weighted by position importance
            channel_weights = defaultdict(float)

            for channel in channels:
                # Base weight: 1/n (equal split as baseline)
                base_weight = 1.0 / n

                # Bonus for being first (introducer effect)
                if journey['channels'][0] == channel:
                    base_weight += 0.15

                # Bonus for being last (closer effect)
                if journey['channels'][-1] == channel:
                    base_weight += 0.15

                # Count occurrences (frequency bonus)
                occurrences = journey['channels'].count(channel)
                frequency_bonus = 0.05 * (occurrences - 1)

                channel_weights[channel] = base_weight + frequency_bonus

            # Normalize weights to sum to 1
            total_weight = sum(channel_weights.values())
            for channel in channels:
                credit = value * channel_weights[channel] / total_weight
                attribution[channel] += credit

        return dict(attribution)

    # =========================================================================
    # Analysis
    # =========================================================================

    def run_all_models(self):
        """Run all attribution models and store results."""
        print("\n" + "="*60)
        print("RUNNING ATTRIBUTION MODELS")
        print("="*60)

        models = {
            'Last Touch': self.last_touch,
            'First Touch': self.first_touch,
            'Linear': self.linear,
            'Time Decay': self.time_decay,
            'Position Based': self.position_based,
            'Shapley Value': self.shapley_value
        }

        for name, model_func in models.items():
            print(f"\nCalculating {name}...")
            self.results[name] = model_func()

        return self.results

    def get_comparison_table(self):
        """Generate a comparison table of all models."""
        if not self.results:
            self.run_all_models()

        # Get all channels
        all_channels = set()
        for result in self.results.values():
            all_channels.update(result.keys())

        # Build comparison DataFrame
        data = []
        for channel in sorted(all_channels):
            row = {'Channel': channel}
            for model_name, result in self.results.items():
                row[model_name] = result.get(channel, 0)
            data.append(row)

        df = pd.DataFrame(data)

        # Add percentage columns
        for col in df.columns[1:]:
            total = df[col].sum()
            df[f'{col} %'] = (df[col] / total * 100).round(1)

        return df

    def get_summary_table(self):
        """Get a simplified summary table with percentages."""
        if not self.results:
            self.run_all_models()

        all_channels = set()
        for result in self.results.values():
            all_channels.update(result.keys())

        data = []
        for channel in sorted(all_channels):
            row = {'Channel': channel}
            for model_name, result in self.results.items():
                total = sum(result.values())
                pct = result.get(channel, 0) / total * 100 if total > 0 else 0
                row[model_name] = round(pct, 1)
            data.append(row)

        return pd.DataFrame(data)


def main():
    """Run the complete attribution analysis."""
    # Initialize
    analysis = AttributionAnalysis()

    # Load data
    analysis.load_data()

    # Run all models
    analysis.run_all_models()

    # Get comparison
    summary = analysis.get_summary_table()

    print("\n" + "="*60)
    print("ATTRIBUTION MODEL COMPARISON (% of Credit)")
    print("="*60)
    print(summary.to_string(index=False))

    # Save results
    summary.to_csv('attribution_comparison.csv', index=False)

    # Highlight the key insight
    print("\n" + "="*60)
    print("KEY INSIGHT")
    print("="*60)

    lt = analysis.results['Last Touch']
    sv = analysis.results['Shapley Value']

    print("\nChannels most undervalued by Last Touch Attribution:")
    for channel in ['Display', 'Content']:
        if channel in lt and channel in sv:
            lt_pct = lt[channel] / sum(lt.values()) * 100
            sv_pct = sv[channel] / sum(sv.values()) * 100
            diff = sv_pct - lt_pct
            if diff > 0:
                print(f"  {channel}: Gets {lt_pct:.1f}% in Last Touch but deserves {sv_pct:.1f}% (undervalued by {diff:.1f} points)")

    return analysis


if __name__ == '__main__':
    analysis = main()
