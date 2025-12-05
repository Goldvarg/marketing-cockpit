"""
Generate Customer Journey Data for Multi-Touch Attribution

This script creates synthetic customer journey data designed to illustrate
the difference between last-touch and multi-touch attribution models.

Key design: Some channels (Display, Content Marketing) are "introducers" -
they start customer journeys but rarely close them. Last-touch attribution
would undervalue these channels, but multi-touch models reveal their true value.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# =============================================================================
# Channel Configuration
# =============================================================================

# Channel roles and characteristics
CHANNELS = {
    'Display': {
        'role': 'introducer',       # First touch, rarely last
        'first_touch_prob': 0.35,   # High probability of being first
        'last_touch_prob': 0.05,    # Low probability of being last
        'conversion_influence': 0.20 # Contributes 20% to conversion likelihood
    },
    'Content': {
        'role': 'introducer',
        'first_touch_prob': 0.25,
        'last_touch_prob': 0.08,
        'conversion_influence': 0.15
    },
    'Social': {
        'role': 'nurture',          # Middle of funnel
        'first_touch_prob': 0.15,
        'last_touch_prob': 0.12,
        'conversion_influence': 0.15
    },
    'Email': {
        'role': 'nurture',
        'first_touch_prob': 0.10,
        'last_touch_prob': 0.20,
        'conversion_influence': 0.20
    },
    'Paid Search': {
        'role': 'closer',           # Bottom of funnel
        'first_touch_prob': 0.10,
        'last_touch_prob': 0.35,
        'conversion_influence': 0.15
    },
    'Direct': {
        'role': 'closer',
        'first_touch_prob': 0.05,
        'last_touch_prob': 0.20,
        'conversion_influence': 0.15
    }
}


def generate_customer_journey(user_id, start_date):
    """
    Generate a realistic customer journey.

    Journeys follow patterns:
    - Introducers (Display, Content) tend to appear early
    - Nurturers (Social, Email) appear in the middle
    - Closers (Paid Search, Direct) appear at the end
    """
    # Determine journey length (1-8 touchpoints)
    journey_length = np.random.choice(
        [1, 2, 3, 4, 5, 6, 7, 8],
        p=[0.15, 0.20, 0.25, 0.20, 0.10, 0.05, 0.03, 0.02]
    )

    # Generate touchpoints
    touchpoints = []
    current_date = start_date

    for i in range(journey_length):
        position = i / max(journey_length - 1, 1)  # 0 to 1

        # Channel selection based on position in journey
        if position < 0.3:  # Early stage
            weights = [CHANNELS[ch]['first_touch_prob'] for ch in CHANNELS]
        elif position > 0.7:  # Late stage
            weights = [CHANNELS[ch]['last_touch_prob'] for ch in CHANNELS]
        else:  # Middle stage
            weights = [0.15, 0.15, 0.25, 0.25, 0.10, 0.10]

        # Normalize weights
        weights = np.array(weights) / sum(weights)

        channel = np.random.choice(list(CHANNELS.keys()), p=weights)

        # Add some time between touchpoints (1 hour to 7 days)
        if i > 0:
            hours_gap = np.random.exponential(scale=48)  # Average 2 days
            hours_gap = max(1, min(hours_gap, 168))  # Cap at 1 week
            current_date += timedelta(hours=hours_gap)

        touchpoints.append({
            'user_id': user_id,
            'channel': channel,
            'timestamp': current_date,
            'touchpoint_order': i + 1
        })

    return touchpoints


def determine_conversion(touchpoints):
    """
    Determine if a journey results in conversion based on channel mix.

    The conversion probability depends on having a good mix of channels.
    Journeys with only closers convert less than those with introducers + closers.
    """
    channels_in_journey = set(tp['channel'] for tp in touchpoints)

    # Base conversion probability
    base_prob = 0.10

    # Add influence from each channel present
    for channel in channels_in_journey:
        base_prob += CHANNELS[channel]['conversion_influence']

    # Bonus for having both introducers and closers (synergy effect)
    has_introducer = any(
        CHANNELS[ch]['role'] == 'introducer' for ch in channels_in_journey
    )
    has_closer = any(
        CHANNELS[ch]['role'] == 'closer' for ch in channels_in_journey
    )

    if has_introducer and has_closer:
        base_prob += 0.15  # Synergy bonus

    # Cap probability
    conversion_prob = min(base_prob, 0.95)

    return np.random.random() < conversion_prob


def generate_dataset(n_users=5000):
    """
    Generate complete customer journey dataset.
    """
    print(f"Generating journeys for {n_users} users...")

    all_touchpoints = []
    conversions = []

    start_date = datetime(2023, 1, 1)

    for user_id in range(1, n_users + 1):
        # Random start date within the year
        user_start = start_date + timedelta(days=np.random.randint(0, 365))

        journey = generate_customer_journey(user_id, user_start)
        all_touchpoints.extend(journey)

        # Determine conversion
        converted = determine_conversion(journey)

        if converted:
            # Conversion happens shortly after last touchpoint
            conversion_time = journey[-1]['timestamp'] + timedelta(hours=np.random.randint(1, 48))
            conversion_value = np.random.lognormal(mean=4.5, sigma=0.5)  # Avg ~$90

            conversions.append({
                'user_id': user_id,
                'conversion_timestamp': conversion_time,
                'conversion_value': round(conversion_value, 2),
                'journey_length': len(journey),
                'first_channel': journey[0]['channel'],
                'last_channel': journey[-1]['channel']
            })

    # Create DataFrames
    touchpoints_df = pd.DataFrame(all_touchpoints)
    conversions_df = pd.DataFrame(conversions)

    print(f"Generated {len(touchpoints_df)} touchpoints")
    print(f"Generated {len(conversions_df)} conversions ({len(conversions_df)/n_users*100:.1f}% conversion rate)")

    return touchpoints_df, conversions_df


def save_data(touchpoints_df, conversions_df, output_dir='.'):
    """Save datasets to CSV."""
    touchpoints_df.to_csv(f'{output_dir}/touchpoints.csv', index=False)
    conversions_df.to_csv(f'{output_dir}/conversions.csv', index=False)
    print(f"\nSaved to {output_dir}/")


def print_summary(touchpoints_df, conversions_df):
    """Print dataset summary."""
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)

    print("\nTouchpoint Distribution:")
    touch_counts = touchpoints_df['channel'].value_counts()
    for channel, count in touch_counts.items():
        pct = count / len(touchpoints_df) * 100
        print(f"  {channel}: {count:,} ({pct:.1f}%)")

    print("\nFirst Touch Distribution:")
    first_touches = touchpoints_df[touchpoints_df['touchpoint_order'] == 1]['channel'].value_counts()
    for channel, count in first_touches.items():
        pct = count / first_touches.sum() * 100
        print(f"  {channel}: {count:,} ({pct:.1f}%)")

    print("\nLast Touch Distribution (Converting Journeys):")
    last_channel_counts = conversions_df['last_channel'].value_counts()
    for channel, count in last_channel_counts.items():
        pct = count / len(conversions_df) * 100
        print(f"  {channel}: {count:,} ({pct:.1f}%)")

    print("\n" + "="*60)
    print("KEY INSIGHT: Notice how Display and Content dominate first touches")
    print("but Paid Search and Direct dominate last touches. Last-touch")
    print("attribution would give all credit to the closers!")
    print("="*60)


if __name__ == '__main__':
    touchpoints_df, conversions_df = generate_dataset(n_users=5000)
    save_data(touchpoints_df, conversions_df)
    print_summary(touchpoints_df, conversions_df)
