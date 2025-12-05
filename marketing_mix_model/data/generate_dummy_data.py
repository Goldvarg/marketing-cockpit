"""
Generate realistic dummy data for Marketing Mix Modeling.

This script creates synthetic marketing data with known ground truth effects,
making it ideal for testing and comparing different MMM approaches.

The data includes:
- 5 marketing channels with different effectiveness levels
- Adstock effects (carryover from past spend)
- Saturation effects (diminishing returns at high spend)
- Seasonality and holiday effects
- Random noise to simulate real-world variability
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

# =============================================================================
# Configuration - The "ground truth" we're trying to recover with MMM
# =============================================================================

# Date range: 2 years of daily data
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2023, 12, 31)

# Channel configurations with TRUE effects
# These are the values we hope our models will recover
CHANNELS = {
    'Google': {
        'avg_daily_spend': 5000,
        'spend_volatility': 0.3,
        'true_roi': 2.5,           # $2.50 revenue per $1 spend
        'adstock_decay': 0.4,      # 40% carryover to next day
        'saturation_point': 15000  # Diminishing returns after this spend
    },
    'Meta': {
        'avg_daily_spend': 4000,
        'spend_volatility': 0.35,
        'true_roi': 2.0,
        'adstock_decay': 0.3,
        'saturation_point': 12000
    },
    'LinkedIn': {
        'avg_daily_spend': 2000,
        'spend_volatility': 0.4,
        'true_roi': 1.8,
        'adstock_decay': 0.2,
        'saturation_point': 6000
    },
    'TV': {
        'avg_daily_spend': 8000,
        'spend_volatility': 0.5,
        'true_roi': 1.5,
        'adstock_decay': 0.6,      # TV has longer carryover
        'saturation_point': 25000
    },
    'Email': {
        'avg_daily_spend': 500,
        'spend_volatility': 0.2,
        'true_roi': 4.0,           # Email is very efficient but limited scale
        'adstock_decay': 0.1,
        'saturation_point': 1500
    }
}

# Base revenue (what we'd get with zero marketing)
BASE_REVENUE = 50000

# Seasonality strength (0 = no seasonality, 1 = strong seasonality)
SEASONALITY_STRENGTH = 0.2

# Noise level (standard deviation as % of signal)
NOISE_LEVEL = 0.1


# =============================================================================
# Helper Functions
# =============================================================================

def apply_adstock(spend_series, decay_rate):
    """
    Apply adstock transformation (carryover effect).

    Adstock models the fact that marketing effects don't disappear immediately.
    A portion of yesterday's spend "carries over" to today's effect.
    """
    adstocked = np.zeros(len(spend_series))
    adstocked[0] = spend_series[0]

    for t in range(1, len(spend_series)):
        adstocked[t] = spend_series[t] + decay_rate * adstocked[t-1]

    return adstocked


def apply_saturation(spend_series, saturation_point):
    """
    Apply saturation transformation (diminishing returns).

    Uses a Hill function to model the fact that doubling spend
    doesn't double results - there are diminishing returns.
    """
    # Hill function: x / (x + K) where K is the half-saturation point
    half_sat = saturation_point / 2
    return spend_series / (spend_series + half_sat)


def generate_seasonality(dates):
    """
    Generate seasonal pattern with weekly and yearly components.
    """
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    day_of_week = np.array([d.weekday() for d in dates])

    # Yearly seasonality (higher in Q4, lower in Q1)
    yearly = np.sin(2 * np.pi * (day_of_year - 1) / 365) * 0.15
    yearly += np.sin(4 * np.pi * (day_of_year - 1) / 365) * 0.05

    # Weekly seasonality (lower on weekends)
    weekly = np.where(day_of_week >= 5, -0.1, 0.02)

    return yearly + weekly


def identify_holidays(dates):
    """
    Flag major US holidays and create promo periods.
    """
    holidays = []
    promos = []

    for d in dates:
        # Major holidays (simplified)
        is_holiday = (
            (d.month == 1 and d.day == 1) or      # New Year
            (d.month == 7 and d.day == 4) or      # July 4th
            (d.month == 11 and d.day >= 22 and d.day <= 28 and d.weekday() == 3) or  # Thanksgiving
            (d.month == 12 and d.day == 25)       # Christmas
        )

        # Black Friday / Cyber Monday period
        is_promo = (
            (d.month == 11 and d.day >= 24 and d.day <= 30) or  # Black Friday week
            (d.month == 12 and d.day >= 15 and d.day <= 24)     # Holiday shopping
        )

        holidays.append(is_holiday)
        promos.append(is_promo)

    return np.array(holidays), np.array(promos)


# =============================================================================
# Generate the Data
# =============================================================================

def generate_mmm_data():
    """
    Generate complete MMM dataset with known ground truth.
    """
    # Create date range
    dates = pd.date_range(START_DATE, END_DATE, freq='D')
    n_days = len(dates)

    print(f"Generating {n_days} days of marketing data...")

    # Initialize dataframe
    df = pd.DataFrame({'date': dates})

    # Generate seasonality and events
    seasonality = generate_seasonality(dates)
    holidays, promos = identify_holidays(dates)

    df['is_holiday'] = holidays.astype(int)
    df['is_promo'] = promos.astype(int)
    df['seasonality_index'] = 1 + seasonality * SEASONALITY_STRENGTH

    # Generate competitor activity (random walk)
    competitor_index = np.cumsum(np.random.normal(0, 0.01, n_days))
    competitor_index = 1 + 0.2 * (competitor_index - competitor_index.mean()) / competitor_index.std()
    df['competitor_spend_index'] = competitor_index

    # Generate channel spend and impressions
    total_marketing_contribution = np.zeros(n_days)

    for channel, config in CHANNELS.items():
        # Generate base spend with some autocorrelation
        base_spend = config['avg_daily_spend']
        volatility = config['spend_volatility']

        # Random walk with mean reversion for realistic spend patterns
        spend_noise = np.random.normal(0, volatility, n_days)
        spend_raw = base_spend * (1 + np.cumsum(spend_noise) * 0.1)
        spend_raw = np.maximum(spend_raw, base_spend * 0.2)  # Floor at 20% of average

        # Increase spend during promos
        spend_raw = spend_raw * (1 + 0.5 * promos)

        df[f'spend_{channel.lower()}'] = np.round(spend_raw, 2)

        # Generate impressions (correlated with spend but with variation)
        cpm = np.random.normal(10, 2, n_days)  # Cost per thousand impressions
        cpm = np.maximum(cpm, 3)
        impressions = (spend_raw / cpm) * 1000
        df[f'impressions_{channel.lower()}'] = np.round(impressions).astype(int)

        # Calculate marketing contribution with adstock and saturation
        adstocked = apply_adstock(spend_raw, config['adstock_decay'])
        saturated = apply_saturation(adstocked, config['saturation_point'])

        # Scale by true ROI
        contribution = saturated * config['saturation_point'] * config['true_roi'] * 0.5
        total_marketing_contribution += contribution

    # Calculate total revenue
    base = BASE_REVENUE * df['seasonality_index']
    holiday_boost = 1 + 0.3 * df['is_holiday'] + 0.4 * df['is_promo']
    competitor_effect = 1 - 0.1 * (df['competitor_spend_index'] - 1)

    revenue_signal = (base + total_marketing_contribution) * holiday_boost * competitor_effect

    # Add noise
    noise = np.random.normal(0, NOISE_LEVEL * revenue_signal.mean(), n_days)
    df['revenue'] = np.round(np.maximum(revenue_signal + noise, 0), 2)

    # Reorder columns nicely
    col_order = ['date']
    col_order += [f'spend_{ch.lower()}' for ch in CHANNELS.keys()]
    col_order += [f'impressions_{ch.lower()}' for ch in CHANNELS.keys()]
    col_order += ['revenue', 'is_holiday', 'is_promo', 'seasonality_index', 'competitor_spend_index']

    df = df[col_order]

    return df


def save_ground_truth():
    """
    Save the ground truth parameters for model validation.
    """
    truth = []
    for channel, config in CHANNELS.items():
        truth.append({
            'channel': channel,
            'true_roi': config['true_roi'],
            'adstock_decay': config['adstock_decay'],
            'saturation_point': config['saturation_point'],
            'avg_daily_spend': config['avg_daily_spend']
        })

    truth_df = pd.DataFrame(truth)
    truth_df.to_csv('ground_truth_parameters.csv', index=False)
    print("\nGround truth parameters saved to ground_truth_parameters.csv")
    return truth_df


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # Generate data
    df = generate_mmm_data()

    # Save to CSV
    output_path = 'mmm_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\nData saved to {output_path}")
    print(f"Shape: {df.shape}")

    # Save ground truth
    truth = save_ground_truth()

    # Print summary statistics
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total days: {len(df)}")

    print("\nSpend Summary (daily averages):")
    for channel in CHANNELS.keys():
        col = f'spend_{channel.lower()}'
        print(f"  {channel}: ${df[col].mean():,.0f}")

    print(f"\nRevenue Summary:")
    print(f"  Mean: ${df['revenue'].mean():,.0f}")
    print(f"  Std:  ${df['revenue'].std():,.0f}")
    print(f"  Min:  ${df['revenue'].min():,.0f}")
    print(f"  Max:  ${df['revenue'].max():,.0f}")

    print("\n" + "="*60)
    print("GROUND TRUTH (what models should recover)")
    print("="*60)
    print(truth.to_string(index=False))
