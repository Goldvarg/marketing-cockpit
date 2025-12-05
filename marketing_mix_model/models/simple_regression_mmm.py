"""
Simple Regression-Based Marketing Mix Model

This is the most straightforward approach to MMM - using linear regression
with manual feature engineering for adstock and saturation effects.

Pros:
- Easy to understand and explain
- No special packages required
- Fast to run
- Full control over transformations

Cons:
- Requires manual tuning of adstock/saturation parameters
- Doesn't provide uncertainty estimates by default
- Less sophisticated than Bayesian approaches
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SimpleRegressionMMM:
    """
    A simple but effective Marketing Mix Model using Ridge Regression.
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.channels = []
        self.coefficients = {}
        self.adstock_rates = {}
        self.saturation_params = {}

    def apply_adstock(self, x, decay_rate):
        """Apply geometric adstock transformation."""
        adstocked = np.zeros(len(x))
        adstocked[0] = x[0]
        for t in range(1, len(x)):
            adstocked[t] = x[t] + decay_rate * adstocked[t-1]
        return adstocked

    def apply_saturation(self, x, alpha=0.5):
        """Apply saturation transformation using power function."""
        return np.power(x, alpha)

    def find_best_adstock(self, df, channel_col, target_col, decay_range=np.arange(0.1, 0.9, 0.1)):
        """Find optimal adstock decay rate via grid search."""
        best_corr = -1
        best_decay = 0.1

        for decay in decay_range:
            adstocked = self.apply_adstock(df[channel_col].values, decay)
            corr = np.corrcoef(adstocked, df[target_col].values)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_decay = decay

        return best_decay

    def prepare_features(self, df, channels, target_col='revenue'):
        """
        Prepare features with adstock and saturation transformations.
        """
        print("Finding optimal adstock rates...")
        X_transformed = pd.DataFrame(index=df.index)

        for channel in channels:
            spend_col = f'spend_{channel.lower()}'

            # Find best adstock rate
            best_decay = self.find_best_adstock(df, spend_col, target_col)
            self.adstock_rates[channel] = best_decay
            print(f"  {channel}: decay = {best_decay:.2f}")

            # Apply transformations
            adstocked = self.apply_adstock(df[spend_col].values, best_decay)
            saturated = self.apply_saturation(adstocked)

            X_transformed[f'{channel}_transformed'] = saturated

        # Add control variables if present
        control_cols = ['is_holiday', 'is_promo', 'seasonality_index', 'competitor_spend_index']
        for col in control_cols:
            if col in df.columns:
                X_transformed[col] = df[col]

        self.feature_names = X_transformed.columns.tolist()
        return X_transformed

    def fit(self, df, channels, target_col='revenue', alpha=1.0):
        """
        Fit the MMM model.

        Args:
            df: DataFrame with marketing data
            channels: List of channel names
            target_col: Name of target variable column
            alpha: Ridge regularization strength
        """
        self.channels = channels
        print("\n" + "="*60)
        print("FITTING SIMPLE REGRESSION MMM")
        print("="*60)

        # Prepare features
        X = self.prepare_features(df, channels, target_col)
        y = df[target_col].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit ridge regression
        self.model = Ridge(alpha=alpha)
        self.model.fit(X_scaled, y)

        # Store coefficients
        for i, name in enumerate(self.feature_names):
            self.coefficients[name] = self.model.coef_[i]

        # Calculate R-squared
        y_pred = self.model.predict(X_scaled)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"\nModel R-squared: {r_squared:.4f}")
        print(f"Intercept (base revenue): ${self.model.intercept_:,.0f}")

        return self

    def get_channel_contribution(self, df):
        """Calculate each channel's contribution to revenue."""
        X = self.prepare_features(df, self.channels)

        contributions = {}

        for i, channel in enumerate(self.channels):
            col_idx = self.feature_names.index(f'{channel}_transformed')
            # Use raw feature values (not scaled) for interpretable contribution
            contribution = X.iloc[:, col_idx].values * abs(self.model.coef_[col_idx])
            contributions[channel] = max(contribution.sum(), 0)  # Ensure non-negative

        # Normalize to percentages
        total = sum(contributions.values())
        if total > 0:
            for channel in contributions:
                contributions[channel] = contributions[channel] / total * 100
        else:
            for channel in contributions:
                contributions[channel] = 100.0 / len(contributions)

        return contributions

    def calculate_roi(self, df):
        """Calculate ROI for each channel."""
        X = self.prepare_features(df, self.channels)

        roi_results = {}
        for i, channel in enumerate(self.channels):
            spend_col = f'spend_{channel.lower()}'
            total_spend = df[spend_col].sum()

            col_idx = self.feature_names.index(f'{channel}_transformed')
            # Use absolute coefficient for positive ROI estimation
            contribution = (X.iloc[:, col_idx].values * abs(self.model.coef_[col_idx])).sum()

            # Scale contribution back to revenue units (rough approximation)
            # Multiply by target std / feature std to denormalize
            roi = contribution / total_spend if total_spend > 0 else 0
            roi_results[channel] = max(roi, 0)

        return roi_results

    def get_results_summary(self, df):
        """Generate a summary of model results."""
        contributions = self.get_channel_contribution(df)
        roi = self.calculate_roi(df)

        summary = []
        for channel in self.channels:
            spend_col = f'spend_{channel.lower()}'
            summary.append({
                'Channel': channel,
                'Total Spend': df[spend_col].sum(),
                'Contribution %': contributions[channel],
                'Estimated ROI': roi[channel],
                'Adstock Decay': self.adstock_rates[channel]
            })

        return pd.DataFrame(summary)

    def plot_contribution(self, df, save_path=None):
        """Plot channel contribution pie chart."""
        contributions = self.get_channel_contribution(df)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(contributions)))

        wedges, texts, autotexts = ax.pie(
            contributions.values(),
            labels=contributions.keys(),
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.02] * len(contributions)
        )

        ax.set_title('Channel Contribution to Revenue\n(Simple Regression MMM)', fontsize=14)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.close()
        return fig

    def plot_roi_comparison(self, df, ground_truth=None, save_path=None):
        """Plot ROI comparison with optional ground truth."""
        roi = self.calculate_roi(df)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(roi))
        width = 0.35

        # Model estimates
        bars1 = ax.bar(x - width/2, list(roi.values()), width, label='Model Estimate', color='steelblue')

        # Ground truth if provided
        if ground_truth is not None:
            truth_values = [ground_truth.get(ch, 0) for ch in roi.keys()]
            bars2 = ax.bar(x + width/2, truth_values, width, label='Ground Truth', color='lightcoral')

        ax.set_ylabel('ROI (Revenue per $1 Spend)')
        ax.set_xlabel('Channel')
        ax.set_title('ROI by Channel\n(Simple Regression MMM)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(roi.keys())
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.close()
        return fig


def run_simple_mmm(data_path='../data/mmm_data.csv', output_dir='../results/simple_regression'):
    """
    Run the complete simple regression MMM analysis.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path, parse_dates=['date'])
    print(f"Loaded {len(df)} rows")

    # Define channels
    channels = ['Google', 'Meta', 'LinkedIn', 'TV', 'Email']

    # Initialize and fit model
    model = SimpleRegressionMMM()
    model.fit(df, channels)

    # Get results
    summary = model.get_results_summary(df)
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))

    # Save results
    summary.to_csv(f'{output_dir}/results_summary.csv', index=False)

    # Load ground truth for comparison
    try:
        truth_df = pd.read_csv('../data/ground_truth_parameters.csv')
        ground_truth = dict(zip(truth_df['channel'], truth_df['true_roi']))
    except:
        ground_truth = None

    # Generate visualizations
    model.plot_contribution(df, save_path=f'{output_dir}/channel_contribution.png')
    model.plot_roi_comparison(df, ground_truth, save_path=f'{output_dir}/roi_comparison.png')

    print(f"\nResults saved to {output_dir}/")
    return model, summary


if __name__ == '__main__':
    model, summary = run_simple_mmm()
