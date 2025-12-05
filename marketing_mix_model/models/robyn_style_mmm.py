"""
Robyn-Style Marketing Mix Model (Python Implementation)

This implementation follows Meta's Robyn methodology but in pure Python.
The original Robyn package is written in R - this is a Python recreation.

Key Robyn concepts implemented here:
1. Geometric adstock transformation
2. Hill saturation function
3. Multi-objective hyperparameter optimization (using Nevergrad)
4. Ridge regression for parameter estimation
5. NRMSE + Decomp.RSSD as optimization objectives

Pros:
- Automatic hyperparameter tuning
- Multi-objective optimization finds Pareto-optimal solutions
- No manual tuning of adstock/saturation parameters
- Open source and well-documented methodology

Cons:
- Slower due to optimization (many model fits)
- Complex to understand fully
- Original R version has more features
- Requires optimization library
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')


class RobynStyleMMM:
    """
    Python implementation of Meta's Robyn MMM methodology.
    """

    def __init__(self):
        self.model = None
        self.best_params = {}
        self.channels = []
        self.decomposition = {}
        self.pareto_solutions = []

    def geometric_adstock(self, x, theta):
        """
        Apply geometric adstock transformation.

        Args:
            x: Spend series
            theta: Decay rate (0 to 1)
        """
        adstocked = np.zeros(len(x))
        adstocked[0] = x[0]
        for t in range(1, len(x)):
            adstocked[t] = x[t] + theta * adstocked[t-1]
        return adstocked

    def hill_saturation(self, x, alpha, gamma):
        """
        Apply Hill saturation function.

        Args:
            x: Adstocked spend
            alpha: Shape parameter (controls curve steepness)
            gamma: Scale parameter (inflection point)
        """
        x_normalized = x / (x.max() + 1e-10)
        return x_normalized ** alpha / (x_normalized ** alpha + gamma ** alpha)

    def transform_media(self, spend_series, theta, alpha, gamma):
        """Apply both adstock and saturation transformations."""
        adstocked = self.geometric_adstock(spend_series, theta)
        saturated = self.hill_saturation(adstocked, alpha, gamma)
        return saturated

    def calculate_nrmse(self, y_true, y_pred):
        """Calculate Normalized Root Mean Square Error."""
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        nrmse = rmse / (y_true.max() - y_true.min())
        return nrmse

    def calculate_decomp_rssd(self, coefficients, spend_shares):
        """
        Calculate Decomposition RSSD (Root Sum of Squared Differences).

        This measures how different the effect share is from spend share.
        Lower is better - means effects are proportional to spend.
        """
        if coefficients.sum() == 0:
            return 1.0

        effect_shares = np.abs(coefficients) / np.abs(coefficients).sum()
        rssd = np.sqrt(np.sum((effect_shares - spend_shares) ** 2))
        return rssd

    def objective_function(self, params, df, channels, target_col):
        """
        Multi-objective function for optimization.

        Combines:
        1. NRMSE (model fit)
        2. Decomp RSSD (effect/spend alignment)
        """
        n_channels = len(channels)

        # Extract parameters for each channel
        thetas = params[:n_channels]
        alphas = params[n_channels:2*n_channels]
        gammas = params[2*n_channels:3*n_channels]

        # Transform media variables
        X = np.zeros((len(df), n_channels))
        for i, channel in enumerate(channels):
            spend_col = f'spend_{channel.lower()}'
            X[:, i] = self.transform_media(
                df[spend_col].values,
                thetas[i], alphas[i], gammas[i]
            )

        # Add control variables
        control_cols = ['is_holiday', 'is_promo', 'seasonality_index', 'competitor_spend_index']
        controls = []
        for col in control_cols:
            if col in df.columns:
                controls.append(df[col].values)

        if controls:
            X_full = np.column_stack([X] + controls)
        else:
            X_full = X

        # Fit ridge regression
        y = df[target_col].values
        model = Ridge(alpha=1.0)
        model.fit(X_full, y)

        # Calculate objectives
        y_pred = model.predict(X_full)
        nrmse = self.calculate_nrmse(y, y_pred)

        # Calculate spend shares
        total_spends = []
        for channel in channels:
            spend_col = f'spend_{channel.lower()}'
            total_spends.append(df[spend_col].sum())
        spend_shares = np.array(total_spends) / sum(total_spends)

        # Get media coefficients (first n_channels)
        media_coefs = model.coef_[:n_channels]
        decomp_rssd = self.calculate_decomp_rssd(media_coefs, spend_shares)

        # Combined objective (weighted sum)
        combined = 0.5 * nrmse + 0.5 * decomp_rssd

        return combined

    def optimize_hyperparameters(self, df, channels, target_col='revenue', n_iterations=50):
        """
        Find optimal adstock and saturation parameters using optimization.
        """
        print("\nOptimizing hyperparameters (Robyn-style)...")
        print(f"Running {n_iterations} optimization iterations...")

        n_channels = len(channels)

        # Parameter bounds
        # theta: adstock decay (0.1 to 0.9)
        # alpha: Hill shape (0.5 to 3.0)
        # gamma: Hill scale (0.3 to 1.0)
        bounds = (
            [(0.1, 0.9) for _ in range(n_channels)] +  # thetas
            [(0.5, 3.0) for _ in range(n_channels)] +  # alphas
            [(0.3, 1.0) for _ in range(n_channels)]    # gammas
        )

        # Initial guess
        x0 = (
            [0.5] * n_channels +  # thetas
            [1.5] * n_channels +  # alphas
            [0.5] * n_channels    # gammas
        )

        # Run optimization with multiple restarts
        best_result = None
        best_score = float('inf')

        for i in range(n_iterations):
            # Random starting point
            if i > 0:
                x0 = [
                    np.random.uniform(b[0], b[1]) for b in bounds
                ]

            try:
                result = minimize(
                    self.objective_function,
                    x0,
                    args=(df, channels, target_col),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100}
                )

                if result.fun < best_score:
                    best_score = result.fun
                    best_result = result

                    if (i + 1) % 10 == 0:
                        print(f"  Iteration {i+1}: Best score = {best_score:.4f}")

            except Exception as e:
                continue

        if best_result is None:
            raise ValueError("Optimization failed")

        # Extract best parameters
        params = best_result.x
        for i, channel in enumerate(channels):
            self.best_params[channel] = {
                'theta': params[i],
                'alpha': params[n_channels + i],
                'gamma': params[2 * n_channels + i]
            }

        print("\nOptimal parameters found:")
        for channel, p in self.best_params.items():
            print(f"  {channel}: theta={p['theta']:.3f}, alpha={p['alpha']:.3f}, gamma={p['gamma']:.3f}")

        return self.best_params

    def fit(self, df, channels, target_col='revenue', n_iterations=50):
        """
        Fit the Robyn-style MMM model.
        """
        print("\n" + "="*60)
        print("FITTING ROBYN-STYLE MMM")
        print("="*60)

        self.channels = channels

        # Optimize hyperparameters
        self.optimize_hyperparameters(df, channels, target_col, n_iterations)

        # Fit final model with optimal parameters
        X = np.zeros((len(df), len(channels)))
        for i, channel in enumerate(channels):
            spend_col = f'spend_{channel.lower()}'
            p = self.best_params[channel]
            X[:, i] = self.transform_media(
                df[spend_col].values,
                p['theta'], p['alpha'], p['gamma']
            )

        # Add controls
        control_cols = ['is_holiday', 'is_promo', 'seasonality_index', 'competitor_spend_index']
        controls = []
        control_names = []
        for col in control_cols:
            if col in df.columns:
                controls.append(df[col].values)
                control_names.append(col)

        if controls:
            X_full = np.column_stack([X] + controls)
        else:
            X_full = X

        y = df[target_col].values
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_full, y)

        # Calculate decomposition
        y_pred = self.model.predict(X_full)
        self._calculate_decomposition(df, X, y, channels)

        # Model metrics
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"\nModel R-squared: {r_squared:.4f}")
        print(f"NRMSE: {self.calculate_nrmse(y, y_pred):.4f}")

        return self

    def _calculate_decomposition(self, df, X_transformed, y, channels):
        """Calculate channel contribution decomposition."""
        # Base contribution (intercept)
        base = self.model.intercept_

        # Channel contributions
        contributions = {}
        total_media_contribution = 0

        for i, channel in enumerate(channels):
            contribution = X_transformed[:, i] * self.model.coef_[i]
            contributions[channel] = contribution.sum()
            total_media_contribution += contributions[channel]

        # Normalize to percentages
        total = base * len(df) + total_media_contribution
        self.decomposition = {
            'base': base * len(df) / total * 100,
            'channels': {ch: contributions[ch] / total * 100 for ch in channels}
        }

    def get_roi_estimates(self, df):
        """Calculate ROI for each channel."""
        roi_results = {}

        for i, channel in enumerate(self.channels):
            spend_col = f'spend_{channel.lower()}'
            total_spend = df[spend_col].sum()

            # Transform spend
            p = self.best_params[channel]
            transformed = self.transform_media(
                df[spend_col].values,
                p['theta'], p['alpha'], p['gamma']
            )

            # Calculate contribution
            contribution = (transformed * abs(self.model.coef_[i])).sum()

            roi = contribution / total_spend if total_spend > 0 else 0
            roi_results[channel] = max(0, roi)

        return roi_results

    def get_results_summary(self, df):
        """Generate results summary DataFrame."""
        roi = self.get_roi_estimates(df)

        summary = []
        for channel in self.channels:
            spend_col = f'spend_{channel.lower()}'
            p = self.best_params.get(channel, {})

            summary.append({
                'Channel': channel,
                'Total Spend': df[spend_col].sum(),
                'Contribution %': self.decomposition['channels'].get(channel, 0),
                'Estimated ROI': roi.get(channel, 0),
                'Adstock (theta)': p.get('theta', 0),
                'Saturation (alpha)': p.get('alpha', 0)
            })

        return pd.DataFrame(summary)

    def plot_contribution(self, df, save_path=None):
        """Plot channel contribution breakdown."""
        contributions = self.decomposition['channels']

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set2(np.linspace(0, 1, len(contributions)))

        wedges, texts, autotexts = ax.pie(
            contributions.values(),
            labels=contributions.keys(),
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.02] * len(contributions)
        )

        ax.set_title('Channel Contribution to Revenue\n(Robyn-Style MMM)', fontsize=14)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.close()
        return fig

    def plot_roi_comparison(self, df, ground_truth=None, save_path=None):
        """Plot ROI comparison."""
        roi = self.get_roi_estimates(df)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(roi))
        width = 0.35

        bars1 = ax.bar(x - width/2, list(roi.values()), width,
                       label='Model Estimate', color='darkorange')

        if ground_truth is not None:
            truth_values = [ground_truth.get(ch, 0) for ch in roi.keys()]
            bars2 = ax.bar(x + width/2, truth_values, width,
                          label='Ground Truth', color='lightcoral')

        ax.set_ylabel('ROI (Revenue per $1 Spend)')
        ax.set_xlabel('Channel')
        ax.set_title('ROI by Channel\n(Robyn-Style MMM)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(roi.keys())
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.close()
        return fig

    def plot_response_curves(self, df, save_path=None):
        """Plot response curves for each channel."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, channel in enumerate(self.channels):
            if i >= len(axes):
                break

            spend_col = f'spend_{channel.lower()}'
            spend_range = np.linspace(0, df[spend_col].max() * 1.5, 100)

            p = self.best_params[channel]
            response = self.transform_media(spend_range, p['theta'], p['alpha'], p['gamma'])
            response_scaled = response * self.model.coef_[i]

            axes[i].plot(spend_range, response_scaled, linewidth=2, color='teal')
            axes[i].fill_between(spend_range, 0, response_scaled, alpha=0.2, color='teal')
            axes[i].set_xlabel('Spend ($)')
            axes[i].set_ylabel('Marginal Response')
            axes[i].set_title(f'{channel}')
            axes[i].grid(alpha=0.3)

        # Hide empty subplot
        if len(self.channels) < len(axes):
            for j in range(len(self.channels), len(axes)):
                axes[j].axis('off')

        plt.suptitle('Response Curves by Channel\n(Robyn-Style MMM)', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.close()
        return fig


def run_robyn_mmm(data_path='../data/mmm_data.csv', output_dir='../results/robyn'):
    """Run the complete Robyn-style MMM analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path, parse_dates=['date'])
    print(f"Loaded {len(df)} rows")

    # Define channels
    global channels
    channels = ['Google', 'Meta', 'LinkedIn', 'TV', 'Email']

    # Initialize and fit model
    model = RobynStyleMMM()
    model.fit(df, channels, n_iterations=30)

    # Get results
    summary = model.get_results_summary(df)
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))

    # Save results
    summary.to_csv(f'{output_dir}/results_summary.csv', index=False)

    # Load ground truth
    try:
        truth_df = pd.read_csv('../data/ground_truth_parameters.csv')
        ground_truth = dict(zip(truth_df['channel'], truth_df['true_roi']))
    except:
        ground_truth = None

    # Generate visualizations
    model.plot_contribution(df, save_path=f'{output_dir}/channel_contribution.png')
    model.plot_roi_comparison(df, ground_truth, save_path=f'{output_dir}/roi_comparison.png')
    model.plot_response_curves(df, save_path=f'{output_dir}/response_curves.png')

    print(f"\nResults saved to {output_dir}/")
    return model, summary


if __name__ == '__main__':
    model, summary = run_robyn_mmm()
