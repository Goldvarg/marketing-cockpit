"""
Google Meridian Marketing Mix Model

Meridian is Google's open-source Bayesian Marketing Mix Model framework.
It uses Bayesian inference to estimate channel effectiveness with uncertainty.

Pros:
- Provides uncertainty estimates (credible intervals)
- Handles adstock and saturation automatically
- Industry-standard approach from Google
- Built-in diagnostics and visualization

Cons:
- Slower to run (MCMC sampling)
- Steeper learning curve
- Requires TensorFlow Probability
- Less flexibility in model specification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MeridianMMM:
    """
    Wrapper for Google's Meridian Marketing Mix Model.
    """

    def __init__(self):
        self.model = None
        self.input_data = None
        self.channels = []
        self.results = {}

    def prepare_data(self, df, channels, target_col='revenue'):
        """
        Prepare data in the format required by Meridian.
        """
        from meridian.data import data_frame_input_data_builder as dfib

        # Rename columns to Meridian's expected format
        data = df.copy()
        data = data.rename(columns={'date': 'time'})

        # Build input data
        builder = dfib.DataFrameInputDataBuilder(
            kpi_type='revenue',
            default_kpi_column=target_col
        )

        # Add KPI (target variable)
        builder = builder.with_kpi(data, time_col='time')

        # Add control variables if present
        control_cols = []
        for col in ['is_holiday', 'is_promo', 'seasonality_index', 'competitor_spend_index']:
            if col in data.columns:
                control_cols.append(col)

        if control_cols:
            builder = builder.with_controls(data, control_cols=control_cols, time_col='time')

        # Add media channels
        media_cols = [f'impressions_{ch.lower()}' for ch in channels]
        spend_cols = [f'spend_{ch.lower()}' for ch in channels]

        builder = builder.with_media(
            data,
            media_cols=media_cols,
            media_spend_cols=spend_cols,
            media_channels=channels,
            time_col='time'
        )

        self.input_data = builder.build()
        self.channels = channels

        print(f"Prepared data with {len(channels)} channels")
        return self.input_data

    def fit(self, df, channels, target_col='revenue',
            n_chains=4, n_adapt=1000, n_burnin=500, n_keep=1000):
        """
        Fit the Meridian MMM model.

        Args:
            df: DataFrame with marketing data
            channels: List of channel names
            target_col: Name of target variable
            n_chains: Number of MCMC chains
            n_adapt: Number of adaptation steps
            n_burnin: Number of burn-in steps
            n_keep: Number of samples to keep
        """
        import tensorflow_probability as tfp
        from meridian.model import prior_distribution, spec, model

        print("\n" + "="*60)
        print("FITTING GOOGLE MERIDIAN MMM")
        print("="*60)

        # Prepare data
        self.prepare_data(df, channels, target_col)

        # Set up priors
        # Using weakly informative priors for ROI
        roi_mu, roi_sigma = 0.2, 0.9  # Log-normal prior centered around ROI ~1.2
        priors = prior_distribution.PriorDistribution(
            roi_m=tfp.distributions.LogNormal(roi_mu, roi_sigma, name='roi_m')
        )

        # Model specification
        model_spec = spec.ModelSpec(
            prior=priors,
            max_lag=14,  # Maximum adstock lag in days
            hill_before_adstock=False,
            knots=10
        )

        # Initialize model
        self.model = model.Meridian(
            input_data=self.input_data,
            model_spec=model_spec
        )

        # Sample prior (for diagnostics)
        print("\nSampling prior...")
        self.model.sample_prior(200)

        # Sample posterior
        print(f"Sampling posterior ({n_chains} chains, {n_keep} samples each)...")
        print("This may take a few minutes...")

        self.model.sample_posterior(
            n_chains=n_chains,
            n_adapt=n_adapt,
            n_burnin=n_burnin,
            n_keep=n_keep,
            seed=42
        )

        print("Sampling complete!")
        return self

    def get_roi_estimates(self):
        """Extract ROI estimates with credible intervals."""
        from meridian.analysis import analyzer

        anal = analyzer.Analyzer(self.model)
        roi_summary = anal.summary_metrics()

        return roi_summary

    def plot_model_fit(self, save_path=None):
        """Plot model fit vs actual data."""
        from meridian.analysis import visualizer

        model_fit = visualizer.ModelFit(self.model)
        fig = model_fit.plot_model_fit()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()

        return fig

    def plot_channel_contribution(self, save_path=None):
        """Plot channel contribution breakdown."""
        from meridian.analysis import visualizer

        media_summary = visualizer.MediaSummary(self.model)
        fig = media_summary.plot_contribution_pie_chart()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()

        return fig

    def plot_roi_comparison(self, save_path=None):
        """Plot ROI by channel."""
        from meridian.analysis import visualizer

        media_summary = visualizer.MediaSummary(self.model)
        fig = media_summary.plot_roi_bar_chart()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()

        return fig

    def plot_response_curves(self, save_path=None):
        """Plot response curves showing diminishing returns."""
        from meridian.analysis import visualizer

        media_effects = visualizer.MediaEffects(self.model)
        fig = media_effects.plot_response_curves()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()

        return fig

    def plot_rhat_diagnostics(self, save_path=None):
        """Plot convergence diagnostics."""
        from meridian.analysis import visualizer

        diagnostics = visualizer.ModelDiagnostics(self.model)
        fig = diagnostics.plot_rhat_boxplot()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()

        return fig

    def get_results_summary(self):
        """Generate comprehensive results summary."""
        from meridian.analysis import analyzer

        anal = analyzer.Analyzer(self.model)

        # Get ROI estimates
        try:
            roi_df = anal.roi_summary()
            return roi_df
        except:
            # Fallback to basic metrics
            return pd.DataFrame({
                'channel': self.channels,
                'note': ['See visualizations for detailed results'] * len(self.channels)
            })


def run_meridian_mmm(data_path='../data/mmm_data.csv', output_dir='../results/meridian'):
    """
    Run the complete Meridian MMM analysis.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path, parse_dates=['date'])
    print(f"Loaded {len(df)} rows")

    # Define channels
    channels = ['Google', 'Meta', 'LinkedIn', 'TV', 'Email']

    # Initialize and fit model
    model = MeridianMMM()

    try:
        model.fit(
            df, channels,
            n_chains=4,
            n_adapt=1000,
            n_burnin=500,
            n_keep=1000
        )

        # Generate visualizations
        print("\nGenerating visualizations...")
        model.plot_model_fit(save_path=f'{output_dir}/model_fit.png')
        model.plot_channel_contribution(save_path=f'{output_dir}/channel_contribution.png')
        model.plot_roi_comparison(save_path=f'{output_dir}/roi_comparison.png')
        model.plot_response_curves(save_path=f'{output_dir}/response_curves.png')
        model.plot_rhat_diagnostics(save_path=f'{output_dir}/convergence_diagnostics.png')

        # Get summary
        summary = model.get_results_summary()
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(summary)

        if hasattr(summary, 'to_csv'):
            summary.to_csv(f'{output_dir}/results_summary.csv', index=False)

        print(f"\nResults saved to {output_dir}/")

    except Exception as e:
        print(f"\nError during model fitting: {e}")
        print("This may be due to data format issues or computational constraints.")

    return model


if __name__ == '__main__':
    model = run_meridian_mmm()
