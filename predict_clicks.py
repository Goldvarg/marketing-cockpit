import pandas as pd
import numpy as np

DATA_PATH = 'data/funnel_data.csv'

# Load data
funnel_df = pd.read_csv(DATA_PATH, parse_dates=['clicked_at'])
funnel_df['month'] = funnel_df['clicked_at'].dt.to_period('M')
monthly_clicks = funnel_df.groupby(['channel', 'month']).size().rename('clicks')

forecasts = {}
for channel, series in monthly_clicks.groupby(level=0):
    ts = series.droplevel(0)
    ts.index = ts.index.to_timestamp()
    ts = ts.resample('M').sum()
    pred_index = pd.date_range(ts.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')
    if len(ts) >= 2:
        x = np.arange(len(ts))
        y = ts.values
        slope, intercept = np.polyfit(x, y, 1)
        future_x = np.arange(len(ts), len(ts) + 12)
        pred_vals = intercept + slope * future_x
    else:
        pred_vals = np.repeat(ts.iloc[-1], 12)
    forecasts[channel] = pd.Series(pred_vals, index=pred_index).clip(lower=0)

forecast_df = pd.concat(forecasts, axis=1)
forecast_df.index.name = 'month'
forecast_df.to_csv('click_forecast.csv')
print('Forecast saved to click_forecast.csv')
print(forecast_df)
