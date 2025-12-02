# Multi-Touch Attribution: Shapley Value Model

This folder contains a complete implementation of **Shapley value attribution** for marketing channels. The model calculates the fair marginal contribution of each marketing touchpoint in a customer's conversion journey.

## What is Shapley Value Attribution?

Shapley value attribution is a game-theory-based approach that fairly distributes credit among marketing channels by:

1. **Considering all possible orderings**: It evaluates how each channel contributes across all possible sequences of touchpoints
2. **Measuring marginal contributions**: For each channel, it calculates the additional value when that channel is added to different combinations
3. **Fair allocation**: The Shapley value represents the average marginal contribution, providing a mathematically fair way to attribute credit

### Shapley Value vs Other Attribution Models

| Model | Description | Pros | Cons |
|-------|-------------|------|------|
| **Last Touch** | 100% credit to last touchpoint | Simple | Ignores customer journey |
| **First Touch** | 100% credit to first touchpoint | Simple | Ignores nurturing |
| **Linear** | Equal credit to all touchpoints | Fair distribution | No consideration of importance |
| **Time Decay** | More credit to recent touchpoints | Considers recency | Arbitrary decay function |
| **Shapley Value** | Credit based on marginal contribution | Mathematically fair, considers all combinations | Computationally intensive |

## Files

- **`shapley_attribution.py`**: Core implementation of the Shapley value attribution model
- **`run_attribution_analysis.py`**: Script to run complete analysis with visualizations
- **`README.md`**: This documentation file

## Installation

Ensure you have the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn
```

## Usage

### Quick Start

Run the complete attribution analysis:

```bash
cd /home/user/marketing-cockpit
python multi_touch_attribution/run_attribution_analysis.py
```

This will:
1. Load marketing clicks and order data from `raw_data/`
2. Build customer journeys
3. Calculate Shapley values for each channel
4. Calculate removal effects
5. Generate visualizations (PNG files)
6. Create a comprehensive report

### Using the ShapleyAttribution Class

You can also use the model programmatically:

```python
from multi_touch_attribution.shapley_attribution import ShapleyAttribution

# Initialize model
model = ShapleyAttribution()

# Load data
model.load_data(
    marketing_clicks_path='raw_data/marketing_clicks.csv',
    orders_path='raw_data/orders.csv',
    attribution_window_days=30  # Attribution window
)

# Calculate Shapley values
shapley_values = model.calculate_shapley_values(
    use_revenue=True,      # Use revenue (True) or conversion count (False)
    sample_size=50000      # Sample for faster computation (optional)
)

# Calculate removal effects
removal_effects = model.calculate_removal_effects()

# Get summary
summary = model.get_attribution_summary()
print(summary)

# Analyze channel interactions
interactions = model.analyze_channel_interactions()
print(interactions.head(10))
```

## Key Metrics

### 1. Shapley Value

The **Shapley value** for a channel represents its average marginal contribution across all possible combinations of channels.

**Formula**:
```
φᵢ = Σ [|S|! × (n - |S| - 1)! / n!] × [v(S ∪ {i}) - v(S)]
```

Where:
- `S` is a subset of channels not containing channel `i`
- `v(S)` is the conversion value for subset `S`
- `n` is the total number of channels
- `|S|` is the size of subset `S`

**Interpretation**:
- Higher Shapley value = greater contribution to conversions
- Can be used to optimize budget allocation
- Fair attribution across all channels

### 2. Removal Effect

The **removal effect** shows the impact of removing a channel from the full set of channels.

**Formula**:
```
Removal Effect(i) = v(All Channels) - v(All Channels \ {i})
```

**Interpretation**:
- Shows the value lost if a channel is eliminated
- Useful for understanding channel dependencies
- Helps identify critical channels

## Output Files

After running the analysis, the following files are generated in `multi_touch_attribution/`:

### Data Files
- **`shapley_results.csv`**: Summary table with Shapley values and removal effects for each channel
- **`channel_interactions.csv`**: Conversion values for different channel combinations

### Visualizations
- **`shapley_values.png`**: Bar chart of Shapley values by channel
- **`removal_effects.png`**: Bar chart of removal effects by channel
- **`attribution_comparison.png`**: Pie charts comparing Shapley and removal distributions
- **`shapley_vs_removal.png`**: Side-by-side comparison of both metrics

### Report
- **`ATTRIBUTION_REPORT.md`**: Complete analysis report with findings and recommendations

## Data Requirements

The model expects two CSV files in the `raw_data/` folder:

### 1. marketing_clicks.csv

| Column | Description |
|--------|-------------|
| `user_id` | Unique identifier for the user |
| `click_ts` | Timestamp of the click |
| `channel` | Marketing channel (e.g., Google, Meta, LinkedIn) |

### 2. orders.csv

| Column | Description |
|--------|-------------|
| `user_id` | Unique identifier for the user |
| `order_ts` | Timestamp of the order |
| `revenue` | Revenue from the order |

## Configuration

You can adjust the following parameters in `run_attribution_analysis.py`:

```python
ATTRIBUTION_WINDOW = 30    # Days to look back for attributing clicks
SAMPLE_SIZE = 50000        # Number of journeys to sample (faster computation)
USE_REVENUE = True         # Use revenue (True) or conversion count (False)
```

## Algorithm Details

### Step 1: Build Customer Journeys

The model creates customer journeys by:
1. Grouping marketing clicks by user
2. Linking clicks to conversions within the attribution window
3. Creating sequences of touchpoints for each conversion

### Step 2: Calculate Conversion Values

For each possible subset of channels, calculate:
- Total conversions or revenue when those channels are present
- Average conversion value per journey

### Step 3: Calculate Shapley Values

For each channel:
1. Iterate through all possible subsets not containing the channel
2. Calculate marginal contribution when adding the channel
3. Weight by the Shapley formula
4. Sum across all subsets

### Step 4: Calculate Removal Effects

For each channel:
1. Calculate value with all channels
2. Calculate value without the channel
3. Compute the difference

## Performance Considerations

- **Computational Complexity**: O(2^n) where n is the number of channels
- **Sampling**: For large datasets, use `sample_size` parameter to sample journeys
- **Memory**: Stores conversion values for all channel subsets
- **Recommended**: Use sampling for datasets with >100k journeys

## Example Output

```
ATTRIBUTION SUMMARY
================================================================================
    channel  shapley_value  removal_effect  shapley_pct  removal_pct
     Google         145.23          189.45        42.3%        45.2%
       Meta          98.67          121.34        28.7%        28.9%
   LinkedIn          79.45           95.23        23.1%        22.7%
     Direct          20.12           13.45         5.9%         3.2%
```

## Interpretation Guide

### High Shapley Value + High Removal Effect
→ **Critical channel**: Major contributor, would be costly to remove

### High Shapley Value + Low Removal Effect
→ **Independent contributor**: Works well on its own, less dependent on other channels

### Low Shapley Value + High Removal Effect
→ **Supporting channel**: Enhances other channels, important for synergy

### Low Shapley Value + Low Removal Effect
→ **Limited impact**: Consider reducing investment or eliminating

## References

- Shapley, L. S. (1953). "A Value for n-person Games"
- Dalessandro, B., et al. (2012). "Causally Motivated Attribution for Online Advertising"
- Shao, X., & Li, L. (2011). "Data-driven Multi-touch Attribution Models"

