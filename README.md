# Marketing Cockpit

A practical toolkit for understanding marketing effectiveness through data analysis.

## What's Inside

### 1. [Marketing Mix Modeling](marketing_mix_model/)
Answers: **"How much revenue does each marketing channel generate?"**

Compares three different approaches:
- **Simple Regression** - Fast and easy to understand
- **Google Meridian** - Bayesian model with uncertainty estimates
- **Meta Robyn-Style** - Automated hyperparameter optimization

### 2. [Multi-Touch Attribution](multi_touch_attribution/)
Answers: **"Which channels deserve credit for conversions?"**

Reveals how last-touch attribution undervalues "introducer" channels (Display, Content) while overvaluing "closer" channels (Paid Search, Direct).

---

## Quick Start

```bash
# Marketing Mix Modeling
cd marketing_mix_model
python run_all_models.py

# Multi-Touch Attribution
cd multi_touch_attribution
python run_analysis.py
```

---

## Key Findings

### From Marketing Mix Modeling
| Channel | ROI | Verdict |
|---------|-----|---------|
| Email | ~4.0x | Best efficiency, limited scale |
| Google | ~2.5x | Strong performer |
| Meta | ~2.0x | Solid returns |
| LinkedIn | ~1.8x | Good for B2B |
| TV | ~1.5x | Brand building |

### From Multi-Touch Attribution
| Channel | Last Touch Credit | Fair Credit | Issue |
|---------|-------------------|-------------|-------|
| Display | 6% | 20% | Severely undervalued |
| Paid Search | 36% | 19% | Overvalued |

---

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, scikit-learn, scipy
- google-meridian (for Meridian model)

```bash
pip install pandas numpy matplotlib scikit-learn scipy google-meridian
```
