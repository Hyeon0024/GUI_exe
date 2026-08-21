# NaOH Lignin Removal Predictor

A Streamlit GUI for predicting **lignin removal (%)** from lignocellulosic biomass under NaOH alkaline pretreatment, using pre-trained machine learning models.

## Overview

Given four process parameters, the app predicts lignin removal using five candidate regression models and lets you compare their predictions, uncertainty, and sensitivity interactively.

**Input parameters**

| Parameter | Description | Training range |
|---|---|---|
| Severity Factor (R₀) | log₁₀ severity factor of the pretreatment | 4.25 – 4.95 |
| Bark inclusion | whether the biomass sample includes bark | 0 (without) / 1 (with) |
| Pretreatment Time (h) | duration of NaOH pretreatment | 12 – 24 |
| NaOH Concentration (%) | NaOH solution concentration (w/v) | 0.5 – 2.0 |

**Models** (ranked by test R², random 80/20 split, mean ± std over 5 repeats)

| Model | Test R² | Test RMSE (%) |
|---|---|---|
| XGBoost | 0.9848 ± 0.0135 | 0.2984 ± 0.0893 |
| Extra Trees | 0.9533 ± 0.0868 | 0.3911 ± 0.4730 |
| SVR | 0.9239 ± 0.0647 | 0.6899 ± 0.2972 |
| Random Forest | 0.9193 ± 0.0548 | 0.7281 ± 0.2556 |
| Polynomial Ridge | 0.6548 ± 0.1272 | 1.5514 ± 0.1243 |

## Features

- **Prediction** — gauge chart with predicted lignin removal, comparison to training mean, percentile, and an approximate 95% interval (±2×Test RMSE).
- **Extrapolation warning** — flags inputs that fall outside the training data's discrete experimental levels.
- **Sensitivity analysis** — sweeps one feature across its experimental range while holding the others fixed, showing predictions from all five models at once.
- **Multi-model comparison** — side-by-side point predictions with error bars for the current input.
- **Training data distribution** — histogram and summary statistics of the observed lignin removal values, with the current prediction overlaid.

## Project structure

```
GUI_exe/
├── data/
│   └── wood_lignin_NaOH.csv       # training data (36 experimental conditions)
├── models/
│   ├── tuned_xgboost.pkl
│   ├── tuned_extra_trees.pkl
│   ├── tuned_svr.pkl
│   ├── tuned_random_forest.pkl
│   └── tuned_polynomial_ridge.pkl
├── scripts/
│   └── predict_gui.py             # Streamlit app
└── run_gui.bat                    # Windows launcher
```

## Requirements

- Python 3.9+
- streamlit
- pandas
- numpy
- joblib
- plotly
- scikit-learn
- xgboost

Install with:

```bash
pip install streamlit pandas numpy joblib plotly scikit-learn xgboost
```

## Usage

**Windows**

Double-click `run_gui.bat`, or run it from a terminal:

```bat
run_gui.bat
```

**Cross-platform**

```bash
python -m streamlit run scripts/predict_gui.py
```

The app opens in your browser at `http://localhost:8501`.

## Notes

- All performance metrics are computed on a held-out test set (random 80/20 split, *n* = 108 samples from 36 experimental conditions, repeated 5× with `random_state=42–46`), not on the training set.
- Predictions for inputs outside the discrete levels used during training (R₀, time, concentration) are extrapolations and are flagged in the UI accordingly.
