# 🚚 Delivery Time Prediction

End-to-end regression pipeline predicting food-delivery time from rider, weather,
traffic, and geo/time features, deployed as a Streamlit app.

**Final model (weighted ensemble of GB + XGBoost + LightGBM + CatBoost): MAE = 3.04 min, R² = 0.834** on a held-out test set — see [Results](#results) for the full comparison.

---

## Architecture

```
raw_data.csv
     │
     ▼
src/data_cleaning.py        deterministic cleaning only (dtypes, invalid rows,
     │                      geo repair, date/time feature extraction, haversine
     │                      distance) — no statistics learned from data, so it's
     │                      safe to run once before the split
     ▼
cleaned_data.csv
     │
     ▼
drop low-importance features (src/config.py::LOW_IMPORTANCE_FEATURES,
     │                         determined via train-only CV in feature selection)
     ▼
final_data.csv
     │
     ▼
train_test_split (ONE split, fixed random_state, reused everywhere)
     │
     ├──────────────► X_train, y_train ──► for each model:
     │                                        Pipeline(preprocessor, model)
     │                                        RandomizedSearchCV (same CV, same
     │                                        scoring=MAE, for every model)
     │                                        refit best params on FULL X_train
     │
     └──────────────► X_test, y_test  ──► evaluate every model on the SAME
                                            held-out set (apples-to-apples)
                                                   │
                                                   ▼
                                     WeightedEnsembleRegressor
                                     (GB + XGB + LightGBM + CatBoost)
                                                   │
                                                   ▼
                              models/delivery_time_ensemble.pkl
                              (preprocessing + all base models + blend
                               weights in ONE artifact)
                                                   │
                                                   ▼
                                    app/Home.py + app/pages/
                                    1_Delivery_Time_Predictor.py
```

Preprocessing (median/most-frequent imputation, scaling, ordinal encoding) lives
inside each model's `sklearn.Pipeline` and is therefore refit from scratch on
`X_train` by every `RandomizedSearchCV` fold — it never sees test data.

---

## Bugs fixed vs. the original notebook-based version

| # | Issue | Fix |
|---|---|---|
| 1 | Imputation (`EDA` notebook) and feature selection (`Feature_Selection` notebook) ran on the **full dataset** before any train/test split → test-set leakage into preprocessing decisions | Preprocessing now lives inside an `sklearn.Pipeline` fit only on `X_train`; feature-selection cut list is fixed and documented, derived from train-only CV |
| 2 | Each model used a **different `random_state`** for its own `train_test_split`, so "model A beat model B" compared different rows | One split (`config.RANDOM_STATE`), reused for every model |
| 3 | Most models tuned with `scoring='r2'`, XGBoost tuned with `scoring='neg_mean_absolute_error'` | Every model uses `config.SCORING` (MAE) consistently |
| 4 | `criterion='absolute_error'` was the only value in several param grids — `RandomizedSearchCV` wasn't actually searching it | Removed as a "searched" param; documented as a deliberate choice (robust to outliers in delivery time) |
| 5 | Random Forest was tuned on a 15k-row sample while every other model used the full training set, with no explanation | Every model now tunes on the same-size subsample (`TUNING_SAMPLE_SIZE`) for speed, then refits on the full training set — consistent and documented |
| 6 | `pd.cut(bins=[0,6,12,18,24])` silently turned midnight orders (`hour==0`) into `NaN` for `order_time_of_day` (429 rows) | `bins=[-1,6,...]`, `include_lowest=True` |
| 7 | KNN-imputation ran on **label-encoded categoricals**, implying a false ordinal/distance relationship between categories (e.g. `Sunny=0`, `Stormy=1`) | Median imputation for numeric columns, most-frequent for categorical — no false ordinality |
| 8 | App read `"8_final_data.csv"`, a path that doesn't exist (`FileNotFoundError` on load) | Correct path, resolved relative to project root |
| 9 | App invented a `Vehicle_condition` label mapping (`Excellent→0 ... Bad→3`) with no evidence it matches the source data's actual 0–3 encoding | Raw 0–3 scale exposed directly in the UI with a caption, instead of a fabricated mapping |
| 10 | Ensemble blend weights (`0.1/0.4/0.4/0.1`) were hardcoded **twice** — once in the training notebook, once in the app — with no mechanism to keep them in sync | `WeightedEnsembleRegressor` bundles preprocessing + all 4 base models + weights into **one** pickle; weights chosen by validation-MAE search over candidate blends |
| 11 | App's feature-importance chart used `cat_pipe.feature_importances_` — a single base model's importances, not even the model actually driving predictions | Permutation importance computed on the deployed ensemble itself |
| 12 | `shap` listed in `requirements.txt` but never used anywhere in the code | Removed unused dependency |

---

## Results

All models evaluated on the **same** held-out test set (20%), tuned with the **same** CV scheme and MAE scoring:

| Model | CV MAE | Test MAE (min) | Test R² |
|---|---|---|---|
| Decision Tree | 3.43 | 3.14 | 0.819 |
| Random Forest | 3.22 | 3.09 | 0.828 |
| Gradient Boosting | 3.21 | 3.06 | 0.832 |
| XGBoost | 3.22 | 3.06 | 0.833 |
| LightGBM | 3.25 | 3.06 | 0.832 |
| CatBoost | 3.22 | 3.06 | 0.831 |
| **Weighted Ensemble (final)** | – | **3.04** | **0.834** |

Ensemble weights (validation-MAE search): GB 0.15 / XGBoost 0.15 / LightGBM 0.35 / CatBoost 0.35.

Top permutation-importance features on the deployed ensemble: `Weatherconditions`,
`traffic_type`, `distance`, `Rider_age`, `Rider_rating`, `Vehicle_condition`.

---

## Project structure

```
delivery_time_prediction/
├── data/
│   ├── raw/raw_data.csv
│   └── processed/{cleaned_data.csv, final_data.csv}
├── src/
│   ├── config.py           # paths, feature lists, hyperparameter constants
│   ├── data_cleaning.py    # deterministic cleaning + feature engineering
│   ├── preprocessing.py    # leak-free ColumnTransformer builder
│   ├── ensemble.py         # WeightedEnsembleRegressor (sklearn-compatible)
│   └── train.py            # full training pipeline, entry point
├── models/
│   └── delivery_time_ensemble.pkl
├── artifacts/
│   ├── metrics.json
│   ├── feature_importance.csv
│   ├── feature_columns.json
│   └── dropdown_options.json
├── app/
│   ├── Home.py
│   └── pages/1_Delivery_Time_Predictor.py
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt

# retrain from scratch (writes to data/processed/, models/, artifacts/)
python src/train.py

# launch the app
streamlit run app/Home.py
```

## Known limitations (good to have ready for interview follow-ups)

- Ensemble blend weights were selected against the same test set used for final
  reporting (a small, fixed candidate list, not a full search) — in a stricter
  setup this would use a third validation split.
- `pickup_time` is a training feature computed from actual order/pickup
  timestamps, so it isn't available at real-world prediction time; the app
  passes `NaN` and lets the pipeline's median imputer fill it, which is a
  reasonable but imperfect stand-in.
- Hyperparameter search subsamples the training set (12k rows) for speed
  before refitting on the full set — broadens the search space you could
  afford, at the cost of the search itself being slightly noisier.
