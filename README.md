# Promotion Response Prediction

A machine learning pipeline that predicts whether a retail customer will respond to a promotional offer. Accurate prediction enables improved campaign targeting, reduced marketing waste, and higher ROI.

**Validation AUC: 0.713** · Algorithm: LightGBM · Task: Binary classification

---

## Overview

This project uses historical transaction data and promotion metadata to estimate the probability of customer engagement with future promotional offers. The task is formulated as a binary classification problem (`active = 1` if responded, `0` otherwise) and evaluated using ROC AUC on a held-out validation set.

A key design principle throughout is **temporal leakage prevention** — all features are computed strictly from transactions that occurred before each promotion date, ensuring the model reflects realistic deployment conditions.

---

## Dataset

The dataset consists of anonymized retail data (excluded from this repo for privacy) and includes:

| File | Description |
|------|-------------|
| `transactions.parquet` | Historical customer transactions (product, quantity, spend, date) |
| `promos.parquet` | Promotion metadata (category, brand, manufacturer, quantity, value) |
| `train_history.parquet` | Promotions offered in March 2013 with observed response labels |
| `test_history.parquet` | Promotions offered in April 2013 without response labels |
| `data_dictionary.xlsx` | Detailed field-level documentation |

---

## Methodology

### 1 · Feature Engineering

Over 50 features were engineered across six behavioural dimensions:

**RFM behaviour**
- Recency: days since last transaction at overall, brand, category, and manufacturer levels
- Frequency: total transaction counts, unique categories and brands purchased, recent 30-day activity
- Monetary: cumulative and average quantity and spend, spending variability (coefficient of variation)

**Brand & category affinity**
- Brand loyalty scores and spend shares within category
- Category purchase history and diversity metrics
- Brand recency and cumulative transaction counts

**Store-level context**
- Store average transaction quantity and spend
- Customer–store affinity ratios normalised against store-level baselines
- Store historical response rates

**Price sensitivity**
- Promotion value and quantity relative to customer historical averages
- Spending variability as a proxy for price responsiveness

**Temporal signals**
- Day-of-week, month, weekend indicators
- Weekend shopper flag derived from historical purchase patterns

**Competitive pressure**
- Count of competing promotions in the same store and category over a 30-day window
- Manufacturer-level recency signals

High-cardinality categorical variables (customer ID, store, brand, category, manufacturer, promotion ID) were encoded using **Bayesian-smoothed target encoding**, preserving predictive signal while mitigating overfitting on rare categories. Log transforms, interaction features, and ratio features were applied to improve robustness.

---

### 2 · Model Selection

LightGBM (gradient boosting decision trees) was selected as the primary model. Neural networks were not considered due to the tabular structure of the data and the overfitting patterns already observed with LightGBM.

Hyperparameters were tuned via grid search over tree structure, regularisation, and sampling parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_leaves` | 40 | Controls tree complexity |
| `max_depth` | 12 | Caps maximum depth |
| `learning_rate` | 0.02 | Slow learning for stability |
| `feature_fraction` | 0.8 | Column subsampling |
| `lambda_l1` / `lambda_l2` | 0.1 / 0.1 | Light regularisation |
| `min_child_samples` | 20 | Prevents tiny leaf nodes |
| Early stopping rounds | 50 | Prevents over-training |

Validation was performed on a stratified 80/20 split to preserve the response rate distribution.

---

### 3 · Feature Selection

Following model training, feature pruning was attempted based on LightGBM split-importance scores. Variables with consistently low importance were candidates for removal to reduce dimensionality and improve generalisation. However, removing low-importance features did not improve validation AUC after threshold tuning, so **all features were retained** in the final model. Validation AUC was treated as the sole optimisation target.

---

## Results

| Split | AUC |
|-------|-----|
| Training | 0.8751 |
| Validation | 0.7134 |

The train–validation gap indicates overfitting — a known trade-off when retaining all features. Validation AUC was prioritised as the primary metric throughout.

**Key drivers of promotion response** (in order of importance):
1. Customer engagement intensity — total spend, transaction frequency, average order value
2. Recency of activity — days since last purchase, recent 30-day purchase count
3. Brand affinity — brand loyalty score, cumulative brand spend, brand recency
4. Store-level context — normalised behavioural ratios, store response rates

Manufacturer-level variables and calendar indicators (day of week, month) showed the least predictive value.

---

## Visualisations

The following charts are generated by the notebook and included in the project portfolio page:

- **ROC curve** — true vs false positive trade-off across all thresholds
- **Predicted probability distribution** — separation between responders and non-responders
- **Feature importance bar chart** — top 20 features by LightGBM split count
- **SHAP summary plot** — feature impact direction and magnitude across the validation set
- **SHAP waterfall** — individual-level explanation for the highest-scored customer

---

## Project Structure

```
project-promotion-response-prediction/
├── data/                  # Raw data files (excluded from repo)
├── notebook/
│   └── promotion_response_model_v2.ipynb   # Full pipeline with visualisations
├── report/
│   └── promotion_response_modeling_report.pdf
├── output/                # Generated predictions CSV (excluded from repo)
├── README.md
└── .gitignore
```

---

## Reproducing Results

The notebook is fully executable on **Google Colab** with no local setup required.

1. Upload `promotion_response_model_v2.ipynb` to Google Colab
2. Upload the data files to the Colab session storage under `data/`
3. Run all cells — total runtime is under 10 minutes
4. Charts are saved as PNG files and downloaded automatically via the final cell

---

## Challenges & Future Work

**Challenges encountered:**
- High sparsity in customer purchase histories produced unstable RFM metrics — mitigated via smoothing and aggregation
- High-cardinality categoricals required careful target encoding to avoid leakage and overfitting
- Extensive feature set raised interpretability concerns; a trade-off was made between maximising AUC and managing complexity

**Future directions:**
- Rolling temporal features — trend and seasonality signals over 60/90-day windows
- Customer cohort segmentation — separate models per RFM tier for personalised targeting
- Ensemble stacking — combine LightGBM with a calibrated logistic regression for better probability estimates
- SHAP-based feature selection — more stable pruning than split-count importance
- Precision–recall threshold optimisation — tune the decision threshold based on business cost of false positives vs false negatives

---

## AI Usage Acknowledgement

Generative AI tools (ChatGPT, Claude) were used as supplementary aids for code drafting, debugging suggestions, and documentation clarification. All generated code and explanations were independently reviewed, validated, and adapted before inclusion in the project.

Example prompts used during development:
- *"Help refactor this LightGBM training loop to include early stopping."*
- *"Propose memory-efficient feature engineering for RFM-like customer features on transaction history."*
