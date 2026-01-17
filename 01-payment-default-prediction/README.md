## Payment Default Prediction Model

**Author**: Mathias Gomez Chan  
**Date**: January 2026  
**Status**: Portfolio Project #1

## Executive Summary

This project develops a logistic regression model to predict credit card payment defaults using 30,000 customer records from April-September 2005. Through feature engineering and business cost-driven threshold optimization, the model achieves 94.8% recall in identifying defaulters, reducing business costs by NT$55.7 million (77.9% reduction) compared to the baseline threshold.

## Business Problem

Credit card default prediction is critical for financial institutions to minimize losses while maintaining customer satisfaction. Incorrectly identifying customers as high-risk leads to lost revenue from rejected applicants, while missing actual defaulters results in significant financial losses. With missed defaults costing 25 times more than false rejections (NT$65,055 vs NT$2,602), accurate prediction with high recall is essential for profitability.

## Dataset

- **Source**: [UCI Credit Card Default Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- **Size**: 30,000 customers, 24 features
- **Target**: Payment default in October 2005 (22.12% default rate)
- **Time Period**: April-September 2005 payment history

## Data Setup

The dataset is not included in this repository. Download it from:
- [UCI Credit Card Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

Place the downloaded `UCI_Credit_Card.csv` file in the `data/` directory.

## Methodology

### 1. Exploratory Data Analysis

Statistical hypothesis testing was used to rank 23 features by their association with default status:
- **Continuous features**: Mann-Whitney U test with point-biserial correlation
- **Categorical features**: Chi-square test with Cramér's V
- **Finding**: Payment history variables (PAY_0-PAY_6) dominated feature importance (Cramér's V > 0.25)

### 2. Feature Engineering

To address multicollinearity among the 6 correlated payment status variables (r > 0.7 for adjacent months), four engineered features were created:

1. **PAY_recent_avg**: Average payment status (last 3 months)
   - Correlation with default: 0.31
   - Captures current payment behavior

2. **PAY_max_delay**: Maximum delay across all 6 months
   - Correlation with default: 0.33 (strongest predictor)
   - Captures worst historical behavior

3. **PAY_trend**: Payment behavior trajectory (PAY_0 - PAY_6)
   - Correlation with default: 0.13
   - Positive = worsening, Negative = improving

4. **PAY_std**: Standard deviation of payment status
   - Correlation with default: 0.25
   - Captures payment consistency (higher = more erratic)

### 3. Modeling

- **Algorithm**: Logistic Regression (statsmodels for statistical inference)
- **Features**: 5 (4 engineered payment features + LIMIT_BAL)
- **Training**: 80/20 stratified split (24,000 train / 6,000 test)
- **Evaluation**: Confusion matrix, Precision, Recall, ROC-AUC, Business Cost

**All 5 features were statistically significant (p < 0.05)**, with payment behavior features showing the strongest effects.

### 4. Threshold Optimization

The 0.05 threshold is driven by cost analysis: missed defaults cost NT$65,055 versus NT$2,602 for false alarms—a 25:1 ratio. At this threshold, the model catches 95% of defaulters instead of 18%, reducing total business costs by 78% (NT$55.7 million saved). While precision drops from 62% to 22%, the business math clearly favors high recall when false negatives are 25 times more expensive than false positives.

## Results

| Metric | Baseline (0.5) | Optimized (0.05) | Improvement |
|--------|---------------|------------------|-------------|
| Recall | 17.63% | **94.80%** | +77.2pp |
| Precision | 61.58% | 22.42% | -39.2pp |
| F1-Score | 27.42% | 36.26% | +8.8pp |
| ROC-AUC | 71.54% | 71.54% | - |
| Missed Defaults (FN) | 1,093 | **69** | -93.7% |
| Total Cost | NT$71.5M | **NT$15.8M** | **-77.9%** |

**Key Finding**: Business cost-driven threshold optimization reduced losses by **NT$55.7 million (77.9%)** while maintaining strong discriminative ability (ROC-AUC = 71.54%).

## Key Technologies

- Python 3.13
- pandas, numpy (data manipulation)
- scikit-learn (modeling pipeline, metrics)
- statsmodels (logistic regression with MLE, statistical inference)
- scipy (hypothesis testing)
- matplotlib, seaborn (visualization)

## Repository Structure
```
payment-default-prediction/
├── README.md
├── requirements.txt
├── notebook.ipynb          # Main analysis
└── data/
    └── UCI_Credit_Card.csv
```

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Mathias70473/data-science-portfolio.git
cd data-science-portfolio/01-payment-default-prediction
```

### 2. Download the Dataset
The dataset is not included in this repository. Download from [Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) and place `UCI_Credit_Card.csv` in the `data/` folder.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Notebook
```bash
jupyter notebook payment_default_prediction.ipynb
```

**Note**: Make sure the dataset is in `data/UCI_Credit_Card.csv` before running the notebook.
```

## Business Impact

This model demonstrates cost-based decision optimization in a real-world credit risk scenario. By aligning the classification threshold with business costs rather than relying on default ML metrics (e.g., accuracy, F1), the model achieves significant ROI while maintaining high recall for default detection. The approach shows how domain knowledge and business constraints should drive model deployment decisions.

## Future Improvements

- Test ensemble methods (Random Forest, XGBoost) for improved discrimination
- Implement time-series cross-validation for temporal robustness
- Engineer interaction features (e.g., payment trend × credit limit)
- Build interpretability tools (SHAP values, partial dependence plots)
- Deploy as REST API for real-time credit scoring

## Contact

www.linkedin.com/mathiasgomez-ds | https://github.com/Mathias70473 | mathias70473@gmail.com


---

**Built as Portfolio Project #1 | January 2026**
