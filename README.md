# Fraud Detection – Task 1 (Data Prep)

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-EDA%20%26%20Prep-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Modeling-F7931E?logo=scikitlearn&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)

Repository for e-commerce and credit card fraud detection. Task 1 covers data cleaning, EDA, feature engineering, scaling/encoding, and imbalance handling.

## Folder Structure

```
fraud-detection/
├── .github/workflows/       # CI
├── .vscode/                 # Editor settings
├── data/
│   ├── raw/                 # Place raw data here (Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv)
│   └── processed/           # Cleaned + processed outputs (ignored by git)
├── models/                  # Saved scalers/models
├── notebooks/
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   └── shap-explainability.ipynb
├── scripts/                 # (reserved)
├── tests/
├── requirements.txt
└── README.md
```

## Prerequisites
- Python 3.11+ (conda env recommended)
- Raw datasets in `data/raw/`:
	- `Fraud_Data.csv`
	- `IpAddress_to_Country.csv`
	- `creditcard.csv`

## Setup

```bash
conda activate kaim_week1  # or python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## How to Run Task 1

1) EDA (optional but recommended)
- Open `notebooks/eda-fraud-data.ipynb` and `notebooks/eda-creditcard.ipynb`
- Run all cells to view class imbalance, univariate/bivariate plots, and country patterns (Fraud_Data only).

2) Feature Engineering & Processing
- Open `notebooks/feature-engineering.ipynb`
- Run all cells top-to-bottom. It will:
	- Clean data (types, missing, duplicates)
	- Map IP → country via range lookup
	- Create time, velocity, and behavioral features
	- One-hot encode + scale numeric features
	- Apply SMOTE to training splits only
	- Save outputs to `data/processed/`:
		- `fraud_data_cleaned.csv`, `ecommerce_train_processed.csv`, `ecommerce_test_processed.csv`
		- `creditcard_cleaned.csv`, `creditcard_train_processed.csv`, `creditcard_test_processed.csv`
		- Scalers in `models/` (e.g., `ecommerce_scaler.pkl`, `creditcard_scaler.pkl`)

## Notes
- `data/` is git-ignored; keep raw and processed files there.
- SMOTE is applied only on the training folds; class distributions before/after are printed in the notebook.
- Modeling and SHAP analysis continue in `notebooks/modeling.ipynb` and `notebooks/shap-explainability.ipynb`.