# Titanic — Feature Engineering & Data Quality

Track I (AI Data Science) — I2: Feature Engineering & Data Quality gate project.

## Objective

Improve predictive signal for passenger survival through hypothesis-driven feature engineering, and audit the dataset for quality issues that could make a model unreliable or biased.

## Dataset

Kaggle's [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) `train.csv` (891 passengers). Kaggle's `test.csv` has no `Survived` column, so it isn't used for evaluation — a held-out test set is created from `train.csv` instead.

**Target:** `Survived` (1 = survived, 0 = died) — binary classification.

## Project structure

titanic-feature-engineering/
├── data/
│ └── train.csv
├── feature_engineering.ipynb
├── requirements.txt
└── README.md