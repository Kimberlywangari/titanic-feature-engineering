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


## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Place `train.csv` inside the `data/` folder, then open `I2_feature_engineering.ipynb` and run all cells top to bottom.

## Notebook structure

1. Setup & data loading
2. Data-quality audit
3. Leakage & class imbalance checks
4. Train/test split & baseline pipeline
5. Feature engineering (with hypotheses)
6. Engineered model
7. Before/after comparison & hypothesis evaluation
8. Conclusion

## Data-quality findings

- **`Age`** missing for 19.9% of passengers, and missingness is not random — concentrated in 3rd class, where survival is also lowest.
- **`Cabin`** missing for 77.1% of passengers; having a recorded cabin correlates with class and survival even within the same class.
- **`Fare`** is a ticket-level price, not per-person — shared across passengers on the same ticket, with 15 zero-fare rows.
- **39% of passengers share a ticket** with another passenger (families/groups), so observations are not independent — random splits can leak family information across train/test.
- Strong survival gaps across protected attributes (`Sex`, `Pclass`), reflecting the ship's rescue policy rather than a fair or generalizable rule.

## Engineered features

| Feature | Source columns | Hypothesis |
|---|---|---|
| `Title` | `Name` | Recovers status/age/marital signal lost to missing `Age` |
| `FamilyGroup` | `SibSp`, `Parch` | Bucketing captures the non-linear survival curve by family size |
| `HasCabin` | `Cabin` | Flags upper-deck passengers; treated cautiously (possible leakage) |
| `FarePerPerson` | `Fare`, `Ticket` | Corrects `Fare` being a shared ticket-level price |

## Results

| Version | CV AUC | Test AUC | Test Recall | Test F1 |
|---|---|---|---|---|
| Baseline | 0.8526 | 0.8464 | 0.7826 | 0.7606 |
| Engineered | 0.8710 | 0.8690 | 0.8261 | 0.7862 |

The engineered model improved on every metric and won 22 of 25 cross-validation folds. Almost all of the gain came from `Title`; `FamilySize` (raw count) added nothing, and a ticket-grouped CV check showed part of the improvement was due to family-ticket leakage rather than new signal.

## Key takeaways

- Not every engineered feature adds value — a feature that's a linear combination of existing columns (raw `FamilySize`) adds nothing to a linear model.
- Data-quality issues (missingness patterns, shared tickets, protected-attribute skew) directly affect how much a result should be trusted.
- Model improvement should always be checked against the same split, same folds, and same metric to be meaningful.