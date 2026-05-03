# Adult Income Analysis & Prediction

Exploratory data analysis and income classification on the UCI Adult Census dataset, extended with country-level drug use statistics. The project visualizes how demographic and socioeconomic factors relate to income level (≤50K vs >50K), then applies forward-feature-selection with XGBoost to identify the most predictive variables.

---

## Project Overview

- **EDA & Visualization** — Interactive bar charts (Altair) breaking down high-income rates across marital status, education, work class, race, and sex
- **Feature Engineering** — Label-encodes categorical variables; merges census data with country-level drug use rates (DALYs, 2019) and a flag/country lookup
- **Feature Selection** — Greedy forward selection using 5-fold cross-validated XGBoost accuracy to find the minimal high-performing feature set
- **Target Variable** — Binary classification: `income ≤50K` (1) vs `>50K` (0)

---

## Dataset

`cleaned_adult_data.csv` — a pre-cleaned version of the [UCI Adult dataset](https://archive.ics.uci.edu/ml/datasets/adult) with the following columns:

| Column | Description |
|---|---|
| `age` | Individual's age |
| `workclass` | Employment type (Private, Self-emp, Gov, etc.) |
| `education` | Highest education level |
| `education-num` | Education level encoded as a number |
| `marital-status` | Marital status |
| `occupation` | Job category |
| `relationship` | Family relationship role |
| `race` | Race |
| `sex` | Sex |
| `capital-gain` / `capital-loss` | Investment gains/losses |
| `hours-per-week` | Weekly working hours |
| `native-country` | Country of origin |
| `income` | Target label: `<=50K` or `>50K` |
| `flagCode` | Country flag code (merged) |
| `country` | Country name (merged) |
| `DrugUseRateByCountry_2023` | Country-level drug use rate (2023) |
| `DrugUseDALYs_2019` | Disability-adjusted life years from drug use (2019) |

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### Install Dependencies

```bash
pip install pandas altair xgboost scikit-learn notebook
```

Or if you prefer a `requirements.txt`:

```bash
pip install -r requirements.txt
```

<details>
<summary>requirements.txt contents</summary>

```
pandas
altair
xgboost
scikit-learn
notebook
```

</details>

### Run the Notebook

```bash
jupyter notebook code.ipynb
```

Make sure `cleaned_adult_data.csv` is in the same directory as the notebook before running.
