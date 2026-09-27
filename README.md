# Titanic Survival Prediction

This project is an introductory, end-to-end machine-learning walkthrough for
predicting whether a passenger survived the Titanic disaster. It uses the
well-known Kaggle Titanic training data and a logistic-regression classifier.
The complete analysis, visual exploration, preprocessing, training, and
evaluation workflow lives in a Jupyter notebook.

> **Scope:** This repository is designed as a learning project and baseline
> model. It demonstrates a clear, reproducible workflow rather than claiming
> production-ready predictive performance.

## Project highlights

- Explores **891 labeled passenger records** and their missing values.
- Visualizes survival counts by survival outcome, sex, and passenger class.
- Cleans missing values and encodes categorical model inputs.
- Trains a `scikit-learn` `LogisticRegression` model on a deterministic
  80/20 train/test split.
- Reports accuracy for both the training and held-out test partitions.

## Repository layout

| Path | Purpose |
| --- | --- |
| `titanic_survival_prediction.ipynb` | Main notebook containing the full workflow. |
| `sample_data/train.csv` | Labeled training data used by the notebook. |
| `sample_data/test.csv` | Unlabeled Titanic test data included for future inference work. |
| `sample_data/gender_submission.csv` | Example Kaggle submission-format file. |
| `README.md` | Project documentation. |

## Dataset

The notebook reads `sample_data/train.csv`. Each row represents one passenger;
the training file contains 891 rows and 12 columns. `Survived` is the binary
target:

| Value | Meaning |
| --- | --- |
| `0` | Did not survive. |
| `1` | Survived. |

### Source columns

| Column | Description | Used by the baseline model? |
| --- | --- | --- |
| `PassengerId` | Passenger identifier. | No |
| `Survived` | Survival label (training data only). | Target |
| `Pclass` | Ticket class (`1`, `2`, or `3`). | Yes |
| `Name` | Passenger name. | No |
| `Sex` | Passenger sex. | Yes |
| `Age` | Age in years. | Yes |
| `SibSp` | Number of siblings/spouses aboard. | Yes |
| `Parch` | Number of parents/children aboard. | Yes |
| `Ticket` | Ticket number. | No |
| `Fare` | Passenger fare. | Yes |
| `Cabin` | Cabin number. | No; dropped during cleaning |
| `Embarked` | Port of embarkation (`S`, `C`, or `Q`). | Yes |

The original data has missing values in `Age`, `Cabin`, and `Embarked`.
`Cabin` is removed because it is sparsely populated; `Age` is imputed with its
column mean; and `Embarked` is imputed with its most frequent value.

## Methodology

The notebook follows these steps:

1. **Load and inspect data** — display sample rows, dimensions, data types,
   summary statistics, and null counts.
2. **Clean missing data** — drop `Cabin`, fill `Age` with its mean, and fill
   `Embarked` with its mode.
3. **Explore the data** — generate Seaborn count plots for survival, sex,
   passenger class, and survival broken down by sex and class.
4. **Encode categories** — map `Sex` from `male`/`female` to `0`/`1`, and map
   `Embarked` from `S`/`C`/`Q` to `0`/`1`/`2`.
5. **Select model inputs** — use `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`,
   `Fare`, and `Embarked`; exclude identifier, name, ticket, and target
   columns.
6. **Split the data** — reserve 20% of records for testing with
   `random_state=2` (712 training rows and 179 test rows).
7. **Train and evaluate** — fit `LogisticRegression(max_iter=1000)` and
   calculate accuracy on both partitions.

## Quick start

### Prerequisites

- Python 3.9 or newer is recommended.
- Jupyter Notebook or JupyterLab.
- The Python packages listed below.

### 1. Clone and enter the repository

```bash
git clone <your-fork-or-repository-url>
cd Titanic-Survival-Prediction
```

### 2. Create an isolated environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\\Scripts\\activate           # Windows PowerShell
python -m pip install --upgrade pip
```

### 3. Install dependencies

```bash
python -m pip install jupyter pandas numpy matplotlib seaborn scikit-learn
```

### 4. Run the notebook

Start Jupyter from the repository root so that the relative data path
`./sample_data/train.csv` resolves correctly:

```bash
jupyter notebook titanic_survival_prediction.ipynb
```

Open the notebook in your browser and choose **Run All Cells**. JupyterLab is
also supported:

```bash
jupyter lab
```

## Baseline results

With the checked-in data, the notebook's fixed split (`test_size=0.2`,
`random_state=2`) reports the following accuracies:

| Partition | Accuracy |
| --- | ---: |
| Training | 0.8090 (80.90%) |
| Test | 0.7821 (78.21%) |

These values are useful as a baseline for the current notebook, not as a
generalization guarantee. Accuracy can change if the data, split, features,
library versions, or preprocessing choices change.

## Using the supplied test data

`sample_data/test.csv` does not include the `Survived` label, so it is not used
by the current evaluation cells. To create predictions for it, apply **the same
preprocessing** used for training before calling `model.predict(...)`:

```python
test = pd.read_csv("sample_data/test.csv")
test = test.drop(columns="Cabin")
test["Age"] = test["Age"].fillna(titanic_data["Age"].mean())
test["Fare"] = test["Fare"].fillna(titanic_data["Fare"].median())
test["Embarked"] = test["Embarked"].fillna("S")
test = test.replace({
    "Sex": {"male": 0, "female": 1},
    "Embarked": {"S": 0, "C": 1, "Q": 2},
})

test_features = test[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
predictions = model.predict(test_features)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
submission.to_csv("submission.csv", index=False)
```

The resulting `submission.csv` has the same two-column structure as
`sample_data/gender_submission.csv`. Unlike the training data, the supplied
test data includes a missing fare, so the example fills it with the training
fare median before prediction. The notebook's training data has no missing
fares.

## Limitations and next steps

- The baseline uses a single random hold-out split; cross-validation would give
  a more robust estimate of performance.
- The numeric category encoding for `Embarked` can imply an artificial order.
  One-hot encoding is a stronger default for linear models.
- Feature engineering (for example, titles from names, family size, ticket
  groups, or deck extracted from cabin) may improve the model.
- Scaling numeric features and comparing alternative classifiers can provide a
  fairer model-selection experiment.
- Package versions are not pinned. Add a `requirements.txt` or lockfile before
  relying on exact environment reproducibility.

## License

No license file is currently included. Add an explicit license before reusing
or distributing this project beyond the terms that apply to its data source.
