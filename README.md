# AI Programming Foundations — Data Workflow Project

## Project Description

This project is a reusable, end-to-end data workflow built in a Jupyter notebook. It loads the Kaggle Titanic dataset, applies two reusable cleaning functions (missing-value handling and feature standardization), runs a structured exploratory data analysis function, and produces three labeled visualizations of the survival patterns in the data. It is designed as a foundation that could be extended into a machine learning or deep learning pipeline.

## What I Built

- Two data-cleaning functions (`handle_missing_values`, `standardize_features`) with docstrings.
- One exploratory analysis function (`explore_dataset`) that returns shape, dtypes, summary statistics, target-mean breakdowns, and a correlation matrix.
- Three labeled visualizations: survival rate by class and sex, age distribution by survival status, and a correlation heatmap.
- A written summary and interpretation section inside the notebook.

## Dataset

[Titanic — Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/c/titanic/data) — the `train.csv` file (891 rows, 12 columns). The CSV is stored at `data/titanic/train.csv` relative to the notebook.

## How to Run the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

To regenerate the file after adding packages:

```
pip freeze > requirements.txt
```

### 2. Open and run the notebook

```
jupyter notebook data_workflow.ipynb
```

In the Jupyter interface, choose **Cell → Run All** (or **Kernel → Restart & Run All**) to execute every cell top to bottom. The dataset path is relative to the notebook, so no configuration is required.

## Bias Awareness — Where Poor Data Cleaning Could Introduce Bias

Cleaning choices in this workflow could introduce or amplify bias in several ways. Imputing missing `Age` values with the within-group median by `Pclass` and `Sex` shrinks variance inside those groups and reinforces whatever stereotypes the group structure already encodes; a model trained on this data could learn a sharper sex/class age profile than truly exists. Converting `Cabin` to a binary `HasCabin` flag treats missingness as informative, but if the missingness actually reflects uneven record-keeping for lower-class passengers, the flag is partly a proxy for class and embeds that historical inequity into any downstream feature. Filling `Embarked` with the modal port silently overrides real but rare cases and could under-represent the experience of passengers from less common embarkation points. Dropping rows with missing values, an alternative I did not take, would have a similar risk because the rows with missing fields are not a random sample.

## Future Integration Reflections

### How this workflow would change for a machine learning pipeline

For an ML pipeline, the cleaning and feature-engineering steps would be wrapped in `scikit-learn` transformers (or a `Pipeline`/`ColumnTransformer`) so they can be fit on training data and reused on validation, test, and inference data without leakage. Imputation parameters such as group medians would be learned on the training split only and stored. Categorical features would be encoded numerically (one-hot or target-encoded), continuous features scaled, and a train/test split or cross-validation loop added before any model is trained. Evaluation would move from descriptive statistics to held-out metrics (accuracy, ROC AUC, calibration).

### Preparing this dataset for a neural network

A neural network is more sensitive to input scale and representation than a tree-based model, so additional preparation is needed. Continuous features (`Age`, `Fare`, `FamilySize`) should be standardized or min-max scaled, and skewed features such as `Fare` may benefit from a log transform. Categorical features should be one-hot encoded or passed through learned embeddings rather than left as pandas categoricals. The target would be reshaped into the format expected by the framework (a single sigmoid output for binary `Survived`). Because the dataset is small, regularization (dropout, weight decay) and early stopping would be more important than network depth.

### Agentic automation potential

Much of this workflow is a strong candidate for agentic automation. An agent could profile any incoming tabular dataset, propose cleaning steps (imputation strategy, columns to drop or convert to indicators), execute them, and then run a default EDA template and a set of visualizations — flagging anomalies (high missingness, suspicious correlations, class imbalance) for human review. The reusable, function-based structure of this notebook makes it a natural starting point: the agent's job becomes choosing *which* function to call and with what arguments, rather than writing the cleaning logic from scratch each time.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── data_workflow.ipynb
└── data/
    └── titanic/
        ├── train.csv
        ├── test.csv
        └── gender_submission.csv
```

## Git Workflow

This repository uses a main branch plus at least one development branch, with regular commits as each task was completed. See `git log --all --oneline` for the full history.
