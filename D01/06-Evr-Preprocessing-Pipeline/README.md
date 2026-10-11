# Build the preprocessing pipeline

In this activity, you will fix the telco table's `TotalCharges` column, then put every preparation step (impute, scale, encode) inside one `ColumnTransformer` that learns only from the training rows.

**Time:** 10 minutes · **Data:** `data/telco_churn.csv` · **Starter:** `Unsolved/preprocessing.ipynb`

One row is one customer. `Churn` says whether they left in the last month.

## Instructions

1. Load the table and look at the type of `TotalCharges`.

    ```python
    import pandas as pd
    df = pd.read_csv("data/telco_churn.csv")
    df["TotalCharges"].dtype
    ```

    pandas reads it as text (`str`): 11 rows hold a single space instead of a number.

2. Turn it into numbers. The 11 blanks become `NaN`, which the pipeline will fill.

    ```python
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].isna().sum()
    ```

3. Separate the target, drop the identifier, and list the two kinds of column.

    ```python
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["customerID", "Churn"])
    num_cols = list(X.select_dtypes(include="number").columns)
    cat_cols = list(X.select_dtypes(include=["str", "object"]).columns)
    ```

4. Split before anything is fitted.

    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    ```

5. Build the `ColumnTransformer`: numbers get the median and a scale, text gets one-hot encoding. Fit it on the training rows and look at what comes out.

    ```python
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    pre.set_output(transform="pandas")
    Xt = pre.fit_transform(X_train)
    Xt.shape, Xt.isna().sum().sum()
    ```

6. Put a model at the end and score it against a dummy.

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    model = make_pipeline(pre, LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    print(model.score(X_test, y_test))
    print(DummyClassifier().fit(X_train, y_train).score(X_test, y_test))
    ```

## Check yourself

| Step | Value to check |
|---|---|
| 1 | `str` |
| 2 | `11` |
| 3 | 4 numeric columns (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`), 15 text columns |
| 4 | 5,634 training rows (8 of them with `TotalCharges` missing), 1,409 test rows |
| 5 | `(5634, 45)` and `0` missing values |
| 6 | Model accuracy `0.806`; dummy `0.735` |

The median the imputer learned from the training rows is `1398.1`. The median over all rows is `1397.5`: close here, but only the first one is allowed.

The 11 blank `TotalCharges` belong to customers with tenure 0, who have not been billed yet, so the true value is 0. Filling them with 0 instead of the median gives the same accuracy, `0.806`; prefer 0 because it is what the blank means.

## Hint

If step 5 fails with "Cannot use median strategy with non-numeric data", step 2 did not run: `TotalCharges` is still text.

## Bonus

Change one test row's `PaymentMethod` to `"Crypto wallet"` and predict it. Then remove `handle_unknown="ignore"` and try again. What does the error tell you about new categories in production?

## Known leaks and quirks in this dataset

- `TotalCharges` is text with 11 blank values; the 11 rows are customers with `tenure` 0, who have not been billed yet.
- `customerID` is an identifier, never a feature.
- `TotalCharges` is close to `tenure` times `MonthlyCharges` (correlation with `tenure` 0.83).
- `SeniorCitizen` is coded 0 / 1 while every other yes / no column uses the words `Yes` / `No`.
- `OnlineSecurity` and five other service columns have a third value, `No internet service`, for the 1,526 customers without internet.

## References

IBM Telco Customer Churn, https://github.com/IBM/telco-customer-churn-on-icp4d, IBM sample data for educational use. Course copy taken 2026-10-02 (sha256 starts `a0ea74c48c1063be`). Numbers from `D01/reference/06_telco_preprocess.py` and `03_telco_quality.py`.
