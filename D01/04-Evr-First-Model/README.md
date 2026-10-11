# Your first model, end to end

In this activity, you will run the whole path from a CSV file to one prediction on your own VM, compare the model with a baseline that knows nothing, and then swap the model for three others while everything else stays the same.

**Time:** 15 minutes · **Data:** `data/penguins.csv` · **Starter:** `Unsolved/first_model.ipynb`

One row is one penguin. The question: given four body measurements, which of three species is it?

## Instructions

1. Load the table and keep four measurement columns as `X` and the species as `y`.

    ```python
    import pandas as pd
    df = pd.read_csv("data/penguins.csv")
    features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    X, y = df[features], df["species"]
    ```

2. Split once, before anything learns from the data. A quarter of the rows go to the test set; `stratify=y` keeps the same share of each species in both parts.

    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    ```

3. Build the pipeline: fill missing values with the median, scale, then classify. Fit it on the training rows only.

    ```python
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    ```

4. Score it on the test rows, and score a model that always answers the most common species.

    ```python
    from sklearn.metrics import accuracy_score
    from sklearn.dummy import DummyClassifier
    print(accuracy_score(y_test, model.predict(X_test)))
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    print(accuracy_score(y_test, dummy.predict(X_test)))
    ```

5. Ask the model about a penguin it has never seen: bill 47.0 mm long and 15.0 mm deep, flipper 215 mm, mass 5,200 g.

    ```python
    new = pd.DataFrame([{"bill_length_mm": 47.0, "bill_depth_mm": 15.0,
                         "flipper_length_mm": 215.0, "body_mass_g": 5200.0}])
    print(model.predict(new), model.predict_proba(new).round(3))
    ```

6. Swap the last step of the pipeline for `DecisionTreeClassifier(random_state=42)`, then `RandomForestClassifier(random_state=42)`, then `KNeighborsClassifier()`. Keep the imputer, the scaler and the split exactly as they are. Write down each test accuracy.

## Check yourself

| Step | Value to check |
|---|---|
| 2 | 258 training rows, 86 test rows |
| 4 | Logistic regression test accuracy `1.0` (86 of 86); dummy `0.442` |
| 5 | `Gentoo`, with probability `0.996` |
| 6 | Decision tree `0.988`, random forest `0.988`, k-nearest neighbors `0.988` |

A perfect score on 86 penguins is believable here: the three species differ in body shape, and the dummy shows how far from luck that is. On business data, a perfect score is a reason to look for a leak (this afternoon).

## Hint

To swap the model, change only the last argument of `make_pipeline(...)` and import the new class. Everything before it stays.

## Bonus

Add `island` as a fifth feature. What error do you get, and why? (5.6 solves it.)

## Known quirks in the data

2 rows have no measurements; with this split both land in the training rows, and the imputer fills them with the training medians. `sex` is missing in 11 rows and is not used here.

## References

Palmer Penguins, https://github.com/allisonhorst/palmerpenguins, CC0. Course copy taken 2026-10-02. Numbers from `D01/reference/02_penguins_first_model.py`.
