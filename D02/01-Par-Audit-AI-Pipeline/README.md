# Audit the AI-written pipeline

In this activity, you and a partner will review a churn model a chat assistant wrote, find the four faults that make its number untrustworthy, and measure what each one hides.

**Time:** 20 minutes · **Data:** `data/telco_churn.csv` · **Starter:** `Unsolved/audit.ipynb`

## The situation

A colleague asked a chat assistant for "a churn model in scikit-learn" and pasted the answer into a notebook. It runs without an error and prints `Accuracy: 0.82`. The colleague wants to send that number to the retention manager this afternoon. You are the reviewer.

One row is one telco customer. `Churn` is `Yes` when the customer left in the last month.

```python
df = pd.read_csv("data/telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
y = (df["Churn"] == "Yes").astype(int)
X = pd.get_dummies(df.drop(columns=["customerID", "Churn"]))
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
```

## Instructions

Work in pairs: one person types, the other reads the code and asks "which rows did that line learn from?". Swap after step 3.

1. Read the code before you run it. For every line, write whether it **learns** something from the data (a mean, a list of columns, weights) and, if so, from which rows: all rows, or training rows only.

2. Run the starter. Note the accuracy and the number of test customers.

3. **Finding 1.** Your partner asks: "Did anything learn from the test customers?"

    Translate: compare the mean of `tenure` over all rows with its mean over the training rows only. Then rewrite the code so the scaler sits inside a `make_pipeline` that is fitted on `X_train` only. Does the accuracy move?

4. **Finding 2.** The retention manager asks: "Is 0.82 good?"

    Translate: score a `DummyClassifier(strategy="most_frequent")` on the same split. Then count how many of the test customers who really churned the model flagged: `((pred == 1) & (y_test == 1)).sum()` against `y_test.sum()`.

5. **Finding 3.** The manager asks: "Who decided which customers count as 'will churn'?"

    Translate: `predict()` flags a customer when the predicted probability is above 0.5. Print how many test customers are flagged at 0.5, then use `predict_proba(X_test)[:, 1] >= 0.3` and count again: how many flagged, how many of them really churned?

6. **Finding 4.** Your partner asks: "Would another split give the same number?"

    Translate: run the corrected pipeline with `random_state` 0, 1, 2, 3 and 4 and print the five accuracies. Then take the course pipeline from D01 5.6 (the starter builds it, with its stratified 80 / 20 split) and score it with 5-fold cross-validation on its training rows: `cross_val_score(pipe, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42))`. Print the mean and the standard deviation.

7. Write the review comment for your colleague: four lines, each naming the code line, what it hides, and the fix.

## Check yourself

| Step | Value to check |
|---|---|
| 2 | Accuracy `0.82` on 1,409 test customers; 45 columns after `get_dummies` |
| 3 | `tenure` mean 32.37 on all rows and 32.37 on the training rows; accuracy still `0.82` once the scaler is fitted on the training rows only |
| 4 | Dummy accuracy `0.735`. The model flagged 223 of the 373 customers who churned (373, not D01's 374: this split is not stratified) |
| 5 | 326 flagged at 0.5; at 0.3, 540 flagged and 299 of the 373 churners among them |
| 6 | Five accuracies `0.798, 0.812, 0.807, 0.817, 0.791` (range 0.791 to 0.817: a spread wider than the gap between 0.82 and 0.806); with the course pipeline from D01 and 5-fold cross-validation, accuracy `0.802` with a standard deviation of `0.012` |

A different count is information, not failure. Re-read the step, then check which rows each line was fitted on.

Step 3 surprises most people: fixing the leak did not change the score. That is why this leak survives in so much code. The habit matters because the same mistake with a target-based encoder or a feature selected on all rows does move the score.

## Hint

Every `fit` and every `fit_transform` learns. Find them first. `predict` uses a threshold you never wrote; `predict_proba` lets you choose it.

## Bonus

There is a fifth finding: `pd.get_dummies` decides the columns from all rows before the split. What happens next month when the file arrives with a payment method the training rows never had? (D01 5.6 answered this with `OneHotEncoder(handle_unknown="ignore")`.)

## Known leaks and quirks in this dataset

- `TotalCharges` is read as text: 11 cells hold a single space. They belong to customers with `tenure` 0, so 0 is the true value (D01 5.6).
- `customerID` is an identifier, never a feature.
- `tenure`, `MonthlyCharges` and `TotalCharges` are near-collinear: `TotalCharges` is roughly `tenure × MonthlyCharges`.
- No column is recorded after the churn month, so there is no target leak; the leaks in this activity come from the code, not the data.

## References

IBM Telco Customer Churn, https://github.com/IBM/telco-customer-churn-on-icp4d, IBM sample data for educational use. Course copy taken 2026-10-02. Numbers from `D02/reference/01_audit.py`.
