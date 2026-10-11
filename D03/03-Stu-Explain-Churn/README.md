# Explain the churn model to a retention manager

In this activity, you will explain the boosted churn model three ways: which columns it relies on, how the predicted risk changes with tenure and contract, and why it flags one customer. Then you will write the explanation for a manager who does not read code.

**Time:** 25 minutes · **Data:** `data/telco_churn.csv` · **Starter:** `Unsolved/explain_churn.ipynb`

## The situation

The starter fits `make_pipeline(tree_pre, HistGradientBoostingClassifier(random_state=42))` on the 5,634 training customers (test ROC AUC `0.835`). `tree_pre` imputes and one-hot encodes but does not scale, so every value you read is in months, dollars or categories.

The retention manager: "Before my team acts on this list, I want to know what the model looks at, and why it put customer 5194 on it. And should we push everyone onto two-year contracts?"

## Instructions

1. Compute permutation importance on the **test** customers: `permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42)`. Put `importances_mean` and `importances_std` in a table sorted from largest to smallest.

2. Which columns lose the most ROC AUC when shuffled? Which ones lose almost nothing?

3. Draw the partial dependence of churn on `tenure`, with ICE lines: `PartialDependenceDisplay.from_estimator(model, X_test_f, ["tenure"], kind="both", subsample=50, random_state=42)`, where `X_test_f = X_test.astype({"tenure": "float64"})`. Read the average curve at 0 and at 72 months.

4. Draw the partial dependence on `Contract` with `categorical_features=["Contract"]`. Read the three values.

5. Explain one customer with SHAP. Transform the test rows with the fitted preprocessing (`Xt = model[0].transform(X_test)`), build `explainer = shap.TreeExplainer(model[-1])` and `sv = explainer(Xt)`, then draw `shap.plots.waterfall(sv[i])` for the row of customer 5194 (`i = list(X_test.index).index(5194)`).

6. Draw `shap.plots.beeswarm(sv)`. Which color sits on the right for `tenure`, and what does that mean?

7. The manager asks: "What does the model look at, why customer 5194, and should we push everyone onto two-year contracts?"

    Translate: the top two columns from step 1; the three largest pushes from step 5; and whether a model that predicts churn tells you what happens when you change a customer's contract. Write four sentences a manager can read.

## Check yourself

| Step | Value to check |
|---|---|
| 1 | `Contract` `0.125 ± 0.008`, `tenure` `0.049 ± 0.006`, `InternetService` `0.016 ± 0.003` lead; `gender` `0.001 ± 0.001` |
| 3 | Average predicted churn `0.371` at tenure 0 and `0.074` at 72 months. Across all test customers, ICE values at tenure 0 range from 0.001 to 0.960 (your 50-line plot shows part of that range) |
| 4 | Month-to-month `0.29`, one year `0.121`, two year `0.03` |
| 5 | Customer 5194: tenure 22 months, month-to-month, fiber, 89.4 a month; probability `0.502`. Largest pushes: `Contract_Month-to-month` +0.88, `InternetService_Fiber optic` +0.33, `PaymentMethod_Electronic check` +0.30 (log-odds) |
| 6 | For `tenure`, low values (blue) sit on the right: short tenure pushes the prediction toward churn |

SHAP values for this model are in log-odds, the same units as D02 4.1's weighted sum `z`. The waterfall starts from the base value `-1.9697` and ends at the customer's `z`; the probability is the sigmoid of `z`.

## Hint

For step 7, look at the bars in step 1 again: the model leans on `Contract`, but nothing in this table shows what a customer does after their contract changes. That needs an experiment, such as offering a contract change to a random half of a customer group and comparing the two halves.

## Bonus

Refit the model without `TotalCharges` and compute permutation importance again. What happens to `tenure`, and why? Check: `0.049` with `TotalCharges`, `0.08` without; the two columns correlate at 0.829, so they shared the credit.

## Known leaks and quirks in this dataset

- `TotalCharges` is read as text: 11 cells hold a single space, all for customers with `tenure` 0.
- `customerID` is an identifier, never a feature.
- `tenure`, `MonthlyCharges` and `TotalCharges` are near-collinear: importance is shared between them, and dropping one moves credit to the others.
- Importance, PDP and SHAP describe the model, not the customers: a column the model relies on is not necessarily a cause of churn.

## References

IBM Telco Customer Churn, https://github.com/IBM/telco-customer-churn-on-icp4d, IBM sample data for educational use. Course copy taken 2026-10-02. Numbers from `D03/reference/05_explain.py`. SHAP 0.52.0 (`tools/requirements.txt`).
