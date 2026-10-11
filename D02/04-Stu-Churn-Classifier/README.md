# Build a churn classifier and evaluate it with the right metric

In this activity, you will fit a logistic regression to telco churn, read its confusion matrix cell by cell, and decide which of two models the retention manager should use, with the metric that fits her decision.

**Time:** 25 minutes · **Data:** `data/telco_churn.csv` · **Starter:** `Unsolved/churn_classifier.ipynb`

## The situation

One row is one telco customer; `Churn` is `Yes` when the customer left in the last month. The starter repeats D01 5.6: `TotalCharges` converted to numbers, the `ColumnTransformer` `pre`, and the 80 / 20 split with `stratify=y, random_state=42`, so your test customers are the same 1,409 as in the 4.2 demonstration.

The retention manager: "My team calls the customers you flag and offers a discount. A call costs us little. A customer who leaves costs us a year of bills."

## Instructions

1. Fit `make_pipeline(pre, LogisticRegression(max_iter=1000))`. Print the test accuracy and the accuracy of `DummyClassifier(strategy="most_frequent")`.

2. Print `confusion_matrix(y_test, model.predict(X_test))`. Write down what each of the four cells counts, in words: "customers who stayed and whom we did not flag", and so on.

3. Print `classification_report(y_test, pred, digits=3)`. Find precision, recall and F1 for the churn class (`1`), and check precision and recall by hand from your four cells.

4. Compute `roc_auc_score` and `average_precision_score` from `model.predict_proba(X_test)[:, 1]`. Compare the average precision with the share of churners in the test rows.

5. Fit a second model with `LogisticRegression(max_iter=1000, class_weight="balanced")`. Repeat steps 2 to 4 for it.

6. The manager asks: "Which model should my team use?"

    Translate: which of her two costs is larger, a wasted call (a false positive) or a missed churner (a false negative)? Which metric counts the misses? Answer in two sentences with the numbers from both models.

## Check yourself

| Step | Value to check |
|---|---|
| 1 | Accuracy `0.806`; dummy `0.735` |
| 2 | TN 926, FP 109, FN 165, TP 209 (rows: actual stayed, actual churned; columns: predicted stay, predicted churn) |
| 3 | Churn class: precision `0.657`, recall `0.559`, F1 `0.604` |
| 4 | ROC AUC `0.842`; average precision `0.633` against a base rate of `0.265` |
| 5 | Balanced: TN 747, FP 288, FN 81, TP 293; precision `0.504`, recall `0.783`, F1 `0.614`; ROC AUC `0.841` |
| 6 | The balanced model finds 293 of the 374 churners instead of 209 (84 more), at the cost of 179 more calls to customers who would have stayed |

ROC AUC barely moves between the two models (0.842 and 0.841): `class_weight` changes where the model draws the line, not how well it ranks customers.

## Hint

`confusion_matrix` puts the true class in rows and the predicted class in columns, both in the order 0, 1. Precision reads down the "predicted churn" column; recall reads across the "actually churned" row.

## Bonus

Plot precision and recall against the threshold: `p = model.predict_proba(X_test)[:, 1]`, then for `t` from 0.05 to 0.95 compute both with `(p >= t)`. At which threshold do the two curves cross? This picture is for understanding only: choosing a threshold by looking at the test rows is a leak. D03 2.2 chooses it on validation folds with a cost per error.

## Known leaks and quirks in this dataset

- `TotalCharges` is read as text: 11 cells hold a single space, all for customers with `tenure` 0.
- `customerID` is an identifier, never a feature.
- `tenure`, `MonthlyCharges` and `TotalCharges` are near-collinear, so their coefficients can take opposite signs; read them together, not one at a time.
- 26.5% of customers churned: accuracy rewards a model that predicts "stays" for everyone (0.735).

## References

IBM Telco Customer Churn, https://github.com/IBM/telco-customer-churn-on-icp4d, IBM sample data for educational use. Course copy taken 2026-10-02. Numbers from `D02/reference/06_class_metrics.py`.
