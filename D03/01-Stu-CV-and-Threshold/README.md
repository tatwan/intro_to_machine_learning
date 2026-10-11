# Model comparison with cross-validation, then the threshold

In this activity, you will compare four churn models on the same five validation folds, pick one from the mean and the spread, and then choose its threshold from the retention manager's costs.

**Time:** 25 minutes · **Data:** `data/telco_churn.csv` · **Starter:** `Unsolved/cv_threshold.ipynb`

## The situation

One row is one telco customer; `Churn` is `Yes` when the customer left in the last month. The starter repeats D01 5.6 and D02: `TotalCharges` converted to numbers, the `ColumnTransformer` `pre` (median and scaling for numbers, one-hot for text), a second one `tree_pre` with no scaling for the tree models, and the 80 / 20 split with `stratify=y, random_state=42`. Your 5,634 training customers and 1,409 test customers are the same as yesterday's.

The retention manager: "Each customer you flag gets a retention offer, which costs us 100 dollars whether they would have left or not. A customer who leaves without an offer costs us 400 dollars. Which model, and how many customers should we flag?"

## Instructions

1. Create the folds once and reuse them for every model: `cv = StratifiedKFold(5, shuffle=True, random_state=42)`.

2. For each of the four models below, run `cross_validate(model, X_train, y_train, cv=cv, scoring=["roc_auc", "average_precision", "neg_brier_score"], return_train_score=True)`. Every model sits inside its own pipeline, so every fold fits its own preprocessing.

    - `make_pipeline(pre, DummyClassifier(strategy="prior"))`
    - `make_pipeline(pre, LogisticRegression(max_iter=1000))`
    - `make_pipeline(tree_pre, RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))`
    - `make_pipeline(tree_pre, HistGradientBoostingClassifier(random_state=42))`

3. Put the results in one table: mean and standard deviation of the validation ROC AUC, mean average precision, mean Brier score (minus `test_neg_brier_score`), and the mean training ROC AUC.

4. The manager asks: "Which model ranks customers best?"

    Translate: the highest mean validation ROC AUC. Is the gap between the top two models larger than one standard deviation? Which model has the largest gap between training and validation AUC, and what does that gap mean (D02 3.1)?

5. Write the cost of a set of predictions as a function: `cost = 100 * FP + 400 * FN + 100 * TP`, from `confusion_matrix(y_true, y_pred).ravel()`. Wrap minus the mean cost per customer in `make_scorer`.

6. Choose the threshold for your model on the folds: `TunedThresholdClassifierCV(model, scoring=scorer, cv=cv, random_state=42).fit(X_train, y_train)`. Print `best_threshold_` and minus `best_score_` (the cost per customer on the folds).

7. Only now, score the test customers once: the total cost at the 0.5 default (`model.predict`) and at your threshold (`tuned.predict`), and how many customers each flags.

8. The manager asks: "How many customers should we flag, and what does that save?"

    Translate: the number flagged at the tuned threshold, and the difference in total cost against the default. Answer in two sentences with the numbers.

## Check yourself

| Model | Validation ROC AUC | Avg precision | Brier | Training ROC AUC |
|---|---:|---:|---:|---:|
| Dummy | 0.500 ± 0.000 | 0.265 | 0.195 | 0.5 |
| Logistic | 0.846 ± 0.013 | 0.661 | 0.135 | 0.85 |
| Random forest | 0.821 ± 0.012 | 0.613 | 0.146 | 1 |
| Boosted | 0.838 ± 0.007 | 0.65 | 0.14 | 0.963 |

| Step | Value to check |
|---|---|
| 4 | Logistic leads by 0.008, less than its own standard deviation of 0.013. The forest scores 1 on its training folds and 0.821 on validation: it memorized the training rows |
| 6 | Logistic: threshold `0.278`, cost per customer on the folds `61.68` |
| 7 | Default 0.5: 318 flagged, total cost 97,800. Tuned: 564 flagged, total cost 89,200 |
| 8 | Flag 564 of 1,409 test customers; the tuned threshold saves 8,600 dollars against the default |

If you chose the boosted model, its tuned threshold is `0.209` and its test cost 88,600. That is 600 dollars below the logistic model on these 1,409 customers. On the folds the logistic model was cheaper (61.68 against 62.68 per customer), so a 600-dollar gap on one test set is not a reason to change the choice.

A different value in the third decimal usually means different folds. Check `shuffle=True, random_state=42` and that you reuse the same `cv` object.

## Hint

`cross_validate` returns `test_*` keys for the validation folds; it calls them "test" even though they come from the training rows. For step 5, `make_scorer(gain)`, where `gain` returns minus the cost per customer, because the tuner maximizes its score.

## Bonus

Check the break-even rule: with these costs, an offer pays when the probability of churn is above 100 / 400 = 0.25. The tuned threshold of 0.278 is close to it because logistic regression gives fairly honest probabilities. Fit the same tuner on `LogisticRegression(max_iter=1000, class_weight="balanced")`: where does its threshold land, and why?

## Known leaks and quirks in this dataset

- `TotalCharges` is read as text: 11 cells hold a single space, all for customers with `tenure` 0.
- `customerID` is an identifier, never a feature.
- `tenure`, `MonthlyCharges` and `TotalCharges` are near-collinear (`tenure` and `TotalCharges` correlate at 0.829).
- 26.5% of customers churned: a model that always says "stays" is 0.735 accurate and finds nobody.
- Choosing the threshold by looking at the test rows is a leak. Choose it on the folds, then score the test rows once.

## References

IBM Telco Customer Churn, https://github.com/IBM/telco-customer-churn-on-icp4d, IBM sample data for educational use. Course copy taken 2026-10-02. Numbers from `D03/reference/03_compare_cv.py`. The 100- and 400-dollar costs are the course's example business case, not IBM data.
