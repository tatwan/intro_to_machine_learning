# Lab: four models, one split

In this lab, you will fit a logistic regression, a decision tree, a random forest and a gradient-boosted model to the same credit-default split, score them with the metrics that fit the question, and recommend one to the bank's risk team.

**Time:** 35 minutes · **Data:** `data/credit_default.csv` · **Starter:** `Unsolved/model_comparison.ipynb`

## The situation

The risk team reviews clients who might default next month. One row is one credit-card client (see the 5.2 README for the columns); `default` is 1 when the client defaulted. The team asks: "Which model ranks the risky clients best, and how many defaulters would it find if we review the top 10%?"

The starter loads the data and splits it 80 / 20 with `stratify=y, random_state=42`, the same split as 5.2. It defines `scores(model)`, which returns accuracy, ROC AUC, average precision, and precision, recall and F1 at the default 0.5 threshold on the test rows.

Work through the checkpoints in order. C1 is the minimum, C2 is today's target, C3 is for those who finish C2.

## C1 · Two baselines

1. Score a `DummyClassifier(strategy="most_frequent")`.

2. Fit `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))` and score it.

3. Fit `DecisionTreeClassifier(max_depth=5, random_state=42)` and score it.

## C2 · Two ensembles

4. Fit `RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)` and score it. It is the slowest of the four to fit.

5. Fit `HistGradientBoostingClassifier(random_state=42)` and score it. Print `n_iter_`: early stopping is on by default when there are more than 10,000 rows.

6. Put the five rows in one table, sorted by ROC AUC.

7. The team asks: "Which model ranks the risky clients best?"

    Translate: the highest ROC AUC and average precision. Then say why the accuracy column could not have answered the question.

## C3 · The review list

8. For the boosted model and the logistic regression, take the 600 test clients with the highest predicted probability (10% of 6,000) and count how many defaulted.

9. Refit both with `class_weight="balanced"`. Print precision, recall and ROC AUC again. What changed, and what did not?

10. Write three sentences for the risk team: the model you recommend, its test ROC AUC, and how many defaulters it puts in a review list of 600.

## Check yourself

| Model | Accuracy | ROC AUC | Avg precision | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Dummy | 0.779 | 0.5 | 0.221 | 0 | 0 | 0 |
| Logistic | 0.808 | 0.708 | 0.494 | 0.687 | 0.24 | 0.355 |
| Tree, depth 5 | 0.817 | 0.742 | 0.488 | 0.661 | 0.356 | 0.463 |
| Random forest | 0.814 | 0.755 | 0.536 | 0.637 | 0.365 | 0.464 |
| Boosted | 0.817 | 0.781 | 0.556 | 0.656 | 0.363 | 0.468 |

| Step | Value to check |
|---|---|
| 1 | 24,000 training and 6,000 test clients; 1,327 test defaulters (base rate `0.221`) |
| 5 | `n_iter_` 72 of a maximum of 100 |
| 7 | Accuracy spans 0.808 to 0.817 across the four models; ROC AUC spans 0.708 to 0.781 |
| 8 | Top 600: boosted 419 defaulters (`0.698`), logistic 401 (`0.668`); 133 expected at random |
| 9 | Balanced logistic: precision `0.367`, recall `0.62`, AUC `0.708`. Balanced boosted: precision `0.477`, recall `0.623`, AUC `0.776` |

## Hint

For step 8: `top = np.argsort(-proba)[:600]`, then `y_test.values[top].sum()`. For step 9, `class_weight` moves the 0.5 line; the ranking, and so the AUC, stays almost where it was.

## Bonus

Fit `RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)`. Does limiting the depth of each tree help the forest here? Check: test AUC `0.773` against `0.755`.

## Known leaks and quirks in this dataset

- `EDUCATION` codes 0, 5 and 6 and `MARRIAGE` code 0 are not in the documentation (14, 280, 51 and 54 clients).
- `PAY_0` to `PAY_6` use -2 and 0, which the documentation does not define; -1 means paid on time.
- The September repayment column is called `PAY_0`; there is no `PAY_1`.
- All columns are known before the October outcome; there is no target leak. A default rate of 22.1% means accuracy flatters every model.

## More practice

Not checked, not submitted.

- Run the five models on the telco churn split from 4.3 (use D01's `pre` in front of each). Does boosting win there too?
- Read the scikit-learn example "Categorical feature support in gradient boosting", convert `SEX`, `EDUCATION` and `MARRIAGE` with `.astype("category")`, and fit with `categorical_features="from_dtype"`.

## References

Yeh and Lien, Default of Credit Card Clients, UCI Machine Learning Repository dataset 350, https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients, CC BY 4.0. Numbers from `D02/reference/10_comparison.py` and `08_ensembles.py`.
