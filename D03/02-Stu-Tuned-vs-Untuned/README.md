# Tuned against untuned: is the search worth it?

In this activity, you will tune a gradient-boosted model with a random search and a decision tree with a grid search on the bank-marketing table, compare each with its defaults, and say for which model the search paid.

**Time:** 20 minutes · **Data:** `data/bank_marketing.csv` · **Starter:** `Unsolved/tuning.ipynb`

## The situation

One row is one client the bank called during a deposit campaign; `y` is `yes` when the client subscribed. The starter applies D01 5.5's decision (`duration` is dropped: it is known only after the call), splits 80 / 20 with `stratify=y, random_state=42` (36,168 training and 9,043 test clients, 11.7% subscribers), and defines `pre`, a `ColumnTransformer` that one-hot encodes the text columns and passes the numbers through.

The campaign lead: "A colleague spent a day tuning last year's model. Was that a good use of a day?"

## Instructions

1. Create three folds: `cv = StratifiedKFold(3, shuffle=True, random_state=42)`. Three folds keep the search short on the course VM; the demonstration used five.

2. Score the untuned boosted model: `cross_val_score(make_pipeline(pre, HistGradientBoostingClassifier(random_state=42)), X_train, y_train, cv=cv, scoring="roc_auc").mean()`.

3. Run a random search with 15 candidates over four settings of the same model:

    ```python
    P = "histgradientboostingclassifier__"
    dist = {P + "learning_rate": loguniform(0.01, 0.5), P + "max_leaf_nodes": randint(7, 128),
            P + "min_samples_leaf": randint(5, 200), P + "l2_regularization": loguniform(1e-3, 10)}
    search = RandomizedSearchCV(model, dist, n_iter=15, cv=cv, scoring="roc_auc",
                                random_state=42, n_jobs=-1).fit(X_train, y_train)
    ```

    Print `best_params_` and `best_score_`. How many models did the search fit?

4. Sort `pd.DataFrame(search.cv_results_)` by `rank_test_score` and look at the best and worst `mean_test_score`. How wide is the range?

5. Score the test clients once with each model: the defaults fitted on all training rows, and `search.best_estimator_`.

6. Do the same for `DecisionTreeClassifier(random_state=42)`: its 5-fold CV score with the defaults, then a `GridSearchCV` over `max_depth` in `[3, 4, 5, 6, 8, 10]` and `min_samples_leaf` in `[20, 50, 100, 200]`.

7. The campaign lead asks: "Was the day of tuning worth it?"

    Translate: for each model, the gain from tuning in validation ROC AUC, set against the spread across folds. Answer in two sentences, one per model, with the numbers.

## Check yourself

| Step | Value to check |
|---|---|
| 2 | Untuned boosted, 3-fold CV ROC AUC `0.795` |
| 3 | 45 fits. Best: `learning_rate` 0.046, `max_leaf_nodes` 56, `min_samples_leaf` 108, `l2_regularization` 0.005; best CV ROC AUC `0.796` |
| 4 | Best to worst mean score: 0.796 to 0.771 |
| 5 | Test ROC AUC: untuned `0.803`, tuned `0.804` (a gain of 0.001) |
| 6 | Tree defaults, 5-fold: `0.615 ± 0.012` (training AUC 1). Tuned: `max_depth` 10, `min_samples_leaf` 200, CV `0.749`, test `0.759` |
| 7 | Boosting gained 0.001, inside its fold-to-fold spread (0.003 in the 5-fold demonstration). The tree gained 0.134: its defaults let it grow until every leaf is pure |

## Hint

A setting's name inside a pipeline is the step name, two underscores, then the setting: `histgradientboostingclassifier__learning_rate`. `make_pipeline` names each step after its class, in lower case.

## Bonus

The tree's best `max_depth` (10) and `min_samples_leaf` (200) are both at the edge of the grid. What does that tell you to do next? Widen the grid past both edges and run it again.

## Known leaks and quirks in this dataset

- **`duration` is the call length, known only after the call ends.** It predicts the outcome very well and must be dropped (D01 5.5). The starter drops it.
- `pdays = -1` means "never contacted before" in this copy (bank-full); it is a code, not a number of days.
- `unknown` is a category of its own in `job`, `education`, `contact` and `poutcome`.
- 11.7% of clients subscribed: compare models with ROC AUC or average precision, not accuracy.
- `day_of_week` holds the day of the month, despite its name.

## References

UCI Bank Marketing (bank-full), https://archive.ics.uci.edu/dataset/222/bank+marketing, CC BY 4.0. Numbers from `D03/reference/04_tuning.py`.
