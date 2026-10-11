# Fit a tree, plot it, change its depth

In this activity, you will fit a decision tree to credit-card clients, draw it, and watch training and test scores move apart as you let it grow.

**Time:** 10 minutes · **Data:** `data/credit_default.csv` · **Starter:** `Unsolved/tree_depth.ipynb`

## The situation

A bank in Taiwan recorded 30,000 credit-card clients in 2005. One row is one client: the credit limit, sex, education, marriage, age, the repayment status for each month from April to September (`PAY_0` is September, `PAY_2` August, and so on), six monthly bill amounts and six payments. The target `default` is 1 when the client defaulted the next month. The starter loads the file with readable column names and splits it 80 / 20 with `stratify=y, random_state=42`.

We do this together: type each cell with the instructor and stop at each checkpoint.

## Instructions

1. Run the first cell. Note the training and test rows and the dummy accuracy.

2. Fit `DecisionTreeClassifier(max_depth=2, random_state=42)` and draw it with `plot_tree(tree, feature_names=X.columns, filled=True)`. Read the root: which column, which threshold?

3. Print training and test accuracy, and the test ROC AUC from `predict_proba(X_test)[:, 1]`.

4. Change `max_depth` to 5, then remove it (`max_depth=None`). Print the same three numbers each time, plus `tree.get_n_leaves()` and `tree.get_depth()`.

## Check yourself

| Step | Value to check |
|---|---|
| 1 | 24,000 training and 6,000 test clients; dummy accuracy `0.779` |
| 2 | Root: `PAY_0 <= 1.5` (paid on time or at most one month late in September) |
| 3 | Depth 2: train accuracy `0.820`, test `0.817`, test AUC `0.690` |
| 4 | Depth 5: train `0.825`, test `0.817`, AUC `0.742`, 31 leaves. No limit: train `0.999`, test `0.715`, AUC `0.607`, 3,791 leaves, depth 41 |

Accuracy hardly moves between depth 2 and depth 5 because 77.9% of clients do not default; the AUC shows the deeper tree ranks clients better. With no limit the tree memorizes the training clients: almost perfect on them, worse than the dummy on new ones.

## Hint

`plot_tree` draws a depth-5 tree too small to read. Use `max_depth=2` for the picture and `export_text(tree, feature_names=list(X.columns))` for anything deeper.

## Bonus

Wrap the depth-5 tree in `make_pipeline(StandardScaler(), ...)`. Does the test accuracy change? Why would a tree not care about the units of a column?

## Known leaks and quirks in this dataset

- `EDUCATION` holds codes 0, 5 and 6 that the documentation does not define (14, 280 and 51 clients); `MARRIAGE` holds 0 for 54 clients.
- `PAY_0` to `PAY_6` hold -2 and 0, which the documentation does not define either; -1 means paid on time and 1 to 8 mean months late. Treat -2, -1 and 0 as "no delay".
- There is no `PAY_1`: the September column is called `PAY_0` in the original file. The loader keeps that name.
- Every column describes April to September 2005; the target is October. No column is recorded after the outcome.

## References

Yeh and Lien, Default of Credit Card Clients, UCI Machine Learning Repository dataset 350, https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients, CC BY 4.0. Numbers from `D02/reference/07_trees.py`.
