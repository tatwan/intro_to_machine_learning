# Regularization: linear against Ridge against Lasso

In this activity, you will fit the same Ames pipeline three ways, plot a learning curve for each, and recommend one model to the agency with the numbers that justify it.

**Time:** 20 minutes · **Data:** `data/ames.csv` · **Starter:** `Unsolved/regularization.ipynb`

## The situation

The agency liked the step-2.2 model but asked why one coefficient is so large: `GarageQual` = `Po` (a poor-quality garage) cuts the predicted price by 39.2%, and only one training house has that value. The starter repeats the 2.2 setup: D01 6.4's decisions, the 75 / 25 split, `pre`, and a scorer `rmse_dollars` for errors in dollars. `cv = KFold(5, shuffle=True, random_state=42)`.

## Instructions

1. Refit the step-2.2 model (`make_pipeline(pre, LinearRegression())`) and print its test RMSE in dollars and the coefficient of `none__GarageQual_Po`. Find it with `pd.Series(model[-1].coef_, index=model[0].get_feature_names_out())`.

2. Replace `LinearRegression()` with `RidgeCV(alphas=np.logspace(-3, 3, 13), cv=cv)`. Print the chosen `alpha_`, the test RMSE and the same coefficient.

3. Replace it with `LassoCV(alphas=np.logspace(-5, -1, 13), cv=cv, max_iter=50000, random_state=42)`. Print `alpha_`, the test RMSE, the same coefficient, and how many of the coefficients are not zero. Count a coefficient as zero when its absolute value is below 1e-10: `(coef != 0).sum()` also counts two values near 1e-16, which are rounding leftovers.

4. The agency asks: "Is the plain model memorizing our houses?"

    Translate: `learning_curve` for each of the three models, on the training rows only, with `cv=cv`, `train_sizes=np.linspace(0.15, 1.0, 8)`, `scoring=rmse_dollars`, `shuffle=True, random_state=42`. Plot training and validation RMSE against the number of rows, one chart per model. Use `Ridge(alpha=...)` and `Lasso(alpha=...)` with the alphas from steps 2 and 3.

5. Read the three charts: for each model, the gap between validation and training RMSE at the largest size.

6. Recommend one model to the agency in two sentences: which one, the test RMSE, and one thing it does that the others do not.

## Check yourself

| Step | Value to check |
|---|---|
| 1 | Test RMSE 22,091; `GarageQual_Po` coefficient `-0.497` |
| 2 | `alpha_` 31.62; test RMSE 20,045; `GarageQual_Po` `-0.001` |
| 3 | `alpha_` 0.001; test RMSE 19,610; 75 of 283 coefficients not zero; `GarageQual_Po` `0.000` |
| 4 | Linear at 130 rows: training RMSE 0 (it fits every house exactly), validation 93,439 |
| 5 | Gap at 873 rows: linear 8,450; Ridge 3,112; Lasso 2,587 |

A coefficient of `-0.497` on the log price means the model multiplies the price by `exp(-0.497)` = 0.608 for that one category. One training house has it, and the plain model believed that house completely.

## Hint

`alpha_` exists only after `fit`. Reach the last step of a pipeline with `model[-1]`. Lasso warns about convergence when `max_iter` is too small; 50,000 is enough here.

## Bonus

Try `ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], alphas=np.logspace(-5, -1, 13), cv=cv, max_iter=50000, random_state=42)`. Which `l1_ratio_` does it choose, how many coefficients does it keep, and what is its test RMSE? Check: 0.8, 83 kept, RMSE 19,517.

## Known leaks and quirks in this dataset

- `SaleType` and `SaleCondition` are recorded when the sale closes: dropped.
- `Id` is an identifier.
- The 4 houses over 4,000 sq ft are removed before the split (De Cock's advice).
- Rare categories are where an unpenalized linear model goes wrong first: a one-hot column with a single 1 in it can earn any coefficient.

## References

Ames Housing, De Cock (2011), *Journal of Statistics Education* 19(3); OpenML 42165, released for teaching. Numbers from `D02/reference/04_regularization.py`.
