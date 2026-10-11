# Build a regression model

In this activity, you will fit a linear regression to Ames house prices, first on one column and then on all of them, and report its errors in dollars the agency can read.

**Time:** 20 minutes · **Data:** `data/ames.csv` · **Starter:** `Unsolved/ames_regression.ipynb`

## The situation

The real estate agency from D01's lab has your EDA decisions and wants a first price model. One row is one house sale in Ames, Iowa, 2006 to 2010. The starter already applies D01 6.4's decisions: it reads the file with `keep_default_na=False, na_values=[""]`, removes the 4 houses over 4,000 sq ft, takes `y = np.log(SalePrice)`, drops `Id`, `SaleType` and `SaleCondition`, and splits 75 / 25 with `random_state=42`. It also builds D01's three-block `ColumnTransformer` as `pre`.

## Instructions

1. Run the first cell. Check the number of training and test houses.

2. The agency asks: "Roughly, what is a square foot worth here?"

    Translate: fit `LinearRegression()` on `X_train[["GrLivArea"]]` against the price in dollars, `np.exp(y_train)`. Print `coef_` and `intercept_`, and predict the price of a 1,500 sq ft house.

3. Score that one-column model on the test houses: MAE, RMSE and R² in dollars (`mean_absolute_error`, `root_mean_squared_error`, `r2_score`). Then score a `DummyRegressor()` (it always predicts the mean training price) the same way.

4. Now use every column: `make_pipeline(pre, LinearRegression())`, fitted on `X_train, y_train` (the log price). Predict the test houses, turn the predictions back into dollars with `np.exp`, and print MAE, RMSE and R² in dollars. Print R² on the training houses too.

5. Plot the residuals: predicted price on the x-axis, actual minus predicted on the y-axis. Find the house with the largest miss and print its living area, neighborhood, actual and predicted price.

6. The agency asks: "How far off is a typical estimate?"

    Translate: answer in one sentence with the MAE from step 4, and add the median of `abs(predicted / actual - 1)` as a percentage.

## Check yourself

| Step | Value to check |
|---|---|
| 1 | 1,092 training and 364 test houses; 77 feature columns |
| 2 | Slope `116.8` dollars per sq ft (116.81 unrounded); intercept 5,647; 1,500 sq ft predicts 180,867 |
| 3 | MAE 37,880, RMSE 52,336, R² `0.453`. Predicting the mean: RMSE 71,312, R² `-0.015` |
| 4 | Test MAE 15,354, RMSE 22,091, R² `0.903`; training R² `0.956` |
| 5 | Largest miss: 1,973 sq ft in `NridgHt`, sold for 395,000, predicted 280,974 |
| 6 | MAE 15,354; median error `0.065` (6.5%) |

## Hint

The model learned log prices, so `model.predict` returns log prices. Turn both the predictions and `y_test` back with `np.exp` before you compute an error in dollars. An RMSE of 0.14 is not fourteen cents.

## Bonus

Add a column `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF` to both `X_train` and `X_test` and refit the step-4 pipeline. Why does the test RMSE not move at all? (Think about what a linear model can already do with the three columns it has.) Then try only five columns: `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `YearBuilt`. Check: TotalSF RMSE 22,091; five columns RMSE 27,035, R² `0.854`.

## Known leaks and quirks in this dataset

- `SaleType` and `SaleCondition` are recorded when the sale closes, after the price is agreed: they are dropped (D01 6.4).
- `Id` is an identifier.
- Four houses over 4,000 sq ft are a different population; De Cock, who published the data, recommends removing them.
- Many missing values mean "none" (no garage, no basement, no pool); the text `None` in `MasVnrType` is a category, which is why the file is read with `keep_default_na=False`.

## References

Ames Housing, De Cock (2011), *Journal of Statistics Education* 19(3); OpenML 42165, released for teaching. Numbers from `D02/reference/02_ames_linreg.py`.
