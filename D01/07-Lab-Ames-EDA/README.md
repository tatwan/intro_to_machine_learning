# Lab: exploratory analysis of house sales

In this lab, you will explore 1,460 house sales from Ames, Iowa the way you would before any model: what the target looks like, what is missing and why, which columns move with price, which houses do not fit, and which columns say the same thing twice. You finish with three feature decisions, each tied to something you found.

**Time:** 50 minutes · **Data:** `data/ames.csv` · **Starter:** `Unsolved/ames_eda.ipynb`

One row is one house sale, recorded between 2006 and 2010. The target is `SalePrice`, in US dollars. A local real estate agency is the client: it wants a price model, and it wants to know what in the data it can trust first.

Work through the checkpoints in order. C1 is the minimum, C2 is today's target, C3 is for those who finish C2.

## C1 · The target

1. Load the file and print its shape. Say in one sentence what one row is.

2. The agency asks: "What does a typical house sell for here?"

    Translate: `SalePrice.describe()`, then the mean and the median. Which one would you quote to the agency, and why?

3. Plot a histogram of `SalePrice` (`df["SalePrice"].hist(bins=30)` or `sns.histplot`). Which side has the long tail? Print `df["SalePrice"].skew()`.

## C2 · Missing values, relationships, outliers

4. The agency asks: "Is the data complete?"

    Translate: count missing values per column, sorted, and keep the columns with at least one. Then, for `PoolQC` and `GarageType`, check whether a missing value means "no pool" or "no garage": compare with `PoolArea == 0` and `GarageArea == 0`.

5. Read `MasVnrType` again with `pd.read_csv(path, keep_default_na=False)` and count each value. How many of the "missing" values were really the word `None`?

6. The agency asks: "What moves the price most?"

    Translate: the correlation of every numeric column with `SalePrice`, sorted. Keep the top five.

7. Plot `GrLivArea` (living area in square feet) against `SalePrice`. Find the houses with more than 4,000 sq ft. How many are there, and what did the cheapest ones sell for? Look at their `SaleCondition`.

8. The agency asks: "Which sale prices are unusual?"

    Translate: Tukey's rule on `SalePrice`. Compute Q1 and Q3 with `quantile([0.25, 0.75])`, the IQR (Q3 − Q1), and the upper fence Q3 + 1.5 × IQR. Count the prices above it. Draw a box plot (`plt.boxplot(df["SalePrice"], orientation="horizontal")`) and find the fence on it.

## C3 · From findings to decisions

9. Plot a histogram of `np.log(df["SalePrice"])` and print its skew.

10. Group by `OverallQual` (1 to 10) and print the median price for each level. Do the same for `Neighborhood` and name the cheapest and the dearest.

11. Find one pair of columns that says almost the same thing (hint: the garage). Print their correlation.

12. Write three feature decisions for the agency, one line each, in the form "Because *finding*, we will *action*." Example: "Because `PoolQC` is missing exactly where `PoolArea` is 0, we will fill it with `None` as a category of its own."

## Check yourself

| Step | Value to check |
|---|---|
| 1 | `(1460, 81)` |
| 2 | Mean 180,921; median 163,000; min 34,900; max 755,000 |
| 3 | Long tail to the right; skew `1.88` |
| 4 | 19 columns with missing values; top five `PoolQC` 1,453, `MiscFeature` 1,406, `Alley` 1,369, `Fence` 1,179, `MasVnrType` 872. All 1,453 missing `PoolQC` rows have `PoolArea` 0; all 81 missing `GarageType` rows have `GarageArea` 0 |
| 5 | 864 of the 872 were the word `None`; 8 are really blank |
| 6 | `OverallQual` 0.79, `GrLivArea` 0.71, `GarageCars` 0.64, `GarageArea` 0.62, `TotalBsmtSF` 0.61 (`1stFlrSF` also 0.61) |
| 7 | 4 houses; the two cheapest sold for 160,000 and 184,750, both `Partial` sales |
| 8 | Q1 129,975, Q3 214,000, IQR 84,025; upper fence 340,037; 61 prices above it |
| 9 | Skew `0.12` |
| 10 | Quality 1: 50,150, quality 10: 432,390. Cheapest neighborhood `MeadowV` 88,000; dearest `NridgHt` 315,000 |
| 11 | `GarageCars` and `GarageArea`: 0.88 |

A different count is information, not failure. Check how you read the file first: `keep_default_na` changes step 4 and step 5.

## Hint

`df.isna().sum().sort_values(ascending=False).head(8)` answers step 4. `df.select_dtypes(include="number").corr()["SalePrice"]` answers step 6.

## Bonus

Correlation measures straight-line association. Compute the Spearman correlation (`method="spearman"`) for `OverallQual` and `GrLivArea`. Which one changes more, and what does that say about the shape of the relationship?

## More practice (not checked)

- Plot `SalePrice` by `OverallQual` as box plots. Where do the boxes overlap?
- `LotArea` has a skew of 12.21. Plot it with and without a log scale.

## Known leaks and quirks in this dataset

- Many `NaN` mean "none": no pool, no alley access, no fence, no garage, no basement. They are categories, not gaps.
- pandas reads the text `None` (in `MasVnrType`) as missing by default.
- `LotFrontage` has 259 values that are really missing.
- Four houses over 4,000 sq ft; De Cock, who published the data, recommends removing them before modeling.
- `SaleType` and `SaleCondition` are recorded when the sale closes. A model that prices a house before it is listed cannot see them: treat them as a leak.
- `Id` is an identifier, never a feature.

## References

De Cock, D. (2011), "Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project", *Journal of Statistics Education* 19(3), https://jse.amstat.org/v19n3/decock.pdf. Course copy: OpenML dataset 42165 (the 1,460-row training split), released for teaching. Numbers from `D01/reference/07_ames_eda.py`.
