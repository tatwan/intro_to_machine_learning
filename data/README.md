# data/

The seven tables the course uses, fixed so that every notebook prints the values in its README's Check yourself table. Read them with a path from the repo root, such as `pd.read_csv("data/penguins.csv")`.

Do not edit these files. If you want to change a table, change a copy in your notebook.

## The files

| File | Rows | Target | Source and license | Copy used here |
|---|---:|---|---|---|
| `penguins.csv` | 344 | `species` | Palmer Penguins, Horst, Hill and Gorman, https://github.com/allisonhorst/palmerpenguins. CC0 | The package's `penguins.csv`, unchanged |
| `telco_churn.csv` | 7,043 | `Churn` | IBM Telco Customer Churn, https://github.com/IBM/telco-customer-churn-on-icp4d. IBM sample data for educational use | `Telco-Customer-Churn.csv`, unchanged |
| `bank_marketing.csv` | 45,211 | `y` | Moro, Rita and Cortez, Bank Marketing, UCI Machine Learning Repository dataset 222, https://archive.ics.uci.edu/dataset/222/bank+marketing. CC BY 4.0 | bank-full, downloaded with the `ucimlrepo` package, which names the day-of-month column `day_of_week` and writes the code `unknown` as an empty cell in `job`, `education`, `contact` and `poutcome` (the notebooks turn it back into `unknown`) |
| `ames.csv` | 1,460 | `SalePrice` | De Cock (2011), Ames Housing, *Journal of Statistics Education* 19(3); OpenML dataset 42165, the 1,460-row training split. Released for teaching | Downloaded with scikit-learn's `fetch_openml`. Missing cells are empty; every other cell is as published, so the word `None` stays text |
| `credit_default.csv` | 30,000 | `Y` (the notebooks call it `default`) | Yeh, Default of Credit Card Clients, UCI Machine Learning Repository dataset 350, https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients. CC BY 4.0 | Downloaded with the `ucimlrepo` package, which names the 23 features `X1` to `X23`. The notebooks give them the names of UCI's variable table, in order: `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`, `PAY_0`, `PAY_2` to `PAY_6`, `BILL_AMT1` to `BILL_AMT6`, `PAY_AMT1` to `PAY_AMT6`. The `ID` column, 1 to 30,000, is dropped |
| `bike_hourly.csv` | 17,379 | `cnt` | Fanaee-T, Bike Sharing, UCI Machine Learning Repository dataset 275 (`hour.csv`), https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset. CC BY 4.0 | `hour.csv`, downloaded with the `ucimlrepo` package |
| `online_retail_ii.csv` | 541,910 | none | Chen, Online Retail II, UCI Machine Learning Repository dataset 502, https://archive.ics.uci.edu/dataset/502/online+retail+ii. CC BY 4.0 | The second sheet of the UCI file, "Year 2010-2011": the year 2010-12-01 to 2011-12-09, which the lab uses. The full table covers two years and is about 93 MB as CSV, close to GitHub's 100 MB limit, so this repo keeps one year. `InvoiceDate` is written in ISO 8601 date-and-time format |

## Known leaks and quirks

Every activity README repeats the ones that matter for that activity.

- **`penguins.csv`**: 2 rows have no measurements at all; `sex` is missing in 11 rows. `year` is the study year, not a measurement.
- **`telco_churn.csv`**: `TotalCharges` is text with 11 blank values, all for customers with `tenure` 0, who have not been billed yet. `customerID` is an identifier, never a feature. `tenure`, `MonthlyCharges` and `TotalCharges` are near-collinear. `SeniorCitizen` is coded 0 / 1 while the other yes / no columns use words.
- **`bank_marketing.csv`**: **`duration` is known only after the call ends: a leak.** `pdays = -1` means "never contacted before", a code rather than a number of days. `day_of_week` holds the day of the month, despite its name.
- **`ames.csv`**: many empty cells mean "none" (no pool, no alley, no garage, no basement), not a gap. pandas reads the word `None` in `MasVnrType` as missing unless you pass `keep_default_na=False`. Four houses over 4,000 sq ft are outliers De Cock recommends removing. `SaleType` and `SaleCondition` are recorded when the sale closes, so a model that prices a house before it is listed cannot use them.
- **`credit_default.csv`**: `EDUCATION` holds codes 0, 5 and 6 and `MARRIAGE` holds 0, none of them documented. `PAY_0` to `PAY_6` use -2 and 0, also undocumented; -1 means paid on time. The September column is `PAY_0`; there is no `PAY_1`.
- **`bike_hourly.csv`**: **`casual + registered = cnt` in every row: drop both before predicting `cnt`.** The rows are in time order, so a shuffled split lets the model see the future; 2011 against 2012 shows drift.
- **`online_retail_ii.csv`**: cancellations have an `Invoice` starting with `C` and a negative `Quantity`. About a quarter of the rows have no `Customer ID`. Stock codes such as `POST`, `M`, `D` and `BANK CHARGES` are not products. `Price` times `Quantity` is heavy-tailed.

## Checksums

The first 16 hexadecimal characters of each file's SHA-256, for checking a copy: `sha256sum data/*.csv` (macOS: `shasum -a 256 data/*.csv`).

| File | Rows | Columns | sha256[:16] |
|---|---:|---:|---|
| `penguins.csv` | 344 | 8 | `f2d6493fc64ad958` |
| `telco_churn.csv` | 7,043 | 21 | `a0ea74c48c1063be` |
| `bank_marketing.csv` | 45,211 | 17 | `1ae070f431c41d2c` |
| `ames.csv` | 1,460 | 81 | `b214efc1b289b567` |
| `credit_default.csv` | 30,000 | 25 | `70b04919874ce714` |
| `bike_hourly.csv` | 17,379 | 17 | `11e08374c4482296` |
| `online_retail_ii.csv` | 541,910 | 8 | `9f3414a3d06748ef` |
