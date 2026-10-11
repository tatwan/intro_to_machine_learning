# Leakage hunt

In this activity, you will find the column in a bank's marketing table that the model could never have at prediction time, remove it, and measure how much of the model's score it was faking.

**Time:** 15 minutes · **Data:** `data/bank_marketing.csv` · **Starter:** `Unsolved/leakage_hunt.ipynb`

## The situation

A Portuguese bank phoned clients to sell a term deposit. One row is one client in the campaign, described as of their last call. The target `y` says whether they subscribed (`yes` / `no`).

The marketing director has a model from a contractor: "It is 90% accurate. Next month we use it to choose whom to call." The starter notebook rebuilds that model. It uses every column.

## The columns

| Column | What it holds |
|---|---|
| `age`, `job`, `marital`, `education` | The client |
| `default`, `balance`, `housing`, `loan` | Credit in default? Average yearly balance in euros; housing loan? personal loan? |
| `contact` | How the last call was made: cellular, telephone, unknown |
| `day_of_week`, `month` | Day of the month and month of the last call (UCI names the day column `day_of_week`) |
| `duration` | Length of the last call, in seconds |
| `campaign` | Calls to this client in this campaign, including the last one |
| `pdays` | Days since the client was last called in an earlier campaign; `-1` means never |
| `previous`, `poutcome` | Calls before this campaign, and how the earlier campaign ended |

## Instructions

1. Run the starter notebook. It splits the rows 75 / 25 with `stratify=y`, fits a pipeline (scale the numbers, one-hot encode the text, logistic regression) and prints two numbers on the 11,303 test rows: accuracy, and the call-list hit rate.

    **Call-list hit rate**: the model scores every test client; take the 10% it ranks highest (1,130 clients, the people the bank would call first) and count how many of them subscribed. Calling at random hits 11.7%, the share of subscribers overall.

2. The director asks: "Can we trust that number for next month's calls?"

    Translate: for each column, decide whether its value exists **before** the call you are deciding to make. Mark each column *before* or *after*.

3. Remove every *after* column from the feature list. Refit and print both metrics again.

4. Print the accuracy of a `DummyClassifier(strategy="most_frequent")` on the same split. Compare it with step 1.

5. Look at `pdays`. What share of rows is `-1`? A model reads `-1` as "one day less than 0". Add a column `contacted_before` (yes / no) and set `pdays` to 0 where it was `-1`. Refit.

6. Write two sentences for the director: what the contractor's score was, what the honest score is, and why the gap exists.

## Check yourself

| Step | Value to check |
|---|---|
| 1 | Accuracy `0.901`; hit rate `0.604` on 1,130 called clients |
| 2 | One column is *after*: `duration` |
| 3 | Hit rate `0.496`; accuracy `0.895` |
| 4 | Dummy accuracy `0.883` (88.3% of clients said no); random calls hit `0.117` |
| 5 | 36,954 rows (`0.817` of all rows) have `pdays = -1` |

Accuracy hardly moved in step 3 (0.901 to 0.895) because almost everyone says no: accuracy hides the leak, the hit rate shows it. Without `duration` the model still beats random calling by a wide margin; that is the number to report.

## Hint

You cannot know how long a call lasted before you dial. The dataset's own documentation says the same.

## Bonus

`campaign`, `day_of_week` and `month` describe the *last* call. They are known before that call only if the model scores each client just before each call. If the model picks the call list once, at the start of the month, which of them are still *before*?

## Known leaks and quirks in this dataset

- **`duration` is known only after the call ends.** The authors' variable notes in the UCI download say to keep it only for benchmarks and discard it for a realistic model.
- `pdays = -1` is a code for "never contacted", not a number of days.
- `poutcome` is `unknown` for every client with no earlier campaign (36,959 rows).
- Depending on the download route, `unknown` in `job`, `education`, `contact` and `poutcome` arrives as the text `unknown` or as a missing value. The starter fills missing values back with `"unknown"`, so it is a category of its own either way.

## References

Moro, Rita and Cortez, Bank Marketing, UCI Machine Learning Repository dataset 222, https://archive.ics.uci.edu/dataset/222/bank+marketing, CC BY 4.0. Numbers from `D01/reference/05_bank_leakage.py`.
