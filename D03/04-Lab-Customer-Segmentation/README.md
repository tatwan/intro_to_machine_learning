# Lab: customer segmentation with RFM and k-means

In this lab, you will turn a year of online-shop transactions into one row per customer, cluster the customers with k-means, choose the number of clusters, and give each segment a name and an action the marketing team can use.

**Time:** 25 minutes · **Data:** `data/online_retail_ii.csv` · **Starter:** `Unsolved/segmentation.ipynb`

## The situation

The shop is a UK online retailer of gifts; the UCI page says many of its customers are wholesalers. One row of the raw table is one product line on one invoice. The starter loads the year from 2010-12-01 to 2011-12-09 with `Invoice` and `StockCode` read as text: 541,910 rows.

The marketing lead: "We send the same email to everyone. Give me a few groups of customers that behave differently, say how big each one is, and what you would send each."

Work through the checkpoints in order. C1 is the minimum, C2 is today's target, C3 is for those who finish C2.

## C1 · One row per customer

1. Clean the rows, one rule at a time, and count what each rule removes: rows with no `Customer ID`; cancellations (`Invoice` starts with `C`); rows with `Quantity` or `Price` not above 0; stock codes that are not products (`POST`, `D`, `M`, `C2`, `DOT`, `BANK CHARGES` and the others listed in the starter).

2. Add `Revenue = Quantity * Price`. Set `snapshot` to the day after the last invoice.

3. Group by `Customer ID` and build the RFM table: **Recency** (days from the customer's last invoice to `snapshot`), **Frequency** (distinct invoices), **Monetary** (total revenue).

## C2 · Cluster and choose k

4. Look at the three columns with `describe()` and `skew()`. Then build `prep = make_pipeline(FunctionTransformer(np.log1p), StandardScaler())` and `Z = prep.fit_transform(rfm)`.

5. For k from 2 to 10, fit `KMeans(n_clusters=k, n_init=10, random_state=42)` on `Z` and record `inertia_` and `silhouette_score(Z, labels)`. Plot both against k.

6. The marketing lead asks: "How many groups?"

    Translate: where the inertia curve bends, what the silhouette score says, and how many groups the team can act on. Fit the k you chose.

## C3 · Name the segments

7. For your k, compute per segment: number of customers, median recency, median frequency, median monetary, and the segment's share of total revenue.

8. Give each segment a name in the marketing lead's words and one action. Write it as a table with four columns: name, size, what they look like, what to send.

## Check yourself

| Step | Value to check |
|---|---|
| 1 | Removed: no Customer ID 135,080; cancellations 8,905; quantity or price not above 0: 40; not a product 1,548. Left: 396,337 rows |
| 3 | 4,334 customers; snapshot 2011-12-10. Medians: recency 50 days, frequency 2, monetary 668.1 |
| 4 | Skew before the log: recency 1.243, frequency 11.949, monetary 19.564; after: -0.467, 1.214, 0.396 |
| 5 | Inertia k = 2: 6,479.0; k = 4: 3,955.7; k = 10: 2,019.1. Silhouette k = 2: 0.434; k = 3: 0.342; k = 4: 0.338 |
| 7 (k = 4) | See the table below |

| Customers | Median recency (days) | Median frequency | Median monetary | Revenue share |
|---:|---:|---:|---:|---:|
| 689 | 8 | 10 | 3,735 | 0.641 |
| 1,182 | 53 | 4 | 1,358 | 0.242 |
| 815 | 16 | 2 | 472 | 0.051 |
| 1,648 | 173 | 1 | 297 | 0.065 |

The silhouette score is highest at k = 2, which splits recent buyers from lapsed ones. That is a correct answer to "how many natural groups?" and a weak one to "how many emails should we write?". k = 3 and k = 4 score almost the same (0.342 and 0.338). The demonstration uses 4 because each of the four profiles above leads to a different action. A different k with a reason is a good answer too.

Cluster numbers (0, 1, 2, 3) are arbitrary labels and can come out in a different order on your machine; compare the profiles, not the numbers.

## Permitted help

Helpers may point to the step, the hint and the error message, and may name a function (`groupby`, `nunique`, `silhouette_score`). They do not type in a learner's notebook.

## Hint

`df.groupby("Customer ID").agg(last=("InvoiceDate", "max"), frequency=("Invoice", "nunique"), monetary=("Revenue", "sum"))`, then `recency = (snapshot - last).dt.days`.

## Bonus

Fit k-means with k = 4 on the raw RFM columns, with no log and no scaling. Count the customers in each cluster. Check: 4,296, 31, 5 and 2. Monetary has a standard deviation of 8,907.5 and recency 100.2, so distance is almost entirely about money.

## More practice (not checked, not submitted)

Add two columns per customer, the number of distinct products and the mean basket value (monetary / frequency), and run PCA on the five scaled columns. How many components keep 90% of the variance? Which two columns does PCA treat as almost one?

## Known leaks and quirks in this dataset

- 24.9% of the raw rows have no `Customer ID`: they cannot be put in a customer segment and are dropped.
- Cancellations have an `Invoice` starting with `C` and a negative `Quantity`. Dropping them leaves the original purchase in place. Customer 12346 bought 74,215 units of one product and canceled them the same day, so after cleaning they show monetary 77,183.60 from one invoice. Look at your biggest customers before you trust a segment.
- Stock codes such as `POST` (postage), `M` (manual), `D` (discount) and `BANK CHARGES` are not products.
- Monetary is heavy-tailed: the largest customer spent 279,138.0 against a median of 668.1. That is why the columns are logged before scaling.
- 89.3% of the cleaned rows are from the United Kingdom.

## References

UCI Online Retail II, https://archive.ics.uci.edu/dataset/502/online+retail+ii, CC BY 4.0; the year 2010-12-01 to 2011-12-09. Numbers from `D03/reference/06_kmeans.py`.
