# Six business problems · sample answers

Shown at the 2.3 Review. A group answer that differs here can still be right if it meets the points in the README.

| # | One row is | Kind | Answer column | Why |
|---|---|---|---|---|
| 1 | One subscriber at the end of a quarter | Supervised classification | `canceled_next_quarter` (yes / no), from past quarters | The answer is a category; past quarters give labeled rows. The team can call 500, so the model's job is to rank subscribers by risk |
| 2 | One day | Supervised regression (a time series) | `tickets` that day | The answer is a number. Rows are ordered in time, so the split must keep the past for training and the future for testing |
| 3 | One refund claim | No ML: a rule | none needed | The rule is known and exact: `amount >= 2000`. A model can only add mistakes to it |
| 4 | One customer | Unsupervised (clustering) | none | No column says which group a customer is in. Clustering proposes groups; marketing decides whether they are useful |
| 5 | One imported transaction | Supervised multiclass classification | `category`, from the categories customers already chose | Many past transactions are already categorized by customers; the merchant text is the strongest feature |
| 6 | One customer email | Generative AI | none in the tabular sense | The output is text, not a column. An LLM drafts; an agent edits and sends |

**Most common wrong reading:** card 2 called classification because "it predicts". A prediction of a number is regression.

**Second most common:** card 3 called anomaly detection. A claim above a known threshold is not an anomaly; it is a rule the business has already written.
