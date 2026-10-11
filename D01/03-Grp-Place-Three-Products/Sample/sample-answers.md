# Place three products on the grid · sample answers

Shown at the 3.4 Review.

| Card | Quadrant | Evidence in the card |
|---|---|---|
| A Churn scores | Scheduled training, batch inference | "every night ... writes a risk score to a table"; "retrained once a quarter" |
| B Payment check | Scheduled training, online inference | "before it is approved ... under 100 milliseconds"; "retrained every Sunday" |
| C Help-article ranking | Continuous training, online inference | "when a user types"; "updates from clicks every few minutes" |

Empty quadrant: **continuous training, batch inference**. Example: a demand forecast that updates its model as each day's sales arrive, and publishes tomorrow's forecast as one table each evening. Rare, because a batch product has time to retrain on a schedule.

## Sketch for card A

`Question` (who will cancel next quarter) → `Data` (subscriptions, payments, support tickets, nightly snapshot) → `Features` (tenure, plan, tickets in the last 90 days) → `Train` (quarterly) → `Evaluate` (on the last quarter held out) → `Batch` (nightly job writes `churn_scores`) → `Monitor` (score distribution each night; actual cancellations each quarter).

Every day: data, features, batch, monitor. Rarely: train and evaluate.

**Most common wrong reading:** card B placed in the continuous row because it is "online". The weekly retrain is a scheduled job; online describes how predictions are served.
