# Place three products on the grid

In this activity, your group will place three ML products on the training-cadence by inference-mode grid and sketch the boxes each one needs, from data to monitoring.

**Time:** 15 minutes · **Groups:** 3 to 5 people · **Data:** the three product cards below · **Report:** one grid and one sketch per group

## Instructions

1. Draw the grid on paper or a whiteboard. Across: **inference mode**, batch on the left, online on the right. Up: **training cadence**, scheduled at the bottom, continuous at the top.

    - **Batch inference:** the model scores many rows at once, on a schedule, and writes the scores to a table.
    - **Online inference:** the model scores one request at a time, while someone or something waits for the answer.
    - **Scheduled training:** the model is retrained offline, on a schedule or by hand.
    - **Continuous training:** the model updates as new labeled data arrives, without a person starting it.

2. Place each product card in one quadrant. Write one sentence on the evidence in the card that put it there.

3. For one product of your choice, sketch the boxes from the lifecycle slide: question, data, features, train, evaluate, batch or API, monitor. Mark which boxes run every day and which run rarely.

### The cards

| Card | Product |
|---|---|
| A | **Churn scores.** Every night, a job scores every subscriber and writes a risk score to a table. The retention team reads the table each morning. The model is retrained once a quarter. |
| B | **Payment check.** Each card payment is scored before it is approved. The answer must come back in under 100 milliseconds. The model is retrained every Sunday on last week's confirmed fraud. |
| C | **Help-article ranking.** When a user types a question in the help panel, articles are ranked for them. The ranking model updates from clicks every few minutes. |

## What a good answer includes

- Card A in **scheduled, batch**: the scores are written to a table nightly and nobody waits on a single prediction.
- Card B in **scheduled, online**: one payment at a time, with a time limit. Weekly retraining is still scheduled training, because a job retrains offline and someone can check the new model before it goes live.
- Card C in **continuous, online**: the model changes between one user's question and the next.
- A sketch that shows where the scores go (a table for batch, an API for online) and what is watched once the model is live.
- The common wrong reading, so you can test for it: calling card B "online learning" because it is online. Online *inference* is how predictions are served; online *learning* is how the model is updated. Card B serves online and learns on a schedule.

Your answer can differ from the sample and still be right. Check it against these points, not against the wording.

## Hint

Two questions per card. Who or what is waiting for one prediction? Who starts a retrain?

## Bonus

Which quadrant is empty? Describe a product that would sit there.

## References

Google Machine Learning Crash Course, "Production ML systems: static versus dynamic training" and "static versus dynamic inference", https://developers.google.com/machine-learning/crash-course/production-ml-systems (updated 2025-12-03, read 2026-10-02). The three product cards are written for this course.
