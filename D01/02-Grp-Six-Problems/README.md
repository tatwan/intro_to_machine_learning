# Six business problems

In this activity, your group will decide, for six requests a product team might send you, which kind of machine learning fits, or whether a rule or a query does the job without ML.

**Time:** 15 minutes · **Groups:** 3 to 5 people · **Data:** the six cards below · **Report:** one answer per card, one speaker per group

## Instructions

1. Read the six cards. For each one, write down:

    - **What one row is**: one customer, one transaction, one day.
    - **The kind of ML**, if any: supervised classification, supervised regression, unsupervised (clustering), generative AI, or no ML.
    - **The answer column**: the column a past record would need for a model to learn from it. Write "none" if no such column exists.

2. Mark the one card your group argued about longest. That is the card your speaker reports.

### The cards

| # | The request, as the product team wrote it |
|---|---|
| 1 | "Which small-business subscribers will cancel next quarter? The retention team can call 500 of them." |
| 2 | "How many support tickets will arrive each day next month? We set staffing two weeks ahead." |
| 3 | "Any refund claim of 2,000 or more must go to a person for review. Can ML do that?" |
| 4 | "Marketing wants to split 40,000 small-business customers into groups. Nobody has defined the groups yet." |
| 5 | "When a customer imports bank transactions, suggest a category for each one: travel, meals, software, and so on." |
| 6 | "Draft a polite first reply to customers who email about a late invoice. An agent edits and sends it." |

## What a good answer includes

- One row named for every card, before the kind of ML. "One row is one subscriber" makes card 1 easier to argue about.
- An answer column for every supervised card, and where it comes from: past cancellations, past ticket counts, categories customers already chose.
- Card 3 answered with a rule. The threshold is already known, so a model can only add errors.
- Card 4 named as unsupervised, with the reason: there is no column that says which group a customer belongs to.
- Card 6 named as generative AI, with a person checking the output before it reaches a customer.
- The common wrong reading, so you can test for it: "it predicts something, so it must be classification". Card 2 predicts a number, which makes it regression.

Your answer can differ from the sample and still be right. Check it against these points, not against the wording.

## Hint

Ask of each card: in last year's data, does a column already hold what you want to predict? If yes, it is supervised. If no, it is unsupervised, generative, or not ML.

## Bonus

Which card would you ship first with no ML at all, and what would that first version do?

## References

The six cards are written for this course. Rules of ML #1 and #4: Google, "Rules of Machine Learning", https://developers.google.com/machine-learning/guides/rules-of-ml (page updated 2025-08-25, read 2026-10-02).
