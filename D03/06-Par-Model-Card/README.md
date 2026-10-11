# Write the model card for the churn model

In this activity, you and a partner will write a one-page model card for the churn model you shipped in 6.2, so that someone who never saw this course can decide whether to use it and how.

**Time:** 15 minutes · **Data:** the numbers below, from `D03/reference/10_model_card.py` · **Starter:** `Unsolved/model_card.md`

## The situation

A model card (Mitchell et al., 2019) is a short document that travels with a model: what it is for, what data it learned from, how well it does overall and for groups of customers, and where it should not be used. The retention team's new analyst will read it before using the scores, and so will the person who audits the model next year.

The model: logistic regression after D01 5.6's `ColumnTransformer`, 19 input columns, fitted on 5,634 training customers, flagging a customer when the predicted probability is at least `0.278` (chosen on validation folds from the 100-dollar offer and 400-dollar lost customer in 2.2).

## Instructions

1. Open `Unsolved/model_card.md`. It has eight headings: **Model details, Intended use, Out of scope, Training data, Evaluation data, Metrics, Results by group, Caveats and recommendations.**

2. One partner fills in the first four headings, the other the last four. Use the numbers in the two tables below; write in sentences a manager can read.

3. Under **Results by group**, say where the model is weaker, and what that means for the retention team. Look at the `Contract` rows first.

4. Under **Out of scope**, name at least two uses the model must not be put to, and why.

5. Swap and read each other's half. Mark one sentence each that a reader could misunderstand, and rewrite it.

**Overall, 1,409 test customers**

| Measure | Value |
|---|---|
| ROC AUC | 0.842 |
| Brier score | 0.1381 |
| Threshold | 0.278 |
| Customers flagged | 564 |
| Recall at the threshold | 0.781 |
| Precision at the threshold | 0.518 |
| Cost per customer at the threshold (100 per offer, 400 per missed churner) | 63.31, against 69.41 at the 0.5 default |

**By group, at the same threshold**

| Group | Customers | Churn rate | Share flagged | Recall | Precision | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| Female | 687 | 0.281 | 0.400 | 0.772 | 0.542 | 0.838 |
| Male | 722 | 0.251 | 0.400 | 0.790 | 0.495 | 0.847 |
| Not senior | 1,187 | 0.233 | 0.350 | 0.728 | 0.484 | 0.842 |
| Senior | 222 | 0.441 | 0.671 | 0.929 | 0.611 | 0.784 |
| Month-to-month | 773 | 0.426 | 0.699 | 0.863 | 0.526 | 0.743 |
| One year | 300 | 0.120 | 0.080 | 0.222 | 0.333 | 0.740 |
| Two year | 336 | 0.027 | 0.000 | 0.000 | none flagged | 0.756 |

## What a good answer includes

- **Intended use** names the decision: which customers get a retention offer this month, with a person reviewing the list. It says the scores are probabilities and the flag is a cost choice that changes if the offer price changes.
- **Training data** says what one row is (one customer, one snapshot), the 26.5% churn rate, the source and license, and that the data has no date, so nobody can tell how old the customers' behavior is.
- **Results by group** says that the model flags no two-year customer and finds 0.222 of the one-year churners, so the team's list misses churners on longer contracts. Senior customers have a lower ROC AUC (0.784 against 0.842 overall): the ranking is weaker for them.
- **Out of scope** rules out at least two of: pricing or credit decisions about a person; deciding who gets worse service; customers outside this company's market; any use after the product range or prices change, until the model is checked again.
- **Model details** or **Caveats** say that `gender` and `SeniorCitizen` are inputs to the model, and whether that is acceptable for this decision: results by group are similar for women and men, but a reviewer should know the columns are there.
- **Caveats** say that the model describes association, not cause: a customer on a month-to-month contract is flagged more often, which does not show that changing their contract would keep them (4.1). And they say when to retrain, with the 6.3 drift rule.
- Every number in the card is one of the numbers above, with its unit.

Your card can differ from the sample and still be right. Check it against these points, not against the wording.

## Hint

Write **Out of scope** by asking: "Who could be harmed if someone used these scores for a different decision?" A churn score that is fine for choosing who gets a discount is not fine for choosing who gets a worse price.

## Bonus

Add a line to **Caveats**: under the EU AI Act, would this model be high-risk? Check the Annex III list on the 6.3 slide. A churn model for retention offers is not on it; the same score used for creditworthiness would be.

## Known leaks and quirks in this dataset

- `TotalCharges` is read as text: 11 cells hold a single space, all for customers with `tenure` 0.
- `customerID` is an identifier, never a feature.
- `tenure`, `MonthlyCharges` and `TotalCharges` are near-collinear.
- The IBM sample has no dates: the card cannot say when the customers were observed.

## References

Mitchell, Wu, Zaldivar, Barnes, Vasserman, Hutchinson, Spitzer, Raji, Gebru, "Model Cards for Model Reporting", FAT* 2019, https://arxiv.org/abs/1810.03993. Hugging Face model card guide, https://huggingface.co/docs/hub/en/model-card-guidebook. IBM Telco Customer Churn, https://github.com/IBM/telco-customer-churn-on-icp4d, IBM sample data for educational use. Numbers from `D03/reference/10_model_card.py` and `02_threshold.py`.
