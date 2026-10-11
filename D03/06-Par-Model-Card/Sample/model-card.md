# Model card · telco churn, retention offers

Sample answer for D03 6.4 and 6.5. Numbers from `D03/reference/10_model_card.py` and `02_threshold.py`.

## Model details

Logistic regression (scikit-learn 1.9.1, `LogisticRegression(max_iter=1000)`) after one `ColumnTransformer`: median imputation and scaling for the 4 numeric columns, one-hot encoding for the 15 text columns. Registered in MLflow as `telco-churn`, version 1, alias `champion`. Chosen over a random forest and a boosted model on 5-fold cross-validated ROC AUC (0.846 ± 0.013 against 0.838 and 0.821).

## Intended use

Each month, rank current customers by their predicted probability of leaving and flag those at or above 0.278 for a retention offer. A person in the retention team reviews the list before any offer goes out. The threshold comes from two costs, 100 dollars per offer and 400 dollars per customer lost without one. If either cost changes, choose the threshold again on validation folds; do not edit it by hand.

## Out of scope

- Any decision that makes a customer worse off: prices, credit, service level, contract terms. The model was never checked for that use, and a wrong score would harm the customer instead of costing an offer.
- Customers of another company, region or product range. The model learned one company's customers at one point in time.
- Automatic decisions with no person reviewing the list.

## Training data

IBM Telco Customer Churn sample (IBM sample data for educational use): 7,043 customers, one row per customer, 19 input columns. `Churn` is "the customer left within the last month"; 26.5% did. 5,634 customers trained the model (80%, stratified, seed 42). The table has no dates, so we cannot say how old the behavior is. `TotalCharges` is blank for 11 customers with tenure 0; the median fills them.

## Evaluation data

The other 1,409 customers (20%), held out from training, threshold choice and model choice, and scored once.

## Metrics

ROC AUC 0.842 (ranking). Brier score 0.1381 (honesty of the probabilities; 0.195 for a model that gives everyone the base rate). At the 0.278 threshold: 564 customers flagged, recall 0.781, precision 0.518, cost 63.31 dollars per customer against 69.41 at the 0.5 default.

## Results by group

Results are similar for women and men (recall 0.772 and 0.790; ROC AUC 0.838 and 0.847). Senior customers churn more (0.441) and the model flags 0.671 of them; its ranking is weaker for them (ROC AUC 0.784). The largest gap is by contract: the model flags no two-year customer and finds 0.222 of the one-year churners. Churn is rare on longer contracts (0.027 and 0.120), but the churners there are invisible to this list. If those customers matter, they need a separate rule or model.

## Caveats and recommendations

- The model describes association, not cause. Month-to-month customers are flagged more often; that does not show that moving them to a longer contract would keep them. Test such a change on a random half of a group first.
- `tenure`, `MonthlyCharges` and `TotalCharges` move together, so do not read their coefficients one at a time.
- `gender` and `SeniorCitizen` are inputs. Results by group are close for women and men, but the retention team should decide whether a protected attribute belongs in a model that chooses who gets an offer; dropping it does not remove its proxies (D01 5.2), so re-check the group table after any change.
- Monitor monthly: the PSI of the score distribution and of the churn rate against the same month a year earlier, and recall on the customers whose outcome is known. Retrain when PSI passes 0.25, or when recall falls clearly below the 0.781 measured here (6.3).
- Not a high-risk use under the EU AI Act Annex III as written (retention offers). Using the same score for creditworthiness would be.
