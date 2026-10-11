# Ship the churn model: three runs, one alias, one endpoint, one batch

In this activity, you will log three churn models to MLflow, register the best one under the alias `champion`, load it back by that alias, serve it behind a `/predict` endpoint, and score a file of customers with a batch script.

**Time:** 20 minutes, typed together with the instructor · **Data:** `data/telco_churn.csv` · **Starter:** `Unsolved/ship.ipynb`; `app.py`, `score_batch.py`, `customer.json` and `customers.csv` are in this folder

## Before you start

- Your environment from D01 1.2 is active in the VS Code terminal (`source .venv/bin/activate`), and `python -c "import mlflow, fastapi, skops"` prints nothing.
- You are in the activity folder: `cd D03/05-Evr-Ship-Churn-Model`.
- One row is one customer; the pipeline is D01 5.6's, with the 2.2 threshold of 0.278.

## Steps

1. **Log three runs.** Run the first cells of `ship.ipynb`. Each run fits one model inside `mlflow.start_run(run_name=...)`, logs its 5-fold CV ROC AUC with `mlflow.log_metric("cv_auc", ...)`, and saves the fitted pipeline with `mlflow.sklearn.log_model(model, name="model", skops_trusted_types=trusted)`. `mlflow.sklearn.autolog(log_models=False)` records every setting for you.

2. **Compare the runs.** `mlflow.search_runs(order_by=["metrics.cv_auc DESC"])[["tags.mlflow.runName", "metrics.cv_auc"]]`.

3. **Register the best run and give it an alias.** `mv = mlflow.register_model(uri_of_best, "telco-churn")`, then `MlflowClient().set_registered_model_alias("telco-churn", "champion", mv.version)`.

4. **Load it by alias**, the way a service would: `champion = mlflow.sklearn.load_model("models:/telco-churn@champion")`. Score the test customers and print the ROC AUC.

5. **Save the pipeline for the service**: `joblib.dump(champion, "churn_model.joblib")`.

6. **Start the endpoint** in a second terminal: `uvicorn app:app --port 8000`. Leave it running.

7. **Call it** from the first terminal: `curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @customer.json`. `customer.json` holds test customer 4376-KFVRS in the training table's 19 columns.

8. **Break it on purpose**: delete the `tenure` line from a copy of `customer.json` and call again. Read the status code and the message.

9. **Score a batch**: `python score_batch.py customers.csv scores.csv --model churn_model.joblib --threshold 0.278`. Open `scores.csv`.

## Check yourself

| Step | What you should see |
|---|---|
| 2 | Three rows: logistic `0.846`, boosted `0.838`, forest `0.821` |
| 3 | `Created version '1' of model 'telco-churn'.` |
| 4 | Test ROC AUC `0.842`: the same model as D02 4.1 |
| 7 | `{"churn_probability": 0.045, "flag": false, "threshold": 0.278}` (a two-year contract, 72 months) |
| 8 | Status 422, `"type": "missing"`, `"loc": ["body", "tenure"]`, `"msg": "Field required"` |
| 9 | `scored 1409 customers, flagged 564`; `scores.csv` has the columns customerID, churn_probability, flag |

## If something goes wrong

- `MlflowException: ... Untrusted types found in the file: ['numpy.dtype']` at step 1: MLflow 3.16 saves scikit-learn models in the skops format by default, and skops refuses types it was not told to trust. Pass `skops_trusted_types=skops.io.get_untrusted_types(data=skops.io.dumps(model))`, as the starter does.
- `Address already in use` at step 6: another endpoint is still running. Stop it with Ctrl+C in its terminal, or use `--port 8001` and change the `curl` line.
- `mlflow ui` fails with `ModuleNotFoundError: No module named 'flask_cors'`: the course environment installs `mlflow-skinny`, which tracks runs but has no web interface. Compare runs with `search_runs` (step 2).
- Different numbers from the Check yourself table: check that the starter's split uses `stratify=y, random_state=42`.

## Bonus

Register the `class_weight="balanced"` model as version 2 and give it the alias `challenger`. Load `models:/telco-churn@challenger` and compare its test recall with the champion's. The service code does not change when an alias moves: that is the point of an alias.

## Known leaks and quirks in this dataset

- `TotalCharges` is read as text: 11 cells hold a single space, all for customers with `tenure` 0. The endpoint accepts `TotalCharges` as empty (`null`) for the same reason; the pipeline's imputer fills it.
- `customerID` is an identifier, never a feature: the batch script copies it to the output and does not pass it to the model.
- The endpoint receives raw columns, so the preprocessing must live inside the saved pipeline. A model saved without its `ColumnTransformer` would need the caller to one-hot encode, and one spelling difference would break it.

## References

IBM Telco Customer Churn, https://github.com/IBM/telco-customer-churn-on-icp4d, IBM sample data for educational use. Course copy taken 2026-10-02. Numbers from `D03/reference/08_ship.py`; `app.py` and `score_batch.py` are in `D03/reference/`. MLflow 3.16.1 (`mlflow-skinny`), FastAPI 0.142.2, uvicorn 0.54.0, skops 0.16.0 (`tools/requirements.txt`).
