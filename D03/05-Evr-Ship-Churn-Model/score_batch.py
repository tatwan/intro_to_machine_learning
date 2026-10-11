"""Score a file of customers with the saved churn pipeline (D03 6.1 and 6.2): batch inference.

    python score_batch.py customers.csv scores.csv [--model churn_model.joblib] [--threshold 0.278]

Run it from this folder (D03/05-Evr-Ship-Churn-Model), after step 5 of ship.ipynb has saved
churn_model.joblib here.

Reads every row of the input CSV, writes customerID, churn_probability and flag to the
output CSV, and prints how many customers were flagged. Run it on a schedule (cron, Airflow)
and the scores land in a table the retention team reads each morning.
"""
import argparse

import joblib
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--model", default="churn_model.joblib")
    ap.add_argument("--threshold", type=float, default=0.278)
    a = ap.parse_args(argv)
    model = joblib.load(a.model)
    df = pd.read_csv(a.infile)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    X = df.drop(columns=[c for c in ("customerID", "Churn") if c in df.columns])
    p = model.predict_proba(X)[:, 1]
    scores = pd.DataFrame({"customerID": df["customerID"], "churn_probability": p.round(3), "flag": p >= a.threshold})
    scores.to_csv(a.outfile, index=False)
    print(f"scored {len(scores)} customers, flagged {int(scores['flag'].sum())}, wrote {a.outfile}")
    return scores


if __name__ == "__main__":
    main()
