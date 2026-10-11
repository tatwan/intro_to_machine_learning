# Introduction to Machine Learning · labs

The notebooks, datasets and activity briefs for the three-day Introduction to Machine Learning course. You clone this repo once, on the first morning, and work in it for all three days.

Each activity folder holds:

- `README.md`: the brief. What you do, the data, the steps, and a **Check yourself** table with the values a correct answer prints.
- `Unsolved/`: the starter notebook. Setup the brief calls "the starter" is already filled in; each step has a markdown cell and an empty cell for your code.
- `Solved/`: the solution, with its outputs. The instructor releases it at the activity's review, so leave it closed until then. Its last cell prints every Check yourself value.
- `Sample/`: sample answers for the group and partner discussions, which have no notebook.

## Set up (D01, step 1.2)

You need Python 3.13, [uv](https://docs.astral.sh/uv/), git, and VS Code with the Python and Jupyter extensions. The course VM has all four.

```bash
git clone <labs-repo-url> ml-labs
cd ml-labs
uv venv --python 3.13
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

The install takes a few minutes the first time. Then, in VS Code:

1. **File > Open Folder**, and choose `ml-labs`.
2. Open `D01/01-Evr-Environment-Check/Unsolved/smoke.ipynb`.
3. **Select Kernel > Python Environments**, and choose the one that ends in `.venv`.
4. **Run All**, and compare the printed versions with that folder's README.

`requirements.txt` pins every package to the version the course was built and checked with (scikit-learn 1.9.1, pandas 3.0.6). A different version can print different numbers from the ones in the Check yourself tables.

## How the notebooks find the data

Every notebook reads its data with a path from the repo root, such as `data/penguins.csv`. `.vscode/settings.json` tells VS Code to run notebooks from the repo root, which is why you open the whole `ml-labs` folder rather than a single notebook.

In JupyterLab or another editor, a notebook starts in its own folder, three levels down. Run `%cd ../../..` in the notebook's first cell, or start from VS Code. `FileNotFoundError: ... data/...` almost always means this.

## The activities

The step is the lesson plan's step label, which the slides show on every activity. Durations are the time in class.

| Day | Folder | Step | What you do | Minutes |
|---|---|---|---|---:|
| D01 | `01-Evr-Environment-Check` | 1.2 | Clone, install, run the smoke notebook | 25 |
| D01 | `02-Grp-Six-Problems` | 2.2 | Decide which kind of ML fits six requests | 15 |
| D01 | `03-Grp-Place-Three-Products` | 3.3 | Place three products on the cadence and inference grid | 15 |
| D01 | `04-Evr-First-Model` | 4.2 | First model end to end on penguins | 15 |
| D01 | `05-Stu-Leakage-Hunt` | 5.4 | Find the leak in the bank-marketing model | 15 |
| D01 | `06-Evr-Preprocessing-Pipeline` | 5.7 | One `ColumnTransformer` for the telco table | 10 |
| D01 | `07-Lab-Ames-EDA` | 6.2 | Exploratory analysis of house sales | 50 |
| D02 | `01-Par-Audit-AI-Pipeline` | 1.2 | Review a churn model a chat assistant wrote | 20 |
| D02 | `02-Stu-Ames-Regression` | 2.2 | Linear regression on house prices, errors in dollars | 20 |
| D02 | `03-Stu-Regularization` | 3.3 | Linear, Ridge and Lasso with learning curves | 20 |
| D02 | `04-Stu-Churn-Classifier` | 4.3 | Churn classifier and the metric that fits the decision | 25 |
| D02 | `05-Evr-Tree-Depth` | 5.2 | A decision tree, drawn, at three depths | 10 |
| D02 | `06-Lab-Model-Comparison` | 6.3 | Four models on one credit-default split | 35 |
| D03 | `01-Stu-CV-and-Threshold` | 2.3 | Compare models with cross-validation, then choose a threshold | 25 |
| D03 | `02-Stu-Tuned-vs-Untuned` | 3.2 | Is hyperparameter search worth it? | 20 |
| D03 | `03-Stu-Explain-Churn` | 4.2 | Permutation importance, partial dependence and SHAP | 25 |
| D03 | `04-Lab-Customer-Segmentation` | 5.2 | RFM and k-means on a year of online sales | 25 |
| D03 | `05-Evr-Ship-Churn-Model` | 6.2 | MLflow runs, an alias, an endpoint and a batch script | 20 |
| D03 | `06-Par-Model-Card` | 6.4 | Write the model card for the churn model | 15 |

In the folder names, `Evr` is an Everyone Do (we type it together), `Stu` a Student Do (on your own), `Par` a Partner Do, `Grp` a Group Do, and `Lab` a longer activity with checkpoints C1, C2 and C3.

D03's `05-Evr-Ship-Churn-Model` needs two terminals: one runs the endpoint, the other calls it. It writes `mlflow.db`, `mlruns/`, `churn_model.joblib` and `scores.csv` into its own folder; `.gitignore` keeps them out of git.

## If something goes wrong

- **A version differs from the smoke notebook's README**: the notebook is not on the `.venv` kernel. Pick it again (set-up step 3).
- **`ModuleNotFoundError`**: same cause, or the install in set-up did not finish. Run `uv pip install -r requirements.txt` again with `.venv` active.
- **`FileNotFoundError` for a file in `data/`**: the notebook is not running from the repo root (see "How the notebooks find the data").
- **A value differs from Check yourself**: compare your split first (`test_size`, `stratify`, `random_state=42`), then the step's code. A different value is information, not failure.
- **`Address already in use` in D03 6.2**: an endpoint is still running. Stop it with Ctrl+C in its terminal.

## Updating during the course

The instructor may fix a brief between days. Before each day, from the `ml-labs` folder:

```bash
git pull
```

If git refuses because you changed a notebook, keep your work under a new name (`cp file.ipynb file-mine.ipynb`), then `git checkout -- file.ipynb` and pull again.

## Data

Seven tables in `data/`. `data/README.md` gives each file's source, license, rows, the copy used here and its known leaks and quirks.

| File | Dataset | License |
|---|---|---|
| `penguins.csv` | Palmer Penguins (Horst, Hill and Gorman) | CC0 |
| `telco_churn.csv` | IBM Telco Customer Churn | IBM sample data for educational use |
| `bank_marketing.csv` | Bank Marketing (Moro, Rita and Cortez), UCI 222 | CC BY 4.0 |
| `ames.csv` | Ames Housing (De Cock, 2011), OpenML 42165 | Released for teaching |
| `credit_default.csv` | Default of Credit Card Clients (Yeh), UCI 350 | CC BY 4.0 |
| `bike_hourly.csv` | Bike Sharing (Fanaee-T), UCI 275 | CC BY 4.0 |
| `online_retail_ii.csv` | Online Retail (Chen), UCI 352, under Online Retail II's column names | CC BY 4.0 |

The four UCI tables are used under the Creative Commons Attribution 4.0 license, https://creativecommons.org/licenses/by/4.0/. `data/README.md` lists the changes made to each copy.
