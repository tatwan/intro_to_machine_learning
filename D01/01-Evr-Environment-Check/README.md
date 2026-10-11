# Environment check

In this activity, you will clone the labs repository on your course VM, install the course packages with `uv`, and run a smoke notebook that proves every later notebook will run.

**Time:** 25 minutes · **Data:** `data/penguins.csv` · **Starter:** `Unsolved/smoke.ipynb`

We do this together: the instructor types each command, you type the same one, and nobody moves on until the checkpoint shows.

## Instructions

1. On your course VM, open **VS Code**, then open a terminal with **Terminal > New Terminal**. Check the tools the VM image should already have:

    ```bash
    python3 --version      # Python 3.13.x
    uv --version
    git --version
    ```

    If one of them prints `command not found`, raise your hand: a helper fixes it before the next step.

2. Clone the labs repository and move into it. The instructor gives the URL in the group channel.

    ```bash
    git clone <labs-repo-url> ml-labs
    cd ml-labs
    ```

3. Create the course environment and install the pinned packages. The install takes a few minutes the first time.

    ```bash
    uv venv --python 3.13
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```

4. Open the folder in VS Code with **File > Open Folder** and choose `ml-labs`. Open `D01/01-Evr-Environment-Check/Unsolved/smoke.ipynb`.

5. Click **Select Kernel** (top right of the notebook), then **Python Environments**, then the one that ends in `.venv`. If VS Code asks to install `ipykernel`, accept.

6. Click **Run All**. The notebook imports every library the course uses, prints their versions, and loads the penguins table.

## Check yourself

| Line the notebook prints | Expected |
|---|---|
| `scikit-learn` | `1.9.1` |
| `pandas` | `3.0.6` |
| `polars` | `1.44.2` |
| `numpy` | `2.5.3` |
| `penguins shape` | `(344, 8)` |
| Missing values in `bill_length_mm` | `2` |
| Missing values in `sex` | `11` |

A different version number means the kernel is not the `.venv` one. Check the kernel name in the top right corner first.

## Hint

`ModuleNotFoundError: No module named 'sklearn'` almost always means the notebook runs on the system Python. Pick the `.venv` kernel again (step 5).

`FileNotFoundError: ... data/penguins.csv` means VS Code opened the notebook without the repo folder. Use **File > Open Folder** on `ml-labs` (step 4), so notebooks run from the repo root.

## Bonus

Open a new cell and run `df.dtypes`. Which columns does pandas 3 read as `str`, and which as numbers?

## Fallback: Google Colab

Only if your VM does not work and the helpers cannot fix it in five minutes. Upload `smoke.ipynb` to Colab and run this cell first:

```python
!pip install -q numpy==2.5.3 scikit-learn==1.9.1 pandas==3.0.6 polars==1.44.2
```

Then restart the runtime, upload `penguins.csv` with the file panel, and change the path in the notebook from `data/penguins.csv` to `penguins.csv` (Colab puts uploads in its working folder). Move back to your VM at the next break.

## Known quirks in the data

`penguins.csv`: 2 rows have no measurements at all; `sex` is missing in 11 rows. Both are real gaps in the field data, not errors in the course copy. `year` is the study year, not a measurement.

## References

Palmer Penguins, Horst, Hill and Gorman, https://github.com/allisonhorst/palmerpenguins, CC0. Course copy taken 2026-10-02 (`data/penguins.csv`, sha256 starts `f2d6493fc64ad958`).
