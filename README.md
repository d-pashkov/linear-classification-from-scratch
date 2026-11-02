# Linear Classification from Scratch

An executed notebook studying linear classifiers, empirical risk minimization, gradient-based optimization, and L1/L2 regularization on the Wisconsin Diagnostic Breast Cancer dataset. The notebook implements the optimization and prediction routines directly with NumPy, then compares a gradient-trained classifier with a Ridge baseline.

## Dataset and evaluation

The dataset contains 569 observations and 30 numeric features. The label convention in this project is `+1` for malignant and `-1` for benign. `data.csv` is included so the notebook can run without downloading data. Its feature values and original class labels match `sklearn.datasets.load_breast_cancer()`; the target column is recoded to the convention above.

The saved experiment uses a stratified 80/20 split with `random_state=42`. Hyperparameters are selected with five-fold cross-validation on the training partition, and scaling is fitted within each training fold. On the held-out 114 examples, the saved results report:

| Model | Correct | Accuracy |
| --- | ---: | ---: |
| Ridge | 109/114 | 95.61% |
| Gradient classifier | 112/114 | 98.25% |

These are results from this fixed split, not a generalization guarantee. The notebook also evaluates learning curves over ten additional fixed splits. The saved CSV files and figures are outputs from the executed notebook.

## Run

Python 3.12 is the version used for the saved run.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install jupyter
jupyter lab lab1.ipynb
```

Run the notebook cells from top to bottom. Existing figures and result tables can be inspected without rerunning training under `figures/` and `results/`.

## Contents

- `lab1.ipynb` — implementation, numerical checks, cross-validation, and experiments.
- `data.csv` — local copy of the dataset with the project's label encoding.
- `figures/` — four saved experiment plots.
- `results/` — saved metrics, histories, coefficient paths, and diagnostics.
- `requirements.txt` — dependency versions from the original notebook environment.

## Contribution and provenance

This is a personal coursework export. The notebook contains the student implementation and experiment work; the assignment was completed within an ITMO course context. The original course prompt and local planning/defense notes are not included. The source notebook's own requirements and structure informed the work; no claim is made that the assignment scaffold or dataset is original to this repository.

The included CSV was checked against scikit-learn's built-in Breast Cancer dataset. If `data.csv` is absent, the notebook contains a fallback that creates it using scikit-learn. See `PROVENANCE.md` for the source and export details.

