# RNA-folding-predictor

## Introduction

**RNA-folding-predictor** predicts the **chemical reactivity profile of an RNA** from its nucleotide sequence, using the [Stanford Ribonanza RNA Folding](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding) Kaggle dataset. The project compares simple baselines (means, Ridge regression with a sliding window of nucleotides) with a deep learning model (Transformer in Keras, optionally fed with base-pair probabilities).

**Why reactivity?** An RNA is a chain of nucleotides (`A`, `C`, `G`, `U`) that folds in space. A chemical reagent mostly "marks" the nucleotides that are flexible and accessible (unpaired) and much less the ones that are protected (paired with another nucleotide). Each nucleotide therefore gets a number, its **reactivity**: high means probably unpaired, low means probably paired. The goal is to predict this reactivity at each position.

**Metric:** MAE (mean absolute error). As in the competition, targets and predictions are clipped to [0, 1] before computing the error.

## Setup environment

We use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage dependencies and the project environment.

Clone the GitHub repository:

```sh
git clone https://github.com/InesZenati/RNA-folding-predictor.git
cd RNA-folding-predictor
```

Sync dependencies:

```sh
uv sync
```

This installs numpy, pandas, matplotlib and scikit-learn, which is enough for **all the scripts** and for the exploration notebook.

To run the Transformer notebook (heavier install because of TensorFlow) and open notebooks with Jupyter:

```sh
uv sync --group deep --group notebooks
```

| Group | Content | Needed for |
|---|---|---|
| *(default)* | numpy, pandas, matplotlib, scikit-learn | scripts, exploration |
| `deep` | tensorflow, keras, keras-hub | `notebook/transformers_bpp.ipynb` |
| `notebooks` | jupyterlab, ipykernel | opening the notebooks |

To add a dependency later: `uv add <package>` (or `uv add --group deep <package>`). Commit `pyproject.toml` and `uv.lock` afterwards.

## Data

The data are **not** included in the repository (too large). A Kaggle account is needed, and the rules of the competition must be accepted.

### Training file

Download `train_data_QUICK_START.csv` from the *Data* tab of the competition and put it in a `data/` folder at the root of the repository (the folder is ignored by Git):

```
data/train_data_QUICK_START.csv
```

Each sequence appears on **2 rows**, one per experiment type:

| Column | Description |
|---|---|
| `sequence_id` | Identifier of the sequence |
| `sequence` | Nucleotide sequence (`A`, `C`, `G`, `U`), up to 206 positions |
| `experiment_type` | `DMS_MaP` or `2A3_MaP`: the two chemical reagents, i.e. two different measurements of the same sequence |
| `dataset_name` | Name of the original sub-dataset |
| `reactivity_0001` … `reactivity_0206` | **Values to predict**: one reactivity per position. `NaN` when the position was not measured (for example after the end of the sequence) |
| `reactivity_error_0001` … `reactivity_error_0206` | Uncertainty of each measurement (the larger, the less reliable) |

Many values are missing: this is normal and is studied in `scripts/exploration.py`. The full competition file also has `SN_filter` and `signal_to_noise` columns (signal quality); they are absent from the `QUICK_START` file, so the scripts use the measurement error as a noise filter instead.

### Base pair probabilities (BPP) files

Only needed for the BPP part of the Transformer notebook. For each sequence there is a text file `<sequence_id>.txt` in which each line is `i j p`: the probability `p` that nucleotide `i` is paired with nucleotide `j`. The notebook sums them into one value per position and uses it as an additional input.

The complete archive (`bpp_full.zip`) is far too large to be fully decompressed, so **only the files of the 2000 sequences used by the notebook are extracted directly from the zip**. These 2000 sequences are the first 2000 `sequence_id` in **alphabetical order** among the sequences that have both experiments (not necessarily the beginning of the CSV file). The script below applies the same selection and copies the files flat into `notebook/bpp_2000/`, whatever the sub-folder they are stored in inside the archive.

Put `bpp_full.zip` at the root of the repository, save the script as `extract_bpp.py` and run:

```sh
uv run extract_bpp.py
```

```python
import zipfile
from pathlib import Path
import pandas as pd

ZIP_PATH = "bpp_full.zip"
CSV_PATH = "data/train_data_QUICK_START.csv"
OUT_DIR = Path("notebook/bpp_2000")
N_SEQUENCES = 2000

# Same selection as the notebook: sequences with both experiments,
# sorted by sequence_id, first N_SEQUENCES.
df = pd.read_csv(CSV_PATH, usecols=["sequence_id", "experiment_type"])
ids_dms = set(df[df["experiment_type"] == "DMS_MaP"]["sequence_id"])
ids_2a3 = set(df[df["experiment_type"] == "2A3_MaP"]["sequence_id"])
ids = sorted(ids_dms & ids_2a3)[:N_SEQUENCES]

OUT_DIR.mkdir(parents=True, exist_ok=True)
missing = 0

with zipfile.ZipFile(ZIP_PATH) as z:
    # Index: file name without extension -> full path inside the zip
    index = {}
    for name in z.namelist():
        if name.endswith(".txt"):
            index[Path(name).stem] = name

    for sid in ids:
        if sid in index:
            with z.open(index[sid]) as src:
                (OUT_DIR / f"{sid}.txt").write_bytes(src.read())
        else:
            missing = missing + 1

print(len(ids) - missing, "files extracted,", missing, "missing")
```

Only the requested files are read from the archive. If the files in your zip are not `.txt`, adapt the `name.endswith(".txt")` test. `unzip -l bpp_full.zip | head` shows how the archive is organized. Do not commit `notebook/bpp_2000/`.

## Usage

### Scripts

Always run the scripts **from the root of the repository**, with `uv run`:

```sh
uv run scripts/<script>.py
```

| Script | Description | Output |
|---|---|---|
| `exploration.py` | Full exploratory analysis: dimensions, unique sequences, experiments, sequence length distribution, reactivity profile of one sequence, missing values (by position, by experiment) | PNG figures in `FIGURES_DIR` |
| `analyse_donnee.py` | Quick statistics read by chunks of 10,000 rows: rows per experiment, percentage of missing values, sequence lengths | Terminal |
| `verification_modele.py` | Sanity check: predict the mean (or the median) reactivity at position 27 on `2A3_MaP` | Terminal |
| `modele_1.py` | Position 28: global mean, mean per nucleotide, mean per 3-nucleotide motif, then Ridge regression (with and without interactions). Train / validation split **by sequence** | Terminal |
| `Model_sur_petit_ech.py` | One Ridge model **per position** (`DMS_MaP`). 70 / 15 / 15 train / validation / test split, window size and `alpha` chosen on validation, comparison plots | `graphiques/` |
| `modele_globale_RIDGE.py` | Most complete script: compares three strategies (`2A3_MaP` only, `DMS_MaP` only, global model with the experiment type as a feature). Window size and `alpha` tuned on validation, a single pass on test, MAE per position and global | `graphiques_comparaison/` |

`exploration.py` accepts a file path:

```sh
uv run scripts/exploration.py -f data/train_data_QUICK_START.csv
```

Common principles of the Ridge scripts:

- **Window:** the reactivity of a nucleotide is predicted from the nucleotide and its neighbors (for example 7 nucleotides centered on it), one-hot encoded.
- **Split by sequence:** a sequence is never in both train and test sets (otherwise scores would be artificially good).
- **Clipping:** predictions and targets are clipped to [0, 1].

The main parameters (number of rows read `N_LIGNES`, window and `alpha` grids, noise thresholds...) are at the top of each script. For a first try of `modele_globale_RIDGE.py`, set `N_LIGNES = 20000` instead of `None`: the full file takes a long time.

### Notebooks

The notebooks can be run **locally with uv** or **on Google Colab** (see below).

**Locally:**

```sh
uv sync --group deep --group notebooks
uv run jupyter lab
```

Then open a notebook in `notebook/`. Relative paths are computed from the notebook folder.

| Notebook | Description |
|---|---|
| `data_exploration.ipynb` | Graphical exploration of the `QUICK_START` file (does not need the `deep` group) |
| `transformers_bpp.ipynb` | Keras Transformer split into functions (loading and encoding, targets, model creation, training, curves, evaluation, prediction on a random sequence). Each nucleotide is encoded as an integer, goes through an embedding with positional encoding, then through several attention blocks. The model predicts **two values per position** (DMS and 2A3). The loss is a **masked** MAE: positions without measurement (`NaN`) are ignored. The second half of the notebook adds the base-pair probabilities (BPP) as an extra input. By default, it uses the **first 2000 sequences** |

**On Google Colab** (no local install, free GPU available):

1. Upload the notebook to [Google Colab](https://colab.research.google.com) (*File > Upload notebook*) or open it directly from GitHub.
2. Put the data on your Google Drive: `train_data_QUICK_START.csv` and, for the BPP part, the `bpp_2000/` folder (extracted as described in the *Data* section).
3. Mount Google Drive in the first cell:

   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```

4. Replace the data paths in the notebook by the Drive ones, for example:

   ```python
   PATH = "/content/drive/MyDrive/train_data_QUICK_START.csv"
   ```

   (this is the path already used in `data_exploration.ipynb`). In `transformers_bpp.ipynb`, do the same for `train_data_QUICK_START.csv` and `bpp_2000/`.
5. For `transformers_bpp.ipynb`, install keras-hub in a first cell, since it is usually not preinstalled on Colab (TensorFlow and Keras normally are):

   ```python
   !pip install -q keras-hub
   ```

   To use a GPU: *Runtime > Change runtime type*.

## Project structure

```
RNA-folding-predictor/
├── notebook/
│   ├── data_exploration.ipynb      # Graphical exploration of the data
│   └── transformers_bpp.ipynb      # Transformer model, without then with BPP
├── scripts/
│   ├── exploration.py              # Full exploratory analysis + figures
│   ├── analyse_donnee.py           # Quick statistics by chunks of rows
│   ├── verification_modele.py      # Mean vs median baseline (position 27)
│   ├── modele_1.py                 # Baselines + Ridge on position 28
│   ├── Model_sur_petit_ech.py      # Ridge on every position (DMS_MaP)
│   └── modele_globale_RIDGE.py     # 2A3 only / DMS only / global model comparison
├── pyproject.toml                  # Project metadata and dependencies (managed with uv)
└── uv.lock                         # Locked dependency versions
```

`data/` (training file), `graphiques/` and `graphiques_comparaison/` (results) are created by the user or by the scripts and should not be committed.

## Output

| Script | Folder | Content |
|---|---|---|
| `exploration.py` | `FIGURES_DIR` (set in the script) | Length distribution, reactivity profile, missing-value maps |
| `Model_sur_petit_ech.py` | `graphiques/` | Model comparison, true vs predicted, error distribution |
| `modele_globale_RIDGE.py` | `graphiques_comparaison/` | `resume_comparaison.csv` (MAE per strategy), `resultats_comparaison.csv` (detail per position), `grille_reglages_validation.csv` (window × alpha trials) and PNG plots |

To interpret a MAE, always compare it with the "global mean" baseline: it is the score to beat.

## Notes

**Data paths.** Not all scripts look for the data at the same place. Check this table before running a script:

| File | Where it reads the data | What to do |
|---|---|---|
| `Model_sur_petit_ech.py`, `modele_globale_RIDGE.py` | `data/train_data_QUICK_START.csv` | nothing |
| `modele_1.py`, `verification_modele.py`, `analyse_donnee.py` | `OLD/train_data.csv` | put the CSV in `OLD/` **or** replace the path in the script |
| `exploration.py` | `DATA_DIR` and `FIGURES_DIR` at the top of the script (lines 10 and 12) | replace with your own folders, or pass the CSV with `-f` |
| `notebook/data_exploration.ipynb` | `PATH` in the first cell (Google Drive / Colab path) | keep it on Google Colab, or replace with the local path |
| `notebook/transformers_bpp.ipynb` | `train_data_QUICK_START.csv` and `bpp_2000/`, **relative to the notebook folder** | put the CSV in `notebook/` (or adapt the path); `bpp_2000/` is created in `notebook/` by the extraction script. On Colab, use the Google Drive paths |

**Common problems.**

| Message | Probable cause | Fix |
|---|---|---|
| `FileNotFoundError` on the CSV | File not at the path expected by the script, or script not run from the repository root | See the table above; run from `RNA-folding-predictor/` |
| `PermissionError` / folder error with `exploration.py` | `FIGURES_DIR` points to a folder that does not exist on your machine | Edit `DATA_DIR` and `FIGURES_DIR` at the top of the script |
| `Dossier bpp_2000/ manquant` | BPP files not extracted, or not in the notebook folder | See *Base pair probabilities (BPP) files* |
| `ModuleNotFoundError: keras` (or `keras_hub`) | `deep` group not installed | `uv sync --group deep --group notebooks` |
| Script is very slow | The whole file is read | Reduce `N_LIGNES` at the top of the script |