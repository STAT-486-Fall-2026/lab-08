# Lab 08: Boosting Pokémon

<!-- STAT 486 · Fall 2026 -->

## Objective

Fit XGBoost and CatBoost for an initial comparison, then choose one library to study class weighting, learning rates, individual prediction trajectories, and primary-type classification. Supplied decision trees provide the baselines.

## Before you begin

You need Git, a GitHub account, and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed on your computer.

## Set up your submission repository

### 1. Create your private submission repository

On [`STAT-486-Fall-2026/lab-08`](https://github.com/STAT-486-Fall-2026/lab-08), select **Use this template** and then **Create a new repository**. Set the owner to the `STAT-486-Fall-2026-Labs` organization, name the repository exactly:

```text
lab-08-<netid>
```

For example, a student whose NetID is `jdoe42` must create `lab-08-jdoe42`. Use your institutional NetID even if it differs from your GitHub username. Make the repository **private**, then create it. Do not fork the starter repository.

### 2. Clone your repository

Copy the HTTPS URL for **your repository**, then run the following commands. Replace the NetID placeholder.

```bash
git clone https://github.com/STAT-486-Fall-2026-Labs/lab-08-<netid>.git
cd lab-08-<netid>
```

### 3. Create and lock your environment

The starter repository includes a lab-specific `pyproject.toml`. From the repository directory, run:

```bash
uv sync
```

This creates your local `.venv` and generates `uv.lock`. Keep the supplied `pyproject.toml` unchanged unless the lab directions explicitly say otherwise.

### 4. Complete the lab

Open `lab-08.ipynb` in your preferred editor and configure it to use the Python interpreter or notebook kernel in this repository's `.venv`.

Begin with the Pokémon description and compact variable guide. Run the supplied preparation and baseline cells, then complete the required TODOs and nine written responses; no prior Pokémon knowledge is needed.

The local dataset is `data/pokemon.csv`; [data/README.md](data/README.md) records its source, checksum, variable meanings, and known quirks. `lab_helpers.py` supplies cleaning, splitting, and reporting support. Keep `data/pokemon.csv`, `lab_helpers.py`, and the supplied environment files unchanged. No Kaggle account, GPU, data download, or additional package installation inside the notebook is needed.

`uv sync` installs **XGBoost** and **CatBoost**, along with the other required libraries. The setup cell reports installed package versions. The lab uses their CPU implementations with two threads. If Windows reports a missing XGBoost DLL, consult the [official installation guide](https://xgboost.readthedocs.io/en/release_3.1.0/install.html) for the Microsoft Visual C++ runtime requirement.

Use the supplied task-specific splits. Keep early stopping, class weights, and model selection within training/validation; evaluate the test sets only at the end. Replace the commented TODOs with executable code. Restart the kernel and run the completed notebook from top to bottom before submission. Include readable outputs and labeled plots. Do not submit trained-model files or a separate answer document.

### 5. Commit and push your work

```bash
git add lab-08.ipynb
git commit -m "Complete Lab 08"
git push
```

## Submission

Paste the root URL of your repository into the Canvas Lab 08 submission textbox:

```text
https://github.com/STAT-486-Fall-2026-Labs/lab-08-<netid>
```

Do not open a pull request and do not upload the notebook file to Canvas.
