"""Supplied data, splitting, and reporting helpers for Lab 08.

These functions do not fit boosting models, select models, or access test scores.
Read the docstrings when a notebook prompt uses a helper.
"""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, log_loss,
    precision_score, recall_score,
)
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 486
DATA_SHA256 = "d11af5a43aeb5a3661060f1eddc5c63ec4b4657ffe0a8018ba5aa6b260b4c97a"
TYPE_NAMES = (
    "bug", "dark", "dragon", "electric", "fairy", "fighting", "fire",
    "flying", "ghost", "grass", "ground", "ice", "normal", "poison",
    "psychic", "rock", "steel", "water",
)
BATTLE_STATS = ("hp", "attack", "defense", "sp_attack", "sp_defense", "speed")
LEGENDARY_FEATURES = (*BATTLE_STATS, "height_m", "weight_kg")
DROP_ALWAYS = (
    "abilities", "japanese_name", "name", "pokedex_number", "classfication",
    "base_total",
)


def load_data(path="data/pokemon.csv"):
    """Read the unmodified local snapshot after verifying its exact bytes."""
    path = Path(path)
    if sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("The dataset differs from the supplied snapshot; restore data/pokemon.csv.")
    return pd.read_csv(path, encoding="utf-8")


def make_features(raw, task):
    """Apply deterministic, documented feature rules; learn no statistics.

    Legendary: use the six battle statistics, height, and weight.
    Type: exclude both type columns, is_legendary, and every against_* column.
    Convert capture_rate to numeric after the shared row filters.
    Both boosters receive identical pandas categorical columns and numeric NaNs.
    """
    if task not in {"legendary", "type"}:
        raise ValueError("task must be 'legendary' or 'type'.")
    if task == "legendary":
        return raw.loc[:, list(LEGENDARY_FEATURES)].copy()
    drop = set(DROP_ALWAYS) | {"is_legendary"}
    drop |= {"type1", "type2"}
    drop |= {column for column in raw if column.startswith("against_")}
    X = raw.drop(columns=sorted(drop)).copy()
    if "capture_rate" in X:
        X["capture_rate"] = pd.to_numeric(X["capture_rate"])
    if "generation" in X:
        values = X["generation"].astype("string").fillna("unknown")
        # XGBoost 3.1 expects object-backed category names, not StringDtype.
        X["generation"] = pd.Categorical(values.astype(object))
    return X


def categorical_columns(X):
    """Names to pass as CatBoost's cat_features; XGBoost uses category dtypes."""
    return X.select_dtypes(include="category").columns.tolist()


def make_splits(y, *, seed=RANDOM_STATE):
    """Return disjoint row-index arrays, approximately 60/20/20 per class.

    Each class contributes max(1, round(0.2*n_class)) rows to validation and
    the same number to test. The rest train the models. Thus a class with
    only three observations contributes exactly one to each partition.
    Stratification uses labels only; feature preprocessing stays within training.
    """
    if not y.index.is_unique or y.isna().any():
        raise ValueError("Labels must have unique row indices and no missing values.")
    if y.value_counts().min() < 3:
        raise ValueError("Each class needs at least three observations for this protocol.")
    rng = np.random.default_rng(seed)
    parts = {name: [] for name in ("train", "validation", "test")}
    for label in sorted(y.unique()):
        indices = rng.permutation(y.index[y.eq(label)].to_numpy())
        n_holdout = max(1, round(0.2 * len(indices)))
        parts["validation"].extend(indices[:n_holdout])
        parts["test"].extend(indices[n_holdout:2 * n_holdout])
        parts["train"].extend(indices[2 * n_holdout:])
    return {name: rng.permutation(indices) for name, indices in parts.items()}


def split_counts(y, splits):
    """Show class support in each partition; this reports no model performance."""
    return pd.DataFrame({name: y.loc[indices].value_counts() for name, indices in splits.items()}).fillna(0).astype(int).sort_index()


def baseline_preprocessor(X):
    """Unfitted median imputation and one-hot encoding for the shallow tree.

    Put this inside a Pipeline so all learned transformations fit on training
    rows only. The boosters instead handle numeric NaNs and categories natively.
    """
    cats = categorical_columns(X)
    nums = [column for column in X if column not in cats]
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), nums),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
    ])


def binary_metrics(y, probability, *, threshold=0.5):
    """Unweighted metrics and detection counts; class 1 is legendary."""
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    return pd.Series({
        "accuracy": accuracy_score(y, prediction),
        "average_precision": average_precision_score(y, probability),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "log_loss": log_loss(y, probability, labels=[0, 1]),
        "legendary_found": int(np.sum((np.asarray(y) == 1) & (prediction == 1))),
        "false_alarms": int(np.sum((np.asarray(y) == 0) & (prediction == 1))),
        "missed_legendary": int(np.sum((np.asarray(y) == 1) & (prediction == 0))),
    })


def case_table(raw, y, probability, *, threshold=0.5):
    """Join held-out probabilities to names and sort by individual log loss.

    y's index controls which rows are shown. Losses are unweighted so hard
    cases can be compared independently of the training class weights.
    """
    probability = np.asarray(probability, dtype=float)
    true_probability = np.where(y.to_numpy() == 1, probability, 1 - probability)
    result = raw.loc[y.index, ["name", "type1", "type2", "is_legendary"]].copy()
    result["p_legendary"] = probability
    result["predicted_legendary"] = (probability >= threshold).astype(int)
    result["p_true_class"] = true_probability
    result["log_loss"] = -np.log(np.clip(true_probability, 1e-15, 1.0))
    result["correct"] = result["predicted_legendary"].to_numpy() == y.to_numpy()
    return result.sort_values("log_loss", ascending=False, kind="stable")


def stage_probabilities(model, X, rounds=None):
    """Binary P(legendary) at checkpoints; columns are numbers of rounds.

    XGBoost's explicit iteration_range overrides automatic best-iteration
    prediction. CatBoost uses staged_predict_proba. A fixed-length diagnostic
    fit with no early stopping is needed to inspect all 300 rounds.
    """
    is_xgboost = hasattr(model, "get_booster")
    count = model.get_booster().num_boosted_rounds() if is_xgboost else model.tree_count_
    if rounds is None:
        rounds = sorted({1, count, *range(10, count + 1, 10)})
    rounds = sorted({int(t) for t in rounds if 1 <= t <= count})
    if not rounds:
        raise ValueError("No requested round lies inside the fitted model.")
    values = {}
    if is_xgboost:
        for t in rounds:
            values[t] = model.predict_proba(X, iteration_range=(0, t))[:, 1]
    else:
        wanted = set(rounds)
        for t, probability in enumerate(model.staged_predict_proba(X, ntree_end=max(rounds), thread_count=2), start=1):
            if t in wanted:
                values[t] = probability[:, 1]
    return pd.DataFrame(values, index=X.index)


def loss_curve(model, X_train, y_train, X_validation, y_validation):
    """Compute comparable, unweighted binary log loss at model checkpoints."""
    train = stage_probabilities(model, X_train)
    validation = stage_probabilities(model, X_validation, rounds=train.columns)
    return pd.DataFrame([
        {"round": t,
         "training_log_loss": log_loss(y_train, train[t], labels=[0, 1]),
         "validation_log_loss": log_loss(y_validation, validation[t], labels=[0, 1])}
        for t in train
    ])
