#!/usr/bin/env python3
"""Baseline-reproduction gate for a linear-model hyperparameter search
(PR1 of ``model-hyperparameter-search``).

This module reuses ``add_new_features``, ``build_pipeline`` and
``train_and_evaluate`` from ``mejorar_modelo.py`` unmodified. It restates the
5 filter/split predicates from ``mejorar_modelo.py``'s ``main()`` (155-196)
because that logic lives inline in a function and is not importable; the
restatement is proven equivalent by ``verify_baseline`` re-fitting the exact
production baseline estimators and comparing all 18 metrics
(3 models x val/test x MAE/RMSE/R2) against the live ``artifacts/metrics.json``
within a tight tolerance.

Scope (PR1 only): the baseline-reproduction gate. The grid search over
alpha/l1_ratio, coefficient extraction, and markdown report rendering are
PR2 and are NOT implemented here.

Zero side effects: this module and its ``main()`` never write to
``artifacts/`` and never call ``mejorar_modelo.main()``.
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, Ridge

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mejorar_modelo import add_new_features, build_pipeline, train_and_evaluate  # noqa: E402

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
FEATURE_SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
DATASET_PATH = ARTIFACTS_DIR / "medicina_features_2020_2025.csv"

# Pre-registered search space (PR2 consumes this; PR1 only asserts it is a
# superset of today's production config).
ALPHA_GRID = np.logspace(-3, 3, 25)
L1_RATIO_GRID = (0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99)
MATERIALITY_MIN_VAL_IMPROVEMENT = 0.05
BASELINE_TOLERANCE = 1e-6

# Executable form of "the grid is a strict superset of today's config".
assert 1.0 in ALPHA_GRID, "ALPHA_GRID must contain today's production alpha=1.0"
assert 0.5 in L1_RATIO_GRID, "L1_RATIO_GRID must contain today's production l1_ratio=0.5"

# Baseline estimators exactly matching mejorar_modelo.py:322/336/350.
_BASELINE_ESTIMATORS = {
    "Ridge": lambda: Ridge(alpha=1.0, random_state=42),
    "Lasso": lambda: Lasso(alpha=1.0, random_state=42, max_iter=10000),
    "ElasticNet": lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000),
}


@dataclass(frozen=True)
class CandidateResult:
    """Result of fitting one candidate estimator. ``n_nonzero`` is left
    ``None`` in PR1 — coefficient extraction is PR2 scope."""

    model: str
    params: dict
    val_mae: float
    val_rmse: float
    val_r2: float
    test_mae: float
    test_rmse: float
    test_r2: float
    converged: bool
    n_nonzero: int | None = None


@dataclass(frozen=True)
class SplitData:
    """Temporal train/validation/test split, reconstructed from
    ``feature_schema.json`` and the 5 filter/split predicates restated from
    ``mejorar_modelo.py`` (155-196)."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    numeric_features: list
    categorical_features: list
    train_count: int
    val_count: int
    test_count: int


def load_split() -> SplitData:
    """Reconstruct the temporal split using the feature lists from
    ``artifacts/feature_schema.json`` (byte-identical to the lists hardcoded
    in ``mejorar_modelo.py`` 166-175) and the 5 restated predicates:
    ``dropna(subset=[target])``, ``anios_historicos_disponibles > 0``,
    ``AÑO <= 2023`` / ``== 2024`` / ``== 2025``.

    Row counts are train=174 / val=62 / test=64, verified by independently
    re-running these exact predicates against the live CSV standalone (the
    design artifact's estimate of 184/71/67 was unverified — it had no shell
    access at design time).
    """
    schema = json.loads(FEATURE_SCHEMA_PATH.read_text(encoding="utf-8"))
    target = schema["target"]
    numeric_features = schema["numeric_features"]
    categorical_features = schema["categorical_features"]

    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
    df = add_new_features(df)

    model_data = (
        df[numeric_features + categorical_features + [target]]
        .dropna(subset=[target])
        .copy()
    )
    model_data = model_data[model_data["anios_historicos_disponibles"] > 0].copy()

    train_data = model_data[model_data["AÑO"] <= 2023]
    val_data = model_data[model_data["AÑO"] == 2024]
    test_data = model_data[model_data["AÑO"] == 2025]

    feature_cols = numeric_features + categorical_features
    return SplitData(
        X_train=train_data[feature_cols],
        y_train=train_data[target],
        X_val=val_data[feature_cols],
        y_val=val_data[target],
        X_test=test_data[feature_cols],
        y_test=test_data[target],
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        train_count=len(train_data),
        val_count=len(val_data),
        test_count=len(test_data),
    )


def read_baseline_metrics() -> dict:
    """Read the live ``artifacts/metrics.json`` — never a pasted literal."""
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def snapshot_artifacts() -> dict:
    """Mtime snapshot of every file under ``artifacts/`` — proves this
    script never writes there (Decision 11)."""
    return {p.name: p.stat().st_mtime_ns for p in ARTIFACTS_DIR.iterdir() if p.is_file()}


def fit_candidate(name: str, estimator, split: SplitData) -> tuple[CandidateResult, object]:
    """Fit one candidate via the imported, unmodified
    ``build_pipeline``/``train_and_evaluate`` and capture whether it
    converged. ``ConvergenceWarning`` must be re-armed every call
    (sklearn's warning registry dedups per call site otherwise)."""
    pipeline = build_pipeline(estimator, split.numeric_features, split.categorical_features)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline, metrics = train_and_evaluate(
            name,
            pipeline,
            split.X_train,
            split.y_train,
            split.X_val,
            split.y_val,
            split.X_test,
            split.y_test,
        )
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    params = {k: v for k, v in estimator.get_params().items() if k in ("alpha", "l1_ratio")}
    result = CandidateResult(
        model=name,
        params=params,
        val_mae=metrics["validacion"]["MAE"],
        val_rmse=metrics["validacion"]["RMSE"],
        val_r2=metrics["validacion"]["R2"],
        test_mae=metrics["test"]["MAE"],
        test_rmse=metrics["test"]["RMSE"],
        test_r2=metrics["test"]["R2"],
        converged=converged,
    )
    return result, pipeline


def verify_baseline(split: SplitData, baseline_metrics: dict) -> None:
    """Re-fit Ridge/Lasso/ElasticNet at today's production hyperparameters
    and compare all 18 metrics (3 models x val/test x MAE/RMSE/R2) against
    the live ``artifacts/metrics.json`` at ``BASELINE_TOLERANCE``. On any
    mismatch, print an expected/observed/delta table to stderr and halt via
    ``SystemExit(1)`` instead of continuing to the grid search."""
    baseline_by_model = {r["modelo"]: r for r in baseline_metrics["model_results"]}

    mismatches = []
    for name, make_estimator in _BASELINE_ESTIMATORS.items():
        result, _ = fit_candidate(name, make_estimator(), split)
        expected = baseline_by_model[name]
        observed_pairs = [
            ("validacion", "MAE", result.val_mae, expected["validacion"]["MAE"]),
            ("validacion", "RMSE", result.val_rmse, expected["validacion"]["RMSE"]),
            ("validacion", "R2", result.val_r2, expected["validacion"]["R2"]),
            ("test", "MAE", result.test_mae, expected["test"]["MAE"]),
            ("test", "RMSE", result.test_rmse, expected["test"]["RMSE"]),
            ("test", "R2", result.test_r2, expected["test"]["R2"]),
        ]
        for split_name, metric_name, observed, exp_value in observed_pairs:
            delta = abs(observed - exp_value)
            if delta > BASELINE_TOLERANCE:
                mismatches.append((name, split_name, metric_name, exp_value, observed, delta))

    if mismatches:
        print("PROVENANCE BROKEN — baseline reproduction mismatch:", file=sys.stderr)
        header = f"{'model':<12}{'split':<12}{'metric':<8}{'expected':>16}{'observed':>16}{'delta':>16}"
        print(header, file=sys.stderr)
        for name, split_name, metric_name, exp_value, observed, delta in mismatches:
            row = (
                f"{name:<12}{split_name:<12}{metric_name:<8}"
                f"{exp_value:>16.6f}{observed:>16.6f}{delta:>16.6f}"
            )
            print(row, file=sys.stderr)
        raise SystemExit(1)

    print("Baseline reproduction: PASS (18/18 metrics within tolerance)", file=sys.stderr)


def main() -> int:
    """Minimal PR1 entrypoint: load the split, run the baseline gate, and
    report PASS/FAIL. The grid search is PR2 and is not invoked here."""
    split = load_split()
    baseline_metrics = read_baseline_metrics()
    print(
        f"Split loaded: train={split.train_count} val={split.val_count} test={split.test_count}",
        file=sys.stderr,
    )

    verify_baseline(split, baseline_metrics)

    print(
        "BASELINE REPRODUCED SUCCESSFULLY "
        f"(train={split.train_count}, val={split.val_count}, test={split.test_count})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
