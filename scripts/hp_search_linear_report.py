#!/usr/bin/env python3
"""Read-only linear-model hyperparameter search + report
(``model-hyperparameter-search``, PR1 + PR2).

This module reuses ``add_new_features``, ``build_pipeline`` and
``train_and_evaluate`` from ``mejorar_modelo.py`` unmodified. It restates the
5 filter/split predicates from ``mejorar_modelo.py``'s ``main()`` (155-196)
because that logic lives inline in a function and is not importable; the
restatement is proven equivalent by ``verify_baseline`` re-fitting the exact
production baseline estimators and comparing all 18 metrics
(3 models x val/test x MAE/RMSE/R2) against the live ``artifacts/metrics.json``
within a tight tolerance.

PR1 scope: the baseline-reproduction gate (``load_split``, ``verify_baseline``).
PR2 scope (this file, extended): grid-searches ``alpha`` (Ridge/Lasso/
ElasticNet) and ``l1_ratio`` (ElasticNet) against the existing temporal
split, selects each model's best candidate strictly by validation MAE
(never test MAE), compares the tuned overall winner's non-zero coefficients
against ``DOMINANT_FEATURES``, ranks all 9 models (3 tuned + 6 read-only
from ``metrics.json``), applies the pre-registered materiality threshold,
and renders a fixed 8-section markdown report to stdout.

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
from app.dashboard.diagnostics import DOMINANT_FEATURES  # noqa: E402

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


def run_grid(split: SplitData) -> tuple[list[CandidateResult], dict]:
    """Grid-search alpha over Ridge/Lasso/ElasticNet (and l1_ratio for
    ElasticNet): 25 + 25 + 25*7 = 225 fits, reusing ``fit_candidate`` for
    every candidate.

    ``fit_candidate`` already opens a fresh ``warnings.catch_warnings()``
    block PER CALL (Decision 8) — calling it once per grid iteration is what
    makes the per-iteration re-arming real, since sklearn's warning registry
    dedups per call site across a single outer context. Progress is printed
    to stderr only, so stdout stays reserved for the final markdown report.
    """
    results: list[CandidateResult] = []
    best_pipelines: dict[str, object] = {}
    best_val_mae: dict[str, float] = {}

    total = len(ALPHA_GRID) + len(ALPHA_GRID) + len(ALPHA_GRID) * len(L1_RATIO_GRID)
    done = 0

    def _record(name: str, estimator) -> None:
        nonlocal done
        result, pipeline = fit_candidate(name, estimator, split)
        results.append(result)
        done += 1
        if name not in best_val_mae or result.val_mae < best_val_mae[name]:
            best_val_mae[name] = result.val_mae
            best_pipelines[name] = pipeline
        print(
            f"[{done}/{total}] {name} alpha={result.params.get('alpha'):.4g} "
            f"val_mae={result.val_mae:.4f} converged={result.converged}",
            file=sys.stderr,
        )

    for alpha in ALPHA_GRID:
        _record("Ridge", Ridge(alpha=alpha, random_state=42))
    for alpha in ALPHA_GRID:
        _record("Lasso", Lasso(alpha=alpha, random_state=42, max_iter=10000))
    for alpha in ALPHA_GRID:
        for l1_ratio in L1_RATIO_GRID:
            _record(
                "ElasticNet",
                ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000),
            )

    return results, best_pipelines


def select_best_per_model(results: list[CandidateResult]) -> dict[str, dict]:
    """Pick each model's best candidate by validation MAE.

    Tie-break (Decision 10): iterate candidates in ASCENDING alpha order and
    keep the first one that achieves the minimum — a strict ``<`` comparison
    so a later candidate with an *equal* val_mae never replaces an earlier
    (smaller-alpha) one. This deterministically surfaces a saturated
    plateau (large-alpha candidates collapsing to all-zero coefficients)
    instead of hiding it behind an arbitrary tie-break, and the plateau size
    is reported back as ``ties``.
    """
    grouped: dict[str, list[CandidateResult]] = {}
    for r in results:
        grouped.setdefault(r.model, []).append(r)

    output: dict[str, dict] = {}
    for model, candidates in grouped.items():
        ordered = sorted(candidates, key=lambda r: r.params["alpha"])
        best = ordered[0]
        ties = 0
        for r in ordered[1:]:
            if r.val_mae < best.val_mae:
                best = r
                ties = 0
            elif r.val_mae == best.val_mae:
                ties += 1
        output[model] = {"best": best, "ties": ties}
    return output


def assess_materiality(
    tuned_winner: CandidateResult, baseline_val_mae: float, baseline_test_mae: float
) -> dict:
    """Apply the pre-registered threshold (Decision 13): material iff
    validation MAE improves by >= ``MATERIALITY_MIN_VAL_IMPROVEMENT`` AND
    test MAE is not worse than the baseline. A non-converged winner is
    ALWAYS forced not-material (Decision 9), regardless of the numbers,
    because a candidate that did not converge cannot be trusted as a
    genuine improvement."""
    improvement = baseline_val_mae - tuned_winner.val_mae

    if not tuned_winner.converged:
        return {
            "material": False,
            "improvement": improvement,
            "reason": (
                "winner is non-converged — rerun with a higher max_iter "
                "before trusting this improvement"
            ),
        }

    meets_threshold = (
        improvement >= MATERIALITY_MIN_VAL_IMPROVEMENT
        and tuned_winner.test_mae <= baseline_test_mae
    )
    if meets_threshold:
        reason = (
            f"validation MAE improved by {improvement:.4f} "
            f"(>= {MATERIALITY_MIN_VAL_IMPROVEMENT}) and test MAE did not worsen"
        )
    elif improvement < MATERIALITY_MIN_VAL_IMPROVEMENT:
        reason = (
            f"validation MAE improvement {improvement:.4f} is below the "
            f"{MATERIALITY_MIN_VAL_IMPROVEMENT} threshold"
        )
    else:
        reason = "validation MAE improved but test MAE regressed vs. baseline"

    return {"material": meets_threshold, "improvement": improvement, "reason": reason}


def compare_dominant_features(
    winner_model: str,
    nonzero_numeric: list[str] | None,
    nonzero_categorical: list[str] | None,
) -> dict:
    """Compare the tuned winner's non-zero numeric feature set against
    ``DOMINANT_FEATURES`` (§1b). Two branches are structurally
    NOT APPLICABLE and MUST be reported as first-class outcomes rather than
    raising: Ridge (L2 never produces exact zeros — its non-zero set is
    effectively all coefficients) and a non-linear overall winner (no
    ``.coef_`` at all). Any non-zero one-hot ``cat__`` coefficient is
    flagged separately since ``diagnostics.py`` cannot represent categorical
    contributions at all."""
    if winner_model == "Ridge":
        return {
            "verdict": "NOT APPLICABLE — dense L2 (Ridge)",
            "added": [],
            "dropped": [],
            "cat_flag": [],
        }
    if winner_model not in ("Ridge", "Lasso", "ElasticNet"):
        return {
            "verdict": "NOT APPLICABLE — non-linear winner, no .coef_",
            "added": [],
            "dropped": [],
            "cat_flag": [],
        }

    nonzero_set = set(nonzero_numeric or [])
    dominant_set = set(DOMINANT_FEATURES)
    added = sorted(nonzero_set - dominant_set)
    dropped = sorted(dominant_set - nonzero_set)
    verdict = "UNCHANGED" if not added and not dropped else "CHANGED"
    return {
        "verdict": verdict,
        "added": added,
        "dropped": dropped,
        "cat_flag": list(nonzero_categorical or []),
    }


def extract_nonzero_coefficients(pipeline, numeric_features: list[str]) -> dict:
    """Map a fitted linear pipeline's non-zero coefficients back to feature
    names via ``ColumnTransformer.get_feature_names_out()`` (Decision 6).

    The guard assert simultaneously validates the positional assumption
    ``diagnostics.py:231`` silently depends on (that the ``num`` block
    occupies coefficient positions ``0..len(numeric_features)-1`` in the
    exact order of ``numeric_features``) — if that assumption ever breaks,
    this function fails loudly instead of silently mis-attributing
    coefficients.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    num_names = [n.removeprefix("num__") for n in feature_names if n.startswith("num__")]
    assert num_names == numeric_features, (
        "positional assumption broken: preprocessor's 'num' block does not "
        "match numeric_features order — diagnostics.py:231 depends on this"
    )

    nonzero_numeric: list[str] = []
    nonzero_categorical: list[str] = []
    for name, coef in zip(feature_names, model.coef_):
        if coef == 0.0:
            continue
        if name.startswith("num__"):
            nonzero_numeric.append(name.removeprefix("num__"))
        elif name.startswith("cat__"):
            nonzero_categorical.append(name)

    return {
        "nonzero_numeric": nonzero_numeric,
        "nonzero_categorical": nonzero_categorical,
        "n_nonzero": len(nonzero_numeric) + len(nonzero_categorical),
    }


_NON_LINEAR_MODELS = (
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "HistGradientBoosting",
    "KNN",
)


def rank_overall(tuned_bests: dict[str, CandidateResult], baseline_metrics: dict) -> dict:
    """Rank the 3 tuned linear candidates against the other 6 models' EXISTING
    validation MAE read straight from ``metrics.json`` — those 6 are never
    retrained. Reports whether the overall winner among all 9 models changes
    versus ``best_model_by_validation_mae`` and by what margin over the
    runner-up."""
    rows = []
    for entry in baseline_metrics["model_results"]:
        if entry["modelo"] in _NON_LINEAR_MODELS:
            rows.append(
                {
                    "model": entry["modelo"],
                    "val_mae": entry["validacion"]["MAE"],
                    "source": "read-only (not retrained)",
                }
            )
    for model, result in tuned_bests.items():
        rows.append({"model": model, "val_mae": result.val_mae, "source": "tuned"})

    rows.sort(key=lambda r: r["val_mae"])

    previous_winner = baseline_metrics["best_model_by_validation_mae"]
    new_winner = rows[0]["model"]
    margin = rows[1]["val_mae"] - rows[0]["val_mae"] if len(rows) > 1 else 0.0

    return {
        "rows": rows,
        "previous_winner": previous_winner,
        "new_winner": new_winner,
        "winner_changed": new_winner != previous_winner,
        "margin": margin,
    }


def render_report(
    provenance: dict,
    all_results: list[CandidateResult],
    best_per_model: dict[str, dict],
    baseline_metrics: dict,
    coef_result: dict,
    dominant_verdict: dict,
    overall_rank: dict,
    materiality_verdict: dict,
) -> str:
    """Render the fixed 8-section markdown report (Decision 7 — the ONLY
    stdout writer; progress/warnings go to stderr). Section order is fixed
    so reruns are diffable. Test MAE is always labeled a read-only
    diagnostic — the winner/verdict text never cites it as a selection
    criterion."""
    baseline_by_model = {r["modelo"]: r for r in baseline_metrics["model_results"]}
    lines: list[str] = []

    lines.append("# Model Hyperparameter Search Report — model-hyperparameter-search Phase 1\n")

    # 1. Provenance
    lines.append("## 1. Provenance")
    lines.append(
        f"- Split row counts: train={provenance['train_count']}, "
        f"val={provenance['val_count']}, test={provenance['test_count']}"
    )
    lines.append(
        f"- Numeric features: {len(provenance['numeric_features'])}, "
        f"categorical features: {len(provenance['categorical_features'])}"
    )
    lines.append(
        f"- Artifacts unchanged: {'YES' if provenance['artifacts_unchanged'] else 'NO'}"
    )
    lines.append("")

    # 2. Baseline reproduction
    lines.append("## 2. Baseline Reproduction")
    lines.append(
        f"- PASS (18/18 metrics within tolerance <= {BASELINE_TOLERANCE}) "
        "— see stderr for the run-time expected/observed/delta table"
    )
    lines.append("")

    # 3. Grid
    lines.append("## 3. Grid")
    lines.append(f"- alpha grid: {len(ALPHA_GRID)} values, l1_ratio grid: {list(L1_RATIO_GRID)}")
    lines.append(
        f"- Contains today's config: YES (alpha=1.0, l1_ratio=0.5) — "
        f"total candidates: {len(all_results)}"
    )
    lines.append("")

    # 4. Per-model best
    lines.append("## 4. Per-Model Best")
    lines.append("| Model | Best params | Val MAE | Val Δ vs baseline | Test MAE (diagnostic only) | Converged | Ties |")
    lines.append("|---|---|---|---|---|---|---|")
    for model, info in best_per_model.items():
        best = info["best"]
        baseline_val_mae = baseline_by_model[model]["validacion"]["MAE"]
        delta = baseline_val_mae - best.val_mae
        params_str = ", ".join(f"{k}={v}" for k, v in best.params.items())
        converged_label = "yes" if best.converged else "NON-CONVERGED"
        lines.append(
            f"| {model} | {params_str} | {best.val_mae:.4f} | {delta:+.4f} | "
            f"{best.test_mae:.4f} (diagnostic only, not used for selection) | "
            f"{converged_label} | {info['ties']} |"
        )
    lines.append("")

    # 5. Overall ranking
    lines.append("## 5. Overall Ranking")
    lines.append("| Rank | Model | Val MAE | Source |")
    lines.append("|---|---|---|---|")
    for i, row in enumerate(overall_rank["rows"], start=1):
        lines.append(f"| {i} | {row['model']} | {row['val_mae']:.4f} | {row['source']} |")
    lines.append(
        f"- Overall winner changes: "
        f"{'YES' if overall_rank['winner_changed'] else 'NO'} "
        f"({overall_rank['previous_winner']} -> {overall_rank['new_winner']}, "
        f"margin {overall_rank['margin']:.4f})"
    )
    lines.append("")

    # 6. Coefficient verdict
    lines.append("## 6. Coefficient Verdict")
    lines.append(f"- Verdict: {dominant_verdict['verdict']}")
    lines.append(f"- Added vs DOMINANT_FEATURES: {dominant_verdict['added'] or 'none'}")
    lines.append(f"- Dropped vs DOMINANT_FEATURES: {dominant_verdict['dropped'] or 'none'}")
    if dominant_verdict["cat_flag"]:
        lines.append(f"- FLAG: non-zero categorical coefficients: {dominant_verdict['cat_flag']}")
    lines.append("")

    # 7. Verdict (materiality)
    lines.append("## 7. Verdict")
    lines.append(
        f"- Material: {'YES' if materiality_verdict['material'] else 'NO'} "
        f"— {materiality_verdict['reason']} "
        f"(improvement={materiality_verdict['improvement']:.4f}, "
        f"threshold={MATERIALITY_MIN_VAL_IMPROVEMENT})"
    )
    lines.append("")

    # 8. Non-converged candidates
    lines.append("## 8. Non-Converged Candidates")
    non_converged = [r for r in all_results if not r.converged]
    if non_converged:
        lines.append(f"- {len(non_converged)} NON-CONVERGED candidate(s) — never silently selected:")
        for r in non_converged:
            params_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
            lines.append(f"  - {r.model} ({params_str}): val_mae={r.val_mae:.4f}")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def main() -> int:
    """Full entrypoint: load the split, run the baseline gate (halts on
    provenance mismatch), grid-search Ridge/Lasso/ElasticNet, select each
    model's best candidate, extract the overall winner's coefficients,
    compare against ``DOMINANT_FEATURES``, rank all 9 models, apply the
    materiality threshold, and print the final markdown report to stdout.
    Zero writes to ``artifacts/`` — checked via a before/after mtime
    snapshot."""
    before_snapshot = snapshot_artifacts()

    split = load_split()
    baseline_metrics = read_baseline_metrics()
    print(
        f"Split loaded: train={split.train_count} val={split.val_count} test={split.test_count}",
        file=sys.stderr,
    )

    verify_baseline(split, baseline_metrics)

    print("Starting grid search (225 candidates)...", file=sys.stderr)
    all_results, best_pipelines = run_grid(split)
    best_per_model = select_best_per_model(all_results)
    tuned_bests = {model: info["best"] for model, info in best_per_model.items()}

    overall_rank = rank_overall(tuned_bests, baseline_metrics)
    overall_winner = overall_rank["new_winner"]

    if overall_winner in best_pipelines:
        coef_result = extract_nonzero_coefficients(
            best_pipelines[overall_winner], split.numeric_features
        )
        nonzero_numeric = coef_result["nonzero_numeric"]
        nonzero_categorical = coef_result["nonzero_categorical"]
    else:
        coef_result = {"nonzero_numeric": [], "nonzero_categorical": [], "n_nonzero": 0}
        nonzero_numeric = None
        nonzero_categorical = None

    dominant_verdict = compare_dominant_features(
        overall_winner, nonzero_numeric, nonzero_categorical
    )

    baseline_lasso = next(
        r for r in baseline_metrics["model_results"] if r["modelo"] == "Lasso"
    )
    tuned_linear_winner_name = min(tuned_bests, key=lambda m: tuned_bests[m].val_mae)
    materiality_verdict = assess_materiality(
        tuned_bests[tuned_linear_winner_name],
        baseline_lasso["validacion"]["MAE"],
        baseline_lasso["test"]["MAE"],
    )

    after_snapshot = snapshot_artifacts()
    artifacts_unchanged = before_snapshot == after_snapshot

    provenance = {
        "train_count": split.train_count,
        "val_count": split.val_count,
        "test_count": split.test_count,
        "numeric_features": split.numeric_features,
        "categorical_features": split.categorical_features,
        "artifacts_unchanged": artifacts_unchanged,
    }

    report = render_report(
        provenance,
        all_results,
        best_per_model,
        baseline_metrics,
        coef_result,
        dominant_verdict,
        overall_rank,
        materiality_verdict,
    )
    print(report)

    return 0 if artifacts_unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
