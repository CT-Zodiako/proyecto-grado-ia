"""Tests for the PR1 baseline-reproduction gate in
``scripts/hp_search_linear_report.py``.

PR1 scope only: constants/grid asserts, ``load_split``,
``read_baseline_metrics``, ``snapshot_artifacts``, ``fit_candidate``,
``verify_baseline`` and a minimal ``main()`` that runs only the baseline
gate. The grid search, coefficient extraction and report rendering are PR2
and are not exercised here.

Conventions mirrored from ``tests/test_diagnostics.py``: real artifacts are
optional in this environment, so integration tests are ``skipif``-guarded on
the presence of the source CSV and ``metrics.json``.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import Lasso

from scripts.hp_search_linear_report import (
    ALPHA_GRID,
    L1_RATIO_GRID,
    MATERIALITY_MIN_VAL_IMPROVEMENT,
    CandidateResult,
    load_split,
    read_baseline_metrics,
    snapshot_artifacts,
    verify_baseline,
    select_best_per_model,
    assess_materiality,
    compare_dominant_features,
    extract_nonzero_coefficients,
    rank_overall,
    run_grid,
    render_report,
    main,
)
from app.dashboard.diagnostics import DOMINANT_FEATURES
from mejorar_modelo import build_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DATASET_PATH = ARTIFACTS_DIR / "medicina_features_2020_2025.csv"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

REAL_DATA_MISSING = not (DATASET_PATH.exists() and METRICS_PATH.exists())
REAL_DATA_SKIP_REASON = (
    "artifacts/medicina_features_2020_2025.csv or artifacts/metrics.json "
    "not present in this environment"
)


# ---------------------------------------------------------------------------
# Structural: grid constants contain today's production config
# ---------------------------------------------------------------------------


def test_alpha_grid_contains_config():
    # Structural/single-output check: the grid must be an executable superset
    # of the config already running in production (mejorar_modelo.py:336/350).
    # Triangulation skipped: single possible output (exact float membership),
    # no branching logic to exercise with a second case.
    assert 1.0 in ALPHA_GRID
    assert 0.5 in L1_RATIO_GRID


# ---------------------------------------------------------------------------
# load_split — real temporal split reconstruction
# ---------------------------------------------------------------------------


@pytest.mark.skipif(REAL_DATA_MISSING, reason=REAL_DATA_SKIP_REASON)
def test_load_split_returns_expected_row_counts():
    # NOTE: the design artifact estimated 184/71/67 without shell access to
    # verify. Running mejorar_modelo.py's exact predicates standalone against
    # the live CSV produces 174/62/64 — confirmed correct independently by
    # test_verify_baseline_passes_on_real_metrics matching all 18 live
    # metrics.json values to 1e-6.
    split = load_split()
    assert split.train_count == 174
    assert split.val_count == 62
    assert split.test_count == 64
    # Non-trivial: the split actually carries rows, not empty frames.
    assert len(split.X_train) == 174
    assert len(split.X_val) == 62
    assert len(split.X_test) == 64


# ---------------------------------------------------------------------------
# verify_baseline — the core PR1 deliverable (fail-fast provenance gate)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(REAL_DATA_MISSING, reason=REAL_DATA_SKIP_REASON)
def test_verify_baseline_halts_on_mismatch(capsys):
    split = load_split()
    baseline_metrics = read_baseline_metrics()
    tampered = copy.deepcopy(baseline_metrics)
    for result in tampered["model_results"]:
        if result["modelo"] == "Lasso":
            # Deliberately wrong expected value — proves the gate actually gates.
            result["validacion"]["MAE"] = 999.0

    with pytest.raises(SystemExit) as exc_info:
        verify_baseline(split, tampered)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "PROVENANCE BROKEN" in captured.err
    assert "Lasso" in captured.err
    assert "999" in captured.err


@pytest.mark.skipif(REAL_DATA_MISSING, reason=REAL_DATA_SKIP_REASON)
def test_verify_baseline_passes_on_real_metrics(capsys):
    split = load_split()
    baseline_metrics = read_baseline_metrics()

    # Must NOT raise — real re-fit reproduces the live artifacts/metrics.json
    # baseline within BASELINE_TOLERANCE for Ridge, Lasso and ElasticNet.
    verify_baseline(split, baseline_metrics)

    captured = capsys.readouterr()
    assert "PASS" in captured.err


# ---------------------------------------------------------------------------
# main() — zero side effects, correct exit contract
# ---------------------------------------------------------------------------


@pytest.mark.skipif(REAL_DATA_MISSING, reason=REAL_DATA_SKIP_REASON)
def test_main_does_not_modify_artifacts_and_exits_zero():
    before = snapshot_artifacts()
    exit_code = main()
    after = snapshot_artifacts()

    assert exit_code == 0
    assert before == after


@pytest.mark.skipif(REAL_DATA_MISSING, reason=REAL_DATA_SKIP_REASON)
def test_main_stdout_contains_full_report(capsys):
    # Integration: proves the full PR2 wiring (grid -> selection ->
    # coefficients -> verdicts -> render_report) actually runs end-to-end
    # and reaches stdout as the fixed 8-section markdown report.
    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    for i in range(1, 9):
        assert f"## {i}." in captured.out
    assert "diagnostic" in captured.out.lower()


@pytest.mark.skipif(REAL_DATA_MISSING, reason=REAL_DATA_SKIP_REASON)
def test_read_baseline_metrics_matches_metrics_json_on_disk():
    on_disk = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert read_baseline_metrics() == on_disk


# ---------------------------------------------------------------------------
# select_best_per_model — PR2: strict-< ascending-alpha tie-break
# ---------------------------------------------------------------------------


def _candidate(model, alpha, val_mae, converged=True, l1_ratio=None):
    params = {"alpha": alpha}
    if l1_ratio is not None:
        params["l1_ratio"] = l1_ratio
    return CandidateResult(
        model=model,
        params=params,
        val_mae=val_mae,
        val_rmse=val_mae * 1.2,
        val_r2=0.8,
        test_mae=val_mae * 1.1,
        test_rmse=val_mae * 1.3,
        test_r2=0.75,
        converged=converged,
    )


def test_select_best_per_model_tie_break():
    # Plateau: alpha=10 and alpha=100 both saturate to the same val_mae
    # (all-zero coefficients). Strict "<" tie-break must keep the SMALLEST
    # alpha of the plateau, not the largest, and must report the tie count.
    results = [
        _candidate("Lasso", 0.001, 5.0),
        _candidate("Lasso", 0.01, 4.5),
        _candidate("Lasso", 10.0, 4.0),  # plateau start (smallest alpha)
        _candidate("Lasso", 100.0, 4.0),  # plateau continues, same val_mae
        _candidate("Lasso", 1000.0, 4.0),  # plateau continues, same val_mae
    ]
    best = select_best_per_model(results)
    assert best["Lasso"]["best"].params["alpha"] == 10.0
    assert best["Lasso"]["ties"] == 2


def test_select_best_per_model_picks_min_val_mae_per_model():
    # Triangulation: two different models in the same call, each must get
    # its OWN best candidate — proves grouping-by-model is real, not a
    # single global minimum.
    results = [
        _candidate("Ridge", 0.001, 6.0),
        _candidate("Ridge", 1.0, 5.0),
        _candidate("ElasticNet", 0.001, 4.8),
        _candidate("ElasticNet", 1.0, 4.2),
    ]
    best = select_best_per_model(results)
    assert best["Ridge"]["best"].val_mae == 5.0
    assert best["Ridge"]["best"].params["alpha"] == 1.0
    assert best["Ridge"]["ties"] == 0
    assert best["ElasticNet"]["best"].val_mae == 4.2
    assert best["ElasticNet"]["ties"] == 0


# ---------------------------------------------------------------------------
# assess_materiality — pre-registered threshold truth table (Decisions 9, 13)
# ---------------------------------------------------------------------------

BASELINE_VAL_MAE = 4.011104118746772
BASELINE_TEST_MAE = 3.8486189543610996


def test_assess_materiality_material_when_both_conditions_met():
    # improvement = 4.0111 - 3.90 = 0.111 >= 0.05, and test MAE improves too.
    winner = _candidate("Lasso", 0.5, val_mae=3.90)
    winner = CandidateResult(**{**winner.__dict__, "test_mae": 3.80})
    verdict = assess_materiality(winner, BASELINE_VAL_MAE, BASELINE_TEST_MAE)
    assert verdict["material"] is True
    assert verdict["improvement"] == pytest.approx(0.111104118746772, abs=1e-9)


def test_assess_materiality_not_material_when_test_worse():
    # Validation improves enough, but test MAE regresses vs baseline -> NOT material.
    winner = _candidate("Lasso", 0.5, val_mae=3.90)
    winner = CandidateResult(**{**winner.__dict__, "test_mae": 4.20})
    verdict = assess_materiality(winner, BASELINE_VAL_MAE, BASELINE_TEST_MAE)
    assert verdict["material"] is False


def test_assess_materiality_not_material_below_threshold():
    # improvement = 4.0111 - 3.98 = 0.0311 < 0.05 -> NOT material, even
    # though test MAE would be fine.
    winner = _candidate("Lasso", 0.8, val_mae=3.98)
    winner = CandidateResult(**{**winner.__dict__, "test_mae": 3.70})
    verdict = assess_materiality(winner, BASELINE_VAL_MAE, BASELINE_TEST_MAE)
    assert verdict["material"] is False


def test_assess_materiality_forced_not_material_when_non_converged():
    # Decision 9: even if the numbers would otherwise qualify, a
    # non-converged winner must NEVER be reported material.
    winner = _candidate("Lasso", 0.001, val_mae=3.50, converged=False)
    winner = CandidateResult(**{**winner.__dict__, "test_mae": 3.40})
    verdict = assess_materiality(winner, BASELINE_VAL_MAE, BASELINE_TEST_MAE)
    assert verdict["material"] is False
    assert "non-converged" in verdict["reason"].lower()


# ---------------------------------------------------------------------------
# compare_dominant_features — first-class NOT APPLICABLE outcomes (§1b)
# ---------------------------------------------------------------------------


def test_compare_dominant_features_unchanged_when_set_matches():
    result = compare_dominant_features("Lasso", list(DOMINANT_FEATURES), [])
    assert result["verdict"] == "UNCHANGED"
    assert result["added"] == []
    assert result["dropped"] == []


def test_compare_dominant_features_changed_when_set_differs():
    # Triangulation: drop one dominant feature, add a new one -> CHANGED,
    # with the exact added/dropped feature names reported.
    nonzero = ["maximo_historico", "promedio_global_anterior", "otra_feature"]
    result = compare_dominant_features("Lasso", nonzero, [])
    assert result["verdict"] == "CHANGED"
    assert result["added"] == ["otra_feature"]
    assert result["dropped"] == sorted(
        set(DOMINANT_FEATURES) - {"maximo_historico", "promedio_global_anterior"}
    )


def test_compare_dominant_features_not_applicable_for_ridge():
    # Ridge (L2) never zeroes coefficients -> the comparison is structurally
    # inapplicable, must be reported as a labeled outcome, not raise.
    result = compare_dominant_features("Ridge", list(range(105)), [])
    assert result["verdict"] == "NOT APPLICABLE — dense L2 (Ridge)"


def test_compare_dominant_features_not_applicable_for_nonlinear_winner():
    # A non-linear overall winner (e.g. LightGBM) has no .coef_ at all.
    result = compare_dominant_features("LightGBM", None, None)
    assert result["verdict"] == "NOT APPLICABLE — non-linear winner, no .coef_"


def test_compare_dominant_features_flags_nonzero_categorical():
    result = compare_dominant_features(
        "Lasso", list(DOMINANT_FEATURES), ["cat__NOMBRE_DEPARTAMENTO_Antioquia"]
    )
    assert result["cat_flag"] == ["cat__NOMBRE_DEPARTAMENTO_Antioquia"]


# ---------------------------------------------------------------------------
# extract_nonzero_coefficients — synthetic fitted pipeline (Decision 6)
# ---------------------------------------------------------------------------


def _synthetic_frame():
    numeric_features = ["x1", "x2"]
    categorical_features = ["cat1"]
    X = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x2": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "cat1": ["a", "b", "a", "b", "a", "b"],
        }
    )
    y = pd.Series([3.0, 6.0, 9.0, 12.0, 15.0, 18.0])  # y = 3 * x1 exactly
    return X, y, numeric_features, categorical_features


def test_extract_nonzero_coefficients_all_zero_at_saturating_alpha():
    X, y, numeric_features, categorical_features = _synthetic_frame()
    pipeline = build_pipeline(
        Lasso(alpha=1e9, max_iter=10000), numeric_features, categorical_features
    )
    pipeline.fit(X, y)
    result = extract_nonzero_coefficients(pipeline, numeric_features)
    assert result["nonzero_numeric"] == []
    assert result["nonzero_categorical"] == []
    assert result["n_nonzero"] == 0


def test_extract_nonzero_coefficients_maps_names_at_low_alpha():
    # Triangulation: a DIFFERENT alpha exercises the non-empty branch and
    # proves the "num__" prefix is stripped back to the original feature
    # name (x1 fully determines y, so it must survive shrinkage).
    X, y, numeric_features, categorical_features = _synthetic_frame()
    pipeline = build_pipeline(
        Lasso(alpha=0.001, max_iter=10000), numeric_features, categorical_features
    )
    pipeline.fit(X, y)
    result = extract_nonzero_coefficients(pipeline, numeric_features)
    assert "x1" in result["nonzero_numeric"]
    assert result["n_nonzero"] >= 1


# ---------------------------------------------------------------------------
# rank_overall — 3 tuned + 6 read-only (never retrained) non-linear models
# ---------------------------------------------------------------------------


def _baseline_metrics_fixture():
    return {
        "model_results": [
            {"modelo": "Random Forest", "validacion": {"MAE": 4.5}, "test": {"MAE": 5.0}},
            {"modelo": "XGBoost", "validacion": {"MAE": 4.3}, "test": {"MAE": 4.9}},
            {"modelo": "LightGBM", "validacion": {"MAE": 4.2}, "test": {"MAE": 4.2}},
            {"modelo": "CatBoost", "validacion": {"MAE": 4.6}, "test": {"MAE": 4.4}},
            {"modelo": "HistGradientBoosting", "validacion": {"MAE": 4.3}, "test": {"MAE": 4.5}},
            {"modelo": "Ridge", "validacion": {"MAE": 5.1}, "test": {"MAE": 6.0}},
            {"modelo": "Lasso", "validacion": {"MAE": 4.0}, "test": {"MAE": 3.8}},
            {"modelo": "ElasticNet", "validacion": {"MAE": 4.2}, "test": {"MAE": 4.1}},
            {"modelo": "KNN", "validacion": {"MAE": 5.3}, "test": {"MAE": 5.3}},
        ],
        "best_model_by_validation_mae": "Lasso",
    }


def test_rank_overall_no_winner_change_when_tuning_does_not_beat_baseline():
    tuned = {
        "Ridge": _candidate("Ridge", 1.0, val_mae=5.1),
        "Lasso": _candidate("Lasso", 1.0, val_mae=4.0),  # matches today's baseline
        "ElasticNet": _candidate("ElasticNet", 1.0, val_mae=4.2),
    }
    result = rank_overall(tuned, _baseline_metrics_fixture())
    assert result["new_winner"] == "Lasso"
    assert result["winner_changed"] is False
    assert result["rows"][0]["model"] == "Lasso"
    assert len(result["rows"]) == 9


def test_rank_overall_detects_winner_change():
    # Triangulation: tuned ElasticNet beats every read-only non-linear model
    # AND the tuned Lasso -> the overall winner changes.
    tuned = {
        "Ridge": _candidate("Ridge", 1.0, val_mae=5.1),
        "Lasso": _candidate("Lasso", 1.0, val_mae=4.0),
        "ElasticNet": _candidate("ElasticNet", 0.05, val_mae=3.5),
    }
    result = rank_overall(tuned, _baseline_metrics_fixture())
    assert result["new_winner"] == "ElasticNet"
    assert result["winner_changed"] is True
    assert result["previous_winner"] == "Lasso"
    assert result["margin"] == pytest.approx(4.0 - 3.5)


# ---------------------------------------------------------------------------
# run_grid — 225 real fits against the real temporal split (integration)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(REAL_DATA_MISSING, reason=REAL_DATA_SKIP_REASON)
def test_run_grid_produces_expected_candidate_count_and_best_pipelines():
    split = load_split()
    results, best_pipelines = run_grid(split)

    # 25 (Ridge) + 25 (Lasso) + 25*7 (ElasticNet) = 225 real fits.
    assert len(results) == len(ALPHA_GRID) * 2 + len(ALPHA_GRID) * len(L1_RATIO_GRID)
    assert len(results) == 225
    assert set(best_pipelines.keys()) == {"Ridge", "Lasso", "ElasticNet"}
    # Every candidate is a REAL fit against real validation data: MAE must be
    # a positive, finite number for all 225 (proves the fits actually ran).
    assert all(r.val_mae > 0.0 for r in results)
    assert {r.model for r in results} == {"Ridge", "Lasso", "ElasticNet"}


# ---------------------------------------------------------------------------
# render_report — fixed 8-section markdown (hand-built fixtures, no I/O)
# ---------------------------------------------------------------------------


def _render_report_fixtures():
    all_results = [
        _candidate("Ridge", 1.0, val_mae=5.1387293006180155),
        _candidate("Lasso", 1.0, val_mae=4.011104118746772),
        _candidate("Lasso", 0.001, val_mae=6.0, converged=False),
        _candidate("ElasticNet", 1.0, val_mae=4.1656694124445925),
    ]
    best_per_model = select_best_per_model(all_results)
    baseline_metrics = _baseline_metrics_fixture()
    tuned_bests = {model: info["best"] for model, info in best_per_model.items()}
    coef_result = {
        "nonzero_numeric": list(DOMINANT_FEATURES),
        "nonzero_categorical": [],
        "n_nonzero": 4,
    }
    dominant_verdict = compare_dominant_features(
        "Lasso", coef_result["nonzero_numeric"], coef_result["nonzero_categorical"]
    )
    overall_rank = rank_overall(tuned_bests, baseline_metrics)
    materiality_verdict = assess_materiality(
        tuned_bests["Lasso"], BASELINE_VAL_MAE, BASELINE_TEST_MAE
    )
    provenance = {
        "train_count": 174,
        "val_count": 62,
        "test_count": 64,
        "numeric_features": ["x1", "x2"],
        "categorical_features": ["cat1"],
        "artifacts_unchanged": True,
    }
    return (
        provenance,
        all_results,
        best_per_model,
        baseline_metrics,
        coef_result,
        dominant_verdict,
        overall_rank,
        materiality_verdict,
    )


def test_render_report_contains_all_8_fixed_sections():
    report = render_report(*_render_report_fixtures())
    for i in range(1, 9):
        assert f"## {i}." in report, f"missing section header for section {i}"


def test_render_report_labels_test_mae_as_diagnostic_only():
    # Spec scenario: test MAE must be labeled diagnostic-only, never cited
    # as a selection criterion.
    report = render_report(*_render_report_fixtures())
    assert "diagnostic" in report.lower()
    assert "float64" not in report  # no unformatted numpy float repr leak


def test_render_report_flags_non_converged_candidate():
    # Triangulation: a fixture with a non-converged candidate must surface
    # it in section 8, never hide it.
    report = render_report(*_render_report_fixtures())
    assert "non-converged" in report.lower() or "NON-CONVERGED" in report
