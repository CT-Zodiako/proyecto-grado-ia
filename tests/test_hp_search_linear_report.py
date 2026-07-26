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

import pytest

from scripts.hp_search_linear_report import (
    ALPHA_GRID,
    L1_RATIO_GRID,
    load_split,
    read_baseline_metrics,
    snapshot_artifacts,
    verify_baseline,
    main,
)

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
def test_read_baseline_metrics_matches_metrics_json_on_disk():
    on_disk = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert read_baseline_metrics() == on_disk
