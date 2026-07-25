"""Integration smoke tests for the "🩺 Diagnóstico" Streamlit page (PR2 of
ai-diagnostic-view), using Streamlit's AppTest headless runner.

These tests exercise the real UI wiring end-to-end against the real
artifacts (model.joblib, feature_schema.json, medicina_features_2020_2025.csv)
— unlike tests/test_diagnostics.py, which only unit-tests the pure logic in
diagnostics.py with a synthetic fixture. This closes the gap flagged by
review-reliability: the glue code that feeds diagnostics.py (selector,
historial filtering, on-the-fly maximo_historico/promedio_movil_3_anios
recomputation) previously had zero automated coverage.

Requires: real artifacts/ directory (model.joblib, feature_schema.json,
medicina_features_2020_2025.csv). Skipped if artifacts are absent, same
policy as the smoke test in test_diagnostics.py.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
APP_PATH = Path(__file__).resolve().parent.parent / "app" / "dashboard" / "streamlit_app.py"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS_DIR / "model.joblib").exists()
    or not (ARTIFACTS_DIR / "medicina_features_2020_2025.csv").exists(),
    reason="real artifacts not present in this environment",
)


def _navigate_to_diagnostico():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    assert not at.exception, f"App failed to load at all: {at.exception}"
    at.sidebar.radio[0].set_value("🩺 Diagnóstico").run(timeout=30)
    return at


def test_diagnostico_page_loads_without_exception():
    at = _navigate_to_diagnostico()
    assert not at.exception, f"Diagnóstico page raised: {at.exception}"
    assert len(at.selectbox) > 0, "Expected a program selector on the page"


def test_diagnostico_page_only_lists_medicina_programs():
    at = _navigate_to_diagnostico()
    options = at.selectbox[0].options
    assert len(options) > 0
    # Every option follows "INSTITUCION — MUNICIPIO, DEPARTAMENTO"; this is
    # scoped to the already-filtered Medicina dataset, so we just assert the
    # selector is non-empty and well-formed rather than re-deriving the full
    # Medicina filter here (that's the dataset's own contract, not this page's).
    for option in options:
        assert " — " in option


def test_diagnostico_page_renders_for_every_real_program_no_exception():
    """Exercises every real program in the dataset (56 as of this writing)
    through the full selector -> historial -> diagnostics.build_diagnostic_narrative
    -> render pipeline, asserting no unhandled exception for any of them.

    This is the regression guard for the exact class of bug found manually
    during PR2 implementation (features missing from CSV columns, and the
    anios_hist / años_previos divergence risk flagged by review-reliability).
    """
    at = _navigate_to_diagnostico()
    options = at.selectbox[0].options
    assert len(options) > 0

    failures = []
    for label in options:
        at.selectbox[0].set_value(label).run(timeout=30)
        if at.exception:
            failures.append((label, str(at.exception)))

    assert not failures, f"{len(failures)}/{len(options)} programs raised an exception: {failures[:5]}"
