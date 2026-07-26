"""Integration tests for the two-group sidebar navigation
(dashboard-navigation-redesign): Principal (always visible) vs Detalle
técnico (collapsed expander), and the Overview CTA that jumps to
Diagnóstico.

Uses Streamlit's AppTest headless runner, same pattern as
test_streamlit_diagnostico_page.py.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
APP_PATH = Path(__file__).resolve().parent.parent / "app" / "dashboard" / "streamlit_app.py"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS_DIR / "model.joblib").exists(),
    reason="real artifacts not present in this environment",
)

PRINCIPAL_OPTIONS = ["📊 Overview", "🩺 Diagnóstico", "🔮 Predicción"]
DETALLE_OPTIONS = ["📈 EDA", "📋 Recomendaciones", "✅ Validación", "🤖 Modelos", "🔍 Explicabilidad"]


def _fresh_app():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    assert not at.exception, f"App failed to load: {at.exception}"
    return at


def test_two_radios_exist_with_correct_grouping():
    at = _fresh_app()
    assert len(at.sidebar.radio) == 2
    assert at.sidebar.radio[0].options == PRINCIPAL_OPTIONS
    assert at.sidebar.radio[1].options == DETALLE_OPTIONS


def test_selecting_principal_clears_detalle():
    at = _fresh_app()
    at.sidebar.radio[0].set_value("🩺 Diagnóstico").run(timeout=30)
    assert not at.exception
    assert at.sidebar.radio[1].value is None


def test_selecting_detalle_clears_principal():
    at = _fresh_app()
    at.sidebar.radio[1].set_value("🤖 Modelos").run(timeout=30)
    assert not at.exception
    assert at.sidebar.radio[0].value is None


def test_overview_cta_jumps_to_diagnostico():
    at = _fresh_app()
    at.button(key="cta_start_diagnostico").click().run(timeout=30)
    assert not at.exception
    assert at.sidebar.radio[0].value == "🩺 Diagnóstico"
    assert at.sidebar.radio[1].value is None
    subheaders = [s.value for s in at.subheader]
    assert "1. ¿Qué programa querés diagnosticar?" in subheaders


def test_overview_content_preserved():
    """Regression: the navigation redesign must not remove existing
    Overview content (metrics, model comparison table, etc.)."""
    at = _fresh_app()
    metric_labels = [m.label for m in at.metric]
    assert "Período" in metric_labels
    assert "Programas" in metric_labels


@pytest.mark.parametrize("option", PRINCIPAL_OPTIONS + DETALLE_OPTIONS)
def test_every_page_still_reachable_no_exception(option):
    """All 8 original pages remain reachable via one radio or the other,
    with identical content/behavior (non-goal: no page removed/changed)."""
    at = _fresh_app()
    if option in PRINCIPAL_OPTIONS:
        at.sidebar.radio[0].set_value(option).run(timeout=30)
    else:
        at.sidebar.radio[1].set_value(option).run(timeout=30)
    assert not at.exception, f"Page {option} raised: {at.exception}"
