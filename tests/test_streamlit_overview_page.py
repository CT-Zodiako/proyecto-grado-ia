"""Integration tests for the redesigned Overview page
(overview-showcase-redesign): metrics sourced live from `load_metrics()`,
4 model-family tabs, and removal of every stale "Ridge won" claim across
Overview, Validación, and Modelos.

Uses Streamlit's AppTest headless runner, same pattern as
test_streamlit_navigation.py.
"""
import json
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

FAMILY_LABELS = [
    "Regresión lineal regularizada",
    "Gradient boosting",
    "Ensemble de árboles",
    "Basado en instancias",
]


def _fresh_app():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    assert not at.exception, f"App failed to load: {at.exception}"
    return at


def _metrics():
    return json.loads((ARTIFACTS_DIR / "metrics.json").read_text(encoding="utf-8"))


def _all_rendered_text(at):
    """Collect every piece of rendered text on the current page so 'no Ridge
    claim anywhere' assertions inspect markdown, success/info/warning boxes,
    metrics, and subheaders alike."""
    chunks = []
    for md in at.markdown:
        chunks.append(md.value)
    for s in at.success:
        chunks.append(s.value)
    for i in at.info:
        chunks.append(i.value)
    for w in at.warning:
        chunks.append(w.value)
    for sh in at.subheader:
        chunks.append(sh.value)
    for m in at.metric:
        chunks.append(f"{m.label} {m.value}")
    return "\n".join(chunks)


def test_overview_renders_four_family_tabs():
    at = _fresh_app()
    assert len(at.tabs) == 4
    assert [t.label for t in at.tabs] == FAMILY_LABELS


def test_overview_winner_comes_from_metrics_json():
    at = _fresh_app()
    metrics = _metrics()
    best_name = metrics["best_model_by_validation_mae"]
    assert best_name == "Lasso"
    text = _all_rendered_text(at)
    assert f"{best_name}** fue el modelo seleccionado" in text


def test_overview_test_metrics_come_from_metrics_json():
    at = _fresh_app()
    metrics = _metrics()
    best_test = metrics["best_test_metrics"]
    text = _all_rendered_text(at)
    assert f"{best_test['MAE']:.3f}" in text or f"{best_test['MAE']:.1f}" in text


def test_overview_has_no_stale_ridge_winner_claim():
    at = _fresh_app()
    text = _all_rendered_text(at)
    assert "seleccionado final es **Ridge**" not in text
    assert "Ridge v2" not in text
    assert "¿Por qué ganó Ridge?" not in text


def test_validacion_page_has_no_stale_ridge_winner_claim():
    at = _fresh_app()
    at.sidebar.radio[1].set_value("✅ Validación").run(timeout=30)
    assert not at.exception
    text = _all_rendered_text(at)
    assert "¿Por qué ganó Ridge?" not in text


def test_modelos_page_has_no_stale_ridge_winner_claim():
    at = _fresh_app()
    at.sidebar.radio[1].set_value("🤖 Modelos").run(timeout=30)
    assert not at.exception
    text = _all_rendered_text(at)
    metrics = _metrics()
    best_name = metrics["best_model_by_validation_mae"]
    assert "¿Por qué ganó Ridge?" not in text
    assert f"¿Por qué ganó {best_name}?" in text


def test_overview_cta_start_diagnostico_still_present():
    at = _fresh_app()
    assert at.button(key="cta_start_diagnostico") is not None


def test_overview_cta_open_modelos_navigates_to_modelos():
    at = _fresh_app()
    at.button(key="cta_open_modelos").click().run(timeout=30)
    assert not at.exception
    assert at.sidebar.radio[1].value == "🤖 Modelos"
    assert at.sidebar.radio[0].value is None


def test_overview_cta_open_explicacion_navigates_to_explicacion():
    at = _fresh_app()
    at.button(key="cta_open_explicacion").click().run(timeout=30)
    assert not at.exception
    assert at.sidebar.radio[0].value == "🔍 Explicación"
    assert at.sidebar.radio[1].value is None


def test_overview_family_tab_lists_only_its_own_models():
    """Triangulation: the linear-regularized tab must list Ridge/Lasso/
    ElasticNet rows and must NOT list a boosting model like XGBoost — proves
    the per-family filter actually runs, not just that 4 tabs exist."""
    at = _fresh_app()
    linear_tab = at.tabs[0]
    dataframes = linear_tab.dataframe
    assert len(dataframes) == 1
    modelos_col = dataframes[0].value["Modelo"].tolist()
    assert set(modelos_col) == {"Ridge", "Lasso", "ElasticNet"}
    assert "XGBoost" not in modelos_col


def test_overview_model_metric_shows_winner_without_api():
    """REQ-1 scenario 3: no FastAPI backend in this test environment
    (health_data is falsy), yet the 'Modelo' metric must still show the
    live winner from metrics.json, not 'No disponible'."""
    at = _fresh_app()
    metrics = _metrics()
    best_name = metrics["best_model_by_validation_mae"]
    modelo_metrics = [m for m in at.metric if m.label == "Modelo"]
    assert len(modelo_metrics) == 1
    assert modelo_metrics[0].value == best_name
