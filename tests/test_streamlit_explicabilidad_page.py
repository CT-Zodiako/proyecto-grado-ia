"""Integration tests for the redesigned Explicabilidad page
(explicabilidad-methodology-page): tabs skeleton, leakage forensics (§A),
prediction-mechanism content with the live Lasso coefficient chart (§B),
training methodology with honest alpha framing (§C), and demotion of the
stale Random Forest chart into a historical annex (§D).

Uses Streamlit's AppTest headless runner, same pattern as
test_streamlit_overview_page.py / test_streamlit_navigation.py.
"""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
APP_PATH = Path(__file__).resolve().parent.parent / "app" / "dashboard" / "streamlit_app.py"

sys.path.insert(0, str(APP_PATH.parent))
import diagnostics  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS_DIR / "model.joblib").exists(),
    reason="real artifacts not present in this environment",
)

TAB_LABELS = ["🧪 Fuga de datos", "⚙️ Cómo se predice", "🎓 Cómo se entrenó"]

LEAKED_FEATURES = [
    "tasa_crecimiento_anual",
    "diferencia_maximo_historico",
    "ranking_departamento",
]


def _fresh_app():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    assert not at.exception, f"App failed to load: {at.exception}"
    return at


def _go_to_explicabilidad(at):
    at.sidebar.radio[0].set_value("🔍 Explicación").run(timeout=30)
    assert not at.exception, f"Explicabilidad page raised: {at.exception}"
    return at


def _all_rendered_text(at):
    """Collect every piece of rendered text so 'no stale claim anywhere'
    assertions inspect markdown, warning/info/success boxes, captions,
    subheaders, and code blocks alike."""
    chunks = []
    for md in at.markdown:
        chunks.append(md.value)
    for s in at.success:
        chunks.append(s.value)
    for i in at.info:
        chunks.append(i.value)
    for w in at.warning:
        chunks.append(w.value)
    for e in at.error:
        chunks.append(e.value)
    for sh in at.subheader:
        chunks.append(sh.value)
    for c in at.caption:
        chunks.append(c.value)
    for code in at.code:
        chunks.append(code.value)
    for t in at.title:
        chunks.append(t.value)
    return chunks


def _plotly_titles(at):
    titles = []
    for chart in at.get("plotly_chart"):
        spec = json.loads(chart.proto.spec)
        title = spec.get("layout", {}).get("title", {}).get("text")
        if title:
            titles.append(title)
    return titles


def test_explicabilidad_page_renders_no_exception():
    at = _fresh_app()
    _go_to_explicabilidad(at)


def test_explicabilidad_has_three_tabs_in_order():
    at = _fresh_app()
    _go_to_explicabilidad(at)
    assert len(at.tabs) == 3
    assert [t.label for t in at.tabs] == TAB_LABELS


def test_explicabilidad_leaked_feature_names_present():
    at = _fresh_app()
    _go_to_explicabilidad(at)
    text = "\n".join(_all_rendered_text(at))
    for feature in LEAKED_FEATURES:
        assert feature in text


def test_explicabilidad_leakage_states_recalculated_not_removed():
    at = _fresh_app()
    _go_to_explicabilidad(at)
    text = "\n".join(_all_rendered_text(at))
    assert "recalcul" in text.lower()
    assert "no se eliminaron" in text.lower() or "no eliminadas" in text.lower()


def test_explicabilidad_v2_split_counts_present_v1_absent():
    at = _fresh_app()
    _go_to_explicabilidad(at)
    text = "\n".join(_all_rendered_text(at))
    assert "174" in text
    assert "62" in text
    assert "64" in text
    # v1 counts must never appear anywhere on the redesigned page.
    assert "184" not in text
    assert "71" not in text


def test_explicabilidad_no_stale_unlabeled_rf_chart_title():
    at = _fresh_app()
    _go_to_explicabilidad(at)
    text = "\n".join(_all_rendered_text(at))
    titles = _plotly_titles(at)
    all_text = text + "\n".join(t for t in titles if t)
    assert "Importancia de Variables (Random Forest)" not in all_text
    # Every remaining "Random Forest" mention must co-occur with a
    # historical/v1 qualifier somewhere in the same rendered chunk set.
    if "Random Forest" in all_text:
        assert "históric" in all_text.lower() or "v1" in all_text.lower()


def test_explicabilidad_alpha_honesty_no_element_mixes_0178_and_servido():
    at = _fresh_app()
    _go_to_explicabilidad(at)
    chunks = _all_rendered_text(at)
    for chunk in chunks:
        if "0.178" in chunk or "0,178" in chunk:
            assert "servido" not in chunk.lower()


def test_explicabilidad_alpha_1_0_reported_as_live_served_value():
    at = _fresh_app()
    _go_to_explicabilidad(at)
    text = "\n".join(_all_rendered_text(at))
    assert "1.0" in text


def test_explicabilidad_no_stale_ridge_winner_claim():
    """Regression guard: this project's recurring bug pattern is a stale
    'Ridge ganó' claim surviving after Lasso became the winner (already
    fixed once on Overview/Validación/Modelos in a prior change). This page
    is new and must never introduce that claim in the first place."""
    at = _fresh_app()
    _go_to_explicabilidad(at)
    text = "\n".join(_all_rendered_text(at))
    assert "¿Por qué ganó Ridge?" not in text
    assert "El seleccionado final es **Ridge**" not in text
    metrics = json.loads((ARTIFACTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    best_name = metrics["best_model_by_validation_mae"]
    assert best_name == "Lasso"
    assert f"α = 1.0" in text or "alpha=1.0" in text or "Lasso" in text


def test_explicabilidad_annex_expander_present():
    at = _fresh_app()
    _go_to_explicabilidad(at)
    assert len(at.expander) >= 1


def test_explicabilidad_placeholder_removed_from_prediction_tab():
    """PR2 supersedes PR1's one-line placeholder with real §B content."""
    at = _fresh_app()
    _go_to_explicabilidad(at)
    text = "\n".join(_all_rendered_text(at))
    assert "sección en preparación" not in text.lower()


def test_explicabilidad_prediction_paths_named():
    at = _fresh_app()
    _go_to_explicabilidad(at)
    text = "\n".join(_all_rendered_text(at))
    assert "api_predict" in text
    assert "POST /predict" in text
    assert "ModelService.predict" in text
    assert "compute_feature_contributions" in text


def test_explicabilidad_formula_snippet_verbatim():
    at = _fresh_app()
    _go_to_explicabilidad(at)
    codes = [c.value for c in at.code]
    assert any("contribution = coefficient * scaled_value" in c for c in codes)


def test_explicabilidad_worked_example_has_no_selector_widget():
    at = _fresh_app()
    _go_to_explicabilidad(at)
    assert len(at.selectbox) == 0
    assert len(at.number_input) == 0
    assert len(at.slider) == 0


def test_explicabilidad_coefficient_chart_caption_matches_live_model():
    at = _fresh_app()
    _go_to_explicabilidad(at)
    hp = diagnostics.load_model_hyperparameters(ARTIFACTS_DIR)
    text = "\n".join(_all_rendered_text(at))
    assert f"{hp['n_nonzero_coefficients']} de {hp['n_coefficients']}" in text
    titles = _plotly_titles(at)
    assert any(hp["model_class"] in (title or "") for title in titles)


def test_explicabilidad_cta_navigates_to_diagnostico():
    at = _fresh_app()
    _go_to_explicabilidad(at)
    at.button(key="cta_explica_to_diagnostico").click().run(timeout=30)
    assert not at.exception, f"CTA click raised: {at.exception}"
    assert at.sidebar.radio[0].value == "🩺 Diagnóstico"
    assert at.sidebar.radio[1].value is None


def test_explicabilidad_no_shap_rationale_present():
    at = _fresh_app()
    _go_to_explicabilidad(at)
    text = "\n".join(_all_rendered_text(at)).lower()
    assert "¿por qué no usamos shap?" in text
    assert "exacta" in text or "exacto" in text
    # The old broken "install shap in Colab" instructions must never return.
    assert "colab" not in text
    assert "pip install shap" not in text


def test_explicabilidad_prediction_contract_preserved():
    """'Contrato de Predicción' (numeric/categorical feature list + target)
    must be re-indented into the ⚙️ tab, reused verbatim from the old
    flat page."""
    at = _fresh_app()
    _go_to_explicabilidad(at)
    text = "\n".join(_all_rendered_text(at))
    assert "Contrato de Predicción" in text
    with open(ARTIFACTS_DIR / "feature_schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    for feat in schema.get("numeric_features", []):
        assert f"`{feat}`" in text
    for feat in schema.get("categorical_features", []):
        assert f"`{feat}`" in text
