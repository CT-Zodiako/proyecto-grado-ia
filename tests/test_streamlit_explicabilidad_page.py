"""Integration tests for the redesigned Explicabilidad page
(explicabilidad-methodology-page): tabs skeleton, leakage forensics (§A),
training methodology with honest alpha framing (§C), and demotion of the
stale Random Forest chart into a historical annex (§D).

PR1 scope only — the prediction-mechanism content (§B) and the live Lasso
coefficient chart ship in PR2 and are NOT asserted here.

Uses Streamlit's AppTest headless runner, same pattern as
test_streamlit_overview_page.py / test_streamlit_navigation.py.
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
    at.sidebar.radio[1].set_value("🔍 Explicabilidad").run(timeout=30)
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


def test_explicabilidad_annex_expander_present():
    at = _fresh_app()
    _go_to_explicabilidad(at)
    assert len(at.expander) >= 1


def test_explicabilidad_placeholder_present_in_prediction_tab():
    """PR1 leaves the ⚙️ tab with a one-line placeholder; PR2 fills it in."""
    at = _fresh_app()
    _go_to_explicabilidad(at)
    text = "\n".join(_all_rendered_text(at))
    assert "preparación" in text.lower()
