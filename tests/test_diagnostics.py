"""Unit tests for app/dashboard/diagnostics.py.

All tests use the synthetic ``synthetic_bundle`` fixture from conftest.py —
no real model.joblib/feature_schema.json is required for the pure-function
tests. Only the optional smoke test at the bottom touches real artifacts.
"""
from pathlib import Path

import pytest

from app.dashboard.diagnostics import (
    DOMINANT_FEATURES,
    ConfidenceState,
    build_diagnostic_narrative,
    classify_confidence_state,
    compute_feature_contributions,
    dominant_contribution,
    load_diagnostic_model_bundle,
    select_narrative_template,
)


# ---------------------------------------------------------------------------
# classify_confidence_state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "years,expected",
    [
        (0, ConfidenceState.ZERO),
        (1, ConfidenceState.LOW),
        (2, ConfidenceState.NORMAL),
        (5, ConfidenceState.NORMAL),
    ],
)
def test_classify_confidence_state(years, expected):
    assert classify_confidence_state(years) == expected


def test_classify_confidence_state_negative_raises():
    with pytest.raises(ValueError):
        classify_confidence_state(-1)


# ---------------------------------------------------------------------------
# compute_feature_contributions
# ---------------------------------------------------------------------------

def test_compute_feature_contributions_exact_values(synthetic_bundle):
    row = {
        "maximo_historico": 185.0,       # scaled = (185-170)/15 = 1.0 -> contrib = 6.0
        "promedio_global_anterior": 150.0,  # scaled = (150-165)/15 = -1.0 -> contrib = -5.0
        "promedio_movil_3_anios": 166.0,    # scaled = 0.0 -> contrib = 0.0
        "promedio_movil_2_anios": 181.0,    # scaled = 1.0 -> contrib = 0.1
    }
    contributions = compute_feature_contributions(synthetic_bundle, row)

    by_feature = {c["feature"]: c for c in contributions}
    assert by_feature["maximo_historico"]["contribution"] == pytest.approx(6.0)
    assert by_feature["promedio_global_anterior"]["contribution"] == pytest.approx(-5.0)
    assert by_feature["promedio_movil_3_anios"]["contribution"] == pytest.approx(0.0)
    assert by_feature["promedio_movil_2_anios"]["contribution"] == pytest.approx(0.1)

    # sorted by descending abs(contribution): maximo_historico(6.0) > promedio_global_anterior(5.0)
    # > promedio_movil_2_anios(0.1) > promedio_movil_3_anios(0.0)
    ordered = [c["feature"] for c in contributions]
    assert ordered == [
        "maximo_historico",
        "promedio_global_anterior",
        "promedio_movil_2_anios",
        "promedio_movil_3_anios",
    ]


def test_compute_feature_contributions_returns_all_dominant_features(synthetic_bundle):
    row = {f: 100.0 for f in DOMINANT_FEATURES}
    contributions = compute_feature_contributions(synthetic_bundle, row)
    assert {c["feature"] for c in contributions} == set(DOMINANT_FEATURES)
    assert len(contributions) == 4


# ---------------------------------------------------------------------------
# dominant_contribution
# ---------------------------------------------------------------------------

def test_dominant_contribution_mixed_signs():
    contributions = [
        {"feature": "a", "raw_value": 0, "scaled_value": 0, "coefficient": 0, "contribution": 3.0},
        {"feature": "b", "raw_value": 0, "scaled_value": 0, "coefficient": 0, "contribution": -7.0},
        {"feature": "c", "raw_value": 0, "scaled_value": 0, "coefficient": 0, "contribution": 1.0},
    ]
    dominant = dominant_contribution(contributions)
    assert dominant["feature"] == "b"
    assert dominant["contribution"] == -7.0


def test_dominant_contribution_all_positive():
    contributions = [
        {"feature": "a", "raw_value": 0, "scaled_value": 0, "coefficient": 0, "contribution": 2.0},
        {"feature": "b", "raw_value": 0, "scaled_value": 0, "coefficient": 0, "contribution": 4.5},
    ]
    assert dominant_contribution(contributions)["feature"] == "b"


def test_dominant_contribution_all_negative():
    contributions = [
        {"feature": "a", "raw_value": 0, "scaled_value": 0, "coefficient": 0, "contribution": -2.0},
        {"feature": "b", "raw_value": 0, "scaled_value": 0, "coefficient": 0, "contribution": -4.5},
    ]
    assert dominant_contribution(contributions)["feature"] == "b"


def test_dominant_contribution_empty_raises():
    with pytest.raises(ValueError):
        dominant_contribution([])


# ---------------------------------------------------------------------------
# select_narrative_template
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("feature", DOMINANT_FEATURES)
@pytest.mark.parametrize("contribution,direction", [(1.0, "positive"), (-1.0, "negative")])
def test_select_narrative_template_all_combos(feature, contribution, direction):
    sentence = select_narrative_template(feature, contribution)
    assert isinstance(sentence, str)
    assert len(sentence) > 0
    # Technical honesty requirement: never claim SHAP.
    assert "shap" not in sentence.lower()


def test_select_narrative_template_zero_contribution_is_positive_direction():
    # direction = "positive" if contribution >= 0 else "negative" (per design §3 step 8)
    sentence = select_narrative_template(DOMINANT_FEATURES[0], 0.0)
    assert isinstance(sentence, str)


def test_select_narrative_template_unknown_feature_raises():
    with pytest.raises(KeyError):
        select_narrative_template("unknown_feature", 1.0)


# ---------------------------------------------------------------------------
# build_diagnostic_narrative
# ---------------------------------------------------------------------------

def test_build_diagnostic_narrative_zero_history_no_narrative(synthetic_bundle):
    row = {f: 100.0 for f in DOMINANT_FEATURES}
    narrative = build_diagnostic_narrative(synthetic_bundle, row, anios_historicos_disponibles=0)
    assert narrative["confidence_state"] == ConfidenceState.ZERO
    assert narrative["contributions"] is None
    assert narrative["dominant"] is None
    assert narrative["narrative_sentence"] is None


def test_build_diagnostic_narrative_low_confidence_still_has_narrative(synthetic_bundle):
    row = {
        "maximo_historico": 185.0,
        "promedio_global_anterior": 150.0,
        "promedio_movil_3_anios": 166.0,
        "promedio_movil_2_anios": 181.0,
    }
    narrative = build_diagnostic_narrative(synthetic_bundle, row, anios_historicos_disponibles=1)
    assert narrative["confidence_state"] == ConfidenceState.LOW
    assert narrative["contributions"] is not None
    assert narrative["dominant"] is not None
    assert narrative["narrative_sentence"] is not None


def test_build_diagnostic_narrative_normal_confidence(synthetic_bundle):
    row = {
        "maximo_historico": 185.0,
        "promedio_global_anterior": 150.0,
        "promedio_movil_3_anios": 166.0,
        "promedio_movil_2_anios": 181.0,
    }
    narrative = build_diagnostic_narrative(synthetic_bundle, row, anios_historicos_disponibles=2)
    assert narrative["confidence_state"] == ConfidenceState.NORMAL
    assert narrative["contributions"] is not None
    assert narrative["dominant"]["feature"] == "maximo_historico"


def test_build_diagnostic_narrative_is_deterministic(synthetic_bundle):
    row = {
        "maximo_historico": 185.0,
        "promedio_global_anterior": 150.0,
        "promedio_movil_3_anios": 166.0,
        "promedio_movil_2_anios": 181.0,
    }
    first = build_diagnostic_narrative(synthetic_bundle, row, anios_historicos_disponibles=3)
    second = build_diagnostic_narrative(synthetic_bundle, row, anios_historicos_disponibles=3)
    assert first == second


# ---------------------------------------------------------------------------
# load_diagnostic_model_bundle — optional smoke test against real artifacts
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


@pytest.mark.skipif(
    not (ARTIFACTS_DIR / "model.joblib").exists(),
    reason="artifacts/model.joblib not present in this environment",
)
def test_load_diagnostic_model_bundle_smoke_test_real_artifacts():
    bundle = load_diagnostic_model_bundle(ARTIFACTS_DIR)
    assert bundle["model_name"] == "Lasso"
    for feature in DOMINANT_FEATURES:
        assert feature in bundle["coefficients"]
        assert feature in bundle["scaler_mean"]
        assert feature in bundle["scaler_scale"]
        # Locked assumption: these 4 features are the Lasso's non-zero coefficients.
        assert bundle["coefficients"][feature] != 0.0


def test_load_diagnostic_model_bundle_missing_artifacts_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_diagnostic_model_bundle(tmp_path)
