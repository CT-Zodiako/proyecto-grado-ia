"""Shared pytest fixtures for diagnostics.py unit tests.

No real ``model.joblib`` or ``feature_schema.json`` is required: a synthetic
``DiagnosticModelBundle`` is built by hand so tests stay independent of the
actual artifacts directory and can assert exact expected numbers.
"""
import pytest

from app.dashboard.diagnostics import DOMINANT_FEATURES, DiagnosticModelBundle


@pytest.fixture
def synthetic_bundle() -> DiagnosticModelBundle:
    """A hand-built bundle covering all 4 DOMINANT_FEATURES with simple,
    easy-to-hand-verify coefficient/scaler values.
    """
    return DiagnosticModelBundle(
        numeric_features=list(DOMINANT_FEATURES),
        coefficients={
            "maximo_historico": 6.0,
            "promedio_global_anterior": 5.0,
            "promedio_movil_3_anios": 1.5,
            "promedio_movil_2_anios": 0.1,
        },
        scaler_mean={
            "maximo_historico": 170.0,
            "promedio_global_anterior": 165.0,
            "promedio_movil_3_anios": 166.0,
            "promedio_movil_2_anios": 166.0,
        },
        scaler_scale={
            "maximo_historico": 15.0,
            "promedio_global_anterior": 15.0,
            "promedio_movil_3_anios": 15.0,
            "promedio_movil_2_anios": 15.0,
        },
        model_name="Lasso",
    )
