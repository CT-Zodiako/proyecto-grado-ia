"""Pure diagnostic-narrative helpers for the "Diagnóstico" dashboard page.

No Streamlit imports. All functions except ``load_diagnostic_model_bundle``
are deterministic, side-effect-free, and take/return plain Python types so
they are unit-testable without Streamlit, the FastAPI service, or a live
artifacts directory (tests build a ``DiagnosticModelBundle`` by hand).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import joblib

# Exactly the Lasso model's 4 non-zero-coefficient features (locked decision).
DOMINANT_FEATURES: tuple[str, ...] = (
    "maximo_historico",
    "promedio_global_anterior",
    "promedio_movil_3_anios",
    "promedio_movil_2_anios",
)


class FeatureContribution(TypedDict):
    feature: str
    raw_value: float
    scaled_value: float
    coefficient: float
    contribution: float  # coefficient * scaled_value


class DiagnosticModelBundle(TypedDict):
    numeric_features: list[str]
    coefficients: dict[str, float]
    scaler_mean: dict[str, float]
    scaler_scale: dict[str, float]
    model_name: str


class ConfidenceState:
    """String constants — not an enum, to keep JSON/dict round-tripping
    trivial in Streamlit."""

    ZERO = "zero"
    LOW = "low"
    NORMAL = "normal"


class DiagnosticNarrative(TypedDict):
    confidence_state: str
    contributions: list[FeatureContribution] | None
    dominant: FeatureContribution | None
    narrative_sentence: str | None


# Fixed 8-entry template table — module-level constant, not user-editable at
# runtime, for determinism. Never mentions "SHAP"; always frames this as
# feature-contribution analysis to stay technically honest (the model is a
# sparse Lasso, not a SHAP explainer).
NARRATIVE_TEMPLATES: dict[tuple[str, str], str] = {
    ("maximo_historico", "positive"): (
        "El techo histórico del programa (su mejor resultado registrado) sigue "
        "siendo alto, y es el factor que más empuja la predicción hacia arriba."
    ),
    ("maximo_historico", "negative"): (
        "El techo histórico del programa es bajo en relación al resto de "
        "programas, y es el factor que más empuja la predicción hacia abajo."
    ),
    ("promedio_global_anterior", "positive"): (
        "El promedio del año anterior estuvo por encima de lo típico, y es el "
        "factor que más empuja la predicción hacia arriba."
    ),
    ("promedio_global_anterior", "negative"): (
        "El promedio del año anterior bajó respecto a lo típico, y es el "
        "factor que más empuja la predicción hacia abajo."
    ),
    ("promedio_movil_3_anios", "positive"): (
        "El promedio móvil de los últimos 3 años viene en alza, y es el factor "
        "que más empuja la predicción hacia arriba."
    ),
    ("promedio_movil_3_anios", "negative"): (
        "El promedio móvil de los últimos 3 años viene en baja, y es el factor "
        "que más empuja la predicción hacia abajo."
    ),
    ("promedio_movil_2_anios", "positive"): (
        "El promedio móvil de los últimos 2 años viene en alza, y es el factor "
        "que más empuja la predicción hacia arriba."
    ),
    ("promedio_movil_2_anios", "negative"): (
        "El promedio móvil de los últimos 2 años viene en baja, y es el factor "
        "que más empuja la predicción hacia abajo."
    ),
}


def classify_confidence_state(anios_historicos_disponibles: int) -> str:
    """0 -> ZERO, 1 -> LOW, >=2 -> NORMAL. Raises ValueError for negative
    input (defensive; should never occur given the data's semantics)."""
    if anios_historicos_disponibles < 0:
        raise ValueError(
            "anios_historicos_disponibles must be >= 0, got "
            f"{anios_historicos_disponibles}"
        )
    if anios_historicos_disponibles == 0:
        return ConfidenceState.ZERO
    if anios_historicos_disponibles == 1:
        return ConfidenceState.LOW
    return ConfidenceState.NORMAL


def compute_feature_contributions(
    bundle: DiagnosticModelBundle,
    row: dict[str, float],
) -> list[FeatureContribution]:
    """Compute contribution = coefficient * scaled_value for each feature in
    DOMINANT_FEATURES, where scaled_value = (row[feature] - scaler_mean[feature])
    / scaler_scale[feature]. Returns a list of 4 FeatureContribution dicts,
    sorted by descending abs(contribution). Pure function, deterministic, no I/O.
    """
    contributions: list[FeatureContribution] = []
    for feature in DOMINANT_FEATURES:
        raw_value = row[feature]
        mean = bundle["scaler_mean"][feature]
        scale = bundle["scaler_scale"][feature]
        coefficient = bundle["coefficients"][feature]
        scaled_value = (raw_value - mean) / scale
        contribution = coefficient * scaled_value
        contributions.append(
            FeatureContribution(
                feature=feature,
                raw_value=raw_value,
                scaled_value=scaled_value,
                coefficient=coefficient,
                contribution=contribution,
            )
        )
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return contributions


def dominant_contribution(
    contributions: list[FeatureContribution],
) -> FeatureContribution:
    """Return the entry with the largest abs(contribution). Raises
    ValueError if ``contributions`` is empty."""
    if not contributions:
        raise ValueError("contributions must not be empty")
    return max(contributions, key=lambda c: abs(c["contribution"]))


def select_narrative_template(feature: str, contribution: float) -> str:
    """Look up the fixed Spanish sentence for (feature, direction) in the
    8-entry NARRATIVE_TEMPLATES table. direction = "positive" if
    contribution >= 0 else "negative". Raises KeyError if ``feature`` is not
    one of DOMINANT_FEATURES."""
    direction = "positive" if contribution >= 0 else "negative"
    return NARRATIVE_TEMPLATES[(feature, direction)]


def build_diagnostic_narrative(
    bundle: DiagnosticModelBundle,
    row: dict[str, float],
    anios_historicos_disponibles: int,
) -> DiagnosticNarrative:
    """Orchestrates the full "why" computation for one program/year.

    When confidence_state == ZERO, no narrative is attempted (there is no
    prior year to compare against) — contributions/dominant/narrative_sentence
    are all None.
    """
    confidence_state = classify_confidence_state(anios_historicos_disponibles)

    if confidence_state == ConfidenceState.ZERO:
        return DiagnosticNarrative(
            confidence_state=confidence_state,
            contributions=None,
            dominant=None,
            narrative_sentence=None,
        )

    contributions = compute_feature_contributions(bundle, row)
    dominant = dominant_contribution(contributions)
    narrative_sentence = select_narrative_template(
        dominant["feature"], dominant["contribution"]
    )
    return DiagnosticNarrative(
        confidence_state=confidence_state,
        contributions=contributions,
        dominant=dominant,
        narrative_sentence=narrative_sentence,
    )


def load_diagnostic_model_bundle(artifacts_dir: Path) -> DiagnosticModelBundle:
    """Load model.joblib + feature_schema.json from ``artifacts_dir`` and
    extract the ordered numeric_features list, Lasso coefficients, and
    StandardScaler mean_/scale_ arrays, indexed into per-feature dicts for
    just DOMINANT_FEATURES.

    Raises FileNotFoundError if model.joblib or feature_schema.json is
    missing (mirrors ModelService.load()'s existing error behavior).

    This is the ONLY function in this module that touches disk. Callers
    (the Streamlit page) should wrap it with st.cache_resource to load once
    per session.
    """
    model_path = artifacts_dir / "model.joblib"
    schema_path = artifacts_dir / "feature_schema.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found: {schema_path}")

    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    numeric_features: list[str] = schema["numeric_features"]
    model_name: str = schema.get("model_name", "")

    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps["preprocessor"]
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    lasso = pipeline.named_steps["model"]

    coefficients: dict[str, float] = {}
    scaler_mean: dict[str, float] = {}
    scaler_scale: dict[str, float] = {}
    for feature in DOMINANT_FEATURES:
        idx = numeric_features.index(feature)
        coefficients[feature] = float(lasso.coef_[idx])
        scaler_mean[feature] = float(scaler.mean_[idx])
        scaler_scale[feature] = float(scaler.scale_[idx])

    return DiagnosticModelBundle(
        numeric_features=numeric_features,
        coefficients=coefficients,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        model_name=model_name,
    )


class ModelHyperparameters(TypedDict):
    model_class: str  # e.g. "Lasso" — type(model).__name__, not the schema
    alpha: float | None  # None for models without an alpha
    n_coefficients: int | None  # len(model.coef_), incl. one-hot columns; None if no coef_
    n_nonzero_coefficients: int | None  # int((model.coef_ != 0).sum()); None if no coef_


def load_model_hyperparameters(artifacts_dir: Path) -> ModelHyperparameters:
    """Read the FITTED estimator's hyperparameters + coefficient sparsity
    straight from ``model.joblib``.

    Raises FileNotFoundError if the model is missing (mirrors
    ``load_diagnostic_model_bundle``). Second and last disk-touching
    function in this module.
    """
    model_path = artifacts_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    pipeline = joblib.load(model_path)
    estimator = pipeline.named_steps["model"]

    alpha = getattr(estimator, "alpha", None)
    if hasattr(estimator, "coef_"):
        n_coefficients = int(len(estimator.coef_))
        n_nonzero_coefficients = int((estimator.coef_ != 0).sum())
    else:
        n_coefficients = None
        n_nonzero_coefficients = None

    return ModelHyperparameters(
        model_class=type(estimator).__name__,
        alpha=alpha,
        n_coefficients=n_coefficients,
        n_nonzero_coefficients=n_nonzero_coefficients,
    )
