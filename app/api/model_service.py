"""
Servicio de carga y predicción del modelo de Medicina Saber Pro.

Módulo separado para mantener FastAPI del modelo y facilitar pruebas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import warnings

# Suprimir advertencia de LightGBM sobre nombres de features después de ColumnTransformer
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but .* was fitted with feature names",
    category=UserWarning,
)

import joblib
import numpy as np
import pandas as pd

# Ruta configurable: usa la variable de entorno ARTIFACTS_DIR o el default
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
if env_dir := __import__("os").environ.get("ARTIFACTS_DIR"):
    ARTIFACTS_DIR = Path(env_dir)

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
VALIDATION_PATH = ARTIFACTS_DIR / "validation_results.json"
RECOMMENDATIONS_PATH = ARTIFACTS_DIR / "recomendaciones_programa.csv"


class ModelService:
    """Carga el modelo, schema y métricas exportados y realiza predicciones."""

    model: Any
    schema: dict
    metrics: dict
    validation: dict
    recommendations: pd.DataFrame | None

    def __init__(self, artifacts_dir: Path | None = None):
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
        self.model = None
        self.schema = {}
        self.metrics = {}
        self.validation = {}
        self.recommendations = None

    def load(self) -> None:
        """Carga todos los artefactos desde el directorio configurado."""
        model_path = self.artifacts_dir / "model.joblib"
        schema_path = self.artifacts_dir / "feature_schema.json"
        metrics_path = self.artifacts_dir / "metrics.json"
        validation_path = self.artifacts_dir / "validation_results.json"
        recommendations_path = self.artifacts_dir / "recomendaciones_programa.csv"

        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
        if not schema_path.exists():
            raise FileNotFoundError(f"No se encontró el schema: {schema_path}")

        self.model = joblib.load(model_path)
        self.schema = json.loads(schema_path.read_text(encoding="utf-8"))

        if metrics_path.exists():
            self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        if validation_path.exists():
            self.validation = json.loads(validation_path.read_text(encoding="utf-8"))

        if recommendations_path.exists():
            self.recommendations = pd.read_csv(recommendations_path, encoding="utf-8")

    @property
    def numeric_features(self) -> list[str]:
        return self.schema.get("numeric_features", [])

    @property
    def categorical_features(self) -> list[str]:
        return self.schema.get("categorical_features", [])

    @property
    def all_input_features(self) -> list[str]:
        return self.numeric_features + self.categorical_features

    @property
    def target(self) -> str:
        return self.schema.get("target", "promedio_global_anual")

    @property
    def best_model_name(self) -> str | None:
        # Fuente de verdad: el schema describe el modelo realmente cargado (model.joblib).
        # metrics.json queda como fallback; validation_results.json es un reporte
        # histórico (EDA v1) y NO debe definir la identidad del modelo servido.
        return (
            self.schema.get("model_name")
            or self.metrics.get("best_model_by_validation_mae")
            or self.validation.get("best_model_by_validation_mae")
        )

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Predice a partir de un diccionario con las variables de entrada."""
        if self.model is None:
            raise RuntimeError("El modelo no ha sido cargado. Ejecutá load() primero.")

        # Construir DataFrame con una fila y las columnas en el orden del schema
        input_df = pd.DataFrame([data])

        missing = [col for col in self.all_input_features if col not in input_df.columns]
        if missing:
            raise ValueError(f"Faltan variables requeridas: {missing}")

        input_df = input_df[self.all_input_features]

        # Convertir numéricas al tipo adecuado
        for col in self.numeric_features:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

        prediction = self.model.predict(input_df)
        pred_value = float(prediction[0])

        return {
            "prediccion": pred_value,
            "variable_objetivo": self.target,
            "modelo": self.best_model_name,
            "features_utilizadas": self.all_input_features,
        }


# Singleton para reutilizar en la aplicación
_service: ModelService | None = None


def get_model_service() -> ModelService:
    global _service
    if _service is None:
        _service = ModelService()
        _service.load()
    return _service


if __name__ == "__main__":
    svc = ModelService()
    svc.load()
    print("Modelo cargado:", svc.best_model_name)
    print("Features:", svc.all_input_features)
