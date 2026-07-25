from pydantic import BaseModel, Field
from typing import Optional, Any


class PredictionRequest(BaseModel):
    """Contrato de entrada para /predict.

    Variables numéricas: son históricas y contextuales del modelo limpio.
    Variables categóricas: deben coincidir con categorías vistas por el OneHotEncoder.
    """

    AÑO: float = Field(..., example=2024, description="Año a predecir")
    promedio_global_anterior: Optional[float] = Field(
        None, example=160.0, description="Promedio global del año anterior"
    )
    promedio_movil_2_anios: Optional[float] = Field(
        None, example=158.5, description="Promedio móvil de los 2 años previos"
    )
    desviacion_historica_2_anios: Optional[float] = Field(
        None, example=3.5, description="Desviación estándar de los 2 años previos"
    )
    anios_historicos_disponibles: Optional[float] = Field(
        None, example=2, description="Cantidad de años históricos disponibles"
    )

    # Nuevas variables del modelo v2
    promedio_movil_3_anios: Optional[float] = Field(
        None, example=157.0, description="Promedio móvil de los 3 años previos"
    )
    desviacion_historica_3_anios: Optional[float] = Field(
        None, example=4.0, description="Desviación estándar de los 3 años previos"
    )
    tasa_crecimiento_anual: Optional[float] = Field(
        None, example=1.5, description="Tasa de crecimiento anual respecto al año anterior"
    )
    maximo_historico: Optional[float] = Field(
        None, example=165.0, description="Máximo promedio global histórico hasta el año anterior"
    )
    minimo_historico: Optional[float] = Field(
        None, example=150.0, description="Mínimo promedio global histórico hasta el año anterior"
    )
    diferencia_maximo_historico: Optional[float] = Field(
        None, example=-5.0, description="Diferencia con el máximo histórico"
    )
    anios_desde_inicio: Optional[float] = Field(
        None, example=4, description="Años transcurridos desde el primer registro del programa"
    )
    ranking_departamento: Optional[float] = Field(
        None, example=3.0, description="Ranking del programa dentro del departamento en el último año"
    )

    NOMBRE_REGION: str = Field(..., example="ANDINA")
    NOMBRE_DEPARTAMENTO: str = Field(..., example="ANTIOQUIA")
    NOMBRE_MUNICIPIO: str = Field(..., example="MEDELLÍN")
    NOMBRE_INSTITUCION: str = Field(..., example="UNIVERSIDAD DE ANTIOQUIA")
    NOMBRE_PROGRAMA_ACAD: str = Field(..., example="MEDICINA")


class PredictionResponse(BaseModel):
    """Respuesta de /predict."""

    prediccion: float
    variable_objetivo: str
    modelo: Optional[str]
    features_utilizadas: list[str]


class HealthResponse(BaseModel):
    """Respuesta de /health."""

    status: str
    modelo_cargado: bool
    modelo_nombre: Optional[str]
    artefactos_dir: str
    timestamp: str


class MetadataResponse(BaseModel):
    """Respuesta de /metadata."""

    target: str
    modelo: Optional[str]
    numeric_features: list[str]
    categorical_features: list[str]
    excluded_features: list[str]
    best_validation_metrics: Optional[dict[str, Any]]
    best_test_metrics: Optional[dict[str, Any]]
    artifacts_dir: str
