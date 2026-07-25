from fastapi import FastAPI, HTTPException
from datetime import datetime
import os

from .schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    MetadataResponse,
)
from .model_service import get_model_service

app = FastAPI(
    title="API Medicina Saber Pro",
    description="API para predecir el desempeño de programas de Medicina en Saber Pro.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    svc = get_model_service()
    return HealthResponse(
        status="OK",
        modelo_cargado=svc.model is not None,
        modelo_nombre=svc.best_model_name,
        artefactos_dir=str(svc.artifacts_dir),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/metadata", response_model=MetadataResponse)
def metadata():
    svc = get_model_service()
    return MetadataResponse(
        target=svc.target,
        modelo=svc.best_model_name,
        numeric_features=svc.numeric_features,
        categorical_features=svc.categorical_features,
        excluded_features=svc.schema.get("excluded_features", []),
        best_validation_metrics=svc.validation.get("best_validation_metrics"),
        best_test_metrics=svc.validation.get("best_test_metrics"),
        artifacts_dir=str(svc.artifacts_dir),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        svc = get_model_service()
        data = request.dict()
        result = svc.predict(data)
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
