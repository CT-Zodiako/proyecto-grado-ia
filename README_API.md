# API Medicina Saber Pro

Esta carpeta contiene una API mínima en **FastAPI** para servir el modelo limpio entrenado en `modelo_medicina.ipynb`.

## Requisitos

Instalá las dependencias de la API:

```bash
pip install -r requirements-api.txt
```

## Estructura

```text
app/
├── api/
│   ├── __init__.py
│   ├── main.py              # Endpoints FastAPI
│   ├── model_service.py     # Carga del modelo y predicción
│   └── schemas.py           # Contratos Pydantic
```

## Cómo ejecutar

Desde la raíz del proyecto:

```bash
uvicorn app.api.main:app --reload
```

O usando el módulo directamente:

```bash
python -m app.api.main
```

Por defecto la API escucha en:

```text
http://localhost:8000
```

## Endpoints

| Método | Path | Descripción |
|---|---|---|
| `GET` | `/health` | Verifica que la API y el modelo estén cargados. |
| `GET` | `/metadata` | Muestra schema, features y métricas del modelo. |
| `POST` | `/predict` | Recibe datos de un programa y devuelve la predicción de `promedio_global_anual`. |

## Documentación interactiva

FastAPI genera automáticamente documentación en:

```text
http://localhost:8000/docs
```

## Ejemplo de predicción con curl

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "AÑO": 2025,
    "promedio_global_anterior": 160.0,
    "promedio_movil_2_anios": 158.5,
    "desviacion_historica_2_anios": 3.5,
    "anios_historicos_disponibles": 2,
    "promedio_movil_3_anios": 159.0,
    "desviacion_historica_3_anios": 3.8,
    "tasa_crecimiento_anual": 1.5,
    "maximo_historico": 165.0,
    "minimo_historico": 152.0,
    "diferencia_maximo_historico": -5.0,
    "anios_desde_inicio": 3,
    "ranking_departamento": 5.0,
    "NOMBRE_REGION": "ANDINA",
    "NOMBRE_DEPARTAMENTO": "ANTIOQUIA",
    "NOMBRE_MUNICIPIO": "MEDELLÍN",
    "NOMBRE_INSTITUCION": "UNIVERSIDAD DE ANTIOQUIA",
    "NOMBRE_PROGRAMA_ACAD": "MEDICINA"
  }'
```

Respuesta esperada:

```json
{
  "prediccion": 162.34,
  "variable_objetivo": "promedio_global_anual",
  "modelo": "Lasso",
  "features_utilizadas": [...]
}
```

## Nota importante

El modelo acepta **18 variables de entrada** (13 numéricas + 5 categóricas). Todas deben estar presentes en el request.

El modelo limpio **no usa** estas variables:

```text
cantidad_evaluados_media_pruebas
cantidad_evaluados_max_pruebas
PROMEDIO_PRUEBA
DESVIACION
NIVEL1-NIVEL4
```

Esto permite predecir antes de conocer los resultados detallados del periodo.

### Corrección de leakage (2026-07-13)

Durante una auditoría se detectó y eliminó target leakage exacto en 3 features (`tasa_crecimiento_anual`, `diferencia_maximo_historico`, `ranking_departamento`) que usaban el target del mismo año en una versión anterior (`mejorar_modelo.py`). Las métricas previas del Ridge v2 (MAE 0.670 / R² 0.996) eran inválidas y fueron descartadas. El modelo vigente (Lasso) fue re-entrenado con features lagueadas correctamente.

## Variables categóricas

Las variables categóricas deben coincidir con categorías vistas por el modelo durante entrenamiento. Si enviás un valor desconocido, el modelo lo manejará con `handle_unknown='ignore'` del OneHotEncoder, pero la predicción puede ser menos confiable.

## Despliegue en Colab/VS Code

Si estás usando VS Code conectado a un entorno Colab:

1. Asegurate de que los artefactos estén en `artifacts/` o en `/content/artifacts`.
2. Seteá la variable de entorno si el directorio es otro:

```bash
export ARTIFACTS_DIR=/content/artifacts
```

3. Ejecutá `uvicorn app.api.main:app --reload`.

## Próximos pasos

- Agregar `/recommend` usando `recomendaciones_programa.csv`.
- Agregar `/metrics/model` y `/summary/regions`.
- Crear el dashboard en Streamlit que consuma esta API.
