# Medicina Saber Pro - API + Dashboard

Este directorio contiene la **API FastAPI** y el **Dashboard Streamlit** del proyecto de Inteligencia Artificial para analizar y predecir el desempeño de los programas de **Medicina** en las pruebas Saber Pro de Colombia.

El proyecto filtra estrictamente por:

```python
NBC = 'MEDICINA' AND NOMBRE_PROGRAMA_ACAD contiene 'MEDICINA'
```

---

## Estructura

```text
app/
├── api/
│   ├── __init__.py
│   ├── main.py              # Endpoints FastAPI
│   ├── model_service.py     # Carga del modelo y predicciones
│   └── schemas.py           # Contratos Pydantic
├── dashboard/
│   ├── __init__.py
│   └── streamlit_app.py     # Dashboard visual
└── README.md                # Este archivo
```

---

## Requisitos

Se recomienda usar un **entorno virtual** para evitar conflictos con otras librerías del sistema.

### Crear y activar entorno virtual

Desde la raíz del proyecto:

```bash
python3 -m venv venv
source venv/bin/activate
```

En Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Instalar dependencias de la API

Con el entorno activado:

```bash
python3 -m pip install -r requirements-api.txt
```

### Instalar dependencias del Dashboard

```bash
python3 -m pip install -r requirements-dashboard.txt
```

O instalá todo junto:

```bash
python3 -m pip install -r requirements-api.txt -r requirements-dashboard.txt
```

---

## Cómo levantar el proyecto completo

Asegurate de tener el entorno virtual activado:

```bash
source venv/bin/activate
```

Necesitás correr **dos servicios** al mismo tiempo: la API y el Dashboard.

### Paso 1: Levantar la API

Desde la raíz del proyecto:

```bash
python3 -m app.api.main
```

La API queda disponible en:

```text
http://localhost:8000
```

Documentación interactiva:

```text
http://localhost:8000/docs
```

### Paso 2: Levantar el Dashboard

En otra terminal, también desde la raíz:

```bash
streamlit run app/dashboard/streamlit_app.py
```

El dashboard queda disponible en:

```text
http://localhost:8501
```

---

## Cerrar el entorno virtual

Cuando termines, en cada terminal ejecutá:

```bash
deactivate
```

---

## Endpoints de la API

| Método | Path | Descripción |
|---|---|---|
| `GET` | `/health` | Verifica que la API y el modelo estén cargados. |
| `GET` | `/metadata` | Devuelve features, métricas y metadatos del modelo. |
| `POST` | `/predict` | Predice el `PROMEDIO_GLOBAL` de un programa de Medicina. |

### Ejemplo de predicción

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "AÑO": 2024,
    "promedio_global_anterior": 165.0,
    "promedio_movil_2_anios": 162.5,
    "desviacion_historica_2_anios": 3.5,
    "anios_historicos_disponibles": 4,
    "NOMBRE_REGION": "ANDINA",
    "NOMBRE_DEPARTAMENTO": "ANTIOQUIA",
    "NOMBRE_MUNICIPIO": "MEDELLÍN",
    "NOMBRE_INSTITUCION": "UNIVERSIDAD DE ANTIOQUIA",
    "NOMBRE_PROGRAMA_ACAD": "MEDICINA"
  }'
```

---

## Páginas del Dashboard

| Página | Descripción |
|---|---|
| **Overview** | Información general del proyecto y metadatos del modelo. |
| **EDA** | Análisis exploratorio con histogramas y filtros. |
| **Predicción** | Formulario para predecir el desempeño de un programa. |
| **Recomendaciones** | Tabla de recomendaciones filtrable y casos de riesgo. |
| **Validación** | Métricas del modelo, comparación y residuos. |
| **Explicabilidad** | Importancia de variables, variables excluidas y nota SHAP. |

---

## VS Code conectado a Colab

Si estás usando **VS Code con la extensión de Colab**, recordá que:

- El Python corre en el **runtime remoto de Colab**.
- Los artefactos deben estar en `artifacts/` o en `/content/artifacts`.
- Si usás Google Drive para persistir artefactos, seteá la variable de entorno antes de levantar la API:

```bash
export ARTIFACTS_DIR=/content/artifacts
```

- Para abrir el dashboard en tu navegador local, podés necesitar **Port Forwarding** desde VS Code.

---

## Ubicación de los artefactos

La API carga el modelo y los datos desde el directorio de artefactos:

```text
artifacts/
├── model.joblib
├── feature_schema.json
├── metrics.json
├── recomendaciones_programa.csv
├── validation_results.json
└── README_artifacts.md
```

Si cambiás la ubicación, usá la variable de entorno:

```bash
export ARTIFACTS_DIR=/ruta/a/tus/artifacts
```

---

## Nota metodológica

El modelo limpio **no usa** estas variables como predictores:

```text
cantidad_evaluados_media_pruebas
cantidad_evaluados_max_pruebas
PROMEDIO_PRUEBA
DESVIACION
NIVEL1, NIVEL2, NIVEL3, NIVEL4
```

Esto permite predecir el desempeño sin depender de variables del mismo período que solo se conocerían después de la evaluación.

---

## Próximos pasos sugeridos

- Agregar más endpoints: `/recommend`, `/metrics/model`, `/summary/regions`.
- Agregar tests automatizados para la API.
- Dockerizar la API y el dashboard.
- Integrar gráficos SHAP en el dashboard cuando `shap` esté disponible.

