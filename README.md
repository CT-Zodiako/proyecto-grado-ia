# Modelo de IA para Programas de Medicina en Saber Pro

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg)](https://streamlit.io/)

Este proyecto desarrolla un modelo de Inteligencia Artificial para **analizar, comprender y predecir el desempeño de los programas de Medicina en Colombia** a partir de los resultados históricos de las pruebas Saber Pro del ICFES (2020-2024).

Incluye un **notebook analítico completo**, una **API en FastAPI**, un **dashboard en Streamlit**, un sistema de **recomendaciones basado en evidencia** e **Inteligencia Artificial Explicable (XAI)**.

---

## 🎯 Objetivo

Construir un sistema que permita:

- Analizar el comportamiento histórico de los programas de Medicina.
- Identificar patrones por región, departamento e institución.
- Predecir el `PROMEDIO_GLOBAL` futuro de un programa.
- Explicar qué variables influyen en cada predicción.
- Generar recomendaciones descriptivas para apoyar la toma de decisiones académicas.

---

## 🏗️ Arquitectura

```text
┌─────────────────────────────────────────────────────────────┐
│                     Notebook                                │
│  (carga, EDA, feature engineering, entrenamiento,           │
│   validación, SHAP, recomendaciones, exportación)          │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      Artefactos                              │
│  model.joblib, feature_schema.json, metrics.json,          │
│  recomendaciones_programa.csv, validation_results.json       │
└──────────────────────┬────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
┌──────────────────┐    ┌─────────────────────┐
│      API          │    │     Dashboard       │
│   FastAPI         │    │   Streamlit         │
│   /health         │    │                     │
│   /metadata       │    │   Overview          │
│   /predict        │◄───│   EDA               │
│                   │    │   Predicción        │
└──────────────────┘    │   Recomendaciones   │
                        │   Validación        │
                        │   Explicabilidad    │
                        └─────────────────────┘
```

---

## 📁 Estructura del proyecto

```text
.
├── app/
│   ├── api/
│   │   ├── main.py              # Endpoints FastAPI
│   │   ├── model_service.py     # Carga del modelo y predicciones
│   │   └── schemas.py           # Contratos Pydantic
│   ├── dashboard/
│   │   └── streamlit_app.py     # Dashboard visual
│   └── README.md                # Cómo levantar API + Dashboard
├── artifacts/
│   ├── model.joblib             # Modelo entrenado
│   ├── feature_schema.json      # Contrato de entrada
│   ├── metrics.json             # Métricas del modelo
│   ├── medicina_features.csv    # Historial completo por programa-año
│   ├── recomendaciones_programa.csv
│   ├── validation_results.json
│   └── README_artifacts.md
├── documentos_para_estudiar/
│   ├── GUIA_DASHBOARD.md        # Guía detallada del dashboard
│   ├── GUIA_PREDICCION.md       # Cómo funciona la predicción
│   ├── GUIA_TECNICA_MODELOS.md  # Explicación técnica de modelos y algoritmos
│   └── RETROSPECTIVA_FASE_POR_FASE.md
├── modelo_medicina.ipynb        # Notebook principal
├── modelo_medicina_executed.ipynb
├── DOCUMENTACION_EJECUCION.md   # Explicación paso a paso del notebook
├── PDR.md                        # Estructura del proyecto por fases
├── README_API.md                 # Guía de la API
├── README_DASHBOARD.md           # Guía del dashboard
├── STRICT_FILTER_NOTE.md         # Nota sobre el filtro estricto
├── requirements-api.txt          # Dependencias de la API
├── requirements-dashboard.txt    # Dependencias del dashboard
└── .gitignore
```

---

## 🚀 Instalación rápida

Se recomienda usar un **entorno virtual**.

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar (Mac/Linux)
source venv/bin/activate

# Activar (Windows PowerShell)
# .\venv\Scripts\Activate.ps1

# Instalar dependencias
python3 -m pip install -r requirements-api.txt
python3 -m pip install -r requirements-dashboard.txt
```

---

## ▶️ Cómo ejecutar el proyecto

Necesitás correr **dos servicios** al mismo tiempo.

### Terminal 1: API

```bash
python3 -m app.api.main
```

La API estará disponible en:

```text
http://localhost:8000
```

Documentación interactiva:

```text
http://localhost:8000/docs
```

### Terminal 2: Dashboard

Asegurate de tener el entorno virtual activado:

```bash
source venv/bin/activate
streamlit run app/dashboard/streamlit_app.py
```

El dashboard estará disponible en:

```text
http://localhost:8501
```

Para más detalles, ver [`app/README.md`](app/README.md).

---

## 📊 Métricas principales del modelo

| Métrica | Validación (2024) | Test (2025) |
|---|---:|---:|
| MAE | 4.011 | 3.849 |
| RMSE | 5.424 | 5.522 |
| R² | 0.872 | 0.845 |

- **Modelo seleccionado**: **Lasso** (pipeline sklearn: preprocesamiento + Lasso α=1.0), seleccionado por menor MAE de validación en un benchmark de 9 modelos (`mejorar_modelo.py`).
- **Variable objetivo**: `PROMEDIO_GLOBAL`.
- **Variables de entrada**: 18 (13 numéricas + 5 categóricas).
- **Filtro aplicado**: `NBC = 'MEDICINA'` AND `NOMBRE_PROGRAMA_ACAD` contiene `'MEDICINA'`.
- **Split temporal**: entrenamiento 2020-2023, validación 2024, test 2025.

---

## 🔍 Variables más importantes

Según los **coeficientes no nulos del Lasso** (sobre features escaladas), el modelo es esparso: de 105 coeficientes totales, solo **4 son distintos de cero**. Esto lo hace más interpretable que un bosque de árboles:

| Orden | Variable | Coeficiente (abs) | Qué indica |
|---|---:|---:|---|
| 1 | `maximo_historico` | 6.47 | El mejor desempeño histórico del programa es el predictor más fuerte. |
| 2 | `promedio_global_anterior` | 5.39 | El desempeño del año inmediato anterior es el segundo predictor. |
| 3 | `promedio_movil_3_anios` | 1.56 | La tendencia suavizada de los últimos 3 años también influye. |
| 4 | `promedio_movil_2_anios` | 0.09 | El promedio móvil de 2 años tiene un peso menor pero no nulo. |

> **Nota:** Lasso no tiene "importancia de árboles" como Random Forest. Los coeficientes son lineales: un valor positivo significa que, a mayor valor de la feature, mayor predicción, manteniendo el resto constante. El modelo ignora automáticamente todas las demás features (incluidas las categóricas), lo cual es una ventaja de interpretabilidad.

Ver [`documentos_para_estudiar/GUIA_PREDICCION.md`](documentos_para_estudiar/GUIA_PREDICCION.md) para más detalles.

---

## 📚 Documentación disponible

| Documento | Propósito |
|---|---|
| [`DOCUMENTACION_EJECUCION.md`](DOCUMENTACION_EJECUCION.md) | Explicación paso a paso del notebook con código y resultados. |
| [`documentos_para_estudiar/GUIA_DASHBOARD.md`](documentos_para_estudiar/GUIA_DASHBOARD.md) | Cómo funciona el dashboard. |
| [`documentos_para_estudiar/GUIA_PREDICCION.md`](documentos_para_estudiar/GUIA_PREDICCION.md) | Cómo funciona la predicción, métricas y limitaciones. |
| [`documentos_para_estudiar/GUIA_TECNICA_MODELOS.md`](documentos_para_estudiar/GUIA_TECNICA_MODELOS.md) | Explicación técnica de cada modelo y algoritmo utilizado. |
| [`documentos_para_estudiar/RETROSPECTIVA_FASE_POR_FASE.md`](documentos_para_estudiar/RETROSPECTIVA_FASE_POR_FASE.md) | Retrospectiva detallada por fase. |
| [`app/README.md`](app/README.md) | Instrucciones para levantar API y dashboard. |
| [`README_API.md`](README_API.md) | Guía de la API. |
| [`README_DASHBOARD.md`](README_DASHBOARD.md) | Guía del dashboard. |
| [`STRICT_FILTER_NOTE.md`](STRICT_FILTER_NOTE.md) | Nota sobre el filtro estricto de Medicina. |

---

## 🛠️ Tecnologías utilizadas

- **Python 3.9+**
- **Pandas** y **NumPy** para manipulación de datos.
- **Scikit-learn** para modelos y preprocesamiento.
- **FastAPI** para la API.
- **Streamlit** para el dashboard.
- **Plotly** para visualizaciones.
- **SHAP** (opcional) para explicabilidad.
- **Joblib** para serialización del modelo.

---

## ⚠️ Limitaciones importantes

- El modelo predice el `PROMEDIO_GLOBAL`, no resultados individuales ni causas.
- El error promedio es de aproximadamente **3.8 puntos** (test MAE 3.849).
- El modelo fue entrenado exclusivamente con programas llamados **Medicina**.
- No incluye variables operativas del mismo período (como cantidad de evaluados) para evitar fuga de información.
- Las recomendaciones son **descriptivas y no causales**.
- **Corrección de leakage (2026-07):** se detectó y eliminó target leakage exacto en 3 features (`tasa_crecimiento_anual`, `diferencia_maximo_historico`, `ranking_departamento`) que usaban el target del mismo año. Las métricas previas (Ridge MAE 0.670 / R² 0.996) eran inválidas y fueron descartadas.

Ver [`documentos_para_estudiar/GUIA_PREDICCION.md`](documentos_para_estudiar/GUIA_PREDICCION.md) para profundizar.

---

## 🧪 Prueba rápida de la API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "AÑO": 2025,
    "promedio_global_anterior": 165.0,
    "promedio_movil_2_anios": 162.5,
    "desviacion_historica_2_anios": 3.5,
    "anios_historicos_disponibles": 4,
    "promedio_movil_3_anios": 163.0,
    "desviacion_historica_3_anios": 4.0,
    "tasa_crecimiento_anual": 2.0,
    "maximo_historico": 170.0,
    "minimo_historico": 155.0,
    "diferencia_maximo_historico": -5.0,
    "anios_desde_inicio": 5,
    "ranking_departamento": 3.0,
    "NOMBRE_REGION": "ANDINA",
    "NOMBRE_DEPARTAMENTO": "ANTIOQUIA",
    "NOMBRE_MUNICIPIO": "MEDELLÍN",
    "NOMBRE_INSTITUCION": "UNIVERSIDAD DE ANTIOQUIA",
    "NOMBRE_PROGRAMA_ACAD": "MEDICINA"
  }'
```

---

## 🔄 Próximos pasos sugeridos

- [ ] Agregar tests automatizados para la API y el servicio.
- [ ] Implementar endpoints adicionales: `/recommend`, `/metrics/model`, `/summary/regions`.
- [ ] Dockerizar la API y el dashboard.
- [ ] Integrar gráficos SHAP en el dashboard.
- [ ] Agregar autenticación para despliegue público.
- [ ] Conectar con base de datos para persistencia de predicciones.

---

## 📌 Contexto académico

Este proyecto corresponde a un **Proyecto de Especialización en Inteligencia Artificial** y sigue la estructura propuesta en `PDR.md`.

El enfoque es **educativo y de investigación**: no pretende ser un producto de producción final sin validación adicional.

---

## 🤝 Créditos

- Datos: ICFES Saber Pro (2020-2024).
- Metodología: Machine Learning, IA Explicable (SHAP) y Análisis Exploratorio de Datos.
- Stack: FastAPI + Streamlit + Scikit-learn.
