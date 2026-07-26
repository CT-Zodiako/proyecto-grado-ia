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
└──────────────────┘    │   Diagnóstico       │
                        │   Recomendaciones   │
                        │   Validación        │
                        │   Modelos           │
                        │   Explicabilidad    │
                        └─────────────────────┘
```

> 💡 **En criollo:** el notebook "entrena" el modelo una sola vez y deja los
> resultados guardados en archivos (los `artifacts/`). La API y el dashboard
> después solo *leen* esos archivos — no vuelven a entrenar nada cada vez que
> los abrís. Por eso podés levantar el dashboard sin tener que correr el
> notebook primero.

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
│   │   ├── streamlit_app.py     # Dashboard visual (todas las páginas)
│   │   ├── diagnostics.py       # Lógica del diagnóstico IA por programa (sin Streamlit, testeable sola)
│   │   └── similar_programs.py  # Clustering de programas por perfil histórico (sin Streamlit, testeable solo)
│   └── README.md                # Cómo levantar API + Dashboard
├── tests/                        # Tests automatizados (pytest)
│   ├── test_diagnostics.py      # Tests de la lógica de diagnóstico
│   ├── test_similar_programs.py # Tests del clustering de programas similares
│   └── test_streamlit_diagnostico_page.py  # Tests de la página del dashboard
├── artifacts/
│   ├── model.joblib             # Modelo entrenado
│   ├── feature_schema.json      # Contrato de entrada
│   ├── metrics.json             # Métricas del modelo
│   ├── medicina_features.csv    # Historial completo por programa-año
│   ├── recomendaciones_programa.csv
│   ├── validation_results.json
│   └── README_artifacts.md
├── documentos_para_estudiar/
│   ├── COMO_FUNCIONA_EL_PROYECTO.md  # Explicación completa: datos, entrenamiento, predicción, diagnóstico
│   ├── GUIA_DASHBOARD.md        # Guía detallada del dashboard
│   ├── GUIA_PREDICCION.md       # Cómo funciona la predicción
│   ├── GUIA_TECNICA_MODELOS.md  # Explicación técnica de modelos y algoritmos
│   └── RETROSPECTIVA_FASE_POR_FASE.md
├── modelo_medicina.ipynb        # Notebook principal
├── modelo_medicina_executed.ipynb
├── PDR.md                        # Estructura del proyecto por fases
├── README_API.md                 # Guía de la API
├── README_DASHBOARD.md           # Guía del dashboard
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

Para más detalles, ver [`README_API.md`](README_API.md) y [`README_DASHBOARD.md`](README_DASHBOARD.md).

---

## ✅ Cómo correr los tests (verificar que todo funciona)

Esto no es obligatorio para usar el dashboard, pero sirve para confirmar que
el código funciona correctamente antes de tocar algo (por ejemplo después de
clonar el proyecto o de hacer un cambio).

```bash
source venv/bin/activate
pytest tests/ -v
```

Si todo está bien, vas a ver algo como `54 passed` al final. Estos tests
cubren la lógica interna del diagnóstico IA, el clustering de programas
similares, y la página completa del dashboard (se prueba automáticamente con
los 56 programas reales del dataset).

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

## 🩺 Página de Diagnóstico (nuevo)

Además de predecir un número, el dashboard tiene una página llamada
**"🩺 Diagnóstico"** pensada para cualquier persona interesada en entender un
programa de Medicina, no solo para técnicos. Para el programa que elijas, te
responde tres preguntas en este orden:

1. **¿Cómo le fue?** — el puntaje del último año disponible y cómo veniía la
   tendencia histórica.
2. **¿Por qué?** — una frase en español simple explicando qué factor pesó
   más en esa predicción (por ejemplo, si el techo histórico del programa es
   alto o si el año anterior vino en baja). Abajo de esa frase hay un detalle
   técnico opcional para quien quiera profundizar.
3. **¿Qué tan confiable es esto?** — si el programa tiene poca historia
   (menos de 2 años de datos), se muestra una advertencia clara explicando
   por qué la estimación es menos confiable en ese caso.

Esta explicación **no usa la librería SHAP** (a pesar de que el modelo es un
modelo lineal simple — Lasso — la contribución de cada variable se puede
calcular de forma exacta con sus coeficientes, sin necesidad de aproximarla).

Además, la misma página tiene un quinto bloque, **"Programas similares"**:
agrupa los 73 programas de Medicina por su **perfil histórico de desempeño**
(promedio, tendencia y volatilidad) usando clustering no supervisado
(K-means), y te muestra 3 a 5 programas parecidos al que elegiste — **nunca**
por cercanía geográfica, solo por cómo les fue en el tiempo. Sirve para
responder "¿mi programa es un caso típico o una excepción?" comparando con
programas que tuvieron un comportamiento parecido, sin importar dónde queden.

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
| [`documentos_para_estudiar/COMO_FUNCIONA_EL_PROYECTO.md`](documentos_para_estudiar/COMO_FUNCIONA_EL_PROYECTO.md) | Explicación completa de punta a punta: datos, entrenamiento, artefactos, predicción, diagnóstico y clustering. |
| [`documentos_para_estudiar/GUIA_DASHBOARD.md`](documentos_para_estudiar/GUIA_DASHBOARD.md) | Cómo funciona el dashboard. |
| [`documentos_para_estudiar/GUIA_PREDICCION.md`](documentos_para_estudiar/GUIA_PREDICCION.md) | Cómo funciona la predicción, métricas y limitaciones. |
| [`documentos_para_estudiar/GUIA_TECNICA_MODELOS.md`](documentos_para_estudiar/GUIA_TECNICA_MODELOS.md) | Explicación técnica de cada modelo y algoritmo utilizado. |
| [`documentos_para_estudiar/RETROSPECTIVA_FASE_POR_FASE.md`](documentos_para_estudiar/RETROSPECTIVA_FASE_POR_FASE.md) | Retrospectiva detallada por fase. |
| [`README_API.md`](README_API.md) | Guía de la API. |
| [`README_DASHBOARD.md`](README_DASHBOARD.md) | Guía del dashboard. |

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

- [x] Agregar tests automatizados para el diagnóstico IA y la página del dashboard (`tests/`, 54 tests con pytest).
- [x] Agregar comparación con programas similares (clustering por perfil histórico, sin geografía).
- [ ] Agregar tests automatizados para la API (`app/api/`).
- [ ] Corregir un bug detectado en la página "Predicción": dos variables importantes del modelo (`maximo_historico`, `promedio_movil_3_anios`) no se están enviando a la API, que las reemplaza en silencio por un valor promedio (mediana). Ver detalle en el historial de decisiones del proyecto.
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
