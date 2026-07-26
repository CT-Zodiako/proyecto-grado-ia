# Dashboard Medicina Saber Pro

Este dashboard en **Streamlit** consume la **API FastAPI** del modelo de Medicina Saber Pro y permite visualizar datos, realizar predicciones y explorar recomendaciones.

## Requisitos

Instalá las dependencias del dashboard:

```bash
pip install -r requirements-dashboard.txt
```

## Estructura

```text
app/
├── dashboard/
│   ├── __init__.py
│   └── streamlit_app.py    # Dashboard principal
```

## Cómo ejecutar

### Opción 1: Solo el dashboard (con API corriendo)

Primero asegurate de que la API esté corriendo:

```bash
python3 -m app.api.main
```

Luego ejecutá el dashboard:

```bash
streamlit run app/dashboard/streamlit_app.py
```

### Opción 2: En Colab/VS Code

Si estás usando VS Code conectado a Colab:

1. Asegurate de que los artefactos estén en `artifacts/`.
2. Ejecutá la API en una terminal:

```bash
python3 -m app.api.main
```

3. En otra terminal, ejecutá el dashboard:

```bash
streamlit run app/dashboard/streamlit_app.py
```

### Opción 3: Configurar URL de la API

Si la API corre en otro lugar, seteá la variable de entorno:

```bash
export API_BASE_URL=http://localhost:8000
streamlit run app/dashboard/streamlit_app.py
```

O usá el campo de texto en el sidebar del dashboard para cambiar la URL.

## Páginas del Dashboard

| Página | Descripción |
|---|---|
| **📊 Overview** | Información general del proyecto, metadatos del modelo |
| **📈 EDA** | Análisis exploratorio: histogramas, distribuciones, filtros |
| **🔮 Predicción** | Formulario para predecir PROMEDIO_GLOBAL de un programa |
| **🩺 Diagnóstico** | Cómo/por qué/confianza de un programa en lenguaje simple + contribución exacta de variables (Lasso) + programas similares por perfil histórico (clustering) |
| **📋 Recomendaciones** | Tabla de recomendaciones filtrable, casos de riesgo |
| **✅ Validación** | Métricas del modelo, comparación de modelos, residuos |
| **🤖 Modelos** | Página dedicada con todos los modelos probados, tabla comparativa y explicaciones |
| **🔍 Explicabilidad** | Importancia de variables, contrato de predicción, nota SHAP |

## Notas importantes

- El dashboard carga datos desde `artifacts/` localmente.
- Las predicciones requieren que la API esté corriendo.
- Si la API no está disponible, las secciones de predicción mostrarán un warning.
- Los gráficos SHAP se generan en el notebook, no en el dashboard directamente (y corresponden al modelo v1/Random Forest, no al Lasso vigente — ver `documentos_para_estudiar/GUIA_DASHBOARD.md`).
- La página Diagnóstico no usa SHAP: calcula la contribución exacta de cada variable a partir de los coeficientes del Lasso.

## Próximos pasos

- Agregar autenticación si se despliega públicamente.
- Conectar con base de datos para actualización automática.
- Agregar más visualizaciones interactivas.
