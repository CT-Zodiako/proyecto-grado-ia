# Artefactos exportados - Modelo Medicina Saber Pro

Estos archivos fueron generados desde `modelo_medicina.ipynb` para alimentar la arquitectura API + Dashboard.

## Ruta usada

```text
/Users/zodiakomac/DEV/proyecto-grado-ia/artifacts
```

## Archivos

- `model.joblib`: pipeline del mejor modelo seleccionado (actualmente **Lasso v2**, benchmark limpio post-corrección de leakage del 2026-07-13).
- `model_v1.joblib`: modelo original Random Forest (2020-2024) para referencia.
- `model_v2.joblib`: modelo ganador del benchmark de 9 modelos (Lasso, seleccionado por menor MAE de validación).
- `feature_schema.json`: contrato de entrada para `/predict` (v2 con nuevas variables históricas).
- `feature_schema_v1.json`: schema original del modelo v1.
- `feature_schema_v2.json`: schema v2 con las nuevas variables.
- `metrics.json`: métricas comparativas de todos los modelos probados.
- `metrics_v1.json`: métricas del modelo v1.
- `metrics_v2.json`: métricas comparativas v2.
- `medicina_features_2020_2024_legacy.csv`: dataset histórico 2020-2024.
- `medicina_features_2025_solo.csv`: observaciones de 2025.
- `dataset_entrenamiento_2020_2025.csv`: dataset extendido 2020-2025 con variables recalculadas.
- `agregacion_anual_2025_pre_features.csv`: agregación anual de 2025 por institución-programa.
- `inputs_prediccion_2025.csv`: inputs calculados para predecir 2025.
- `predicciones_vs_reales_test_2025.csv`: predicciones vs valores reales de 2025.
- `recomendaciones_programa.csv`: recomendaciones por institución-programa (último año disponible).
- `validation_results.json`: resumen global de validaciones (fusionado con v2).

## Nota metodológica

El modelo exportado no usa `cantidad_evaluados_*`, `PROMEDIO_PRUEBA`, `DESVIACION` ni `NIVEL1`-`NIVEL4` como predictores.
Esto lo hace más adecuado para escenarios donde se quiere predecir antes de conocer resultados o variables operativas del mismo periodo.

## Si estás en VS Code conectado a Colab

Los archivos se escriben en el runtime remoto de Colab. Si `EXPORT_STORAGE` es `google_drive`, buscalos en:

```text
Mi unidad/proyecto_medicina_artifacts
```

Si `EXPORT_STORAGE` es fallback local de Colab, buscalos en:

```text
/content/artifacts
```
