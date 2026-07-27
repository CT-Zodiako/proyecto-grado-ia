# Artefactos - Modelo Medicina Saber Pro

Estos son los archivos que la API y el dashboard leen en producción. Generados por `mejorar_modelo.py` a partir de `dataset_entrenamiento_2020_2025.csv`.

## Archivos

- `model.joblib`: pipeline del modelo servido (Lasso, `alpha=1.0`, seleccionado entre 9 candidatos por menor MAE de validación tras la corrección de leakage de 2026-07-13).
- `feature_schema.json`: contrato de entrada para `/predict` — qué variables espera el modelo.
- `metrics.json`: métricas de validación/test de los 9 modelos comparados, y cuál ganó.
- `validation_results.json`: reporte histórico de validación (generación v1 — ver nota de vigencia en `documentos_para_estudiar/DICCIONARIO_DE_DATOS.md`).
- `dataset_entrenamiento_2020_2025.csv`: **dataset principal** (373 filas, 2020-2025) — entrena el modelo y alimenta el historial que usa el dashboard.
- `medicina_features_2020_2024_legacy.csv`: dataset anterior (306 filas, solo hasta 2024) — se usa únicamente como respaldo si falta el archivo principal.
- `recomendaciones_programa.csv`: una recomendación por combinación institución-programa (generación v1, ver nota de vigencia — pendiente de regenerar con el modelo actual).

## Nota metodológica

El modelo no usa `cantidad_evaluados_*`, `PROMEDIO_PRUEBA`, `DESVIACION` ni `NIVEL1`-`NIVEL4` como predictores, ni las 3 variables que originalmente tenían fuga de datos en su forma sin corregir (`tasa_crecimiento_anual`, `diferencia_maximo_historico`, `ranking_departamento` — sí se usan, pero recalculadas con datos solo hasta el año anterior).

Para el detalle de cada variable ver `documentos_para_estudiar/DICCIONARIO_DE_DATOS.md`. Para el detalle de cómo se generó cada archivo CSV (incluidos los intermedios que ya no están en este repo) ver `documentos_para_estudiar/GUIA_ARCHIVOS_CSV.md`.
