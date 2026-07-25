# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Actualización documental en /Users/zodiakomac/DEV/proyecto-grado-ia. Los docs están en ESPAÑOL: mantené idioma, tono, formato y estructura existente. NO hagas commits. NO toques código ni artefactos — solo los 5 archivos .md listados.

CONTEXTO VERIFICADO (usá exactamente estos hechos, no los re-derives):
- Modelo vigente: **Lasso** (pipeline sklearn: ColumnTransformer[SimpleImputer+StandardScaler para numéricas, OneHotEncoder para categóricas] + Lasso alpha=1.0), seleccionado por benchmark de 9 modelos en `mejorar_modelo.py` por menor MAE de validación. Artefactos: artifacts/model.joblib, metrics.json, feature_schema.json.
- Split temporal: train 2020-2023, validación 2024, test 2025. Dataset: artifacts/medicina_features_2020_2025.csv (373 filas, 6 años).
- Métricas del modelo vigente (Lasso): Validación 2024 → MAE 4.011, RMSE 5.424, R² 0.872. Test 2025 → MAE 3.849, RMSE 5.522, R² 0.845.
- Tabla completa del benchmark limpio (val MAE / val R² | test MAE / test R²):
  Random Forest 4.477/0.855 | 5.048/0.769 — XGBoost 4.236/0.856 | 4.949/0.775 — LightGBM 4.158/0.872 | 4.204/0.821 — CatBoost 4.552/0.844 | 4.413/0.805 — HistGradientBoosting 4.305/0.866 | 4.458/0.813 — Ridge 5.139/0.813 | 6.001/0.678 — **Lasso 4.011/0.872 | 3.849/0.845** — ElasticNet 4.166/0.851 | 4.120/0.826 — KNN 5.313/0.777 | 5.275/0.739.
- 18 features: 13 numéricas (AÑO, promedio_global_anterior, promedio_movil_2_anios, desviacion_historica_2_anios, anios_historicos_disponibles, promedio_movil_3_anios, desviacion_historica_3_anios, tasa_crecimiento_anual, maximo_historico, minimo_historico, diferencia_maximo_historico, anios_desde_inicio, ranking_departamento) + 5 categóricas (NOMBRE_REGION, NOMBRE_DEPARTAMENTO, NOMBRE_MUNICIPIO, NOMBRE_INSTITUCION, NOMBRE_PROGRAMA_ACAD).
- Lasso es esparso: solo 4 coeficientes no nulos de 105 → maximo_historico (|6.47|), promedio_global_anterior (|5.39|), promedio_movil_3_anios (|1.56|), promedio_movil_2_anios (|0.09|). Reemplaza cualquier sección de "importancia del Random Forest" por esto (Lasso no tiene feature importance de árboles; son coeficientes lineales sobre features escaladas).
- CORRECCIÓN DE LEAKAGE (2026-07-13): se detectó y corrigió target leakage exacto en 3 features que usaban el target del mismo año (tasa_crecimiento_anual, diferencia_maximo_historico, ranking_departamento). Las métricas previas del Ridge v2 (MAE 0.670/0.861, R² 0.996/0.995) eran inválidas y quedan descartadas. Sin leak, Ridge cae a MAE 5.139/6.001. Donde un doc cite esas métricas viejas, corregilas.
- API: /predict exige las 18 features YA COMPUTADAS (las engineered las calcula el llamador). Respuesta incluye "modelo": "Lasso". Un ejemplo curl con solo 10 features falla con "Faltan variables requeridas".
- El notebook `modelo_medicina.ipynb` (v1) queda como referencia histórica: entrena Baseline/Ridge/DecisionTree/RandomForest con 10 features, split train 2020-2022/valid 2023/test 2024, ganador Random Forest (MAE 5.450/5.896, R² 0.780/0.766).

TRABAJO POR ARCHIVO:

1. `README.md`
   - Tabla "Métricas principales del modelo": reemplazar con métricas Lasso (val 2024 / test 2025). Ajustar encabezados de split si dice 2023/2024.
   - "Modelo seleccionado: Random Forest Regressor" → Lasso (con nota breve: benchmark de 9 modelos, ver mejorar_modelo.py).
   - Sección "Variables más importantes": reescribir con los 4 coeficientes no nulos del Lasso (explicar que son coeficientes lineales, no importancia de árboles).
   - Ejemplo curl de /predict: actualizar a las 18 features (valores plausibles para un programa de Medicina; podés basarte en UNIVERSIDAD DE ANTIOQUIA 2025: pred ~181).
   - Limitaciones: "error promedio ~5.9 puntos" → ~3.8 puntos (test MAE 3.849). Si menciona SHAP/Random Forest, ajustar.
   - Agregar mención breve de la corrección de leakage donde encaje (p.ej. en limitaciones o métricas).

2. `documentos_para_estudiar/GUIA_PREDICCION.md`
   - Reemplazar "Random Forest Regressor" por Lasso en todas sus ocurrencias (L64, L131, L336 aprox.).
   - "usa 10 variables" → 18 variables (13 numéricas + 5 categóricas); actualizar la lista si está detallada.
   - Métricas: Test MAE 5.896 → 3.849; Test R² 0.766 → 0.845; validación 5.450 → 4.011; split Validación(2023)/Test(2024) → Validación(2024)/Test(2025).
   - Agregar una subsección sobre la corrección de leakage del 2026-07-13 (qué features, por qué las métricas viejas eran inválidas, qué se hizo).

3. `DOCUMENTACION_EJECUCION.md`
   - Este doc describe el notebook v1: NO reescribirlo entero. Agregar al inicio una nota/advertencia clara: "Este documento describe el notebook v1 (Random Forest, 2020-2024). El modelo vigente es Lasso v2 (ver README.md y GUIA_TECNICA_MODELOS.md)".
   - Corregir la línea ~2413 que dice "model.joblib … Random Forest" → Lasso.
   - Al final, agregar sección corta "Actualización v2 (2026-07)": benchmark de 9 modelos, corrección de leakage, Lasso final con sus métricas.

4. `documentos_para_estudiar/RETROSPECTIVA_FASE_POR_FASE.md`
   - Agregar una fase final nueva: "Fase de mejora v2 y corrección de leakage (2026-07)": qué se intentó (benchmark de 9 modelos, nuevas features históricas), qué salió mal (leakage exacto → métricas falsas R² 0.996), cómo se detectó (auditoría: recuperación exacta del target), cómo se corrigió (features lagueadas), resultado final (Lasso, test MAE 3.849). Tono de lección aprendida, coherente con el resto del doc.
   - Si el doc afirma en su cierre que el modelo final es Random Forest con MAE 5.450, agregar fe de erratas/actualización apuntando a la nueva fase.

5. `README_API.md`
   - Ejemplo de respuesta con "modelo": "Random Forest" → "modelo": "Lasso".
   - Si hay ejemplo de request /predict con 10 features, actualizarlo a las 18 features (mismo estilo que README.md).

VERIFICACIÓN ANTES DE TERMINAR:
- grep sobre los 5 archivos por "Random Forest", "5.450", "5.896", "0.766", "0.780", "0.996", "0.670" → las únicas ocurrencias restantes deben ser en contexto histórico v1 claramente etiquetado (DOCUMENTACION_EJECUCION nota v1, RETROSPECTIVA fases viejas, narrativa del leak).
- grep por "Ridge" → no debe aparecer como modelo vigente (solo en la historia del leak o benchmark).

ENTREGABLE: resumen por archivo de qué cambiaste (bullets), y la salida de los grep de verificación.

## Acceptance Contract
Acceptance level: checked
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Implement the requested change without widening scope

Required evidence: changed-files, tests-added, commands-run, residual-risks, no-staged-files

Finish with a fenced JSON block tagged `acceptance-report` in this shape:
Use empty arrays when no items apply; array fields contain strings unless object entries are shown.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "specific proof"
    }
  ],
  "changedFiles": [
    "src/file.ts"
  ],
  "testsAddedOrUpdated": [
    "test/file.test.ts"
  ],
  "commandsRun": [
    {
      "command": "command",
      "result": "passed",
      "summary": "short result"
    }
  ],
  "validationOutput": [
    "validation output or concise summary"
  ],
  "residualRisks": [
    "none"
  ],
  "noStagedFiles": true,
  "diffSummary": "short description of the diff",
  "reviewFindings": [
    "blocker: file.ts:12 - issue found, or no blockers"
  ],
  "manualNotes": "anything else the parent should know"
}
```