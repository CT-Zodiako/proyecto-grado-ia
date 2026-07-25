# Task for scout

Auditoría READ-ONLY de consistencia documental en /Users/zodiakomac/DEV/proyecto-grado-ia. No escribas ni modifiques nada, no ejecutes el notebook.

Contexto: README.md afirma "Modelo seleccionado: Random Forest Regressor" con MAE 5.450/5.896 y R² 0.780/0.766. El usuario dice que "utilizamos otros modelos". Hay directorio catboost_info/ y script mejorar_modelo.py.

Objetivo: determinar el modelo REALMENTE usado hoy y qué documentos están desactualizados.

Inspeccioná:
1. artifacts/metrics.json — modelo y métricas reales
2. validation_results.json (raíz) — idem
3. mejorar_modelo.py — qué hace, qué modelo entrena/guarda y dónde
4. app/api/model_service.py — qué archivo de modelo carga la API
5. ls artifacts/ y ls catboost_info/ — qué artefactos de modelo existen (.joblib/.cbm)
6. grep acotado en modelo_medicina_executed.ipynb por "CatBoost", "RandomForest", "best_model", "modelo_final"
7. Escaneo de claims en docs: documentos_para_estudiar/GUIA_TECNICA_MODELOS.md, documentos_para_estudiar/GUIA_PREDICCION.md, DOCUMENTACION_EJECUCION.md, README_API.md, README_DASHBOARD.md, app/README.md, artifacts/README_artifacts.md, documentos_para_estudiar/RETROSPECTIVA_FASE_POR_FASE.md — ¿qué modelo/métricas declara cada uno?

Entregable (español, conciso):
- MODELO REAL ACTUAL: nombre, archivo artefacto, métricas reales (con fuente)
- TABLA: documento → Actualizado/Desactualizado/Parcial + evidencia concreta (valor contradictorio)
- LISTA PRIORIZADA de docs a corregir: valor viejo → valor nuevo
- Contradicciones entre artefactos (metrics.json vs validation_results.json vs lo que carga la API) si existen

---
**Output:**
Write your findings to exactly this path: /Users/zodiakomac/DEV/proyecto-grado-ia/.pi-subagents/artifacts/outputs/075c3e94/context.md
This path is authoritative for this run.
Ignore any other output filename or output path mentioned elsewhere, including output destinations in the base agent prompt, system prompt, or task instructions.

## Acceptance Contract
Acceptance level: attested
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Return concrete findings with file paths and severity when applicable

Required evidence: review-findings, residual-risks

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