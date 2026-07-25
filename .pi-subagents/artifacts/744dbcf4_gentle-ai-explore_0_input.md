# Task for gentle-ai-explore

Auditoría read-only de consistencia documental en /Users/zodiakomac/DEV/proyecto-grado-ia. NO escribas ni modifiques nada.

Contexto: el README.md afirma "Modelo seleccionado: Random Forest Regressor" con métricas MAE 5.450/5.896 y R² 0.780/0.766. Pero el usuario dice que "utilizamos otros modelos" y existen señales de cambio: directorio `catboost_info/`, script `mejorar_modelo.py`, `modelo_medicina_executed.ipynb`.

Tu tarea: determinar cuál es el modelo REALMENTE usado hoy y qué documentos están desactualizados.

Archivos a inspeccionar (evidencia de la realidad):
1. `artifacts/metrics.json` — qué modelo y métricas reporta
2. `validation_results.json` (raíz) — idem
3. `mejorar_modelo.py` — qué hace, qué modelo entrena/guarda y dónde
4. `app/api/model_service.py` — qué archivo de modelo carga la API en producción (¿model.joblib? ¿otro?)
5. `artifacts/` — listar contenido: ¿hay model.joblib, catboost, otros .joblib/.cbm?
6. `catboost_info/` — listar para confirmar que se entrenó CatBoost
7. `documentos_para_estudiar/GUIA_TECNICA_MODELOS.md` — qué modelo declara como seleccionado
8. `documentos_para_estudiar/GUIA_PREDICCION.md` — qué modelo y métricas declara
9. `DOCUMENTACION_EJECUCION.md` — qué modelo y métricas declara
10. `README_API.md`, `README_DASHBOARD.md`, `app/README.md`, `artifacts/README_artifacts.md`, `documentos_para_estudiar/RETROSPECTIVA_FASE_POR_FASE.md`, `STRICT_FILTER_NOTE.md` — escaneo rápido: ¿mencionan Random Forest, métricas viejas, o modelos que ya no aplican?
11. Si es rápido: `grep` dentro de `modelo_medicina_executed.ipynb` por "CatBoost", "Random Forest", "best_model", "modelo_final" para ver qué quedó como modelo final ejecutado. No ejecutes el notebook.

Entregable (en español):
- **Modelo real actual**: nombre, archivo artefacto que lo contiene, métricas reales (fuente: metrics.json / validation_results.json).
- **Tabla de consistencia**: para cada documento listado arriba → ¿Actualizado / Desactualizado / Parcial? + evidencia concreta (línea o valor que contradice la realidad).
- **Lista priorizada de documentos a actualizar** con qué corregir en cada uno (qué valor viejo → qué valor nuevo).
- Si hay contradicción entre artefactos (ej. metrics.json vs validation_results.json vs lo que carga la API), señalala explícitamente.

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