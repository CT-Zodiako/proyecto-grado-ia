# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Apply a strict filter to the entire Medicina project: NBC='MEDICINA' AND NOMBRE_PROGRAMA_ACAD='MEDICINA'. This replaces the previous filter that only used NBC='MEDICINA'. Update and re-execute everything consistently. Files to edit: `modelo_medicina.ipynb`, `modelo_medicina_executed.ipynb`, `DOCUMENTACION_EJECUCION.md`, `validation_results.json`, `app/api/main.py`, `app/api/model_service.py`, `app/api/schemas.py` if needed, `app/dashboard/streamlit_app.py` if needed, `artifacts/*`, `README_API.md` and `README_DASHBOARD.md` if needed.

Requirements:
1. In the notebook, change the filter to:
   ```python
   solo_medicina = df[
       df['NBC'].astype(str).str.upper().eq('MEDICINA') &
       df['NOMBRE_PROGRAMA_ACAD'].astype(str).str.upper().str.contains('MEDICINA')
   ].copy()
   ```
   Update all related markdown and validation cells to reflect that the project is now strictly about programs named MEDICINA.
2. Re-run all notebook cells with nbclient and save a fresh `modelo_medicina_executed.ipynb`. Update validation outputs (counts, metrics, recommendations) in `validation_results.json`.
3. Re-export artifacts to `artifacts/` using the new model and recommendations. This includes `model.joblib`, `feature_schema.json`, `metrics.json`, `recomendaciones_programa.csv`, `validation_results.json`, `README_artifacts.md`.
4. Update `app/api/model_service.py` and `app/api/schemas.py` if categorical values changed. Update `README_API.md` if example values changed.
5. Update `app/dashboard/streamlit_app.py` to load new recommendations and use new categorical values; adjust README_DASHBOARD.md if needed.
6. Update `DOCUMENTACION_EJECUCION.md` consistently: all stage explanations, counts, metrics, and examples must match the new strict filter. Add a note explaining why both NBC and program name are used.
7. Run local import tests for API and dashboard; if API predicts with new model, report a sample prediction.
8. If Engram save is available, save important decisions/discoveries with project `proyecto-grado-ia` before returning.

Be thorough: do not leave stale metrics or old examples from the previous NBC-only filter in the final notebook or docs. Return concise summary of the new dataset size, model metrics, recommendation counts, and files changed.

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