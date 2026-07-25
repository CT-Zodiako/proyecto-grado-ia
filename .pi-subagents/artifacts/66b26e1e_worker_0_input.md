# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Continue the PDR in Spanish technical artifacts. Edit only `modelo_medicina.ipynb`, `modelo_medicina_executed.ipynb`, `DOCUMENTACION_EJECUCION.md`, and `validation_results.json` if needed. Add stage 9: Arquitectura del sistema for the product direction API + Dashboard. Do not implement actual FastAPI/Streamlit files yet; this is architecture/design stage.

Requirements:
1. Add a notebook section after stage 8: `## 9. Arquitectura del sistema: API + Dashboard`.
2. Explain recommended stack: FastAPI, Streamlit, joblib, pandas, scikit-learn, optional SHAP, file-based artifacts first; PostgreSQL optional later.
3. Include clear architecture diagram(s) in markdown using Mermaid or ASCII. Must show: notebook/training pipeline -> exported artifacts -> FastAPI -> Streamlit dashboard -> user. Also show artifact/data flow.
4. Define proposed repository structure, e.g. `app/api`, `app/dashboard`, `app/artifacts`, `app/services`, `data`, `reports`.
5. Define artifacts to export from notebook: model.joblib, feature schema JSON, metrics JSON, recommendations CSV, validation_results JSON, optional SHAP outputs.
6. Define API endpoints with method, path, purpose, input/output summary: `/health`, `/metadata`, `/predict`, `/recommend`, `/metrics/model`, `/summary/regions`, `/summary/departments`.
7. Define dashboard pages/sections: overview, EDA, prediction, recommendations, validation/errors, explainability.
8. Define model serving contract and input schema fields based on clean model features (no `cantidad_evaluados_*`). Be explicit about required fields and what can be selected from known categories.
9. Define non-goals and risks: not causal, not production-grade auth yet, data drift, missing SHAP install, model retraining, privacy.
10. Add validation cell(s) that create Python dictionaries/lists for `arquitectura_componentes`, `api_endpoints`, `dashboard_paginas`, `artefactos_exportacion`, `contrato_prediccion`, and assert required keys/paths/endpoints are present. This keeps the architecture auditable.
11. Update `DOCUMENTACION_EJECUCION.md` with a new stage 9 section explaining architecture decisions, diagrams, endpoint table, artifacts, and next implementation step.
12. Execute notebook with nbclient and save `modelo_medicina_executed.ipynb`; keep source notebook outputs cleared if possible. Update `validation_results.json` with `architecture_validation: OK`, stack list, endpoint count, dashboard page count, artifact count.
13. If Engram save is available, save important decisions/discoveries with project `proyecto-grado-ia` before returning.

Keep it concise, academically clear, and suitable for later implementation.

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