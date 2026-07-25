# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Implement the next step: a minimal FastAPI API for the exported Medicina model artifacts. Write technical artifacts in Spanish where documentation/user-facing text already uses Spanish, but code identifiers can be English if clearer.

Scope and files:
- You may create/edit: `app/api/main.py`, `app/api/schemas.py`, `app/api/model_service.py`, `app/api/__init__.py`, `app/__init__.py`, `requirements-api.txt`, `README_API.md`, and optionally small test/validation scripts under `scripts/`.
- Do not modify notebook files unless absolutely necessary.
- Use existing artifacts in `artifacts/`: `model.joblib`, `feature_schema.json`, `metrics.json`, `validation_results.json`, `recomendaciones_programa.csv`.

Requirements:
1. Implement FastAPI endpoints:
   - `GET /health`: returns status OK and artifact availability.
   - `GET /metadata`: returns model target, feature schema, best model/metrics, exported artifact info.
   - `POST /predict`: accepts one record matching the clean model input contract and returns predicted `promedio_global_anual`.
2. Prediction input must exclude `cantidad_evaluados_*`. Required fields should match `feature_schema.json` model features: numeric + categorical.
3. Use Pydantic schemas for request/response validation. Allow numeric historical fields to be optional where the model imputer can handle missing values, but categorical fields required as strings where needed.
4. Load model once with joblib in a service module. Use pandas DataFrame with columns in schema order.
5. Provide a clear `README_API.md` with how to install, run, and test from VS Code/Mac/Colab context. Include example curl/JSON body.
6. Add `requirements-api.txt` with minimal dependencies.
7. Validate by importing the app/service and performing a sample prediction using a realistic row from artifacts or a constructed input. If FastAPI/TestClient dependencies are unavailable, at least run direct service prediction. Report commands run and results.
8. If you make important discoveries/decisions, save to Engram via available memory save tool with project `proyecto-grado-ia` before returning.

Keep implementation small and robust. Do not implement dashboard yet.

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