# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Continue the PDR in Spanish technical artifacts. Edit only `modelo_medicina.ipynb`, `modelo_medicina_executed.ipynb`, `DOCUMENTACION_EJECUCION.md`, and `validation_results.json` if needed. Add stage 10: Exportación de artefactos for a user running the notebook from VS Code connected to a Colab runtime.

Requirements:
1. Add notebook section after stage 9: `## 10. Exportación de artefactos`.
2. Explain VS Code + Colab runtime distinction: Python writes to Colab remote filesystem; use Google Drive for persistence, with local `/content/artifacts` fallback.
3. Add robust code to choose export directory:
   - `USE_GOOGLE_DRIVE = True` default.
   - Try mounting Drive only if running in Colab and Google Drive requested.
   - If Drive unavailable/not Colab, fallback to local `artifacts/` or `/content/artifacts` depending environment.
   - Do not make local nbclient execution fail due missing `google.colab`.
4. Export artifacts:
   - `model.joblib`: `mejor_modelo` clean pipeline.
   - `feature_schema.json`: numeric_features, categorical_features, excluded features, target, split strategy, clean model note.
   - `metrics.json`: model results, best model, validation/test metrics, validation summary.
   - `recomendaciones_programa.csv`.
   - `validation_results.json` from notebook data/current known values if possible. If reading existing file is easier, export/update it safely.
   - Optional: `README_artifacts.md` explaining files.
5. Add validation cell: assert exported files exist and non-empty, model can be loaded with joblib, schema includes no `cantidad_evaluados_*` predictors, recommendations CSV has rows.
6. Update `validation_results.json` with `artifact_export_validation: OK`, export file names, export dir used during local validation, artifact count.
7. Update `DOCUMENTACION_EJECUCION.md` with stage 10 explanation, exact code behavior for VS Code+Colab, and instructions for finding files in Drive.
8. Execute notebook with nbclient and save `modelo_medicina_executed.ipynb`; keep source notebook outputs cleared if possible. Local execution should export to a safe local folder, not require Drive.
9. If Engram save is available, save important decisions/discoveries with project `proyecto-grado-ia` before returning.

Do not implement API/dashboard files yet.

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