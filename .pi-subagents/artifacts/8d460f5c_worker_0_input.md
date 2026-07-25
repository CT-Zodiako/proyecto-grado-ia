# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Implement the recommended methodological correction before SHAP. Edit only `modelo_medicina.ipynb`, `modelo_medicina_executed.ipynb`, `DOCUMENTACION_EJECUCION.md`, and `validation_results.json` if needed.

Context: Stage 5/6 currently include `cantidad_evaluados_media_pruebas` and `cantidad_evaluados_max_pruebas` as model features. The parent recommendation is to remove these same-period variables before SHAP, because the intended prediction may need to happen before knowing the number of evaluated students.

Requirements:
1. Update stage 5 model feature selection so `cantidad_evaluados_media_pruebas` and `cantidad_evaluados_max_pruebas` are excluded from `numeric_features` / model predictors.
2. Keep these variables in EDA/feature engineering as descriptive context, but not as model inputs.
3. Re-run the temporal split and compare the same models: Baseline mean, Ridge, Decision Tree, Random Forest.
4. Recompute best model by validation MAE and preserve `mejor_modelo_nombre`, `mejor_modelo`, `resultados_modelos` semantics for later SHAP.
5. Update stage 6 validation/error analysis so it uses the recalculated clean model outputs.
6. Update markdown explanations in the notebook: explicitly say this is the cleaner pre-SHAP model without same-period evaluated-count features.
7. Update `DOCUMENTACION_EJECUCION.md` with a correction note and the new observed metrics. Do not leave contradictory old claims as final truth; if preserving old metrics, label them as previous benchmark superseded.
8. Update `validation_results.json`: include new clean-model feature list, excluded operational variables, best model and metrics. Keep old metrics only if clearly labeled as previous/superseded.
9. Execute notebook with nbclient and save `modelo_medicina_executed.ipynb`; keep source notebook outputs cleared if possible.
10. Return concise summary with changed files, new best model, validation/test metrics, and whether performance changed.
11. If Engram save is available, save important decisions/discoveries with project `proyecto-grado-ia` before returning.

Do not add SHAP yet.

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