# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Continue the PDR in Spanish technical artifacts. Edit only `modelo_medicina.ipynb`, `modelo_medicina_executed.ipynb`, `DOCUMENTACION_EJECUCION.md`, and `validation_results.json` if needed. Add stage 6: Validación del modelo and error analysis after model training. Do not add SHAP yet.

Requirements:
1. Use `mejor_modelo`, `mejor_modelo_nombre`, `X_valid`, `y_valid`, `X_test`, `y_test`, and the corresponding `valid_data`/`test_data` from stage 5.
2. Create prediction tables for validation and test, e.g. `validacion_predicciones` and `test_predicciones`, including actual, prediction, residual/error, absolute error, squared error, and context columns: AÑO, region, departamento, municipio, institución, programa.
3. Add residual diagnostics: distribution of residuals, actual vs predicted scatter, residuals vs predicted plot.
4. Add grouped error analysis by region, department, institution, and academic program. Use MAE, RMSE, bias/mean residual, n. Avoid overinterpreting groups with very low n; include `MIN_GRUPO_VALIDACION` threshold and explain.
5. Add overfitting/generalization check comparing train/validation/test metrics for best model and baseline. Use existing `resultados_modelos`.
6. Add a concise conclusion: whether model generalizes reasonably, where errors are larger, and risks/limitations.
7. Add validation/assertion cells: prediction row counts equal y lengths, no null predictions, required columns present, metrics finite.
8. Update `DOCUMENTACION_EJECUCION.md` with a new section explaining stage 6 code and observed execution results.
9. Run notebook with nbclient and save `modelo_medicina_executed.ipynb`; keep source notebook outputs cleared if possible. Update `validation_results.json` with validation-stage summary: best model, test MAE/RMSE/R2, largest-error groups, residual mean, validation status.
10. If you make important discoveries/decisions, save to Engram via available memory save tool with project `proyecto-grado-ia` before returning.

Keep it educational and clear. No SHAP, no dashboard, no recommendations stage yet.

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