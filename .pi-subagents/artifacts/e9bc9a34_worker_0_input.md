# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Continue the PDR in Spanish technical artifacts. Edit only `modelo_medicina.ipynb`, `modelo_medicina_executed.ipynb`, `DOCUMENTACION_EJECUCION.md`, and `validation_results.json` if needed. Add stage 8: Sistema de recomendaciones after IA Explicable. Do not add dashboard/Streamlit yet.

Context: We have clean Random Forest model, predictions (`test_predicciones`, `validacion_predicciones`), feature engineering (`medicina_features`), validation/error analysis, and SHAP scaffold/fallback feature importance. Recommendations must not invent information; they must be based on observed model outputs, historical trends, errors, and available fields.

Requirements:
1. Add markdown explaining the recommendation philosophy: evidence-based, descriptive/decision-support, not causal proof.
2. Build a recommendation dataset/table, e.g. `recomendaciones_programa`, at program/institution level using latest available year/test predictions when possible and historical features. Include context columns: region, department, municipality, institution, program.
3. Create rule-based recommendation categories based on evidence, for example:
   - performance risk if predicted/actual PROMEDIO_GLOBAL is below thresholds or lower quartile;
   - declining trend if `variacion_anual` is negative and historical features support it;
   - stability/volatility using `desviacion_historica_2_anios` or recent absolute variation;
   - model uncertainty/risk if absolute prediction error is high in test;
   - positive performance if high predicted/actual and stable/improving.
   Calibrate thresholds from quantiles of the data, not arbitrary constants where possible.
4. Generate a text recommendation column with clear Spanish sentences like: “El programa presenta una tendencia descendente reciente...” Include evidence values in the text. Avoid pretending causality.
5. Produce summary tables: count by recommendation category, top risk cases, top opportunity/strong cases, and groups by region/department.
6. Add validation/assertion cells: no empty recommendation text, required columns present, categories finite, no recommendations for missing core evidence without noting missing info.
7. Update `DOCUMENTACION_EJECUCION.md` with a new stage 8 section explaining the logic, code, thresholds, and observed results.
8. Execute notebook with nbclient and save `modelo_medicina_executed.ipynb`; keep source notebook outputs cleared if possible. Update `validation_results.json` with recommendation status, number of recommendations, category counts, thresholds used, and top examples.
9. If Engram save is available, save important decisions/discoveries with project `proyecto-grado-ia` before returning.

Keep recommendations academically defensible and concise. No UI/dashboard yet.

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