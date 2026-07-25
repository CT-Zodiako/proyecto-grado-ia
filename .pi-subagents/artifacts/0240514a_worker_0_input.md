# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Continue the PDR in Spanish technical artifacts. Edit only `modelo_medicina.ipynb`, `modelo_medicina_executed.ipynb`, `DOCUMENTACION_EJECUCION.md`, and `validation_results.json` if needed. Add stage 7: IA Explicable with SHAP after model validation. Use the clean model currently selected (`mejor_modelo`, `mejor_modelo_nombre`) and do not reintroduce `cantidad_evaluados_*` predictors.

Requirements:
1. Add notebook markdown explaining why SHAP is used and that the model explained is the clean pre-SHAP Random Forest without `cantidad_evaluados_*`.
2. Add robust setup cell:
   - Try importing `shap`.
   - If not installed, do not silently fail. In notebook, provide a commented Colab install line `# !pip install shap` and raise/print a clear instruction OR implement a safe fallback that skips SHAP plots but keeps notebook execution successful. Prefer keeping nbclient execution successful in this local environment while clearly telling Colab users how to install SHAP.
3. Extract the trained Random Forest estimator and preprocessed feature matrix from the sklearn Pipeline/ColumnTransformer. Get transformed feature names from the preprocessor.
4. Use a manageable sample from test set (and/or validation) for SHAP to avoid slow notebooks, e.g. max 100 rows.
5. Produce:
   - model-native/permutation or Random Forest feature importance table as a simpler reference;
   - SHAP feature importance table using mean absolute SHAP values when SHAP is available;
   - SHAP summary plot;
   - SHAP dependence plot for the top feature;
   - SHAP waterfall plot for one representative test prediction.
6. Add written interpretation after each output: what the table/plot means, not just code.
7. Add validation cells/assertions: feature names length matches transformed matrix columns; SHAP values shape matches sample if SHAP available; no `cantidad_evaluados_*` appears in features; required objects exist.
8. Update `DOCUMENTACION_EJECUCION.md` with a new stage 7 section explaining the code and observed results. If SHAP cannot execute locally because package unavailable, document that local validation skipped SHAP rendering but notebook includes Colab install instruction and fallback; include feature importance fallback results.
9. Execute notebook with nbclient and save `modelo_medicina_executed.ipynb`; keep source notebook outputs cleared if possible. Update `validation_results.json` with explainability status, whether SHAP was available, top features if available/fallback, and validation status.
10. If Engram save is available, save important decisions/discoveries with project `proyecto-grado-ia` before returning.

Do not add recommendations stage, dashboard, or Streamlit yet.

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