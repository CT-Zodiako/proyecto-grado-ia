# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Continue the PDR in Spanish technical artifacts. Edit only `modelo_medicina.ipynb`, `DOCUMENTACION_EJECUCION.md`, and `validation_results.json` if needed. Add stage 4: Ingeniería de Características after the EDA section and before the final next-step note. Requirements:

1. Create an annual aggregated dataset `medicina_anual` at the level of year + institution + academic program (use stable identifiers and names). Aggregate from `solo_medicina`/`eda_data`:
   - `PROMEDIO_GLOBAL` mean as `promedio_global_anual`
   - `PROMEDIO_PRUEBA` mean as descriptive only
   - sum or mean for `CANTIDADEVALUADOS` but choose and justify carefully. Prefer a conservative aggregation that avoids double-counting across NOMBRE_PRUEBA; if unsure, compute both `cantidad_evaluados_media_pruebas` and `cantidad_evaluados_max_pruebas` and explain.
   - count distinct `NOMBRE_PRUEBA` as `cantidad_pruebas`
2. Create `medicina_features` sorted by institution/program/year with historical features that do not use future information:
   - `promedio_global_anterior`
   - `variacion_anual`
   - `variacion_porcentual`
   - `promedio_movil_2_anios` based on previous values only (shift before rolling)
   - `desviacion_historica_2_anios` based on previous values only
   - `crecimiento_acumulado_desde_inicio`
   - `mejora_vs_anio_anterior` boolean/int
   - `disminuye_vs_anio_anterior` boolean/int
   - `anios_historicos_disponibles` using cumcount
3. Add markdown explanations before/after code: what each feature means, why useful, and leakage safeguards.
4. Add validation cells: shape, null counts of engineered columns, sample rows, and assertions that previous-year features are null for first observation per program.
5. Update `DOCUMENTACION_EJECUCION.md` with a new section explaining stage 4 code and observed execution results.
6. Run the notebook or equivalent validation. Keep the notebook JSON valid and clear source notebook outputs if possible; update executed notebook only if you choose, but report clearly.
7. If you make important discoveries/decisions, save to Engram via available memory save tool with project `proyecto-grado-ia` before returning.

Do not add ML training yet.

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