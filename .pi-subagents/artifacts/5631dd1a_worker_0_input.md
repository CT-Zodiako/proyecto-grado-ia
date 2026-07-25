# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Implement the next PDR stage in this project. Files to edit: `modelo_medicina.ipynb` and `DOCUMENTACION_EJECUCION.md` only. Add a documented EDA section to the notebook after the current dataset/model_data setup. Include code cells for: PROMEDIO_GLOBAL distribution, yearly evolution 2020-2024, region comparison, department comparison, top institutions by mean PROMEDIO_GLOBAL with enough records, CANTIDADEVALUADOS distribution, outlier detection for PROMEDIO_GLOBAL using IQR, and relationship between CANTIDADEVALUADOS and PROMEDIO_GLOBAL. Use pandas + matplotlib + seaborn only; set figure sizes and titles. Add markdown interpretation prompts/notes after plots, not just code. Then update `DOCUMENTACION_EJECUCION.md` by complementing it with a new EDA section explaining each added block and expected/observed results. Run/validate the notebook or equivalent Python snippets if possible, and report exact files changed and validation summary. Keep technical artifacts in Spanish because the existing document is Spanish. If you make important discoveries, decisions, or fix bugs, save them to Engram via the available memory save tool with project: 'proyecto-grado-ia' before returning.

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