# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Implement the next step: a minimal Streamlit dashboard for the Medicina Saber Pro project. Write technical artifacts in Spanish where the project already uses Spanish.

Scope and files:
- You may create/edit: `app/dashboard/streamlit_app.py`, `app/dashboard/__init__.py`, `requirements-dashboard.txt`, `README_DASHBOARD.md`, and optionally small helper scripts.
- Do not modify notebook files unless absolutely necessary.
- The dashboard should consume the FastAPI API endpoints /health, /metadata, /predict.

Requirements:
1. Create `app/dashboard/` package with `__init__.py` and `streamlit_app.py`.
2. Add pages/sections using streamlit radio sidebar:
   - **Overview**: project title, period, data source, clean model status from /health and /metadata.
   - **EDA**: load `recomendaciones_programa.csv` locally and show histogram of `promedio_global_anual`, distribution by region, and top institutions; include cached @st.cache_data loader.
   - **Predicción**: form with inputs for all numeric and categorical fields from /metadata, call /predict, show result and interpretation (above/below thresholds).
   - **Recomendaciones**: load `recomendaciones_programa.csv` and show category counts, filterable table by category/region/department, and top risk cases.
   - **Validación**: display validation_results.json or metrics.json metrics and model comparison if available.
   - **Explicabilidad**: show feature importances from `feature_schema.json` or a fallback CSV; mention that SHAP plots render only if shap is installed.
3. Use requests to call the API. Allow API base URL configurable via environment variable or sidebar input (default `http://localhost:8000`). Gracefully handle API not available with a warning message.
4. Add `requirements-dashboard.txt` with streamlit, pandas, requests, plotly or matplotlib, and optionally shap.
5. Create `README_DASHBOARD.md` with instructions to run in local or Colab, including how to run both API and dashboard together.
6. Validate that the dashboard module imports without errors. If streamlit is not installed, use a lightweight import test after installing. Since user previously agreed to install deps locally, try installing requirements-dashboard.txt if needed. Report the result and any limitations.
7. If you make important discoveries/decisions, save to Engram via available memory save tool with project `proyecto-grado-ia` before returning.

Keep dashboard simple and educational. No production auth.

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