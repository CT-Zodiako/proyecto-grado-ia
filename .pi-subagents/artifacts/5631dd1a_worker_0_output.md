Implemented la etapa EDA del PDR.

Changed files:
- `modelo_medicina.ipynb`
- `DOCUMENTACION_EJECUCION.md`

Validation:
- Notebook ejecutado en memoria con `nbclient`: OK.
- Validaciones EDA por snippets: OK.
- JSON del notebook válido: OK.

Open risks/questions:
- No pude guardar en Engram porque este subagente no tiene herramienta `mem_save` disponible.
- El directorio no está dentro de un repo Git, así que `git status` no puede verificar staging.

Recommended next step:
- Continuar con **Ingeniería de Características**.