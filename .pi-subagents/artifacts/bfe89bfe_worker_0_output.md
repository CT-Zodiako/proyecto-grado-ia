Implemented la Etapa 5: entrenamiento y comparación inicial de modelos.

Changed files:
- `modelo_medicina.ipynb`
- `modelo_medicina_executed.ipynb`
- `DOCUMENTACION_EJECUCION.md`
- `validation_results.json`

Validation:
- Notebook ejecutado completo con `nbclient`: OK.
- Notebook fuente quedó sin outputs.
- Mejor modelo inicial: `Random Forest`.
- Split temporal: train `184`, validación `71`, test `66`.
- Validación: MAE `5.315`, RMSE `7.289`, R² `0.770`.
- Test: MAE `5.284`, RMSE `6.771`, R² `0.802`.

Open risks/questions:
- `cantidad_evaluados_*` se incluyó como variable operativa del mismo periodo; si la predicción debe hacerse antes de conocer evaluados reales, habría que excluirla.
- No hay repo Git en este directorio, así que no existe staging area.
- No tuve herramienta Engram disponible.

Recommended next step:
- Etapa 6: validación del modelo y análisis de errores por región, departamento, institución y programa.