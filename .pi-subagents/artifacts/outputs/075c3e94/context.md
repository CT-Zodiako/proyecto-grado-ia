# Auditoría de consistencia documental — Modelo real vs. documentos

Fecha de auditoría: 2026-07-13 · Modo: READ-ONLY (no se modificó nada, no se ejecutó el notebook)

---

## 1. MODELO REAL ACTUAL (lo que la API carga HOY)

**Modelo: Ridge Regression (v2)** — NO Random Forest.

| Aspecto | Valor | Fuente |
|---|---|---|
| Archivo de modelo que carga la API | `artifacts/model.joblib` | `app/api/model_service.py:31` (`MODEL_PATH = ARTIFACTS_DIR / "model.joblib"`) |
| Identidad del modelo | `model_name: "Ridge"` | `artifacts/feature_schema.json` (el schema que la API carga) |
| Tamaño del artefacto | **8506 bytes** (idéntico a `model_v2.joblib` = 8506 bytes) | `stat` de `artifacts/` |
| Comparación con v1 | El viejo Random Forest `model_v1.joblib` pesa **544195 bytes**; un RF de 300 árboles no puede pesar 8.5 KB → confirma que `model.joblib` es el pipeline lineal Ridge | `stat` de `artifacts/` |
| Confirmación explícita | "`model.joblib`: pipeline del mejor modelo seleccionado (actualmente **Ridge v2**)." | `artifacts/README_artifacts.md` |

### Métricas reales del modelo actual (Ridge v2)
Fuente: `artifacts/metrics.json` y `artifacts/metrics_v2.json` (idénticos), split `temporal_train_2020_2023_valid_2024_test_2025`:

| Split | MAE | RMSE | R² |
|---|---:|---:|---:|
| Validación (2024) | **0.670** | 0.956 | **0.996** |
| Test (2025) | **0.861** | 1.023 | **0.995** |

`best_model_by_validation_mae: "Ridge"` en ambos archivos.

### Cómo se llegó a este modelo
- `mejorar_modelo.py` entrena **9 modelos** (Random Forest, XGBoost, LightGBM, **CatBoost**, HistGradientBoosting, Ridge, Lasso, ElasticNet, KNN) sobre `artifacts/medicina_features_2020_2025.csv` con 13 features numéricas nuevas, selecciona el de menor MAE de validación (**Ridge = 0.670**) y exporta `model_v2.joblib`, `metrics_v2.json`, `feature_schema_v2.json`.
- El directorio `catboost_info/` es el subproducto de haber entrenado CatBoost en ese script (CatBoost fue probado pero **NO** seleccionado: quedó 4º con Test MAE 2.39).
- Luego hubo una **promoción** (2026-07-06 20:01:51) que copió los artefactos v2 a los nombres sin sufijo que la API consume: `model.joblib`, `metrics.json`, `feature_schema.json`.

### Qué hace el notebook (modelo v1, ya no es el cargado)
`modelo_medicina_executed.ipynb` solo entrena Baseline / Ridge / Decision Tree / Random Forest (sin CatBoost/XGBoost/LightGBM) y selecciona **Random Forest** (Validación MAE 5.509, Test MAE 5.486, split 2023/2024). Ese es el modelo **v1** (`model_v1.joblib`), hoy solo de referencia. El claim del README (MAE 5.450/5.896) corresponde a una corrida v1 con split validación 2023 / test 2024.

**Conclusión:** el usuario tiene razón — "utilizamos otros modelos". El modelo en producción hoy es **Ridge v2** (benchmark de 9 modelos), y el README está desactualizado.

---

## 2. CONTRADICCIONES ENTRE ARTEFACTOS (importante)

| Artefacto | Qué declara | Estado |
|---|---|---|
| `artifacts/metrics.json` + `artifacts/metrics_v2.json` | best = **Ridge**, MAE 0.670/0.861 | ✅ Actual (v2) |
| `artifacts/feature_schema.json` | `model_name: "Ridge"`, 13 num + 5 cat | ✅ Actual (v2) |
| `artifacts/model.joblib` (8506 B) | pipeline Ridge v2 | ✅ Actual (v2) |
| `artifacts/validation_results.json` | best = **Random Forest**, MAE 5.45/5.896, R² 0.78/0.766 | ❌ **STALE (v1)** — modificado 20:03:07 (tras la promoción v2) pero su contenido sigue siendo del notebook v1 |
| `validation_results.json` (raíz) | idem Random Forest 5.45/5.896 | ❌ STALE (v1), modificado 17:32:53 |

### Impacto vivo en la API (bug de consistencia)
`app/api/model_service.py:106-108`:
```python
@property
def best_model_name(self) -> str | None:
    return self.validation.get("best_model_by_validation_mae") or self.metrics.get("best_model")
```
- Lee `validation_results.json` **primero** → devuelve **"Random Forest"**, aunque el modelo cargado es Ridge.
- El fallback `self.metrics.get("best_model")` **nunca se ejecuta**: (a) `validation` ya devolvió un valor, y (b) `metrics.json` usa la clave `best_model_by_validation_mae`, no `best_model`.
- **Resultado:** `/predict` y `/metadata` reportan `"modelo": "Random Forest"` mientras sirven predicciones de **Ridge**. Inconsistencia real en runtime, no solo documental.

---

## 3. TABLA DE DOCUMENTOS

| Documento | Estado | Evidencia concreta (valor contradictorio) |
|---|---|---|
| `README.md` | ❌ **Desactualizado** | L168 "Modelo seleccionado: Random Forest Regressor"; L164-166 MAE 5.450/5.896, R² 0.780/0.766; L176 "Según la importancia del Random Forest" |
| `documentos_para_estudiar/GUIA_PREDICCION.md` | ❌ **Desactualizado** | L131 "Random Forest Regressor"; L64 "Entrenamiento del modelo Random Forest"; L80 "usa 10 variables"; L166 Test MAE=5.896; L198 Test R²=0.766; L209-210 split Validación(2023)/Test(2024); L336 "Algoritmo: Random Forest" |
| `DOCUMENTACION_EJECUCION.md` | ❌ **Desactualizado** | L1358/L1405/L1417/L1855/L2731 "Mejor modelo limpio: Random Forest"; L1348-1349 val 5.450 / test 5.896; L2413 "model.joblib … + Random Forest"; L1560/1574/1654 Random Forest. Documenta el notebook v1. |
| `documentos_para_estudiar/RETROSPECTIVA_FASE_POR_FASE.md` | ❌ **Desactualizado (no refleja v2)** | L259/L268/L605/L667/L681 "Random Forest"; L682-685 MAE 5.450/R² 0.780, Test 5.896/0.766. No menciona Ridge v2 ni `mejorar_modelo.py`. |
| `README_API.md` | ⚠️ **Parcial** | L85 ejemplo de respuesta `"modelo": "Random Forest"` (coincide con el bug de runtime, pero el modelo real es Ridge). |
| `artifacts/README_artifacts.md` | ✅ **Actualizado** | "model.joblib … (actualmente **Ridge v2**)"; lista model_v1/v2 y metrics_v1/v2. |
| `documentos_para_estudiar/GUIA_TECNICA_MODELOS.md` | ✅ **Actualizado** | L11 "El modelo final fue **Ridge Regression** … MAE de validación (0.67)"; tabla con los 9 modelos (Ridge 0.67/0.86/0.995, XGBoost, CatBoost, LightGBM, RF 2.45/2.62/0.920); split train 2020-2023/valid 2024/test 2025. |
| `README_DASHBOARD.md` | ➖ Neutral / OK | No declara modelo ni métricas; solo menciona una página "Modelos". |
| `app/README.md` | ➖ Neutral / OK | Lista artefactos (`model.joblib`, etc.) sin nombrar modelo ni métricas. |

---

## 4. LISTA PRIORIZADA DE CORRECCIONES (valor viejo → valor nuevo)

### Prioridad 1 — Identidad del modelo (user-facing, alto impacto)
1. **`README.md:168`** — "Modelo seleccionado: Random Forest Regressor" → "**Ridge Regression (v2)**".
2. **`README.md:164-166`** — tabla de métricas → MAE 5.450→**0.670** (val) / 5.896→**0.861** (test); RMSE 7.143→**0.956** / 7.355→**1.023**; R² 0.780→**0.996** / 0.766→**0.995**. Actualizar también etiquetas de split: Validación 2023→**2024**, Test 2024→**2025**.
3. **`README.md:176`** — "Según la importancia del Random Forest" → Ridge es lineal (coeficientes, no importancia de árboles); reescribir la sección de variables importantes.
4. **`README_API.md:85`** — ejemplo `"modelo": "Random Forest"` → `"modelo": "Ridge"`.

### Prioridad 2 — Runtime / artefactos (corrige el bug, no solo el texto)
5. **Regenerar `artifacts/validation_results.json` y la copia en raíz** para reflejar Ridge v2, **o** corregir `app/api/model_service.py:106-108` para que `best_model_name` lea `metrics.json` (`best_model_by_validation_mae`). Hoy la API sirve Ridge pero reporta "Random Forest".

### Prioridad 3 — Guías y retrospectiva
6. **`GUIA_PREDICCION.md`** — "Random Forest Regressor"→"Ridge Regression"; "10 variables"→"**18 variables** (13 numéricas + 5 categóricas)"; Test MAE 5.896→**0.861**; Test R² 0.766→**0.995**; split Validación(2023)/Test(2024)→**Validación(2024)/Test(2025)**; actualizar L64, L147, L300, L336, L367.
7. **`DOCUMENTACION_EJECUCION.md`** — aclarar que describe el notebook **v1** (Random Forest) y añadir sección v2/Ridge; corregir L2413 ("model.joblib … Random Forest" → Ridge).
8. **`RETROSPECTIVA_FASE_POR_FASE.md`** — añadir fase final de mejora v2 (benchmark de 9 modelos, selección Ridge) o anotar que 5.450/5.896 corresponden a v1.

---

## 5. Riesgos / preguntas abiertas (fuera del alcance doc, pero conviene flaggear)
- **R² 0.996 (val) / 0.995 (test) del Ridge v2 es inusualmente alto.** `GUIA_TECNICA_MODELOS.md` lo atribuye a linealidad fuerte, pero un salto de R² ~0.77 (v1) a ~0.99 (v2) con MAE 5.45→0.67 merece una verificación de **target leakage** en las nuevas features v2 (p.ej. `promedio_global_anterior`, `promedio_movil_*`, `maximo_historico`) antes de publicar estas métricas como definitivas. No se pudo re-ejecutar (auditoría read-only).
- `catboost_info/` es residuo de ejecución de `mejorar_modelo.py`; no implica que CatBoost esté en producción.
- La discrepancia `metrics.json` (Ridge) vs `validation_results.json` (Random Forest) confundirá a cualquier consumidor que lea `validation_results.json` (incluida la propia API).
