## Nota sobre el filtro estricto de Medicina

Este proyecto utiliza un filtro estricto para seleccionar únicamente programas de **Medicina**:

```python
solo_medicina = df[
    df['NBC'].astype(str).str.upper().eq('MEDICINA') &
    df['NOMBRE_PROGRAMA_ACAD'].astype(str).str.upper().str.contains('MEDICINA')
].copy()
```

Esto garantiza que el análisis, modelo y recomendaciones se centren exclusivamente en programas académicos cuyo nombre contiene 'MEDICINA', no en otros programas del mismo núcleo de conocimiento (como Odontología, Biotecnología, etc.).

Resultados del filtro estricto:
- Filas seleccionadas: 2,173 (vs 2,250 con solo NBC)
- Programas únicos: 70 (vs 75 con solo NBC)
- Recomendaciones: 70 (vs 75 con solo NBC)

---

