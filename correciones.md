esto de overview se va Este dashboard te muestra, en lenguaje simple, cómo le fue a un programa de Medicina, por qué le fue así, y qué tan confiable es esa lectura.
[x] HECHO — se eliminó esa línea de intro en Overview.

en metadatos del modelo de overview se muestra AÑO ocn la o abajo
[x] HECHO — era un bug de renderizado de la fuente monoespaciada (el glifo de "Ñ" en fuente de código no calzaba bien con la tilde). Se sacó el formato de código (backticks) de la lista de variables en "Metadatos del Modelo"; ahora se ve en texto normal, sin el problema.

en hallazgos de rigor la fugade tatos explcica tmabien el tarjet leakage que encima de la plaba al apsar el clci apareca un ventand para explicar que es igual para todas esas palbras tecicas que no estan explcicdas
[x] HECHO — "target leakage" ahora tiene un tooltip al pasar el mouse (subrayado punteado) explicando qué es. También se agregó el mismo tipo de tooltip (ícono ⓘ) a las métricas Test MAE / Test RMSE / Test R² en la sección "El hallazgo principal" de Overview, que eran del mismo tipo de jerga técnica sin explicar ahí. Si hay otras palabras técnicas puntuales que falten, decime cuáles y las agrego con el mismo patrón.
