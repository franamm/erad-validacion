# Consulta de resultados por caso

Los archivos están organizados dentro de `artefactos/`:

- `validacion/`: estudio de la tesis, barridos y casos base existentes.
- `oberrhein/`: segunda red y sus corridas guardadas.

El selector **Caso de estudio** cambia la carpeta que leen **Optimización y análisis**
y **Resultados**. No reinicia el servidor, no ejecuta el pipeline ni vuelve a resolver
optimizaciones. Optimización y análisis lee la red, E4, los estudios y los casos base del caso elegido.
Los nuevos trabajos reciben esa misma ruta de forma explícita. Cambiar el selector
no lanza trabajos ni vuelve a resolver corridas guardadas.

Los barridos del modelo cargado conservan sus vistas de análisis. Para consultar
otro caso, o un caso sin barridos, se ofrecen directamente los resultados guardados:
CSV, series NPZ, `caso_base_*h.pkl`, `prueba_milp.pkl` y corridas `.pkl.gz`.
Los pickle se reducen a tablas horarias para visualizar ENS y potencia, sin
conservar el objeto completo en la caché. Los pickle grandes pueden tardar al abrirse.

La caché distingue la ruta completa, fecha de modificación y tamaño del archivo.
**Actualizar archivos** vuelve a leer los datos; no calcula corridas faltantes.

Los nombres anteriores `artefactos` y `artefactos_oberrhein` se reconocen al
resolver las rutas del pipeline. Los nuevos casos se crean dentro de `artefactos/`.
