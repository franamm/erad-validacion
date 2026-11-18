# Plan de reorganización de la herramienta de resiliencia

> **Reemplazado el 2026-09-15.** La interfaz se rehizo reutilizando las vistas existentes; el
> estado vigente está en `README_APP_V8.md`. Este documento queda como antecedente de diseño.

## Estado de implementación (2026-09-15)

Revisión de presentación solicitada: se recuperó la vista de entradas de la
v7 y la navegación superior de tres pestañas. Se plegaron controles avanzados
y tablas extensas; Resultados abre directamente en Red y ENS y conserva Ver
una corrida. La guía se trasladó al pie de la herramienta.

La v8 se implementó en `app_v8_estudio.py` y `estudio_v8/`; el archivo de
entrada dejó de ser una copia idéntica y ahora inicia la interfaz modular.
Las 12 medidas fueron confirmadas por el usuario:
`reconfig_topo,dg_sol,dg_eol,dg_fos,ess_4h,ess_5h,ess_6h,meg,mess,edrp,refuerzo,soterramiento`.
La guía operativa vigente y las limitaciones están en `README_APP_V8.md`.

Se integraron ejecución persistente, resultados y matrices; se reutilizan
los motores existentes. La autonomía se ofrece desde artefactos preparados,
con exportación de código e insumos. La reconstrucción climática completa
continúa requiriendo sus fuentes externas. Los apartados siguientes conservan
la propuesta original como referencia de diseño.

## Alcance de la propuesta original

Se creó `app_v8_estudio.py` como copia exacta de `app_v7_carga_corregida.py`.
La versión original se conserva. La copia todavía comparte módulos y artefactos
con ella: no es aún un paquete independiente ni implementa este rediseño.
Este documento especifica la propuesta de modificación solicitada.

Objetivo: completar desde la herramienta el recorrido **datos → estudio →
ejecución → resultados → selección de un plan**, con trazabilidad hasta la
corrida y un comando reproducible para cada cálculo.

## 1. Estructura de la interfaz

### Datos de entrada

Conservar las vistas actuales de modelo cargado y preparación de datos.
Añadir únicamente una ficha de estado: red, versión de fallas, horizonte,
escenarios disponibles, costos, variante de aislación, límites de DG y casos
base compatibles. Si falta una etapa, mostrar qué falta y permitir prepararla.
La aplicación debe poder abrir esta pantalla aunque falten artefactos;
actualmente `data_loader.py` puede fallar al importarse.

### Optimización y análisis

Sustituir Optimización y la parte de ejecución de Barrido por un único flujo:

1. **Elegir estudio:** corrida puntual; medidas individuales; combinación;
   evaluación de planes fijos. Incluir una receta «Estudio de 12 medidas en
   10 escenarios», cuyo catálogo y muestra quedan visibles y guardados.
2. **Configurar:** medidas habilitadas, escenarios y supuestos. Opciones
   avanzadas plegadas: límites, tiempos, tolerancia del optimizador, recursos,
   equipos móviles y nivel de detalle guardado.
3. **Elegir productos:** indicadores, curvas de abatimiento, solapamiento,
   rangos de inversión y matriz de decisión. Explicar qué productos solo leen
   resultados y cuáles requieren nuevas optimizaciones.
4. **Revisar y ejecutar:** mostrar corridas totales, existentes y pendientes,
   dependencias, salida y comando. Un botón inicia o reanuda las pendientes.
5. **Seguimiento:** progreso persistente, registro de ejecución, errores,
   detener/reanudar y acceso directo a «Ver una corrida».

Reutilizar `barrido_lanzador.py`: ya lanza procesos separados y registra su
estado. Generalizarlo para corrida puntual, pipeline y plan fijo; proteger
salidas frente a escrituras simultáneas y distinguir detenido, fallido,
factible con límite y terminado. El avance no dependerá de `session_state`.
La supervivencia al reinicio del equipo requiere reanudación desde disco.

### Resultados

Cabecera común: estudio, caso/medida/combinación, plan si aplica, escenario
individual o conjunto, hora o resumen temporal, zona y caso base asociado.
Mostrar siempre cobertura y calidad: por ejemplo, 8 de 10 escenarios válidos.
Separar el escenario usado para diseñar un plan del usado para evaluarlo.

| Vista | Contenido |
|---|---|
| Resumen | Pregunta del estudio, alcance, indicadores principales, cobertura y resultados disponibles |
| Red y plan instalado | Grafo horario, resumen temporal, ubicación, capacidad y tabla de inversiones |
| Indicadores | ENS, SAIDI, SAIFI, ASAI; CAPEX, OPEX, costo de ENS, costo total y beneficio, con unidades |
| ENS territorial y temporal | ENS por zona y por hora, comparación con el caso base; agregación diaria/mensual |
| Medidas y abatimiento | Comparación individual, MAC de ENS/SAIDI/SAIFI y tablas descargables |
| Combinaciones | Medidas habilitadas y efectivamente instaladas/usadas, retención, solapamiento y comparación entre combinaciones |
| Rangos y decisión | Variabilidad de planes, matriz plan × escenario, matriz de arrepentimiento y ranking por criterio |
| Ver una corrida | Conservar la experiencia actual del Barrido y ampliarla a todas las fuentes de resultados |

En conjuntos de escenarios, ofrecer promedio ponderado, mínimo, máximo y
dispersión. Identificar el escenario del peor caso según el indicador elegido.
Un grafo agregado mostrará frecuencia de instalación/falla y rangos de
capacidad; se rotulará como resumen estadístico, no como una red físicamente
operada. No unir inversiones de planes distintos como si fueran un único plan.
En el resumen de horas, distinguir energía acumulada y potencia instantánea.

### Guía de uso

Integrar una guía por tarea: preparar datos, resolver un escenario, barrer
medidas, calcular combinaciones, evaluar planes fijos y exportar.
Cada receta debe mostrar requisitos, parámetros, comando exacto, carpeta de
salida y dónde consultar el resultado. El mismo constructor de argumentos
alimentará el botón de ejecución y el comando mostrado.

## 2. Secuencia de análisis propuesta

1. Validar los casos base de los 10 escenarios con idénticos supuestos al estudio.
2. Resolver 12 medidas individuales × 10 escenarios: **120 corridas**.
3. Construir indicadores y curvas por escenario y sobre la muestra común.
4. Resolver cada combinación elegida en los 10 escenarios: **10 corridas por
   combinación**. La fase 3 actual usa representativos por defecto; aquí se
   debe solicitar explícitamente la muestra completa.
5. Comparar beneficios y despliegue real de cada combinación con sus medidas
   individuales, siempre contra el mismo caso base y escenario.
6. Tomar los planes obtenidos para una combinación y evaluar cada inversión
   fija en los 10 escenarios, permitiendo adaptar la operación. Con 10 planes
   son **100 celdas**; pueden requerirse 90 cálculos nuevos si se reutilizan
   las 10 diagonales válidas. Deduplicar planes equivalentes solo con una
   comparación explícita de todas sus decisiones de inversión.
7. Calcular criterios y explicar qué plan elige cada uno y por qué.

Los pasos 3 y 7 son posprocesamiento si existen datos suficientes. No lanzar
un nuevo MILP al cambiar un filtro o elegir otro criterio.

## 3. Definiciones para interpretar correctamente los resultados

### Catálogo y activación

El código contiene **17 opciones**, no una lista inequívoca de 12. Antes de
implementar la receta, identificar las 12 medidas del estudio y guardar sus
claves; no usar `--medidas todas` para representar esas 12.
Mantener disponibles otras medidas como selección personalizada.

Distinguir «habilitada para optimizar», «inversión positiva» y «utilizada en
la operación». Usar umbrales por unidad y tolerancia numérica. Una política
operativa sin CAPEX requiere evidencia de operación, no una capacidad instalada.

### Solapamiento y curvas

Para un indicador que se desea reducir, definir el beneficio individual como
caso base menos resultado. Comparar la suma de beneficios de los miembros de
una combinación con el beneficio conjunto. La diferencia describe falta de
aditividad; no atribuye causalmente el aporte de cada medida dentro del conjunto.
Esa atribución requiere contrafactuales adicionales, por ejemplo retirar una
medida y recalcular, o un diseño factorial/Shapley con cobertura suficiente.

Revisar `_solapamiento`: suma las filas individuales del escenario que recibe;
la integración debe garantizar que solo reciba miembros de la combinación.
Mostrar por separado coincidencia de ubicaciones/activaciones y solapamiento
de beneficios. No son el mismo resultado.

Preservar el cálculo agregado costo esperado / mejora esperada, usando la
misma muestra y pesos. Si no hay mejora material, el costo unitario es no
aplicable. No presentar la suma de abatimientos independientes como reducción
alcanzable por una cartera. Una curva marginal de cartera requiere calcular
sus incrementos sucesivos y declarar el orden de incorporación.

### Matriz y reglas de decisión

Filas = planes de inversión fijos; columnas = escenarios de evaluación.
Ofrecer matrices de costo objetivo, ENS, SAIDI y arrepentimiento. Una tabla de
óptimos recalculados por escenario no sustituye esta evaluación de un plan fijo.

| Criterio | Si la matriz representa costos/pérdidas | Si representa beneficios |
|---|---|---|
| Conservador | Minimax: minimizar el máximo | Maximin: maximizar el mínimo |
| Optimista | Minimin: minimizar el mínimo | Maximax: maximizar el máximo |
| Arrepentimiento | Minimizar el peor exceso sobre la referencia del escenario | Minimizar la peor pérdida frente a la referencia del escenario |
| Valor esperado | Minimizar el promedio ponderado | Maximizar el promedio ponderado |

Explicar esta orientación junto al selector para resolver la ambigüedad de
«minmás/máxmás». `run_plan_fijo.py` ya calcula valor esperado, minimax de
objetivo y minimax regret; falta integrar la vista y completar las reglas.
Actualmente usa pesos iguales y limita los criterios a columnas completas
con diagonal disponible. Mostrar esa cobertura y ampliar la referencia si se
evalúan escenarios fuera del conjunto de planes; no omitirlos silenciosamente.
Identificar si el arrepentimiento se mide contra el mejor plan candidato o
contra una solución por escenario con información perfecta. Reportar gap y
calidad de esta referencia: una solución con tolerancia no es un óptimo exacto.

### Rangos del plan

Mostrar mínimo, mediana, máximo y frecuencia de instalación por medida y
ubicación sobre los planes optimizados para cada escenario. Para un plan fijo,
las inversiones permanecen iguales: lo que varía es su operación y desempeño.
No interpretar la frecuencia en una muestra seleccionada como probabilidad
poblacional sin justificar sus pesos y diseño muestral.

## 4. Implementación por etapas

1. **Aislar la versión nueva y los datos.** Crear módulos propios para la v8
   antes de modificarlos; inventariar dependencias, definir rutas configurables
   y salidas por estudio. Leer resultados anteriores conservando su procedencia.
2. **Unificar el estudio y la ejecución.** Configuración serializable común,
   catálogo, validaciones, recetas, cola persistente y comandos reproducibles.
3. **Unificar resultados.** Adaptadores para historial, barrido y plan fijo;
   conservar grafo y «Ver una corrida», eliminar selectores contradictorios.
4. **Integrar análisis y decisión.** Solapamiento por miembros, activación,
   rangos, MAC y matrices; distinguir datos ausentes de ceros.
5. **Cerrar la autonomía.** Guía integrada, dependencias y arranque documentados,
   paquete exportable con configuración, entradas necesarias y resultados.
   La herramienta seguirá requiriendo Python y el solver/licencia correspondiente.

Estructura objetivo orientativa: `app_v8_estudio.py` como entrada ligera;
`estudio_v8/` con configuración, ejecución, lectores y vistas; `estudios/<id>/`
con manifiesto, corridas, logs y exportaciones. Evitar duplicar fórmulas de
indicadores entre pantallas. Cargar resultados detallados y preparar imágenes
o Excel únicamente cuando se soliciten.

## 5. Verificación para aceptar el rediseño

- Abrir la herramienta sin artefactos y recibir una ruta de preparación utilizable.
- Recuperar resultados actuales sin alterarlos; comparar indicadores con sus
  lectores existentes para la misma configuración.
- Completar una corrida corta; lanzar, detener y reanudar un barrido pequeño
  sin repetir resultados válidos ni perder el avance al cerrar el navegador.
- Comprobar coincidencia de escenario, plan, hora y caso base entre tablas,
  grafo y exportación; mantener identificadores internos sin desplazar índices.
- Verificar el solapamiento con una combinación que use solo parte del catálogo.
- Evaluar una matriz pequeña: inversión idéntica por fila, operación adaptable,
  control de diagonal y manejo de fallos/inviabilidad y celdas ausentes.
- Verificar criterios con una matriz de resultados conocidos, incluyendo
  orientación costo/beneficio, empates y referencia de arrepentimiento.
- Mostrar productos disponibles y faltantes; no declarar un estudio completo
  cuando solo una fracción de escenarios produjo resultados utilizables.

## 6. Comandos actuales de referencia

Ejecutar desde `abc/mi_proyecto/app/app`. Son referencias a las interfaces
revisadas en el código, no cálculos ejecutados como parte de esta propuesta.

```powershell
python -m streamlit run app_v8_estudio.py
python pipeline.py --estado
python run_barrido_medidas.py --help
python run_plan_fijo.py --help
```

Ejemplo de previsualización de barrido con la muestra documentada en
`GUIA_BARRIDO.md` (validar su vigencia para la red/fallas elegidas):

```powershell
python run_barrido_medidas.py --escenarios 17,26,43,46,49,53,54,61,81,98 --seccionadores 0,2,9,10 --sin-fallas LIM-0.30-ALIM --lim-dg 0.30 --lim-dg-modo alimentador --fases 1,2 --listar
```

Ese ejemplo usa el catálogo predeterminado de 17 opciones. Para el estudio de
12, añadir `--medidas` con sus 12 claves explícitas. Tras verificar la
configuración y una salida propia, quitar `--listar` ejecuta el cálculo.
Para fase 3, usar `--combo`, `--no-todas` si corresponde y
`--combo-escenarios` con los escenarios deseados.

Ejemplo de inspección del plan de ejecución, usando un barrido existente:

```powershell
python run_plan_fijo.py --barrido "sec-0-2-9-10 severo" --listar
```

Por defecto el script elige la carpeta de combinación con más componentes;
la nueva interfaz siempre pedirá una combinación explícita y mostrará su nombre.
`--solo-matriz` recompone matrices con corridas guardadas. El cálculo de plan
fijo necesita recuperar la configuración desde una corrida `.pkl.gz` de la
combinación; conservar el detalle necesario desde la fase 3.

Fuentes locales revisadas: `app_v7_carga_corregida.py`, `barrido_vista.py`,
`barrido_lanzador.py`, `run_barrido_medidas.py`, `run_plan_fijo.py`,
`data_loader.py`, `MAPA_APP_V7.md` y `GUIA_BARRIDO.md`.
La guía contiene afirmaciones antiguas (por ejemplo, cálculo solo por consola)
que deben actualizarse a las capacidades actuales del lanzador.
