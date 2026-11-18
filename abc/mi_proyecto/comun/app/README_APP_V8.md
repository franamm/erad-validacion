# Herramienta de estudio de resiliencia v8

Entrada: `python -m streamlit run app_v8_estudio.py` (desde esta carpeta). La v7
(`app_v7_carga_corregida.py`) sigue funcionando igual.

## Estructura

`app_v8_estudio.py` solo llama a `estudio_v8/ui.py`. Cada pestaña vive en su módulo y
**reutiliza** las vistas existentes, así que los números y el formato de figuras son
los mismos de la v7, del Barrido y de `graficos_validacion.ipynb`.

| Módulo | Qué hace |
|---|---|
| `ui.py` | Navegación, Datos de entrada, guía (checkbox al pie) |
| `entradas_clasicas.py` | Vista del modelo cargado de la v7 |
| `optimizacion.py` | Estudio, análisis elegidos, medidas, supuestos, escenarios, pasos pendientes, ejecución y seguimiento |
| `resultados.py` | Red y plan instalado, indicadores, ENS, MAC, solapamiento, reglas de decisión, Ver una corrida |
| `analisis.py` | Medidas activas, solapamiento, rangos del plan, reglas de decisión, muestra por percentil (puro, con tests) |
| `estado.py`, `progreso.py` | Inventario de corridas en disco; avance, cronómetro y gap de Gurobi desde el registro |
| `jobs.py`, `worker.py` | Procesos aparte con bloqueo por carpeta; un trabajo puede encadenar varios comandos |

Funciones compartidas movidas para no duplicar: `formato_figuras.py` (figuras pantalla/exportación),
`barrido_vista.seccion_ver_corrida` y `barrido_vista.CONDICIONALES`, `graficos_valor_esperado.render_mac`.

## Casos de estudio

Un **caso** es una carpeta de artefactos con su red, sus perfiles, sus fallas y sus
resultados (`casos.py`). `artefactos/` es el caso de validación de la tesis; cualquier
otra red se prepara en su propia carpeta `artefactos_*` sin tocarlo.

La pestaña *Datos de entrada → Origen de los datos* abre con el selector de caso y,
según lo que encuentre, muestra lo que el caso ya tiene o guía para armarlo. Desde ahí
se lanzan las etapas en segundo plano (`pipeline.py --artefactos <caso>`), la matriz de
fallas real y —solo para probar que el modelo corre— `fallas_sinteticas.py`.

La carpeta **activa del proceso** se fija al arrancar con `ERAD_ARTEFACTOS`:

```powershell
$env:ERAD_ARTEFACTOS = "$PWD\artefactos_mi_red"; python iniciar_v8.py
```

Preparar un caso distinto del activo funciona completo, pero las pestañas de
optimización, resultados y barrido siguen mostrando el activo hasta reiniciar, porque
`data_loader` lee sus artefactos al importarse y once módulos copian esos valores en su
propio import. La pestaña lo avisa y da el comando.

## Optimización y análisis

1. **Estudio**: una tanda existente (hereda sus supuestos de los `.json`) o uno nuevo.
2. **Análisis**: red, indicadores y ENS siempre; opcionales Curvas MAC, Solapamiento y Reglas de
   decisión. Cada casilla agrega solo las corridas que necesita.
3. **Medidas**: toggles de la v7; botón con las 12 de la tesis.
4. **Supuestos y solver**: E4, seccionadores, tope DG, horizonte, MIPGap, hilos, memoria, detalle horario.
5. **Escenarios**: muestra del estudio, **por percentil** de severidad del caso base
   (`seleccionar_escenarios.estratificar`, exige los casos base de toda la población y ofrece
   calcularlos) o personalizada. Índices desde 0.
6. **Revisar y ejecutar**: pasos con corridas totales, guardadas y pendientes, comandos exactos y un
   botón. El seguimiento se refresca cada 5 s: corridas guardadas, tiempo resolviendo la actual,
   incumbente/cota/gap de Gurobi y resultados preliminares a medida que termina cada escenario.

Planes fijos: la diagonal **no** se re-resuelve por defecto (90 MILP en vez de 100). La matriz usa
`cst_objetivo_plan`, que no depende de la constante del E4; antes se ofrece `--recostear-dg` (sin
MILP) cuando el E4 es posterior a la combinación.

## Resultados

Si falta algo para una vista, la vista lo dice y ofrece calcularlo. Cambiar filtros o criterios
nunca lanza una optimización. El grafo por hora necesita el `.pkl.gz`; si no existe, se puede
calcular un solo escenario en `detalle_horario/` sin tocar el estudio.

**Pendiente de decidir**: la referencia del arrepentimiento (diagonal = información perfecta, o
mejor plan candidato). La vista ofrece ambas, con la diagonal por defecto.

## Validación

```powershell
python -m unittest estudio_v8.test_core estudio_v8.test_jobs estudio_v8.test_package estudio_v8.test_analisis estudio_v8.test_progreso -v
python -m estudio_v8.check_ui
python check_entrada_red.py
```

`test_analisis` contrasta medidas activas, rangos de la librería completa y abatimiento conjunto con
las tablas de `Tesis_Francisca_Muñoz_M/Figuras/Resultados/`. `check_ui` recorre todas las pestañas
sobre los artefactos locales sin lanzar optimizaciones.

`check_entrada_red.py` hace lo mismo con las ramas de la pestaña de datos de entrada, que dependen de
lo que haya en disco y no se ven abriéndola una vez: caso de validación, caso no activo, creación de
un caso y el asistente con una red cargada. Tampoco pulsa botones de ejecución.
