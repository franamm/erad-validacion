# Mapa de `app_v7_carga_corregida.py`

Referencia rápida del entry point de la app (3517 líneas). Los números de línea son
del estado al 2026-09-08; si el archivo cambió mucho, buscar por el título de la
sección (`st.header(...)`) en vez de saltar a la línea.

## Esqueleto

| Líneas | Bloque |
|---|---|
| 1–98 | Imports + `_resolver_con_veg()` (despacha al MILP; la rama estocástica está rota, ver abajo) |
| 99–137 | `set_page_config` + **sidebar**: granularidad, `vista_global` (Promedio / Peor caso / Escenario), zonas |
| 138–176 | `st.tabs([Datos de entrada, Optimización, Resultados, Barrido])` + `esc_sel` global |
| 178–201 | Supuesto de **aislación** leído del `session_state` → elige la variante de baseline |
| 203–259 | Modo **estocástico** leído del `session_state` + carga de `results_sin` (baselines) |
| 261–290 | `_pos_esc()` (posición real de un escenario en los resultados) + `SAIDI_sin` |
| 292–425 | **Tab 1 · sub-pestaña "Modelo cargado"** |
| 428–1443 | **Tab 2 · Optimización** |
| 1445–3517 | **Tab 3 · Resultados** |

`tab1/sub_origen` → `entrada_red_vista.render()`. `tab4` → `barrido_vista.render()`.
Ambos se declaran arriba (líneas 152–165) y viven en sus propios módulos.

El truco estructural: los contenedores (`with tab3:`) se declaran arriba y se
**reabren** mil líneas más abajo. Por eso el orden visual de la app no coincide con
el orden del archivo.

## Tab 1 — Modelo cargado (300–425)

Grafo de la red por hora (slider 1-based) → tablas de buses y líneas → métricas Monte
Carlo → parámetros económicos (`r`, `n`, `frp`, VoLL) → tabla de costos de medidas →
perfil de demanda de las primeras 24 h.

## Tab 2 — Optimización (432–1443)

- **435–519** Toggles de las 17 medidas en 4 columnas + botones "Marcar/Desmarcar todas".
- **521–635** Panel `🔒 Aislación de la falla`: **no es una medida**, es un supuesto de red.
  Previsualiza línea por línea qué arrastra cada falla usando la misma
  `zona_aislada()` del optimizador.
- **637–761** Parámetros condicionales (MEG / MESS / MS-RCS), solo si el toggle está activo.
- **763–825** Selector de escenario MC + toggle de modelo estocástico de dos etapas.
- **827–876** Se arma `config` (el dict que recibe el MILP).
- **878–1020** Botones `▶️ Calcular`, `🗑️ Limpiar`, `▶️ Calcular cada medida` (barrido corto
  en sesión, uno por medida activa).
- **1026–1091** Corrida única → guarda en `st.session_state["historial"][label]`
  (`{results, saidi, config, tiempo}`).
- **1093–1106** Tabla del historial.
- **1108–1443** Bloque de **sidebar** "Guardar / Cargar resultados": serializa el historial
  a `.pkl.gz` con firma del modelo (`pipeline.firma_modelo()`) y lo vuelve a cargar
  validando que la firma calce. Ojo: es código de sidebar escrito *dentro* de `with tab2`.

## Tab 3 — Resultados (1449–3517)

Un solo scroll larguísimo, sin sub-pestañas. Secciones en orden:

| Línea | Sección |
|---|---|
| 1491 | **A — Indicadores de Confiabilidad** (radio Red/Zona/Barra + slider de escenario propio) |
| 1569 | SAIDI/SAIFI mensual acumulado (solo si `N_HORAS >= 720`) |
| 1610 | **Grafo de la Red** (por hora) — sin letra en el título |
| 1771 | **Resumen de la simulación** — grafo agregado del horizonte + descargas PNG/HTML/Excel. Radio `caso_grafo_resumen` con tres casos: *Caso base* (baseline del escenario; dibuja los seccionadores preexistentes), *Estrategia activa* (la del historial) y *Todas las fallas* (mapa de exposición: `grafo.resumir_fallas_mc` funde los N escenarios del Monte Carlo, 0.1 s, no necesita baselines). El agregado NO es un estado realizable de la red — ver el comentario en `grafo.agregar_resumenes` |
| 1789 | **C — Capacidades Instaladas** (DG, MEG, MESS, ESS, tie-lines, líneas tratadas, switches) |
| 2110 | **D — Costos y Beneficio Neto** |
| 2148 | **E — Comparación de Estrategias** |
| 2182 | └ Panel 0: tarjeta por estrategia (`expander(expanded=True)`) con semáforo de rentabilidad |
| 2408 | └ Panel 1: tablas comparativas (confiabilidad / económicos / costo por unidad de mejora) |
| 2591 | └ Curvas MAC de ENS, SAIDI y SAIFI (`construir_curva_abatimiento_2`) en 3 tabs |
| 2785 | └ Panel 1.5: exportar tabla comparativa CSV/Excel |
| 2826 | └ Panel 2: SAIDI por escenario MC |
| 2868 | └ Tabs SAIFI por estrategia / ENS por zona |
| 3014 | └ ENS por período (usa la granularidad de la sidebar) |
| 3058 | └ EDRP por bloque y por nodo |
| 3118 | **F — Análisis de Sinergias** (diseño factorial 2×2, IC bootstrap, Shapley) |
| 3345 | **Exportar Reporte** (Excel con hojas KPIs / mensual / Medidas / Costos) |

## Estado compartido (`st.session_state`)

| Clave | Quién la escribe | Quién la lee |
|---|---|---|
| `historial` | tab2 (calcular / cargar archivo) | tab3 entero |
| `esc_sel_global`, `esc_tab2` | slider de tab2 | cabecera del script (línea 174) |
| `esc_t3_kpi` | slider de tab3 | **solo** la sección A de tab3 |
| `t_aislacion`, `sec_existentes_ui` | panel de aislación | cabecera (elige la variante de baseline) |
| `estoc_activo`, `estoc_escenarios` | toggle estocástico | cabecera (elige el set de baselines) |
| `_resumen_sim`, `_resumen_sim_key` | caché del grafo resumen | sección de resumen |
| `resultados_gz_*` | botones "Preparar" | botón de descarga |

Los widgets de tab2 se leen en la **cabecera** del script porque Streamlit re-ejecuta
todo en cada interacción: para cuando el usuario aprieta "Calcular", esas variables ya
reflejan la elección de la corrida anterior.

## Resuelto el 2026-09-08

- **Un solo selector de escenario.** El slider vive ahora en la sidebar (bloque
  "Escenario Monte Carlo", ~línea 133) y rige para toda la app. Antes había uno en
  tab2 (`esc_tab2`) y otro en los KPIs de tab3 (`esc_t3_kpi`) sin sincronizar: la
  misma pantalla podía mostrar dos escenarios distintos. Los dos sliders viejos se
  reemplazaron por avisos de solo lectura.
- **`_pos_esc()` aplicado en todas partes.** Ya no quedan `min(esc_sel - 1, len(valid) - 1)`
  sobre listas de resultados. Los dos `min()` que sobreviven (secciones de KPIs y de
  Excel) indexan el array crudo `demand_factor`, donde el clamp sí corresponde.
- **Cada estrategia se compara contra SU propio caso base.** Era el bug que hacía
  aparecer aumentos de ENS donde había reducciones. `_norm()` re-indexa los
  resultados de la estrategia a `{0: […]}` y borra el índice Monte Carlo, mientras
  `baseline_esc()` devuelve `{esc_seleccionado: […]}`; ambos quedaban como listas de
  un elemento y `_results_to_list` emparejaba posición 0 con posición 0 aunque fueran
  escenarios distintos. Como la ENS del caso base va de 4.469 kWh (esc. 34) a 47.045
  (esc. 21) —factor 10—, una medida que reducía 30% su ENS podía mostrarse como
  −637%. Ahora `_base_de(cfg)` reconstruye el baseline del escenario y del supuesto
  de aislación con que se resolvió cada estrategia (lee `config["esc_debug"]` /
  `["escenarios_estoc"]` y `["aislacion_fallas_fija"]`). La tabla comparativa y las
  tarjetas ganaron columna «Escenario», y hay avisos cuando el historial mezcla
  escenarios (Δ correctos pero no comparables entre sí) o supuestos de aislación
  (directamente incomparables).
- **Costos unitarios solo con mejora real.** `USD/kWh`, `USD/(h/año)` y
  `USD/(int./año)` quedan en blanco si Δ ≤ 0: con Δ negativo el cociente salía
  negativo y se leía como un costo negativo. De paso se protegió la división por cero
  que reventaba la tarjeta cuando una medida no mejoraba nada.
- **Sección E2 — valor esperado (2026-09-09).** Panel nuevo en tab3, entre E y F. Lee
  el barrido (medida × N escenarios) reusando los cargadores de `barrido_vista`
  (`listar_barridos`, `cargar_barrido`, `cargar_base`, `enriquecer`, `_unitario`,
  `UMBRAL_MEJORA`) en vez de duplicarlos, para que las dos vistas no se
  desincronicen. Promedia los Δ que `enriquecer` calcula intra-escenario, excluye las
  corridas `sospechosa` (peores que el caso base más allá del MIPGap, o cortadas por
  TimeLimit con gap > 5×MIPGap) e informa cuántas. Los costos unitarios son
  **E[costo]/E[Δ]**, no el promedio de los cocientes, y respetan `UMBRAL_MEJORA`: sin
  ese umbral un Δ ENS de polvo numérico daba cifras de 1,2e16 USD/kWh.
  Los gráficos de esa sección viven en `graficos_valor_esperado.py` (ENS por zona, ENS
  horaria y curvas de abatimiento, todo promediado sobre los mismos escenarios que el
  caso base). Desde el 2026-09-09 aceptan **medidas individuales y combinaciones**: un
  caso es `(_tipo, _caso)` y el tipo decide el subdirectorio del barrido
  (`individual/` o `combinadas/`), antes estaba fijo en `individual/`.
- **Pantalla vs. exportación.** Helpers `_fig_pantalla()` y `_descargas_figura()`
  (~línea 99). Las figuras se siguen construyendo en tamaño de tesis —eso es lo que
  se descarga— y a `st.plotly_chart` va una copia reescalada. Aplicado al grafo de
  red de tab3, ENS por zona, ENS por período y las 3 curvas MAC. El grafo por zona y
  el de período ganaron botones de descarga PNG/HTML, detrás de una casilla para no
  renderizar en cada rerun.

## Deuda conocida (candidatos a arreglar)

1. **Todo el trabajo pesado corre siempre.** Las 3 curvas MAC (2591–2675) se construyen
   aunque no se abra su tab, y `_generar_excel()` (3508) se ejecuta en cada rerun antes
   del botón de descarga. El patrón bueno ya existe en el archivo: el checkbox
   "Preparar la imagen…" de la línea 1725.
2. **Tab 3 sin sub-pestañas.** ~2000 líneas de scroll continuo; A/C/D/E/F son candidatas
   naturales a `st.tabs`. Además la numeración está rota: hay A, luego el grafo sin
   letra, y después C (la B quedó en tab1).
3. **Tarjetas siempre abiertas.** Panel 0 usa `expander(..., expanded=True)` por cada
   estrategia del historial.
4. **La sección F (sinergias) sigue usando `results_sin`.** El análisis factorial 2×2
   compara A, B y AB contra el caso base del escenario SELECCIONADO, no contra el de
   cada corrida. `claves_escenarios_comunes()` amortigua el problema (exige los mismos
   escenarios en las cuatro corridas), pero conviene hacerle el mismo tratamiento que
   a los paneles E.
5. **APIs deprecadas y mezcladas.** `df.style.applymap` (pandas ≥2.1 pide `.map`) y
   convivencia de `use_container_width=True` con `width="stretch"` en la sección EDRP.
6. **Código muerto.** `NODEFILL` se redefine dos veces seguidas y ya venía
   importado de `helpers_operativos`; hay bloques comentados grandes en las secciones
   de ESS y de KPIs económicos.
7. **Toggle estocástico roto.** `_resolver_con_veg` llama a
   `resolver_optimizacion_estocastica` pero el import está comentado (cerca de la
   línea 66) → `NameError`. Para estocástico se usa `run_estocastico.py` desde consola.
