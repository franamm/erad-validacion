# Guía del barrido de medidas

Cómo correr el barrido medida × escenario, elegir el escenario representativo
de cada medida y visualizar los resultados.

**Regla de oro: el cálculo va por consola, la app solo visualiza.** Son cientos
de MILP de 8760 h que tardan horas. Streamlit re-ejecuta el script completo en
cada interacción de widget y pierde el `session_state` al cerrar la pestaña, así
que resolverlos desde la UI es perder el trabajo. `run_barrido_medidas.py` guarda
cada corrida apenas termina y es reanudable.

---

## 1. Qué hace, en tres fases

| fase | qué hace | corridas |
|---|---|---|
| **1** | Cada medida **por separado** sobre los N escenarios. Una corrida determinista de `optimization_6_oficial` por cada par (medida, escenario), con `esc_debug = escenario`. | medidas × escenarios |
| **2** | Promedia los indicadores sobre esos escenarios y elige, **para cada medida**, el escenario más cercano a su promedio: el *representativo*. No resuelve nada, solo lee la fase 1. | 0 |
| **3** | Corre **todas las medidas de la fase 1 encendidas a la vez** sobre cada escenario representativo distinto. | ≤ escenarios representativos |

La fase 2 es instantánea, así que se puede repetir con otro criterio sin
recalcular nada.

---

## 2. Antes de correr: qué tiene que existir

```bash
cd abc/mi_proyecto/app/app
python pipeline.py --estado
```

Tiene que aparecer `[x]` en E0 a E4, y baselines E5 **de la misma variante de
aislación** con la que vas a correr el barrido. Si falta algo, el propio mensaje
de error dice qué etapa correr.

Los baselines de una variante de aislación se generan así (una vez, ~1 min por
escenario):

```bash
python pipeline.py --etapas E5 \
    --escenarios 17,26,43,46,49,53,54,61,81,98 \
    --seccionadores 0,2,9,10
```

Esto escribe `artefactos/baselines/sec-0-2-9-10/` sin tocar los baselines
históricos. **Dos supuestos de aislación distintos no son comparables entre sí**:
si el caso base se calcula con aislación perfecta y la estrategia con aislación
realista, el "ahorro de ENS" sale negativo aunque la medida sirva. Por eso cada
variante lleva su propio juego de baselines.

### E4 con otro tope de capacidad de DG

El tope de DG por barra (`lim_dg`, como fracción de `P_D_base`) es un parámetro
de configuración, no algo que haya que editar en el código:

```bash
# E4 con DG hasta el 100% de la demanda de cada barra
python pipeline.py --etapas E4 --lim-dg 1.0 --etiqueta LIM-1.00

# E4 con el tope histórico del 30%
python pipeline.py --etapas E4 --lim-dg 0.3 --etiqueta LIM-0.30

# ver qué variantes de E4 existen y con qué tope se calculó cada una
python pipeline.py --estado
```

Sin `--etiqueta` se sobrescribe `sin_fallas.pkl`. **Usa siempre etiqueta al
cambiar el tope**: el CAPEX/OPEX de DG sin fallas que E4 entrega al objetivo
depende de cuánta capacidad se dejó instalar, así que dos topes distintos son
dos artefactos distintos y no intercambiables.

Después el barrido tiene que correr con **el mismo** tope:

```bash
python run_barrido_medidas.py ... --sin-fallas LIM-0.30-ALIM \
    --lim-dg 0.30 --lim-dg-modo alimentador
```

Si no coinciden, el script **se detiene** con un mensaje que dice qué hacer. Es
una comprobación deliberada: la incompatibilidad no produce ningún error por sí
sola, simplemente deja el resultado mal en silencio. Lo mismo se verifica para el
horizonte, porque un E4 de 24 h no alcanza a amortizar el CAPEX y deja el crédito
de DG en cero.

> **Estado actual del código, a revisar.** `optimization_base.py` (que calcula
> E4 y E5) trae como default `lim_dg = 0.3`, y `optimization_6_oficial.py` (las
> estrategias) trae `1.0`. Los defaults solo aplican cuando nadie pasa la clave;
> `pipeline.config_medida` la fija siempre, así que todo lo que pase por el
> pipeline o por el barrido queda consistente. Pero **la app arma su config por
> su cuenta y no pasa `lim_dg`**, así que ahí siguen rigiendo los defaults
> distintos de cada archivo. Conviene unificarlos.

---

## 3. La corrida configurada para esta tesis

```bash
cd abc/mi_proyecto/app/app

# 1. Ver el plan sin resolver nada (no crea archivos)
python run_barrido_medidas.py \
    --escenarios 17,26,43,46,49,53,54,61,81,98 \
    --seccionadores 0,2,9,10 \
    --sin-fallas LIM-0.30-ALIM \
    --lim-dg 0.30 --lim-dg-modo alimentador \
    --listar

# 2. Fases 1 y 2: barrido individual + escenario representativo
python run_barrido_medidas.py \
    --escenarios 17,26,43,46,49,53,54,61,81,98 \
    --seccionadores 0,2,9,10 \
    --sin-fallas LIM-0.30-ALIM \
    --lim-dg 0.30 --lim-dg-modo alimentador \
    --fases 1,2

# 3. Fase 3: todas las medidas juntas sobre los representativos
python run_barrido_medidas.py \
    --escenarios 17,26,43,46,49,53,54,61,81,98 \
    --seccionadores 0,2,9,10 \
    --sin-fallas LIM-0.30-ALIM \
    --lim-dg 0.30 --lim-dg-modo alimentador \
    --fases 3
```

Los resultados quedan en `artefactos/barrido/sec-0-2-9-10/`.

### Por qué esos tres parámetros

**`--escenarios`** — muestra estratificada por **percentil de ENS del caso
base**: se ordenan los 100 escenarios Monte Carlo por su ENS, se parten en
deciles y se toma el del centro de cada uno (percentiles 5, 15, 25 … 95).

> **Desde el 2026-09-08 esto no se hace a mano.** Lo calcula
> `seleccionar_escenarios.py`, que además deja el respaldo en
> `artefactos/escenarios_estratificados.json`:
>
> ```bash
> python seleccionar_escenarios.py --seccionadores 0,2,9,10
> python seleccionar_escenarios.py --json-solo    # sólo la lista, para pipes
> ```
>
> | flag | para qué |
> |---|---|
> | `--criterio {ens,saidi,fallas}` | ENS por defecto — ver abajo |
> | `--n-estratos 5` | menos corridas |
> | `--seccionadores` | lee los baselines de esa variante de aislación |

Muestra vigente con la matriz de Dunn — **`17,26,43,46,49,53,54,61,81,98`**:

| decil | pct | esc | ENS caso base |
|---|---|---|---|
| 1 | 5 % | 43 | 7.050 |
| 2 | 15 % | 54 | 12.758 |
| 3 | 25 % | 81 | 14.746 |
| 4 | 35 % | 17 | 17.731 |
| 5 | 45 % | 53 | 20.986 |
| 6 | 55 % | 49 | 23.687 |
| 7 | 65 % | 46 | 26.884 |
| 8 | 75 % | 26 | 29.848 |
| 9 | 85 % | 61 | 32.554 |
| 10 | 95 % | 98 | 41.911 |

Media 22.815 contra 22.772 de los 100 → **sesgo +0,2 %**. Determinista y
reproducible: no depende de una semilla.

> La muestra anterior era `0,3,19,28,35,52,66,76,83,86` con ENS de 34.632 a
> 100.295. Cambió porque cambió la matriz de fallas (curva de Dunn sobre ráfaga,
> enlaces inmunes: 727 fallas donde antes había 1.747). **Los índices ya no
> significan lo mismo**: el escenario 3 de hoy no es el 3 de la corrida
> archivada, así que no se pueden empalmar resultados — sólo comparar rankings.

**Por qué ENS y no cantidad de fallas.** La correlación entre ambas es **0,754**:
un escenario con muchas fallas en líneas poco críticas puede tener menos ENS que
uno con pocas fallas bien ubicadas. Estratificar por fallas daría
`21,25,26,50,55,64,79,94,97,99` — comparte **un solo** escenario con la muestra
por ENS. Se estratifica por lo que las medidas van a mejorar.

Se estratificó por ENS del **caso base** y no por `fn_objetivo` porque el
objetivo depende de qué medida apliques: habría una estratificación distinta por
medida, y para conocerla habría que resolver los 100 escenarios × 17 medidas, que
es exactamente lo que el barrido evita. La ENS del caso base mide la severidad
del escenario en sí y sale gratis de los baselines E5.

**`--seccionadores 0,2,9,10`** — líneas `1-2, 3-4, 3-8, 12-13`. Sin esto, el
modelo supone aislación perfecta (sale solo la línea averiada), lo que deja
degeneradas cinco medidas: la reconfiguración repone el 100% de la energía por
8.213 USD/año y switches y unidades móviles no instalan nada porque no hay nada
que aislar.

**`--sin-fallas LIM-0.30-ALIM` con `--lim-dg 0.30 --lim-dg-modo alimentador`** —
usa el artefacto de E4 calculado con tope de DG del 30 % **agregado por
alimentador**, que deja al modelo elegir dónde emplazar dentro del grupo en vez
de topar cada barra por su propia demanda.

Los tres van juntos: el CAPEX/OPEX de DG sin fallas que E4 entrega al objetivo
depende de cuánta capacidad se dejó instalar, así que el barrido tiene que
declarar **el mismo** tope y el mismo modo con que se calculó el E4. Si no
coinciden, el script se detiene. Ver las etiquetas disponibles con
`python pipeline.py --estado`.

---

## 3.bis Las 17 medidas: cómo se llaman

La clave de la izquierda es la que va en `--medidas` y la que lee
`optimization_6_oficial` desde `config`. Se pasan separadas por coma y sin
espacios: `--medidas dg_sol,ess_4h,veg_mgmt`.

### Generación y almacenamiento

| clave | nombre | qué decide el modelo |
|---|---|---|
| `dg_sol` | DG solar | Capacidad en kW por barra (variable continua), con perfil solar horario. Tope `Lim · P_D_base` por barra. |
| `dg_eol` | DG eólica | Capacidad en kW por barra, con perfil eólico diferenciado por zona. |
| `dg_fos` | DG fósil | Capacidad en kW más despacho horario, con costo variable de combustible diésel. |
| `ess_4h` | ESS 4 h | Capacidad en kWh por barra de almacenamiento fijo, con 4 h de duración a potencia nominal. |
| `ess_5h` | ESS 5 h | Ídem, 5 h de duración. |
| `ess_6h` | ESS 6 h | Ídem, 6 h de duración. |

### Red y maniobras

| clave | nombre | qué decide el modelo |
|---|---|---|
| `reconfig_topo` | Reconfiguración topológica | Opera las tie-lines **ya existentes** (L12, L13, L14) para re-alimentar tramos sanos. Sin CAPEX: solo paga el costo de maniobra. |
| `inv_tie` | Enlaces nuevos | **Construir** tie-lines: CAPEX y OPEX por km de la línea de enlace. Es una decisión de inversión, distinta de `reconfig_topo`. |
| `inv_switch_rcs` | Switches RCS | Instalar reconectadores controlados remotamente en líneas candidatas: aíslan de forma automática e inmediata. |
| `inv_switch_ms` | Switches MS | Instalar seccionadores manuales: más baratos, pero requieren cuadrilla y responden con el retardo de `--t_maniobra_ms`. |

### Endurecimiento de la red

Las tres actúan sobre la **matriz de fallas**: el modelo elige a qué líneas
candidatas aplicarlas, sopesando su CAPEX/OPEX por km contra la ENS evitada.

| clave | nombre | qué decide el modelo |
|---|---|---|
| `veg_mgmt` | Gestión de vegetación | Poda: usa `faultlinMC_veg`, adelgazada con el modelo de caída de árboles de Canham (DBH de 30 a 10 cm). |
| `refuerzo` | Refuerzo estructural | Postes clase 3 en vez de clase 5: usa `faultlinMC_ref`, con la curva de fragilidad de Darestani desplazada. |
| `soterramiento` | Soterramiento | Línea subterránea: usa `faultlinMC_ug`, con ρ = 0,95 de reducción de fallas. |

### Recursos móviles y demanda

| clave | nombre | qué decide el modelo |
|---|---|---|
| `meg` | MEG, generación móvil | Cuántas unidades y a qué barra se trasladan en cada evento. Dimensionar con `--n_meg`, `--p_meg_max`, `--t_tra_meg`. |
| `mess` | MESS, almacenamiento móvil | Ídem para baterías móviles. Dimensionar con `--n_mess`, `--h_mess`, `--p_mess_max`, `--t_tra_mess`. |
| `edrp` | EDRP | Programa de respuesta de demanda de emergencia: desplaza demanda en cuatro bloques, cada uno con su incentivo. |
| `prior_criticos` | Priorización de cargas críticas | Multiplica ×100 el VoLL de las barras críticas (6, 11 y 14), de modo que el modelo las corte último. No tiene CAPEX. |

### El modificador que no es medida

`microredes` **no** está en `--medidas`: no es una medida por sí sola, es un
modificador que habilita la operación en isla, y el modelo la exige junto a
switches RCS o MS. Se activa con la bandera `--microredes` y solo aplica a las
corridas que ya incluyen `inv_switch_rcs` o `inv_switch_ms`.

---

## 4. Qué se guarda

```
artefactos/barrido/sec-0-2-9-10/
├── individual/<medida>/esc_XXX.json    indicadores + cartera + trazabilidad
├── individual/<medida>/esc_XXX.npz     series horarias (P_LC, P_SUB, dano, fuera)
├── resumen_individual.csv              tabla medida × escenario  (fase 1)
├── promedios.csv                       valor esperado, rango y representativo por medida (fase 2)
├── representativos.json                ranking completo de distancias  (fase 2)
├── combinadas/<combo>/esc_XXX.{json,npz}
└── resumen_combinadas.csv              fase 3
```

Cada `.json` registra con qué se calculó: `medidas_on`, `variante_aislacion`,
`e4_sin_fallas`, `mip_gap` y `horas`. Ningún resultado queda sin trazabilidad.

Los `.csv` salen en `utf-8-sig`, así que Excel los abre con los acentos bien.

### Qué trae cada `.json`

**Indicadores** — `objetivo`, `ens`, `saidi`, `saifi`, `asai`, `costo_ens`.

**Capacidad instalada** — `cap_dg_sol_kW`, `cap_dg_eol_kW`, `cap_dg_fos_kW`,
`cap_ess_4h_kWh`, `cap_ess_5h_kWh`, `cap_ess_6h_kWh`, `cap_meg_MW`,
`cap_mess_MWh`.

**Equipos instalados y km intervenidos** — `n_rcs`, `n_ms`, `n_tie_nuevas`,
`n_mess`, `n_veg`, `n_ref`, `n_ug`, `km_veg`, `km_ref`, `km_ug`.

**Energía del año** — `E_dg_sol`, `E_dg_eol`, `E_dg_fos`, `E_meg`,
`E_ess_descarga`, `E_ess_carga`, `E_mess_descarga`, `E_mess_carga`, `E_edrp`,
`E_sub` (compra a la subestación), `E_servida`.

**Operación** — `maniobras_tie`, `maniobras_sw`, `horas_isla`.

**`plan`** — la cartera **por elemento**, o sea la inversión fija: capacidad por
barra (`CAP_DG_sol_kW`, `CAP_ESS_kWh_4h`, `CAP_MEG_MW`, `CAP_MESS_MWh`…) y qué
líneas recibieron cada equipo (`y_rcs`, `y_ms`, `y_tie`, `x_veg`, `x_ref`,
`x_ug`, `y_mess`). O sea: **cuáles se refuerzan, cuáles se podan, cuáles se
soterran y dónde van los switches.** Solo los valores distintos de cero.

**`ubicacion`** — dónde actuó lo que **varía en el tiempo**, que `plan` no cubre:

| clave | qué dice |
|---|---|
| `tie_horas_cerrada` | **cuál enlace se cierra** y cuántas horas |
| `tie_maniobras` | cuántas veces se maniobra cada enlace |
| `sw_horas_abierta` | qué línea candidata queda abierta y cuánto |
| `sw_maniobras` | cuántas veces se maniobra cada switch |
| `meg_horas_por_bus`, `meg_horas_por_unidad_y_bus` | **a qué barra va cada MEG** |
| `mess_horas_por_bus`, `mess_horas_por_unidad_y_bus` | por dónde se mueve cada MESS |
| `isla_horas_por_bus` | qué barras operaron como fuente de isla |

**`resumen_red`** — el mismo dict que produce `grafo.resumir_simulacion`, o sea
lo que alimenta la figura de resumen de red de la pestaña Resultados. Se calcula
en el barrido (donde todavía existen las 8760 horas) y se guarda: pesa unos pocos
KB frente a los ~100 MB que ocuparían las horas completas.

Todas estas columnas viajan también a `resumen_individual.csv`,
`resumen_combinadas.csv` y, resumidas, a `promedios.csv`.

### Cómo resume `promedios.csv` (fase 2)

Una fila por medida, y **los escenarios entran equiprobables**: el promedio es
el valor esperado sobre los N escenarios del barrido.

| sufijo | qué es | en qué columnas |
|---|---|---|
| `_prom`, `_std` | valor esperado y desviación entre escenarios | indicadores (`objetivo`, `ens`, `saidi`, `saifi`, `asai`, `costo_ens`) y descomposición económica (`cst_*`) |
| `_prom` | valor esperado | resultados físicos (energía, operación) |
| `_min`, `_max`, `_n_instala` | **rango** entre escenarios y en cuántos instala algo | dimensionamiento: `cap_*`, `n_*`, `km_*` |
| `_rep` | valor en el escenario representativo | todas |

El dimensionamiento va como rango y no como promedio a secas porque **es una
decisión, no una magnitud continua**: `cap_dg_fos_kW_prom = 393 kW` con
`_min = 0`, `_max = 1.693` y `_n_instala = 4` dice que DG fósil no es un parque
de 393 kW, sino que en 6 de 10 escenarios no se instala y en los otros 4 se
instala grande. El promedio solo lo escondería.

Los **costos unitarios** (USD/kWh evitado, USD por h/cliente, B/C) no están en
el CSV a propósito: se calculan aguas abajo como **cociente de promedios**
(`E[costo] / E[mejora]`, en `barrido_vista._resumen_eco`), no como promedio de
cocientes, porque promediar razones le da el mismo peso a un escenario que
evitó 100 kWh que a uno que evitó 10.000.

La fase 2 se puede rehacer sola, sin re-resolver nada:

```bash
python run_barrido_medidas.py --escenarios 17,26,43,46,49,53,54,61,81,98 \
    --seccionadores 0,2,9,10 --fases 2 --criterio objetivo \
    --lim-dg 0.3 --lim-dg-modo alimentador --sin-fallas LIM-0.30-ALIM
```

Los supuestos (`--lim-dg`, `--lim-dg-modo`, `--sin-fallas`, `--horas`) tienen
que coincidir con los de la fase 1 o el script se detiene antes de tocar nada.

### Qué trae cada `.npz`

Series horarias de 8760 valores, más el detalle espacial. Unos 42 KB por corrida
comprimidos (≈7 MB las 170).

| grupo | arreglos |
|---|---|
| caso base (igual que E5) | `P_LC` (8760×barras), `P_SUB` (8760×2), `dano`, `fuera` |
| generación | `P_DG_sol`, `P_DG_eol`, `P_DG_fos`, `P_MEG` |
| almacenamiento | `P_ESS_4h`, `P_ESS_5h`, `P_ESS_6h`, `P_MESS`, `SOC_4h`, `SOC_5h`, `SOC_6h`, `SOC_MESS` |
| demanda | `EDRP`, `carga_servida` |
| maniobras | `tie_cerradas`, `maniobras_tie`, `sw_en_servicio`, `maniobras_sw`, `buses_isla` |
| energía por barra | `BUSES`, `E_DG_sol_bus`, `E_DG_eol_bus`, `E_DG_fos_bus`, `E_MEG_bus`, `E_MESS_bus`, `E_EDRP_bus`, `E_LC_bus` |

**Convención de signo del almacenamiento**: `P_ESS` y `P_MESS` son
`descarga − carga`, o sea **positivo es inyección a la red** y negativo es
consumo ([optimization_6_oficial.py:1524](optimization_6_oficial.py#L1524)).

Se guardan los **totales del sistema por hora**, no el detalle por barra de cada
tecnología: eso último serían ~1 MB por arreglo y por corrida. El detalle
espacial se conserva como energía acumulada por barra (`E_*_bus`), que es lo que
se necesita para mapas y para saber dónde terminó instalándose cada recurso.

Para leer uno a mano:

```python
import numpy as np, json
z = np.load("artefactos/barrido/sec-0-2-9-10/individual/dg_sol/esc_000.npz")
print(z.files)
ens_por_hora = z["P_LC"].sum(axis=1)
reg = json.load(open("artefactos/barrido/sec-0-2-9-10/individual/dg_sol/esc_000.json",
                     encoding="utf-8"))
print(reg["cap_dg_sol_kW"], reg["plan"]["CAP_DG_sol_kW"])
```

**Las corridas anteriores a este cambio no traen los campos nuevos.** La app las
muestra igual, sin las secciones que no aplican. Para rellenarlas hay que
recalcularlas con `--forzar`.

---

## 5. Reanudar, rehacer, acotar

**Es reanudable.** Si una corrida ya tiene su `.json`, se omite. Puedes cortar con
Ctrl+C y relanzar el mismo comando: sigue donde iba. Las corridas sin solución
también se registran, para no reintentarlas en cada pasada.

> ### ⚠️ Una carpeta = un juego de supuestos
>
> La carpeta de salida solo codifica la **variante de aislación**. El tope de DG
> (`--lim-dg`), el artefacto E4 (`--sin-fallas`) y el horizonte (`--horas`)
> también cambian los números y **no** entran en ese nombre.
>
> Como el barrido es reanudable, correr dos veces con distinto `--lim-dg` en la
> misma carpeta no solo sobrescribe: **omite** las corridas que ya existen, y te
> deja una tabla que promedia dos supuestos distintos sin avisar.
>
> El script se detiene si lo detecta y te ofrece las dos salidas: una carpeta
> aparte (`--salida artefactos/barrido/sec-0-2-9-10__lim0.3`), que conserva lo ya
> calculado, o `--forzar`, que rehace todo con los supuestos nuevos. En la app,
> el campo "Carpeta de resultados" del formulario hace lo mismo.
>
> Las corridas anteriores a que se registrara `lim_dg` aparecen como
> desconocidas: se avisan pero no bloquean, porque desconocido no es lo mismo
> que incompatible.

```bash
# rehacer todo desde cero
... --forzar

# solo algunas medidas
... --medidas dg_sol,ess_4h,veg_mgmt --fases 1

# recalcular la fase 2 con otro criterio (instantáneo, no resuelve nada)
... --fases 2 --criterio multi

# combinación extra en la fase 3 (repetible)
... --fases 3 --combo dg_sol,ess_4h --combo veg_mgmt,refuerzo

# SOLO tu combinación, sin la de "todas las medidas juntas",
# y sobre un escenario que tú eliges
... --fases 3 --no-todas --combo dg_sol,ess_4h,meg,mess --combo-escenarios 66 \
    --n_meg 3 --p_meg_max 300 --t_tra_meg 2 \
    --n_mess 3 --h_mess 3 --p_mess_max 200 --t_tra_mess 2
```

**Tiempos medidos** (8760 h, MIPGap 2%): `dg_sol` 74 s, `reconfig_topo` 75 s,
`mess` 70 s, `inv_switch_rcs` 365 s. Las medidas con binarias indexadas por hora
(switches, MESS, reconfiguración) son las lentas. Cada corrida está acotada por
`--time_limit` (default 1800 s).

Si vas justo de tiempo: `--mip_gap 0.05` acelera bastante, a costa de más ruido
en el objetivo.

---

## 6. El criterio del escenario representativo

`--criterio` decide qué significa "más cerca del promedio":

| valor | qué mide |
|---|---|
| `objetivo` (default) | `\|fn_objetivo_esc − promedio\|`. Es lo que el modelo minimiza: CAPEX+OPEX de la medida + ENS valorizada (`VoLL·P_LC·N_eventos`) − ahorro de subestación. |
| `ens` | `\|ENS_esc − promedio\|`, solo energía no suministrada. |
| `multi` | Distancia euclidiana sobre SAIDI, SAIFI y ENS estandarizados (z-score entre escenarios). |

Con `objetivo`, las constantes `−CAPEX/OPEX_sin_fallas` de DG solar y eólica **no
contaminan** la elección: son idénticas en todos los escenarios de una misma
medida, así que no alteran cuál queda más cerca del promedio.

**Ojo con el MIPGap.** El objetivo trae hasta `--mip_gap` de holgura de
optimalidad. Si la dispersión entre escenarios de una medida cae por debajo de
eso, el representativo se elige por ruido del solver, no por la física. El script
lo detecta y avisa al final de la fase 2:

```
[!] Dispersion entre escenarios menor al MIPGap (2.00%) en: [...]
```

Si aparece, baja `--mip_gap` o usa `--criterio ens` / `multi` para esas medidas.

---

## 7. Ver los resultados en la app

> **App v8** (`streamlit run app_v8_estudio.py`): lanza estos mismos comandos como proceso aparte
> desde ⚙️ Optimización y análisis (sin perder el trabajo al cerrar el navegador) y muestra los
> resultados en 📊 Resultados. Ver `README_APP_V8.md`. Lo que sigue describe la pestaña Barrido de la v7.

```bash
streamlit run app_v7_carga_corregida.py
```

Pestaña **🧪 Barrido**. Solo lee los artefactos, no calcula nada. Cuatro
secciones:

- **Resultados por medida** — matriz medida × escenario con gradiente de color,
  boxplot de dispersión y descarga a CSV. El selector **Ver** cambia entre
  *Indicadores* (SAIDI, SAIFI, ENS, ASAI, objetivo) y *Resultados físicos*
  (capacidad instalada, equipos, energía por tecnología, maniobras), así que se
  puede ver la matriz de, por ejemplo, "capacidad DG solar instalada" por medida
  y escenario.
- **Escenario representativo** — tabla de representativos con su desviación
  respecto al promedio, y por medida un gráfico de barras por escenario con la
  línea del promedio y el representativo destacado en rojo.
- **Medidas combinadas** — individual vs combinada **en el mismo escenario**, que
  es la única comparación válida. Si la combinada no baja tanto como la suma de
  las individuales, hay solapamiento entre medidas.
- **Series horarias** — para una corrida concreta:
  - ENS de la estrategia contra la del caso base, con la ENS evitada como KPI.
  - **Potencia por tecnología**: DG solar/eólica/fósil, MEG, ESS por duración,
    MESS, EDRP, carga servida y estados de carga. Multiselect para superponer;
    solo aparecen las series que no están en cero.
  - **Operación de la red**: líneas fuera de servicio, líneas averiadas, enlaces
    cerrados, maniobras de enlace y de switch, barras con inyección local.
  - **Energía acumulada por barra**: dónde terminó inyectando cada recurso y
    dónde quedó la energía no suministrada.
  - **Resumen de red**: la **misma figura** que la pestaña Resultados, dibujada
    con `grafo.build_fig_grafo_resumen` a partir del `resumen_red` guardado —
    no es una reimplementación, es la misma función. Debajo, las tablas por
    línea y por barra con descarga a CSV.
  - **Dónde actuó la medida**: qué enlace se cerró y cuántas horas, qué switch
    se maniobró, a qué barra fue cada MEG y por dónde pasó cada MESS.
  - **Resultados físicos** de la corrida como KPIs, y la cartera por elemento en
    un desplegable.

  La granularidad (horaria / diaria / mensual) agrega sumando la energía y
  promediando los conteos de estado, porque sumar 24 valores de "líneas fuera de
  servicio" no tendría interpretación.

El selector de arriba elige la variante de aislación. Si corriste el barrido con
la app abierta, el botón **🔄 Recargar** vuelve a leer el disco.

---

## 8. Errores frecuentes

**`Faltan baselines (variante 'sec-...') para los escenarios [...]`**
No generaste E5 para esa variante. El mensaje trae el comando exacto a correr.

**`No existe el artefacto de E4 'sin_fallas_X.pkl'`**
La etiqueta de `--sin-fallas` no existe. Lista las disponibles con
`python pipeline.py --estado`.

**Una medida reporta ENS idéntica al caso base y `plan` vacío**
El solver no instaló nada: esa medida no resultó rentable, o el supuesto de red
no le deja margen. No es un error del script. Con aislación perfecta le pasa a
switches y unidades móviles de forma sistemática.

**Si filtras la salida con `grep`, cuidado con el código de salida.** En bash el
exit code de un pipe es el del último comando, así que un fallo de Python queda
enmascarado por un `grep` exitoso. Usa `${PIPESTATUS[0]}`:

```bash
python run_barrido_medidas.py ... 2>&1 | grep "^\[barrido\]"; echo "EXIT=${PIPESTATUS[0]}"
```

---

## 9. Limitaciones a declarar en la tesis

**La estratificación es por ENS, no por SAIDI.** La correlación entre ambos sobre
los 100 escenarios es 0,907 — alta, pero no 1. La muestra queda perfectamente
estratificada en ENS (percentiles 5 a 95) pero no en SAIDI, donde los percentiles
se apelotonan entre 50 y 70 y no hay nada entre 15 y 43 ni sobre 86. Las medias
sí son insesgadas en ambos indicadores, así que **el promedio de la fase 2 no se
ve afectado**; lo que no cubre bien es la *distribución* del SAIDI. Si en algún
momento reportas rangos o colas de SAIDI, hay que decirlo.

**El muestreo por centro de decil excluye las colas.** La muestra va de 34.632 a
100.295 de ENS, mientras los 100 escenarios van de 19.164 (esc 68) a 132.464
(esc 82). El peor caso no está en la muestra.

**El escenario 3, el del estudio congelado, está en el percentil 95.** No es un
problema del muestreo: es un hallazgo. Los resultados históricos de esta tesis
salen de un escenario casi de peor caso, no de uno típico.

**El conjunto de seccionadores es un supuesto, no un dato.** La red es el
benchmark CIGRE MV y su tabla `net.switch` define solo tres switches de línea, los
tres en las tie-lines (S1, S2, S3), más los dos interruptores de cabecera: **no
especifica ningún seccionador a lo largo de los alimentadores**. El supuesto que
el modelo usaba antes (aislación perfecta) equivale a suponer un seccionador en
cada línea, que es el más alejado del benchmark de todos. El conjunto
`{1-2, 3-4, 3-8, 12-13}` sigue el criterio estándar de seccionar en la cabecera
de cada alimentador y en las ramas que salen del nodo de derivación, pero hay que
declararlo como supuesto y mostrar sensibilidad. Correr otra variante es solo
generar sus baselines E5 y repetir el barrido con otro `--seccionadores`: cada
variante vive en su propia carpeta y no pisa a la otra.

**`Cap 3_2.tex` necesita un ajuste.** El texto dice hoy que la ausencia de switch
no implica "la apertura de una línea sana". Con la aislación activada, las líneas
sanas **sí** quedan fuera de servicio: no porque el modelo las abra
voluntariamente, sino porque la falla las arrastra. Son dos mecanismos distintos
y siguen siendo compatibles, pero la frase hay que matizarla.

---

## 10. Referencia rápida de flags

| flag | default | para qué |
|---|---|---|
| `--escenarios` | `0-9` | índices MC: `0-9`, `0,3,7` |
| `--medidas` | `todas` | claves separadas por coma (ver abajo) |
| `--horas` | `8760` | horizonte de cada corrida |
| `--fases` | `1,2,3` | qué fases ejecutar |
| `--criterio` | `objetivo` | `objetivo`, `ens` o `multi` |
| `--seccionadores` | *(vacío)* | activa la aislación realista |
| `--sin-fallas` | *(histórico)* | etiqueta del artefacto E4 |
| `--lim-dg` | `1.0` | tope de DG por barra, en fracción de `P_D_base`; tiene que coincidir con el del E4 |
| `--combo` | — | combinación extra en fase 3, repetible |
| `--todas` / `--no-todas` | on | fase 3: correr además la combinación con todas las medidas de `--medidas` juntas |
| `--combo-escenarios` | *(los representativos)* | fase 3: sobre qué escenarios correr las combinaciones |
| `--completo` / `--no-completo` | on | guardar además el `.pkl.gz` que abre la corrida en Resultados |
| `--microredes` | off | operación en isla; solo aplica con switch RCS/MS |
| `--forzar` | off | rehacer corridas ya guardadas |
| `--listar` | off | mostrar el plan y salir |
| `--salida` | `artefactos/barrido/<variante>` | carpeta de resultados |
| `--mip_gap` | `0.02` | holgura de optimalidad |
| `--time_limit` | `1800` | tope de segundos por corrida |
| `--threads`, `--mem_gb` | auto | recursos de Gurobi |

**Las 17 claves de `--medidas`** (detalle en la sección 3.bis):

```
dg_sol          dg_eol          dg_fos
ess_4h          ess_5h          ess_6h
reconfig_topo   inv_tie         inv_switch_rcs  inv_switch_ms
veg_mgmt        refuerzo        soterramiento
meg             mess            edrp            prior_criticos
```

Para copiar y pegar:

```
dg_sol,dg_eol,dg_fos,ess_4h,ess_5h,ess_6h,reconfig_topo,inv_tie,inv_switch_rcs,inv_switch_ms,veg_mgmt,refuerzo,soterramiento,meg,mess,edrp,prior_criticos
```

`microredes` no va acá: es un modificador, se activa con `--microredes`.

Dimensionamiento de unidades móviles y switches (defaults iguales a los de la UI):
`--n_meg 3`, `--p_meg_max 50`, `--t_tra_meg 2`, `--n_mess 3`, `--h_mess 3`,
`--p_mess_max 200`, `--t_tra_mess 2`, `--t_maniobra_ms 1`.

**No uses `--horas` menor a 8760 con `--criterio objetivo`.** El crédito de
subestación del objetivo compara contra un baseline E5 de 8760 h, así que con un
horizonte recortado el objetivo queda distorsionado en valor absoluto. Para
probar el flujo sirve; para elegir el representativo, no. El script lo avisa al
arrancar.
