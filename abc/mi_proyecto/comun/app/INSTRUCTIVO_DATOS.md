# Instructivo de datos: de cero a los artefactos del modelo

Cómo se obtienen, desde el principio, todos los insumos del estudio: los datos
climáticos (ARClim + Open-Meteo), los perfiles de generación, la matriz de
fallas, la red eléctrica y los artefactos que consume la app.

Compañero de `GUIA_BARRIDO.md`, que empieza donde este termina.

**Regla que atraviesa todo: el cálculo pesado va por consola, la app prepara y
visualiza.** La pestaña 📥 *Datos de entrada → 🗺️ Origen de los datos* arma y
valida los archivos de entrada y muestra el comando que sigue; no descarga
APIs, no corre el Monte Carlo y no resuelve MILP.

---

## 1. El mapa completo

Dos cadenas que nacen separadas y se cruzan en la matriz de fallas.

```
CADENA 1 — CLIMA                          CADENA 2 — RED
(Proyecto oficial/Scripts/)               (la pestaña, o Prueba 12 oficial.ipynb)

SystemzoneV1.xlsx                         topología (CIGRE MV, .json, .dss, CSV)
  lat/lon por zona                          + zona climática de cada bus
        │                                   + demanda por bus
        ├── Open-Meteo  → Hourlyhistoricdata.nc              │
        ├── ARClim      → Dailyfuturedata.nc                 │
        │   (ssp585)      Dailyfuturedata_relativehumidity.nc│
        ▼                                                     ▼
  BaseFhorPELP4v20z2anos.nc  (base horaria por zona)    red_nueva.pkl / app.pkl
        ▼                                                     │
  BaseHoraEWEPELP20z.nc  (viento, humedad, FFDI, fuego)      │
        │         │                                           │
        │         └──→ construir_perfiles.py → perfiles_horarios.npz
        │                                                     │
        └────────────────┬────────────────────────────────────┘
                         ▼
              construir_faultlin.py
              viento de zona → curvas de fragilidad → Monte Carlo
                         ▼
        FaultlinMC.npy  +  _veg / _ref / _ug
                         ▼
                pipeline.py  E0 … E6
                         ▼
                   artefactos/  →  app + barrido
```

El cruce es **por zona**: cada línea toma el viento del punto geográfico de la
zona de su **bus de origen** (`from_bus`). Si la red define 3 zonas y el
`SystemzoneV1.xlsx` tiene 4 puntos —o al revés— el mapeo se corre en silencio.
Por eso la pestaña exporta las dos cosas juntas.

---

## 2. Cadena 1 — datos climáticos

Todo vive en `abc/mi_proyecto/app/Proyecto oficial/`, la copia consolidada (los
mismos scripts están repartidos en `Trend database with hourly resolution/` y
`Extreme Weather Events database/`, las carpetas de trabajo originales).

```
Proyecto oficial/
├── Scripts/     los 4 pasos + el orquestador
├── Inputs/      SystemzoneV1.xlsx  ← lo único que se prepara a mano
└── Outputs/     todo lo que se genera
```

**El rol de cada fuente.** Open-Meteo aporta el **pasado horario** (reanálisis
2017–2021), que da la forma intradiaria; ARClim aporta el **futuro diario**
proyectado bajo SSP5-8.5, que da el nivel. El paso 3 los combina porque ARClim
entrega solo datos diarios y la curva de fragilidad necesita resolución horaria.
Que las dos fuentes difieran en una zona es esperable —una es reanálisis del
pasado, la otra proyección climática— y es justamente lo que hace que unas zonas
concentren más eventos extremos que otras.

### 2.1 Lo único que se prepara a mano: `SystemzoneV1.xlsx`

| columna | obligatoria | qué es |
|---|---|---|
| `Zona` | recomendada | identificador de la zona (1, 2, 3…). Si falta, se usa el número de fila. |
| `Lat` | **sí** | latitud decimal, negativa en Chile |
| `Lon` | **sí** | longitud decimal, negativa en Chile |
| `Comuna` | no | documental |

**El orden de las filas importa.** Fija el `point_id` con el que queda indexado
el eje `lat` del NetCDF de salida, y por lo tanto qué columna de viento le toca
a cada zona en `construir_faultlin.py`. La pestaña lo exporta ordenado por
`Zona` para que no dependa del orden de edición.

Se genera desde la app (📥 *Datos de entrada* → 🗺️ *Origen de los datos* →
paso 4 → **Descargar SystemzoneV1.xlsx**) o se edita a mano. El archivo vigente
lleva anotaciones de versiones anteriores del caso de estudio; qué es cada cosa
está en §7.1.

### 2.2 Paso 1 — histórico horario (Open-Meteo)

```bash
cd "abc/mi_proyecto/app/Proyecto oficial/Scripts"
python download_openmeteo_historical_ready_v2.py
```

Deja `Inputs/Hourlyhistoricdata.nc` con `t2m` (°C), `wind` (m/s), `tp` (mm/h) y
`hur` (%), horarios por punto.

| flag | default | nota |
|---|---|---|
| `--start-date` / `--end-date` | 2017-01-01 / 2021-12-31 | 5 años; es el patrón horario que usa el paso 3 |
| `--timezone` | `America/Santiago` | |
| `--model` | `best_match` | **no cambiar a `era5_land`**: no entrega todas las variables de viento |
| `--sleep` | 1.0 s | subirlo si la API responde 429 |

Requiere internet. Unos minutos por punto.

### 2.3 Paso 2 — futuro diario (ARClim)

```bash
python download_arclim_future_to_netcdf_ready_v4.py --scenario ssp585 \
       --start-year 2025 --end-year 2040
```

Genera `Inputs/Dailyfuturedata.nc` (`tas`, `tasmax`, `tasmin`, `sfcWind`, `pr`)
y `Inputs/Dailyfuturedata_relativehumidity.nc` (`hur`). Baja `tasmax`, `tasmin`,
`pr`, `huss` y `ps`; la temperatura media sale de `(tasmax+tasmin)/2` y la
humedad relativa se estima desde `huss`, `ps` y `tas`.

| flag | default | nota |
|---|---|---|
| `--scenario` | `ssp585` | altas emisiones; también `ssp245` y `ref` |
| `--start-year` / `--end-year` | 2025 / 2040 | debe **contener** el periodo del paso 4 |
| `--aggregation` | `mean` | cómo colapsar las columnas GCM de ARClim |
| `--wind-source` | `auto` | intenta `sfcWind` y cae a `uas`+`vas` |
| `--humidity-file-mode` | `daily` | dejar así; `compatible` es el indexado legado |

⚠️ **Trampa del archivo de humedad.** En
`Dailyfuturedata_relativehumidity.nc` la dimensión `lat` **no es latitud**: es
el índice de punto `z`, en el mismo orden de filas del Excel, y `lon` es dummy
`[0,1]`. Los pasos 3 y 4 lo leen así (`.isel(lat=z, lon=0)`).

Nota sobre la grilla: los puntos dispersos se guardan en una grilla rectangular
lat × lon, así que la mayoría de las celdas quedan en `NaN` y solo las
combinaciones reales tienen datos (verificado: 3 celdas con dato, 6 en NaN, en
las posiciones correctas). Todo el código aguas abajo consulta con
`sel(lat=…, lon=…, method="nearest")` usando los pares exactos del Excel, así
que cae en la celda correcta — pero no inventes un punto intermedio.

Diagnósticos que vale la pena mirar la primera vez, en `Outputs/`:
`diagnostico_arclim_compatibilidad_ssp585.csv`,
`diagnostico_arclim_wind_source_ssp585.csv`,
`diagnostico_openmeteo_validez_por_punto.csv`.

### 2.4 Paso 3 — base horaria por zona

```bash
python Hourly_climate_zone_humidity_daily_patch.py
```

Cruza el patrón **horario** del histórico con el nivel **diario** del futuro →
`Outputs/BaseFhorPELP4v20z2anos.nc`: dos años horarios (2029–2030 = 17.520 h).

Este script **no tiene CLI**: los años están escritos adentro
(`yearBASE = range(2025, 2041)`, `year = range(2029, 2031)`,
`year0 = range(2017, 2022)`). Si cambias el periodo hay que editarlo.

### 2.5 Paso 4 — eventos extremos

```bash
python Extreme_events_arclim_final.py --start-year 2029 --end-year 2030
```

Genera **`Outputs/BaseHoraEWEPELP20z.nc`**, que alimenta todo lo demás:

```
dims:  time (17520) × lat (n_zonas) × lon (2)
vars:  tem [°C], wind [m/s], prec [mm], hur [%], FFDI [-], Fire [0/1]
lat  = point_id, en el orden de filas de SystemzoneV1.xlsx  (¡no es latitud!)
lon  = escenario:  0 = "baseline"  ·  1 = "with_EWE"
coords extra: zone_id, Lat_real, Lon_real, original_row_SystemzoneV1
```

En `with_EWE` las horas sobre el umbral de viento se multiplican por el factor
de ráfaga y la humedad baja se escala; en `baseline` queda la tendencia sin esa
amplificación.

| flag | default | qué controla |
|---|---|---|
| `--wind-threshold` | 20.8 m/s | velocidad máxima tras ráfaga |
| `--wind-gust-factor` | 1.263 | factor de ráfaga en `with_EWE` |
| `--humidity-threshold` | 15 % | umbral de humedad relativa baja |
| `--humidity-scaling` | 0.7 | factor sobre la humedad en días de baja HR |
| `--ffdi-threshold` | 15 | FFDI sobre el cual el día se marca como incendio |

Esta versión **desactiva a propósito** olas de calor, olas de frío y sequía:
conserva viento extremo, baja humedad relativa, FFDI e incendios.

Salidas adicionales: `resumen_EWE_arclim.csv`,
`orden_zonas_BaseHoraEWEPELP20z.csv` (el mapeo columna ↔ zona ↔ fila del Excel;
consúltalo si dudas del orden) y `EWEwind_hora_anual.npy`.

### 2.6 Correr los cuatro pasos de una

```bash
python Pipeline_extreme_events.py                    # todo
python Pipeline_extreme_events.py --skip-if-exists   # solo lo que falta
python Pipeline_extreme_events.py --only-step 4      # rehacer un paso
python Pipeline_extreme_events.py --dry-run          # ver comandos sin correr
python Pipeline_extreme_events.py --step2-args "--scenario ssp245"
```

⚠️ El orquestador tiene los nombres de archivo **sin guiones bajos**
(`Dailyfuturedatarelativehumidity.nc`, `resumenEWEarclim.csv`) y los scripts los
escriben **con** guión bajo. Consecuencia: `--skip-if-exists` no los detecta y
los pasos 2 y 4 se re-ejecutan aunque ya estén, y al final imprime
`[AVISO] … NO fueron creados` sobre archivos que sí existen. Es cosmético, pero
no te fíes de ese aviso.

### 2.7 Verificar y copiar

```python
import xarray as xr
ds = xr.open_dataset("Outputs/BaseHoraEWEPELP20z.nc")
print(ds); print(ds.zone_id.values)
print(ds.wind.values[8760:17520, :, 0].max())   # viento máx del año 2
```

Después hay que **copiarlo** a `abc/mi_proyecto/app/Scenario Database/Inputs/`,
que es donde `pipeline.py`, `construir_faultlin.py` y `construir_perfiles.py` lo
buscan (`RUTA_WIND_NC`).

---

## 3. Perfiles de generación

### 3.1 Cómo entra el perfil al modelo

El perfil es un **factor de planta normalizado en [0, 1]** que acota la
generación de cada unidad:

```
P_DG_sol[t,g]  ≤  alphaDGsolar[t,esc,g]  · CAP_DG_sol[g]
P_DG_eol[t,g]  ≤  alphaDGeolica[t,esc,g] · CAP_DG_eol[g] · a_eol_wind[t,g]
```

`CAP_DG_*` es la capacidad instalada (variable de decisión, en kW), así que el
valor del perfil en una hora **es** la fracción de esa capacidad disponible en
esa hora.

`a_eol_wind[t,g]` es un binario aparte, calculado dentro del modelo: **apaga
toda la generación eólica de una zona en cualquier hora en que haya al menos una
línea dañada en esa zona.** Conviene tenerlo presente porque es agresivo (una
línea caída anula el aporte eólico de toda la zona, esté o no afectada la
turbina) y porque hace que, en las horas de falla, el valor del perfil eólico de
esa zona sea irrelevante.

### 3.2 Generarlos para los puntos del caso de estudio

```bash
cd abc/mi_proyecto/app/app
python construir_perfiles.py --estado        # qué hay hoy
python construir_perfiles.py --forzar        # horaria, eólico del .nc
```

Escribe `artefactos/perfiles_horarios.npz` con `solar_por_zona` y
`eolico_por_zona`, ambos (n_horas × n_zonas). E1 lo prefiere automáticamente
sobre los Excel de 24 h.

| fuente | de dónde sale | por qué |
|---|---|---|
| eólico (default) | el **mismo `.nc`** que genera las fallas, mismo corte de horas y mismo eje de escenario | la turbina y la línea ven el mismo viento; cualquier otra fuente rompe esa coherencia |
| eólico alternativo | `--fuente-eolico openmeteo` | reanálisis histórico |
| solar | Open-Meteo (`--fuente-solar openmeteo`) | el `.nc` no trae irradiancia |
| cualquiera | `--fuente-solar explorador` / `--fuente-eolico explorador` | exportación del Explorador (§3.4) |

Modelos de conversión: fotovoltaico `CF = (GHI/1000)·[1 + γ(T_celda − 25)]` con
temperatura de celda por NOCT; eólico con curva de potencia genérica (cut-in
3 m/s, nominal 12, cut-out 25) y corrección de altura de buje por Hellmann
(`--altura-buje`, default 30 m).

Para el solar, la serie histórica se promedia por **(día del año, hora)** sobre
todos los años disponibles y se mapea al calendario del horizonte. Así conserva
estacionalidad y ciclo diario, y suaviza el ruido de un año puntual —que no
aporta nada, porque el año de fallas es otro.

### 3.3 Por qué horaria y no un perfil de 24 h

El estudio venía usando un perfil de 24 h repetido los 365 días
(`_perfil_diario_anual`). Tres razones para dejar de hacerlo:

1. **Estacionalidad.** En Aysén (−47°) el día de verano dura ~17 h y el de
   invierno ~7,5 h. Un perfil único hace enero idéntico a julio: el modelo ve
   sol donde no lo hay y no lo ve donde sí. Como las fallas caen en horas
   concretas del año, la hora del corte deja de ser informativa.
2. **Qué día representa.** El Excel vigente llega a **1,034** al mediodía. Un
   factor de planta de ~1 es un **día despejado**, no un promedio anual (el
   promedio de 5 años en el mismo punto llega a 0,51). Repetirlo 365 veces
   sobrestima la energía anual. Con resolución horaria la pregunta desaparece.
3. **Correlación con el evento.** Para el eólico esto pesa menos de lo que
   parece, porque `a_eol_wind` ya anula la generación de la zona dañada; pero
   sigue importando para las zonas **no** afectadas durante la misma tormenta y
   para las horas alrededor del evento, que definen con qué carga llega el
   almacenamiento.

El artefacto `perfiles.npz` **ya guardaba (n_horas × n_buses)**, así que esto no
cambia ningún formato: lo que aplanaba era el tilado.

Si por comparabilidad se quiere seguir con el formato viejo:
`--resolucion diaria24` reescribe los dos Excel, colapsando con el percentil 90
por hora del día (es lo que reproduce la convención del archivo vigente, no el
promedio).

### 3.4 Explorador Solar / Eólico y comparación de fuentes

`solar.minenergia.cl` y `eolico.minenergia.cl` son la referencia nacional (WRF a
~1 km, calibrado para Chile) y la única de las fuentes que resuelve topografía
de valle. Son aplicaciones web sin API pública: la serie se descarga a mano
eligiendo el punto en el mapa, y el script lee ese archivo.

```bash
# comparar todas las fuentes disponibles, sin escribir perfiles
python construir_perfiles.py --comparar

# con los exports del Explorador
python construir_perfiles.py --comparar \
    --solar  Z1=sol_z1.csv Z2=sol_z2.csv Z3=sol_z3.csv \
    --eolico Z1=eol_z1.csv Z2=eol_z2.csv Z3=eol_z3.csv
```

Deja `artefactos/comparacion_fuentes.csv` con, por zona y fuente: velocidad
media / p90 / máxima, CF medio, CF en el 1 % de horas de mayor viento, GHI medio
y CF solar. El lector de exports identifica las columnas por nombre (acepta
"Fecha", "Radiación Global", "Velocidad", "Factor de planta"…) y salta las filas
de metadatos del encabezado; si el archivo ya trae factor de planta lo usa
directo, sin aplicar ningún modelo encima.

---

## 4. La matriz de fallas

### 4.1 Qué es `FaultlinMC.npy`

Arreglo `int8` de forma **(horas, escenarios, líneas)** = hoy `(8760, 100, 15)`,
donde `1` significa "esta línea falla en esta hora, en este escenario". Es la
**fuente de verdad** de las fallas; la clave `data["faultlinMC"]` del `app.pkl`
quedó en ceros y no sirve (ver `pipeline.py`, invariante 2).

### 4.2 Cómo se calcula

```bash
python construir_faultlin.py --estado                 # insumos, sin escribir
python construir_faultlin.py --semilla 42 --forzar    # recalcular
```

Este script **es nuevo**: la simulación existía solo *comentada* dentro de
`Prueba 12 oficial.ipynb` (celdas 11 a 14), con tres variantes y sin marca de
cuál era la vigente. La vigente es la **lognormal de la celda 14**, y se sabe
porque es la misma curva con la que E2 calibra el refuerzo estructural: si la
matriz base viniera de otra curva, el "ahorro" del refuerzo estaría medido
contra una física distinta de la que generó las fallas.

**a) Viento a la altura del conductor.** `wind` del `.nc` está medido a 10 m; se
corrige a los 12 m del poste con el perfil de potencia de Hellmann:
`v₁₂ = v₁₀ · (12/10)^(1/7)` → factor 1,0263.

**b) Curva de fragilidad.** Hay dos, y se eligen con `--curva`:

| `--curva` | forma | fuente | qué la distingue |
|---|---|---|---|
| `lognormal` (default) | `P = Φ(ln(v/θ)/β)` | Panteli et al. 2017, ajustada | estructural: modela la rotura del poste |
| **`dunn`** | `P = α · l[km] · v^exp`, α = 2·10⁻¹⁷, exp = 9,91 | Dunn et al. 2018 vía Shi et al. 2021 ec. (5) | **empírica**, escala con el largo del tramo e **incluye la caída de árboles** |

**La vigente es `dunn`.** Tres razones, en orden de peso:

1. **Incluye los árboles**, que son el mecanismo dominante de falla en
   distribución (Karagiannakis et al. 2025, §4.3). Las analíticas modelan sólo
   la rotura estructural del poste — y entonces multiplicar por un factor de
   vegetación (Canham) sobre ellas mezcla dos cosas distintas.
2. **Escala con el largo del tramo**, así que no hay que suponer un número de
   postes por línea.
3. Es empírica sobre **12.000 fallas reales** de líneas de distribución en el
   Reino Unido (2003-2010), no transferida desde transmisión.

Ojo con el exponente ~10: satura en `P = 1` a 48,4 m/s para un tramo de 1 km
(antes si el tramo es más largo), de ahí el clip a 1 en `p_falla_dunn`.

**Sobre ráfaga, no sostenido.** `--escenario-clima with_EWE` (eje `lon = 1`).
La convención de la literatura de fragilidad es ráfaga de 3 s (Xu et al. 2025;
Darestani y Shafieezadeh 2019 construyen su demanda sobre cargas ASCE 2013, que
son de ráfaga). Con sostenido no hay fallas en Arauco: el máximo del año es
19,6 m/s y ninguna curva publicada despega ahí — ver §4.7.

> El perfil eólico de `construir_perfiles.py` hace lo **contrario**: usa
> `baseline` (sostenido), porque la curva de potencia IEC 61400 se define sobre
> media de 10 min. Es el mismo campo de viento; cada uso toma el estadístico que
> le corresponde. Pasarle la ráfaga infla el factor de planta 3,3× (0,18 → 0,60).

**c) Monte Carlo con reparación.** Cada hora y escenario se sortea cada línea
*disponible*; la que falla queda fuera las horas de reparación y no puede volver
a fallar mientras se repara. Sin esa memoria, una tormenta de 10 h generaría 10
fallas seguidas en la misma línea.

Verificación de orden de magnitud: 720 h × 5 escenarios da ≈1,4 fallas por
escenario → ≈17 extrapolado a 8760 h, contra las **17,5 por escenario** del
`FaultlinMC.npy` histórico (1747/100). Reproduce el archivo congelado.

| flag | default | nota |
|---|---|---|
| `--horas` / `--escenarios` | 8760 / 100 | dimensiones de la matriz |
| `--viento` | el `.nc` histórico | apuntar al del caso de estudio |
| `--anio-offset` | 8760 | hora inicial en el `.nc`; 8760 = segundo año (2030) |
| `--escenario-clima` | `baseline` | **usar `with_EWE`**: la fragilidad va sobre ráfaga |
| `--curva` | `lognormal` | **usar `dunn`** |
| `--theta` / `--beta` | 30.0 / 0.25 | sólo aplican a `--curva lognormal` |
| `--enlaces-inmunes` | — | anula las fallas de los enlaces — ver abajo |
| `--horas-espera` | 3 | horas de reparación — §4.4 |
| `--semilla` | ninguna | sin semilla **no es reproducible** |
| `--sin-mitigadas` | — | ⚠️ **también reescribe `faultlin_meta.json`** declarando que no hay curvas directas, y entonces E2 cae al *fallback* de adelgazamiento. Si se usa para una verificación, rehacer después la corrida completa. |
| `--forzar` | — | necesario para sobrescribir |

**`--enlaces-inmunes`.** Los enlaces (`tie_indices` de `red.pkl`, hoy las líneas
12, 13 y 14) están normalmente **abiertos** y con `z_tie` fijo en 0: su falla no
puede producir ENS, sólo infla la cuenta de fallas y ensucia la correlación
entre amenaza y consecuencia. Con el flag pasa de 798 a 727 fallas y la
correlación ENS–fallas sube de 0,710 a 0,754.

Se anulan **después** del sorteo, nunca antes: sacarlos de las columnas activas
cambiaría cuántos números consume el generador por hora y desplazaría toda la
matriz. Anulándolos después, el resto queda **bit a bit idéntico** a la misma
corrida sin el flag (verificado).

La corrida vigente:

```bash
python construir_faultlin.py \
    --viento "../Proyecto oficial/Outputs/BaseHoraEWE_arauco.nc" \
    --escenario-clima with_EWE --curva dunn --anio-offset 8760 \
    --escenarios 100 --semilla 42 --enlaces-inmunes --salida FaultlinMC.npy
# → 727 / 484 / 168 / 36  (base / veg / ref / ug)
```

La sensibilidad a los parámetros de la curva es fuerte (la falla vive en la cola
de la distribución), así que conviene reportar los valores elegidos y una
sensibilidad. Con la lognormal, β pesa **240×** y θ **35×** sobre las fallas
anuales.

### 4.3 Las medidas de hardening, con su propia curva

Por defecto se simulan **cuatro matrices en la misma corrida**:

| archivo | medida | curva | fallas/año |
|---|---|---|---:|
| `FaultlinMC.npy` | red tal cual | Dunn empírica | **727** |
| `FaultlinMC_veg.npy` | gestión de vegetación | base × caída de árboles de Canham, DBH 30 → 10 cm | 484 (−33 %) |
| `FaultlinMC_ref.npy` | refuerzo estructural | curva evaluada en `v/1,1671` (Darestani, poste clase 5 → clase 3) | 168 (−77 %) |
| `FaultlinMC_ug.npy` | soterramiento | base × (1 − 0,95) | 36 (−95 %) |

**El refuerzo es el mismo desplazamiento físico en las dos curvas**, expresado
distinto. La razón clase 3 / clase 5 de Darestani es 1,1671, y la curva se
evalúa en `v/1,1671`:

- con la lognormal eso equivale a mover θ (30 → 35,0, o 42,43 → 49,5);
- con la ley de potencia, a **multiplicar la probabilidad por 1,1671⁻⁹·⁹¹ =
  0,216** — un factor constante, igual que `factor_vegetacion` y `rho_ug`.

No son dos supuestos distintos: es la misma medida en dos parametrizaciones.

**Por qué es mejor que lo anterior.** Antes las mitigadas se obtenían por
*adelgazamiento de Bernoulli*: se tomaba la matriz base y cada falla ya sorteada
se "curaba" con probabilidad ρ. Era la única opción, porque el `FaultlinMC.npy`
histórico se generó **sin semilla** y su sorteo no se puede reproducir. Ahora que
la corrida lleva semilla, la medida entra donde corresponde: como una curva de
fragilidad propia.

**Números aleatorios comunes.** El sorteo uniforme se hace una vez por hora y lo
usan las cuatro variantes. La comparación queda pareada —la medida se evalúa
contra la misma secuencia de ráfagas, no contra otro año simulado— y toda falla
mitigada es también falla base en el mismo sorteo (verificado: cero fallas
mitigadas fuera de la base). Lo que sí se separa son las trayectorias, y debe
ser así: una línea que con poda no falló está disponible antes que su gemela sin
poda, y puede fallar en una hora en que la otra todavía se repara. Eso el
adelgazamiento no podía representarlo.

| medida | curva directa | E2 con el `.nc` real | E2 con *fallback* plano |
|---|---|---|---|
| vegetación | −29 % | −35 % | −30 % (ρ = 0,30, Cerrai et al. 2019) |
| refuerzo | **−82 %** | **−83 %** | −42 % (ρ = 0,42, Hu et al. 2021) |
| soterramiento | −93 % | −95 % | −95 % (ρ = 0,95) |

Los dos primeros métodos convergen; el que se sale es el *fallback* plano, que
es lo que el estudio tiene calculado hoy (§4.5).

E2 usa las matrices del script si existen, comprobando contra
`artefactos/faultlin_meta.json` que declare
`metodo: curva_directa_numeros_comunes`. No basta con que el archivo exista
—`FaultlinMC_veg.npy` está en el repo desde antes, generado por adelgazamiento
en la celda 15 del notebook— así que el meta es lo que ata las matrices a la
corrida que las produjo. Si falta o no calza la forma, E2 vuelve al
adelgazamiento y lo registra en `fallas_meta.json` como `metodo_mitigadas`.

### 4.4 Las horas de reparación son un parámetro de entrada

Cuánto tarda en reponerse una línea **no es una constante del código**: define
cuánto dura la interrupción, entra directo en SAIDI y ENS, y fija cuánto vale
cualquier medida que acorte la indisponibilidad. Se ingresa en los tres lugares
donde se usa:

| dónde | cómo | para qué sirve ahí |
|---|---|---|
| la pestaña | campo *"Horas de reparación de una línea fallada"* (paso 5) | se guarda en `red_nueva.pkl` y se inyecta en los comandos generados |
| Monte Carlo | `construir_faultlin.py --horas-espera N` | bloquea la línea mientras se repara |
| E2 | `pipeline.py --etapas E2 --horas-reparacion N` | construye `repairMC`, con la que el MILP sabe cuándo vuelve la línea |

Los dos últimos son el mismo fenómeno físico y **deberían coincidir**. E2 sin
`--horas-reparacion` toma el valor que registró el Monte Carlo
(`faultlin_meta.json`); si se le da uno distinto lo usa, pero avisa.

> **Herencia:** el notebook usaba `linespera = 3` en el sorteo y `pipeline.py`
> tenía `HORAS_REPARACION = 4` fija para `repairMC`. Los dos conviven en lo ya
> calculado. Los defaults se mantienen (3 y 4 **solo cuando no hay registro**)
> para no cambiar en silencio resultados existentes; en cualquier corrida nueva
> conviene fijar uno solo.

### 4.5 El fallback silencioso de E2

Si `pipeline.py` no puede leer el `.nc` de viento, E2 **no falla**: usa
porcentajes planos de literatura (ρ_veg = 0,30, ρ_ref = 0,42) en vez de la curva
calibrada hora a hora. Queda registrado en `artefactos/fallas_meta.json` como
`"fuente_viento": "fallback_plano"` y en ningún otro lado. La pestaña 🗺️
*Origen de los datos* ahora lo muestra arriba de todo.

> **Resuelto el 2026-09-08.** `fallas_meta.json` decía `fallback_plano`; se
> rehizo la etapa y hoy dice `curva_directa`. El refuerzo pasó de −42 % a −77 %.
>
> Se vio el fallback en vivo, y conviene saber cómo se manifiesta: una corrida
> de verificación con `--sin-mitigadas` reescribió `faultlin_meta.json` dejando
> `mitigadas: {metodo: null}`, y E2 —sin error ni aviso— volvió al
> adelgazamiento. Se detectó porque `ref` saltó de 168 a **613** fallas. **Si
> los totales de E2 no calzan con los que imprimió `construir_faultlin.py`, es
> esto.**

### 4.6 Decisiones heredadas, y cuáles se cerraron

1. ~~**Se usa `lon = 0` (`baseline`), no `with_EWE`.**~~ **Revertido el
   2026-09-08.** El notebook leía `Base085.wind.values[8760:17520, :, 0]`, o sea
   viento sostenido. Hoy las fallas se sortean con **ráfaga** (`with_EWE`), que
   es la convención de la literatura de fragilidad. El motivo está en §4.7: con
   sostenido, Arauco no produce **ninguna** falla bajo ninguna curva publicada.
2. **El sorteo original no tiene semilla.** `random.seed(42)` en el notebook
   afecta al módulo `random`, que la celda vigente no usa (usa
   `np.random.random`). El `FaultlinMC.npy` histórico es irreproducible; toda
   corrida nueva lleva `--semilla 42`.
3. **Los enlaces entraban al sorteo y su falla no tenía consecuencia.** Se
   cerró por los dos lados: `--enlaces-inmunes` los saca de la matriz, y en
   `optimization_base.py` / `optimization_6_oficial.py` se agregó la restricción
   que faltaba —un enlace dañado no se puede cerrar—, replicando lo que ya
   existía para `sw_cand_list`. Sin eso, `reconfig_topo` e `inv_tie` podían usar
   un enlace roto y quedaban sobrevaloradas.

**Tres bugs latentes corregidos el 2026-09-08**, los tres del mismo tipo:
parámetros del módulo usados donde correspondía el de la corrida. Ninguno se
notaba con los defaults, y los tres habrían falseado cualquier sensibilidad.

| dónde | qué hacía | qué hace ahora |
|---|---|---|
| `construir_faultlin.theta_refuerzo()` | usaba `THETA_DEFAULT` (30) en vez de la θ de la corrida | recibe `theta_base`. Con `--theta 42.43` el refuerzo quedaba en 35,0 m/s, **por debajo** de la base: "reforzar" aumentaba las fallas un 6386 % |
| `curvas_mitigadas` | usaba `P.BETA_REF` en vez del `beta` recibido | usa `beta`. Coincidían por casualidad (ambos 0,25) |
| `faultlin_meta.json` | declaraba `curva: lognormal, θ=30, β=0.25` aunque la corrida fuera Dunn | declara la curva real y pone θ/β en `null` |

Queda **uno abierto**: la ruta de *fallback* de `pipeline.py` (líneas 748 y 752)
tiene `theta_ref = 30.0 * ratio` y `_p_falla_lognormal(wind, 30.0, 0.25)`
hardcodeados. Es código muerto mientras existan las tres matrices mitigadas,
pero si alguna vez se corre sin ellas devuelve un ρ calculado con la curva
equivocada, en silencio.

Y un patrón latente en **13 lugares**: `tie_indices = set(LINES[-3:])` en vez de
leerlo de `red.pkl`. Hoy coinciden (ambos dan `[12, 13, 14]`), así que no hay
bug activo. Corregidos `grafo.py` y `data_loader.py`; siguen con el hardcodeo
los dos optimizadores vigentes, `helpers_inversion.py` y la UI.

### 4.7 De dónde sale la curva, y por qué sobre ráfaga

Esta sección es el respaldo de la elección. Sirve de borrador para el capítulo
de metodología.

**El punto de partida estaba mal leído.** El `θ = 30 m/s` heredado es el
`w_critical` de Panteli et al. (2017), ec. (6) — la velocidad a la que la
probabilidad **empieza a despegar** desde la tasa de buen tiempo, donde vale
`P = 0,01`. El código lo usaba como **mediana** de una lognormal, donde vale
`P = 0,50`. Cincuenta veces más, a la misma velocidad. No se notaba porque con
viento sostenido el máximo de Arauco es 19,6 m/s y la curva operaba en su cola
izquierda; al pasar a ráfagas quedó expuesto.

**Tres rutas independientes convergen.** A 34,6 m/s, la ráfaga máxima del año:

| ruta | nivel | P |
|---|---|---:|
| Panteli 2017, ajustada a lognormal (θ = 42,43, β = 0,149) | corredor de transmisión | 0,0855 |
| Darestani y Shafieezadeh 2019 + agregación en serie, 25 postes | poste de distribución | 0,0839 |
| **Dunn et al. 2018** (`α·l·v^9,91`), tramo de 2,4 km | **línea de distribución, empírica** | **0,0858** |

Transmisión analítica, postes analítica y distribución empírica dan lo mismo
dentro de un 2 %. Esa coincidencia es el argumento: la elección de curva no
determina el resultado.

**Sobre qué variable.** Aplicando cada curva a la variable que le corresponde,
Arauco **no produce fallas**:

| curva | sostenido | ráfaga |
|---|---:|---:|
| Panteli 30/60 (transmisión) | 0,00 | 2,89 |
| Nourian y Kazemi 2022, 20/40 (distribución) | 0,10 | 288,6 |
| la misma, reescalada a ráfaga con la razón del sitio (1,82) | — | 0,08 |
| Dunn empírica | 0,02 | 7,68 |

Las combinaciones internamente coherentes dan ≈ 0. **Arauco no tiene régimen de
viento sostenido para tumbar líneas**: el máximo es 19,6 m/s y el umbral de
distribución de Nourian es 20.

Se resuelve así: la matriz de fallas es un **generador de escenarios**, no un
pronóstico. Se corre sobre ráfaga, apoyado en que Darestani —la curva de postes
de distribución— está construida sobre cargas **ASCE 2013**, que son de ráfaga
de 3 s, y en que Xu et al. (2025) declaran esa convención explícitamente. Es un
supuesto y hay que declararlo como tal.

**Referencias, por si se citan.** Nourian y Kazemi (2022), *J. Energy Storage*
48, es la más cercana al estudio: red de distribución activa, eólica,
almacenamiento y cargas flexibles sobre IEEE 33 barras, con la **misma
estructura de fragilidad** (su ec. 47) y valores de distribución 20/40 m/s
—esto es, transmisión ÷ 1,5—. Confirma que el andamiaje es práctica estándar.

**Supuestos a declarar:**

1. La ráfaga de Open-Meteo se trata como ráfaga de 3 s. Su duración real **no
   está documentada**.
2. El "sostenido" de Open-Meteo es **media horaria**, no de 1 minuto: por eso la
   razón ráfaga/sostenido del sitio es 1,82 y no el τ_G = 1,49 de Xu.
3. El muestreo horario sobre 8760 h **sobreestima** las fallas frente a un
   muestreo más grueso: `1 − ∏(1 − f(t_k))` crece con la resolución (Xu et al.
   2025). El remedio publicado es el método HRSRA, que no se implementó.
4. Los enlaces se declaran inmunes al viento (§4.2).

---

## 5. Lectura de los datos de la red

### 5.1 Cómo se leía hasta ahora

En `Prueba 12 oficial.ipynb`, mezclado en tres celdas:

- **celda 2** — `create_cigre_network_mv()`, lee `SystemzoneV1.xlsx`, proyecta
  con `pyproj` y escribe `net.bus['geo']`. Todos los buses de una zona reciben
  **el mismo punto** (`x_bus = x0`, sin desplazamiento): el estudio no modela la
  posición individual de cada barra, solo a qué zona climática pertenece. Útil
  saberlo antes de leer un mapa donde 5 buses caen exactamente encima.
- **celda 5** — pisa los parámetros de línea: la CIGRE viene con **cable
  subterráneo** y el estudio modela **línea aérea**, que es la expuesta al
  viento. R = 0,510 Ω/km · X = 0,366 Ω/km · C = 10,097 nF/km · I_max = 0,145 kA.
  También fija `bus_zone` a mano.
- **celda 8** — cada línea hereda la zona de su `from_bus`.

⚠️ En el `bus_zone` de la celda 5 hay **tres versiones comentadas** una encima
de otra. La vigente es la última:
`{0:1, 1:1, 2:1, 3:1, 4:1, 5:3, 6:3, 7:2, 8:2, 9:3, 10:3, 11:3, 12:2, 13:2, 14:2}`.
Y `lin_Z1/Z2/Z3` de la celda 2 (que asignan las **coordenadas**) usan un reparto
**distinto** del de `bus_zone` (que asigna el **viento**). Vale la pena verificar
cuál corresponde: cambia qué línea se sortea contra qué serie.

### 5.2 Qué exige el modelo de la red

Lo que realmente lee el MILP; el resto se ignora:

| tabla | columnas | dónde se usa |
|---|---|---|
| `net.bus` | índice, `name`, `vn_kv` | conjuntos `BUSES`, `GENS`, `ESS` |
| `net.line` | `from_bus`, `to_bus`, `length_km`, `x_ohm_per_km`, `max_i_ka` | reactancia, capacidad, costos por km |
| `net.load` | `bus`, `p_mw`, `scaling`, `in_service` | `P_D_base[b] = Σ p_mw · scaling · 1000` [kW] |
| `net.trafo` | `sn_mva` (fila 0) | cota de inyección de **cada** subestación |

Requisitos no obvios que el validador de la pestaña revisa:

- **Índices contiguos 0..n−1** en `bus` y `line`. El código mezcla
  `net.line.length_km.iloc[i]` con `net.line.from_bus[l]`: con índices no
  contiguos las dos formas dejan de coincidir y el modelo confunde líneas sin
  dar error.
- **Índice desde 0 en `trafo` y `load`**. `net.trafo.sn_mva[0]` es acceso por
  ETIQUETA, no por posición: una red bajada de un catálogo trae otros números
  (en MV Oberrhein los trafos son 114 y 142) y el modelo revienta al
  construirse con `KeyError: 0`.
- **Las cabeceras de subestación tienen que ser los buses 1 y 12.** Está
  escrito a fuego en el modelo (`sub_p = m.P_SUB[t,0] if b == 1 else … if
  b == 12`) y es el requisito más peligroso, porque si no se cumple el modelo
  **no falla**: inyecta la subestación en un bus interior cualquiera y
  entrega un resultado plausible y falso. El bus 0, además, es la referencia
  angular (`theta[t,0].fix(0)`).
- **Al menos una fila de `trafo`**, o el modelo revienta con `IndexError`.
- **Demanda base ≤ capacidad de las subestaciones.** No es un requisito del
  código sino de coherencia: si la demanda pico supera lo que las cabeceras
  pueden inyectar, el MILP recorta carga *sin que haya ninguna falla* y la ENS
  del caso base deja de medir el efecto del clima. Es lo que pasa con MV
  Oberrhein si se ignora el `scaling` de 0,6 de sus cargas: 61,9 MW contra
  50 MVA instalados.

Los tres primeros los arregla la pestaña con el botón **🔧 Adaptar la red**
(`entrada_red.adaptar_para_milp`), que renumera buses y líneas y deja las
cabeceras en 1 y 12 **sin tocar topología, parámetros eléctricos ni
coordenadas**, y muestra la lista de lo que cambió. Eso último no es
cosmético: después de adaptar, el bus 39 del archivo original puede ser el
bus 1, y cruzar resultados con la fuente por número daría cualquier cosa. Para
eso está la columna de nombre, que no se renumera.

### 5.3 La pestaña 📥 Datos de entrada → 🗺️ Origen de los datos

Reemplaza a las celdas 2, 5 y 8 y deja las dos salidas coherentes entre sí.

**Paso 0 — elegir el caso de estudio.** Un **caso** es una carpeta de artefactos
(`casos.py`): su red, sus perfiles, sus fallas y sus resultados. La pestaña abre
con el selector y **no muestra nada más hasta que haya uno elegido**, porque un
✅ solo significa algo referido a un caso concreto: hasta el 2026-09-22 abría con
el estado de `artefactos/` aunque se hubiera entrado a cargar otra red, o sea
mostraba datos correctos del caso equivocado.

Cada caso se prepara sin tocar a los demás. Esto reemplaza la receta manual de
§7 ("archiva lo anterior en vez de pisarlo"): ahora se crea una carpeta nueva y
el caso viejo sigue intacto.

Dos cosas que conviene tener claras:

- La carpeta **activa del proceso** se fija con `ERAD_ARTEFACTOS` al arrancar
  (`pipeline.usar_artefactos`). Preparar otro caso desde la pestaña funciona
  completo, pero las pestañas de optimización, resultados y barrido siguen
  mostrando el caso con el que arrancó la app: `data_loader` leyó sus artefactos
  al importarse y once módulos copiaron esos valores en su propio import. La
  pestaña avisa y da el comando para reabrir la app en el caso nuevo.
- `artefactos/` es el caso de validación de la tesis y está protegido: la
  pestaña y `fallas_sinteticas.py` se niegan a escribirle fallas inventadas.

**Paso 1 — cargar la red.** Cuatro formatos:

| formato | cómo | qué importa |
|---|---|---|
| CIGRE MV integrada | un botón | la opción "convertir a línea aérea" = la celda 5 |
| pandapower | subir `.json` / `.p` / `.xlsx`, o ruta local | `pp.from_json` / `from_pickle` / `from_excel` |
| OpenDSS | **ruta local** al `Master.DSS` | buses, líneas, cargas y coordenadas. Reguladores, capacitores y curvas horarias se ignoran: el MILP no los usa. Se pide ruta y no archivo porque un caso DSS son varios archivos enlazados con `Redirect`. |
| CSV | tres tablas (buses / líneas / cargas) | formato en el desplegable de la pestaña |

El importador de OpenDSS convierte las unidades de la línea (ft, m, mi…) tanto
en el largo **como** en R/X/C, que OpenDSS entrega "por unidad de largo".
Probado contra `IEEE_13_Bus-G`: 2,50 km totales, 0,036 Ω/km, 0,075 Ω/km.

**Paso 2 — validación y adaptación.** Chequea §5.2 y muestra buses, líneas,
demanda base (ya con `scaling`), largo total y capacidad de las subestaciones
con su holgura. Un error en rojo impide continuar, pero los de numeración se
arreglan ahí mismo con **🔧 Adaptar la red**, porque casi ninguna red bajada de
un catálogo cumple los supuestos del modelo y trabar a la usuaria en un
problema mecánico no aporta nada. El detalle de la renumeración queda visible
en un desplegable. Si falta el transformador, ofrece declararlo.

En `.pkl` la pestaña acepta las tres formas que existen en el repo: el formato
de `pp.to_pickle`, una `pandapowerNet` guardada con `pickle.dump`, y un paquete
del pipeline (`red.pkl`, que es un dict con la red en la clave `net`). Antes
solo entendía la primera y con las otras fallaba con `'pandapowerNet' instance
has no attribute 'version'`, que no le dice a nadie qué hacer.

**Paso 3 — zonas por bus.** Tabla editable, con resumen por zona (buses, líneas
y demanda) para no dejar una zona con punto climático y ninguna línea.

**Paso 4 — punto geográfico por zona.** Si la red trae coordenadas geográficas
chilenas, propone el centroide de cada zona. Si trae coordenadas que **no** son
lat/lon de Chile —la CIGRE de pandapower viene con `(7, 16)`, un lat/lon
perfectamente válido en África— lo dice y obliga a fijarlas: ARClim y Open-Meteo
se consultan con lat/lon reales y con cualquier otra cosa devuelven el punto de
grilla equivocado, sin error. También hay un botón para partir de los puntos del
`SystemzoneV1.xlsx` vigente.

**Paso 5 — parámetros y exportación.** Escenarios Monte Carlo, horizonte,
`N_eventos`, **horas de reparación** (§4.4), buses críticos, tie-lines,
seccionadores y perfiles.

- **Tie-lines**: vienen propuestas como **las líneas con switch abierto**, que es
  la definición física de un enlace normalmente abierto. El viejo default
  `LINES[-3:]` solo era correcto en la red del estudio, donde las tres últimas
  resultaban ser las de respaldo. Los optimizadores ahora las reciben por config
  (`tie_indices`), así que lo que se elija acá es lo que se usa.
- **Seccionadores**: se muestra la tabla de switches de la red y se marcan los
  que de verdad seccionan tramos. **No se deducen automáticamente**, porque un
  switch cerrado puede ser solo de extremo de línea: en MV Oberrhein los 316
  switches cerrados no son seccionadores, y darlos por tales inventaría una
  capacidad de aislación que la red no tiene. Vacío es una respuesta válida y
  significa el supuesto conservador (una falla deja fuera el alimentador
  completo); la pestaña lo dice en vez de dejarlo implícito. Sin esta lista, la
  variante de aislación realista no se puede correr en un caso nuevo.
- **Perfiles**: se eligen los archivos de demanda, solar y eólico del caso. Si se
  heredan los de `app/`, que son los de la red de 15 barras, la pestaña advierte:
  la curva de carga de otro sistema sirve para probar el modelo, no para
  concluir.

Salen `SystemzoneV1.xlsx` (descarga o escritura directa en las carpetas
`Inputs/`) y `<caso>/red_nueva.pkl`.

**Después de exportar — las etapas, desde la misma pestaña.** Ya no hay un bloque
de comandos para copiar a ciegas ni una tabla de chequeos aparte: cada etapa (E0 a
E5, más el Monte Carlo de fallas) es un paso con su estado, su tiempo estimado y
un botón que la corre **en segundo plano** con `--artefactos <caso>`, reutilizando
los jobs de la v8 (`estudio_v8/jobs.py`). El comando exacto se muestra igual, en
un desplegable, y sale de la misma lista que se ejecuta, así que no pueden
divergir: si prefieres consola, o vas a cerrar la app, copias de ahí.

**De dónde sale la red en E0/E1.** `--fuente app_pkl` lee `app/app.pkl`, el
registro congelado del estudio (711 MB), y la red que trae es la CIGRE de 15
barras. Por eso **solo vale para el caso de validación**: para cualquier otro caso
la red sale de su propio paquete (`--fuente red_nueva`). Si se piden E0 o E1 con
`app_pkl` sobre otro caso, `pipeline.py` se niega y dice el comando correcto; antes
llenaba ese caso con la red del estudio sin dar error, y se descubría mucho
después, cuando los resultados no calzaban con la red que se había subido. La
pestaña, por su parte, no ofrece E0/E1 mientras el caso no tenga paquete.

Los escenarios de E5 se eligen con los mismos criterios que en «Optimización y
análisis», no con un rango escrito a mano: **toda la población** (lo que E5 existe
para producir, y lo que la selección por percentil de severidad exige), la
**muestra estratificada** de `seleccionar_escenarios.py` si ya se corrió en el
caso, o una lista personalizada con índices desde 0.

Para las fallas hay dos caminos, en pestañas separadas y marcados:

| camino | qué hace | sirve para la tesis |
|---|---|---|
| Cadena climática | `construir_faultlin.py` + E2: viento real del `.nc`, curva de fragilidad, Monte Carlo | **sí**, es el del estudio |
| `fallas_sinteticas.py` | tormentas sorteadas por zona, sin viento ni fragilidad | **no**, solo para ver si el modelo corre |

El generador sintético deja `fuente_viento = "sintetico"` y `no_validado = true`
en `fallas_meta.json` —el mismo archivo que escribe E2— así que el aviso aparece
en toda la app mientras esas fallas estén en uso, sin código nuevo. Y se niega a
escribir en `artefactos/`.

### 5.4 Por qué no se escribe un `app.pkl` nuevo

`app.pkl` son ~740 MB donde la red convive con los resultados congelados, y de
ahí el pipeline solo rescata **seis claves**: `net`, `bus_zone`, `P_D_base`,
`N_eventos`, `numesc` y los perfiles. `red_nueva.pkl` son esas seis claves y
nada más (~60 kB): se puede versionar, revisar y rehacer sin mover 740 MB, y
`app.pkl` sigue siendo de solo lectura (invariante 1 de `pipeline.py`).

Los perfiles tampoco necesitan el pickle: con `--fuente red_nueva`, E1 los
reconstruye desde `perfiles_horarios.npz` si existe, y si no desde los archivos
sueltos de `app/` (`Perfil_demanda.csv` y los dos Excel), que es lo que hacía la
celda 38 del notebook.

---

## 6. Cadena 2 — el pipeline de la app

```bash
python pipeline.py --estado                 # qué hay y qué falta
python pipeline.py --etapas E0,E1,E2,E3     # datos: segundos, sin solver
python pipeline.py --etapas E4              # caso sin fallas: 3 MILP
python pipeline.py --etapas E5 --escenarios 0-19
```

| etapa | qué produce | lee | tarda |
|---|---|---|---|
| **E0** | `red.pkl` — red, conjuntos, zonas, `P_D_base`, tie-lines, candidatas a switch | `app.pkl` o `red_nueva.pkl` | segundos |
| **E1** | `perfiles.npz` — demanda, solar y eólico por bus | ídem, o `perfiles_horarios.npz` / los Excel | segundos |
| **E2** | `fallas.npz` — matriz base + `repairMC` + las tres mitigadas | `FaultlinMC*.npy` + el `.nc` | ~1 min |
| **E3** | `costos.pkl` — CAPEX/OPEX de todas las medidas | `red.pkl` | segundos |
| **E4** | `sin_fallas.pkl` — caso base sin fallas (3 corridas) | E0–E3 | minutos |
| **E5** | `baselines/esc_XXX.npz` — caso base con fallas, por escenario | E0–E3 | ~1 min por escenario |
| **E6** | `meta.json` — índice y estado | — | instantáneo |

Detalle de E1: `demand_factor` y `alphaDG*` viven en el pickle como `np.tile` de
(8760, 100, 15) = 105 MB cada uno, pero el contenido real es un perfil de 8760
valores. Se guardan compactos y `expandir_perfiles` los devuelve como vistas de
solo lectura (`np.broadcast_to`), lo que ahorra ~315 MB de RAM.

Con `--fuente red_nueva`, E1 guarda además `solar_por_bus`: en el artefacto
congelado el perfil solar quedó como **uno solo para las tres zonas** (columna
Z3), aunque el notebook usaba Z1/Z2/Z3. Se respeta el valor histórico donde ya
está, pero no se replica en una red nueva.

Banderas útiles de E4/E5 (detalle completo en `GUIA_BARRIDO.md`): `--lim-dg` +
`--etiqueta` para el tope de capacidad de DG, `--seccionadores` para la variante
de aislación, `--horas` para pruebas cortas.

---

## 7. Cambiar de red o de datos: qué hay que rehacer

Cambiar la red **invalida casi todo aguas abajo**: la matriz de fallas está
dimensionada a `(horas, escenarios, n_líneas)`, y los baselines de E5 y el
barrido se calcularon con esa matriz.

```bash
# 0. En la app: cargar la red, fijar zonas, puntos y horas de reparación,
#    y exportar las dos salidas.

# 1. Clima — SOLO si cambiaron los puntos geográficos.   [minutos: consulta APIs]
cd "abc/mi_proyecto/app/Proyecto oficial/Scripts"
python construir_ano_operativo.py --anio-previo 2023 --anio-operativo 2024 \
       --salida-nc BaseHoraEWE_arauco.nc
#    El .nc se pasa con --viento a los pasos 3 y 4; ya no se copia a
#    "Scenario Database/Inputs/".
#    Años alternativos para sensibilidad: generar_anios_alternativos.py

# 2. Red del modelo, desde el paquete.                              [segundos]
cd "../../app"
python pipeline.py --fuente red_nueva --etapas E0,E3 --forzar

# 3. Perfiles de generación.                                     [~1 min/zona]
#    OJO: --puntos es obligatorio, o toma un SystemzoneV1.xlsx viejo.
#    OJO: --escenario-clima baseline (SOSTENIDO) — ver §4.2.
python construir_perfiles.py --forzar \
       --viento "../Proyecto oficial/Outputs/BaseHoraEWE_arauco.nc" \
       --escenario-clima baseline --anio-offset 8760 \
       --fuente-solar openmeteo \
       --puntos "../Proyecto oficial/Inputs/SystemzoneV1_arauco.xlsx"
python pipeline.py --etapas E1 --forzar

# 4. Matriz de fallas + las tres mitigadas.                           [~1 min]
#    OJO: --escenario-clima with_EWE (RÁFAGA), al revés que el paso 3.
python construir_faultlin.py \
       --viento "../Proyecto oficial/Outputs/BaseHoraEWE_arauco.nc" \
       --escenario-clima with_EWE --curva dunn --anio-offset 8760 \
       --escenarios 100 --horas-espera 3 --semilla 42 \
       --enlaces-inmunes --salida FaultlinMC.npy --forzar

# 5. Ensamblado de fallas (usa las mitigadas y las mismas horas del paso 4).
python pipeline.py --etapas E2 --horas-reparacion 3 --forzar

# 6. Caso sin fallas y baselines.                     [minutos / ~36 s por esc]
python pipeline.py --etapas E4 --lim-dg 0.30 --lim-dg-modo alimentador \
       --etiqueta LIM-0.30-ALIM --forzar
python pipeline.py --etapas E5 --seccionadores 0,2,9,10

# 7. Elegir la muestra del barrido, estratificada por ENS.          [segundos]
python seleccionar_escenarios.py --seccionadores 0,2,9,10

# 8. Verificar.
python pipeline.py --estado
```

**Los dos `--escenario-clima` son distintos a propósito** (§4.2): la fragilidad
va sobre ráfaga, la curva de potencia de la turbina sobre sostenido. Es el mismo
`.nc`, columnas distintas.

**`--seccionadores` no es una etiqueta**: entra al optimizador como
`cfg["aislacion_fallas_fija"]` y cambia si el daño se expande a la sección entre
seccionadores (`optimization_base.py:246`). Sin el flag se resuelve el supuesto
optimista —cae sólo la línea fallada— y se guarda en la variante `base`. Los
baselines de una variante **no sirven** para un barrido de la otra.

**Orden de dependencia**, por si hay dudas de qué quedó obsoleto: los `.pkl` y
`.npz` de `artefactos/` tienen que ser posteriores a sus insumos. Basta con
`ls -la artefactos/` y comparar fechas: si `sin_fallas*.pkl` es anterior a
`perfiles.npz` o a `fallas.npz`, E4 está viejo.

**Antes de empezar, archiva lo anterior** en vez de pisarlo — es lo que ya
hiciste con `artefactos/baselines 01-09/` y `artefactos/barrido 01-09/`. Una vez
rehecho E2, el barrido guardado deja de ser comparable y no hay forma de
recuperarlo salvo volviendo a resolver.

**Desde el 2026-09-22 hay una forma mejor: no rehacer nada, crear un caso.** Una
red distinta es un caso distinto (§5.3), con su carpeta de artefactos propia, así
que no hay nada que archivar ni que invalidar: `artefactos/` se queda como está y
el caso nuevo se arma al lado.

```powershell
# Crear el caso desde la pestaña, o a mano:
mkdir artefactos_mi_red

# Todo lo de abajo acepta --artefactos y escribe SOLO en esa carpeta
python pipeline.py --artefactos artefactos_mi_red --fuente red_nueva --etapas E0,E1,E3
python construir_faultlin.py --artefactos artefactos_mi_red --horas 8760 --escenarios 100 --semilla 42
python pipeline.py --artefactos artefactos_mi_red --etapas E2 --forzar
python pipeline.py --artefactos artefactos_mi_red --etapas E4
python pipeline.py --artefactos artefactos_mi_red --etapas E5   # toda la población
python pipeline.py --artefactos artefactos_mi_red --estado

# Para abrir la app EN ese caso. Lo normal es elegirlo en el selector de la
# barra lateral, que reinicia sola; esto es el equivalente a mano.
$env:ERAD_ARTEFACTOS = "$PWD\artefactos_mi_red"; python iniciar_v8.py
```

La receta de arriba (rehacer sobre `artefactos/`) sigue siendo la correcta para
un cambio de **datos** del mismo caso: otros perfiles, otra semilla, otro
supuesto de reparación.

> **Estado al 2026-09-08.** Todo rehecho con la curva de Dunn sobre ráfaga,
> enlaces inmunes y los perfiles de Arauco:
>
> | etapa | qué hay |
> |---|---|
> | E1 | perfiles de Arauco, eólico sostenido. Solar CF ≈ 0,197; eólico 0,222 / 0,219 / 0,103 |
> | E2 | 727 / 484 / 168 / 36 fallas, curva directa |
> | E4 | `sin_fallas_LIM-0.30-ALIM.pkl` |
> | E5 | 100 escenarios en la variante `base` + 10 en `sec-0-2-9-10` |
>
> Lo anterior quedó archivado en `baselines 01-09/`, `barrido 01-09/`,
> `fallas primer.npz` y `FaultlinMC primer*.npy`.
>
> **Un baseline de la variante `base` no sirve para un barrido con
> `--seccionadores`**, y viceversa: `run_barrido_medidas.py` aborta con "Faltan
> baselines". Es el error más fácil de encontrarse acá.

---

## 8. Puntos abiertos

Ninguno rompe el código: todos producen resultados plausibles. Van acá para que
se decidan explícitamente.

### 8.1 Hay dos `SystemzoneV1.xlsx`, y el default es el viejo

**El caso vigente es Arauco**, en
`Proyecto oficial/Inputs/SystemzoneV1_arauco.xlsx`:

| zona | lat | lon | |
|---|---|---|---|
| 1 | −37,645 | −73,658 | Lebu (costera) — la más severa |
| 2 | −37,715 | −73,636 | Arauco costa sur |
| 3 | −37,645 | −73,538 | Lebu interior |

Separación máxima **12 km**, contra los ~47 km de los puntos de Aysén que se
usaban antes.

⚠️ **`construir_perfiles.py` sin `--puntos` toma un `SystemzoneV1.xlsx` viejo**
de `Trend database with hourly resolution/Inputs/`, que tiene las coordenadas de
Aysén (−47°). Pasó el 2026-09-08: el eólico salió bien porque viene del `.nc`
que se pasa con `--viento`, pero el **solar se descargó en el punto equivocado**
(CF 0,15 en vez de 0,20). La pista fue la línea `distancia máxima 47 km` en el
log. **Siempre pasar `--puntos`.**

El archivo antiguo arrastra además historia que conviene no citar por error: la
columna `Comuna` decía Molina / San Clemente / Río Claro (Maule), una
alternativa descartada, y una columna suelta mencionaba Iquique / Valparaíso /
Concepción, que era **la procedencia de los perfiles heredados**.

### 8.2 Los perfiles heredados vs. los puntos actuales

Los Excel de perfiles vienen de esas zonas de referencia y tienen 28 % (solar) y
18 % (eólico) de diferencia en energía diaria entre columnas — un spread propio
de puntos a cientos de km, no de tres a 47 km. Además su ventana de luz
corresponde a latitudes −21° a −37°: se apaga a las 19 h, mientras que en Aysén
(−47°) en verano hay sol útil hasta las 20–21 h.

`construir_perfiles.py` resuelve esto generándolos para los puntos reales
(§3.2), y E1 avisa por consola si detecta la combinación sospechosa (puntos a
menos de 150 km con perfiles que difieren más de 10 %).

### 8.3 `bus_zone` vs. `lin_Z1/Z2/Z3`

Ver §5.1: en el notebook, coordenadas y viento se asignan con repartos de buses
distintos. Verificar cuál corresponde.

### 8.4 ~~E2 con el fallback plano~~ — CERRADO el 2026-09-08

Ver §4.5. Hoy E2 usa las curvas directas (`ref` −77 %). El fallback sigue ahí y
sigue siendo silencioso: si `faultlin_meta.json` no declara
`metodo: curva_directa_numeros_comunes`, E2 vuelve al adelgazamiento sin avisar.
**Comparar siempre los totales de E2 con los que imprimió
`construir_faultlin.py`.**

### 8.7 Lo que quedó abierto el 2026-09-08

1. **`pipeline.py:748` y `:752`** tienen θ = 30 y β = 0,25 hardcodeados en la
   ruta de fallback. Código muerto hoy, pero devolvería un ρ con la curva
   equivocada si alguna vez se corre sin las matrices mitigadas.
2. **`tie_indices = set(LINES[-3:])`** en 13 archivos, entre ellos los dos
   optimizadores vigentes. Hoy coincide con `red.pkl`; si la red cambia, deja de
   coincidir en silencio. Corregidos `grafo.py` y `data_loader.py`.
3. **Sin desconexión de la turbina por ráfaga.** La curva de potencia corta a
   25 m/s de sostenido y en Arauco eso **no ocurre nunca** (máximo 22,3 m/s a
   30 m de buje). Pero hay **276 horas** con ráfaga sobre 25 m/s a esa altura,
   pico 39,4: una turbina real dispararía. Lo único que apaga la eólica en
   tormenta es `a_eol_wind`, que se activa por **daño de línea en la zona**, no
   por viento. El supuesto **favorece a la medida `dg_eol`**.
4. **La estratificación del barrido usó baselines sin aislación** y el barrido
   corre con ella. Se puede cerrar corriendo E5 con `--seccionadores` sobre los
   100 y re-estratificando; el orden relativo de severidad casi no cambia.
5. **Muestreo horario.** Sobre 8760 h sobreestima las fallas frente a un
   muestreo más grueso (§4.7, supuesto 3).

### 8.5 Horas de reparación: 3 en el sorteo, 4 en `repairMC`

Ver §4.4. Ahora son parámetro de entrada y E2 los sincroniza solo, pero lo ya
calculado arrastra los dos valores.

### 8.6 Menores

- El orquestador climático busca nombres de archivo sin guión bajo (§2.6).
- El toggle "estocástico" de la app está roto (`NameError`): el import está
  comentado en `app_v7_carga_corregida.py`. Para el estocástico usar
  `run_estocastico.py`.

---

## 9. Referencia rápida de archivos

**Cadena climática** — `abc/mi_proyecto/app/Proyecto oficial/`

| archivo | rol |
|---|---|
| **`Inputs/SystemzoneV1_arauco.xlsx`** | **entrada manual del caso vigente**: lat/lon por zona. Pasarlo con `--puntos` (§8.1) |
| `Inputs/SystemzoneV1.xlsx` | el de Aysén, histórico. Es el que se toma por defecto — cuidado |
| **`Scripts/construir_ano_operativo.py`** | **vigente**: año operativo horario → `Outputs/BaseHoraEWE_arauco.nc` |
| **`Scripts/generar_anios_alternativos.py`** | años alternativos para sensibilidad. Ver la nota de abajo |
| `Scripts/download_openmeteo_v3.py` | histórico horario con ráfaga, modelo `ecmwf_ifs` fijado |
| `Scripts/Pipeline_extreme_events.py` | orquestador antiguo de los 4 pasos |
| `Scripts/download_openmeteo_historical_ready_v2.py` | paso 1 → `Hourlyhistoricdata.nc` |
| `Scripts/download_arclim_future_to_netcdf_ready_v4.py` | paso 2 → `Dailyfuturedata*.nc` |
| `Scripts/Hourly_climate_zone_humidity_daily_patch.py` | paso 3 → `BaseFhorPELP4v20z2anos.nc` |
| `Scripts/Extreme_events_arclim_final.py` | paso 4 → `BaseHoraEWEPELP20z.nc` |

> **La tabla de severidad de `generar_anios_alternativos.py` ordena por horas
> sobre umbral, y para una curva empinada ese es el criterio equivocado.** El
> año "severo" (2020, 648 h) da **menos** fallas que el base (2024, 363 h),
> porque su ráfaga máxima es 29,2 m/s contra 34,0. Lo que manda es el **pico**,
> no la acumulación: ordenados por fallas esperadas el ranking es 2021, 2024,
> 2025, …, casi el inverso. El año severo se generó y se descartó por esto.

**Modelo** — `abc/mi_proyecto/app/app/`

| archivo | rol |
|---|---|
| `../Proyecto oficial/Outputs/BaseHoraEWE_arauco.nc` | **el viento vigente**; se pasa con `--viento`. `lon=0` sostenido, `lon=1` ráfaga |
| `../Scenario Database/Inputs/BaseHoraEWEPELP20z.nc` | el default histórico de `RUTA_WIND_NC` |
| `construir_perfiles.py` | perfiles de generación → `artefactos/perfiles_horarios.npz`. **Sostenido** |
| `construir_faultlin.py` | Monte Carlo → `FaultlinMC.npy` + `_veg` / `_ref` / `_ug`. **Ráfaga** |
| **`seleccionar_escenarios.py`** | muestra estratificada por ENS → `artefactos/escenarios_estratificados.json` |
| `entrada_red.py` | lectura/validación de la red y de las zonas (lógica pura) |
| `entrada_red_vista.py` | la pestaña 🗺️ *Origen de los datos* |
| `pipeline.py` | E0–E6 → `artefactos/` |
| `data_loader.py` | cargador puro de `artefactos/` para la app |
| `app.pkl` | estudio congelado, **solo lectura**, ~740 MB |
| `artefactos/red_nueva.pkl` | paquete de red exportado desde la pestaña |
| `artefactos/perfiles_meta.json` | fuente y parámetros de los perfiles |
| `artefactos/comparacion_fuentes.csv` | comparación entre `.nc`, Open-Meteo y Explorador |
| `artefactos/faultlin_meta.json` | parámetros de la última corrida del Monte Carlo |
| `artefactos/fallas_meta.json` | cómo se calibró E2 (`fuente_viento`, `metodo_mitigadas`, `horas_reparacion`) |

**Documentos**

| archivo | qué cubre |
|---|---|
| `INSTRUCTIVO_DATOS.md` | este: de cero a los artefactos |
| `GUIA_BARRIDO.md` | barrido medida × escenario y su visualización |
| `ECUACIONES_OBJETIVO_DG.md` | función objetivo y términos de DG |
