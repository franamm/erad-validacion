# Correr la cadena en un servidor Linux (sin interfaz)

Cómo llevar una **red nueva** desde el archivo de la red hasta el barrido de
medidas completo, todo por consola. Compañero de `INSTRUCTIVO_DATOS.md` (qué es
cada dato) y `GUIA_BARRIDO.md` (qué hace el barrido).

La app de Streamlit sigue sirviendo para **mirar** los resultados: se copian de
vuelta al computador y se abren ahí (§7). Lo que cambia es que ya nada de la
preparación exige la pestaña.

```
crear_caso.py ──► red_nueva.pkl + SystemzoneV1_<caso>.xlsx      (minutos, a mano)
                          │
servidor/correr_caso.sh   ▼
  clima → e0 → perfiles → e1 → fallas → e2 → e3                 (~minutos, internet)
  → e4 → e5 → seleccion → barrido                               (horas a días, MILP)
```

---

## 1. Qué hay en `servidor/`

| archivo | para qué |
|---|---|
| `../crear_caso.py` | el asistente de "🗺️ Origen de los datos", por consola |
| `correr_caso.sh` | toda la cadena, reanudable, con un log por etapa |
| `casos/plantilla.env` | la configuración de un caso, comentada |
| `casos/prueba_cigre.env` | prueba de humo: la cadena completa en ~5 min, sin internet |
| `instalar.sh` | entorno de Python + verificación |
| `requirements_linux.txt` | versiones fijas (las del `erad-env` de Windows) |
| `trabajo_slurm.sbatch` | plantilla, si el servidor usa SLURM |
| `../verificar_entorno.py` | ¿está todo listo? licencia, solver, RAM, disco, red |
| `../recursos.py` | núcleos y RAM **asignados** al proceso (cgroup / SLURM) |

---

## 2. Instalación (una vez)

```bash
# Clonado superficial: la historia del repo pesa 6,4 GB y no hace falta
git clone --depth 1 https://github.com/franamm/erad-tesis.git
cd erad-tesis

bash abc/mi_proyecto/comun/app/servidor/instalar.sh    # crea erad-env/ y verifica
```

**Licencia de Gurobi.** Es WLS y no va en git. Desde Windows:

```powershell
scp C:\Users\<usuario>\gurobi.lic <usuario>@<servidor>:~/gurobi.lic
```

El servidor necesita salida HTTPS a `token.gurobi.com`. Antes de correr en dos
máquinas a la vez, revisa en license.gurobi.com cuántas sesiones simultáneas
admite la licencia.

`instalar.sh` termina con `verificar_entorno.py --red`. Tiene que decir
**"Listo para correr"**. Las fallas que puede reportar, y qué hacer con cada una:

| mensaje | qué hacer |
|---|---|
| licencia de prueba (2000 variables) | falta `~/gurobi.lic` |
| `Gurobi WLS: sin acceso` | el servidor no sale a internet: la licencia no se renueva |
| `hay un Gurobi de sistema en el PATH` | E4/E5 usarían ESE Gurobi (otra versión, otra licencia). Sacarlo del `PATH` |
| `Open-Meteo / ARClim: sin acceso` | correr `clima` y `perfiles` en otra máquina y copiar los `.nc` (§4) |
| `< 50 GB libres` en TMPDIR | apuntar `TMPDIR` a un disco grande (§6) |

**Prueba de humo**, antes de lanzar nada largo:

```bash
cd abc/mi_proyecto/comun/app
erad-env/bin/python crear_caso.py prueba_cigre --red cigre --n-zonas 3 --sin-puntos \
    --criticos 6,11,14 --seccionadores ninguno --escenarios 4 --horas-reparacion 3
servidor/correr_caso.sh servidor/casos/prueba_cigre.env
```

(`erad-env` está en la raíz del repo: desde `comun/app/` es `../../../../erad-env/bin/python`.)

**Código y casos.** El código vive una sola vez en `abc/mi_proyecto/comun/app/`
(y los scripts de clima en `comun/clima/`). Cada caso de estudio es una carpeta
de **solo datos** al lado: `caso_validacion/`, `caso_elmelon/`, … con `datos/`
(donde quedan `artefactos/<CASO>/`) y `Proyecto oficial/{Inputs,Outputs}`. El
caso se elige con la variable `ERAD_CASO`. Sin ella, los scripts lanzados desde
`comun/app/` usan `caso_validacion/` (es lo que hace la prueba de humo).

```bash
export ERAD_CASO="$PWD/../../caso_elmelon"     # para crear_caso.py y verificar_entorno.py
# y en el .env del caso, la misma ruta: ERAD_CASO="$APP_DIR/../../caso_elmelon"
```

---

## 3. Crear el caso de la red nueva

Esto lo decides **tú**: son datos de la red y supuestos del estudio, no cosas que
el código pueda inferir.

| dato | archivo / flag | formato |
|---|---|---|
| la red | `--red` | pandapower `.json/.p/.xlsx`, OpenDSS `Master.dss`, o CSV (`--csv-buses/--csv-lineas/--csv-cargas`, plantillas en `entrada_red.PLANTILLA_CSV`) |
| zona climática de cada bus | `--zonas zonas.csv` | columnas `bus,zona` (o `name,zona`). Zonas 1..Z, sin huecos |
| un punto por zona | `--puntos puntos.csv` | columnas `Zona,Lat,Lon` (opcional `Comuna`). Lat/lon reales de Chile |
| buses críticos | `--criticos 5,17` o `ninguno` | obligatorio: no hay default que valga en otra red |
| seccionadores | `--seccionadores 0,4` o `ninguno` | obligatorio. Un switch cerrado no es necesariamente un seccionador (§5.3 del instructivo) |
| tie-lines | `--ties auto` (switches abiertos) o lista | `auto` es el default |
| escenarios / horizonte | `--escenarios 100 --horas 8760` | |
| horas de reparación | `--horas-reparacion 3` | entra directo en SAIDI y ENS |
| perfil de demanda | `--perfil-demanda demanda.csv` | columna `Power` = factor p.u. sobre `P_D_base`, ≥ 8760 filas |
| capacidad de cabecera | `--sn-mva-trafo 25` | solo si la red no trae transformador |

Los números de bus y de línea van **en la numeración de tu archivo**. El script
adapta la red a lo que el MILP exige: buses contiguos, cabeceras en 1 y 12,
índices desde 0. Además traduce y deja la tabla de equivalencias en
`artefactos/<caso>/crear_caso.json`. Úsala para cruzar resultados con la red
original: el bus 39 de tu archivo puede ser el bus 1 del modelo.

```bash
# 1) Mirar primero, sin escribir nada
erad-env/bin/python crear_caso.py mi_red --red mi_red.json --zonas zonas.csv \
    --puntos puntos.csv --criticos 5,17 --seccionadores ninguno --revisar

# 2) Crear
erad-env/bin/python crear_caso.py mi_red --red mi_red.json --zonas zonas.csv \
    --puntos puntos.csv --criticos 5,17 --seccionadores ninguno \
    --escenarios 100 --horas-reparacion 3 --perfil-demanda demanda_mi_red.csv
```

`--revisar` muestra la validación y el resumen por zona (buses, líneas, kW),
además de los avisos. Un `ERROR` significa que el MILP no correría o daría un
resultado falso: por ejemplo, demanda mayor que la capacidad de las cabeceras o
puntos fuera de Chile. Cada error dice qué corregir.

Escribe, dentro del caso `$ERAD_CASO`, `datos/artefactos/mi_red/red_nueva.pkl`,
`crear_caso.json` y `Proyecto oficial/Inputs/SystemzoneV1_mi_red.xlsx`.

---

## 4. Correr

```bash
cp servidor/casos/plantilla.env servidor/casos/mi_red.env    # y ajustar CASO=mi_red y ERAD_CASO
servidor/correr_caso.sh servidor/casos/mi_red.env --dry-run  # qué va a hacer
servidor/correr_caso.sh servidor/casos/mi_red.env            # todo
```

Cada etapa se salta si ya está hecha **con el mismo comando** y con insumos que
no han cambiado. Si cambias un parámetro del `.env` (horizonte, semilla, tope de
DG), solo se rehacen las etapas afectadas. Si algo se corta, relanzar el mismo
comando retoma desde ahí: el barrido, además, guarda cada corrida apenas termina.

| quiero… | comando |
|---|---|
| solo la preparación (rápida, con internet) | `correr_caso.sh mi_red.env clima,e0,perfiles,e1,fallas,e2,e3` |
| solo los MILP | `correr_caso.sh mi_red.env e4,e5,seleccion,barrido` |
| rehacer una etapa | `correr_caso.sh mi_red.env fallas --forzar` |
| otra tanda de barrido | cambiar `BARRIDO_SALIDA` y relanzar `barrido` |
| conservar algo hecho desde la app | `--adoptar` (sin él, lo que no tiene registro de su comando se rehace) |

Logs: `artefactos/<caso>/logs/<fecha>_<etapa>.log`. El comando con que se hizo
cada etapa queda en `artefactos/<caso>/.correr/`.

**Sin internet en el servidor:** corre `clima` en Windows, con el mismo
`SystemzoneV1_<caso>.xlsx`, y copia `<carpeta del caso>/Proyecto oficial/Outputs/BaseHoraEWE_<caso>.nc`
al servidor. Si lo dejas en otra ruta, apúntalo con `CLIMA_NC=` en el `.env`.
La etapa `perfiles` también descarga datos (radiación desde Open-Meteo).

**Que no muera al cerrar la sesión SSH:**

```bash
tmux new -s erad                      # ... correr_caso.sh ...   Ctrl-b d para salir
tmux attach -t erad                   # volver
# o bien
nohup servidor/correr_caso.sh servidor/casos/mi_red.env > corrida.log 2>&1 &
```

Con SLURM: `servidor/trabajo_slurm.sbatch` (instrucciones adentro).

---

## 5. Recursos: lo que enseñó Oberrhein

MV Oberrhein (179 barras, 8760 h) en una máquina de 28 núcleos y 64 GB:

| corrida | resultado |
|---|---|
| E4 (4 MILP sin fallas) | 4,5 h |
| E5, un escenario | ~6 min |
| combinación de las **12 medidas**, un escenario | **Out of memory** a las 4,8 h, sin incumbente |
| combinación de 7 medidas | resolvió |

O sea, la combinación de todas las medidas es el cuello de botella, y es de
**memoria** más que de tiempo. Para una red de ese tamaño:

- **RAM: 192 GB o más** para la fase 3 con las 12 medidas. Los optimizadores
  dan a Gurobi el 60 % de la RAM **asignada** como `SoftMemLimit`. Así, al
  llegar al límite Gurobi termina ordenadamente y devuelve el mejor incumbente
  con su gap, en vez de que el kernel lo mate.
- Si igual no alcanza: `BARRIDO_EXTRA="... --method 1 --threads 8"`. El simplex
  dual pide mucha menos memoria que barrier en la raíz, y cada hilo guarda su
  propia copia de la relajación.
- `TIME_LIMIT=86400`: un corte por tiempo **no** pierde la corrida. Queda el
  incumbente, y su gap real viaja en `solver_gap` del resumen.
- Fases por separado: primero `BARRIDO_FASES=1,2` (individuales, son muchas pero
  chicas) y después `BARRIDO_FASES=3` (la combinación, que es la grande).

`recursos.py` (y la línea `[RECURSOS]` del log) muestra qué núcleos y qué RAM ve
el proceso. Lee el affinity, el cgroup y las variables de SLURM, porque en un
servidor compartido `os.cpu_count()` devuelve los núcleos de **todo** el nodo.
Para fijarlos a mano: `export ERAD_CPUS=16 ERAD_RAM_GB=120`.

---

## 6. Disco

Gurobi y Pyomo escriben temporales grandes. E4/E5 pasan el modelo por un archivo
LP de 8760 h, y el árbol de branch & bound se vuelca a disco a partir de
`NodefileStart`. `correr_caso.sh` los manda a `artefactos/<caso>/tmp/`. En un
servidor con `/scratch` conviene que vayan ahí:

```bash
export TMPDIR=/scratch/$USER/erad_tmp      # en el .env o antes de correr
```

---

## 7. Traer los resultados a Windows

Los `.pkl`, `.npz` y `.pkl.gz` **no van en git** (pesados, ignorados). Cópialos
con rsync o scp:

```bash
# desde Windows (Git Bash / WSL), en la raíz del repo
rsync -av --exclude tmp/ <usuario>@<servidor>:erad-tesis/abc/mi_proyecto/caso_elmelon/datos/artefactos/mi_red/ \
      abc/mi_proyecto/caso_elmelon/datos/artefactos/mi_red/
```

Después, en la app, elige `mi_red` en el selector de la barra lateral. Las
corridas del barrido se abren en la pestaña Resultados solo si se guardaron
completas: es el default; `--no-completo` lo desactiva.

---

## 8. Qué se corrigió para que esto funcione fuera de la red de 15 barras

Ninguna de estas correcciones cambia el caso de validación. Todas fallaban **en
silencio** en una red nueva:

| dónde | qué pasaba |
|---|---|
| `pipeline.py` E0, `optimization_6_oficial`, `optimization_base`, `data_loader` | una red **sin** tie-lines (`[]`) se trataba como "no declaradas", y `LINES[-3:]` dejaba abiertas tres líneas cualesquiera: carga desconectada sin ninguna falla |
| `helpers_inversion._extraer_costos_objetivo` | el OPEX de maniobra de enlaces se sumaba siempre sobre `LINES[-3:]`. En Oberrhein los enlaces son otros, así que ese término de `cst_*` salía 0 |
| `pipeline.py` E0 | `buses_criticos: []` se convertía en `[6, 11, 14]`, que además quedan fuera de EDRP |
| `data_loader` + `run_barrido_medidas` | el desglose de costos exigía el `sin_fallas.pkl` sin etiqueta. En un caso que solo tiene el E4 etiquetado se perdía entero; en Oberrhein se calculaba contra **otro** E4 que el del objetivo |
| `entrada_red.adaptar_para_milp` | `net.deepcopy()` no existe en pandapower 3.x: el botón "Adaptar la red" reventaba. Además, con dos trafos colgando del mismo bus de alta (CIGRE), intercambiaba los buses 0 y 2 |
| `optimization_6_oficial` / `optimization_base` | Threads y RAM se calculaban con el total del nodo, no con lo asignado |

Pendiente, y conviene saberlo: el docstring de `fallas_sinteticas.py` dice que
la primera tormenta cae en la hora 6, pero en la prueba la primera falla llegó
en la hora ~430. Por eso `prueba_cigre.env` usa 720 h. No afecta a la cadena
real, porque las fallas sintéticas no sirven para la tesis.
