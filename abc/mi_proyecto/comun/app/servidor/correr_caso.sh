#!/usr/bin/env bash
# ============================================================
# correr_caso.sh — La cadena completa de un caso, por consola
# ============================================================
# Reemplaza a los botones de "📥 Datos de entrada" y "Optimización y
# análisis" en un servidor sin pantalla. Corre, en orden:
#
#   clima      Open-Meteo + delta ARClim -> BaseHoraEWE_<caso>.nc   [internet]
#   e0         red.pkl desde red_nueva.pkl (lo escribe crear_caso.py)
#   perfiles   perfiles_horarios.npz para los puntos del caso       [internet]
#   e1         perfiles.npz
#   fallas     Monte Carlo de fallas (construir_faultlin.py), o
#              fallas_sinteticas.py si FALLAS=sinteticas (solo pruebas)
#   e2         fallas.npz (matriz base + mitigadas + repairMC)
#   e3         costos.pkl
#   e4         caso sin fallas (4 MILP del horizonte completo)
#   e5         casos base con fallas, toda la población
#   seleccion  muestra estratificada por ENS para el barrido
#   barrido    run_barrido_medidas.py (fases 1, 2 y 3)
#
# Como un `make`, con memoria de parámetros: una etapa se salta si su
# salida existe, es más nueva que sus insumos Y se hizo con el mismo
# comando que se correría ahora. Se rehace sola si cambió un insumo o un
# parámetro (otro HORAS, otra semilla, otro tope de DG). Por eso relanzar
# tras un corte retoma donde quedó, y nunca se mezclan artefactos de dos
# configuraciones distintas. El comando de cada etapa queda en
# artefactos/<caso>/.correr/<etapa>.cmd.
#
# USO
#   servidor/correr_caso.sh <caso.env> [etapas] [--forzar] [--adoptar] [--dry-run]
#
#   --adoptar  da por buenas las salidas que existen sin registro del comando
#              (hechas desde la app o a mano) en vez de rehacerlas. Solo si se
#              sabe que se hicieron con los mismos parámetros.
#
#   servidor/correr_caso.sh casos/mi_red.env                  # todo
#   servidor/correr_caso.sh casos/mi_red.env clima,e0,perfiles,e1,fallas,e2,e3
#   servidor/correr_caso.sh casos/mi_red.env e4,e5,seleccion,barrido
#   servidor/correr_caso.sh casos/mi_red.env fallas --forzar  # rehace fallas; lo que
#                                    depende de ella se rehace solo en la próxima corrida
#   servidor/correr_caso.sh casos/mi_red.env --dry-run        # qué haría, sin correr
#
# Para no depender de la sesión SSH: tmux / nohup (README_SERVIDOR.md) o
# servidor/trabajo_slurm.sbatch.
# ============================================================
set -Eeuo pipefail

ETAPAS_TODAS="clima e0 perfiles e1 fallas e2 e3 e4 e5 seleccion barrido"

die() { echo "[correr_caso] ERROR: $*" >&2; exit 1; }
log() { echo "[correr_caso $(date '+%F %T')] $*"; }

# ── argumentos ───────────────────────────────────────────────
[ $# -ge 1 ] || die "uso: $0 <caso.env> [etapas] [--forzar] [--adoptar] [--dry-run]"
ENV_FILE="$1"; shift
PEDIDAS="" ; FORZAR=0 ; DRY=0 ; ADOPTAR=0
for a in "$@"; do
  case "$a" in
    --forzar)  FORZAR=1 ;;
    --dry-run) DRY=1 ;;
    --adoptar) ADOPTAR=1 ;;
    -*)        die "opción desconocida: $a" ;;
    *)         PEDIDAS="${a//,/ }" ;;
  esac
done
[ -f "$ENV_FILE" ] || die "no existe $ENV_FILE"
ENV_FILE="$(cd "$(dirname "$ENV_FILE")" && pwd)/$(basename "$ENV_FILE")"
for e in $PEDIDAS; do
  [[ " $ETAPAS_TODAS " == *" $e "* ]] || die "etapa desconocida '$e'. Válidas: $ETAPAS_TODAS"
done
[ "$FORZAR" = 1 ] && [ -z "$PEDIDAS" ] && \
  die "--forzar necesita etapas explícitas (ej: 'fallas --forzar'): forzar TODO rehace horas de MILP"

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # código (comun/app)
cd "$APP_DIR"

# ── configuración del caso (defaults; el .env los pisa) ──────
FALLAS="clima"                 # clima | sinteticas
HORAS=""                       # vacío = horizonte del paquete; 48 = prueba rápida
OPENMETEO_DESDE="2017-01-01"; OPENMETEO_HASTA="2025-12-31"
ANIO_PREVIO=2023; ANIO_OPERATIVO=2024; ARCLIM_MIEMBRO=""
CURVA="dunn"; SEMILLA=42
LIM_DG=0.3; LIM_DG_MODO="alimentador"
SECCIONADORES="paquete"        # 'paquete' = los de red_nueva.pkl; 'ninguno'; o lista
N_ESTRATOS=10
BARRIDO_MEDIDAS="reconfig_topo,dg_sol,dg_eol,dg_fos,ess_4h,ess_5h,ess_6h,meg,mess,edrp,refuerzo,soterramiento"
BARRIDO_ESCENARIOS=""          # vacío = la muestra estratificada de 'seleccion'
BARRIDO_FASES="1,2,3"
BARRIDO_SALIDA="principal"
MIP_GAP=0.02; TIME_LIMIT=86400
BARRIDO_EXTRA=""
# Flota de MEG (vacío = default de run_barrido_medidas: 3 x 50 kW, traslado 2 h, sin existentes).
# Van aparte de BARRIDO_EXTRA para que no se pierdan: cambian el resultado de toda combinación con MEG.
MEG_N=""; MEG_P_MAX=""; MEG_EXISTENTES=""; T_TRA_MEG=""
# shellcheck disable=SC1090
source "$ENV_FILE"
[ -n "${CASO:-}" ] || die "$ENV_FILE no define CASO"
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null || die "no encuentro el intérprete '$PYTHON' (PYTHON en $ENV_FILE)"

# Carpeta del caso de estudio (solo datos): ERAD_CASO del .env o del entorno; si
# no, caso_validacion, igual que rutas.py. Se exporta para que la hereden todos.
ERAD_CASO="${ERAD_CASO:-$APP_DIR/../../caso_validacion}"
[ -d "$ERAD_CASO" ] || die "no existe la carpeta del caso ERAD_CASO=$ERAD_CASO"
ERAD_CASO="$(cd "$ERAD_CASO" && pwd)"; export ERAD_CASO
DATOS_DIR="$ERAD_CASO/datos"
CLIMA_DIR="$ERAD_CASO/Proyecto oficial"      # Inputs/ y Outputs/ del clima
CLIMA_SCRIPTS="$APP_DIR/../clima"            # scripts comunes de la cadena climática

ART="$DATOS_DIR/artefactos/$CASO"
PAQUETE="$ART/red_nueva.pkl"
[ -f "$PAQUETE" ] || die "no existe $PAQUETE. Crear el caso primero con crear_caso.py (README_SERVIDOR.md §3)."
SYSTEMZONE="$CLIMA_DIR/Inputs/SystemzoneV1_${CASO}.xlsx"
HIST_NC="$CLIMA_DIR/Inputs/Hourlyhistoricdata_${CASO}.nc"
CLIMA_NC="${CLIMA_NC:-$CLIMA_DIR/Outputs/BaseHoraEWE_${CASO}.nc}"
LOG_DIR="$ART/logs"; SELLOS="$ART/.correr"
mkdir -p "$LOG_DIR" "$SELLOS"

export ERAD_ARTEFACTOS="$ART"          # lo heredan todos los scripts (casos.py)
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
# UTF-8 siempre: con LANG=C (común en servidores y en trabajos de SLURM) o con
# la salida redirigida a un archivo, un '→' en un mensaje de progreso mata el
# proceso con UnicodeEncodeError. También fija la codificación de open().
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
# Gurobi y Pyomo escriben temporales grandes (LP de 8760 h, nodefiles).
export TMPDIR="${TMPDIR:-$ART/tmp}"; mkdir -p "$TMPDIR"
export ERAD_GUROBI_NODEFILE_DIR="${ERAD_GUROBI_NODEFILE_DIR:-$TMPDIR/gurobi_nodefiles}"

# ── datos del caso: UNA fuente de verdad, el paquete ─────────
# numesc, horizonte, horas de reparación y seccionadores salen de
# red_nueva.pkl para que el Monte Carlo, E2, E5 y el barrido no puedan
# desincronizarse por un número escrito dos veces.
# `tr -d '\r'`: un Python de Windows (Git Bash) termina las líneas en CRLF y
# el \r terminaría pegado a la etiqueta de E4 y al nombre de la variante.
py() { "$PYTHON" "$@" | tr -d '\r'; }
read -r NUMESC N_HORAS HREP SEC_PAQ ETIQUETA_E4 < <(py - "$PAQUETE" "$LIM_DG" "$LIM_DG_MODO" <<'PY'
import pickle, sys
import limites_dg
p = pickle.load(open(sys.argv[1], "rb"))
sec = ",".join(str(int(x)) for x in (p.get("seccionadores_existentes") or [])) or "-"
etq = limites_dg.etiqueta_variante({"lim_dg": float(sys.argv[2]), "lim_dg_modo": sys.argv[3]})
print(int(p["numesc"]), int(p.get("n_horas") or 8760), p.get("horas_reparacion") or "-", sec, etq or "-")
PY
)
[ -n "${ETIQUETA_E4:-}" ] || die "no se pudo leer $PAQUETE con $PYTHON (¿falta el entorno? ver instalar.sh)"
[ "$HREP" = "-" ] && HREP=""
[ "$ETIQUETA_E4" = "-" ] && ETIQUETA_E4=""
case "$SECCIONADORES" in
  paquete) SEC="${SEC_PAQ/-/}" ;;
  ninguno) SEC="" ;;
  *)       SEC="$SECCIONADORES" ;;
esac
H_MILP="${HORAS:-$N_HORAS}"
VARIANTE="$(py -c "from aislacion import variante; s=[int(x) for x in '$SEC'.split(',') if x]; print(variante(bool(s), s))")"
BASE_DIR="$ART/baselines"; [ "$VARIANTE" != "base" ] && BASE_DIR="$ART/baselines/$VARIANTE"
SIN_FALLAS="$ART/sin_fallas${ETIQUETA_E4:+_$ETIQUETA_E4}.pkl"
FAULTLIN="$ART/FaultlinMC.npy"

log "carpeta del caso: $ERAD_CASO"
log "caso $CASO · $NUMESC escenarios · horizonte $N_HORAS h (MILP: $H_MILP h) · reparación ${HREP:-de faultlin} h"
log "fallas: $FALLAS · E4: $(basename "$SIN_FALLAS") · aislación: $VARIANTE${SEC:+ (seccionadores $SEC)}"
log "python: $(py -c 'import sys; print(sys.executable)') · TMPDIR: $TMPDIR"
log "$(py recursos.py)"

# ── maquinaria make-like ─────────────────────────────────────
pedida()  { [ -z "$PEDIDAS" ] && return 0; [[ " $PEDIDAS " == *" $1 "* ]]; }
forzada() { [ "$FORZAR" = 1 ] && pedida "$1"; }
saltar()  { log "--- $1: al día, se omite ($2)"; }

# al_dia SALIDA INSUMO... : 0 si SALIDA existe y ningún insumo existente es más nuevo
al_dia() {
  local out="$1"; shift
  [ -e "$out" ] || return 1
  local dep
  for dep in "$@"; do
    if [ -n "$dep" ] && [ -e "$dep" ] && [ "$dep" -nt "$out" ]; then return 1; fi
  done
  return 0
}
baseline_extremo() {  # -t = el baseline más nuevo, -tr = el más viejo; vacío si no hay
  ls "$1" "$BASE_DIR"/esc_*.npz 2>/dev/null | head -1 || true
}
firma() { printf '%q ' "$@"; }

correr() {  # correr ETAPA comando...
  local etapa="$1"; shift
  local logf="$LOG_DIR/$(date '+%Y%m%d-%H%M%S')_${etapa}.log"
  log ">>> $etapa"
  printf '    %s\n' "$(firma "$@")"
  [ "$DRY" = 1 ] && return 0
  local t0=$SECONDS
  echo "# $(date '+%F %T')  $(firma "$@")" > "$logf"
  if ! "$@" 2>&1 | tee -a "$logf"; then
    die "$etapa falló (log: $logf)"
  fi
  log "<<< $etapa listo en $(( (SECONDS - t0) / 60 )) min (log: $logf)"
}

# ejecutar ETAPA SALIDA comando...   (insumos en el arreglo DEPS)
# Corre si: se forzó, falta la salida, un insumo es más nuevo, o no consta
# que la salida se haya hecho con ESTE comando (sello distinto o ausente).
# Sin sello no hay cómo saber con qué horizonte o qué tope se hizo, así que se
# rehace; --adoptar la da por buena cuando se sabe que es correcta.
DEPS=()
ejecutar() {
  local etapa="$1" out="$2"; shift 2
  local f sello="$SELLOS/$etapa.cmd" motivo=""
  f="$(firma "$@")"
  if forzada "${etapa%%-*}"; then motivo="--forzar"
  elif [ ! -e "$out" ]; then motivo="falta $(basename "$out")"
  elif ! al_dia "$out" ${DEPS[@]+"${DEPS[@]}"}; then motivo="un insumo es más nuevo que $(basename "$out")"
  elif [ ! -e "$sello" ] && [ "$ADOPTAR" = 0 ]; then motivo="sin registro de con qué comando se hizo (--adoptar para darla por buena)"
  elif [ -e "$sello" ] && [ "$(cat "$sello")" != "$f" ]; then motivo="cambiaron los parámetros"
  fi
  if [ -z "$motivo" ]; then
    if [ ! -e "$sello" ] && [ "$DRY" = 0 ]; then printf '%s' "$f" > "$sello"; log "$etapa: adoptada"; fi
    saltar "$etapa" "$(basename "$out")"; return 0
  fi
  log "$etapa: se corre ($motivo)"
  correr "$etapa" "$@"
  if [ "$DRY" = 0 ]; then printf '%s' "$f" > "$sello"; fi
}

# ── etapas ───────────────────────────────────────────────────
etapa_clima() {
  if [ "$FALLAS" = "sinteticas" ] && [ ! -e "$CLIMA_NC" ]; then
    saltar clima "FALLAS=sinteticas, sin .nc"; return; fi
  [ -e "$SYSTEMZONE" ] || die "falta $SYSTEMZONE (lo escribe crear_caso.py con --puntos)"
  DEPS=("$SYSTEMZONE")
  ejecutar clima-openmeteo "$HIST_NC" "$PYTHON" "$CLIMA_SCRIPTS/download_openmeteo_v3.py" \
    --coords-file "$(basename "$SYSTEMZONE")" --output-nc "$(basename "$HIST_NC")" \
    --start-date "$OPENMETEO_DESDE" --end-date "$OPENMETEO_HASTA"
  DEPS=("$SYSTEMZONE" "$HIST_NC")
  ejecutar clima-arclim "$CLIMA_NC" "$PYTHON" "$CLIMA_SCRIPTS/construir_ano_operativo.py" \
    --coords-file "$(basename "$SYSTEMZONE")" --entrada-nc "$(basename "$HIST_NC")" \
    --salida-nc "$(basename "$CLIMA_NC")" \
    --anio-previo "$ANIO_PREVIO" --anio-operativo "$ANIO_OPERATIVO" \
    ${ARCLIM_MIEMBRO:+--miembro "$ARCLIM_MIEMBRO"}
}

etapa_e0() {
  DEPS=("$PAQUETE")
  ejecutar e0 "$ART/red.pkl" "$PYTHON" pipeline.py --artefactos "$ART" --fuente red_nueva \
    --etapas E0 --forzar
}

etapa_perfiles() {
  if [ ! -e "$CLIMA_NC" ] && [ "$FALLAS" = "sinteticas" ]; then
    saltar perfiles "sin .nc: E1 usará los perfiles de 24 h"; return; fi
  [ -e "$CLIMA_NC" ] || [ "$DRY" = 1 ] || die "perfiles necesita $CLIMA_NC (etapa clima)"
  # SOSTENIDO (baseline): la curva de potencia de la turbina va sobre viento
  # sostenido; la fragilidad, sobre ráfaga (INSTRUCTIVO_DATOS.md §4.2).
  DEPS=("$CLIMA_NC" "$PAQUETE")
  ejecutar perfiles "$ART/perfiles_horarios.npz" "$PYTHON" construir_perfiles.py --forzar \
    --viento "$CLIMA_NC" --escenario-clima baseline --anio-offset 8760 \
    --fuente-solar openmeteo --puntos "$SYSTEMZONE"
}

etapa_e1() {
  DEPS=("$ART/red.pkl" "$ART/perfiles_horarios.npz")
  ejecutar e1 "$ART/perfiles.npz" "$PYTHON" pipeline.py --artefactos "$ART" --fuente red_nueva \
    --etapas E1 --forzar
}

etapa_fallas() {
  DEPS=("$ART/red.pkl")
  if [ "$FALLAS" = "sinteticas" ]; then
    log "[!] FALLAS=sinteticas: tormentas inventadas, NO sirven para la tesis"
    ejecutar fallas "$ART/fallas.npz" "$PYTHON" fallas_sinteticas.py --artefactos "$ART" --forzar \
      --horas "$N_HORAS" --escenarios "$NUMESC" ${HREP:+--horas-reparacion "$HREP"}
    return
  fi
  # RÁFAGA (with_EWE), al revés que perfiles.
  DEPS=("$ART/red.pkl" "$CLIMA_NC")
  ejecutar fallas "$FAULTLIN" "$PYTHON" construir_faultlin.py --artefactos "$ART" --forzar \
    --viento "$CLIMA_NC" --escenario-clima with_EWE --curva "$CURVA" --anio-offset 8760 \
    --horas "$N_HORAS" --escenarios "$NUMESC" --semilla "$SEMILLA" --enlaces-inmunes \
    ${HREP:+--horas-espera "$HREP"}
}

etapa_e2() {
  if [ "$FALLAS" = "sinteticas" ]; then saltar e2 "fallas_sinteticas.py ya escribió fallas.npz"; return; fi
  DEPS=("$FAULTLIN" "${FAULTLIN%.npy}_veg.npy" "${FAULTLIN%.npy}_ref.npy" "${FAULTLIN%.npy}_ug.npy")
  ejecutar e2 "$ART/fallas.npz" "$PYTHON" pipeline.py --artefactos "$ART" --etapas E2 --forzar \
    ${HREP:+--horas-reparacion "$HREP"}
}

etapa_e3() {
  DEPS=("$ART/red.pkl")
  ejecutar e3 "$ART/costos.pkl" "$PYTHON" pipeline.py --artefactos "$ART" --etapas E3 --forzar
}

INSUMOS_MILP=("$ART/red.pkl" "$ART/perfiles.npz" "$ART/fallas.npz" "$ART/costos.pkl")

etapa_e4() {
  DEPS=("${INSUMOS_MILP[@]}")
  ejecutar e4 "$SIN_FALLAS" "$PYTHON" pipeline.py --artefactos "$ART" --fuente red_nueva \
    --etapas E4 --forzar --lim-dg "$LIM_DG" --lim-dg-modo "$LIM_DG_MODO" \
    ${ETIQUETA_E4:+--etiqueta "$ETIQUETA_E4"} --horas "$H_MILP"
}

etapa_e5() {
  # E5 se saltea solo los escenarios ya hechos, así que se corre siempre. Lo
  # que se decide acá es si los hechos siguen valiendo: si algún insumo es
  # más nuevo que el baseline más viejo, o cambió el comando (otro horizonte,
  # otros seccionadores), se rehacen todos con --forzar.
  local cmd=("$PYTHON" pipeline.py --artefactos "$ART" --fuente red_nueva --etapas E5
             --escenarios "0-$((NUMESC - 1))" ${SEC:+--seccionadores "$SEC"} --horas "$H_MILP")
  local f sello="$SELLOS/e5-$VARIANTE.cmd" viejo extra=()
  f="$(firma "${cmd[@]}")"
  viejo="$(baseline_extremo -tr)"
  if forzada e5; then extra=(--forzar)
  elif [ -n "$viejo" ] && ! al_dia "$viejo" "${INSUMOS_MILP[@]}"; then
    log "[!] baselines de '$VARIANTE' anteriores a sus insumos: se rehacen"; extra=(--forzar)
  elif [ -n "$viejo" ] && [ -e "$sello" ] && [ "$(cat "$sello")" != "$f" ]; then
    log "[!] baselines de '$VARIANTE' hechos con otros parámetros: se rehacen"; extra=(--forzar)
  elif [ -n "$viejo" ] && [ ! -e "$sello" ] && [ "$ADOPTAR" = 0 ]; then
    log "[!] baselines de '$VARIANTE' sin registro del comando: se rehacen (--adoptar para conservarlos)"
    extra=(--forzar)
  fi
  correr e5 "${cmd[@]}" ${extra[@]+"${extra[@]}"}
  if [ "$DRY" = 0 ]; then printf '%s' "$f" > "$sello"; fi
}

etapa_seleccion() {
  if [ -n "$BARRIDO_ESCENARIOS" ]; then saltar seleccion "BARRIDO_ESCENARIOS=$BARRIDO_ESCENARIOS"; return; fi
  local n=$N_ESTRATOS
  if [ "$n" -gt "$NUMESC" ]; then n=$NUMESC; fi
  DEPS=("$(baseline_extremo -t)")
  ejecutar seleccion "$ART/escenarios_estratificados.json" "$PYTHON" seleccionar_escenarios.py \
    --n-estratos "$n" ${SEC:+--seccionadores "$SEC"} ${HORAS:+--horas "$HORAS"}
}

etapa_barrido() {
  local escs="$BARRIDO_ESCENARIOS"
  if [ -z "$escs" ]; then
    if [ -e "$ART/escenarios_estratificados.json" ]; then
      escs="$(py -c "import json,sys; print(','.join(str(e) for e in json.load(open(sys.argv[1]))['escenarios']))" "$ART/escenarios_estratificados.json")"
    elif [ "$DRY" = 1 ]; then
      escs="<muestra>"
    else
      die "barrido sin escenarios: correr 'seleccion' o fijar BARRIDO_ESCENARIOS"
    fi
  fi
  # El barrido es reanudable y se SALTEA las corridas ya guardadas. Si E4, E5
  # o las fallas cambiaron después, esas corridas son de otros insumos y
  # mezclarlas daría una tabla incoherente sin ningún error. (Los cambios de
  # parámetros los ataja el propio run_barrido_medidas: verificar_supuestos.)
  local marca="$ART/barrido/$BARRIDO_SALIDA/resumen_individual.csv"
  if [ -e "$marca" ] && ! forzada barrido && \
     ! al_dia "$marca" "$SIN_FALLAS" "$ART/fallas.npz" "$(baseline_extremo -t)"; then
    die "el barrido '$BARRIDO_SALIDA' es anterior a sus insumos (E4/E5/fallas). Usar otra BARRIDO_SALIDA o 'barrido --forzar'."
  fi
  local extra=()
  if forzada barrido; then extra=(--forzar); fi
  [ -n "$MEG_N" ] && extra+=(--n_meg "$MEG_N")
  [ -n "$MEG_P_MAX" ] && extra+=(--p_meg_max "$MEG_P_MAX")
  [ -n "$MEG_EXISTENTES" ] && extra+=(--meg_existentes "$MEG_EXISTENTES")
  [ -n "$T_TRA_MEG" ] && extra+=(--t_tra_meg "$T_TRA_MEG")
  # shellcheck disable=SC2086
  correr barrido "$PYTHON" run_barrido_medidas.py --artefactos "$ART" \
    --medidas "$BARRIDO_MEDIDAS" --escenarios "$escs" --fases "$BARRIDO_FASES" \
    --horas "$H_MILP" --salida "$BARRIDO_SALIDA" \
    --lim-dg "$LIM_DG" --lim-dg-modo "$LIM_DG_MODO" ${ETIQUETA_E4:+--sin-fallas "$ETIQUETA_E4"} \
    ${SEC:+--seccionadores "$SEC"} --mip_gap "$MIP_GAP" --time_limit "$TIME_LIMIT" \
    ${extra[@]+"${extra[@]}"} $BARRIDO_EXTRA
}

# ── orden ────────────────────────────────────────────────────
T0=$SECONDS
for e in $ETAPAS_TODAS; do
  if pedida "$e"; then "etapa_$e"; fi
done
log "fin. $(( (SECONDS - T0) / 60 )) min en total."
if [ "$DRY" = 0 ]; then "$PYTHON" pipeline.py --artefactos "$ART" --estado; fi
