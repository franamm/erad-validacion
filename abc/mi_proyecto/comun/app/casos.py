# ============================================================
# casos.py — Un "caso de estudio" = una carpeta de artefactos
# ============================================================
# El estudio nació con una sola red y una sola carpeta (`artefactos/`), así
# que la ruta quedó escrita como constante en ocho módulos distintos. Al
# querer trabajar una segunda red (MV Oberrhein, 179 barras) eso se volvió
# el obstáculo: la única forma de cambiar de red era parchear
# `pipeline.ART_DIR` desde un notebook, y ni eso alcanzaba (ver
# `pipeline.usar_artefactos`).
#
# Este módulo es la respuesta, y tiene dos reglas que lo mantienen barato:
#
#   1. NO importa `pipeline` ni `streamlit`. Lo importan tanto la vista
#      como los scripts de consola, así que un ciclo de imports o una
#      dependencia de Streamlit lo volverían inusable desde consola.
#   2. `estado_caso` NUNCA muta nada. Mirar el estado de un caso distinto
#      del activo tiene que ser una lectura de disco y nada más. Si la
#      vista llamara a `pipeline.usar_artefactos` para "mirar" otro caso,
#      cambiaría la carpeta para `barrido_vista`, `estudio_v8` y todo lo
#      que ya importó `pipeline` en el mismo proceso.
#
# La carpeta activa del proceso se fija con la variable de entorno
# ERAD_ARTEFACTOS; sin ella, es `artefactos/` y nada cambia respecto de
# como funcionó siempre.
# ============================================================

import json
import os
import re
from pathlib import Path

import numpy as np

from rutas import CASO_DIR, DATOS_DIR
# Raíz de los DATOS del caso activo (artefactos/, FaultlinMC.npy, caso_activo.txt),
# que ya no viven junto al código: ver rutas.py. Conserva el nombre APP_DIR
# porque los tests lo parchean.
APP_DIR = DATOS_DIR
VAR_ENTORNO = "ERAD_ARTEFACTOS"
NOMBRE_DEFECTO = "artefactos"
PREFIJO = "artefactos"
RUTA_PUNTERO = APP_DIR / "caso_activo.txt"

# Artefactos que produce cada etapa del pipeline, en orden de dependencia.
# Es la misma lista que recorre `pipeline.mostrar_estado` y la que usa
# `data_loader._cargar_artefactos` para exigirlos.
ARTEFACTOS_ETAPA = (
    ("E0", "red.pkl", "red y topología"),
    ("E1", "perfiles.npz", "perfiles de demanda y generación"),
    ("E2", "fallas.npz", "matriz de fallas y reparaciones"),
    ("E3", "costos.pkl", "parámetros económicos"),
    ("E4", "sin_fallas.pkl", "caso sin fallas (MILP)"),
)


def art_dir_validacion():
    """Carpeta del estudio congelado de la tesis (caso de validación).

    Es la que protege `es_validacion`: nada de lo que ofrezca la pestaña debe
    poder pisarla. Un caso nuevo NO es esto, aunque sea el que abre la app.
    """
    raiz = APP_DIR / NOMBRE_DEFECTO
    return raiz / "validacion" if (raiz / "validacion").is_dir() else raiz


def art_dir_defecto():
    """Carpeta de artefactos con la que abre el caso cuando nadie dice otra.

    Cada caso (`caso_x/`) guarda lo suyo en `datos/artefactos/x/`: la de su
    propio nombre. Sin esto, un caso distinto de la validación abría
    `datos/artefactos/` a secas (la carpeta contenedora, sin red ni E4) y la
    app decía que no había nada, aunque `pipeline.py --artefactos x` sí lo
    encontraba. En el caso de validación el resultado es el de siempre
    (`artefactos/validacion`).
    """
    raiz = APP_DIR / NOMBRE_DEFECTO
    propio = raiz / CASO_DIR.name.removeprefix("caso_")
    if propio.is_dir():
        return propio
    return art_dir_validacion()


def resolver_ruta(ruta):
    """Acepta también los nombres anteriores a la organización por casos."""
    ruta = Path(ruta)
    if not ruta.is_absolute():
        ruta = APP_DIR / ruta
    ruta = ruta.resolve()
    if ruta == APP_DIR / NOMBRE_DEFECTO:
        return art_dir_defecto()
    if ruta.parent == APP_DIR and ruta.name.startswith("artefactos_"):
        nueva = APP_DIR / NOMBRE_DEFECTO / ruta.name.removeprefix("artefactos_")
        if nueva.is_dir():
            return nueva
    if ruta.parent == APP_DIR and not ruta.exists():
        anidada = APP_DIR / NOMBRE_DEFECTO / ruta.name
        if anidada.is_dir():
            return anidada
    return ruta


def art_dir_activo():
    """Carpeta de artefactos con la que arrancó este proceso.

    No es necesariamente el caso que la usuaria está preparando en la
    pestaña: es el que cargó `data_loader` al importarse, y por lo tanto el
    que muestran las demás pestañas. La diferencia entre ambos es
    justamente lo que la vista tiene que avisar.

    Orden de precedencia, del más explícito al más cómodo:

      1. `$ERAD_ARTEFACTOS` — lo que pasa un comando o un job concreto.
      2. `caso_activo.txt` — lo que eligió el selector de la app.
      3. `artefactos/` — el caso de validación, como funcionó siempre.

    La variable de entorno gana porque es la que usan `pipeline --artefactos`
    y los subprocesos que lanza la app: un job apuntado a un caso no puede
    quedar desviado porque alguien tocó el selector mientras corría.
    """
    ruta = os.environ.get(VAR_ENTORNO) or caso_apuntado()
    return resolver_ruta(ruta) if ruta else art_dir_defecto()


def caso_apuntado():
    """Ruta escrita en `caso_activo.txt`, o None si no hay o no es válida.

    Un puntero a una carpeta que ya no existe se ignora en vez de romper el
    arranque: si se borró el caso, la app tiene que abrir igual (en el de
    validación) y decirlo desde la barra lateral.
    """
    try:
        txt = RUTA_PUNTERO.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not txt:
        return None
    ruta = resolver_ruta(txt)
    return str(ruta.resolve()) if ruta.is_dir() else None


def fijar_caso(art_dir):
    """Deja apuntado el caso para el próximo arranque. No afecta a este proceso.

    No toca `os.environ`: cambiar la variable acá desviaría los subprocesos
    que la app lance después, que deben seguir apuntando al caso con el que
    la usuaria está trabajando en esta sesión, no al que eligió para la
    siguiente.
    """
    art_dir = Path(art_dir).resolve()
    if not art_dir.is_dir():
        raise ValueError(f"No existe la carpeta del caso: {art_dir}")
    RUTA_PUNTERO.write_text(str(art_dir), encoding="utf-8")
    return art_dir


def etiqueta_e4(art_dir):
    """Etiqueta del E4 (caso sin fallas) con el que abre la app este caso.

    1. `$ERAD_SIN_FALLAS`, si está: lo fija el barrido para que el desglose de
       costos use el MISMO E4 que el objetivo del MILP.
    2. El histórico (`sin_fallas.pkl`, etiqueta ""), si existe: así funcionó
       siempre el caso de validación.
    3. Si no, el primer E4 etiquetado (`sin_fallas_<etiqueta>.pkl`). Una red
       nueva solo tiene el etiquetado, porque su tope de DG (`lim_dg`) se
       registra en la etiqueta; exigirle el histórico dejaba la app diciendo
       "no hay E4" con el E4 calculado.
    Devuelve "" si no hay ninguno (entonces el histórico es el que falta).
    """
    env = os.environ.get("ERAD_SIN_FALLAS")
    if env:
        return env
    art_dir = Path(art_dir)
    if (art_dir / "sin_fallas.pkl").exists():
        return ""
    etiquetas = sorted(p.stem[len("sin_fallas_"):] for p in art_dir.glob("sin_fallas_*.pkl"))
    return etiquetas[0] if etiquetas else ""


def nombre_e4(art_dir):
    etiqueta = etiqueta_e4(art_dir)
    return f"sin_fallas_{etiqueta}.pkl" if etiqueta else "sin_fallas.pkl"


def listo_para_ver(art_dir):
    """(bool, faltantes). ¿Puede `data_loader` cargar este caso?

    Es exactamente la lista que exige `data_loader._cargar_artefactos`, que
    falla en el import y por lo tanto tumba toda la app, no solo una
    pestaña. De ahí que el selector la consulte ANTES de dejar cambiar de
    caso: un caso a medio armar se prepara sin problema, pero no se puede
    abrir la app en él.
    """
    faltan = [f"{etapa} ({nombre})" for etapa, nombre, _ in ARTEFACTOS_ETAPA
              if not (Path(art_dir) / (nombre_e4(art_dir) if etapa == "E4" else nombre)).exists()]
    return not faltan, faltan


def es_validacion(art_dir):
    """True si es la carpeta del estudio congelado de la tesis.

    Sirve para proteger ese caso: nada de lo que ofrezca la pestaña debería
    poder pisarlo sin que se note.
    """
    return Path(art_dir).resolve() == art_dir_validacion().resolve()


def ruta_faultlin(art_dir):
    """Matriz de fallas del caso.

    El caso de validación la tiene en `app/FaultlinMC.npy`, fuera de
    `artefactos/`, por razones históricas (`pipeline.py` la trata como
    "fuente de verdad" separada de los artefactos regenerables). Ese
    archivo NO se mueve: cualquier otro caso guarda la suya dentro de su
    propia carpeta, que además es lo correcto, porque una matriz de fallas
    solo sirve para la red con la que se sorteó.
    """
    art_dir = Path(art_dir)
    if es_validacion(art_dir):
        return APP_DIR / "FaultlinMC.npy"
    return art_dir / "FaultlinMC.npy"


def listar_casos():
    """Carpetas de casos existentes, con el de validación primero.

    Sin caché, a propósito y por la misma razón que documenta
    `barrido_vista.listar_barridos`: la usuaria crea carpetas mientras la
    app está abierta, y una lista cacheada la dejaría sin ver lo que acaba
    de crear.
    """
    raiz = APP_DIR / NOMBRE_DEFECTO
    encontradas = [d for d in sorted(raiz.iterdir())
                   if d.is_dir() and not d.name.startswith("_")
                   and ((d / "red.pkl").exists() or (d / ".caso").exists())] if raiz.exists() else []
    encontradas += [d for d in sorted(APP_DIR.glob("artefactos_*")) if d.is_dir()]
    defecto = art_dir_defecto()
    if defecto.exists():
        if defecto in encontradas:
            encontradas.remove(defecto)
        encontradas.insert(0, defecto)
    return encontradas


def nombre_valido(txt):
    """Valida un nombre simple dentro de artefactos/<caso>."""
    txt = (txt or "").strip().removeprefix("artefactos_")
    if not txt:
        return None, "Escribe un nombre para el caso."
    if not re.fullmatch(r"[\w .+-]+", txt) or txt in (".", ".."):
        return None, ("Solo letras, números, espacios y . + - _ "
                      "(nada de / o \\: es una carpeta dentro de artefactos/).")
    ruta = APP_DIR / NOMBRE_DEFECTO / txt
    if ruta.exists():
        return ruta, f"'{txt}' ya existe."
    return ruta, None


def _mb(ruta):
    try:
        return ruta.stat().st_size / 1e6
    except OSError:
        return 0.0


def _json_o_none(ruta):
    if not ruta.exists():
        return None
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:
        return None


def _forma_npy(ruta):
    """(forma, total de fallas) de una matriz de fallas, sin cargarla.

    `mmap_mode="r"` lee solo la cabecera para la forma; la suma sí recorre
    el archivo, pero son int8 y el histórico pesa 13 MB.
    """
    if not ruta.exists():
        return None, None
    try:
        a = np.load(ruta, mmap_mode="r")
        return tuple(int(x) for x in a.shape), int(np.asarray(a).sum())
    except Exception:
        return None, None


def estado_caso(art_dir):
    """Qué hay en disco para este caso. Lectura pura, sin efectos.

    Devuelve un dict pensado para que la vista lo formatee sin volver a
    tocar el disco, y para que decida qué mostrar: `armado` es la pregunta
    "¿existe ya una red en este caso?", que es la que separa el modo
    "muéstrame lo que tiene" del modo "guíame para armarlo".
    """
    art_dir = Path(art_dir)
    etapas = {}
    for etapa, nombre, glosa in ARTEFACTOS_ETAPA:
        if etapa == "E4":
            nombre = nombre_e4(art_dir)       # el histórico o, si no hay, el etiquetado
        ruta = art_dir / nombre
        etapas[etapa] = {
            "nombre": nombre, "glosa": glosa, "ruta": ruta,
            "existe": ruta.exists(), "mb": _mb(ruta),
        }

    baselines = art_dir / "baselines"
    escenarios = sorted(int(p.stem.split("_")[1])
                        for p in baselines.glob("esc_*.npz")) if baselines.exists() else []
    variantes = ([d.name for d in sorted(baselines.iterdir())
                  if d.is_dir() and any(d.glob("esc_*.npz"))]
                 if baselines.exists() else [])

    faultlin = ruta_faultlin(art_dir)
    forma, fallas = _forma_npy(faultlin)

    return {
        "art_dir": art_dir,
        "nombre": art_dir.name,
        "existe": art_dir.exists(),
        "es_validacion": es_validacion(art_dir),
        "es_activo": art_dir.resolve() == art_dir_activo(),
        "armado": (art_dir / "red.pkl").exists(),
        "etapas": etapas,
        "escenarios_baseline": escenarios,
        "variantes_aislacion": variantes,
        "sin_fallas_etiquetas": sorted(
            p.stem[len("sin_fallas_"):] for p in art_dir.glob("sin_fallas_*.pkl")),
        "faultlin": {"ruta": faultlin, "existe": faultlin.exists(),
                     "forma": forma, "fallas": fallas, "mb": _mb(faultlin)},
        "paquete": _meta_paquete(art_dir / "red_nueva.pkl"),
        "fallas_meta": _json_o_none(art_dir / "fallas_meta.json"),
        "faultlin_meta": _json_o_none(art_dir / "faultlin_meta.json"),
        "meta": _json_o_none(art_dir / "meta.json"),
    }


def _meta_paquete(ruta):
    """Parámetros del paquete `red_nueva.pkl` que dejó la pestaña.

    Carga el pickle completo, o sea que importa pandapower (~1 s la primera
    vez). No hay atajo: el `_meta` está en el mismo pickle que la red. La
    vista debería cachear el `estado_caso` por mtime si esto se nota.
    """
    if not ruta.exists():
        return None
    import pickle
    try:
        with open(ruta, "rb") as f:
            paquete = pickle.load(f)
    except Exception:
        return {"mb": _mb(ruta), "ilegible": True}
    meta = dict(paquete.get("_meta") or {})
    meta.update({
        "numesc": paquete.get("numesc"),
        "n_horas": paquete.get("n_horas"),
        "horas_reparacion": paquete.get("horas_reparacion"),
        "tie_indices": paquete.get("tie_indices"),
        "perfiles": paquete.get("perfiles"),
        "mb": _mb(ruta),
    })
    meta.setdefault("n_buses", len(paquete.get("bus_zone") or {}))
    return meta


def resumen_red(art_dir):
    """Cifras de `red.pkl`. Carga la red, así que se llama solo cuando se va
    a mostrar (y conviene cachearlo en la vista por mtime del archivo)."""
    import pickle

    ruta = Path(art_dir) / "red.pkl"
    if not ruta.exists():
        return None
    with open(ruta, "rb") as f:
        red = pickle.load(f)
    zonas = sorted(set((red.get("bus_zone") or {}).values()))
    return {
        "n_buses": len(red.get("BUSES") or []),
        "n_lineas": len(red.get("LINES") or []),
        "numesc": red.get("numesc"),
        "zonas": zonas,
        "tie_indices": red.get("tie_indices") or [],
        "seccionadores": red.get("seccionadores_existentes") or [],
        "buses_criticos": red.get("buses_criticos") or [],
        "demanda_kw": sum((red.get("P_D_base") or {}).values()),
        "largo_km": sum((red.get("longitud_km_linea") or {}).values()),
        "net": red.get("net"),
    }
