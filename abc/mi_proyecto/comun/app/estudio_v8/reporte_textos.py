"""Frases del reporte, llenadas con los números del estudio (sin IA, reproducibles).

Cada función recibe el diccionario de `reporte_datos.reunir` y devuelve texto o una
lista de párrafos. Un párrafo es una lista de segmentos (texto, negrita).
Formato numérico chileno: miles con punto y decimales con coma.
"""
import numpy as np


def miles(v, dec=0):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if round(float(v), dec) == 0:
        v = 0.0                    # sin "−0,0"
    return f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def usd(v, dec=0):
    return f"{miles(v, dec)} USD"


def pct(v, dec=1):
    return f"{miles(v, dec)} %"


def mwh(kwh, dec=1):
    return f"{miles(kwh / 1000, dec)} MWh"


def km(v):
    return f"{miles(v * 1000)} m" if v < 0.1 else f"{miles(v, 2)} km"


def lista_y(xs):
    xs = [str(x) for x in xs]
    return "" if not xs else xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " y " + xs[-1]


def P(*segs):
    """Párrafo: P("texto normal ", ("en negrita", True), " sigue")."""
    return [s if isinstance(s, tuple) else (s, False) for s in segs]


REGLA_TXT = {"Valor esperado": "valor esperado", "Minimax": "minimax (mejor peor caso)",
             "Minimax regret": "mínimo arrepentimiento máximo"}


# ── Resumen ejecutivo ───────────────────────────────────────────────────────

def indicadores_plan(D, p=None):
    """Valor esperado del plan sobre todos los escenarios y del caso base."""
    dec = D["decision"]
    p = D["recomendado"]["plan"] if p is None else p
    m = dec["medias"].loc[p]
    b = D["base"].loc[dec["escenarios"]].mean()
    return m, b


def frase_recomendacion(D):
    R, dec = D["recomendado"], D["decision"]
    m, b = indicadores_plan(D)
    n = len(dec["escenarios"])
    d_ens = 100 * (1 - m["ens"] / b["ens"])
    origen = f"diseñado para el escenario {D['num_esc'][R['plan']]}"
    verbo = "Se analiza" if R["forzado"] else "Se recomienda"
    return P(f"{verbo} el ", (f"plan {R['etiqueta']}", True), f", {origen}. Evaluado con su inversión fija en los {n} "
             f"escenarios, cuesta ", (f"{usd(m['cst_costo_total'])}/año", True),
             f" (CAPEX anualizado {usd(m['cst_capex_anual'])} y OPEX {usd(m['cst_opex_anual'])}) y reduce la ENS "
             f"esperada de {mwh(b['ens'])} a {mwh(m['ens'])} al año (", (f"−{pct(d_ens)}", True),
             f"). El SAIDI baja de {miles(b['saidi'], 2)} a {miles(m['saidi'], 2)} h/cliente y el SAIFI de "
             f"{miles(b['saifi'], 2)} a {miles(m['saifi'], 2)} interrupciones por cliente. El costo de la energía no "
             f"suministrada pasa de {usd(b['costo_ens'])} a {usd(m['costo_ens'])} al año, con un beneficio neto de ",
             (f"{usd(m['cst_ahorro_neto'])}/año", True), ".")


def vinetas_inversion(inv, op=None, n_esc=None):
    """Qué instala (inversión) y cómo opera (operación) un plan."""
    out = []
    for nom, x in inv.items():
        if x["unidad"] == "km":
            els = lista_y([f"L{l}" for l, _ in x["elementos"][:4]]) + (" y otras" if x["n"] > 4 else "")
            out.append(P((f"{nom}: ", True), f"{km(x['total'])} en {x['n']} tramo{'s' * (x['n'] > 1)} ({els})."))
        elif nom.startswith("Generador"):
            nuevo = x.get("nuevos", 0)
            txt = (f"solo la unidad existente de {miles(x['total'])} kW, sin compra de unidades nuevas" if not nuevo
                   else f"{x['n']} unidades ({nuevo} nuevas), {miles(x['total'])} kW en total")
            out.append(P((f"{nom}: ", True), txt + "."))
        else:
            top = lista_y([f"B{b} ({miles(v)} kW)" for b, v in x["elementos"][:3]])
            out.append(P((f"{nom}: ", True), f"{miles(x['total'])} kW en {x['n']} barra{'s' * (x['n'] > 1)}; "
                         f"las mayores, {top}."))
    if op:
        n = n_esc or op["n"]
        frec = [f"E{l}" for l, c in op["enlaces"].items() if c >= 0.8 * n]
        if op["enlaces"]:
            out.append(P(("Reconfiguración: ", True), f"cierra {len(op['enlaces'])} enlaces existentes"
                         + (f"; {lista_y(frec)} en al menos el 80 % de los escenarios" if frec else "")
                         + (f", con {miles(op['maniobras'][0])} a {miles(op['maniobras'][1])} maniobras al año"
                            if op["maniobras"] else "") + "."))
        if op["arriendos"] and op["arriendos"][1] > 0:
            out.append(P(("Arriendo de generadores: ", True),
                         f"{miles(op['arriendos'][0])} a {miles(op['arriendos'][1])} arriendos al año, según dónde caigan las fallas."))
        if op["edrp_MWh"] and op["edrp_MWh"][1] > 0.01:
            out.append(P(("Respuesta de demanda: ", True),
                         f"{miles(op['edrp_MWh'][0], 1)} a {miles(op['edrp_MWh'][1], 1)} MWh reducidos al año."))
    return out


def frase_por_que(D):
    dec, et = D["decision"], D["etiqueta"]
    p = D["recomendado"]["plan"]
    gana = [r for r, v in dec["elegidos"].items() if p in v]
    otras = {r: [et[q] for q in v] for r, v in dec["elegidos"].items() if p not in v}
    ind = "el costo del plan (costo anual de las medidas más el costo de la ENS valorizada con el VoLL)"
    if len(gana) == len(dec["elegidos"]):
        txt = P(f"Es el plan elegido por las {len(gana)} reglas de decisión aplicadas sobre {ind}: ",
                (lista_y([REGLA_TXT[r] for r in gana]), True), ". La elección no depende de la actitud frente al riesgo.")
    elif gana:
        txt = P(f"Lo eligen {lista_y([REGLA_TXT[r] for r in gana])} sobre {ind}; "
                + "; ".join(f"{REGLA_TXT[r]} elige {lista_y(v)}" for r, v in otras.items())
                + ". La elección depende de la actitud frente al riesgo.")
    else:
        txt = P(f"No lo elige ninguna regla sobre {ind}: se reporta por elección del usuario.")
    return txt


def frase_alternativa(D):
    R = D["recomendado"]
    if "alternativa" not in R:
        return None
    a = R["alternativa"]
    m, _ = indicadores_plan(D)
    ma, _ = indicadores_plan(D, a["plan"])
    d_c, d_e = ma["cst_costo_total"] - m["cst_costo_total"], (m["ens"] - ma["ens"])
    voll = D["costos"]["voll"]
    marg = d_c / d_e if d_e > 0 else np.nan
    juicio = "sobre" if marg > voll else "bajo"
    return P(("Alternativa: ", True), f"el plan {a['etiqueta']} deja la menor ENS esperada ({mwh(ma['ens'])}), "
             f"pero cuesta {usd(d_c)}/año más que {R['etiqueta']} para evitar {mwh(d_e)} adicionales: "
             f"{miles(marg, 1)} USD por kWh evitado, {juicio} el VoLL de {miles(voll, 0)} USD/kWh.")


def advertencias(D):
    out = []
    am, dec = D["amenaza"], D["decision"]
    n = len(D["escenarios"])
    if am.get("escenarios_poblacion"):
        out.append(P(("Muestra de escenarios: ", True), f"los resultados son el valor esperado sobre {n} escenarios "
                     f"severos elegidos de una población de {am['escenarios_poblacion']} años simulados; no son "
                     "probabilidades de la población completa."))
    if dec and np.isfinite(dec["reglas"]["evpi"]):
        out.append(P(("Valor de la información: ", True), f"comprometerse con un plan sin saber qué escenario ocurrirá "
                     f"cuesta {usd(dec['reglas']['evpi'])}/año más que decidir conociéndolo (EVPI de la muestra)."))
    C = D["combinacion"]
    if C:
        s = C["solapamiento"]["ens"]
        if np.isfinite(s["razon"]):
            out.append(P(("Las medidas no se suman: ", True), f"juntas abaten {pct(100 * s['razon'], 0)} de lo que "
                         "sumarían evaluadas por separado, porque compiten por las mismas fallas. Las fichas de medidas "
                         "individuales no sirven para dimensionar una cartera."))
    I = D["individuales"]
    if I and set(I["escenarios"]) != set(D["escenarios"]):
        out.append(P(("Medidas individuales: ", True), f"se evaluaron en {len(I['escenarios'])} de los {n} escenarios; "
                     "sus curvas de costo de abatimiento usan esa submuestra."))
    return out


# ── Secciones ──────────────────────────────────────────────────────────────

def texto_red(D):
    r = D["red"]
    z = r["zonas"].sort_values("Demanda base [kW]", ascending=False).iloc[0]
    return [P(f"La red tiene {r['barras']} barras y {r['lineas']} líneas ({miles(r['km'], 1)} km), alimentadas desde "
              f"{r['subestaciones']} subestaciones, con {r['enlaces']} enlaces normalmente abiertos entre alimentadores y "
              f"{r['seccionadores']} seccionadores. La demanda base suma {miles(r['demanda_kW'] / 1000, 2)} MW y "
              f"{miles(r['clientes'])} clientes. La zona {z['Zona']} concentra {pct(100 * z['Demanda base [kW]'] / r['demanda_kW'], 0)} "
              "de la demanda.")]


def texto_amenaza(D):
    am, n = D["amenaza"], len(D["escenarios"])
    fe = list(D["fallas_esc"].values())
    out = [P(f"La amenaza es el viento extremo. Una simulación de Monte Carlo genera {am.get('escenarios_poblacion') or '—'} "
             f"años de {miles(am.get('horas') or 8760)} horas con las fallas de cada línea, según curvas de fragilidad que "
             f"dependen de la ráfaga y del largo expuesto de la línea; una línea dañada queda fuera "
             f"{am.get('horas_reparacion') or '—'} horas hasta su reparación.")]
    txt = f"El estudio usa {n} escenarios"
    if D["grupos"]:
        txt += ", agrupados en " + lista_y([f"{k} ({len(v)})" for k, v in D["grupos"].items()])
    if fe:
        txt += f", con {min(fe)} a {max(fe)} fallas por escenario"
    out.append(P(txt + ". Todos los planes se evalúan sobre los mismos escenarios, así la mejora es atribuible a las medidas."))
    return out


def texto_base(D):
    b = D["base"].loc[D["escenarios"]]
    z = D["base_zonal"]
    out = [P(f"Sin medidas, la red deja en promedio ", (f"{mwh(b['ens'].mean())} sin suministrar al año", True),
             f" (entre {mwh(b['ens'].min())} y {mwh(b['ens'].max())} según el escenario), con un SAIDI de "
             f"{miles(b['saidi'].mean(), 2)} h/cliente, un SAIFI de {miles(b['saifi'].mean(), 2)} y un costo de la ENS de "
             f"{usd(b['costo_ens'].mean())}/año.")]
    if z is not None:
        zz = z.sort_values("Caso base", ascending=False).iloc[0]
        out.append(P(f"La {zz['Zona'].replace('Zona ', 'zona ZG')} concentra {pct(100 * zz['Caso base'] / z['Caso base'].sum(), 0)} "
                     "de la ENS: es donde deben actuar las medidas."))
    return out


def texto_individuales(D):
    I = D["individuales"]
    pr = I["prom"]
    ok = pr[pr["usd_kwh"].notna() & (pr["d_ens"] > 1)].sort_values("usd_kwh")
    nulas = pr[~pr["_caso"].isin(ok["_caso"])]["Medida"].tolist()
    voll = D["costos"]["voll"]
    out = [P("Cada medida se optimiza sola, en los mismos escenarios, y se compara con el caso base. La curva de costos "
             "de abatimiento ordena las medidas por su costo por unidad de mejora; el ancho de cada bloque es lo que abate.")]
    if len(ok):
        a, z = ok.iloc[0], ok.iloc[-1]
        bajo = ok[ok["usd_kwh"] < voll]
        out.append(P("La más barata por kWh evitado es ", (f"{a['Medida']} ({miles(a['usd_kwh'], 2)} USD/kWh)", True),
                     f" y la más cara {z['Medida']} ({miles(z['usd_kwh'], 2)} USD/kWh). {len(bajo)} de {len(ok)} medidas "
                     f"cuestan menos que el VoLL de {miles(voll, 0)} USD/kWh."))
    if nulas:
        out.append(P(f"No reducen la ENS en estos escenarios, y el modelo no las instala: {lista_y(nulas)}."))
    return out


def texto_combinacion(D):
    C = D["combinacion"]
    out = []
    for k, nom, u, dec in (("ens", "ENS", "MWh", 1), ("saidi", "SAIDI", "h/cliente", 2), ("saifi", "SAIFI", "int./cliente", 2)):
        s = C["solapamiento"][k]
        f = 1000 if k == "ens" else 1
        out.append(P((f"{nom}: ", True), f"la suma de las medidas activas evaluadas por separado abatiría "
                     f"{miles(s['suma'] / f, dec)} {u}; juntas abaten {miles(s['conjunto'] / f, dec)} {u} "
                     f"({pct(100 * s['razon'], 0)})."))
    return out


def texto_planes(D):
    dec, et = D["decision"], D["etiqueta"]
    n = len(dec["planes"])
    return [P(f"Cada escenario entrega su propia inversión óptima: son {n} planes candidatos ({et[dec['planes'][0]]} a "
              f"{et[dec['planes'][-1]]}, numerados como sus escenarios). Cada plan se vuelve a evaluar con su inversión fija "
              f"en todos los escenarios, dejando libre la operación (enlaces, generadores móviles, respuesta de demanda). "
              f"Eso da una matriz de {n} planes por {len(dec['escenarios'])} escenarios.")]


def texto_pareto(D, frontera, grupo=None):
    et = D["etiqueta"]
    txt = f"En {'el ' + grupo if grupo else 'valor esperado'}, la frontera entre ENS y costo anual de las medidas la forman "
    return P(txt, (lista_y([et[p] for p in frontera]), True),
             ". Los demás planes están dominados: hay otro que cuesta menos y deja menos ENS.")


def texto_reglas(D):
    dec, et = D["decision"], D["etiqueta"]
    out = [P("Las reglas de decisión eligen un plan sobre la matriz del costo del plan (medidas + ENS valorizada con el "
             "VoLL), según la actitud frente al riesgo: el ", ("valor esperado", True), " promedia los escenarios; el ",
             ("minimax", True), " protege del peor escenario; el ", ("mínimo arrepentimiento máximo", True),
             " minimiza lo que se lamentaría frente al plan diseñado para el escenario que ocurre.")]
    out.append(P("Resultado: " + "; ".join(f"{REGLA_TXT[r]} elige {lista_y([et[p] for p in v])}"
                                           for r, v in dec["elegidos"].items()) + "."))
    return out
