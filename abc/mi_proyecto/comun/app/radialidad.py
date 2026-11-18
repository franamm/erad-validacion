"""
radialidad.py — Restricciones de radialidad para el cierre de enlaces (reconfig_topo / inv_tie).

Sin esto, el modelo puede cerrar a la vez los 29 enlaces: cerrar un enlace cuyo lazo
no tiene ninguna línea fuera de servicio deja la red mallada, que una red de
distribución MT no opera (protecciones, coordinación).

Como la aislación de fallas viene de seccionadores fijos (ver `aislacion.py`), el
conjunto de líneas fuera de servicio de cada hora es un DATO. Entonces la radialidad
no necesita variables nuevas:

  1. Las líneas sanas (no enlaces, fuera de servicio = 0) agrupan las barras en
     islas. Las barras de cabecera se tratan como una sola isla (comparten el lado
     de alta tensión).
  2. Un enlace con los dos extremos en la misma isla cierra un ciclo por sí solo:
     se fija en 0.
  3. Los demás enlaces son aristas entre islas. Cerrarlos es radial si no forman
     ciclo: se agrega, por cada ciclo del grafo de islas, sum(z) <= largo - 1.
     Los enlaces paralelos (misma pareja de islas) quedan como sum(z) <= 1.

Limitación: la lista de líneas fuera de servicio es la del daño base. Con medidas
que cambian el daño dentro del MILP (veg/refuerzo/soterramiento) o la aislación
(inv_switch_*) es una aproximación.
"""
import networkx as nx


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def restricciones_radiales(dam_vec, LINES, from_bus, to_bus, tie_list, fuentes,
                           max_ciclos=20000):
    """Devuelve (fijar_cero, grupos) para UNA hora.

    fijar_cero : enlaces que no pueden cerrarse (cerrarían un ciclo por sí solos).
    grupos     : lista de tuplas de enlaces con sum(z) <= len(grupo_de_aristas) - 1;
                 cada elemento es (enlaces, cota).
    dam_vec    : líneas fuera de servicio de esa hora (averiadas + aisladas), 0/1.
    """
    ties = set(tie_list)
    uf = _UF()
    for l in LINES:
        if l in ties or dam_vec[l] == 1:
            continue
        uf.union(from_bus[l], to_bus[l])
    fuentes = list(fuentes)
    for b in fuentes[1:]:
        uf.union(fuentes[0], b)

    fijar, aristas = [], {}
    for l in tie_list:
        if dam_vec[l] == 1:
            continue                       # ya fijado por el modelo (enlace dañado)
        a, b = uf.find(from_bus[l]), uf.find(to_bus[l])
        if a == b:
            fijar.append(l)
        else:
            aristas.setdefault(frozenset((a, b)), []).append(l)

    grupos = []
    for par, ls in aristas.items():
        if len(ls) > 1:                    # enlaces paralelos: a lo más uno
            grupos.append((tuple(ls), 1))

    G = nx.Graph()
    for par, ls in aristas.items():
        a, b = tuple(par)
        G.add_edge(a, b, ties=tuple(ls))
    n = 0
    for ciclo in nx.simple_cycles(G):
        if len(ciclo) < 3:
            continue
        n += 1
        if n > max_ciclos:
            raise RuntimeError(f"radialidad: más de {max_ciclos} ciclos en una hora")
        ls = []
        for i in range(len(ciclo)):
            ls.extend(G[ciclo[i]][ciclo[(i + 1) % len(ciclo)]]["ties"])
        grupos.append((tuple(ls), len(ciclo) - 1))
    return fijar, grupos
