"""Barre potencia/energia por unidad MESS a costo vigente, sin tocar insumos.

Ejecutar desde el entorno con Pyomo y Gurobi:
    python diagnosticar_mess_tamano.py --escenario 0 --barrido individual-esc0
"""
import argparse
import copy
import gzip
import json
import pickle
from pathlib import Path


# (P_mess_max kW, h_mess h) a probar, ademas del caso vigente (200, 3).
COMBOS = [
    (200, 3),    # vigente
    (200, 6),    # mas horas de autonomia, mismo costo dominante (potencia)
    (200, 10),
    (100, 3),    # mitad de potencia -> ~mitad de costo (domina C_op*P)
    (50, 3),
    (400, 3),    # el doble de potencia
    (100, 6),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--escenario', type=int, default=0)
    parser.add_argument('--barrido', default='individual-esc0')
    parser.add_argument('--limite-segundos', type=float, default=300)
    args = parser.parse_args()
    import casos
    raiz = casos.art_dir_activo()
    corrida = raiz / 'barrido' / args.barrido / 'individual' / 'mess'
    nombre = f'esc_{args.escenario:03d}'
    meta = json.loads((corrida / f'{nombre}.json').read_text(encoding='utf-8'))
    with gzip.open(corrida / f'{nombre}.pkl.gz', 'rb') as archivo:
        historial = pickle.load(archivo)['historial']
    cfg_base = copy.deepcopy(next(iter(historial.values()))['config'])
    cfg_base.pop('cache_file', None)
    with (raiz / 'costos.pkl').open('rb') as archivo:
        costos = pickle.load(archivo)
    inv, op, tra = costos['MESS']

    from run_barrido_medidas import construir_data
    from optimization_6_oficial import resolver_optimizacion
    data, _ = construir_data(
        [args.escenario], variante=meta['variante_aislacion'],
        etiqueta_e4=meta['e4_sin_fallas'], lim_dg=meta['lim_dg'],
        horas=meta['horas'], lim_dg_modo=meta['lim_dg_modo'],
    )

    salida = raiz / 'diagnostico_mess'
    salida.mkdir(exist_ok=True)
    for p_max, h in COMBOS:
        cfg = copy.deepcopy(cfg_base)
        cfg['gurobi_time_limit'] = args.limite_segundos
        cfg['gurobi_threads'] = 2
        cfg['P_mess_max'] = p_max
        cfg['h_mess'] = h
        e_cap = p_max * h
        cfg['E_cap_mess'] = e_cap
        cfg['E_cap_mess_total'] = cfg['n_mess'] * e_cap
        costo_anual_unidad = (inv * e_cap + op * p_max) * costos['frp_movil']
        resultados = resolver_optimizacion(dict(data), cfg)
        horas = resultados.get(args.escenario, [])
        if not horas:
            raise RuntimeError(f'P={p_max} h={h}: el solver no devolvio resultados.')
        primero = horas[0]
        fila = {
            'escenario': args.escenario, 'P_mess_max_kW': p_max, 'h_mess': h,
            'E_cap_mess_kWh': e_cap, 'costo_anual_por_unidad': costo_anual_unidad,
            'unidades_candidatas': cfg['n_mess'],
            'unidades_instaladas': sum(primero['y_mess'].values()),
            'energia_descargada_kWh': sum(sum(hr['Pdis_mess'].values()) for hr in horas),
            'objetivo': primero['fn_objetivo'],
            'solver_term': primero['solver_term'], 'solver_gap': primero['solver_gap'],
        }
        (salida / f'{nombre}_tamano_P{p_max}_h{h}.json').write_text(
            json.dumps(fila, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(fila, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
