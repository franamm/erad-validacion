"""Compara costos MESS sin modificar insumos ni resultados del barrido.

Ejecutar desde el entorno con Pyomo y Gurobi:
    python diagnosticar_mess.py --escenario 17
Solo inspeccionar los costos y la configuración (sin solver):
    python diagnosticar_mess.py --escenario 17 --solo-costos
"""
import argparse
import copy
import gzip
import json
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--escenario', type=int, default=17)
    parser.add_argument('--barrido', default='sec-0-2-9-10')
    parser.add_argument('--solo-costos', action='store_true')
    parser.add_argument('--limite-segundos', type=float, default=180)
    args = parser.parse_args()
    import casos
    raiz = casos.art_dir_activo()
    corrida = raiz / 'barrido' / args.barrido / 'individual' / 'mess'
    nombre = f'esc_{args.escenario:03d}'
    meta = json.loads((corrida / f'{nombre}.json').read_text(encoding='utf-8'))
    with gzip.open(corrida / f'{nombre}.pkl.gz', 'rb') as archivo:
        historial = pickle.load(archivo)['historial']
    cfg = copy.deepcopy(next(iter(historial.values()))['config'])
    cfg.pop('cache_file', None)
    with (raiz / 'costos.pkl').open('rb') as archivo:
        costos = pickle.load(archivo)
    potencia = cfg['P_mess_max']
    energia = potencia * cfg['h_mess']
    inv, op, traslado = costos['MESS']
    resumen = {
        'escenario': args.escenario,
        'costos_MESS': costos['MESS'],
        'potencia_por_unidad_kW': potencia,
        'energia_por_unidad_kWh': energia,
        'unidades_candidatas': cfg['n_mess'],
        'costo_anual_por_unidad_sin_transporte':
            (inv * energia + op * potencia) * costos['frp_movil'],
    }
    print(json.dumps(resumen, indent=2, ensure_ascii=False), flush=True)
    if args.solo_costos:
        return
    try:
        from run_barrido_medidas import construir_data
        from optimization_6_oficial import resolver_optimizacion
    except ModuleNotFoundError as exc:
        parser.exit(2, f'Falta dependencia: {exc.name}. Ejecutar en el entorno del modelo.\n')
    data, _ = construir_data(
        [args.escenario], variante=meta['variante_aislacion'],
        etiqueta_e4=meta['e4_sin_fallas'], lim_dg=meta['lim_dg'],
        horas=meta['horas'], lim_dg_modo=meta['lim_dg_modo'],
    )
    cfg['gurobi_time_limit'] = args.limite_segundos
    cfg['gurobi_threads'] = 2
    # Primero reproducir el costo vigente; luego reducir inversión y finalmente
    # todos los costos MESS. No cambiar restricciones entre las tres corridas.
    casos = {'vigente': [inv, op, traslado],
             'inversion_10pct': [inv * .1, op * .1, traslado],
             'sin_costos_MESS': [0, 0, 0]}
    salida = raiz / 'diagnostico_mess'
    salida.mkdir(exist_ok=True)
    for caso, valores in casos.items():
        datos = dict(data, MESS=valores)
        resultados = resolver_optimizacion(datos, copy.deepcopy(cfg))
        horas = resultados.get(args.escenario, [])
        if not horas:
            raise RuntimeError(f'{caso}: el solver no devolvio resultados.')
        primero = horas[0]
        fila = dict(resumen, caso=caso, costos_MESS=valores,
                    unidades_instaladas=sum(primero['y_mess'].values()),
                    energia_descargada_kWh=sum(sum(h['Pdis_mess'].values()) for h in horas),
                    objetivo=primero['fn_objetivo'],
                    solver_term=primero['solver_term'], solver_gap=primero['solver_gap'])
        (salida / f'{nombre}_{caso}.json').write_text(
            json.dumps(fila, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(fila, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
