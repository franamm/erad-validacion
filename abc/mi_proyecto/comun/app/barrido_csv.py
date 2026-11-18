"""Lectura de resúmenes concatenados sin interpretar cabeceras como corridas."""
import csv
import io

import pandas as pd


def leer_resumen(ruta):
    # Filtrar antes de read_csv permite que pandas infiera nuevamente los
    # tipos numéricos de TODAS las columnas, no solo el índice de escenario.
    limpio = io.StringIO()
    with open(ruta, newline='', encoding='utf-8-sig') as archivo:
        lector = csv.reader(archivo)
        cabecera = next(lector, None)
        if cabecera is None:
            return pd.DataFrame()
        escritor = csv.writer(limpio)
        escritor.writerow(cabecera)
        for fila in lector:
            normalizada = [celda.strip().lstrip('\ufeff') for celda in fila]
            if normalizada == cabecera:
                continue
            escritor.writerow(fila)
    limpio.seek(0)
    datos = pd.read_csv(limpio)
    if 'escenario' in datos:
        escenarios = pd.to_numeric(datos['escenario'], errors='coerce')
        validos = escenarios.notna() & escenarios.ge(0) & escenarios.mod(1).eq(0)
        if not validos.all():
            valores = datos.loc[~validos, 'escenario'].head(5).tolist()
            raise ValueError(f'{ruta}: identificadores de escenario inválidos: {valores}')
        datos['escenario'] = escenarios.astype('int64')
    return datos
