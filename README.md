# Estrategias para mejorar la seguridad de suministro en sistemas de distribución ante cambio climático

Código y caso de validación (red CIGRE de media tensión) de la tesis de magíster en Ingeniería Eléctrica,
Universidad de Chile.

## Contenido

- `abc/mi_proyecto/comun/app/` modelo de optimización (Pyomo/Gurobi), pipeline por etapas y apps Streamlit.
- `abc/mi_proyecto/comun/clima/` cadena climática.
- `abc/mi_proyecto/caso_validacion/` caso de validación: insumos mínimos en `datos/artefactos/` y tablas resumen de los barridos.

## Clonar en Windows

Algunas carpetas de resultados tienen nombres largos; antes de clonar:

```bash
git config --global core.longpaths true
```

## Cómo correr

```bash
cd abc/mi_proyecto/caso_validacion
python ../comun/app/pipeline.py --estado
```

El caso activo se decide por la carpeta desde la que se ejecuta (marca `.caso_erad`) o por la variable `ERAD_CASO`.
Se necesita una licencia de Gurobi.

## Qué no está aquí

Otros casos de estudio, datos de terceros y los resultados completos de los barridos (varios GB) se mantienen
en un repositorio privado.

## Licencia

Pendiente de definir.
