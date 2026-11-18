# Función objetivo y atribución del beneficio de DG

Derivación de lo que realmente optimiza `optimization_6_oficial.py` en el término
de generación distribuida, y verificación numérica de si el beneficio que se le
acredita es efectivamente **solo el atribuible a resiliencia**.

Todos los números son del artefacto `sin_fallas_LIM-1.00.pkl` (E4),
`artefactos/baselines/sec-0-2-9-10/esc_000.npz` (E5) y del barrido
`artefactos/barrido/sec-0-2-9-10 LIM-1.00/`.

---

## 1. Notación

| símbolo | qué es | valor / fuente |
|---|---|---|
| $c$ | costo de compra a la subestación | `C_sub[0]` = 0,1116 USD/kWh |
| $V_b$ | VoLL de la barra $b$ | `VoLL_local` |
| $N$ | número de eventos al año | `N_eventos` |
| $x$ | decisión: capacidades DG, despacho, deslastre | variables del MILP |
| $x^{sf}$ | solución óptima **sin fallas** | E4 |
| $K^{sf},\; O^{sf}$ | CAPEX y OPEX anuales **puros** de $x^{sf}$ | E4 |
| $P^{sf}_0$ | $\sum P_{SUB}$ sin fallas, **sin** DG | E4/1 = 155.985.005 kWh |
| $P^{sf}_{tec}$ | $\sum P_{SUB}$ sin fallas, **con** esa DG | E4/2, E4/3 |
| $P^{base}_e$ | $\sum P_{SUB}$ del caso base **con** fallas del escenario $e$ | E5 |
| $P(x),\,K(x),\,O(x)$ | los de la corrida con estrategia | barrido |

Valores de E4 con `lim_dg = 1,0`:

$$P^{sf}_0 = 155.985.005 \qquad P^{sf}_{sol} = 105.412.191 \qquad P^{sf}_{eol} = 57.351.892 \;\;[\text{kWh}]$$

---

## 2. La constante de E4 no es OPEX

`pipeline._capex_opex_dg` (líneas 660-662) guarda:

$$K^{sf}_{\text{guardado}} = K^{sf}$$

$$O^{sf}_{\text{guardado}} \;=\; O^{sf} \;\underbrace{-\; c\,\bigl(P^{sf}_0 - P^{sf}_{tec}\bigr)}_{\text{cost\_ahorro\_sub}}$$

Es decir, `OPEX_DG_*_sin_fallas` **lleva embebido el ahorro de subestación** que esa
DG lograría en un mundo sin fallas. Por eso el valor guardado es negativo.
Descomposición verificada:

| | $\sum P_{SUB}$ sin fallas | ahorro embebido | $O^{sf}$ puro | valor guardado |
|---|---:|---:|---:|---:|
| DG solar | 105.412.191 | −5.643.712 | 253.066 | **−5.390.647** |
| DG eólica | 57.351.892 | −11.007.039 | 981.621 | **−10.025.418** |

Está documentado en `pipeline.py:642-647` y deliberadamente no corregido.

---

## 3. Objetivo tal como está escrito en el código

De `optimization_6_oficial.py`, líneas 1323-1327 (constantes de DG) y 1385
(crédito de subestación):

$$Z(x)=\bigl[K(x)-K^{sf}\bigr]\;+\;\bigl[O(x)-O^{sf}_{\text{guardado}}\bigr]\;+\;N\sum_{t}\sum_{b} V_b\,P_{LC}(x)_{t,b}\;-\;c\bigl[P^{base}_e-P(x)\bigr]$$

---

## 4. Reagrupación: de dónde sale la atribución

Sustituyendo $O^{sf}_{\text{guardado}}$:

$$-\,O^{sf}_{\text{guardado}} \;=\; -\,O^{sf} \;+\; c\,\bigl(P^{sf}_0 - P^{sf}_{tec}\bigr)$$

Ese $+c(P^{sf}_0-P^{sf}_{tec})$ se **fusiona** con el crédito explícito de
subestación. Definiendo

$$A_{obs}(x) \;=\; P^{base}_e - P(x) \qquad\text{(ahorro observado de compra a la subestación)}$$

$$A^{sf} \;=\; P^{sf}_0 - P^{sf}_{tec} \qquad\text{(ahorro que esa DG lograría sin ninguna falla)}$$

queda:

$$\boxed{\;Z(x)\;=\;\underbrace{\bigl[K(x)-K^{sf}\bigr]}_{\Delta K}\;+\;\underbrace{\bigl[O(x)-O^{sf}\bigr]}_{\Delta O}\;+\;N\sum_{t,b} V_b\,P_{LC}(x)_{t,b}\;-\;c\,\bigl[A_{obs}(x)-A^{sf}\bigr]\;}$$

**Restar el OPEX sin fallas es correcto**: es exactamente lo que convierte el
crédito bruto $A_{obs}$ en el crédito neto $A_{obs}-A^{sf}$, es decir, en el ahorro
que la DG entrega *por encima* de lo que habría entregado igual sin tormentas.
Lo mismo ocurre con $\Delta K$ y $\Delta O$: solo se cobra la capacidad instalada
**por sobre** la que se instalaría sin fallas.

### Condición de consistencia

Si no hubiera fallas, el óptimo sería $x=x^{sf}$, y entonces

$$\Delta K = 0,\qquad \Delta O = 0,\qquad ENS = 0,\qquad A_{obs}=A^{sf}$$

$$\Longrightarrow\; Z = 0$$

La DG no aporta ni resta nada al objetivo. **Todo $Z\neq 0$ es efecto de las
fallas.** Ésa es la definición operativa de "solo resiliencia", y la condición
para que se cumpla es que $x = x^{sf}$ en todo lo que no depende de fallas.

---

## 5. Verificación numérica (escenario 0, `lim_dg = 1,0`)

$P^{base}_0 = 155.909.317$ kWh.

Reconstrucción de $Z$ desde sus partes, contra `fn_objetivo` del MILP:

| término | dg_sol | dg_eol |
|---|---:|---:|
| $\Delta K$ | 0 | 196.334 |
| $\Delta O$ | 0 | 27.762 |
| $N\sum V_b P_{LC}$ | 268.655 | 237.662 |
| $-c\,(A_{obs}-A^{sf})$ | +2.609 | −153.423 |
| **suma** | **271.264** | **308.335** |
| `fn_objetivo` del MILP | 271.264 | 308.335 |
| **residuo** | **0** | **0** |

La identidad cierra al peso: la fórmula de §4 **es** lo que el modelo optimiza.

---

## 6. ¿Se cuenta solo resiliencia? Depende de la capacidad instalada

La condición $x=x^{sf}$ se cumple o no según la tecnología:

| | $\sum \text{CAP}^{sf}$ | $\sum \text{CAP}(x)$ | ¿se cumple? |
|---|---:|---:|---|
| DG solar | 29.323,9 kW | 29.323,9 kW | **sí**, exacto (topada por `lim_dg`) |
| DG eólica | 43.511,5 kW | 44.742,2 kW | **no**, +1.230,7 kW (+2,8 %) |

Descomponiendo $A_{obs}(x)-A^{sf}$ hora por hora, separando las 8.684 horas sin
línea fuera de servicio de las 76 horas con falla:

$$A_{obs}(x)-A^{sf}\;=\;\underbrace{\sum_{t\,\in\,\mathcal{T}_{sanas}}\!\!\bigl[\cdot\bigr]}_{\text{fuga: no es resiliencia}}\;+\;\underbrace{\sum_{t\,\in\,\mathcal{T}_{falla}}\!\!\bigl[\cdot\bigr]}_{\text{resiliencia real}}$$

### DG solar — atribución exacta

```
CAPACIDAD    sin fallas (E4)  29.323,9 kW
             con fallas       29.323,9 kW   ═══  IDÉNTICA ✓

A_obs − A_sf, hora por hora:
  8.684 h sanas   ████████████████████████████        0 kWh   ✓
     76 h falla   █                             −23.376 kWh
                                              ──────────────
                                  TOTAL         −2.609 USD

⇒ el 100 % del crédito viene de horas con falla.
```

El crédito es levemente **negativo**: durante las fallas parte del solar queda
aislado y entrega menos que en el mundo sin fallas. Físicamente correcto.

### DG eólica — atribución rota

```
CAPACIDAD    sin fallas (E4)  43.511,5 kW
             con fallas       44.742,2 kW   ≠  +1.230,7 kW (+2,8 %) ✗

A_obs − A_sf, hora por hora:
  8.684 h sanas   ████████████████████████████ +1.952.355 kWh
                                               → +217.875 USD  ← FUGA
     76 h falla   █                              −577.548 kWh
                                               →  −64.452 USD  ← resiliencia
                                              ──────────────
                                  TOTAL         +153.423 USD

⇒ el crédito es 142 % fuga y −42 % resiliencia.
```

La capacidad eólica extra genera **también en las 8.684 horas sanas**, donde no
hay resiliencia que premiar, y ese desplazamiento entra en $A_{obs}$ sin estar en
$A^{sf}$. El efecto de resiliencia real es **negativo** (−64.452 USD), que es lo
esperable: durante una falla el eólico queda aislado y entrega *menos*. El signo
del crédito reportado está invertido respecto del fenómeno.

Magnitud: el objetivo de `dg_eol` es 308.335; sin la fuga sería ≈ 526.210. Es una
distorsión del 41 % del valor reportado, en las 10 corridas de `dg_eol` (la
capacidad instalada es 44.742,2 kW en los 10 escenarios).

---

## 7. Segundo problema: doble conteo al combinar ambas DG

Independiente del anterior. Con `dg_sol` y `dg_eol` activas a la vez, el objetivo
resta **dos** constantes de E4, cada una con su propio ahorro embebido medido
contra el mismo $P^{sf}_0$:

$$Z_{\text{ambas}} \;=\; \ldots \;-\; c\Bigl[A_{obs}(x)\;-\;\underbrace{A^{sf}_{sol}\;-\;A^{sf}_{eol}}_{\text{no son aditivos}}\Bigr]$$

Los dos contrafactuales se solapan: cada uno se midió en un E4 donde esa
tecnología era **lo único** instalado, y ambas compiten por las mismas horas y la
misma demanda.

$$A^{sf}_{sol}+A^{sf}_{eol} = 50.572.814+98.633.113 = 149.205.927\ \text{kWh}$$

$$A_{obs}(\text{combo}) = 155.909.317-48.841.623 = 107.067.694\ \text{kWh}$$

$$A_{obs}-A^{sf}_{sol}-A^{sf}_{eol} = -42.138.233\ \text{kWh}\;\longrightarrow\;-4.702.449\ \text{USD}$$

Ahorro de resiliencia **negativo por 42 GWh**, imposible: el máximo desplazable
es 156 GWh y juntas logran ~107 GWh. El término entra como **penalización** de
4,70 M USD.

Efecto visible en `resumen_combinadas.csv` (escenario 0):

| combinación | objetivo | E_sub |
|---|---:|---:|
| `dg_sol+ess_5h` | −114.389 | 86.468.582 |
| `dg_eol+ess_5h` | 96.511 | 54.210.141 |
| `dg_eol+dg_sol+ess_5h` | **2.296.739** | **48.841.623** |

El combo entrega el mejor desempeño físico y sale 2,4 M peor en objetivo.

Las columnas `cst_*` no sirven de contraste: `_extraer_costos_objetivo`
(`helpers_inversion.py:446-447, 453-454`) replica las mismas restas, por eso
`cst_costo_total` == `objetivo` exacto en todas las filas.

### Alcance

- **No invalida el óptimo de cada corrida**: para un conjunto fijo de medidas es
  una constante aditiva.
- **No afecta la elección de representativos**: `--criterio objetivo` compara
  escenarios dentro de una misma medida, la constante se cancela.
- **Sí invalida la comparación entre combinaciones** con distinto número de
  tecnologías DG — que es el objeto de la fase 3 del barrido.

---

## 8. Correcciones propuestas

### 8.1 Fuga por capacidad (§6)

Evaluar $A^{sf}$ a la capacidad realmente instalada, no a $x^{sf}$. La generación
sin fallas escala aproximadamente lineal con la capacidad, así que el MILP sigue
siendo lineal:

$$A^{sf}(x)\;\approx\;\bigl(P^{sf}_0-P^{sf}_{tec}\bigr)\cdot\frac{\sum_g \text{CAP}(x)_g}{\sum_g \text{CAP}^{sf}_g}$$

Para solar no cambia nada (el cociente vale 1); para eólica elimina la fuga.
Aproximado si hay vertimiento en algunas horas.

### 8.2 Doble conteo (§7)

Correr **una** E4 adicional con `dg_sol=True, dg_eol=True` simultáneas y guardar
`CAPEX/OPEX_DG_ambas_sin_fallas`; usar esa constante cuando ambas estén activas en
lugar de las dos por separado. No requiere rehacer E5 ni las corridas
individuales, solo las combinadas que llevan las dos DG.

### 8.3 Bug latente en `optimization_base`

`optimization_base.py:706` y `:710` suman el crédito de subestación dentro de
`if usar_dg_sol` y de `if usar_dg_eol` por separado. Si alguna vez se corriera E4
con ambas encendidas, se duplicaría. Hoy no ocurre (`pipeline.py:715-719` las corre
una por vez). Es el mismo bug que `optimization_6` ya corrigió sacando el término
fuera de los `if`.

---

## 9. Pendiente

Revisar si `dg_fos` y las demás medidas que consumen constantes de E4 tienen la
misma fuga de §6.
