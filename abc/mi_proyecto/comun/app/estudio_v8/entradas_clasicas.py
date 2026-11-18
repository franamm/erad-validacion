"""Vista de entradas recuperada de la v7; copia aislada de su presentación."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def render():
    from data_loader import net, BUSES, LINES, GENS, ESS, faultlinMC, repairMC, demand_factor, alphaDGsolar, alphaDGeolica, bus_zone, P_D_base, N_eventos, sw_cand_list, tie_cand_list, C_incentivo_edrp, N_HORAS, horas, numesc, esc_esp_ui, r_econ, n_econ, frp, frp_movil, VoLL, data, _norm, buses_criticos, h_ess_4h, h_ess_5h, h_ess_6h, baseline_esc, ESCENARIOS_CON_BASELINE, escenarios_con_baseline, p_sub_total_esc, lim_dg_e4, escenarios_muestra, escenarios_barrido
    from grafo import build_fig_grafo
    from helpers_operativos import PALETA
    esc = st.selectbox("Escenario de entrada (índice desde 0)", list(range(numesc)))
    esc_sel = esc + 1
    results_sin = baseline_esc(esc) if esc in ESCENARIOS_CON_BASELINE else {}
    st.subheader("🕸️ Grafo de la red")

    col_ctrl, col_info = st.columns([3, 1])
    with col_ctrl:
        # Slider 1-based, igual que el campo 'hora' de los resultados y que
        # el eje X de los gráficos (data_loader.horas = range(1, N+1)). Antes
        # iba de 0 a N-1, así que la misma hora física aparecía como "90" acá
        # y como "91" en el gráfico de ENS: una hora de desfase aparente.
        # `key` nuevo a propósito: el valor guardado del slider viejo (0)
        # queda fuera del rango nuevo.
        hora_ui = st.slider("Hora", 1, N_HORAS, 1, key="slider_grafo_inicio_h1")
    with col_info:
        st.caption(f"Hora seleccionada: **{hora_ui}**")

    hora_grafo = hora_ui - 1        # índice 0-based en listas y arrays

    # Estado de fallas para la hora seleccionada
    fmc_hora = {}
    try:
        eidx = min(max(esc_sel - 1, 0), faultlinMC.shape[1] - 1)
        hidx = min(hora_grafo, faultlinMC.shape[0] - 1)
        for li, l in enumerate(LINES):
            if li < faultlinMC.shape[2]:
                fmc_hora[l] = bool(faultlinMC[hidx, eidx, li] > 0)
    except Exception:
        pass

    # Resultados de la hora seleccionada (caso base o estrategia activa)

    res_hora = {}
    # results_sin ahora trae SOLO el escenario seleccionado, con su índice
    # 0-based como clave; el `min(..., len(results_sin)-1)` anterior asumía
    # un dict 0-indexado con todos los escenarios y fallaba salvo esc_sel=1.
    h_list = results_sin.get(esc_sel - 1, [])
    if h_list and hora_grafo < len(h_list):
        res_hora = h_list[hora_grafo]

    # Se pasa la hora 1-based: dentro de build_fig_grafo solo se usa para
    # mostrarla, no para indexar.
    fig_grafo = build_fig_grafo(hora_ui, esc_sel, res_hora, fmc_hora, {}, buses_criticos)
    st.plotly_chart(
        fig_grafo,
        use_container_width=True,
        key="grafo_red_principal",
        config={"scrollZoom": True},
    )

    with st.expander("Barras y líneas de la red"):
        st.subheader("🔌 Red Eléctrica — Buses")
        df_bus_show = net.bus[["name", "vn_kv"]].copy()
        df_bus_show.index.name = "Bus"
        df_bus_show["zona"] = df_bus_show.index.map(lambda b: bus_zone.get(b, "—"))
        df_bus_show.columns = ["Nombre", "Tensión nominal [kV]", "Zona"]
        st.dataframe(df_bus_show, use_container_width=True)

        st.subheader("🔌 Red Eléctrica — Líneas")
        df_line_show = net.line[["name", "from_bus", "to_bus", "length_km",
                                  "r_ohm_per_km", "x_ohm_per_km", "max_i_ka"]].copy()
        df_line_show.columns = ["Nombre", "Bus origen", "Bus destino",
                                 "Long. [km]", "R [Ω/km]", "X [Ω/km]", "I_max [kA]"]
        df_line_show["Candidata switch"] = df_line_show.index.map(lambda l: "✅" if l in sw_cand_list else "—")
        st.dataframe(df_line_show, use_container_width=True)


    # ── Parámetros Monte Carlo ───────────────────────────────
    st.subheader("🎲 Parámetros Monte Carlo")
    col1, col2, col3 = st.columns(3)
    col1.metric("N° Escenarios",  numesc)
    col2.metric("N° Horas",       N_HORAS)
    col3.metric("Tiempo rep. [h]", 3)

    # ── Parámetros económicos ────────────────────────────────
    st.subheader("💰 Parámetros Económicos")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tasa de descuento r", f"{r_econ:.1%}")
    col2.metric("Vida útil n [años]",  n_econ)
    col3.metric("Factor recuperación frp", f"{frp:.4f}")
    voll_mean_display = np.mean(list(VoLL.values())) if VoLL else 0
    col4.metric("VoLL promedio [USD/kWh]", f"{voll_mean_display:.1f}")

    df_voll = pd.DataFrame({"Bus": list(VoLL.keys()), "VoLL [USD/kWh]": list(VoLL.values())})
    with st.expander("VoLL por bus"):
        st.dataframe(df_voll.set_index("Bus"), use_container_width=True)

    # ── Costos de medidas ────────────────────────────────────
    st.subheader("📋 Costos de Medidas de Resiliencia")
    def _scalar(x):
        if isinstance(x, dict):
            return np.mean(list(x.values()))
        return float(x)

    df_costos = pd.DataFrame([
        {"Tecnología": "DG Solar",   "CAPEX [USD/kW]": _scalar(data["DG_sol"][0]), "OPEX [USD/kW·año]": _scalar(data["DG_sol"][1])},
        {"Tecnología": "DG Eólica",  "CAPEX [USD/kW]": _scalar(data["DG_eol"][0]), "OPEX [USD/kW·año]": _scalar(data["DG_eol"][1])},
        {"Tecnología": "DG Fósil",   "CAPEX [USD/kW]": _scalar(data["DG_fos"][0]), "OPEX [USD/kW·año]": _scalar(data["DG_fos"][1])},
        {"Tecnología": "ESS 5h",     "CAPEX [USD/kWh]": _scalar(data["ESS 5h"][0]), "OPEX [USD/kWh·año]": _scalar(data["ESS 5h"][1])},
        {"Tecnología": "ESS 4h",     "CAPEX [USD/kWh]": _scalar(data["ESS 4h"][0]), "OPEX [USD/kWh·año]": _scalar(data["ESS 4h"][1])},
        {"Tecnología": "ESS 6h",     "CAPEX [USD/kWh]": _scalar(data["ESS 6h"][0]), "OPEX [USD/kWh·año]": _scalar(data["ESS 6h"][1])},
        {"Tecnología": "MEG",        "CAPEX [USD/kW]": _scalar(data["MEG"][0]),     "OPEX [USD/kW·año]": _scalar(data["MEG"][1])},
        {"Tecnología": "EDRP (bloque 1)", "CAPEX [USD/kW]": 0, "OPEX [USD/kW·año]": C_incentivo_edrp[0]},
        {"Tecnología": "EDRP (bloque 2)", "CAPEX [USD/kW]": 0, "OPEX [USD/kW·año]": C_incentivo_edrp[1]},
        {"Tecnología": "EDRP (bloque 3)", "CAPEX [USD/kW]": 0, "OPEX [USD/kW·año]": C_incentivo_edrp[2]},
        {"Tecnología": "EDRP (bloque 4)", "CAPEX [USD/kW]": 0, "OPEX [USD/kW·año]": C_incentivo_edrp[3]},
    ])
    st.dataframe(df_costos.set_index("Tecnología"), use_container_width=True)

    # ── Perfil de demanda base ───────────────────────────────
    st.subheader("📈 Perfil de Demanda Base por Bus")
    fig_dem = go.Figure()
    buses_sample = BUSES[:] if len(BUSES) > 8 else BUSES
    n_show = min(N_HORAS, 24)
    for idx_b, b in enumerate(buses_sample):
        color = PALETA[idx_b % len(PALETA)]
        # Perfil real: P_D_base[b] * demand_factor[h, esc=0, bus_idx]
        bi = BUSES.index(b)
        y_perfil = [
            P_D_base.get(b, 0) * demand_factor[h, 0, bi]
            for h in range(n_show)
        ]
        fig_dem.add_trace(go.Scatter(
            x=list(range(1, n_show + 1)),
            y=y_perfil,
            mode="lines", name=f"Bus {b}",
            line=dict(color=color, width=2)
        ))
    fig_dem.update_layout(
        title="Perfil de demanda real [kW] — primeras 24 h (esc. 1)",
        xaxis_title="Hora", yaxis_title="P_D [kW]",
        height=380
    )
    st.plotly_chart(fig_dem, use_container_width=True)
