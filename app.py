"""
app.py
------
Dashboard de Análisis y Diseño de Controladores — Péndulo Invertido sobre Carro
Planta: G(s) = 0.01209 / (0.002846·s² − 0.09678)

Estructura:
  - Pestaña 1: Análisis y Diseño  (respuesta al impulso + LGR + Bode + Polos)
  - Pestaña 2: Simulación del Carro (dinámica en tiempo real con controlador PID)

Dependencias externas: graficos.py (debe estar en el mismo directorio)
"""

import time
import numpy as np
import streamlit as st
import streamlit.components.v1 as components  # noqa: F401  (disponible por si se necesita)
import control as ct

# ---------------------------------------------------------------------------
# Importar funciones de graficos.py
# ---------------------------------------------------------------------------
from graficos import (
    # Datos de la planta
    PLANT, A_SS, B_SS, C_SS, D_SS, M_CART, M_PEND, L_PEND,
    COLOR_MAIN, COLOR_REF, COLORS_OVERLAY,
    # Construcción del sistema
    construir_controlador, sistema_lazo_cerrado,
    # Métricas
    calcular_metricas_impulso, calcular_margenes,
    # Respuestas
    respuesta_impulso,
    # Gráficas de análisis
    fig_impulso, fig_root_locus, fig_bode, fig_polos_lc,
    # Gráficas de simulación
    figura_carro_pendulo, figura_theta_tiempo,
    # Lógica de control
    pid_step,
)

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diseño de Controladores | Péndulo Invertido",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .main { background-color: #f6f8fb; }
        .stApp { background-color: #f6f8fb; }
        h1, h2, h3, h4 { color: #1a2332; }
        p, label, span, div { color: #1a2332; }
        .metric-card {
            background-color: #ffffff;
            border-left: 4px solid #2563eb;
            padding: 10px 14px;
            border-radius: 6px;
            margin-bottom: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            font-size: 0.9rem;
        }
        .stButton>button {
            background-color: #2563eb;
            color: #ffffff;
            border: none;
            font-weight: 600;
            border-radius: 6px;
        }
        .stButton>button:hover { background-color: #1d4ed8; color: #ffffff; }
        div[data-testid="stExpander"] {
            background-color: #ffffff;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }
        div[data-testid="stMetricValue"] { font-size: 1.1rem; }
        div[data-testid="stMetricLabel"] { font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Valores por defecto de las ganancias
# ---------------------------------------------------------------------------
DEFAULTS = {
    "Lazo Abierto": {"Kp": 1.0,  "Ki": 0.0, "Kd": 0.0},
    "P":            {"Kp": 30.0, "Ki": 0.0, "Kd": 0.0},
    "PI":           {"Kp": 30.0, "Ki": 5.0, "Kd": 0.0},
    "PD":           {"Kp": 40.0, "Ki": 0.0, "Kd": 3.0},
    "PID":          {"Kp": 50.0, "Ki": 10.0,"Kd": 5.0},
}

# ---------------------------------------------------------------------------
# Inicialización de Session State
# ---------------------------------------------------------------------------
for ctrl, vals in DEFAULTS.items():
    for k, v in vals.items():
        key = f"{ctrl}_{k}"
        if key not in st.session_state:
            st.session_state[key] = v

if "controlador_actual" not in st.session_state:
    st.session_state.controlador_actual = "PID"

# Estado de la simulación dinámica del carro
if "sim_running"    not in st.session_state:
    st.session_state.sim_running    = False
if "sim_state"      not in st.session_state:
    st.session_state.sim_state      = np.zeros(4)
if "sim_t_hist"     not in st.session_state:
    st.session_state.sim_t_hist     = []
if "sim_theta_hist" not in st.session_state:
    st.session_state.sim_theta_hist = []
if "sim_t_actual"   not in st.session_state:
    st.session_state.sim_t_actual   = 0.0
if "sim_integral"   not in st.session_state:
    st.session_state.sim_integral   = 0.0
if "sim_error_prev" not in st.session_state:
    st.session_state.sim_error_prev = 0.0
if "sim_perturbar"  not in st.session_state:
    st.session_state.sim_perturbar  = False
if "sim_ctrl_tipo"  not in st.session_state:
    st.session_state.sim_ctrl_tipo  = "PID"


# ---------------------------------------------------------------------------
# Helpers de simulación
# ---------------------------------------------------------------------------

def reset_simulacion():
    """Reinicia el estado interno de la simulación del carro."""
    st.session_state.sim_running    = False
    st.session_state.sim_state      = np.zeros(4)
    st.session_state.sim_t_hist     = []
    st.session_state.sim_theta_hist = []
    st.session_state.sim_t_actual   = 0.0
    st.session_state.sim_integral   = 0.0
    st.session_state.sim_error_prev = 0.0
    st.session_state.sim_perturbar  = False


def aplicar_perturbacion(x_state):
    """
    Suma ~15° al ángulo del péndulo usando la pseudo-inversa de C_SS.
    Devuelve el estado modificado.
    """
    try:
        dtheta  = np.radians(15.0)
        delta_x = np.linalg.pinv(C_SS) @ np.array([[dtheta]])
        x_state = x_state + delta_x.flatten()
    except Exception:
        pass
    return x_state


# ===========================================================================
# ENCABEZADO
# ===========================================================================
st.markdown(
    "<h3 style='margin-bottom:0;'>🎛️ Análisis y Diseño de Controladores</h3>"
    "<p style='color:#64748b;margin-top:0;font-size:0.9rem;'>Planta: Péndulo Invertido — "
    "<code>G(s) = 0.01209 / (0.002846·s² − 0.09678)</code> · "
    "<span style='color:#dc2626;'>Inestable en lazo abierto</span></p>",
    unsafe_allow_html=True,
)

tab_analisis, tab_simulacion = st.tabs(["📊 Análisis y Diseño", "🎮 Simulación del Carro"])

# ===========================================================================
# PESTAÑA 1: ANÁLISIS Y DISEÑO
# ===========================================================================
with tab_analisis:
    col_izq, col_der = st.columns([1, 3], gap="medium")

    # ----- Panel izquierdo: selección de controlador y ganancias -----
    with col_izq:
        st.markdown("#### ⚙️ Configuración")

        tipo = st.radio(
            "Controlador actual",
            ["Lazo Abierto", "P", "PI", "PD", "PID"],
            index=["Lazo Abierto", "P", "PI", "PD", "PID"].index(
                st.session_state.controlador_actual
            ),
            horizontal=True,
            key="radio_ctrl",
        )
        st.session_state.controlador_actual = tipo

        if tipo != "Lazo Abierto":
            st.session_state[f"{tipo}_Kp"] = st.slider(
                "Kp", 0.0, 200.0, float(st.session_state[f"{tipo}_Kp"]), 0.5,
                key=f"slider_Kp_{tipo}",
            )
        else:
            st.info("Lazo abierto: la salida diverge.")

        if tipo in ("PI", "PID"):
            st.session_state[f"{tipo}_Ki"] = st.slider(
                "Ki", 0.0, 200.0, float(st.session_state[f"{tipo}_Ki"]), 0.5,
                key=f"slider_Ki_{tipo}",
            )
        if tipo in ("PD", "PID"):
            st.session_state[f"{tipo}_Kd"] = st.slider(
                "Kd", 0.0, 30.0, float(st.session_state[f"{tipo}_Kd"]), 0.05,
                key=f"slider_Kd_{tipo}",
            )

        t_final = st.slider("Tiempo simulación (s)", 1.0, 20.0, 5.0, 0.5)

        st.markdown("---")
        st.markdown("##### 🗂️ Comparar con:")
        overlays_seleccionados = []
        for c in ["Lazo Abierto", "P", "PI", "PD", "PID"]:
            if c == tipo:
                continue
            kp = st.session_state[f"{c}_Kp"]
            ki = st.session_state[f"{c}_Ki"]
            kd = st.session_state[f"{c}_Kd"]
            label = f"{c} (Kp={kp:.1f} Ki={ki:.1f} Kd={kd:.2f})"
            if st.checkbox(label, key=f"chk_{c}"):
                overlays_seleccionados.append(c)

        st.markdown("---")
        if st.button("↺ Restablecer ganancias"):
            for k, v in DEFAULTS[tipo].items():
                st.session_state[f"{tipo}_{k}"] = v
            st.rerun()

    # ----- Panel derecho: gráficas y métricas -----
    with col_der:
        Kp = st.session_state[f"{tipo}_Kp"]
        Ki = st.session_state[f"{tipo}_Ki"]
        Kd = st.session_state[f"{tipo}_Kd"]

        T_sys, C_sys = sistema_lazo_cerrado(tipo, Kp, Ki, Kd)

        # Respuesta al impulso del sistema actual
        t_arr, y_arr = respuesta_impulso(T_sys, t_final=t_final)
        metricas     = calcular_metricas_impulso(T_sys, t_arr, y_arr)

        # Lista de sistemas para gráficas multi-curva
        sistemas    = [{"nombre": f"{tipo} (actual)", "C": C_sys,
                        "color": COLOR_MAIN, "dash": "solid"}]
        curvas_imp  = [{"nombre": f"{tipo} (actual)", "t": t_arr, "y": y_arr,
                        "color": COLOR_MAIN, "dash": "solid"}]

        for idx, ctrl_ov in enumerate(overlays_seleccionados):
            kp_o = st.session_state[f"{ctrl_ov}_Kp"]
            ki_o = st.session_state[f"{ctrl_ov}_Ki"]
            kd_o = st.session_state[f"{ctrl_ov}_Kd"]
            T_o, C_o = sistema_lazo_cerrado(ctrl_ov, kp_o, ki_o, kd_o)
            t_o, y_o = respuesta_impulso(T_o, t_final=t_final)
            color_ov = COLORS_OVERLAY[idx % len(COLORS_OVERLAY)]
            sistemas.append({"nombre": ctrl_ov, "C": C_o,
                              "color": color_ov, "dash": "dash"})
            curvas_imp.append({"nombre": ctrl_ov, "t": t_o, "y": y_o,
                               "color": color_ov, "dash": "dash"})

        # ---- Métricas al impulso ----
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        if metricas["estable"]:
            m1.metric("Estado",   "✅ Estable")
            m2.metric("Pico",     f"{metricas['pico']:.4f}")
            m3.metric("t pico",   f"{metricas['t_pico']:.3f} s")
            m4.metric("Ts (2%)",  f"{metricas['Ts']:.3f} s")
        else:
            m1.metric("Estado",  "❌ Inestable")
            m2.metric("Pico",    "—")
            m3.metric("t pico",  "—")
            m4.metric("Ts (2%)", "—")

        L_actual = C_sys * PLANT
        gm_db_a, pm_a, wcg_a, wcp_a = calcular_margenes(L_actual)
        m5.metric("MF", f"{pm_a:.1f}°"    if pm_a   is not None else "—")
        m6.metric("MG", f"{gm_db_a:.1f} dB" if gm_db_a is not None else "—")

        # ---- Diálogos para maximizar ----
        @st.dialog("Respuesta al Impulso", width="large")
        def dlg_imp():
            st.plotly_chart(
                fig_impulso(curvas_imp, t_final=t_final, height=620),
                use_container_width=True,
            )

        @st.dialog("Lugar Geométrico de las Raíces", width="large")
        def dlg_rl():
            st.plotly_chart(
                fig_root_locus(sistemas, height=620),
                use_container_width=True,
            )

        @st.dialog("Diagrama de Bode", width="large")
        def dlg_bode():
            f, *_ = fig_bode(sistemas, height=620)
            st.plotly_chart(f, use_container_width=True)

        @st.dialog("Polos y Ceros LC", width="large")
        def dlg_pz():
            st.plotly_chart(
                fig_polos_lc(sistemas, height=620),
                use_container_width=True,
            )

        H = 320

        # ---- Fila 1: Respuesta al Impulso + LGR ----
        fila1_a, fila1_b = st.columns(2)
        fila2_a, fila2_b = st.columns(2)

        with fila1_a:
            bc1, bc2 = st.columns([5, 1])
            bc1.markdown("**📈 Respuesta al Impulso**")
            if bc2.button("🔍", key="max_imp", help="Maximizar"):
                dlg_imp()
            st.plotly_chart(
                fig_impulso(curvas_imp, t_final=t_final, height=H),
                use_container_width=True,
            )

        with fila1_b:
            bc1, bc2 = st.columns([5, 1])
            bc1.markdown("**📍 Lugar Geométrico (LGR)**")
            if bc2.button("🔍", key="max_rl", help="Maximizar"):
                dlg_rl()
            st.plotly_chart(
                fig_root_locus(sistemas, height=H),
                use_container_width=True,
            )

        # ---- Fila 2: Bode + Polos y Ceros ----
        with fila2_a:
            bc1, bc2 = st.columns([5, 1])
            bc1.markdown("**🎚️ Diagrama de Bode**")
            if bc2.button("🔍", key="max_bode", help="Maximizar"):
                dlg_bode()
            fig_b, *_ = fig_bode(sistemas, height=H)
            st.plotly_chart(fig_b, use_container_width=True)

        with fila2_b:
            bc1, bc2 = st.columns([5, 1])
            bc1.markdown("**📌 Polos y Ceros LC**")
            if bc2.button("🔍", key="max_pz", help="Maximizar"):
                dlg_pz()
            st.plotly_chart(
                fig_polos_lc(sistemas, height=H),
                use_container_width=True,
            )

        # ---- Detalle del sistema ----
        with st.expander("ℹ️ Detalles del sistema actual", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Controlador C(s):**")
                st.code(str(C_sys), language="text")
            with c2:
                st.markdown("**Polos de lazo cerrado:**")
                try:
                    polos_t = ct.poles(T_sys)
                    for p in polos_t:
                        icon = "🟢" if p.real < 0 else "🔴"
                        st.code(f"{icon} {p.real:.4f} {p.imag:+.4f}j",
                                language="text")
                except Exception:
                    st.code("—", language="text")


# ===========================================================================
# PESTAÑA 2: SIMULACIÓN DEL CARRO
# ===========================================================================
with tab_simulacion:
    col_sim_izq, col_sim_der = st.columns([3, 1], gap="medium")

    # ----- Panel derecho: configuración del controlador -----
    with col_sim_der:
        st.markdown("#### 🎛️ Controlador")

        sim_tipo = st.radio(
            "Tipo",
            ["P", "PI", "PD", "PID"],
            index=["P", "PI", "PD", "PID"].index(
                st.session_state.sim_ctrl_tipo
                if st.session_state.sim_ctrl_tipo in ("P", "PI", "PD", "PID")
                else "PID"
            ),
            horizontal=True,
            key="radio_sim_ctrl",
        )
        st.session_state.sim_ctrl_tipo = sim_tipo

        # Sliders sincronizados con los de la pestaña de análisis
        st.session_state[f"{sim_tipo}_Kp"] = st.slider(
            "Kp", 0.0, 200.0, float(st.session_state[f"{sim_tipo}_Kp"]), 0.5,
            key=f"sim_slider_Kp_{sim_tipo}",
        )
        if sim_tipo in ("PI", "PID"):
            st.session_state[f"{sim_tipo}_Ki"] = st.slider(
                "Ki", 0.0, 200.0, float(st.session_state[f"{sim_tipo}_Ki"]), 0.5,
                key=f"sim_slider_Ki_{sim_tipo}",
            )
        if sim_tipo in ("PD", "PID"):
            st.session_state[f"{sim_tipo}_Kd"] = st.slider(
                "Kd", 0.0, 30.0, float(st.session_state[f"{sim_tipo}_Kd"]), 0.05,
                key=f"sim_slider_Kd_{sim_tipo}",
            )

        st.markdown("---")
        st.markdown("#### ▶️ Controles")

        cb1, cb2 = st.columns(2)
        if not st.session_state.sim_running:
            if cb1.button("▶ Iniciar", use_container_width=True, key="btn_start"):
                st.session_state.sim_running = True
                st.rerun()
        else:
            if cb1.button("■ Detener", use_container_width=True, key="btn_stop"):
                st.session_state.sim_running = False
                st.rerun()

        if cb2.button("↺ Reset", use_container_width=True, key="btn_reset"):
            reset_simulacion()
            st.rerun()

        if st.button("⚡ Perturbar +15°", use_container_width=True,
                     key="btn_perturb", help="Suma 15° al ángulo actual"):
            st.session_state.sim_perturbar = True
            if not st.session_state.sim_running:
                st.rerun()

        st.markdown("---")
        st.markdown("##### 📊 Estado")
        if len(st.session_state.sim_theta_hist) > 0:
            theta_act = st.session_state.sim_theta_hist[-1]
            st.metric("θ actual", f"{theta_act:.2f}°")
            st.metric("Tiempo",   f"{st.session_state.sim_t_actual:.2f} s")
            if abs(theta_act) > 20:
                st.error("⚠ Fuera de zona lineal (>20°)")
            elif abs(theta_act) > 15:
                st.warning("⚠ Cerca del límite lineal")
            else:
                st.success("✓ Dentro de zona lineal")
        else:
            st.metric("θ actual", "0.00°")
            st.metric("Tiempo",   "0.00 s")

        with st.expander("⚙️ Parámetros físicos", expanded=False):
            st.code(
                f"M (carro)   = {M_CART} kg\n"
                f"m (péndulo) = {M_PEND} kg\n"
                f"L (varilla) = {L_PEND} m",
                language="text",
            )

    # ----- Panel izquierdo: visualización + θ(t) -----
    with col_sim_izq:
        ph_carro   = st.empty()
        ph_grafica = st.empty()

        # Aplicar perturbación pendiente (cuando la sim está detenida)
        if st.session_state.sim_perturbar and not st.session_state.sim_running:
            st.session_state.sim_state = aplicar_perturbacion(
                st.session_state.sim_state.copy()
            )
            st.session_state.sim_perturbar = False

        # Config común de Plotly: sin barra de modo para reducir el coste de render.
        PLOTLY_CFG = {"staticPlot": False, "displayModeBar": False}

        # ---- AVANCE DE SIMULACIÓN (un bloque por ejecución del script) ----
        # Cada ejecución hace UN solo render con keys fijas. Esto permite que
        # Streamlit reconozca los mismos widgets entre reruns sucesivos y
        # actualice los datos en lugar de destruir/recrear el componente
        # (que es lo que producía el parpadeo). Las keys fijas no se pueden
        # repetir dentro del MISMO script run, así que el loop interno
        # desaparece: avanzamos varios pasos de simulación, renderizamos una
        # vez, dormimos y disparamos st.rerun() para el siguiente frame.
        if st.session_state.sim_running:
            DT          = 0.04   # paso de avance de tiempo de simulación
            N_SUB       = 4      # subpasos RK4 por paso (dt_int = 0.01 s)
            dt_int      = DT / N_SUB
            STEPS_RERUN = 2      # 2 pasos de sim por render visual (~2× speed)

            Kp_s = st.session_state[f"{sim_tipo}_Kp"]
            Ki_s = st.session_state[f"{sim_tipo}_Ki"]
            Kd_s = st.session_state[f"{sim_tipo}_Kd"]

            x_state    = st.session_state.sim_state.copy().astype(float)
            integral   = st.session_state.sim_integral
            error_prev = st.session_state.sim_error_prev

            for _ in range(STEPS_RERUN):
                # Aplicar perturbación pendiente si la hubo durante el bloque
                if st.session_state.sim_perturbar:
                    x_state = aplicar_perturbacion(x_state)
                    st.session_state.sim_perturbar = False

                # Extraer estados directamente del vector de 4 componentes
                theta = float(x_state[2])
                error = 0.0 - theta   # referencia = 0 (vertical)

                # Acción de control
                u, integral, _ = pid_step(
                    error, dt_int, Kp_s, Ki_s, Kd_s, sim_tipo,
                    integral, error_prev
                )
                error_prev = error   # actualizar estado del derivador

                # Integración Runge-Kutta 4
                for _sub in range(N_SUB):
                    k1 = (A_SS @ x_state.reshape(-1, 1) + B_SS * u).flatten()
                    k2 = (A_SS @ (x_state + 0.5 * dt_int * k1).reshape(-1, 1) + B_SS * u).flatten()
                    k3 = (A_SS @ (x_state + 0.5 * dt_int * k2).reshape(-1, 1) + B_SS * u).flatten()
                    k4 = (A_SS @ (x_state + dt_int * k3).reshape(-1, 1) + B_SS * u).flatten()
                    x_state = x_state + (dt_int / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

                # Avanzar tiempo e historial
                st.session_state.sim_t_actual += DT
                theta_grados = np.degrees(theta)
                st.session_state.sim_t_hist.append(st.session_state.sim_t_actual)
                st.session_state.sim_theta_hist.append(theta_grados)

                # Limitar historial a ~30 s para no consumir RAM
                if len(st.session_state.sim_t_hist) > 1000:
                    st.session_state.sim_t_hist     = st.session_state.sim_t_hist[-800:]
                    st.session_state.sim_theta_hist = st.session_state.sim_theta_hist[-800:]

                # Detener si el péndulo cae completamente
                if abs(theta_grados) > 90:
                    st.session_state.sim_running = False
                    break

            # Persistir el estado del bloque
            st.session_state.sim_state      = x_state
            st.session_state.sim_integral   = integral
            st.session_state.sim_error_prev = error_prev

        # ---- RENDER ÚNICO (keys fijas → Streamlit reutiliza los componentes
        #      entre reruns y elimina el parpadeo) ----
        x_carro_now = float(st.session_state.sim_state[0])
        theta_now   = float(st.session_state.sim_state[2])

        ph_carro.plotly_chart(
            figura_carro_pendulo(theta_now, x_carro_now),
            use_container_width=True,
            key="carro_live",
            config=PLOTLY_CFG,
        )
        ph_grafica.plotly_chart(
            figura_theta_tiempo(
                st.session_state.sim_t_hist,
                st.session_state.sim_theta_hist,
            ),
            use_container_width=True,
            key="theta_live",
            config=PLOTLY_CFG,
        )

        # ---- Si seguimos corriendo, programar el siguiente frame ----
        if st.session_state.sim_running:
            time.sleep(0.04)   # ≈ 25 fps visual; sim avanza ~2× tiempo real
            st.rerun()