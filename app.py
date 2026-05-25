"""
app.py
------
Dashboard de Análisis y Diseño de Controladores — Péndulo Invertido sobre Carro
Planta: G(s) = 0.01209 / (0.002846·s² − 0.09678)

Pestaña 1: Análisis (respuesta al impulso, LGR, Bode, Polos/Ceros)
Pestaña 2: Simulación animada estilo CartPole con PIL + HTML base64
           Sistema completo de 4 estados [x, ẋ, θ, θ̇] — RK4
"""

import time
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import control as ct

from graficos import (
    # Planta y matrices de estado
    PLANT, A_SS, B_SS, C_SS, D_SS,
    M_CART, M_PEND, L_PEND, G_GRAV, I_PEND,
    # Colores
    COLOR_MAIN, COLOR_REF, COLORS_OVERLAY,
    # Sistema
    construir_controlador, sistema_lazo_cerrado,
    # Métricas
    calcular_metricas_impulso, calcular_margenes,
    # Respuestas
    respuesta_condicion_inicial_completa,
    # Gráficas de análisis
    fig_condicion_inicial, fig_root_locus, fig_bode, fig_polos_lc,
    # Animación PIL
    frame_a_html,
    # Gráfica temporal
    figura_theta_tiempo,
    # PID
    pid_step,
)

# ===========================================================================
# Configuración de página
# ===========================================================================
st.set_page_config(
    page_title="Diseño de Controladores | Péndulo Invertido",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main, .stApp { background-color: #f6f8fb; }
    h1,h2,h3,h4 { color: #1a2332; }
    p, label, span, div { color: #1a2332; }
    .stButton>button {
        background-color: #2563eb; color: #ffffff;
        border: none; font-weight: 600; border-radius: 6px;
    }
    .stButton>button:hover { background-color: #1d4ed8; color: #ffffff; }
    div[data-testid="stExpander"] {
        background-color: #ffffff; border-radius: 8px; border: 1px solid #e2e8f0;
    }
    div[data-testid="stMetricValue"] { font-size: 1.1rem; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# Valores por defecto de ganancias
# ===========================================================================
DEFAULTS = {
    "Lazo Abierto": {"Kp": 1.0,  "Ki": 0.0, "Kd": 0.0},
    "P":            {"Kp": 30.0, "Ki": 0.0, "Kd": 0.0},
    "PI":           {"Kp": 30.0, "Ki": 5.0, "Kd": 0.0},
    "PD":           {"Kp": 40.0, "Ki": 0.0, "Kd": 3.0},
    "PID":          {"Kp": 50.0, "Ki": 10.0,"Kd": 5.0},
}

# ===========================================================================
# Session State
# ===========================================================================
for ctrl, vals in DEFAULTS.items():
    for k, v in vals.items():
        if f"{ctrl}_{k}" not in st.session_state:
            st.session_state[f"{ctrl}_{k}"] = v

if "controlador_actual" not in st.session_state:
    st.session_state.controlador_actual = "PID"

# Estado simulación
_sim_keys = {
    "sim_running":    False,
    "sim_state":      np.zeros(4),   # [x, ẋ, θ, θ̇]
    "sim_t_hist":     [],
    "sim_theta_hist": [],
    "sim_x_hist":     [],
    "sim_t_actual":   0.0,
    "sim_integral":   0.0,
    "sim_error_prev": 0.0,
    "sim_perturbar":  False,
    "sim_ctrl_tipo":  "PID",
}
for k, v in _sim_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ===========================================================================
# Helpers
# ===========================================================================

def reset_simulacion():
    st.session_state.sim_running    = False
    st.session_state.sim_state      = np.zeros(4)
    st.session_state.sim_t_hist     = []
    st.session_state.sim_theta_hist = []
    st.session_state.sim_x_hist     = []
    st.session_state.sim_t_actual   = 0.0
    st.session_state.sim_integral   = 0.0
    st.session_state.sim_error_prev = 0.0
    st.session_state.sim_perturbar  = False


def aplicar_perturbacion(x_state):
    """Suma 15° al ángulo del péndulo (estado índice 2)."""
    x_new = x_state.copy()
    x_new[2] += np.radians(15.0)
    return x_new


def rk4_step(x, u, dt):
    """Un paso RK4 del sistema completo de 4 estados."""
    def f(xx):
        return (A_SS @ xx.reshape(-1, 1) + B_SS * u).flatten()
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


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
# PESTAÑA 1: ANÁLISIS
# ===========================================================================
with tab_analisis:
    col_izq, col_der = st.columns([1, 3], gap="medium")

    with col_izq:
        st.markdown("#### ⚙️ Configuración")
        tipo = st.radio(
            "Controlador actual",
            ["Lazo Abierto", "P", "PI", "PD", "PID"],
            index=["Lazo Abierto","P","PI","PD","PID"].index(
                st.session_state.controlador_actual),
            horizontal=True, key="radio_ctrl",
        )
        st.session_state.controlador_actual = tipo

        if tipo != "Lazo Abierto":
            st.session_state[f"{tipo}_Kp"] = st.slider(
                "Kp", 0.0, 200.0, float(st.session_state[f"{tipo}_Kp"]), 0.5,
                key=f"slider_Kp_{tipo}")
        else:
            st.info("Lazo abierto: la salida diverge.")

        if tipo in ("PI", "PID"):
            st.session_state[f"{tipo}_Ki"] = st.slider(
                "Ki", 0.0, 200.0, float(st.session_state[f"{tipo}_Ki"]), 0.5,
                key=f"slider_Ki_{tipo}")
        if tipo in ("PD", "PID"):
            st.session_state[f"{tipo}_Kd"] = st.slider(
                "Kd", 0.0, 30.0, float(st.session_state[f"{tipo}_Kd"]), 0.05,
                key=f"slider_Kd_{tipo}")

        t_final = st.slider("Tiempo simulación (s)", 1.0, 20.0, 5.0, 0.5)

        st.markdown("---")
        st.markdown("##### 🗂️ Comparar con:")
        overlays_seleccionados = []
        for c in ["Lazo Abierto","P","PI","PD","PID"]:
            if c == tipo:
                continue
            kp = st.session_state[f"{c}_Kp"]
            ki = st.session_state[f"{c}_Ki"]
            kd = st.session_state[f"{c}_Kd"]
            if st.checkbox(f"{c} (Kp={kp:.1f} Ki={ki:.1f} Kd={kd:.2f})",
                           key=f"chk_{c}"):
                overlays_seleccionados.append(c)

        st.markdown("---")
        if st.button("↺ Restablecer ganancias"):
            for k, v in DEFAULTS[tipo].items():
                st.session_state[f"{tipo}_{k}"] = v
            st.rerun()

    with col_der:
        Kp = st.session_state[f"{tipo}_Kp"]
        Ki = st.session_state[f"{tipo}_Ki"]
        Kd = st.session_state[f"{tipo}_Kd"]
        T_sys, C_sys = sistema_lazo_cerrado(tipo, Kp, Ki, Kd)
        t_arr, y_arr = respuesta_condicion_inicial_completa(tipo, Kp, Ki, Kd, t_final=t_final)
        metricas     = calcular_metricas_impulso(T_sys, t_arr, y_arr)

        sistemas   = [{"nombre": f"{tipo} (actual)", "C": C_sys,
                       "color": COLOR_MAIN, "dash": "solid"}]
        curvas_imp = [{"nombre": f"{tipo} (actual)", "t": t_arr, "y": y_arr,
                       "color": COLOR_MAIN, "dash": "solid"}]

        for idx, ctrl_ov in enumerate(overlays_seleccionados):
            kp_o = st.session_state[f"{ctrl_ov}_Kp"]
            ki_o = st.session_state[f"{ctrl_ov}_Ki"]
            kd_o = st.session_state[f"{ctrl_ov}_Kd"]
            T_o, C_o = sistema_lazo_cerrado(ctrl_ov, kp_o, ki_o, kd_o)
            t_o, y_o = respuesta_condicion_inicial_completa(ctrl_ov, kp_o, ki_o, kd_o, t_final=t_final)
            col_ov = COLORS_OVERLAY[idx % len(COLORS_OVERLAY)]
            sistemas.append({"nombre": ctrl_ov, "C": C_o, "color": col_ov, "dash": "dash"})
            curvas_imp.append({"nombre": ctrl_ov, "t": t_o, "y": y_o,
                               "color": col_ov, "dash": "dash"})

        # Métricas
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        if metricas["estable"]:
            m1.metric("Estado",  "✅ Estable")
            m2.metric("Pico (°)", f"{metricas['pico']:.2f}°")
            m3.metric("t pico",  f"{metricas['t_pico']:.3f} s")
            m4.metric("Ts (2%)", f"{metricas['Ts']:.3f} s")
        else:
            m1.metric("Estado",  "❌ Inestable")
            m2.metric("Pico (°)", "—")
            m3.metric("t pico",  "—")
            m4.metric("Ts (2%)", "—")
        gm_db_a, pm_a, *_ = calcular_margenes(C_sys * PLANT)
        m5.metric("MF", f"{pm_a:.1f}°"      if pm_a   is not None else "—")
        m6.metric("MG", f"{gm_db_a:.1f} dB" if gm_db_a is not None else "—")

        # Diálogos maximizar
        @st.dialog("Respuesta a Condición Inicial (θ₀=15°)", width="large")
        def dlg_imp():
            st.plotly_chart(fig_condicion_inicial(curvas_imp, t_final=t_final, height=620),
                            use_container_width=True)
        @st.dialog("Lugar Geométrico de las Raíces", width="large")
        def dlg_rl():
            st.plotly_chart(fig_root_locus(sistemas, height=620),
                            use_container_width=True)
        @st.dialog("Diagrama de Bode", width="large")
        def dlg_bode():
            f, *_ = fig_bode(sistemas, height=620)
            st.plotly_chart(f, use_container_width=True)
        @st.dialog("Polos y Ceros LC", width="large")
        def dlg_pz():
            st.plotly_chart(fig_polos_lc(sistemas, height=620),
                            use_container_width=True)

        H = 320
        f1a, f1b = st.columns(2)
        f2a, f2b = st.columns(2)

        with f1a:
            c1, c2 = st.columns([5,1])
            c1.markdown("**📈 Respuesta a CI (θ₀=15°)**")
            if c2.button("🔍", key="max_imp"):  dlg_imp()
            st.plotly_chart(fig_condicion_inicial(curvas_imp, t_final=t_final, height=H),
                            use_container_width=True)
        with f1b:
            c1, c2 = st.columns([5,1])
            c1.markdown("**📍 Lugar Geométrico (LGR)**")
            if c2.button("🔍", key="max_rl"):   dlg_rl()
            st.plotly_chart(fig_root_locus(sistemas, height=H),
                            use_container_width=True)
        with f2a:
            c1, c2 = st.columns([5,1])
            c1.markdown("**🎚️ Diagrama de Bode**")
            if c2.button("🔍", key="max_bode"): dlg_bode()
            fb, *_ = fig_bode(sistemas, height=H)
            st.plotly_chart(fb, use_container_width=True)
        with f2b:
            c1, c2 = st.columns([5,1])
            c1.markdown("**📌 Polos y Ceros LC**")
            if c2.button("🔍", key="max_pz"):   dlg_pz()
            st.plotly_chart(fig_polos_lc(sistemas, height=H),
                            use_container_width=True)

        with st.expander("ℹ️ Detalles del sistema actual", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Controlador C(s):**")
                st.code(str(C_sys), language="text")
            with c2:
                st.markdown("**Polos de lazo cerrado:**")
                try:
                    for p in ct.poles(T_sys):
                        icon = "🟢" if p.real < 0 else "🔴"
                        st.code(f"{icon} {p.real:.4f} {p.imag:+.4f}j",
                                language="text")
                except Exception:
                    st.code("—")

# ===========================================================================
# PESTAÑA 2: SIMULACIÓN DEL CARRO (animación PIL estilo CartPole)
# ===========================================================================
with tab_simulacion:
    col_sim_izq, col_sim_der = st.columns([3, 1], gap="medium")

    # ----- Panel derecho: controles -----
    with col_sim_der:
        st.markdown("#### 🎛️ Controlador")
        sim_tipo = st.radio(
            "Tipo", ["P","PI","PD","PID"],
            index=["P","PI","PD","PID"].index(
                st.session_state.sim_ctrl_tipo
                if st.session_state.sim_ctrl_tipo in ("P","PI","PD","PID") else "PID"),
            horizontal=True, key="radio_sim_ctrl",
        )
        st.session_state.sim_ctrl_tipo = sim_tipo

        st.session_state[f"{sim_tipo}_Kp"] = st.slider(
            "Kp", 0.0, 200.0, float(st.session_state[f"{sim_tipo}_Kp"]), 0.5,
            key=f"sim_Kp_{sim_tipo}")
        if sim_tipo in ("PI","PID"):
            st.session_state[f"{sim_tipo}_Ki"] = st.slider(
                "Ki", 0.0, 200.0, float(st.session_state[f"{sim_tipo}_Ki"]), 0.5,
                key=f"sim_Ki_{sim_tipo}")
        if sim_tipo in ("PD","PID"):
            st.session_state[f"{sim_tipo}_Kd"] = st.slider(
                "Kd", 0.0, 30.0, float(st.session_state[f"{sim_tipo}_Kd"]), 0.05,
                key=f"sim_Kd_{sim_tipo}")

        st.markdown("---")
        st.markdown("#### 🎯 Auto-sintonización")
        st.caption("Busca Kp, Ki, Kd óptimos para el controlador actual usando el modelo lineal.")

        if st.button("🎯 Auto-sintonizar", use_container_width=True, key="btn_auto"):
            with st.spinner(f"Optimizando ganancias para {sim_tipo}... (puede tomar 20-40 s)"):
                from graficos import auto_sintonizar
                res = auto_sintonizar(sim_tipo, A_SS, B_SS)
            if res["exito"]:
                st.session_state[f"{sim_tipo}_Kp"] = res["Kp"]
                st.session_state[f"{sim_tipo}_Ki"] = res["Ki"]
                st.session_state[f"{sim_tipo}_Kd"] = res["Kd"]
                st.success(f"✅ Kp={res['Kp']} | Ki={res['Ki']} | Kd={res['Kd']} | Ts={res['Ts']} s")
                st.rerun()
            else:
                st.error("❌ No se encontró solución estable. Intenta con otro tipo de controlador.")

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

        if st.button("⚡ Perturbar +15°", use_container_width=True, key="btn_perturb"):
            st.session_state.sim_perturbar = True
            if not st.session_state.sim_running:
                st.rerun()

        st.markdown("---")
        st.markdown("##### 📊 Estado")
        if len(st.session_state.sim_theta_hist) > 0:
            theta_act = st.session_state.sim_theta_hist[-1]
            x_act     = st.session_state.sim_x_hist[-1]
            st.metric("θ actual", f"{theta_act:.2f}°")
            st.metric("x carro",  f"{x_act:.3f} m")
            st.metric("Tiempo",   f"{st.session_state.sim_t_actual:.2f} s")
            if abs(theta_act) > 30:
                st.error("⚠ CAÍDA (>30°)")
            elif abs(theta_act) > 15:
                st.warning("⚠ Fuera zona lineal")
            else:
                st.success("✓ Zona lineal")
        else:
            st.metric("θ actual", "0.00°")
            st.metric("x carro",  "0.000 m")
            st.metric("Tiempo",   "0.00 s")

        with st.expander("⚙️ Parámetros físicos", expanded=False):
            st.code(
                f"M (carro)   = {M_CART} kg\n"
                f"m (péndulo) = {M_PEND} kg\n"
                f"L (varilla) = {L_PEND} m\n"
                f"I (pivote)  = {I_PEND:.6f} kg·m²\n"
                f"g           = {G_GRAV} m/s²",
                language="text")

    # ----- Panel izquierdo: animación + gráfica -----
    with col_sim_izq:
        # Placeholder para la imagen PIL (HTML base64 — sin parpadeo)
        ph_anim   = st.empty()
        ph_grafica = st.empty()

        # Aplicar perturbación si fue solicitada y la sim está parada
        if st.session_state.sim_perturbar and not st.session_state.sim_running:
            st.session_state.sim_state    = aplicar_perturbacion(
                st.session_state.sim_state.copy())
            st.session_state.sim_perturbar = False

        # Mostrar estado actual (estático)
        x_st   = st.session_state.sim_state
        x_vis0 = float(x_st[0])
        th_vis0 = float(x_st[2])
        ph_anim.markdown(frame_a_html(x_vis0, th_vis0), unsafe_allow_html=True)
        ph_grafica.plotly_chart(
            figura_theta_tiempo(
                st.session_state.sim_t_hist,
                st.session_state.sim_theta_hist,
                st.session_state.sim_x_hist,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # ---- LOOP DE SIMULACIÓN ----
        if st.session_state.sim_running:
            DT              = 0.033   # ~30 fps objetivo
            N_SUB           = 4       # subpasos RK4 por frame
            dt_int          = DT / N_SUB
            FRAMES_X_BLOQUE = 12      # frames por bloque antes de rerun

            Kp_s = st.session_state[f"{sim_tipo}_Kp"]
            Ki_s = st.session_state[f"{sim_tipo}_Ki"]
            Kd_s = st.session_state[f"{sim_tipo}_Kd"]

            x_state    = st.session_state.sim_state.copy().astype(float)
            integral   = float(st.session_state.sim_integral)
            error_prev = float(st.session_state.sim_error_prev)

            for frame_i in range(FRAMES_X_BLOQUE):
                # Perturbación pendiente
                if st.session_state.sim_perturbar:
                    x_state = aplicar_perturbacion(x_state)
                    st.session_state.sim_perturbar = False

                # Extraer θ y x del estado completo
                x_carro = x_state[0]   # posición carro [m]
                theta   = x_state[2]   # ángulo péndulo [rad]
                error   = theta         # u>0 corrige theta>0 (B[3,0]<0)

                # Acción de control PID
                u, integral, _ = pid_step(
                    error, dt_int, Kp_s, Ki_s, Kd_s,
                    sim_tipo, integral, error_prev)
                error_prev = error

                # Integración RK4 (N_SUB subpasos)
                for _ in range(N_SUB):
                    x_state = rk4_step(x_state, u, dt_int)

                # Avanzar tiempo
                st.session_state.sim_t_actual += DT

                # Guardar historial
                theta_deg = np.degrees(x_state[2])
                x_m       = x_state[0]
                st.session_state.sim_t_hist.append(st.session_state.sim_t_actual)
                st.session_state.sim_theta_hist.append(theta_deg)
                st.session_state.sim_x_hist.append(x_m)

                # Limitar a últimos 30 s (~900 muestras a 30 fps)
                if len(st.session_state.sim_t_hist) > 900:
                    st.session_state.sim_t_hist     = st.session_state.sim_t_hist[-700:]
                    st.session_state.sim_theta_hist = st.session_state.sim_theta_hist[-700:]
                    st.session_state.sim_x_hist     = st.session_state.sim_x_hist[-700:]

                # Condición de parada: caída total o carro fuera de rango
                if abs(theta_deg) > 90:
                    st.session_state.sim_running = False
                    break

                # Renderizar frame (PIL → base64 → HTML, sin parpadeo)
                ph_anim.markdown(
                    frame_a_html(x_state[0], x_state[2]),
                    unsafe_allow_html=True,
                )

                # Actualizar gráfica cada 3 frames para no sobrecargar
                if frame_i % 3 == 0:
                    ph_grafica.plotly_chart(
                        figura_theta_tiempo(
                            st.session_state.sim_t_hist,
                            st.session_state.sim_theta_hist,
                            st.session_state.sim_x_hist,
                        ),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

                time.sleep(max(0, DT - 0.005))

            # Guardar estados del bloque
            st.session_state.sim_state      = x_state
            st.session_state.sim_integral   = integral
            st.session_state.sim_error_prev = error_prev

            if st.session_state.sim_running:
                st.rerun()