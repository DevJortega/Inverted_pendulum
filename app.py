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

from control_fisico import ClienteESP32, FrameTel

from graficos import (
    # Planta y matrices de estado
    PLANT, A_SS, B_SS, C_SS, D_SS,
    M_CART, M_PEND, L_PEND, G_GRAV, I_PEND,
    # Colores
    COLOR_MAIN, COLOR_REF, COLORS_OVERLAY,
    # Sistema
    construir_controlador, sistema_lazo_cerrado,
    TIPOS_COMPENSADOR,
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
    # Diseño de compensadores
    disenar_compensador_adelanto,
    disenar_compensador_atraso,
    disenar_compensador_adelanto_atraso,
    discretizar_compensador,
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
    .btn-inactive {
        opacity: 0.4 !important;
        pointer-events: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# Valores por defecto de ganancias
# ===========================================================================
DEFAULTS = {
    "Lazo Abierto": {"Kp": 1.0,  "Ki": 0.0,  "Kd": 0.0},
    "P":            {"Kp": 30.0, "Ki": 0.0,  "Kd": 0.0},
    "PI":           {"Kp": 30.0, "Ki": 5.0,  "Kd": 0.0},
    "PD":           {"Kp": 40.0, "Ki": 0.0,  "Kd": 3.0},
    "PID":          {"Kp": 50.0, "Ki": 10.0, "Kd": 5.0},
    # Compensadores: Kp=T, Ki=alpha/beta/T2, Kd=Kc/beta
    "Adelanto":     {"Kp": 0.1,  "Ki": 0.1,  "Kd": 50.0},
    "Atraso":       {"Kp": 2.0,  "Ki": 5.0,  "Kd": 50.0},
    "Atr-Adel":     {"Kp": 0.1,  "Ki": 2.0,  "Kd": 5.0},
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
    "sim_comp_state": None,          # estado del compensador (numpy array o None)
    "comp_diseno_texto": None,       # texto resultado diseño automático
}
for k, v in _sim_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "esp32" not in st.session_state:
    st.session_state.esp32 = ClienteESP32()
if "fisico_running" not in st.session_state:
    st.session_state.fisico_running = False
if "fisico_t_hist" not in st.session_state:
    st.session_state.fisico_t_hist = []
if "fisico_theta_hist" not in st.session_state:
    st.session_state.fisico_theta_hist = []
if "fisico_t0" not in st.session_state:
    st.session_state.fisico_t0 = None

if "ctrl_tipo_global" not in st.session_state:
    st.session_state.ctrl_tipo_global = "PID"
if "ctrl_Kp_global" not in st.session_state:
    st.session_state.ctrl_Kp_global = 50.0
if "ctrl_Ki_global" not in st.session_state:
    st.session_state.ctrl_Ki_global = 10.0
if "ctrl_Kd_global" not in st.session_state:
    st.session_state.ctrl_Kd_global = 5.0


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
    st.session_state.sim_comp_state = None


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


# Callbacks de sincronización: on_change actualiza el global ANTES del rerun,
# para que el "set-before" de los demás sliders lea el valor correcto.
def _sync_tipo_analisis():
    v = st.session_state.radio_ctrl
    if v != "Lazo Abierto":
        st.session_state.ctrl_tipo_global = v

def _sync_tipo_sim():
    st.session_state.ctrl_tipo_global = st.session_state.ctrl_tipo_sim

def _sync_tipo_fisico():
    st.session_state.ctrl_tipo_global = st.session_state.ctrl_tipo_fisico

def _sync_Kp_analisis():
    st.session_state.ctrl_Kp_global = st.session_state.ctrl_Kp_analisis

def _sync_Kp_sim():
    st.session_state.ctrl_Kp_global = st.session_state.ctrl_Kp_sim

def _sync_Kp_fisico():
    st.session_state.ctrl_Kp_global = st.session_state.ctrl_Kp_fisico

def _sync_Ki_analisis():
    st.session_state.ctrl_Ki_global = st.session_state.ctrl_Ki_analisis

def _sync_Ki_sim():
    st.session_state.ctrl_Ki_global = st.session_state.ctrl_Ki_sim

def _sync_Ki_fisico():
    st.session_state.ctrl_Ki_global = st.session_state.ctrl_Ki_fisico

def _sync_Kd_analisis():
    st.session_state.ctrl_Kd_global = st.session_state.ctrl_Kd_analisis

def _sync_Kd_sim():
    st.session_state.ctrl_Kd_global = st.session_state.ctrl_Kd_sim

def _sync_Kd_fisico():
    st.session_state.ctrl_Kd_global = st.session_state.ctrl_Kd_fisico


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

tab_analisis, tab_simulacion, tab_fisico = st.tabs(["📊 Análisis y Diseño", "🎮 Simulación del Carro", "🔌 Control Físico"])

# ===========================================================================
# PESTAÑA 1: ANÁLISIS
# ===========================================================================
with tab_analisis:
    col_izq, col_der = st.columns([1, 3], gap="medium")

    with col_izq:
        st.markdown("#### ⚙️ Configuración")
        _opts_a = ["Lazo Abierto", "P", "PI", "PD", "PID", "Adelanto", "Atraso", "Atr-Adel"]
        if st.session_state.ctrl_tipo_global in _opts_a:
            st.session_state.radio_ctrl = st.session_state.ctrl_tipo_global
        tipo = st.radio(
            "Controlador actual",
            _opts_a,
            horizontal=True, key="radio_ctrl",
            on_change=_sync_tipo_analisis,
        )
        st.session_state.controlador_actual = tipo

        if tipo == "Lazo Abierto":
            st.info("Lazo abierto: la salida diverge.")
        elif tipo in TIPOS_COMPENSADOR:
            # ---- Sliders para compensadores ----
            lbl1 = "T₁" if tipo == "Atr-Adel" else "T"
            st.session_state.ctrl_Kp_analisis = min(10.0, max(0.001, float(st.session_state.ctrl_Kp_global)))
            st.session_state.ctrl_Kp_global = st.slider(
                lbl1, 0.001, 10.0, step=0.001,
                key="ctrl_Kp_analisis", on_change=_sync_Kp_analisis)
            st.session_state[f"{tipo}_Kp"] = st.session_state.ctrl_Kp_global

            if tipo == "Adelanto":
                lbl2, mn2, mx2, stp2 = "α", 0.01, 0.99, 0.01
            elif tipo == "Atraso":
                lbl2, mn2, mx2, stp2 = "β", 1.01, 20.0, 0.01
            else:
                lbl2, mn2, mx2, stp2 = "T₂", 0.01, 100.0, 0.01
            st.session_state.ctrl_Ki_analisis = min(mx2, max(mn2, float(st.session_state.ctrl_Ki_global)))
            st.session_state.ctrl_Ki_global = st.slider(
                lbl2, mn2, mx2, step=stp2,
                key="ctrl_Ki_analisis", on_change=_sync_Ki_analisis)
            st.session_state[f"{tipo}_Ki"] = st.session_state.ctrl_Ki_global

            if tipo == "Atr-Adel":
                lbl3, mn3, mx3, stp3 = "β", 1.01, 20.0, 0.01
            else:
                lbl3, mn3, mx3, stp3 = "Kc", 0.1, 100.0, 0.1
            st.session_state.ctrl_Kd_analisis = min(mx3, max(mn3, float(st.session_state.ctrl_Kd_global)))
            st.session_state.ctrl_Kd_global = st.slider(
                lbl3, mn3, mx3, step=stp3,
                key="ctrl_Kd_analisis", on_change=_sync_Kd_analisis)
            st.session_state[f"{tipo}_Kd"] = st.session_state.ctrl_Kd_global

            # ---- Diseño automático ----
            st.markdown("---")
            st.markdown("##### 🔧 Diseño automático")
            pm_des_a = st.number_input("PM deseado (°)", 30.0, 80.0, 45.0, step=1.0,
                                       key="pm_deseado_analisis")
            if st.button("🔧 Diseñar", use_container_width=True, key="btn_diseno_analisis"):
                try:
                    if tipo == "Adelanto":
                        p, txt = disenar_compensador_adelanto(PLANT, pm_des_a, Kc=1.0)
                        st.session_state.ctrl_Kp_global = p["T"]
                        st.session_state.ctrl_Ki_global = p["alpha"]
                        st.session_state.ctrl_Kd_global = p["Kc"]
                    elif tipo == "Atraso":
                        p, txt = disenar_compensador_atraso(PLANT, pm_des_a, Kc=1.0)
                        st.session_state.ctrl_Kp_global = p["T"]
                        st.session_state.ctrl_Ki_global = p["beta"]
                        st.session_state.ctrl_Kd_global = p["Kc"]
                    else:
                        p, txt = disenar_compensador_adelanto_atraso(PLANT, pm_des_a)
                        st.session_state.ctrl_Kp_global = p["T1"]
                        st.session_state.ctrl_Ki_global = p["T2"]
                        st.session_state.ctrl_Kd_global = p["beta"]
                    st.session_state[f"{tipo}_Kp"] = st.session_state.ctrl_Kp_global
                    st.session_state[f"{tipo}_Ki"] = st.session_state.ctrl_Ki_global
                    st.session_state[f"{tipo}_Kd"] = st.session_state.ctrl_Kd_global
                    st.session_state.comp_diseno_texto = txt
                except Exception as e:
                    st.session_state.comp_diseno_texto = f"Error en diseño: {e}"
                st.rerun()
            if st.session_state.get("comp_diseno_texto"):
                with st.expander("📋 Pasos del diseño", expanded=True):
                    st.code(st.session_state.comp_diseno_texto, language="text")
        else:
            # ---- Sliders PID estándar ----
            st.session_state.ctrl_Kp_analisis = min(200.0, float(st.session_state.ctrl_Kp_global))
            st.session_state.ctrl_Kp_global = st.slider(
                "Kp", 0.0, 200.0, step=0.5,
                key="ctrl_Kp_analisis", on_change=_sync_Kp_analisis)
            st.session_state[f"{tipo}_Kp"] = st.session_state.ctrl_Kp_global

            if tipo in ("PI", "PID"):
                st.session_state.ctrl_Ki_analisis = min(200.0, float(st.session_state.ctrl_Ki_global))
                st.session_state.ctrl_Ki_global = st.slider(
                    "Ki", 0.0, 200.0, step=0.5,
                    key="ctrl_Ki_analisis", on_change=_sync_Ki_analisis)
                st.session_state[f"{tipo}_Ki"] = st.session_state.ctrl_Ki_global
            if tipo in ("PD", "PID"):
                st.session_state.ctrl_Kd_analisis = min(30.0, float(st.session_state.ctrl_Kd_global))
                st.session_state.ctrl_Kd_global = st.slider(
                    "Kd", 0.0, 30.0, step=0.05,
                    key="ctrl_Kd_analisis", on_change=_sync_Kd_analisis)
                st.session_state[f"{tipo}_Kd"] = st.session_state.ctrl_Kd_global

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
            if st.checkbox(f"{c} (Kp={kp:.1f} Ki={ki:.1f} Kd={kd:.2f})",
                           key=f"chk_{c}"):
                overlays_seleccionados.append(c)

        st.markdown("---")
        if st.button("↺ Restablecer ganancias"):
            for k, v in DEFAULTS[tipo].items():
                st.session_state[f"{tipo}_{k}"] = v
            if tipo != "Lazo Abierto":
                st.session_state.ctrl_Kp_global = DEFAULTS[tipo]["Kp"]
                st.session_state.ctrl_Ki_global = DEFAULTS[tipo]["Ki"]
                st.session_state.ctrl_Kd_global = DEFAULTS[tipo]["Kd"]
            st.rerun()

    with col_der:
        if tipo == "Lazo Abierto":
            Kp, Ki, Kd = 1.0, 0.0, 0.0
        else:
            Kp = st.session_state.ctrl_Kp_global
            Ki = st.session_state.ctrl_Ki_global
            Kd = st.session_state.ctrl_Kd_global
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
                if tipo in TIPOS_COMPENSADOR:
                    try:
                        ceros_c = ct.zeros(C_sys)
                        polos_c = ct.poles(C_sys)
                        st.markdown("**Ceros de C(s):**")
                        for z in ceros_c:
                            st.code(f"  z = {z.real:.4f} {z.imag:+.4f}j", language="text")
                        st.markdown("**Polos de C(s):**")
                        for p in polos_c:
                            st.code(f"  p = {p.real:.4f} {p.imag:+.4f}j", language="text")
                    except Exception:
                        pass
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
        _opts_sim = ["P", "PI", "PD", "PID", "Adelanto", "Atraso", "Atr-Adel"]
        _tipo_g = st.session_state.ctrl_tipo_global
        st.session_state.ctrl_tipo_sim = _tipo_g if _tipo_g in _opts_sim else "PID"
        sim_tipo = st.radio(
            "Tipo", _opts_sim,
            horizontal=True, key="ctrl_tipo_sim",
            on_change=_sync_tipo_sim,
        )
        st.session_state.ctrl_tipo_global = sim_tipo
        st.session_state.sim_ctrl_tipo = sim_tipo

        if sim_tipo in TIPOS_COMPENSADOR:
            lbl1s = "T₁" if sim_tipo == "Atr-Adel" else "T"
            st.session_state.ctrl_Kp_sim = min(10.0, max(0.001, float(st.session_state.ctrl_Kp_global)))
            st.session_state.ctrl_Kp_global = st.slider(
                lbl1s, 0.001, 10.0, step=0.001,
                key="ctrl_Kp_sim", on_change=_sync_Kp_sim)

            if sim_tipo == "Adelanto":
                lbl2s, mn2s, mx2s, stp2s = "α", 0.01, 0.99, 0.01
            elif sim_tipo == "Atraso":
                lbl2s, mn2s, mx2s, stp2s = "β", 1.01, 20.0, 0.01
            else:
                lbl2s, mn2s, mx2s, stp2s = "T₂", 0.01, 100.0, 0.01
            st.session_state.ctrl_Ki_sim = min(mx2s, max(mn2s, float(st.session_state.ctrl_Ki_global)))
            st.session_state.ctrl_Ki_global = st.slider(
                lbl2s, mn2s, mx2s, step=stp2s,
                key="ctrl_Ki_sim", on_change=_sync_Ki_sim)

            if sim_tipo == "Atr-Adel":
                lbl3s, mn3s, mx3s, stp3s = "β", 1.01, 20.0, 0.01
            else:
                lbl3s, mn3s, mx3s, stp3s = "Kc", 0.1, 100.0, 0.1
            st.session_state.ctrl_Kd_sim = min(mx3s, max(mn3s, float(st.session_state.ctrl_Kd_global)))
            st.session_state.ctrl_Kd_global = st.slider(
                lbl3s, mn3s, mx3s, step=stp3s,
                key="ctrl_Kd_sim", on_change=_sync_Kd_sim)

            st.markdown("---")
            st.markdown("##### 🔧 Diseño automático")
            pm_des_s = st.number_input("PM deseado (°)", 30.0, 80.0, 45.0, step=1.0,
                                       key="pm_deseado_sim")
            if st.button("🔧 Diseñar", use_container_width=True, key="btn_diseno_sim"):
                try:
                    if sim_tipo == "Adelanto":
                        p, txt = disenar_compensador_adelanto(PLANT, pm_des_s, Kc=1.0)
                        st.session_state.ctrl_Kp_global = p["T"]
                        st.session_state.ctrl_Ki_global = p["alpha"]
                        st.session_state.ctrl_Kd_global = p["Kc"]
                    elif sim_tipo == "Atraso":
                        p, txt = disenar_compensador_atraso(PLANT, pm_des_s, Kc=1.0)
                        st.session_state.ctrl_Kp_global = p["T"]
                        st.session_state.ctrl_Ki_global = p["beta"]
                        st.session_state.ctrl_Kd_global = p["Kc"]
                    else:
                        p, txt = disenar_compensador_adelanto_atraso(PLANT, pm_des_s)
                        st.session_state.ctrl_Kp_global = p["T1"]
                        st.session_state.ctrl_Ki_global = p["T2"]
                        st.session_state.ctrl_Kd_global = p["beta"]
                    st.session_state.comp_diseno_texto = txt
                    st.session_state.sim_comp_state = None  # reiniciar estado
                except Exception as e:
                    st.session_state.comp_diseno_texto = f"Error: {e}"
                st.rerun()
            if st.session_state.get("comp_diseno_texto"):
                with st.expander("📋 Pasos del diseño", expanded=False):
                    st.code(st.session_state.comp_diseno_texto, language="text")
        else:
            st.session_state.ctrl_Kp_sim = min(200.0, float(st.session_state.ctrl_Kp_global))
            st.session_state.ctrl_Kp_global = st.slider(
                "Kp", 0.0, 200.0, step=0.5,
                key="ctrl_Kp_sim", on_change=_sync_Kp_sim)
            if sim_tipo in ("PI", "PID"):
                st.session_state.ctrl_Ki_sim = min(200.0, float(st.session_state.ctrl_Ki_global))
                st.session_state.ctrl_Ki_global = st.slider(
                    "Ki", 0.0, 200.0, step=0.5,
                    key="ctrl_Ki_sim", on_change=_sync_Ki_sim)
            if sim_tipo in ("PD", "PID"):
                st.session_state.ctrl_Kd_sim = min(30.0, float(st.session_state.ctrl_Kd_global))
                st.session_state.ctrl_Kd_global = st.slider(
                    "Kd", 0.0, 30.0, step=0.05,
                    key="ctrl_Kd_sim", on_change=_sync_Kd_sim)

        st.markdown("---")
        st.markdown("#### 🎯 Auto-sintonización")
        st.caption("Busca Kp, Ki, Kd óptimos para el controlador actual usando el modelo lineal.")

        if st.button("🎯 Auto-sintonizar", use_container_width=True, key="btn_auto"):
            with st.spinner(f"Optimizando ganancias para {sim_tipo}... (puede tomar 20-40 s)"):
                from graficos import auto_sintonizar
                res = auto_sintonizar(sim_tipo, A_SS, B_SS)
            if res["exito"]:
                st.session_state.ctrl_Kp_global = res["Kp"]
                st.session_state.ctrl_Ki_global = res["Ki"]
                st.session_state.ctrl_Kd_global = res["Kd"]
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
        if cb1.button("▶ Iniciar", use_container_width=True, key="btn_start",
                      disabled=st.session_state.sim_running):
            st.session_state.sim_running = True
            st.rerun()
        if cb2.button("■ Detener", use_container_width=True, key="btn_stop",
                      disabled=not st.session_state.sim_running):
            st.session_state.sim_running = False
            st.rerun()

        if st.button("↺ Reset", use_container_width=True, key="btn_reset"):
            reset_simulacion()
            st.rerun()

        if st.button("⏹ Detener y limpiar", use_container_width=True, key="btn_detener_limpiar"):
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

            Kp_s = st.session_state.ctrl_Kp_global
            Ki_s = st.session_state.ctrl_Ki_global
            Kd_s = st.session_state.ctrl_Kd_global
            tipo_loop = st.session_state.ctrl_tipo_global

            x_state    = st.session_state.sim_state.copy().astype(float)
            integral   = float(st.session_state.sim_integral)
            error_prev = float(st.session_state.sim_error_prev)

            # Preparar compensador SS si corresponde
            _usando_comp = tipo_loop in TIPOS_COMPENSADOR
            if _usando_comp:
                try:
                    _C_tf = construir_controlador(tipo_loop, Kp_s, Ki_s, Kd_s)
                    _C_ss = ct.tf2ss(_C_tf)
                    _Ac = np.asarray(_C_ss.A)
                    _Bc = np.asarray(_C_ss.B)
                    _Cc = np.asarray(_C_ss.C)
                    _Dc = np.asarray(_C_ss.D)
                    _nc = _Ac.shape[0]
                    x_comp = st.session_state.sim_comp_state
                    if x_comp is None or not hasattr(x_comp, "__len__") or len(x_comp) != _nc:
                        x_comp = np.zeros(_nc)
                    x_comp = x_comp.astype(float)
                except Exception:
                    _usando_comp = False
                    x_comp = np.zeros(1)

            for frame_i in range(FRAMES_X_BLOQUE):
                # Perturbación pendiente
                if st.session_state.sim_perturbar:
                    x_state = aplicar_perturbacion(x_state)
                    st.session_state.sim_perturbar = False

                # Extraer θ y x del estado completo
                theta   = x_state[2]   # ángulo péndulo [rad]
                error   = theta         # u>0 corrige theta>0 (B[3,0]<0)

                if _usando_comp:
                    # Acción de control via compensador SS (Euler)
                    u_raw = float(np.dot(_Cc.flatten(), x_comp) + float(_Dc.flatten()[0]) * error)
                    x_comp = x_comp + dt_int * (_Ac @ x_comp + (_Bc * error).flatten())
                    u = float(np.clip(u_raw, -50.0, 50.0))
                else:
                    # Acción de control PID
                    u, integral, _ = pid_step(
                        error, dt_int, Kp_s, Ki_s, Kd_s,
                        tipo_loop, integral, error_prev)
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
            if _usando_comp:
                st.session_state.sim_comp_state = x_comp

            if st.session_state.sim_running:
                st.rerun()

# ===========================================================================
# PESTAÑA 3: CONTROL FÍSICO (ESP32 via UDP)
# ===========================================================================
with tab_fisico:
    esp32: ClienteESP32 = st.session_state.esp32

    col_f_izq, col_f_der = st.columns([3, 1], gap="medium")

    # ----- Panel derecho: conexión y controlador -----
    with col_f_der:

        # --- Conexión ---
        st.markdown("#### 📡 Conexión ESP32")
        st.caption("Conéctate a la red WiFi **PenduloPID** (pass: `pendulo123`) antes de continuar.")

        conectado = esp32.conectado
        if conectado:
            st.success(f"✅ Conectado — {esp32.frames_recibidos} frames rx")
        else:
            st.error("❌ Sin señal del ESP32")

        c1, c2 = st.columns(2)
        if c1.button("🔗 Conectar", use_container_width=True, key="btn_f_conectar"):
            if esp32._sock is None or not esp32._activo:
                ok = esp32.conectar()
                if ok:
                    st.success("Socket abierto")
                else:
                    st.error("Error abriendo socket")
            st.rerun()

        if c2.button("⛔ Desconectar", use_container_width=True, key="btn_f_desconectar"):
            esp32.desconectar()
            st.session_state.fisico_running = False
            st.rerun()

        st.markdown("---")

        # --- Controlador ---
        st.markdown("#### 🎛️ Controlador")
        _opts_f = ["P", "PI", "PD", "PID", "Adelanto", "Atraso", "Atr-Adel"]
        _tipo_gf = st.session_state.ctrl_tipo_global
        st.session_state.ctrl_tipo_fisico = _tipo_gf if _tipo_gf in _opts_f else "PID"
        f_tipo = st.radio(
            "Tipo", _opts_f,
            horizontal=True, key="ctrl_tipo_fisico",
            on_change=_sync_tipo_fisico,
        )
        st.session_state.ctrl_tipo_global = f_tipo

        f_Ki = 0.0
        f_Kd = 0.0

        if f_tipo in TIPOS_COMPENSADOR:
            # ---- Sliders compensador (mismos rangos que análisis/sim) ----
            lbl1f = "T₁" if f_tipo == "Atr-Adel" else "T"
            st.session_state.ctrl_Kp_fisico = min(10.0, max(0.001, float(st.session_state.ctrl_Kp_global)))
            f_Kp = st.slider(lbl1f, 0.001, 10.0, step=0.001,
                             key="ctrl_Kp_fisico", on_change=_sync_Kp_fisico)
            st.session_state.ctrl_Kp_global = f_Kp

            if f_tipo == "Adelanto":
                lbl2f, mn2f, mx2f, stp2f = "α", 0.01, 0.99, 0.01
            elif f_tipo == "Atraso":
                lbl2f, mn2f, mx2f, stp2f = "β", 1.01, 20.0, 0.01
            else:
                lbl2f, mn2f, mx2f, stp2f = "T₂", 0.01, 100.0, 0.01
            st.session_state.ctrl_Ki_fisico = min(mx2f, max(mn2f, float(st.session_state.ctrl_Ki_global)))
            f_Ki = st.slider(lbl2f, mn2f, mx2f, step=stp2f,
                             key="ctrl_Ki_fisico", on_change=_sync_Ki_fisico)
            st.session_state.ctrl_Ki_global = f_Ki

            if f_tipo == "Atr-Adel":
                lbl3f, mn3f, mx3f, stp3f = "β", 1.01, 20.0, 0.01
            else:
                lbl3f, mn3f, mx3f, stp3f = "Kc", 0.1, 100.0, 0.1
            st.session_state.ctrl_Kd_fisico = min(mx3f, max(mn3f, float(st.session_state.ctrl_Kd_global)))
            f_Kd = st.slider(lbl3f, mn3f, mx3f, step=stp3f,
                             key="ctrl_Kd_fisico", on_change=_sync_Kd_fisico)
            st.session_state.ctrl_Kd_global = f_Kd

            # ---- Diseño automático ----
            pm_des_f = st.number_input("PM deseado (°)", 30.0, 80.0, 45.0, step=1.0,
                                       key="pm_deseado_fisico")
            if st.button("🔧 Diseñar", use_container_width=True, key="btn_diseno_fisico"):
                try:
                    if f_tipo == "Adelanto":
                        pcomp, txt = disenar_compensador_adelanto(PLANT, pm_des_f, Kc=1.0)
                        st.session_state.ctrl_Kp_global = pcomp["T"]
                        st.session_state.ctrl_Ki_global = pcomp["alpha"]
                        st.session_state.ctrl_Kd_global = pcomp["Kc"]
                    elif f_tipo == "Atraso":
                        pcomp, txt = disenar_compensador_atraso(PLANT, pm_des_f, Kc=1.0)
                        st.session_state.ctrl_Kp_global = pcomp["T"]
                        st.session_state.ctrl_Ki_global = pcomp["beta"]
                        st.session_state.ctrl_Kd_global = pcomp["Kc"]
                    else:
                        pcomp, txt = disenar_compensador_adelanto_atraso(PLANT, pm_des_f)
                        st.session_state.ctrl_Kp_global = pcomp["T1"]
                        st.session_state.ctrl_Ki_global = pcomp["T2"]
                        st.session_state.ctrl_Kd_global = pcomp["beta"]
                    st.session_state[f"{f_tipo}_Kp"] = st.session_state.ctrl_Kp_global
                    st.session_state[f"{f_tipo}_Ki"] = st.session_state.ctrl_Ki_global
                    st.session_state[f"{f_tipo}_Kd"] = st.session_state.ctrl_Kd_global
                    st.session_state.comp_diseno_texto = txt
                except Exception as e:
                    st.session_state.comp_diseno_texto = f"Error en diseño: {e}"
                st.rerun()
            if st.session_state.get("comp_diseno_texto"):
                with st.expander("📋 Pasos del diseño", expanded=False):
                    st.code(st.session_state.comp_diseno_texto, language="text")

            # ---- Vista previa de coeficientes discretos ----
            try:
                _b0, _b1, _b2, _a1, _a2 = discretizar_compensador(
                    f_tipo,
                    st.session_state.ctrl_Kp_global,
                    st.session_state.ctrl_Ki_global,
                    st.session_state.ctrl_Kd_global,
                    Ts=0.005,
                )
                st.caption("Coeficientes discretos (Tustin, Ts=5ms) a enviar:")
                st.code(
                    f"b0={_b0:.5f}  b1={_b1:.5f}  b2={_b2:.5f}\n"
                    f"a1={_a1:.5f}  a2={_a2:.5f}",
                    language="text",
                )
            except Exception as e:
                st.warning(f"No se pudo discretizar: {e}")
        else:
            # ---- Sliders PID estándar ----
            st.session_state.ctrl_Kp_fisico = float(st.session_state.ctrl_Kp_global)
            f_Kp = st.slider("Kp", 0.0, 400.0, step=1.0,
                              key="ctrl_Kp_fisico", on_change=_sync_Kp_fisico)
            st.session_state.ctrl_Kp_global = f_Kp
            if f_tipo in ("PI", "PID"):
                st.session_state.ctrl_Ki_fisico = min(300.0, float(st.session_state.ctrl_Ki_global))
                f_Ki = st.slider("Ki", 0.0, 300.0, step=1.0,
                                  key="ctrl_Ki_fisico", on_change=_sync_Ki_fisico)
                st.session_state.ctrl_Ki_global = f_Ki
            if f_tipo in ("PD", "PID"):
                st.session_state.ctrl_Kd_fisico = min(50.0, float(st.session_state.ctrl_Kd_global))
                f_Kd = st.slider("Kd", 0.0, 50.0, step=0.5,
                                  key="ctrl_Kd_fisico", on_change=_sync_Kd_fisico)
                st.session_state.ctrl_Kd_global = f_Kd

        if st.button("📤 Enviar parámetros", use_container_width=True, key="btn_f_params"):
            _tg = st.session_state.ctrl_tipo_global
            if _tg in TIPOS_COMPENSADOR:
                b0, b1, b2, a1, a2 = discretizar_compensador(
                    _tg,
                    st.session_state.ctrl_Kp_global,
                    st.session_state.ctrl_Ki_global,
                    st.session_state.ctrl_Kd_global,
                    Ts=0.005,
                )
                esp32.cmd_set_compensador(b0, b1, b2, a1, a2)
                st.success(
                    f"Enviado COMP ({_tg}): "
                    f"b0={b0:.4f} b1={b1:.4f} b2={b2:.4f} a1={a1:.4f} a2={a2:.4f}"
                )
            else:
                esp32.cmd_set_params(
                    _tg,
                    st.session_state.ctrl_Kp_global,
                    st.session_state.ctrl_Ki_global,
                    st.session_state.ctrl_Kd_global,
                )
                st.success(
                    f"Enviado: {_tg} "
                    f"Kp={st.session_state.ctrl_Kp_global} "
                    f"Ki={st.session_state.ctrl_Ki_global} "
                    f"Kd={st.session_state.ctrl_Kd_global}"
                )

        st.markdown("---")

        # --- Controles ---
        st.markdown("#### ▶️ Controles")

        if st.button("🔄 Reset encoder", use_container_width=True, key="btn_f_reset_enc"):
            esp32.cmd_reset_encoder()
            st.session_state.fisico_t_hist     = []
            st.session_state.fisico_theta_hist = []
            st.session_state.fisico_t0         = None
            st.info("Encoder reseteado")

        cc1, cc2 = st.columns(2)
        if cc1.button("▶ Iniciar", use_container_width=True, key="btn_f_start",
                      disabled=st.session_state.fisico_running):
            _tg = st.session_state.ctrl_tipo_global
            if _tg in TIPOS_COMPENSADOR:
                b0, b1, b2, a1, a2 = discretizar_compensador(
                    _tg,
                    st.session_state.ctrl_Kp_global,
                    st.session_state.ctrl_Ki_global,
                    st.session_state.ctrl_Kd_global,
                    Ts=0.005,
                )
                esp32.cmd_set_compensador(b0, b1, b2, a1, a2)
            else:
                esp32.cmd_set_params(
                    _tg,
                    st.session_state.ctrl_Kp_global,
                    st.session_state.ctrl_Ki_global,
                    st.session_state.ctrl_Kd_global,
                )
            time.sleep(0.05)
            esp32.cmd_start()
            st.session_state.fisico_running = True
            st.session_state.fisico_t0      = time.time()
            st.rerun()

        if cc2.button("■ Detener", use_container_width=True, key="btn_f_stop",
                      disabled=not st.session_state.fisico_running):
            esp32.cmd_stop()
            st.session_state.fisico_running = False
            st.rerun()

        st.markdown("---")

        # --- Estado en tiempo real ---
        st.markdown("##### 📊 Estado")
        frame = esp32.ultimo_frame()
        if frame:
            st.metric("θ actual",  f"{frame.angulo:.2f}°")
            st.metric("Error",     f"{frame.error:.2f}°")
            st.metric("PWM",       f"{frame.pwm}")
            st.metric("Estado",    frame.estado)
            col_p, col_i, col_d = st.columns(3)
            col_p.metric("P", f"{frame.P:.1f}")
            col_i.metric("I", f"{frame.I:.1f}")
            col_d.metric("D", f"{frame.D:.1f}")
        else:
            st.metric("θ actual", "—")
            st.metric("Error",    "—")
            st.metric("PWM",      "—")

    # ----- Panel izquierdo: gráfica en tiempo real -----
    with col_f_izq:
        st.markdown("##### 📈 Ángulo θ en tiempo real (datos físicos)")

        # Leer frames nuevos de la cola
        frames_nuevos = esp32.vaciar_cola()
        if frames_nuevos and st.session_state.fisico_t0 is not None:
            t0 = st.session_state.fisico_t0
            for fr in frames_nuevos:
                st.session_state.fisico_t_hist.append(fr.ts - t0)
                st.session_state.fisico_theta_hist.append(fr.angulo)

            # Limitar historial a 60 s
            if len(st.session_state.fisico_t_hist) > 1200:
                st.session_state.fisico_t_hist     = st.session_state.fisico_t_hist[-1000:]
                st.session_state.fisico_theta_hist = st.session_state.fisico_theta_hist[-1000:]

        # Gráfica
        import plotly.graph_objects as go
        fig_f = go.Figure()
        fig_f.add_hline(y=180, line=dict(color="#dc2626", width=1.5, dash="dot"))
        fig_f.add_hline(y=195, line=dict(color="#f59e0b", width=1,   dash="dash"))
        fig_f.add_hline(y=165, line=dict(color="#f59e0b", width=1,   dash="dash"))

        if len(st.session_state.fisico_t_hist) > 0:
            fig_f.add_trace(go.Scatter(
                x=st.session_state.fisico_t_hist,
                y=st.session_state.fisico_theta_hist,
                mode="lines",
                line=dict(color="#2563eb", width=2),
                name="θ físico",
                hovertemplate="t=%{x:.2f}s<br>θ=%{y:.2f}°<extra></extra>",
            ))
            t_max = st.session_state.fisico_t_hist[-1]
            ventana = 20.0
            fig_f.update_xaxes(range=[max(0, t_max - ventana), t_max + 0.5])

        fig_f.update_layout(
            template="plotly_white",
            xaxis_title="Tiempo (s)",
            yaxis_title="θ (°)",
            yaxis=dict(range=[120, 240]),
            height=420,
            margin=dict(l=50, r=20, t=20, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_f, use_container_width=True,
                        config={"displayModeBar": False})

        # Tabla de últimos valores
        if len(st.session_state.fisico_t_hist) > 0:
            with st.expander("📋 Últimos 10 frames", expanded=False):
                import pandas as pd
                n = min(10, len(st.session_state.fisico_t_hist))
                df = pd.DataFrame({
                    "t (s)": [f"{v:.3f}" for v in st.session_state.fisico_t_hist[-n:]],
                    "θ (°)": [f"{v:.2f}" for v in st.session_state.fisico_theta_hist[-n:]],
                })
                st.dataframe(df, use_container_width=True, hide_index=True)

        # Auto-refresh mientras está corriendo
        if st.session_state.fisico_running and conectado:
            time.sleep(0.1)
            st.rerun()