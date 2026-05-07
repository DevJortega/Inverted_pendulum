"""
Dashboard de Análisis y Diseño de Controladores
Planta: Péndulo Invertido sobre Carro
G(s) = 0.01209 / (0.002846 s^2 - 0.09678)
"""

import numpy as np
import streamlit as st
import control as ct
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diseño de Controladores | Péndulo Invertido",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS tema claro / profesional
st.markdown(
    """
    <style>
        .main { background-color: #f6f8fb; }
        .stApp { background-color: #f6f8fb; }
        /* ocultar toolbar de Streamlit y el toggle del sidebar */
        header[data-testid="stHeader"] { display: none !important; }
        [data-testid="collapsedControl"]  { display: none !important; }
        /* reducir padding vertical del contenido principal */
        .block-container { padding-top: 0.75rem !important; padding-bottom: 0.5rem !important; }
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
        /* botones generales */
        .stButton>button {
            background-color: #2563eb;
            color: #ffffff;
            border: none;
            font-weight: 600;
            border-radius: 6px;
        }
        .stButton>button:hover { background-color: #1d4ed8; color: #ffffff; }
        /* botones de expansión: compactos, fondo neutro */
        div[data-testid="column"]:last-child .stButton>button {
            background-color: #e2e8f0;
            color: #1a2332;
            font-size: 1rem;
            padding: 2px 6px;
            border-radius: 5px;
            line-height: 1;
        }
        div[data-testid="column"]:last-child .stButton>button:hover {
            background-color: #cbd5e1;
            color: #1a2332;
        }
        div[data-testid="stExpander"] {
            background-color: #ffffff;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }
        div[data-testid="stMetricValue"] { font-size: 1.0rem; }
        div[data-testid="stMetricLabel"] { font-size: 0.80rem; }
        /* reducir gaps entre filas de gráficas */
        div[data-testid="stVerticalBlock"] > div { gap: 0.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = "plotly_white"
COLOR_MAIN = "#2563eb"
COLOR_REF = "#dc2626"
COLORS_OVERLAY = ["#16a34a", "#ea580c", "#9333ea", "#0891b2", "#ca8a04", "#db2777"]

# ---------------------------------------------------------------------------
# Planta
# ---------------------------------------------------------------------------
NUM = [0.01209]
DEN = [0.002846, 0.0, -0.09678]
PLANT = ct.tf(NUM, DEN)

# ---------------------------------------------------------------------------
# Session State - valores por defecto independientes para cada controlador
# ---------------------------------------------------------------------------
DEFAULTS = {
    "Lazo Abierto": {"Kp": 1.0, "Ki": 0.0, "Kd": 0.0},
    "P":            {"Kp": 30.0, "Ki": 0.0, "Kd": 0.0},
    "PI":           {"Kp": 30.0, "Ki": 5.0, "Kd": 0.0},
    "PD":           {"Kp": 40.0, "Ki": 0.0, "Kd": 3.0},
    "PID":          {"Kp": 50.0, "Ki": 10.0, "Kd": 5.0},
}

for ctrl, vals in DEFAULTS.items():
    for k, v in vals.items():
        key = f"{ctrl}_{k}"
        if key not in st.session_state:
            st.session_state[key] = v

if "controlador_actual" not in st.session_state:
    st.session_state.controlador_actual = "PID"

# ---------------------------------------------------------------------------
# Funciones de control
# ---------------------------------------------------------------------------
def construir_controlador(tipo, Kp, Ki, Kd):
    if tipo == "Lazo Abierto":
        return ct.tf([1], [1])
    if tipo == "P":
        return ct.tf([Kp], [1])
    if tipo == "PI":
        return ct.tf([Kp, Ki], [1, 0])
    if tipo == "PD":
        N = 100
        return ct.tf([Kd * N + Kp, Kp * N], [1, N])
    if tipo == "PID":
        N = 100
        num = [Kp + Kd * N, Kp * N + Ki, Ki * N]
        den = [1, N, 0]
        return ct.tf(num, den)
    return ct.tf([1], [1])


def sistema_lazo_cerrado(tipo, Kp, Ki, Kd):
    C = construir_controlador(tipo, Kp, Ki, Kd)
    if tipo == "Lazo Abierto":
        return PLANT, C
    L = C * PLANT
    T = ct.feedback(L, 1)
    return T, C


def calcular_metricas(T, t, y):
    try:
        polos = ct.poles(T)
        estable = bool(np.all(np.real(polos) < 0))
    except Exception:
        estable = False

    if not estable or not np.all(np.isfinite(y)):
        return {"estable": False, "Ts": None, "OS": None, "SSE": None, "y_ss": None}

    y_ss = y[-1]
    banda = 0.02 * abs(y_ss) if abs(y_ss) > 1e-9 else 0.02
    fuera = np.where(np.abs(y - y_ss) > banda)[0]
    Ts = t[fuera[-1]] if len(fuera) > 0 else 0.0
    if y_ss > 1e-9:
        OS = max(0.0, (np.max(y) - y_ss) / y_ss * 100.0)
    else:
        OS = 0.0
    SSE = 1.0 - y_ss
    return {"estable": True, "Ts": Ts, "OS": OS, "SSE": SSE, "y_ss": y_ss}


def respuesta_escalon(T, t_final=5.0, n_pts=600):
    try:
        t = np.linspace(0, t_final, n_pts)
        t_out, y_out = ct.step_response(T, T=t)
        y_out = np.asarray(y_out).flatten()
        y_out = np.clip(y_out, -1e6, 1e6)
        return t_out, y_out
    except Exception:
        return np.linspace(0, t_final, n_pts), np.full(n_pts, np.nan)


def calcular_margenes(L):
    """Devuelve (gm_db, pm_deg, wcg, wcp) o Nones."""
    try:
        gm, pm, wcg, wcp = ct.margin(L)
        gm_db = 20 * np.log10(gm) if (gm is not None and gm > 0 and np.isfinite(gm)) else None
        pm_val = pm if (pm is not None and np.isfinite(pm)) else None
        wcg_v = wcg if (wcg is not None and np.isfinite(wcg)) else None
        wcp_v = wcp if (wcp is not None and np.isfinite(wcp)) else None
        return gm_db, pm_val, wcg_v, wcp_v
    except Exception:
        return None, None, None, None


# ---------------------------------------------------------------------------
# Funciones de gráficos
# ---------------------------------------------------------------------------
def fig_step(curvas, ref=1.0, t_final=5.0, height=300):
    fig = go.Figure()
    fig.add_hline(
        y=ref, line=dict(color=COLOR_REF, width=1.5, dash="dot"),
        annotation_text="Ref", annotation_position="top right",
    )
    for c in curvas:
        fig.add_trace(go.Scatter(
            x=c["t"], y=c["y"], mode="lines", name=c["nombre"],
            line=dict(color=c["color"], width=2.2, dash=c.get("dash", "solid")),
            hovertemplate="t=%{x:.3f}s<br>y=%{y:.4f}<extra></extra>",
        ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Respuesta al Escalón", font=dict(size=13)),
        xaxis_title="Tiempo (s)",
        yaxis_title="y(t)",
        height=height,
        margin=dict(l=50, r=15, t=42, b=15),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=9),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#e2e8f0", borderwidth=1,
        ),
    )
    fig.update_xaxes(range=[0, t_final])
    return fig


def fig_root_locus(sistemas, height=300):
    """sistemas: lista de dicts {nombre, C, color, dash}"""
    fig = go.Figure()
    all_re, all_im = [], []

    for s in sistemas:
        try:
            L = s["C"] * PLANT
            try:
                rldata = ct.root_locus_map(L)
                roots = rldata.loci
            except AttributeError:
                roots, _ = ct.root_locus(L, plot=False)

            # Trazar ramas
            for i in range(roots.shape[1]):
                fig.add_trace(go.Scatter(
                    x=np.real(roots[:, i]), y=np.imag(roots[:, i]),
                    mode="lines",
                    line=dict(color=s["color"], width=1.8, dash=s.get("dash", "solid")),
                    name=s["nombre"], showlegend=(i == 0),
                    legendgroup=s["nombre"],
                    hovertemplate=f"{s['nombre']}<br>Re=%{{x:.3f}}<br>Im=%{{y:.3f}}<extra></extra>",
                ))
                all_re.extend(np.real(roots[:, i]).tolist())
                all_im.extend(np.imag(roots[:, i]).tolist())

            # Polos LC actuales (cuadrado)
            try:
                T = ct.feedback(L, 1)
                polos_lc = ct.poles(T)
                fig.add_trace(go.Scatter(
                    x=np.real(polos_lc), y=np.imag(polos_lc),
                    mode="markers",
                    marker=dict(symbol="square", size=10, color=s["color"],
                                line=dict(color="white", width=1)),
                    name=f"Polos LC {s['nombre']}",
                    legendgroup=s["nombre"], showlegend=False,
                    hovertemplate=f"Polo LC<br>Re=%{{x:.3f}}<br>Im=%{{y:.3f}}<extra></extra>",
                ))
            except Exception:
                pass
        except Exception:
            continue

    # Polos y ceros de la planta (siempre los mismos)
    try:
        polos_g = ct.poles(PLANT)
        fig.add_trace(go.Scatter(
            x=np.real(polos_g), y=np.imag(polos_g),
            mode="markers",
            marker=dict(symbol="x", size=14, color="#1a2332", line=dict(width=3)),
            name="Polos planta",
        ))
    except Exception:
        pass

    fig.add_vline(x=0, line=dict(color="#94a3b8", width=1, dash="dash"))
    fig.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="dash"))

    # Ajuste automático con margen
    if all_re and all_im:
        re_min, re_max = min(all_re), max(all_re)
        im_min, im_max = min(all_im), max(all_im)
        re_span = max(re_max - re_min, 1.0)
        im_span = max(im_max - im_min, 1.0)
        pad_r = 0.15 * re_span
        pad_i = 0.15 * im_span
        x_range = [re_min - pad_r, re_max + pad_r]
        y_range = [im_min - pad_i, im_max + pad_i]
    else:
        x_range, y_range = None, None

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Lugar Geométrico de las Raíces", font=dict(size=13)),
        xaxis_title="Re",
        yaxis_title="Im",
        height=height,
        margin=dict(l=50, r=15, t=42, b=15),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=9),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#e2e8f0", borderwidth=1,
        ),
    )
    if x_range:
        fig.update_xaxes(range=x_range)
        fig.update_yaxes(range=y_range)
    return fig


def calcular_rango_omega(sistemas):
    """Rango fijo: 1 rad/s a 1 Grad/s (10^9)."""
    return 0, 9

def fig_bode(sistemas, height=300):
    """Múltiples sistemas superpuestos. Marca MF y MG del primero (actual)."""
    exp_min, exp_max = calcular_rango_omega(sistemas)
    omega = np.logspace(exp_min, exp_max, 3000)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Magnitud (dB)", "Fase (°)"),
        row_heights=[0.5, 0.5],
    )

    margenes_actual = (None, None, None, None)

    for idx, s in enumerate(sistemas):
        try:
            L = s["C"] * PLANT
            mag, phase, w = ct.frequency_response(L, omega)
            mag = np.asarray(mag).flatten()
            phase = np.asarray(phase).flatten()
            phase = np.unwrap(phase)   # evita saltos ±π; fase continua
            mag_db = 20 * np.log10(np.maximum(mag, 1e-12))
            phase_deg = np.degrees(phase)

            fig.add_trace(go.Scatter(
                x=w, y=mag_db, mode="lines",
                line=dict(color=s["color"], width=2, dash=s.get("dash", "solid")),
                name=s["nombre"], legendgroup=s["nombre"],
                hovertemplate="ω=%{x:.4g} rad/s<br>%{y:.2f} dB<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=w, y=phase_deg, mode="lines",
                line=dict(color=s["color"], width=2, dash=s.get("dash", "solid")),
                name=s["nombre"], legendgroup=s["nombre"], showlegend=False,
                hovertemplate="ω=%{x:.4g} rad/s<br>%{y:.2f}°<extra></extra>",
            ), row=2, col=1)

            if idx == 0:
                margenes_actual = calcular_margenes(L)
        except Exception:
            continue

    # Líneas de referencia
    fig.add_hline(y=0, line=dict(color="#94a3b8", dash="dash", width=1), row=1, col=1)
    fig.add_hline(y=-180, line=dict(color="#94a3b8", dash="dash", width=1), row=2, col=1)

    gm_db, pm_val, wcg, wcp = margenes_actual

    # Marcado del Margen de Fase
    if wcp is not None and pm_val is not None:
        fig.add_vline(x=wcp, line=dict(color=COLOR_REF, dash="dot", width=1.3), row=1, col=1)
        fig.add_vline(x=wcp, line=dict(color=COLOR_REF, dash="dot", width=1.3), row=2, col=1)
        # En eje tipo "log", la coordenada x de la anotación se da como valor real (no log10)
        fig.add_annotation(
            x=wcp, y=-180 + pm_val,
            text=f"<b>MF={pm_val:.1f}°</b>",
            showarrow=True, arrowhead=2, ax=40, ay=-28,
            bgcolor="rgba(220,38,38,0.9)", font=dict(color="white", size=10),
            xref="x2", yref="y2",
        )

    # Marcado del Margen de Ganancia
    if wcg is not None and gm_db is not None:
        fig.add_vline(x=wcg, line=dict(color="#16a34a", dash="dot", width=1.3), row=1, col=1)
        fig.add_vline(x=wcg, line=dict(color="#16a34a", dash="dot", width=1.3), row=2, col=1)
        fig.add_annotation(
            x=wcg, y=-gm_db,
            text=f"<b>MG={gm_db:.1f} dB</b>",
            showarrow=True, arrowhead=2, ax=40, ay=28,
            bgcolor="rgba(22,163,74,0.9)", font=dict(color="white", size=10),
            xref="x", yref="y",
        )

    # Eje X log con rango FORZADO en log10 (clave: Plotly espera log10 cuando type="log")
    fig.update_xaxes(
        type="log",
        range=[exp_min, exp_max],
        autorange=False,
        row=1, col=1,
    )
    fig.update_xaxes(
        type="log",
        range=[exp_min, exp_max],
        autorange=False,
        title_text="ω (rad/s)",
        row=2, col=1,
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=50, r=15, t=50, b=15),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=9),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#e2e8f0", borderwidth=1,
        ),
    )
    return fig, gm_db, pm_val, wcg, wcp


def fig_polos_lc(sistemas, height=300):
    """Mapa de polos y ceros del lazo cerrado para cada sistema."""
    fig = go.Figure()
    all_re, all_im = [0], [0]

    for s in sistemas:
        try:
            L = s["C"] * PLANT
            T = ct.feedback(L, 1)
            polos = ct.poles(T)
            ceros = ct.zeros(T)

            fig.add_trace(go.Scatter(
                x=np.real(polos), y=np.imag(polos),
                mode="markers",
                marker=dict(symbol="x", size=14, color=s["color"],
                            line=dict(width=3)),
                name=f"Polos {s['nombre']}",
                legendgroup=s["nombre"],
                hovertemplate=f"{s['nombre']}<br>Re=%{{x:.3f}}<br>Im=%{{y:.3f}}<extra></extra>",
            ))
            if len(ceros) > 0:
                fig.add_trace(go.Scatter(
                    x=np.real(ceros), y=np.imag(ceros),
                    mode="markers",
                    marker=dict(symbol="circle-open", size=12, color=s["color"],
                                line=dict(width=2.5)),
                    name=f"Ceros {s['nombre']}",
                    legendgroup=s["nombre"], showlegend=False,
                    hovertemplate=f"{s['nombre']}<br>Re=%{{x:.3f}}<br>Im=%{{y:.3f}}<extra></extra>",
                ))
            all_re.extend(np.real(polos).tolist())
            all_im.extend(np.imag(polos).tolist())
            if len(ceros) > 0:
                all_re.extend(np.real(ceros).tolist())
                all_im.extend(np.imag(ceros).tolist())
        except Exception:
            continue

    fig.add_vline(x=0, line=dict(color="#94a3b8", width=1, dash="dash"))
    fig.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="dash"))

    if all_re:
        re_span = max(max(all_re) - min(all_re), 1.0)
        im_span = max(max(all_im) - min(all_im), 1.0)
        pad_r = 0.2 * re_span
        pad_i = 0.2 * im_span
        x_range = [min(all_re) - pad_r, max(all_re) + pad_r]
        y_range = [min(all_im) - pad_i, max(all_im) + pad_i]
    else:
        x_range, y_range = None, None

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Polos y Ceros de Lazo Cerrado", font=dict(size=13)),
        xaxis_title="Re",
        yaxis_title="Im",
        height=height,
        margin=dict(l=50, r=15, t=42, b=15),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=9),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#e2e8f0", borderwidth=1,
        ),
    )
    if x_range:
        fig.update_xaxes(range=x_range)
        fig.update_yaxes(range=y_range)
    return fig


# ---------------------------------------------------------------------------
# UI - Encabezado
# ---------------------------------------------------------------------------
st.markdown(
    "<h3 style='margin-bottom:0;'>🎛️ Análisis y Diseño de Controladores</h3>"
    "<p style='color:#64748b;margin-top:0;font-size:0.9rem;'>Planta: Péndulo Invertido — "
    "<code>G(s) = 0.01209 / (0.002846·s² − 0.09678)</code> · "
    "<span style='color:#dc2626;'>Inestable en lazo abierto</span></p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Layout principal: 2 columnas
# ---------------------------------------------------------------------------
col_izq, col_der = st.columns([1, 3], gap="medium")

# =========================== PANEL IZQUIERDO ===============================
with col_izq:
    st.markdown("#### ⚙️ Configuración")

    tipo = st.radio(
        "Controlador actual",
        ["Lazo Abierto", "P", "PI", "PD", "PID"],
        index=["Lazo Abierto", "P", "PI", "PD", "PID"].index(st.session_state.controlador_actual),
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

# =========================== PANEL DERECHO ================================
with col_der:
    Kp = st.session_state[f"{tipo}_Kp"]
    Ki = st.session_state[f"{tipo}_Ki"]
    Kd = st.session_state[f"{tipo}_Kd"]
    T_sys, C_sys = sistema_lazo_cerrado(tipo, Kp, Ki, Kd)
    t_arr, y_arr = respuesta_escalon(T_sys, t_final=t_final)
    metricas = calcular_metricas(T_sys, t_arr, y_arr)

    # Lista de sistemas (actual + overlays) para todas las gráficas
    sistemas = [{
        "nombre": f"{tipo} (actual)",
        "C": C_sys,
        "color": COLOR_MAIN,
        "dash": "solid",
    }]
    curvas_step = [{
        "nombre": f"{tipo} (actual)",
        "t": t_arr, "y": y_arr,
        "color": COLOR_MAIN, "dash": "solid",
    }]
    for idx, ctrl_ov in enumerate(overlays_seleccionados):
        kp_o = st.session_state[f"{ctrl_ov}_Kp"]
        ki_o = st.session_state[f"{ctrl_ov}_Ki"]
        kd_o = st.session_state[f"{ctrl_ov}_Kd"]
        T_o, C_o = sistema_lazo_cerrado(ctrl_ov, kp_o, ki_o, kd_o)
        t_o, y_o = respuesta_escalon(T_o, t_final=t_final)
        color_ov = COLORS_OVERLAY[idx % len(COLORS_OVERLAY)]
        sistemas.append({
            "nombre": ctrl_ov,
            "C": C_o,
            "color": color_ov,
            "dash": "dash",
        })
        curvas_step.append({
            "nombre": ctrl_ov,
            "t": t_o, "y": y_o,
            "color": color_ov, "dash": "dash",
        })

    # ----- Métricas en tarjetas -----
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    if metricas["estable"]:
        m1.metric("Estado", "✅ Estable")
        m2.metric("Ts (2%)", f"{metricas['Ts']:.3f} s")
        m3.metric("OS%", f"{metricas['OS']:.2f}")
        m4.metric("SSE", f"{metricas['SSE']:.4f}")
    else:
        m1.metric("Estado", "❌ Inestable")
        m2.metric("Ts", "—")
        m3.metric("OS%", "—")
        m4.metric("SSE", "—")

    # Calcular márgenes del actual para tarjetas
    L_actual = C_sys * PLANT
    gm_db_a, pm_a, wcg_a, wcp_a = calcular_margenes(L_actual)
    m5.metric("MF", f"{pm_a:.1f}°" if pm_a is not None else "—")
    m6.metric("MG", f"{gm_db_a:.1f} dB" if gm_db_a is not None else "—")

    # ----- Diálogos pop-out -----
    @st.dialog("Respuesta al Escalón", width="large")
    def dlg_step():
        st.plotly_chart(fig_step(curvas_step, t_final=t_final, height=650),
                        use_container_width=True)

    @st.dialog("Lugar Geométrico de las Raíces", width="large")
    def dlg_rl():
        st.plotly_chart(fig_root_locus(sistemas, height=650),
                        use_container_width=True)

    @st.dialog("Diagrama de Bode", width="large")
    def dlg_bode():
        f, *_ = fig_bode(sistemas, height=680)
        st.plotly_chart(f, use_container_width=True)

    @st.dialog("Polos y Ceros LC", width="large")
    def dlg_pz():
        st.plotly_chart(fig_polos_lc(sistemas, height=650),
                        use_container_width=True)

    # ----- Bloque 2x2 -----
    H     = 305   # altura para Step / LGR / Polos-Ceros
    H_BOD = 330   # Bode necesita más espacio (2 subpaneles)
    fila1_a, fila1_b = st.columns(2)
    fila2_a, fila2_b = st.columns(2)

    with fila1_a:
        hcol1, hcol2 = st.columns([6, 1])
        hcol1.markdown("**📈 Respuesta al Escalón**")
        if hcol2.button("⛶", key="max_step", help="Ver en pantalla completa"):
            dlg_step()
        st.plotly_chart(fig_step(curvas_step, t_final=t_final, height=H),
                        use_container_width=True)

    with fila1_b:
        hcol1, hcol2 = st.columns([6, 1])
        hcol1.markdown("**📍 Lugar Geométrico (LGR)**")
        if hcol2.button("⛶", key="max_rl", help="Ver en pantalla completa"):
            dlg_rl()
        st.plotly_chart(fig_root_locus(sistemas, height=H),
                        use_container_width=True)

    with fila2_a:
        hcol1, hcol2 = st.columns([6, 1])
        hcol1.markdown("**🎚️ Diagrama de Bode**")
        if hcol2.button("⛶", key="max_bode", help="Ver en pantalla completa"):
            dlg_bode()
        fig_b, *_ = fig_bode(sistemas, height=H_BOD)
        st.plotly_chart(fig_b, use_container_width=True)

    with fila2_b:
        hcol1, hcol2 = st.columns([6, 1])
        hcol1.markdown("**📌 Polos y Ceros LC**")
        if hcol2.button("⛶", key="max_pz", help="Ver en pantalla completa"):
            dlg_pz()
        st.plotly_chart(fig_polos_lc(sistemas, height=H),
                        use_container_width=True)

    # ----- Información del sistema -----
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
                    st.code(f"{icon} {p.real:.4f} {p.imag:+.4f}j", language="text")
            except Exception:
                st.code("—", language="text")